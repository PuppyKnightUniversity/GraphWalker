# In this file, we realize the vllm inference for LLMs.
# It only supports local model.

import sys

def inference(args, model_path:str, 
              prompt_list:list, 
              adapter_path:str = None,
              max_tokens:int = 4096,
              temperature:float = 0.7,
              top_p:float = 0.8,
              top_k:int = 20,
              repetition_penalty:float = 1.05,
              vllm_batch_size:int = 4,
              max_model_len:int = 16384,
              gpu_memory_utilization:float = 0.85,
              save_path:str = None,
              labels:list = None,
              logger=None,
              return_logits:bool = False,
              classification_options:list = None,
              enable_thinking:bool = False)->list:
    """
    Inference the LLMs with vllm.
    Args:
        args: the arguments.
        model_path: the path of the model.
        prompt_list: the list of prompts.
        adapter_path: the path of the adapter.
        max_tokens: the max tokens of the response.
        temperature: the temperature of the response.
        top_p: the top p of the response.
        top_k: the top k of the response.
        repetition_penalty: the repetition penalty of the response.
        vllm_batch_size: the batch size of the response generation.
        max_model_len: the maximum sequence length (default: 16384). 
                      Can be increased if you have sufficient GPU memory.
                      Common values: 16384, 32768, 65536.
        gpu_memory_utilization: GPU memory utilization ratio (default: 0.85).
                               Lower this if you increase max_model_len and encounter OOM.
        save_path: the directory path to save individual responses (optional).
                   If provided, each response will be saved to a separate file.
        logger: the logger instance for progress display (optional).
        enable_thinking: whether to enable thinking mode for qwen3 models (default: False).
                        When True, the model will generate thinking tokens before the final answer.
    Returns:
        the list of responses.
    """
    from vllm import LLM, SamplingParams
    import torch
    import os
    import json
    import numpy as np

    # Save functionality disabled for open source release
    # if save_path:
    #     os.makedirs(save_path, exist_ok=True)
    #     print(f"Responses will be saved to: {save_path}")
    
    # Helper function to save individual responses (disabled for open source)
    # def save_response(response: str, index: int, save_dir: str, prompt: str = None, label_value = None, logits_dict = None):
    #     """Save a response (and prompt) to a separate JSON file."""
    #     save_file = os.path.join(save_dir, f"response_{index:05d}.json")
    #     with open(save_file, 'w', encoding='utf-8') as f:
    #         payload = {"index": index, "response": response}
    #         if prompt is not None:
    #             payload["prompt"] = prompt
    #         if label_value is not None:
    #             payload["label"] = label_value
    #         if logits_dict is not None:
    #             payload["logits_dict"] = logits_dict
    #         json.dump(payload, f, ensure_ascii=False, indent=2)
    
    # Helper function to load existing responses (disabled for open source)
    # def load_existing_responses(save_dir: str, total_count: int):
    #     """Load existing responses from save directory."""
    #     existing_responses = {}
    #     if save_dir and os.path.isdir(save_dir):
    #         # Find all response files
    #         for filename in os.listdir(save_dir):
    #             if filename.startswith("response_") and filename.endswith(".json"):
    #                 try:
    #                     index = int(filename.replace("response_", "").replace(".json", ""))
    #                     if 0 <= index < total_count:
    #                         with open(os.path.join(save_dir, filename), 'r', encoding='utf-8') as f:
    #                             data = json.load(f)
    #                             existing_responses[index] = data["response"]
    #                 except (ValueError, json.JSONDecodeError, KeyError) as e:
    #                     continue
    #     return existing_responses

    # detect gpu count
    gpu_count = torch.cuda.device_count()
    tensor_parallel_size = min(gpu_count, 4)  # Limit to 4 GPUs max for stability
    
    # Check if model path exists
    import os
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    # initialize vllm kwargs
    vllm_kwargs = {"model": model_path,
                   "tensor_parallel_size": tensor_parallel_size,
                   "trust_remote_code": True,
                   "dtype": "bfloat16",
                   "gpu_memory_utilization": gpu_memory_utilization,  # Control GPU memory usage to avoid OOM
                   "max_model_len": max_model_len,  # Maximum sequence length (can be increased if GPU memory allows)
                   "swap_space": 4}  # Use disk swap space in GB if shared memory is insufficient
    
    # Try to disable v1 engine if it's causing issues
    # vLLM v1 engine may be unstable, use enforce_eager as workaround
    try:
        # Disable v1 engine by using enforce_eager (forces eager mode, avoids v1 engine)
        # This may help with v1 engine initialization failures
        vllm_kwargs["enforce_eager"] = False  # Try False first, if fails, can set to True
    except:
        pass
    
    # Add LoRA support if adapter path is provided
    if adapter_path and adapter_path.strip():
        vllm_kwargs["enable_lora"] = True
        vllm_kwargs["max_lora_rank"] = 64
        vllm_kwargs["max_loras"] = 1

    # initialize vllm model with retry logic
    vllm_model = None
    last_error = None
    
    # Normal initialization
    vllm_model = LLM(**vllm_kwargs)

    
    # initialize tokenizer - always load transformers tokenizer for chat template support
    from transformers import AutoTokenizer
    
    # Always load transformers tokenizer for chat template and other operations
    # vLLM's TokenizerGroup doesn't support apply_chat_template
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                  use_fast=True,
                                                  trust_remote_code=True)
    except Exception as e:
        # Fallback to slow tokenizer if fast tokenizer fails
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                  use_fast=False,
                                                  trust_remote_code=True)
    def extract_logits_for_options(output, tokenizer, classification_options):
        """
        Extract log probabilities (logits) for specific classification options from vLLM output.
        
        Args:
            output: vLLM output object
            tokenizer: tokenizer instance
            classification_options: list of option strings (e.g., ["1", "2", "3", "4", "5"]) or None
        
        Returns:
            dict mapping option strings to their log probabilities
        """
        logits_dict = {}
        
        # Get logprobs from the second-to-last generated token
        # The last token is the end-of-sequence (EOS) token, which we don't need
        # We want the logprobs of the actual output token (e.g., "3"), which is at position -2
        if hasattr(output.outputs[0], 'logprobs') and output.outputs[0].logprobs:
            # logprobs is a list, get the second-to-last token's logprobs (the actual output token)
            if len(output.outputs[0].logprobs) >= 2:
                last_token_logprobs = output.outputs[0].logprobs[-2]  # Skip EOS token
            else:
                last_token_logprobs = None
            
            if last_token_logprobs:
                # Logprob objects have a .logprob attribute to access the value
                # If no specific options provided, return all logprobs
                if not classification_options:
                    # Access .logprob attribute from Logprob objects
                    logits_dict = {str(k): (v.logprob if hasattr(v, 'logprob') else float(v)) for k, v in last_token_logprobs.items()}
                else:
                    # last_token_logprobs is a dict mapping token_id to logprob
                    # Get token IDs for classification options
                    option_token_ids = {}
                    for option in classification_options:
                        # Tokenize the option and get the first token ID
                        # Some models might tokenize "1" as a single token, others might need special handling
                        tokens = tokenizer.encode(option, add_special_tokens=False)
                        if tokens:
                            option_token_ids[option] = tokens[0]  # Use first token ID
                    
                    # Extract log probabilities for each option
                    for option, token_id in option_token_ids.items():
                        if token_id in last_token_logprobs:
                            # logprobs returns log probabilities, which are already in log space
                            # For consistency with terminology, we'll call them "logits" (though technically they're logprobs)
                            logprob_obj = last_token_logprobs[token_id]
                            logits_dict[option] = logprob_obj.logprob if hasattr(logprob_obj, 'logprob') else float(logprob_obj)
                        else:
                            # Token not in top-k logprobs, set to a very negative value
                            logits_dict[option] = float('-inf')
                    
                    # Ensure all classification options are in logits_dict (set missing ones to -inf)
                    for option in classification_options:
                        if option not in logits_dict:
                            logits_dict[option] = float('-inf')
                    
                    # Normalize probabilities so they sum to 1
                    # Convert log probabilities to probabilities and normalize
                    logprobs_array = np.array([logits_dict[opt] for opt in classification_options])
                    # Handle -inf values by setting them to a very negative number
                    logprobs_array = np.where(np.isinf(logprobs_array), -1e10, logprobs_array)
                    # Apply softmax to normalize probabilities
                    # Subtract max for numerical stability
                    logprobs_array = logprobs_array - np.max(logprobs_array)
                    exp_logprobs = np.exp(logprobs_array)
                    probs = exp_logprobs / np.sum(exp_logprobs)
                    # Update logits_dict with normalized probabilities
                    for i, option in enumerate(classification_options):
                        logits_dict[option] = float(probs[i])
            else:
                # last_token_logprobs is None, cannot extract logits
                print(f"ERROR: Cannot extract logits - last_token_logprobs is None. logprobs length: {len(output.outputs[0].logprobs) if hasattr(output.outputs[0], 'logprobs') and output.outputs[0].logprobs else 0}")
                sys.exit(1)
        else:
            # logprobs not available, cannot extract logits
            print(f"ERROR: Cannot extract logits - logprobs not available in output")
            sys.exit(1)
        
        # Verify that all classification options are present in logits_dict
        if classification_options:
            missing_options = [opt for opt in classification_options if opt not in logits_dict]
            if missing_options:
                print(f"ERROR: Missing logits for options: {missing_options}")
                sys.exit(1)
        
        return logits_dict

    # initialize sampling params
    sampling_params_dict = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty
    }
    if return_logits:
        if classification_options:
            # Set logprobs to at least the number of classification options + some buffer
            logprobs_value = max(len(classification_options) + 10, 20)
        else:
            # If no specific options, get top 50 logprobs
            logprobs_value = 20
        sampling_params_dict["logprobs"] = logprobs_value
    
    sampling_params = SamplingParams(**sampling_params_dict)    
    # warp messages
    messages_list = []
    if args.vllm_apply_chat_template:
        for prompt in prompt_list:
            if args.llm_name == 'ministral-3-14b-instruct':
                message = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    },
                ]
            else:
                message = [{"role": "system", "content": 'You are a helpful assistant.'},
                        {"role": "user", "content": prompt}]
            # Apply chat template to convert messages to string format
            message_str = tokenizer.apply_chat_template(message, 
                                                       tokenize=False,
                                                       add_generation_prompt=True,
                                                       enable_thinking=enable_thinking)
            messages_list.append(message_str)
    else:
        messages_list = prompt_list

    
    # Checkpoint recovery disabled for open source release
    existing_responses = {}
    # if save_path:
    #     existing_responses = load_existing_responses(save_path, len(messages_list))
    
    # Determine which items need to be generated
    indices_to_generate = [i for i in range(len(messages_list)) if i not in existing_responses]
    
    # Initialize responses_list
    responses_list = [None] * len(messages_list)
    logits_list = [None] * len(messages_list) if return_logits else None

    # Load existing logits disabled for open source
    # if return_logits and save_path:
    #     existing_logits = {}
    #     if os.path.isdir(save_path):
    #         for filename in os.listdir(save_path):
    #             if filename.startswith("response_") and filename.endswith(".json"):
    #                 try:
    #                     index = int(filename.replace("response_", "").replace(".json", ""))
    #                     if 0 <= index < len(messages_list):
    #                         with open(os.path.join(save_path, filename), 'r', encoding='utf-8') as f:
    #                             data = json.load(f)
    #                             if "logits" in data:
    #                                 existing_logits[index] = data["logits"]
    #                 except (ValueError, json.JSONDecodeError, KeyError) as e:
    #                     continue
    #     for idx, logits_dict in existing_logits.items():
    #         logits_list[idx] = logits_dict

    for idx, response in existing_responses.items():
        responses_list[idx] = response
        
        # Filter messages_list to only include items that need generation
        messages_to_generate = [messages_list[i] for i in indices_to_generate]
        
        # Calculate number of batches for progress bar (only for items to generate)
        num_batches = (len(messages_to_generate) + vllm_batch_size - 1) // vllm_batch_size
        
        # Create progress bar if logger is provided
        if logger:
            progress = logger.create_progress("Generating responses with vLLM", len(messages_to_generate))
            with progress:
                task = progress.add_task("Generating responses with vLLM", total=len(messages_to_generate))
                
                for batch_idx in range(0, len(messages_to_generate), vllm_batch_size):
                    batch_end = min(batch_idx + vllm_batch_size, len(messages_to_generate))
                    current_batch_messages = messages_to_generate[batch_idx:batch_end]
                    
                    # Ensure all messages are strings (vLLM expects string list, not dict or other types)
                    current_batch_messages = [str(msg) if not isinstance(msg, str) else msg for msg in current_batch_messages]
                    
                    # generate
                    if adapter_path and adapter_path.strip():
                        # use LoRA for generation
                        from vllm.lora.request import LoRARequest
                        lora_request = LoRARequest("default", 1, adapter_path)
                        outputs = vllm_model.generate(current_batch_messages, sampling_params, lora_request=lora_request, use_tqdm=False)
                    else:
                        outputs = vllm_model.generate(current_batch_messages, sampling_params, use_tqdm=False)
                    
                    # extract answer and store in correct position
                    for local_idx, output in enumerate(outputs):
                        result = output.outputs[0].text
                        # Map back to original index
                        original_idx = indices_to_generate[batch_idx + local_idx]
                        responses_list[original_idx] = result
                        # Extract logits if requested
                        logits_dict = None
                        if return_logits:
                            logits_dict = extract_logits_for_options(output, tokenizer, classification_options)
                            logits_list[original_idx] = logits_dict                        

                        # Save response disabled for open source release
                        # if save_path:
                        #     label_value = None
                        #     if labels is not None and original_idx < len(labels):
                        #         label_value = labels[original_idx]
                        #     save_response(result, original_idx, save_path, prompt_list[original_idx], label_value)
                    
                    # update progress
                    progress.update(task, advance=len(current_batch_messages))
        else:
            # Fallback to simple print without logger
            for batch_idx in range(0, len(messages_to_generate), vllm_batch_size):
                batch_end = min(batch_idx + vllm_batch_size, len(messages_to_generate))
                current_batch_messages = messages_to_generate[batch_idx:batch_end]
                
                # Ensure all messages are strings (vLLM expects string list, not dict or other types)
                current_batch_messages = [str(msg) if not isinstance(msg, str) else msg for msg in current_batch_messages]
                
                # generate
                if adapter_path and adapter_path.strip():
                    # use LoRA for generation
                    from vllm.lora.request import LoRARequest
                    lora_request = LoRARequest("default", 1, adapter_path)
                    outputs = vllm_model.generate(current_batch_messages, sampling_params, lora_request=lora_request, use_tqdm=False)
                else:
                    outputs = vllm_model.generate(current_batch_messages, sampling_params, use_tqdm=False)
                
                # extract answer and store in correct position
                for local_idx, output in enumerate(outputs):
                    result = output.outputs[0].text
                    # Map back to original index
                    original_idx = indices_to_generate[batch_idx + local_idx]
                    responses_list[original_idx] = result
                    # Extract logits if requested
                    logits_dict = None
                    if return_logits:
                        logits_dict = extract_logits_for_options(output, tokenizer, classification_options)
                        logits_list[original_idx] = logits_dict 

                    # Save response disabled for open source release
                    # if save_path:
                    #     label_value = None
                    #     if labels is not None and original_idx < len(labels):
                    #         label_value = labels[original_idx]
                    #     save_response(result, original_idx, save_path, prompt_list[original_idx], label_value, logits_dict)
    
    # Release vLLM model memory
    if vllm_model is not None:
        try:
            if logger:
                logger.info("Releasing vLLM model memory...")
            del vllm_model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            if logger:
                logger.success("vLLM model memory released successfully")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to release vLLM model memory: {e}")
    
    if return_logits:
        return responses_list, logits_list
    else:
        return responses_list, None
    
