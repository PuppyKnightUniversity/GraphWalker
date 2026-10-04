"""Local vLLM inference and task-option likelihoods."""
import gc
import os
from contextlib import nullcontext


def score_classification_options(model, tokenizer, prompts, options, lora_request=None):
    """Score every option at the same answer boundary, including non-top-k tokens."""
    import numpy as np
    from vllm import SamplingParams

    if not options or len(set(options)) != len(options):
        raise ValueError('Classification options must be nonempty and unique')
    requests, spans = [], []
    for prompt in prompts:
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        if not prefix:
            raise ValueError('Classification prompts must be nonempty')
        for option in options:
            suffix = tokenizer.encode(option, add_special_tokens=False)
            if not suffix:
                raise ValueError(f'Empty tokenization for option {option}')
            requests.append({'prompt_token_ids': prefix + suffix})
            spans.append((len(prefix), len(prefix) + len(suffix)))
    kwargs = {'lora_request': lora_request} if lora_request is not None else {}
    outputs = model.generate(
        requests, SamplingParams(temperature=0., max_tokens=1, prompt_logprobs=1),
        use_tqdm=False, **kwargs,
    )
    if len(outputs) != len(requests):
        raise ValueError('Missing classification likelihoods')
    scores = []
    for output, request, (start, end) in zip(outputs, requests, spans):
        ids = request['prompt_token_ids']
        probs = output.prompt_logprobs
        if list(output.prompt_token_ids) != ids or probs is None or len(probs) != len(ids):
            raise ValueError('Incomplete classification prompt logprobs')
        score = 0.
        for pos in range(start, end):
            if probs[pos] is None or ids[pos] not in probs[pos]:
                raise ValueError('Missing observed option-token logprob')
            score += float(probs[pos][ids[pos]].logprob)
        if not np.isfinite(score):
            raise ValueError('Non-finite classification likelihood')
        scores.append(score)
    scores = np.asarray(scores).reshape(len(prompts), len(options))
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return [dict(zip(options, map(float, row))) for row in weights]


def inference(args, model_path: str, prompt_list: list, adapter_path: str = None,
              max_tokens: int = 4096, temperature: float = 0.7, top_p: float = 0.8,
              top_k: int = 20, repetition_penalty: float = 1.05,
              vllm_batch_size: int = 4, max_model_len: int = 16384,
              gpu_memory_utilization: float = 0.85, save_path: str = None,
              labels: list = None, logger=None, return_logits: bool = False,
              classification_options: list = None, enable_thinking: bool = False):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import torch

    if not os.path.exists(model_path):
        raise ValueError(f'Model path does not exist: {model_path}')
    if vllm_batch_size <= 0:
        raise ValueError('vllm_batch_size must be positive')
    if return_logits and not classification_options:
        raise ValueError('classification_options are required for probability output')
    if return_logits and enable_thinking:
        raise ValueError('Option likelihood scoring requires direct-answer inference')
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError('Local vLLM inference requires a CUDA GPU')
    kwargs = dict(model=model_path, tensor_parallel_size=min(gpu_count, 4),
                  trust_remote_code=True, dtype='bfloat16',
                  gpu_memory_utilization=gpu_memory_utilization,
                  max_model_len=max_model_len, swap_space=4,
                  seed=getattr(args, 'seed', 3407))
    lora_request = None
    if adapter_path and adapter_path.strip():
        from vllm.lora.request import LoRARequest
        kwargs.update(enable_lora=True, max_lora_rank=64, max_loras=1)
        lora_request = LoRARequest('default', 1, adapter_path)
    model = LLM(**kwargs)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        from llms.prompt_format import format_model_prompt
        messages = [format_model_prompt(args, tokenizer, prompt, enable_thinking=enable_thinking)
                    for prompt in prompt_list]
        sampling = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                                  top_k=top_k, repetition_penalty=repetition_penalty,
                                  seed=getattr(args, 'seed', 3407))
        responses, probabilities = [], []
        progress = logger.create_progress('Generating responses with vLLM', len(messages)) if logger else None
        with progress if progress is not None else nullcontext():
            task = progress.add_task('Generating responses with vLLM', total=len(messages)) if progress else None
            for start in range(0, len(messages), vllm_batch_size):
                batch = messages[start:start + vllm_batch_size]
                if return_logits:
                    probs = score_classification_options(model, tokenizer, batch, classification_options, lora_request)
                    probabilities.extend(probs)
                    responses.extend(max(p, key=p.get) for p in probs)
                else:
                    generate_kwargs = {'lora_request': lora_request} if lora_request is not None else {}
                    requests = [{'prompt_token_ids': tokenizer.encode(prompt, add_special_tokens=False)}
                                for prompt in batch]
                    outputs = model.generate(requests, sampling, use_tqdm=False, **generate_kwargs)
                    if len(outputs) != len(batch):
                        raise ValueError('vLLM returned an incomplete response batch')
                    responses.extend(output.outputs[0].text for output in outputs)
                if progress:
                    progress.update(task, advance=len(batch))
        return responses, probabilities if return_logits else None
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
