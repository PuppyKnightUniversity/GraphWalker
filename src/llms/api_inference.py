from openai import OpenAI
from utils.logger import EHRLogger
import time
import os
import json

class API_LLM:
    def __init__(self, model_name:str, api_key:str, base_url:str, api_setting_model_name:str, timeout: int = 100, max_retries: int = 3):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.api_setting_model_name = api_setting_model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def inference(self, prompt:str)->str:
        """
        Inference the LLMs with API with retry mechanism.
        The prompt is a string.
        Args:
            prompt: the prompt.
        Returns:
            the response.
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.api_setting_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(1)  # 等待一秒后重试
                else:
                    print(f"All {self.max_retries} attempts failed. Last error: {e}")
                    raise

def inference(model_name:str, prompt_list:list, logger:EHRLogger, timeout: int = 100, max_retries: int = 3, save_path: str = None, labels: list = None)->list:
    """
    Inference the LLMs with API.
    Args:
        model_name: the name of the model.
        prompt_list: the list of prompts.
        timeout: maximum time (in seconds) to wait for a response. Default is 100 seconds.
        max_retries: maximum number of retry attempts. Default is 3.
        save_path: directory to save individual responses as checkpoint files (optional).
        labels: optional list of labels aligned with prompt_list for saving.
    Returns:
        the list of responses.
    """

    # API configuration - Please configure your own API keys and URLs
    # You can set these via environment variables or modify this section
    # For security, it's recommended to use environment variables:
    #   export QWEN_API_KEY="your_api_key"
    #   export QWEN_BASE_URL="your_base_url"
    
    if model_name == 'qwen2-5-72b-instruct':
        # Get from environment variables or use placeholder
        api_key = os.environ.get('QWEN_API_KEY', 'YOUR_API_KEY_HERE')
        base_url = os.environ.get('QWEN_BASE_URL', 'YOUR_BASE_URL_HERE')
        api_setting_model_name = 'qwen2.5:72b'
    elif model_name == 'deepseek-r1':
        api_key = os.environ.get('DEEPSEEK_API_KEY', 'YOUR_API_KEY_HERE')
        base_url = os.environ.get('DEEPSEEK_BASE_URL', 'YOUR_BASE_URL_HERE')
        api_setting_model_name = 'deepseek-r1-64k-local'
    elif model_name == 'xiaobei-32b':
        api_key = os.environ.get('XIAOBEI_API_KEY', 'YOUR_API_KEY_HERE')
        base_url = os.environ.get('XIAOBEI_BASE_URL', 'YOUR_BASE_URL_HERE')
        api_setting_model_name = 'test'
    elif model_name == 'gpt-3.5-turbo':
        api_key = os.environ.get('OPENAI_API_KEY', 'YOUR_API_KEY_HERE')
        base_url = os.environ.get('OPENAI_BASE_URL', 'YOUR_BASE_URL_HERE')
        api_setting_model_name = 'gpt-3.5-turbo'
    elif model_name == 'gpt-5':
        api_key = os.environ.get('OPENAI_API_KEY', 'YOUR_API_KEY_HERE')
        base_url = os.environ.get('OPENAI_BASE_URL', 'YOUR_BASE_URL_HERE')
        api_setting_model_name = 'gpt-5'
    else:
        raise ValueError(f'we have not implemented the inference for {model_name}')
    
    # Validate that API credentials are configured
    if api_key == 'YOUR_API_KEY_HERE' or base_url == 'YOUR_BASE_URL_HERE':
        raise ValueError(
            f'API credentials not configured for {model_name}. '
            f'Please set the appropriate environment variables or modify the API configuration in api_inference.py'
        )
    
    # Prepare save directory if provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # Helpers for checkpointing (disabled for open source release)
    # def save_response(response: str, index: int, save_dir: str, prompt: str, label_value):
    #     file_path = os.path.join(save_dir, f"response_{index:05d}.json")
    #     with open(file_path, 'w', encoding='utf-8') as f:
    #         payload = {"index": index, "prompt": prompt, "response": response}
    #         if label_value is not None:
    #             payload["label"] = label_value
    #         json.dump(payload, f, ensure_ascii=False, indent=2)

    # def load_existing_responses(save_dir: str, total_count: int):
    #     existing = {}
    #     if save_dir and os.path.isdir(save_dir):
    #         for filename in os.listdir(save_dir):
    #             if filename.startswith("response_") and filename.endswith(".json"):
    #                 try:
    #                     idx = int(filename.replace("response_", "").replace(".json", ""))
    #                     if 0 <= idx < total_count:
    #                         with open(os.path.join(save_dir, filename), 'r', encoding='utf-8') as f:
    #                             data = json.load(f)
    #                             existing[idx] = data["response"]
    #                 except (ValueError, json.JSONDecodeError, KeyError):
    #                     continue
    #     return existing

    api_llm = API_LLM(model_name, api_key, base_url, api_setting_model_name, timeout=timeout, max_retries=max_retries)

    # Checkpoint recovery disabled for open source release
    total = len(prompt_list)
    responses_list = [None] * total
    # existing_responses = load_existing_responses(save_path, total) if save_path else {}
    # for idx, resp in existing_responses.items():
    #     responses_list[idx] = resp

    indices_to_generate = list(range(total))
    # indices_to_generate = [i for i in range(total) if i not in existing_responses]

    # if not indices_to_generate:
    #     # Everything already computed
    #     return responses_list

    progress = logger.create_progress(f"Inferencing via {model_name}", len(indices_to_generate))
    with progress:
        task = progress.add_task(f"Inferencing via {model_name}", total=len(indices_to_generate))
        for idx in indices_to_generate:
            prompt = prompt_list[idx]
            response = api_llm.inference(prompt)
            responses_list[idx] = response
            # Save response disabled for open source release
            # if save_path:
            #     label_value = None
            #     if labels is not None and idx < len(labels):
            #         label_value = labels[idx]
            #     save_response(response, idx, save_path, prompt, label_value)
            progress.update(task, advance=1)

    return responses_list