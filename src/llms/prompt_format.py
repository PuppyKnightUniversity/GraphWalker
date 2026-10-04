"""Shared chat serialization for generation and record likelihoods."""


def format_model_prompt(args, tokenizer, prompt, target_span=None, enable_thinking=False):
    if not getattr(args, 'vllm_apply_chat_template', True):
        return (prompt, *target_span) if target_span is not None else prompt

    def render(content):
        if getattr(args, 'llm_name', '') == 'ministral-3-14b-instruct':
            chat = [{'role': 'user', 'content': [{'type': 'text', 'text': content}]}]
        else:
            chat = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': content}]
        return tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)

    formatted = render(prompt)
    if target_span is None:
        return formatted
    marker = '__ehr_user_content_729d81__'
    skeleton = render(marker)
    if skeleton.count(marker) != 1:
        raise ValueError('Chat template must contain the user content exactly once')
    prefix, suffix = skeleton.split(marker)
    if formatted != prefix + prompt + suffix:
        raise ValueError('Chat template transforms EHR text; target offsets cannot be preserved')
    start, end = target_span
    return formatted, len(prefix) + start, len(prefix) + end
