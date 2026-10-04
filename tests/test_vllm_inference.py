from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock
import sys
import math

import pytest
import torch
import transformers
from llms.vllm_inference import inference, score_classification_options


class Tokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]
    def apply_chat_template(self, chat, **kwargs):
        return 'prefix' + chat[-1]['content']


class Model:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
    def generate(self, prompts, params, **kwargs):
        self.calls.append((prompts, params))
        results = []
        for p in prompts:
            if not hasattr(params, 'prompt_logprobs'):
                results.append(SimpleNamespace(outputs=[SimpleNamespace(text='0.25')]))
            else:
                ids = p['prompt_token_ids']
                probs = [None] + [{token: SimpleNamespace(logprob=-1.)} for token in ids[1:]]
                probs[-1] = {ids[-1]: SimpleNamespace(logprob=math.log({'A': .1, 'B': .6, 'C': .2, 'D': .1}[chr(ids[-1])]))}
                results.append(SimpleNamespace(prompt_token_ids=ids, prompt_logprobs=probs))
        return results


@pytest.mark.parametrize('with_logger', [False, True])
@pytest.mark.parametrize('classification', [False, True])
def test_fresh_run_generates_without_cached_responses(monkeypatch, tmp_path, with_logger, classification):
    instance = Model()
    factory = Mock(return_value=instance)
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=factory, SamplingParams=SimpleNamespace))
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: Tokenizer())
    args = SimpleNamespace(seed=17, vllm_apply_chat_template=True, llm_name='qwen3-14b-instruct')
    logger = Mock() if with_logger else None
    if logger:
        progress = Mock()
        progress.__enter__ = Mock(return_value=progress)
        progress.__exit__ = Mock(return_value=False)
        logger.create_progress.return_value = progress
    responses, scores = inference(args, str(tmp_path), ['first', 'second', 'third'], logger=logger,
                                   vllm_batch_size=2, return_logits=classification,
                                   classification_options=['A', 'B', 'C', 'D'] if classification else None)
    assert factory.call_args.kwargs['seed'] == 17
    assert len(instance.calls) == 2
    assert responses == (['B'] * 3 if classification else ['0.25'] * 3)
    if classification:
        for p in scores:
            assert p == pytest.approx({'A': .1, 'B': .6, 'C': .2, 'D': .1})
    else:
        assert scores is None
        sent = [p for requests, _ in instance.calls for p in requests]
        assert sent == [{'prompt_token_ids': Tokenizer().encode('prefix' + value)}
                        for value in ['first', 'second', 'third']]


def test_classification_probabilities_do_not_depend_on_generated_eos(monkeypatch):
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(SamplingParams=SimpleNamespace))
    model = Model()
    scores = score_classification_options(model, Tokenizer(), ['question'], ['A', 'B', 'C', 'D'])
    assert scores[0]['B'] == pytest.approx(.6)
    assert len(model.calls[0][0]) == 4
