from types import SimpleNamespace
from unittest.mock import Mock
import sys

import numpy as np
import pytest
import torch
import igraph as ig
import leidenalg
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from icl.method import graph_walker as gw


@pytest.mark.parametrize('fail', [False, True])
def test_selection_cleans_up_scoring_model_on_success_and_failure(monkeypatch, fail):
    import weakref
    args = SimpleNamespace(embedding_model_name='smart', graph_walker_neighbor_num=1,
                           graph_walker_mode='frontiers-full-greedy')
    train = {'detail': ['case'], 'smart_embedding': torch.ones(1, 2)}
    logger, progress = Mock(), Mock()
    progress.__enter__ = Mock(return_value=progress)
    progress.__exit__ = Mock(return_value=False)
    logger.create_progress.return_value = progress
    references = []
    class Model:
        pass
    def load(*args, **kwargs):
        model = Model()
        model.cycle = model
        references.append(weakref.ref(model))
        return model
    def select(*args, **kwargs):
        if fail:
            raise ValueError('scoring failed')
        return []
    monkeypatch.setattr(gw, '_load_vllm_model', load)
    monkeypatch.setattr(gw, 'select_graph_walker_examples_for_single_patient', select)
    original_cleanup = gw._release_vllm_model
    cleaned = []
    def cleanup(**kwargs):
        cleaned.append(True)
        original_cleanup(**kwargs)
    monkeypatch.setattr(gw, '_release_vllm_model', cleanup)
    if fail:
        with pytest.raises(ValueError, match='scoring failed'):
            gw.select_graph_walker_examples(args, {'detail': ['target']}, train, 1, logger)
    else:
        assert gw.select_graph_walker_examples(args, {'detail': ['target']}, train, 1, logger) == [[]]
        assert references[0]() is None
    assert cleaned == [True]


@pytest.mark.parametrize('size,k', [(0, 8), (1, 8), (3, 1), (3, 8), (3, 0)])
def test_knn_is_simple_symmetric_and_handles_small_bases(size, k):
    angles = torch.tensor([0., .1, .4])[:size]
    embeddings = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    graph = gw._build_graph(None, {'smart_embedding': embeddings}, k)
    assert set(graph) == set(range(size))
    for i, neighbors in graph.items():
        assert i not in dict(neighbors)
        for j, weight in neighbors:
            assert dict(graph[j])[i] == pytest.approx(weight)
    if size == 3 and k == 1:
        assert set(dict(graph[1])) == {0, 2}  # incoming edge 2 -> 1 must be walkable


def test_leiden_uses_binary_configuration_modularity(monkeypatch):
    real_find = leidenalg.find_partition
    calls = []
    def capture(graph, partition_type, **kwargs):
        calls.append((partition_type, kwargs))
        assert 'weight' not in graph.edge_attributes()
        return real_find(graph, partition_type, **kwargs)
    monkeypatch.setattr(leidenalg, 'find_partition', capture)
    embeddings = torch.tensor([[1., 0.], [2., 0.], [-1., 0.]])
    graph = {0: [(1, .9)], 1: [(0, .9), (2, -1.)], 2: [(1, -1.)]}
    membership, cohorts, centroids, g = gw._leiden_cluster_patients(
        {'smart_embedding': embeddings}, graph, args=SimpleNamespace(seed=17))
    assert calls[0][0] is leidenalg.RBConfigurationVertexPartition
    assert calls[0][1] == {'resolution_parameter': 1.0, 'seed': 17}
    assert g.ecount() == 2
    for c, nodes in cohorts.items():
        np.testing.assert_allclose(centroids[c], embeddings[nodes].mean(0))
    # The default objective equals normalized graph modularity.
    part = leidenalg.RBConfigurationVertexPartition(
        g, membership, resolution_parameter=calls[0][1]['resolution_parameter'])
    mod = leidenalg.ModularityVertexPartition(g, membership)
    assert part.quality() / (2 * g.ecount()) == pytest.approx(mod.quality())


def test_semantic_embedding_is_used_for_anchors(monkeypatch):
    train = {'semantic_embedding': torch.eye(2), 'detail': ['one', 'two'], 'y': [0, 1]}
    args = SimpleNamespace(graph_walker_top_k_per_cohort=1,
                           graph_walker_top_l_cohorts=1,
                           graph_walker_parallel_batch_size_for_cal_greedy_score=1)
    walk = Mock(return_value=[1])
    monkeypatch.setattr(gw, '_greedy_graph_walk', walk)
    examples = gw.select_graph_walker_examples_for_single_patient(
        args, {'semantic_embedding': torch.tensor([0., 1.])}, train,
        {}, np.array([[.5, .5]]), {0: [0, 1]}, 1, emb_key='semantic_embedding')
    assert walk.call_args.kwargs['frontiers'] == {1}
    assert examples[0]['label'] == 1


def run_walk(monkeypatch, entropy, graph, frontier, budget=4, **options):
    calls = []
    def evaluate(args, patient, train, compositions, *rest):
        calls.extend(tuple(c) for c in compositions)
        return [entropy(c) for c in compositions]
    monkeypatch.setattr(gw, '_compute_cross_entropy_for_examples_batch', evaluate)
    result = gw._greedy_graph_walk(SimpleNamespace(**options), {}, {}, graph, budget,
                                  None, 4, frontier)
    return result, calls


def test_greedy_expands_frontier_and_stops_on_zero_gain(monkeypatch):
    features = {0: {0, 1, 2}, 1: {1, 2}, 2: {3}, 3: {4, 5}}
    def entropy(nodes):
        covered = set().union(*(features[n] for n in nodes))
        return 10. - len(covered)
    graph = {0: [(3, 1.)], 1: [], 2: [], 3: [(0, 1.)]}
    result, calls = run_walk(monkeypatch, entropy, graph, {0, 1, 2})
    assert result == [0, 3, 2]
    assert len(result) == len(set(result))
    assert calls == [(), (0,), (1,), (2,), (0, 1), (0, 2), (0, 3),
                     (0, 3, 1), (0, 3, 2), (0, 3, 2, 1)]


def test_greedy_recomputes_gains_after_selection(monkeypatch):
    table = {(): 10, (0,): 7, (1,): 8, (2,): 9, (0, 1): 6, (0, 2): 3}
    result, calls = run_walk(monkeypatch, lambda nodes: table[tuple(nodes)], {}, {0, 1, 2}, budget=2)
    assert result == [0, 2]
    assert calls == [(), (0,), (1,), (2,), (0, 1), (0, 2)]


@pytest.mark.parametrize('gain', [0, -1])
def test_nonpositive_gain_stops_without_padding(monkeypatch, gain):
    result, _ = run_walk(monkeypatch, lambda s: 10 - len(s) * gain, {}, {0, 1})
    assert result == []
    ablated, _ = run_walk(monkeypatch, lambda s: 10 - len(s) * gain, {}, {0, 1},
                          graph_walker_no_early_stop=True)
    assert ablated == [0, 1]


def test_empty_frontier_and_zero_budget_do_not_score(monkeypatch):
    fail = Mock(side_effect=AssertionError('unnecessary model call'))
    monkeypatch.setattr(gw, '_compute_cross_entropy_for_examples_batch', fail)
    for budget, frontier in [(0, {0}), (3, set())]:
        assert gw._greedy_graph_walk(SimpleNamespace(), {}, {}, {}, budget, None, 1, frontier) == []


def test_no_expansion_scoring_after_budget(monkeypatch):
    result, calls = run_walk(monkeypatch, lambda s: 10 - len(s), {0: [(1, 1.)]}, {0}, budget=1)
    assert result == [0]
    assert calls == [(), (0,)]


def test_invalid_entropy_is_not_silent_early_stop(monkeypatch):
    with pytest.raises(ValueError, match='non-finite'):
        run_walk(monkeypatch, lambda s: float('nan'), {}, {0})


def patient(dataset):
    return {'X': np.array([[0., 60.], [24., 70.]]), 'record_time': [0., 24.],
            'detail': 'Clinical Features Over Time:\nHeart Rate: [60, 70]', 'y': 'SECRET_TARGET'}


@pytest.mark.parametrize('dataset,label,expected', [
    ('mimic3_mortality', 1, 'mortality'),
    ('mimic3_los', np.int64(2), 'length of stay'),
    ('mimic4_readmission', 0, 'readmitted to the icu'),
])
def test_scoring_prompt_has_correct_task_and_never_target_label(dataset, label, expected):
    p = patient(dataset)
    text, start = gw._build_prompt_from_examples(
        SimpleNamespace(dataset=dataset), p, {'detail': [p['detail']], 'y': [label]},
        [0], return_target_start=True)
    assert expected in text.lower()
    assert 'SECRET_TARGET' not in text
    assert text[start:] == p['detail'] + '\n\nYour Answer:'
    assert start > text.find(p['detail'])  # identical demonstration must not capture the mask
    if dataset == 'mimic3_los':
        assert 'Label: C' in text
    assert '1 means not surviving' not in text


class ScoringModel:
    def __init__(self, tokenizer, missing=False):
        self.tokenizer, self.missing = tokenizer, missing
        self.calls = []
    def get_tokenizer(self):
        return self.tokenizer
    def generate(self, prompts, sampling_params, **kwargs):
        self.calls.extend(prompts)
        result = []
        for prompt in prompts:
            assert isinstance(prompt, dict) and 'prompt_token_ids' in prompt
            ids = prompt['prompt_token_ids']
            # Position-dependent loss makes off-by-one masks visible.
            probs = [None] + [{ids[i]: SimpleNamespace(logprob=-float(i))} for i in range(1, len(ids))]
            if self.missing:
                probs[-1] = {}
            result.append(SimpleNamespace(prompt_token_ids=ids, prompt_logprobs=probs))
        return result


@pytest.fixture
def tokenizer():
    core = Tokenizer(models.BPE(unk_token='[UNK]'))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.train_from_iterator(['Clinical Features Over Time:\nHeart Rate: [60, 70]\nLabel:',
                              'some task example readmission mortality'],
                             trainers.BpeTrainer(vocab_size=100, special_tokens=['[UNK]']))
    return PreTrainedTokenizerFast(tokenizer_object=core, unk_token='[UNK]')


@pytest.fixture(autouse=True)
def vllm_sampling_params(monkeypatch):
    # Scoring test doubles only require a SamplingParams stand-in.
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(SamplingParams=SimpleNamespace))


def test_vllm_scores_first_target_token_without_extra_shift(tokenizer):
    model = ScoringModel(tokenizer)
    result = gw._compute_cross_entropy_loss_vllm(['abc def ghi'], model, tokenizer,
                                               mask_lengths=[2], test_lengths=[5])
    assert result == [3.]  # (2 + 3 + 4) / 3, not (3 + 4) / 2


def test_full_prompt_offsets_with_repeated_markers(tokenizer):
    args = SimpleNamespace(dataset='mimic3_mortality', vllm_apply_chat_template=False)
    p = patient(args.dataset)
    train = {'detail': [p['detail']], 'y': [1]}
    model = ScoringModel(tokenizer)
    value = gw._compute_cross_entropy_for_examples_batch(args, p, train, [[0]], model, None, 2)[0]
    prompt, boundary, end_boundary = gw._build_prompt_from_examples(args, p, train, [0], return_target_span=True)
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    positions = [i for i, (start, end) in enumerate(encoded['offset_mapping']) if end > boundary and start < end_boundary and end > start]
    assert value == pytest.approx(np.mean(positions))
    assert model.calls[0]['prompt_token_ids'] == encoded['input_ids']


def test_missing_observed_token_logprob_raises(tokenizer):
    with pytest.raises(ValueError, match='Missing observed-token'):
        gw._compute_cross_entropy_loss_vllm(['abc def'], ScoringModel(tokenizer, missing=True), tokenizer)


@pytest.mark.parametrize('model_name', ['qwen3-14b-instruct', 'ministral-3-14b-instruct'])
def test_chat_scoring_span_excludes_demonstrations_and_answer_suffix(tokenizer, model_name):
    from llms.prompt_format import format_model_prompt
    tokenizer.chat_template = (
        "{% for message in messages %}{{ '<' + message['role'] + '>' }}"
        "{{ message['content'] if message['content'] is string else message['content'][0]['text'] }}"
        "{{ '</' + message['role'] + '>' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}")
    args = SimpleNamespace(dataset='mimic3_mortality', vllm_apply_chat_template=True,
                           llm_name=model_name, vllm_enable_thinking=False)
    p = patient(args.dataset)
    train = {'detail': [p['detail']], 'y': [1]}
    raw, start, end = gw._build_prompt_from_examples(args, p, train, [0], return_target_span=True)
    text, lo, hi = format_model_prompt(args, tokenizer, raw, (start, end))
    assert text == format_model_prompt(args, tokenizer, raw)
    assert text[lo:hi] == p['detail']
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    positions = [i for i, (begin, end) in enumerate(encoded['offset_mapping'])
                 if end > lo and begin < hi and end > begin]
    # Missing logprobs outside the record do not affect its likelihood.
    model = ScoringModel(tokenizer, missing=True)
    result = gw._compute_cross_entropy_for_examples_batch(args, p, train, [[0]], model, None, 2)
    assert result == pytest.approx([np.mean(positions)])
    assert model.calls[0]['prompt_token_ids'] == encoded['input_ids']
    assert positions[-1] < len(encoded['input_ids']) - 1
