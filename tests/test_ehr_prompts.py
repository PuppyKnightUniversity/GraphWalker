import copy
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from data.labels import icu_readmission_label, los_bin_from_hours
from icl.method.graph_walker import _build_prompt_from_examples
from prompt.EHR_prompt.common import build_ehr_prompt, serialize_patient_record
from prompt.EHR_prompt.prompt_wraper import transform_ehr_to_detail_prompt


TASKS = ['mimic3_mortality', 'mimic3_los', 'mimic4_readmission',
         'mimic4_mortality', 'tjh_mortality']


def patient(dataset):
    values = np.array([[65., np.nan], [70., 99.]])
    times = [25.25, 82.5]
    names = ['Heart Rate', 'Oxygen saturation']
    if dataset.startswith('mimic3'):
        values = np.column_stack((times, values))
        names = ['Hours'] + names
    return dict(X=values, header=names, record_time=times, t=2, y='SECRET_OUTCOME',
                data_protocol='historical_visits', input_visit_ids=['old_a', 'old_b'],
                row_visit_ids=['old_a', 'old_b'], name='patient')


def wrapper(dataset):
    cohort, outcome = dataset.split('_', 1)
    module = importlib.import_module(f'prompt.EHR_prompt.{cohort}.{outcome}.prompt')
    return getattr(module, dataset + '_prompt_wrapper')


@pytest.mark.parametrize('dataset', TASKS)
@pytest.mark.parametrize('few_shot', [False, True])
def test_scoring_and_task_inference_share_historical_records(dataset, few_shot):
    p = patient(dataset)
    args = SimpleNamespace(dataset=dataset, unit=False, reference_range=False)
    raw = {k: [v] for k, v in p.items()}
    prepared = transform_ehr_to_detail_prompt(args, copy.deepcopy(raw), copy.deepcopy(raw), raw)[2]
    p = {k: v[0] for k, v in prepared.items()}
    label = np.int64(2) if dataset.endswith('_los') else np.int64(1)
    example = dict(p, label=label)
    train = {k: [v] for k, v in p.items()}
    train['y'] = [label]
    examples, indices = ([example], [0]) if few_shot else ([], [])
    final = wrapper(dataset)(p, is_few_shot=few_shot, icl_examples_list=examples)
    scored, start, end = _build_prompt_from_examples(args, p, train, indices, return_target_span=True)
    assert final == scored
    assert scored[start:end] == p['detail']
    assert scored[end:] == '\n\nYour Answer:'
    assert final.count(p['detail']) == (2 if few_shot else 1)
    assert 'SECRET_OUTCOME' not in final
    assert '[25.25, 82.50]' in final
    assert 'Visit index for each measurement: [1, 2]' in final
    assert 'hours since the first historical admission' in final
    assert 'NaN' in final and 'nan' not in final
    assert 'Expert Model' not in final
    if dataset.endswith('_readmission'):
        assert 'discharge from the target ICU stay' in final
        assert 'new ICU admission' in final
        assert 'death' not in final and 'dies' not in final
    if few_shot:
        assert ('Outcome Label: C' if dataset.endswith('_los') else 'Outcome Label: 1') in final


def test_empty_selected_set_uses_zero_shot_template():
    p = patient('mimic3_mortality')
    assert wrapper('mimic3_mortality')(p, is_few_shot=True, icl_examples_list=[]) == build_ehr_prompt(p, 'mimic3_mortality')


def test_target_label_and_expert_logits_are_unused_by_default():
    p = patient('mimic4_readmission')
    baseline = build_ehr_prompt(p, 'mimic4_readmission')
    p.update(y='DIFFERENT_SECRET', smart_logits=[-100., 100.], target_visit_id='PRIVATE_VISIT')
    assert build_ehr_prompt(p, 'mimic4_readmission') == baseline


@pytest.mark.parametrize('dataset', ['mimic3_los', 'mimic4_readmission'])
def test_expert_probabilities_are_optional_and_outside_record_loss(dataset):
    p = patient(dataset)
    p['smart_logits'] = [0., 1., 2., 3.] if dataset.endswith('_los') else [0., 1.]
    label = 2 if dataset.endswith('_los') else 1
    example = dict(p, label=label)
    text, start, end = build_ehr_prompt(
        p, dataset, [example], add_smart_logits=True,
        add_smart_logits_for_test_example=True, return_target_span=True)
    assert text.count('Expert Model Probabilities:') == 2
    assert 'Expert Model Probabilities:' not in text[start:end]
    assert text.endswith('Your Answer:')
    if dataset.endswith('_los'):
        assert all(f'{c}: ' in text for c in 'ABCD')
    del p['smart_logits']
    with pytest.raises(ValueError, match='require smart_logits'):
        build_ehr_prompt(p, dataset, add_smart_logits_for_test_example=True)


def test_single_stay_input_cannot_be_described_as_preceding_visits():
    p = patient('mimic4_readmission')
    p['data_protocol'] = 'single_stay'
    with pytest.raises(ValueError, match='preceding visits'):
        build_ehr_prompt(p, 'mimic4_readmission')


@pytest.mark.parametrize('days,letter', [(2.99, 'A'), (3, 'B'), (7, 'B'), (7.5, 'C'),
                                         (8, 'C'), (14, 'C'), (14.01, 'D')])
def test_continuous_los_category_labels(days, letter):
    assert 'ABCD'[los_bin_from_hours(days * 24)] == letter


def test_readmission_requires_an_explicit_icu_outcome():
    with pytest.raises(ValueError, match='readmission_definition'):
        icu_readmission_label({'readmission_30d': 1, 'mortality': 1})
    assert icu_readmission_label({'icu_readmission_30d': 0, 'mortality': 1}) == 0
    assert icu_readmission_label({'icu_readmission_30d': 1, 'mortality': 0}) == 1
    assert icu_readmission_label({'readmission_definition': 'icu_30d', 'y_readmission': [1]}) == 1
    with pytest.raises(ValueError, match='binary'):
        icu_readmission_label({'icu_readmission_30d': .7})


def test_sft_uses_the_shared_prediction_query():
    from run.run_sft.sft_train import _build_zero_shot_prompt, SFTDataset
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    dataset = 'mimic3_los'
    p = patient(dataset)
    p['y'] = 2
    p['detail'] = serialize_patient_record(p, dataset)
    prompt = _build_zero_shot_prompt(p, dataset)
    assert prompt == build_ehr_prompt(p, dataset)
    core = Tokenizer(models.BPE(unk_token='[UNK]'))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.train_from_iterator([prompt, 'C'], trainers.BpeTrainer(vocab_size=300, special_tokens=['[UNK]', '[EOS]']))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, unk_token='[UNK]', eos_token='[EOS]')
    args = SimpleNamespace(dataset=dataset, vllm_apply_chat_template=False)
    data = {k: [v] for k, v in p.items()}
    item = SFTDataset(data, tokenizer, 4096, args=args)[0]
    unmasked = item['labels'][item['labels'] != -100].tolist()
    assert unmasked == tokenizer.encode('C', add_special_tokens=False) + [tokenizer.eos_token_id]
    with pytest.raises(ValueError, match='max_length'):
        SFTDataset(data, tokenizer, 1, args=args)[0]


def test_numeric_feature_serialization_preserves_scientific_notation():
    p = patient('mimic3_mortality')
    p['X'][0, 1] = 1e-5
    text = serialize_patient_record(p, 'mimic3_mortality')
    assert 'Heart Rate: [1e-05, 70.0]' in text


def test_legacy_prompt_entry_point_uses_shared_templates():
    from prompt.prompt_wraper import mimic3_mortality_prompt_wrapper
    from prompt.prompt_template import USERPROMPT_ZERO_SHOT
    from prompt.EHR_prompt.prompt_template import USERPROMPT_ZERO_SHOT as shared
    p = patient('mimic3_mortality')
    assert mimic3_mortality_prompt_wrapper(p) == build_ehr_prompt(p, 'mimic3_mortality')
    assert USERPROMPT_ZERO_SHOT == shared


@pytest.mark.parametrize('module_name', ['sft_eval', 'sft_eval_vllm'])
@pytest.mark.parametrize('dataset', ['mimic3_los', 'mimic4_readmission'])
def test_sft_eval_preserves_task_and_history(module_name, dataset):
    module = importlib.import_module('run.run_sft.' + module_name)
    p = patient(dataset)
    args = SimpleNamespace(dataset=dataset, unit=False, reference_range=False, inference_type='only_answer')
    data = {k: [v] for k, v in p.items()}
    data = module._ensure_detail(args, data)
    assert module._build_prompts(args, data) == [build_ehr_prompt(p, dataset)]


def test_sft_vllm_passes_los_scores_to_evaluation(monkeypatch):
    from unittest.mock import Mock, create_autospec
    from run.run_sft import sft_eval_vllm
    p = patient('mimic3_los')
    p['y'] = 2
    data = {k: [v] for k, v in p.items()}
    args = SimpleNamespace(dataset='mimic3_los', unit=False, reference_range=False,
                           inference_type='only_answer', llm_local_path='model', llm_name='model',
                           llm_adapter_path='adapter', sft_output_dir='output', llm_responses_save_path=None)
    scores = [{'A': .1, 'B': .2, 'C': .6, 'D': .1}]
    generate = Mock(return_value=(['C'], scores))
    evaluate = create_autospec(sft_eval_vllm.llm_response_evaluation, return_value={})
    monkeypatch.setattr(sft_eval_vllm, 'vllm_generate', generate)
    monkeypatch.setattr(sft_eval_vllm, 'llm_response_evaluation', evaluate)
    sft_eval_vllm.run(args, {}, {}, data, Mock())
    assert generate.call_args.kwargs['args'] is args
    assert generate.call_args.kwargs['classification_options'] == list('ABCD')
    assert evaluate.call_args.args[2] == scores
    assert evaluate.call_args.args[1] == ['C']


def test_hf_los_scoring_uses_all_class_probabilities():
    import torch
    from run.run_sft.sft_eval import _score_los_options
    class Tokenizer:
        def encode(self, value, **kwargs):
            return [ord(c) for c in value]
    class Model(torch.nn.Module):
        def forward(self, input_ids):
            logits = torch.full((*input_ids.shape, 128), -50., device=input_ids.device)
            for label, probability in zip('ABCD', [.1, .2, .6, .1]):
                logits[..., ord(label)] = np.log(probability)
            return SimpleNamespace(logits=logits)
    result = _score_los_options(Model(), Tokenizer(), ['patient\nYour Answer:'])
    assert result[0] == pytest.approx(dict(zip('ABCD', [.1, .2, .6, .1])))


@pytest.mark.parametrize('method', ['llm_zero_shot', 'graph_walker'])
def test_cli_defaults_complete_prompt_pipeline(monkeypatch, method):
    import sys
    from unittest.mock import Mock
    from args.ehrbase_args import parse_args
    from prompt.EHR_prompt.prompt_wraper import train_val_test_dataset_prompt_wrapper
    from run.run_smart import smart_embedding
    from icl import select_icl_examples

    command = ['main.py', '--dataset', 'mimic3_mortality', '--dataset_path', '/data',
               '--llm_local_path', '/model', '--method', method, '--use_vllm']
    if method == 'graph_walker':
        command += ['--embedding_model_name', 'smart', '--embedding_model_path', '/encoder']
    monkeypatch.setattr(sys, 'argv', command)
    args = parse_args()
    assert args.method == method
    assert not hasattr(args, 'llm_smart_embedding_topk_add_smart_logits')
    p = patient(args.dataset)
    p.update(y=0, data_smart={})
    splits = [{k: [v] for k, v in p.items()} for _ in range(3)]
    logger, progress = Mock(), Mock()
    progress.__enter__ = Mock(return_value=progress)
    progress.__exit__ = Mock(return_value=False)
    logger.create_progress.return_value = progress
    monkeypatch.setitem(sys.modules, 'tiktoken', SimpleNamespace(
        get_encoding=lambda name: SimpleNamespace(encode=lambda text: list(text))))
    monkeypatch.setattr(smart_embedding, 'calculate_smart_embedding', lambda args, *data: data)
    def select(args, test_dataset, *, train_dataset, **kwargs):
        return [[{'detail': train_dataset['detail'][0], 'label': train_dataset['y'][0]}]]
    monkeypatch.setattr(select_icl_examples, 'FIND_ICL_EXAMPLES', select)
    _, _, test = train_val_test_dataset_prompt_wrapper(
        args, logger, *splits, is_few_shot=method == 'graph_walker')
    prompt = test['data_prompt_fomat'][0]
    assert prompt.endswith('Your Answer:')
    assert ('Selected Patient Examples:' in prompt) == (method == 'graph_walker')
    assert 'Expert Model Probabilities:' not in prompt
