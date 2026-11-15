import copy
import importlib
import json
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from data.longitudinal import prepare_longitudinal_data
from models.smart import Encoder, Classifier
from run.run_smart.dims import resolve_smart_dims, smart_checkpoint_config
from run.run_smart.smart_embedding import _load_embedding_models, calculate_smart_embedding


def args_for(tmp_path, **changes):
    values = dict(dataset='mimic4_mortality', smart_input_dim=None, smart_demo_dim=None,
                  smart_num_class=None, smart_max_len=None, period_length=48,
                  smart_d_model=4, smart_n_heads=1, smart_dropout=0., smart_e_layers=1,
                  smart_save_dir=str(tmp_path), embedding_model_path=None, smart_batch_size=2,
                  ehr_protocol='admission_daily', method='graph_walker', embedding_model_name='smart',
                  train_ratio=.8, data_split_seed=3407, seed=7, dataset_path=str(tmp_path),
                  mid_data_dump_path=str(tmp_path / 'cache'), ehr_records_path=None)
    values.update(changes)
    return SimpleNamespace(**values)


def split_with_width(width, label=0):
    return {'data_smart': [{'x': np.ones((2, width)).tolist(), 'mask': np.ones((2, width), dtype=int).tolist(),
                            'lens': 2, 'labels': label}], 'y': [label]}


@pytest.mark.parametrize('dataset,demo,classes', [('mimic3_mortality', 0, 2), ('mimic3_los', 0, 4),
                                                ('mimic4_mortality', 2, 2), ('mimic4_readmission', 2, 2),
                                                ('mimic4_los', 2, 4), ('tjh_mortality', 2, 2), ('tjh_los', 2, 4)])
def test_mapped_features_are_inferred_for_every_task(tmp_path, dataset, demo, classes):
    args = args_for(tmp_path, dataset=dataset)
    resolve_smart_dims(args, (split_with_width(63), split_with_width(63), split_with_width(63)))
    assert (args.smart_input_dim, args.smart_demo_dim, args.smart_num_class) == (63, demo, classes)
    assert args.smart_max_len == 48


def test_matching_explicit_values_and_custom_demographics_are_preserved(tmp_path):
    args = args_for(tmp_path, smart_input_dim=63, smart_demo_dim=0, smart_num_class=2, smart_max_len=24)
    resolve_smart_dims(args, (split_with_width(63),))
    assert (args.smart_input_dim, args.smart_demo_dim, args.smart_num_class, args.smart_max_len) == (63, 0, 2, 24)


@pytest.mark.parametrize('changes,match', [({'smart_input_dim': 44}, 'data has 63'),
                                         ({'smart_demo_dim': 64}, 'exceeds'),
                                         ({'smart_demo_dim': -1}, 'integer'),
                                         ({'smart_num_class': 4}, 'mortality requires 2'),
                                         ({'smart_input_dim': 0}, 'integer')])
def test_invalid_explicit_dimensions_fail(tmp_path, changes, match):
    with pytest.raises(ValueError, match=match):
        resolve_smart_dims(args_for(tmp_path, **changes), (split_with_width(63),))


def test_dimension_consistency_is_checked_across_all_splits(tmp_path):
    with pytest.raises(ValueError, match='Inconsistent SMART feature dimensions'):
        resolve_smart_dims(args_for(tmp_path), (split_with_width(63), split_with_width(63), split_with_width(62)))


@pytest.mark.parametrize('values', [[], [1, 2], [[], []], [[[1]]], [[1, 2], [3]]])
def test_malformed_smart_matrices_fail(tmp_path, values):
    with pytest.raises(ValueError, match='SMART'):
        resolve_smart_dims(args_for(tmp_path), ({'data_smart': [{'x': values}]},))


def test_empty_splits_cannot_infer_a_dimension(tmp_path):
    with pytest.raises(ValueError, match='feature dimensions'):
        resolve_smart_dims(args_for(tmp_path), ({'data_smart': []},))


def test_cli_dimensions_default_to_inference(monkeypatch):
    from args.ehrbase_args import parse_args
    monkeypatch.setattr(sys, 'argv', ['main.py'])
    args = parse_args()
    assert args.smart_input_dim is args.smart_demo_dim is args.smart_num_class is None
    monkeypatch.setattr(sys, 'argv', ['main.py', '--smart_input_dim', '63', '--smart_demo_dim', '2', '--smart_num_class', '2'])
    args = parse_args()
    assert (args.smart_input_dim, args.smart_demo_dim, args.smart_num_class) == (63, 2, 2)


@pytest.mark.parametrize('key,value', [('smart_input_dim', 62), ('smart_demo_dim', 0)])
def test_encoder_checkpoint_config_mismatch_is_rejected(tmp_path, key, value):
    args = resolve_smart_dims(args_for(tmp_path), (split_with_width(63),))
    config = smart_checkpoint_config(args)
    config[key] = value
    torch.save({'encoder': Encoder(args).state_dict(), 'config': config}, tmp_path / 'checkpoint-mse.pth')
    with pytest.raises(ValueError, match=f'Checkpoint was trained with {key}={value}'):
        _load_embedding_models(args, 'cpu')


def test_encoder_transfer_allows_different_task_class_count(tmp_path):
    args = resolve_smart_dims(args_for(tmp_path), (split_with_width(63),))
    config = dict(smart_checkpoint_config(args), smart_num_class=4)
    torch.save({'encoder': Encoder(args).state_dict(), 'config': config}, tmp_path / 'checkpoint-mse.pth')
    encoder, classifier = _load_embedding_models(args, 'cpu')
    assert encoder.query.shape[0] == 63 and classifier is None


def test_supervised_checkpoint_class_count_is_checked(tmp_path):
    args = resolve_smart_dims(args_for(tmp_path, graph_walker_add_smart_logits=True), (split_with_width(63),))
    torch.save({'encoder': Encoder(args).state_dict(), 'classifier': Classifier(args).state_dict(),
                'config': dict(smart_checkpoint_config(args), smart_num_class=4)}, tmp_path / 'checkpoint-prc.pth')
    with pytest.raises(ValueError, match='smart_num_class=4'):
        _load_embedding_models(args, 'cpu')


def test_embedding_pipeline_accepts_63_features_and_resizes_daily_positions(tmp_path):
    args = args_for(tmp_path)
    saved_args = resolve_smart_dims(copy.deepcopy(args), (split_with_width(63),))
    torch.save({'encoder': Encoder(saved_args).state_dict(), 'config': smart_checkpoint_config(saved_args)},
               tmp_path / 'checkpoint-mse.pth')
    splits = calculate_smart_embedding(args, split_with_width(63), split_with_width(63), split_with_width(63))
    assert args.smart_input_dim == 63 and args.smart_max_len == 2
    for split in splits:
        assert split['smart_embedding'].shape == (1, 252)
        assert torch.isfinite(split['smart_embedding']).all()


@pytest.mark.parametrize('source', ['mimic3', 'mimic4'])
def test_admission_jsonl_accepts_mapped_smart_schema_and_checks_cli(tmp_path, source):
    records = []
    for i in range(6):
        records.append(dict(subject_id=i, hadm_id=i, hospital_admitted_at='2020-01-01',
                            hospital_discharged_at='2020-01-04', mortality=0,
                            feature_names=['llm_feature'], feature_types=['continuous'],
                            smart_feature_names=[f'mapped_{j}' for j in range(63)],
                            smart_feature_types=['continuous'] * 63,
                            icu_stays=[dict(stay_id=i, admitted_at='2020-01-01', discharged_at='2020-01-04',
                                            record_time=[1, 25, 49], values=[[1], [2], [3]],
                                            smart_values=np.ones((3, 63)).tolist())]))
    (tmp_path / 'longitudinal.jsonl').write_text(''.join(json.dumps(record) + '\n' for record in records))
    args = args_for(tmp_path, dataset=source + '_mortality')
    splits = prepare_longitudinal_data(args)
    assert args.smart_input_dim == 63
    assert all(np.asarray(sample['x']).shape == (2, 63) for split in splits for sample in split['data_smart'])
    with pytest.raises(ValueError, match='data has 63'):
        prepare_longitudinal_data(args_for(tmp_path, dataset=source + '_mortality', smart_input_dim=44))


ADAPTERS = ['mimic4_mortality', 'mimic4_los', 'mimic4_readmission', 'tjh_mortality', 'tjh_los']


def processed_records(tmp_path, names=True):
    source = tmp_path / 'processed' / 'split'
    source.mkdir(parents=True)
    records = [dict(id=i, subject_id=i, x_llm_ts=np.ones((2, 63)), x_ts=np.ones((2, 63)),
                    missing_mask=np.zeros((2, 63)), record_time=[1, 25], y_mortality=0,
                    icu_readmission_30d=0, hospital_admitted_at='2020-01-01',
                    hospital_discharged_at='2020-01-04') for i in range(6)]
    for split in ('train', 'val', 'test'):
        with (source / f'{split}_data.pkl').open('wb') as stream:
            pickle.dump(records if split == 'train' else [], stream)
    if names:
        with (source / 'labtest_features.pkl').open('wb') as stream:
            pickle.dump([f'mapped_{i}' for i in range(61)], stream)
    return records


@pytest.mark.parametrize('dataset', ADAPTERS)
@pytest.mark.parametrize('names', [False, True])
def test_processed_adapters_accept_mapped_dimensions_and_headers(tmp_path, dataset, names):
    processed_records(tmp_path, names)
    module = importlib.import_module(f'data.{dataset.split("_")[0]}.prepare_{dataset}')
    splits = module.prepare(args_for(tmp_path, dataset=dataset, ehr_protocol='single_stay'))
    assert sum(len(split['y']) for split in splits) == 6
    for split in splits:
        assert all(np.asarray(sample['x']).shape == (2, 63) for sample in split['data_smart'])
        assert all(len(header) == 63 for header in split['header'])
        assert split['header'][0][2] == ('mapped_0' if names else 'labtest_features[0]')


@pytest.mark.parametrize('dataset', ADAPTERS)
def test_processed_adapters_reject_inconsistent_dimensions(tmp_path, dataset):
    records = processed_records(tmp_path)
    records[-1]['x_ts'] = np.ones((2, 62))
    with (tmp_path / 'processed/split/train_data.pkl').open('wb') as stream:
        pickle.dump(records, stream)
    module = importlib.import_module(f'data.{dataset.split("_")[0]}.prepare_{dataset}')
    with pytest.raises(ValueError, match='维度不匹配'):
        module.prepare(args_for(tmp_path, dataset=dataset, ehr_protocol='single_stay'))


@pytest.mark.parametrize('dataset', ADAPTERS)
def test_processed_adapters_reject_explicit_dimension_mismatch(tmp_path, dataset):
    processed_records(tmp_path)
    module = importlib.import_module(f'data.{dataset.split("_")[0]}.prepare_{dataset}')
    with pytest.raises(ValueError, match='维度不匹配'):
        module.prepare(args_for(tmp_path, dataset=dataset, smart_input_dim=44, ehr_protocol='single_stay'))


def test_header_names_must_match_data_columns(tmp_path):
    from data.smart_preprocessing import processed_feature_header
    with pytest.raises(ValueError, match='61 names'):
        processed_feature_header(np.ones((2, 63)), ['wrong'])


@pytest.mark.parametrize('name', ['run_smart_pretrain', 'run_smart_finetune'])
def test_training_launcher_forwards_explicit_dimension_options(tmp_path, monkeypatch, name):
    module = importlib.import_module('run.run_smart.run')
    launch = Mock()
    monkeypatch.setattr(module.subprocess, 'run', launch)
    args = args_for(tmp_path, smart_input_dim=63, smart_demo_dim=0, smart_num_class=2)
    getattr(module, name)(args, Mock())
    command = launch.call_args.args[0]
    assert command[command.index('--smart_input_dim') + 1] == '63'
    assert command[command.index('--smart_demo_dim') + 1] == '0'
    assert command[command.index('--smart_num_class') + 1] == '2'


@pytest.mark.parametrize('module_name,function_name', [('smart_pretrain', 'smart_pretrain'),
                                                     ('smart_finetune', 'smart_finetune'), ('smart_eval', 'run_eval')])
def test_training_and_eval_validate_dimensions_before_model_setup(tmp_path, module_name, function_name):
    module = importlib.import_module('run.run_smart.' + module_name)
    args = args_for(tmp_path, smart_input_dim=44)
    inputs = (split_with_width(63),) if module_name == 'smart_eval' else (split_with_width(63),) * 3
    with pytest.raises(ValueError, match='data has 63'):
        getattr(module, function_name)(args, *inputs)


def test_pretraining_and_finetuning_save_resolved_checkpoint_config(tmp_path, monkeypatch):
    pretrain = importlib.import_module('run.run_smart.smart_pretrain')
    finetune = importlib.import_module('run.run_smart.smart_finetune')
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *args, **kwargs: self)
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self, *args, **kwargs: self)
    for module in (pretrain, finetune):
        monkeypatch.setattr(module, 'distributed_init', lambda args: None)
        monkeypatch.setattr(module, 'init_logging', lambda *args: None)
        monkeypatch.setattr(module, 'test', Mock())
    monkeypatch.setattr(finetune, 'print_metrics_binary', lambda *args: {'auprc': .8})
    args = args_for(tmp_path, local_rank=0, distributed=False, world_size=1,
                    smart_save_model=True, smart_data_dropout=0, smart_epochs=1,
                    smart_lr=.001, smart_freeze_epochs=0, smart_min_mask_ratio=.25, smart_max_mask_ratio=.75)
    splits = (split_with_width(63), split_with_width(63), split_with_width(63))
    pretrain.smart_pretrain(args, *splits)
    pretrained = torch.load(tmp_path / 'checkpoint-mse.pth', weights_only=True)
    assert pretrained['config'] == {'smart_input_dim': 63, 'smart_demo_dim': 2, 'smart_num_class': 2}
    finetune.smart_finetune(args, *splits)
    supervised = torch.load(tmp_path / 'checkpoint-prc.pth', weights_only=True)
    assert supervised['config'] == pretrained['config']
