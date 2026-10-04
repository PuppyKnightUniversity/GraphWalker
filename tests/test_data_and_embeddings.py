from types import SimpleNamespace
from unittest.mock import Mock
from contextlib import nullcontext
import sys
import pickle
import random

import numpy as np
import pytest
import torch

from data.splits import fixed_train_val_test_split
from models.smart import Encoder, Classifier
from run.run_smart.smart_embedding import _load_embedding_models, _encoder_cls_embedding
from prompt.EHR_prompt.common import format_icl_label, measurement_times
from utils import conditional_entropy as ce


def test_fixed_splits_across_experiment_seeds_and_cache_windows(tmp_path):
    data = {'X': list(range(100)), 'y': list(range(100)), 'subject_id': list(range(100))}
    args = SimpleNamespace(train_ratio=.8, data_split_seed=3407, seed=1,
                           period_length=24, mid_data_dump_path=str(tmp_path))
    random.seed(5)
    before = random.getstate()
    first = fixed_train_val_test_split(args, data, 'mimic3_los')
    assert random.getstate() == before
    args.seed = 99
    assert first == fixed_train_val_test_split(args, data, 'mimic3_los')
    assert [len(d['X']) for d in first] == [80, 10, 10]
    assert len(set(first[0]['X']) | set(first[1]['X']) | set(first[2]['X'])) == 100
    changed = {'X': list(range(100, 200)), 'y': list(range(100)), 'subject_id': list(range(100))}
    args.period_length = 48
    second = fixed_train_val_test_split(args, changed, 'mimic3_los')
    assert second != first
    assert len(list(tmp_path.rglob('*_test.pkl'))) == 2


def smart_args(tmp_path):
    return SimpleNamespace(smart_d_model=4, smart_input_dim=2, smart_max_len=3,
                           smart_n_heads=1, smart_dropout=0., smart_e_layers=1,
                           smart_num_class=2, smart_save_dir=str(tmp_path),
                           embedding_model_path=None)


def test_smart_dropout_clears_observation_mask():
    from run.run_smart.smart_embedding import CustomDataset
    dataset = CustomDataset([{'x': [[7, 0]], 'mask': [[1, 0]], 'labels': 1}])
    dataset.dropout_data(drop_rate=1.0)
    assert dataset[0] == {'x': [[0, 0]], 'mask': [[0, 0]], 'labels': 1}


def test_pretrained_checkpoint_requires_no_classifier_and_returns_frozen_cls(tmp_path):
    args = smart_args(tmp_path)
    torch.manual_seed(7)
    original = Encoder(args).eval()
    torch.save({'encoder': original.state_dict(), 'predictor': {}, 'epoch': 1},
               tmp_path / 'checkpoint-mse.pth')
    # A supervised checkpoint is present too; it must never be preferred.
    torch.save({'encoder': original.state_dict(), 'classifier': Classifier(args).state_dict()},
               tmp_path / 'checkpoint-prc.pth')
    encoder, classifier = _load_embedding_models(args, torch.device('cpu'))
    assert classifier is None and not encoder.training
    assert all(not p.requires_grad for p in encoder.parameters())
    batch = {'x': torch.rand(2, 3, 2), 'mask': torch.ones(2, 3, 2), 'lens': torch.tensor([3, 3])}
    h = encoder(**batch)
    torch.testing.assert_close(h, original(**batch))
    embeddings = _encoder_cls_embedding(h)
    assert embeddings.shape == (2, 8)
    torch.testing.assert_close(embeddings, h[:, :, 0, :].flatten(1))


def test_supervised_checkpoint_is_explicit_ablation(tmp_path):
    args = smart_args(tmp_path)
    path = tmp_path / 'checkpoint-prc.pth'
    torch.save({'encoder': Encoder(args).state_dict(), 'classifier': Classifier(args).state_dict()}, path)
    with pytest.raises(FileNotFoundError, match='checkpoint-mse'):
        _load_embedding_models(args, 'cpu')
    args.embedding_model_path = str(path)
    with pytest.raises(ValueError, match='supervised classifier'):
        _load_embedding_models(args, 'cpu')
    args.graph_walker_add_smart_logits = True
    _, classifier = _load_embedding_models(args, 'cpu')
    assert classifier is not None


def test_checkpoint_missing_learned_weight_fails(tmp_path):
    args = smart_args(tmp_path)
    weights = Encoder(args).state_dict()
    weights.pop('query')
    torch.save({'encoder': weights}, tmp_path / 'checkpoint-mse.pth')
    with pytest.raises(RuntimeError, match='query'):
        _load_embedding_models(args, 'cpu')


@pytest.mark.parametrize('label', [2, 2., np.int64(2), torch.tensor(2), 'C'])
def test_los_scalar_types(label):
    assert format_icl_label(label, 'mimic3_los') == 'C'


def test_readmission_uses_real_timestamps_not_sex_column():
    patient = {'X': np.array([[1, 65], [1, 65]]), 'record_time': [5., 29.]}
    np.testing.assert_array_equal(measurement_times(patient, 'mimic4_readmission'), [5., 29.])
    del patient['record_time']
    with pytest.raises(ValueError, match='record_time'):
        measurement_times(patient, 'mimic4_readmission')


def test_readmission_loader_preserves_timestamps(tmp_path):
    from data.mimic4.prepare_mimic4_readmission import prepare_mimic4_readmission_extract_from_raw
    source = tmp_path / 'processed' / 'split'
    source.mkdir(parents=True)
    record = {'x_llm_ts': np.ones((2, 19)), 'x_ts': np.ones((2, 44)),
              'record_time': [5, 29], 'icu_readmission_30d': 0, 'id': 7, 'subject_id': 70,
              'missing_mask': np.zeros((2, 19))}
    for split in ('train', 'val', 'test'):
        with (source / f'{split}_data.pkl').open('wb') as stream:
            pickle.dump([record] if split == 'train' else [], stream)
    args = SimpleNamespace(period_length=24, dataset_path=str(tmp_path),
                           mid_data_dump_path=str(tmp_path / 'cache'))
    data = prepare_mimic4_readmission_extract_from_raw(args)
    assert data['record_time'] == [[5, 29]]
    assert prepare_mimic4_readmission_extract_from_raw(args)['record_time'] == [[5, 29]]


def test_entropy_report_distinguishes_H_from_delta_H(monkeypatch):
    logger = Mock()
    progress = Mock()
    progress.__enter__ = Mock(return_value=progress)
    progress.__exit__ = Mock(return_value=False)
    logger.create_progress.return_value = progress
    monkeypatch.setattr(ce, '_load_vllm_model', lambda *a, **k: None)
    monkeypatch.setattr(ce, '_release_vllm_model', lambda *a, **k: None)
    monkeypatch.setattr(ce, '_compute_cross_entropy_for_examples',
                        lambda args, patient, train, indices, *rest: 3. if indices else 5.)
    train = {'detail': ['example'], 'y': [0]}
    selected = [[{'detail': 'example', 'label': 0, 'node_index': 0}]]
    result = ce.compute_average_conditional_entropy(
        SimpleNamespace(), train, {'detail': ['target']}, selected, logger, return_metrics=True)
    assert result == {'avg_conditional_entropy': 3., 'avg_delta_H': 2.}
    with pytest.raises(ValueError, match='not found'):
        ce._convert_icl_examples_to_node_indices([{'detail': 'wrong'}], train)


def test_graph_hyperparameter_defaults(monkeypatch):
    from args.ehrbase_args import parse_args
    monkeypatch.setattr(sys, 'argv', ['main.py'])
    args = parse_args()
    assert args.graph_walker_mode == 'frontiers-full-greedy'
    assert args.ehr_protocol == 'historical_visits'
    assert (args.graph_walker_neighbor_num, args.graph_walker_top_l_cohorts,
            args.graph_walker_top_k_per_cohort, args.graph_walker_leiden_resolution) == (8, 3, 3, 1.0)


def test_invalid_cli_flags_do_not_fall_back_to_defaults(monkeypatch):
    from args.ehrbase_args import parse_args
    monkeypatch.setattr(sys, 'argv', ['main.py', '--misspelled-experiment-option'])
    with pytest.raises(SystemExit):
        parse_args()
