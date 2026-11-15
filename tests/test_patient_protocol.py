import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from data.labels import hospital_los_label, los_bin_from_hours
from data.longitudinal import build_patient_history, daily_records, prepare_longitudinal_data
from data.prepare_ehr_data import prepare_ehr_data
from data.splits import fixed_train_val_test_split, processed_patient_id


def split_args(tmp_path, **kwargs):
    values = dict(train_ratio=.8, data_split_seed=3407, seed=1, period_length=48,
                  mid_data_dump_path=str(tmp_path / 'cache'), dataset_path=str(tmp_path),
                  dataset='mimic3_los', ehr_protocol='admission_daily',
                  ehr_records_path=None, embedding_model_name=None,
                  method='graph_walker', toy_dataset=False)
    return SimpleNamespace(**dict(values, **kwargs))


def patient_record(subject=1, hadm=None, features=17):
    return {
        'subject_id': subject, 'hadm_id': hadm if hadm is not None else f'admission-{subject}',
        'hospital_admitted_at': '2020-01-01T12:00:00',
        'hospital_discharged_at': '2020-01-07T12:00:00', 'mortality': 1,
        'feature_names': [f'feature_{i}' for i in range(features)],
        'feature_types': ['continuous'] * features,
        'icu_stays': [
            {'stay_id': 'a', 'admitted_at': '2020-01-01T12:00:00', 'discharged_at': '2020-01-03T12:00:00',
             'record_time': [1, 12, 25], 'values': [[1.] * features, [3.] * features, [5.] * features],
             'icu_readmission_30d': 1},
            {'stay_id': 'b', 'admitted_at': '2020-01-04T12:00:00', 'discharged_at': '2020-01-06T12:00:00',
             'record_time': [1, 24, 36], 'values': [[7.] * features, [9.] * features, [999999.] * features],
             'icu_readmission_30d': 0},
        ],
    }


def write_patients(path, records):
    path.write_text(''.join(json.dumps(record) + '\n' for record in records))


def test_patient_splits_are_disjoint_and_order_independent(tmp_path):
    data = {'X': list(range(40)), 'subject_id': [str(i // 2) for i in range(40)]}
    args = split_args(tmp_path)
    splits = fixed_train_val_test_split(args, data, 'mimic3_los')
    groups = [set(split['subject_id']) for split in splits]
    assert [len(group) for group in groups] == [16, 2, 2]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    reversed_data = {key: list(reversed(value)) for key, value in data.items()}
    args.seed = 999
    again = fixed_train_val_test_split(args, reversed_data, 'mimic3_los')
    assert [set(split['subject_id']) for split in again] == groups
    assert set().union(*(set(split['X']) for split in splits)) == set(data['X'])


def test_mimic3_episodes_share_patient_group(tmp_path):
    data = {'X': list(range(20)), 'name': [f'{i // 2}_episode{i % 2 + 1}_timeseries.csv' for i in range(20)]}
    splits = fixed_train_val_test_split(split_args(tmp_path), data, 'mimic3_mortality')
    for split in splits:
        for subject in set(split['subject_id']):
            assert split['subject_id'].count(subject) == 2


@pytest.mark.parametrize('value', [None, '', 'null', float('nan'), True, 7.2])
def test_missing_or_ambiguous_patient_id_rejected(value):
    with pytest.raises(ValueError):
        processed_patient_id({'subject_id': value})


def test_record_id_requires_explicit_patient_mapping():
    with pytest.raises(ValueError, match='subject_id'):
        processed_patient_id({'id': 7})
    assert processed_patient_id({'id': '007'}, field='id') == '7'


@pytest.mark.parametrize('hours,expected', [(0, 0), (71.9, 0), (72, 1), (167.9, 1),
                                           (168, 1), (168.1, 2), (180, 2), (335.9, 2), (336, 2), (336.1, 3)])
def test_los_hour_boundaries(hours, expected):
    assert los_bin_from_hours(hours) == expected


@pytest.mark.parametrize('hours', [-1, float('nan'), float('inf')])
def test_invalid_los_durations(hours):
    with pytest.raises(ValueError):
        los_bin_from_hours(hours)


def test_los_uses_total_hospital_interval():
    record = patient_record()
    record['y_los'] = 0
    assert hospital_los_label(record) == 1
    sample, _ = build_patient_history(record, 'mimic3_los')
    assert sample['y'] == 1  # Six hospital days, not a two-day ICU stay.
    with pytest.raises(ValueError, match='hospital_admitted_at'):
        hospital_los_label({'y_los': 0})


def test_daily_aggregation_and_forward_fill():
    values, mask, times = daily_records(
        [[2, 1], [4, 3], [None, None], [8, 4]], [1, 2, 49, 73], 96,
        ['continuous', 'categorical'], admitted_at='2020-01-01')
    np.testing.assert_allclose(values, [[3, 3], [3, 3], [8, 4]])
    np.testing.assert_array_equal(mask[:, 0], [True, False, True])
    np.testing.assert_array_equal(times, [2, 49, 73])


@pytest.mark.parametrize('origin', ['2020-01-01T23:00:00', '2020-01-01T23:00:00+08:00'])
def test_calendar_days_not_elapsed_24_hour_bins(origin):
    values, _, times = daily_records([[2], [4], [8]], [.5, 1, 2], 24,
                                     ['continuous'], admitted_at=origin)
    np.testing.assert_array_equal(values[:, 0], [2, 6])
    np.testing.assert_array_equal(times, [.5, 2])


def test_last_seven_calendar_days_keep_earlier_locf_without_inventing_days():
    values, mask, times = daily_records([[5], [None], [9]], [1, 73, 217], 240,
                                       ['continuous'], admitted_at='2020-01-01')
    np.testing.assert_array_equal(times, [73, 217])
    np.testing.assert_array_equal(values[:, 0], [5, 9])
    np.testing.assert_array_equal(mask[:, 0], [False, True])


def test_missing_history_not_filled_from_later_day():
    values, _, _ = daily_records([[None], [4]], [1, 25], 48, ['continuous'], admitted_at='2020-01-01')
    assert np.isnan(values[0, 0]) and values[1, 0] == 4


@pytest.mark.parametrize('dataset', ['mimic3_mortality', 'mimic3_los'])
def test_target_day_excluded_from_llm_and_smart(dataset):
    record = patient_record()
    sample, _ = build_patient_history(record, dataset, need_smart=True)
    record['icu_stays'][-1]['values'][-1] = [-123456.] * 17
    record['icu_stays'].reverse()
    changed, _ = build_patient_history(record, dataset, need_smart=True)
    np.testing.assert_array_equal(sample['X'], changed['X'])
    np.testing.assert_array_equal(sample['X_ts'], changed['X_ts'])
    assert sample['input_visit_ids'] == ['2020-01-01', '2020-01-02', '2020-01-04', '2020-01-05']
    assert sample['target_visit_id'] == '2020-01-06'
    np.testing.assert_array_equal(sample['X'][:, 1], [1, 4, 7, 9])
    assert len(sample['X_ts']) == 4 and not sample['includes_target_visit']


def test_readmission_includes_index_icu_target_day_for_llm_and_smart():
    record = patient_record(features=44)
    sample, _ = build_patient_history(record, 'mimic4_readmission', need_smart=True)
    assert sample['y'] == 0  # Last ICU stay, not the first stay's label of 1.
    assert sample['target_visit_id'] in sample['input_visit_ids']
    assert sample['includes_target_visit'] and sample['X'][-1, 0] == 999999
    assert sample['X_ts'][-1, 0] == 999999
    record['icu_stays'][-1]['values'][-1] = [23.] * 44
    changed, _ = build_patient_history(record, 'mimic4_readmission', need_smart=True)
    assert changed['X'][-1, 0] == changed['X_ts'][-1, 0] == 23
    assert sample['record_time'][-1] == 108  # Last observation, before ICU discharge at 120h.


def test_one_day_admission_allowed_only_for_readmission():
    record = patient_record()
    record['icu_stays'] = [record['icu_stays'][-1]]
    record['icu_stays'][0]['record_time'] = [1]
    record['icu_stays'][0]['values'] = [[7.] * 17]
    sample, _ = build_patient_history(record, 'mimic4_readmission')
    assert len(sample['X']) == 1
    with pytest.raises(ValueError, match='preceding'):
        build_patient_history(record, 'mimic3_mortality')


@pytest.mark.parametrize('change', ['legacy', 'duplicate_stay', 'overlap', 'other_patient', 'other_admission', 'outside_hospital', 'post_discharge'])
def test_invalid_admission_records(change):
    record = patient_record()
    stay = record['icu_stays'][-1]
    if change == 'legacy':
        record['visits'] = record.pop('icu_stays')
    elif change == 'duplicate_stay':
        stay['stay_id'] = 'a'
    elif change == 'overlap':
        stay['admitted_at'] = '2020-01-02'
    elif change == 'other_patient':
        stay['subject_id'] = 2
    elif change == 'other_admission':
        stay['hadm_id'] = 'another-admission'
    elif change == 'outside_hospital':
        stay['discharged_at'] = '2020-01-08'
    else:
        stay['record_time'][-1] = 49
    with pytest.raises(ValueError):
        build_patient_history(record, 'mimic3_los')


def test_multiple_admissions_are_separate_samples_in_one_patient_split(tmp_path):
    records = [patient_record(i, f'{i}-{j}') for i in range(10) for j in range(2)]
    records[1]['icu_stays'][0]['values'][0] = [111.] * 17
    write_patients(tmp_path / 'longitudinal.jsonl', records)
    splits = prepare_longitudinal_data(split_args(tmp_path))
    owners, inputs = {}, {}
    for split_index, split in enumerate(splits):
        for subject, hadm, values in zip(split['subject_id'], split['hadm_id'], split['X']):
            assert owners.setdefault(subject, split_index) == split_index
            inputs[hadm] = values
    assert len(inputs) == 20
    assert inputs['0-0'][0, 1] == 1 and inputs['0-1'][0, 1] == 111


def test_history_scaling_uses_training_patients_only(tmp_path):
    path = tmp_path / 'longitudinal.jsonl'
    records = [patient_record(i) for i in range(20)]
    records[0]['icu_stays'][0]['values'][0][0] = None
    write_patients(path, records)
    args = split_args(tmp_path, embedding_model_name='smart')
    splits = prepare_longitudinal_data(args)
    held_out = set(splits[1]['subject_id'] + splits[2]['subject_id'])
    original_train = copy.deepcopy(splits[0]['data_smart'])
    for record in records:
        if str(record['subject_id']) in held_out:
            for stay in record['icu_stays']:
                stay['values'] = [[1e6] * 17 for _ in stay['values']]
    write_patients(path, records)
    changed = prepare_longitudinal_data(args)
    assert changed[0]['data_smart'] == original_train
    assert all(np.isfinite(sample['x']).all() for split in changed for sample in split['data_smart'])


def test_history_route_prompt_and_frozen_encoder(tmp_path):
    from models.smart import Encoder
    from run.run_smart.smart_embedding import calculate_smart_embedding
    from prompt.EHR_prompt.prompt_wraper import transform_ehr_to_detail_prompt
    path = tmp_path / 'longitudinal.jsonl'
    write_patients(path, [patient_record(i) for i in range(10)])
    args = split_args(tmp_path, embedding_model_name='smart', smart_d_model=4,
                      smart_input_dim=17, smart_max_len=48, smart_n_heads=1,
                      smart_dropout=0., smart_e_layers=1, smart_num_class=4,
                      smart_save_dir=str(tmp_path), smart_batch_size=4,
                      embedding_model_path=None, unit=False, reference_range=False)
    torch.save({'encoder': Encoder(args).state_dict()}, tmp_path / 'checkpoint-mse.pth')
    splits = prepare_ehr_data(args, Mock())
    splits = transform_ehr_to_detail_prompt(args, *splits)
    assert 'within one hospital admission' in splits[2]['detail'][0]
    assert '999999' not in splits[2]['detail'][0]
    splits = calculate_smart_embedding(args, *splits)
    assert args.smart_max_len == 4
    for split in splits:
        assert split['smart_embedding'].shape == (len(split['y']), 68)
        assert torch.isfinite(split['smart_embedding']).all()


def test_no_implicit_fallback_to_single_stay(tmp_path):
    args = split_args(tmp_path)
    with pytest.raises(FileNotFoundError, match='Admission JSONL'):
        prepare_ehr_data(args, Mock())
    args.ehr_protocol = 'single_stay'
    args.ehr_records_path = 'history.jsonl'
    with pytest.raises(ValueError, match='requires'):
        prepare_ehr_data(args, Mock())


def test_duplicate_admission_records_rejected(tmp_path):
    path = tmp_path / 'longitudinal.jsonl'
    write_patients(path, [patient_record(1), patient_record('001', 'admission-1')])
    with pytest.raises(ValueError, match='Each hadm_id'):
        prepare_longitudinal_data(split_args(tmp_path))


def test_mimic3_los_ignores_remaining_icu_label(tmp_path):
    from data.mimic3.prepare_mimic3_los import prepare_mimic3_los_extract_from_raw
    (tmp_path / 'train').mkdir()
    (tmp_path / 'test').mkdir()
    name = '1_episode1_timeseries.csv'
    (tmp_path / 'train' / name).write_text('Hours,Heart Rate\n0,80\n1,90\n')
    for split in ('train', 'val', 'test'):
        (tmp_path / f'{split}_listfile.csv').write_text(
            'stay,period_length,y_true\n' + (f'{name},24,1\n' if split == 'train' else ''))
    (tmp_path / 'hospital_los.csv').write_text(
        'name,hospital_admitted_at,hospital_discharged_at\n'
        f'{name},2020-01-01,2020-01-11\n')
    data = prepare_mimic3_los_extract_from_raw(split_args(tmp_path))
    assert data['y'] == [2]
    (tmp_path / 'hospital_los.csv').unlink()
    with pytest.raises(FileNotFoundError, match='Hospital LOS metadata'):
        prepare_mimic3_los_extract_from_raw(split_args(tmp_path))


@pytest.mark.parametrize('dataset', ['mimic4_readmission', 'mimic4_mortality', 'mimic4_los',
                                     'tjh_mortality', 'tjh_los'])
def test_processed_adapters_keep_subjects_and_finite_features(tmp_path, dataset):
    import importlib
    import pickle
    module = importlib.import_module(f'data.{dataset.split("_")[0]}.prepare_{dataset}')
    source = tmp_path / 'processed' / 'split'
    source.mkdir(parents=True)
    llm_dim, smart_dim = (19, 44) if dataset.startswith('mimic4') else (75, 75)
    records = []
    for i in range(20):
        values = np.ones((2, smart_dim))
        values[0, 0] = np.nan
        records.append({'id': i, 'subject_id': i // 2, 'x_llm_ts': np.ones((2, llm_dim)),
                        'x_ts': values, 'record_time': [1, 25],
                        'missing_mask': np.zeros((2, llm_dim)), 'y_mortality': 0,
                        'icu_readmission_30d': 0, 'y_los': 0,
                        'hospital_admitted_at': '2020-01-01',
                        'hospital_discharged_at': '2020-01-11'})
    for split in ('train', 'val', 'test'):
        with (source / f'{split}_data.pkl').open('wb') as stream:
            pickle.dump(records if split == 'train' else [], stream)
    args = split_args(tmp_path, dataset=dataset, ehr_protocol='single_stay', embedding_model_name='smart')
    output = tmp_path / 'cache' / dataset
    output.mkdir(parents=True)
    splits = module.prepare(args)
    owners = {}
    for split_index, split in enumerate(splits):
        for subject, sample, label in zip(split['subject_id'], split['data_smart'], split['y']):
            assert owners.setdefault(subject, split_index) == split_index
            assert np.isfinite(sample['x']).all()
            assert sample['x'][0][0] == 0 and sample['mask'][0][0] == 0
            assert label == (2 if dataset.endswith('_los') else 0)
    assert len(owners) == 10
    records[0].pop('subject_id')
    with (source / 'train_data.pkl').open('wb') as stream:
        pickle.dump(records, stream)
    with pytest.raises(ValueError, match='subject_id'):
        module.prepare(args)


def test_single_stay_scaling_uses_training_patients_only(tmp_path):
    from data.smart_preprocessing import standardize_smart_data
    from data.splits import patient_split_indices
    data = {'X': list(range(20)), 'subject_id': list(range(20))}
    args = split_args(tmp_path)
    _, groups = patient_split_indices(args, data, 'mimic3_mortality')
    values = [np.array([[float(i)], [float(i + 1)]]) for i in range(20)]
    masks = [np.ones((2, 1)) for _ in values]
    before = standardize_smart_data(args, data, values, masks, 'mimic3_mortality')
    for i in groups[1] + groups[2]:
        values[i] *= 1e6
    after = standardize_smart_data(args, data, values, masks, 'mimic3_mortality')
    assert [before[i] for i in groups[0]] == [after[i] for i in groups[0]]
