"""Deterministic splits with disjoint patient groups."""
import numbers
import pickle
import random
import re
from pathlib import Path


def normalize_patient_id(value):
    if isinstance(value, bool) or value is None:
        raise ValueError('Patient IDs must be nonempty strings or integers')
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, str) and value.strip():
        value = value.strip()
        if value.lower() in ('nan', 'none', 'null'):
            raise ValueError('Patient ID is missing')
        return str(int(value)) if value.isdecimal() else value
    raise ValueError('Patient IDs must be nonempty strings or integers')


def processed_patient_id(record, field=None):
    """Read an explicit patient identifier, never infer it from a visit ID."""
    field = field or next((key for key in ('subject_id', 'patient_id') if key in record), None)
    if field is None or field not in record:
        raise ValueError('Processed records require subject_id or patient_id; '
                         'set --patient_id_field only if another field identifies the patient')
    return normalize_patient_id(record[field])


def patient_ids(patient_all, dataset):
    field = next((key for key in ('subject_id', 'patient_id') if key in patient_all), None)
    if field is not None:
        return [normalize_patient_id(value) for value in patient_all[field]]
    if dataset.startswith('mimic3_') and 'name' in patient_all:
        result = []
        for name in patient_all['name']:
            match = re.fullmatch(r'(\d+)_episode\d+_timeseries\.csv', Path(str(name)).name)
            if match is None:
                raise ValueError(f'Cannot resolve MIMIC-III subject_id from sample name: {name!r}')
            result.append(normalize_patient_id(match[1]))
        return result
    raise ValueError('Patient-level splitting requires subject_id or patient_id')


def patient_split_indices(args, patient_all, dataset):
    ratio = args.train_ratio
    if not 0 < ratio < 1:
        raise ValueError('train_ratio must be between zero and one')
    n = len(patient_all['X'])
    if any(len(values) != n for values in patient_all.values()):
        raise ValueError('All patient fields must have the same length')
    subjects = patient_ids(patient_all, dataset)
    unique_subjects = sorted(set(subjects))
    if len(unique_subjects) < 3:
        raise ValueError('At least three distinct patients are required for train/val/test splits')
    split_seed = getattr(args, 'data_split_seed', 3407)
    random.Random(split_seed).shuffle(unique_subjects)
    n_train = min(max(1, int(len(unique_subjects) * ratio)), len(unique_subjects) - 2)
    n_val = max(1, (len(unique_subjects) - n_train) // 2)
    subject_groups = (set(unique_subjects[:n_train]),
                      set(unique_subjects[n_train:n_train + n_val]),
                      set(unique_subjects[n_train + n_val:]))
    groups = [[i for i, subject in enumerate(subjects) if subject in group]
              for group in subject_groups]
    return subjects, groups


def fixed_train_val_test_split(args, patient_all, dataset):
    subjects, groups = patient_split_indices(args, patient_all, dataset)
    data = dict(patient_all, subject_id=subjects)
    splits = tuple({key: [values[i] for i in group] for key, values in data.items()}
                   for group in groups)
    split_seed = getattr(args, 'data_split_seed', 3407)
    protocol = getattr(args, 'ehr_protocol', 'single_stay')
    out = Path(args.mid_data_dump_path) / dataset / (
        f'split_v3_subjects_{protocol}_seed{split_seed}_period{args.period_length}_train{args.train_ratio:g}')
    out.mkdir(parents=True, exist_ok=True)
    for name, split in zip(('train', 'val', 'test'), splits):
        with (out / f'{dataset}_{name}.pkl').open('wb') as stream:
            pickle.dump(split, stream)
    return splits
