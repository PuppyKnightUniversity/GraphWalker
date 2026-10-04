"""Patient histories stored as JSONL, one patient per line.

Each record contains subject_id, feature_names, feature_types and visits.
Each visit contains visit_id, admitted_at, discharged_at, record_time (hours
from admission) and values (numeric rows aligned with record_time). Feature
types are continuous or categorical. SMART may use a separate smart_values
matrix with smart_feature_names and smart_feature_types on the patient record.
The last visit supplies mortality, icu_readmission_30d, or hospital admission
and discharge timestamps for LOS. ICU readmission denotes a new ICU admission
within 30 days after target ICU discharge. Its measurements are never input.
"""
import json
import math
from pathlib import Path

import numpy as np

from data.labels import hospital_los_hours, hospital_los_label, icu_readmission_label, timestamp
from data.splits import fixed_train_val_test_split, normalize_patient_id


def daily_records(values, times, duration_hours, feature_types):
    """Daily mean (continuous), last value (categorical), then forward fill.

    Retain the latest seven admission-aligned days. Forward filling uses only
    earlier observations in the same visit, including those before the window.
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if not math.isfinite(duration_hours) or duration_hours <= 0:
        raise ValueError('Visit duration must be positive and finite')
    if times.ndim != 1 or values.ndim != 2 or values.shape != (len(times), len(feature_types)) or not len(times):
        raise ValueError('Measurements and times must be nonempty and aligned')
    if not np.isfinite(times).all() or np.isinf(values).any():
        raise ValueError('Times must be finite and feature values cannot be infinite')
    if any(kind not in ('continuous', 'categorical') for kind in feature_types):
        raise ValueError('Feature types must be continuous or categorical')
    if np.any(times < 0) or np.any(times > duration_hours):
        raise ValueError('Measurements must fall inside the visit')
    order = np.argsort(times, kind='stable')
    values, times = values[order], times[order]
    n_days = max(1, math.ceil(duration_hours / 24))
    bins = np.minimum((times // 24).astype(int), n_days - 1)
    first_day = max(0, n_days - 7)
    carry = np.full(values.shape[1], np.nan)
    rows, masks, ends = [], [], []
    # Empty days outside the retained window need no materialization.
    for day in sorted(set(bins) | set(range(first_day, n_days))):
        observed = np.zeros(values.shape[1], dtype=bool)
        entries = values[bins == day]
        for column, kind in enumerate(feature_types):
            valid = entries[:, column][np.isfinite(entries[:, column])]
            if len(valid):
                carry[column] = valid.mean() if kind == 'continuous' else valid[-1]
                observed[column] = True
        if day >= first_day:
            rows.append(carry.copy())
            masks.append(observed)
            ends.append(min(24 * (day + 1), duration_hours))
    return np.asarray(rows), np.asarray(masks), np.asarray(ends)


def _feature_schema(record, smart=False):
    prefix = 'smart_' if smart else ''
    names = record.get(prefix + 'feature_names')
    kinds = record.get(prefix + 'feature_types')
    if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f'{prefix}feature_names must be a nonempty list of names')
    if len(set(names)) != len(names) or not isinstance(kinds, list) or len(kinds) != len(names):
        raise ValueError('Feature names must be unique and aligned with feature types')
    if any(kind not in ('continuous', 'categorical') for kind in kinds):
        raise ValueError('Feature types must be continuous or categorical')
    return tuple(names), tuple(kinds)


def build_patient_history(record, dataset, need_smart=False):
    if dataset not in ('mimic3_mortality', 'mimic3_los', 'mimic4_mortality', 'mimic4_los',
                       'mimic4_readmission', 'tjh_mortality', 'tjh_los'):
        raise ValueError(f'Unsupported EHR dataset: {dataset}')
    subject = normalize_patient_id(record.get('subject_id'))
    names, kinds = _feature_schema(record)
    visits = record.get('visits')
    if not isinstance(visits, list) or len(visits) < 2:
        raise ValueError(f'Patient {subject} requires at least one historical and one target visit')
    visits = sorted(visits, key=lambda visit: timestamp(visit['admitted_at']))
    for visit in visits:
        if 'subject_id' in visit and normalize_patient_id(visit['subject_id']) != subject:
            raise ValueError('Visit subject_id differs from its patient record')
    visit_ids = [normalize_patient_id(visit['visit_id']) for visit in visits]
    if len(set(visit_ids)) != len(visit_ids):
        raise ValueError(f'Duplicate visit IDs for patient {subject}')
    durations = [hospital_los_hours(visit['admitted_at'], visit['discharged_at']) for visit in visits]
    for previous, current in zip(visits, visits[1:]):
        if timestamp(previous['discharged_at']) > timestamp(current['admitted_at']):
            raise ValueError(f'Overlapping visits for patient {subject}')
    target = visits[-1]
    if dataset.endswith('_los'):
        label = hospital_los_label(target)
        if not (timestamp(target['hospital_admitted_at']) <= timestamp(target['admitted_at'])
                < timestamp(target['discharged_at']) <= timestamp(target['hospital_discharged_at'])):
            raise ValueError('The target visit must fall inside its hospital admission')
    elif dataset.endswith('_readmission'):
        label = icu_readmission_label(target)
    else:
        key = 'mortality'
        label = target[key]
        if label not in (0, 1):
            raise ValueError(f'{key} must be a binary outcome')
        label = int(label)
    smart_schema = None
    if need_smart:
        smart_schema = _feature_schema(record, smart=True) if 'smart_feature_names' in record else (names, kinds)
        input_dim = 17 if dataset.startswith('mimic3_') else 44 if dataset.startswith('mimic4_') else 75
        if len(smart_schema[0]) != input_dim:
            raise ValueError(f'{dataset} SMART input requires {input_dim} features in checkpoint order')
    origin = timestamp(visits[0]['admitted_at'])
    all_values, all_times, row_visits, all_smart, all_masks = [], [], [], [], []
    # Aggregate historical visits in chronological order.
    for visit, visit_id, duration in zip(visits[:-1], visit_ids[:-1], durations[:-1]):
        values, _, times = daily_records(visit['values'], visit['record_time'], duration, kinds)
        offset = (timestamp(visit['admitted_at']) - origin).total_seconds() / 3600
        all_values.append(values)
        all_times.extend((times + offset).tolist())
        row_visits.extend([visit_id] * len(times))
        if need_smart:
            if smart_schema == (names, kinds) and 'smart_values' not in visit:
                smart_values = visit['values']
            else:
                smart_values = visit['smart_values']
            smart, mask, _ = daily_records(smart_values, visit['record_time'], duration, smart_schema[1])
            all_smart.append(smart)
            all_masks.append(mask)
    values = np.concatenate(all_values)
    is_mimic3 = dataset.startswith('mimic3_')
    sample = {
        'subject_id': subject, 'name': f'{subject}:{visit_ids[-1]}',
        'target_visit_id': visit_ids[-1], 'input_visit_ids': visit_ids[:-1],
        'row_visit_ids': row_visits, 'record_time': all_times,
        'X': np.column_stack((all_times, values)) if is_mimic3 else values,
        'header': ['Hours', *names] if is_mimic3 else list(names),
        't': len(values), 'y': label, 'data_protocol': 'historical_visits',
    }
    if need_smart:
        sample['X_ts'] = np.concatenate(all_smart)
        sample['smart_observed_mask'] = np.concatenate(all_masks)
    return sample, (names, kinds, smart_schema)


def normalize_smart_splits(splits):
    """Fit scaling on training histories; apply it to all splits."""
    training = np.concatenate(splits[0]['X_ts'])
    finite = np.isfinite(training)
    counts = finite.sum(axis=0)
    mean = np.where(finite, training, 0).sum(axis=0) / np.maximum(counts, 1)
    variance = np.where(finite, (training - mean) ** 2, 0).sum(axis=0) / np.maximum(counts, 1)
    scale = np.sqrt(variance)
    scale = np.where(scale > 0, scale, 1)
    for split in splits:
        samples = []
        for values, mask, label in zip(split['X_ts'], split['smart_observed_mask'], split['y']):
            x = np.where(np.isfinite(values), (values - mean) / scale, 0).astype(np.float32)
            if not np.isfinite(x).all():
                raise ValueError('Nonfinite normalized SMART features')
            samples.append({'x': x.tolist(), 'mask': mask.astype(int).tolist(),
                            'lens': len(x), 'labels': label})
        split['data_smart'] = samples
    return splits


def prepare_longitudinal_data(args):
    if args.dataset.startswith('tjh_'):
        raise ValueError('Patient-level training splits require a MIMIC dataset')
    path = Path(getattr(args, 'ehr_records_path', None) or Path(args.dataset_path) / 'longitudinal.jsonl')
    if not path.is_file():
        raise FileNotFoundError(f'Patient JSONL file not found: {path}. Set --ehr_records_path.')
    need_smart = args.method == 'ehr_model_smart' or getattr(args, 'embedding_model_name', None) == 'smart'
    data, schema, subjects = None, None, set()
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                sample, current_schema = build_patient_history(json.loads(line), args.dataset, need_smart)
                if schema is not None and current_schema != schema:
                    raise ValueError('Feature schema differs between patients')
                if sample['subject_id'] in subjects:
                    raise ValueError('Each patient must appear in exactly one JSONL record')
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f'{path}:{line_number}: {error}') from error
            subjects.add(sample['subject_id'])
            schema = current_schema
            if data is None:
                data = {key: [] for key in sample}
            for key, value in sample.items():
                data[key].append(value)
    if data is None:
        raise ValueError('The patient JSONL file is empty')
    splits = fixed_train_val_test_split(args, data, args.dataset)
    return normalize_smart_splits(splits) if need_smart else splits
