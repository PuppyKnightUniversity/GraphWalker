"""Admission-level JSONL inputs for the paper's daily-observation protocol.

One record is one hospital admission, identified by subject_id and hadm_id.
MIMIC measurements are nested in icu_stays; TJH laboratory measurements are on
an admission record. record_time is in hours from the containing stay/admission.
See README.md for the schema. Legacy cross-admission visits are rejected.
"""
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from data.labels import hospital_los_hours, hospital_los_label, icu_readmission_label, timestamp
from data.splits import fixed_train_val_test_split, normalize_patient_id


def daily_records(values, times, duration_hours, feature_types, *, admitted_at,
                  continuous_agg='last', max_days=7, forward_fill=True):
    """Aggregate one admission's observations by calendar day.

    MIMIC uses the last recorded value of each variable per day, retains the
    final ``max_days`` observed days, and merges all earlier observations into
    one initial step. TJH uses the daily mean without a day cap. ``forward_fill``
    applies LOCF within the admission for the encoder; without it, missing LLM
    entries stay NaN. No unobserved days are materialized. Returned times mark
    the last measurement of each step, in hours from hospital admission.
    Calendar boundaries use the admission timestamp's timezone.
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if not math.isfinite(duration_hours) or duration_hours <= 0:
        raise ValueError('Admission duration must be positive and finite')
    if times.ndim != 1 or values.ndim != 2 or values.shape != (len(times), len(feature_types)) or not len(times):
        raise ValueError('Measurements and times must be nonempty and aligned')
    if not np.isfinite(times).all() or np.isinf(values).any():
        raise ValueError('Times must be finite and feature values cannot be infinite')
    if any(kind not in ('continuous', 'categorical') for kind in feature_types):
        raise ValueError('Feature types must be continuous or categorical')
    if continuous_agg not in ('last', 'mean'):
        raise ValueError('continuous_agg must be last or mean')
    if max_days is not None and (isinstance(max_days, (bool, np.bool_)) or
                                 not isinstance(max_days, (int, np.integer)) or max_days <= 0):
        raise ValueError('max_days must be a positive integer or None')
    if np.any(times < 0) or np.any(times > duration_hours):
        raise ValueError('Measurements must fall inside the admission')
    origin = datetime.fromisoformat(admitted_at.replace('Z', '+00:00'))
    midnight_offset = (origin - origin.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600
    order = np.argsort(times, kind='stable')
    values, times = values[order], times[order]
    bins = np.floor((times + midnight_offset) / 24).astype(int)
    days = np.unique(bins)
    if max_days is not None and len(days) > max_days:
        split_day = days[-(max_days + 1)]
        bins = np.where(bins <= split_day, split_day, bins)
        days = np.unique(bins)
    carry = np.full(values.shape[1], np.nan)
    rows, masks, ends = [], [], []
    for day in days:
        in_day = bins == day
        entries = values[in_day]
        row = np.full(values.shape[1], np.nan)
        observed = np.zeros(values.shape[1], dtype=bool)
        for column, kind in enumerate(feature_types):
            valid = entries[:, column][np.isfinite(entries[:, column])]
            if len(valid):
                use_mean = kind == 'continuous' and continuous_agg == 'mean'
                row[column] = valid.mean() if use_mean else valid[-1]
                observed[column] = True
        if forward_fill:
            carry = np.where(observed, row, carry)
            rows.append(carry.copy())
        else:
            rows.append(row)
        masks.append(observed)
        ends.append(times[in_day][-1])
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


def _admission_measurements(record, dataset, schema, smart_schema):
    """Collect only measurements inside this admission's declared ICU stays."""
    admitted = timestamp(record['hospital_admitted_at'])
    discharged = timestamp(record['hospital_discharged_at'])
    hospital_los_hours(record['hospital_admitted_at'], record['hospital_discharged_at'])
    if dataset.startswith('mimic'):
        stays = record.get('icu_stays')
        if not isinstance(stays, list) or not stays:
            raise ValueError('MIMIC admissions require a nonempty icu_stays list')
        stays = sorted(stays, key=lambda stay: timestamp(stay['admitted_at']))
    else:
        stays = [dict(record, admitted_at=record['hospital_admitted_at'],
                      discharged_at=record['hospital_discharged_at'])]
    all_times, all_values, all_smart = [], [], []
    previous_end, stay_ids = None, set()
    for stay in stays:
        start, end = timestamp(stay['admitted_at']), timestamp(stay['discharged_at'])
        if not admitted <= start < end <= discharged:
            raise ValueError('Every measurement stay must fall inside its hospital admission')
        if previous_end is not None and start < previous_end:
            raise ValueError('ICU stays within an admission cannot overlap')
        previous_end = end
        for key in ('subject_id', 'hadm_id'):
            if key in stay and normalize_patient_id(stay[key]) != normalize_patient_id(record[key]):
                raise ValueError(f'ICU stay {key} differs from its admission record')
        if dataset.startswith('mimic'):
            stay_id = normalize_patient_id(stay.get('stay_id'))
            if stay_id in stay_ids:
                raise ValueError('Duplicate ICU stay_id within an admission')
            stay_ids.add(stay_id)
        times = np.asarray(stay['record_time'], dtype=float)
        duration = (end - start).total_seconds() / 3600
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(times < 0) or np.any(times > duration):
            raise ValueError('record_time must contain finite measurement times inside its stay')
        values = np.asarray(stay['values'], dtype=float)
        if values.shape != (len(times), len(schema[0])) or np.isinf(values).any():
            raise ValueError('Measurement values must align with record_time and feature_names')
        all_times.extend(times + (start - admitted).total_seconds() / 3600)
        all_values.append(values)
        if smart_schema is not None:
            raw = stay['values'] if smart_schema == schema and 'smart_values' not in stay else stay['smart_values']
            smart = np.asarray(raw, dtype=float)
            if smart.shape != (len(times), len(smart_schema[0])) or np.isinf(smart).any():
                raise ValueError('SMART measurements must align with the same record_time')
            all_smart.append(smart)
    return (np.concatenate(all_values), np.asarray(all_times),
            np.concatenate(all_smart) if all_smart else None, stays[-1])


def build_patient_history(record, dataset, need_smart=False):
    """Build one admission sample; each visit is an observed calendar day.

    MIMIC retains seven observed days plus a merged earlier-history step; TJH
    retains all observed days. Mortality and LOS withhold the final day, while
    readmission includes it through the last ICU stay's discharge. LLM missing
    values stay NaN; SMART uses LOCF. No other admission supplies input records.
    """
    if dataset not in ('mimic3_mortality', 'mimic3_los', 'mimic4_mortality', 'mimic4_los',
                       'mimic4_readmission', 'tjh_mortality', 'tjh_los'):
        raise ValueError(f'Unsupported EHR dataset: {dataset}')
    if 'visits' in record:
        raise ValueError('Legacy visits histories are unsupported; provide one hospital admission per JSONL record')
    subject = normalize_patient_id(record.get('subject_id'))
    admission = normalize_patient_id(record.get('hadm_id'))
    names, kinds = _feature_schema(record)
    smart_schema = None
    if need_smart:
        smart_schema = _feature_schema(record, smart=True) if 'smart_feature_names' in record else (names, kinds)
        input_dim = 17 if dataset.startswith('mimic3_') else 44 if dataset.startswith('mimic4_') else 75
        if len(smart_schema[0]) != input_dim:
            raise ValueError(f'{dataset} SMART input requires {input_dim} features in checkpoint order')
    raw, times, smart_raw, index_stay = _admission_measurements(record, dataset, (names, kinds), smart_schema)
    duration = hospital_los_hours(record['hospital_admitted_at'], record['hospital_discharged_at'])
    is_mimic = dataset.startswith('mimic')
    aggregation = dict(continuous_agg='last' if is_mimic else 'mean', max_days=7 if is_mimic else None)
    # LLM input retains missing values, as in Appendix F.
    values, _, daily_times = daily_records(raw, times, duration, kinds,
                                          admitted_at=record['hospital_admitted_at'],
                                          forward_fill=False, **aggregation)
    origin = datetime.fromisoformat(record['hospital_admitted_at'].replace('Z', '+00:00'))
    visit_ids = [(origin + timedelta(hours=float(t))).date().isoformat() for t in daily_times]
    target_visit_id = visit_ids[-1]
    readmission = dataset.endswith('_readmission')
    if dataset.endswith('_los'):
        label = hospital_los_label(record)
    elif readmission:
        label = icu_readmission_label(index_stay)
    else:
        label = record['mortality']
        if label not in (0, 1):
            raise ValueError('mortality must be a binary outcome')
        label = int(label)
    keep = slice(None) if readmission else slice(None, -1)
    values, daily_times, input_visits = values[keep], daily_times[keep], visit_ids[keep]
    if not len(values):
        raise ValueError('Mortality/LOS require an observed day preceding the withheld target day')
    is_mimic3 = dataset.startswith('mimic3_')
    sample = {
        'subject_id': subject, 'hadm_id': admission, 'name': f'{subject}:{admission}',
        'target_visit_id': target_visit_id, 'input_visit_ids': input_visits,
        'row_visit_ids': input_visits, 'record_time': daily_times.tolist(),
        'X': np.column_stack((daily_times, values)) if is_mimic3 else values,
        'header': ['Hours', *names] if is_mimic3 else list(names),
        't': len(values), 'y': label, 'data_protocol': 'admission_daily',
        'includes_target_visit': readmission,
    }
    if need_smart:
        # Only the encoder input uses LOCF within this admission.
        smart, mask, smart_times = daily_records(smart_raw, times, duration, smart_schema[1],
                                                admitted_at=record['hospital_admitted_at'],
                                                forward_fill=True, **aggregation)
        sample['X_ts'] = smart[keep]
        sample['smart_observed_mask'] = mask[keep]
        if not np.array_equal(smart_times[keep], daily_times):
            raise ValueError('LLM and SMART observation days must be identical')
    return sample, (names, kinds, smart_schema)


def normalize_smart_splits(splits):
    """Fit scaling on training admissions; apply it to all splits."""
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
        raise FileNotFoundError(f'Admission JSONL file not found: {path}. Set --ehr_records_path.')
    need_smart = args.method == 'ehr_model_smart' or getattr(args, 'embedding_model_name', None) == 'smart'
    data, schema, admissions = None, None, set()
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                sample, current_schema = build_patient_history(json.loads(line), args.dataset, need_smart)
                if schema is not None and current_schema != schema:
                    raise ValueError('Feature schema differs between admissions')
                if sample['hadm_id'] in admissions:
                    raise ValueError('Each hadm_id must appear in exactly one JSONL record')
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f'{path}:{line_number}: {error}') from error
            admissions.add(sample['hadm_id'])
            schema = current_schema
            if data is None:
                data = {key: [] for key in sample}
            for key, value in sample.items():
                data[key].append(value)
    if data is None:
        raise ValueError('The admission JSONL file is empty')
    splits = fixed_train_val_test_split(args, data, args.dataset)
    return normalize_smart_splits(splits) if need_smart else splits
