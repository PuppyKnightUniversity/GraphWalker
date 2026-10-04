"""Outcome labels with explicit time units."""
from datetime import datetime, timezone
import math


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError('Timestamps must be ISO-8601 strings')
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def hospital_los_hours(admitted_at, discharged_at):
    hours = (timestamp(discharged_at) - timestamp(admitted_at)).total_seconds() / 3600
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError('Hospital discharge must follow hospital admission')
    return hours


def los_bin_from_hours(hours):
    """Classify continuous LOS: <3, [3, 7], (7, 14], or >14 days."""
    hours = float(hours)
    if not math.isfinite(hours) or hours < 0:
        raise ValueError('LOS must be a finite, nonnegative duration in hours')
    if hours < 72:
        return 0
    if hours <= 168:
        return 1
    if hours <= 336:
        return 2
    return 3


def hospital_los_label(record):
    """Derive total hospital LOS from admission and discharge timestamps."""
    required = ('hospital_admitted_at', 'hospital_discharged_at')
    if not all(key in record for key in required):
        raise ValueError('LOS records require hospital_admitted_at and hospital_discharged_at')
    return los_bin_from_hours(hospital_los_hours(*(record[key] for key in required)))


def icu_readmission_label(record):
    """Read a binary ICU readmission outcome within 30 days of ICU discharge."""
    if 'icu_readmission_30d' in record:
        label = record['icu_readmission_30d']
    elif record.get('readmission_definition') == 'icu_30d':
        key = 'readmission_30d' if 'readmission_30d' in record else 'y_readmission'
        label = record[key]
    else:
        raise ValueError('Readmission requires icu_readmission_30d or readmission_definition=icu_30d')
    if isinstance(label, (list, tuple)):
        if len(label) != 1:
            raise ValueError('ICU readmission must have exactly one outcome label')
        label = label[0]
    if hasattr(label, 'item'):
        label = label.item()
    if label not in (0, 1):
        raise ValueError('icu_readmission_30d must be a binary outcome')
    return int(label)
