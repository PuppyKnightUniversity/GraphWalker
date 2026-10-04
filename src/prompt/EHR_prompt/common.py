"""Shared EHR prompt metadata and task label formatting."""
import numpy as np


def format_icl_label(label, dataset):
    """Render scalar NumPy/Torch labels as the task output, including LOS A-D."""
    if hasattr(label, 'item'):
        label = label.item()
    if dataset.endswith('_los'):
        if isinstance(label, str) and label in ('A', 'B', 'C', 'D'):
            return label
        if label not in (0, 1, 2, 3):
            raise ValueError(f"Invalid LOS class: {label}")
        return 'ABCD'[int(label)]
    return str(label)


def measurement_times(patient, dataset):
    if 'record_time' in patient:
        times = np.asarray(patient['record_time'], dtype=float)
    elif dataset.startswith('mimic3_'):
        times = np.asarray(patient['X'][:, 0], dtype=float)
    else:
        raise ValueError("Missing record_time metadata; regenerate the raw data cache. "
                         "Feature values or row indices must not be labeled as hours.")
    if times.ndim != 1 or len(times) != len(patient['X']) or not np.isfinite(times).all():
        raise ValueError("record_time must contain one finite time per measurement")
    return times


def history_context(patient):
    if patient.get('data_protocol') != 'historical_visits':
        return ''
    visits = {visit: i + 1 for i, visit in enumerate(patient['input_visit_ids'])}
    indices = [visits[visit] for visit in patient['row_visit_ids']]
    return ('The measurements describe visits preceding the target visit.\n'
            'Measurement times are hours since the first historical admission.\n'
            f'Visit index for each measurement: {indices}\n\n')


def serialize_patient_record(patient, dataset, unit=False, reference_range=False):
    """Serialize observed measurements and their time/visit metadata."""
    times = measurement_times(patient, dataset)
    if dataset.startswith('mimic3_'):
        from prompt.EHR_prompt.mimic3.mortality.prompt import transform_mimic3_mortality_ehr_to_detail_prompt
        detail = transform_mimic3_mortality_ehr_to_detail_prompt(
            patient, unit=unit, reference_range=reference_range, smooth_hourly_data=False)
    else:
        values = np.asarray(patient['X'])
        headers = patient['header']
        if values.ndim != 2 or values.shape[1] != len(headers):
            raise ValueError('Feature names and measurement columns must be aligned')
        lines = []
        for column, name in enumerate(headers):
            entries = []
            for value in values[:, column]:
                text = str(value)
                entries.append('NaN' if value is None or text.strip().lower() in ('', 'nan', 'none') else text)
            lines.append(f"- {name}: [{', '.join(entries)}]")
        detail = '\n'.join(lines)
    return (f'- Number of measurements: {len(times)}\n'
            f"- Measurement times in the historical records: [{', '.join(f'{t:.2f}' for t in times)}]\n"
            + history_context(patient) + 'Clinical Features Over Time:\n' + detail)


def expert_probabilities(example, dataset):
    if 'smart_logits' not in example:
        raise ValueError('Expert probabilities require smart_logits')
    logits = example['smart_logits']
    if hasattr(logits, 'detach'):
        logits = logits.detach().cpu().numpy()
    logits = np.asarray(logits, dtype=float)
    classes = 4 if dataset.endswith('_los') else 2
    if logits.shape != (classes,) or not np.isfinite(logits).all():
        raise ValueError(f'Expected {classes} finite expert logits')
    weights = np.exp(logits - logits.max())
    probabilities = weights / weights.sum()
    if classes == 4:
        return ', '.join(f'{label}: {prob:.4f}' for label, prob in zip('ABCD', probabilities))
    return f'{probabilities[1]:.4f}'


def build_ehr_prompt(patient, dataset, examples=(), *, is_few_shot=None,
                     inference_type='only_answer', unit=False, reference_range=False,
                     add_smart_logits=False, add_smart_logits_for_test_example=False,
                     return_target_span=False):
    """Build an outcome query and optionally return its target-record character span."""
    from prompt.EHR_prompt.prompt_template import (
        TASK_DESCRIPTION, RESPONSE_FORMAT_ONLY_ANSWER, USERPROMPT_ZERO_SHOT,
        USERPROMPT_FEW_SHOT, USERPROMPT_FEW_SHOT_SMART_WITH_LOGITS,
    )
    if dataset not in TASK_DESCRIPTION:
        raise ValueError(f'Unsupported EHR task: {dataset}')
    if inference_type != 'only_answer':
        raise ValueError(f'Unsupported inference type: {inference_type}')
    if patient.get('data_protocol', 'historical_visits') != 'historical_visits':
        raise ValueError('Historical EHR prompts require preceding visits with the target visit excluded')
    examples = list(examples or ())
    few_shot = bool(examples) if is_few_shot is None else is_few_shot and bool(examples)
    rendered = []
    for index, example in enumerate(examples if few_shot else (), 1):
        if example.get('data_protocol', 'historical_visits') != 'historical_visits':
            raise ValueError('Demonstrations require historical visits')
        detail = example.get('detail')
        if detail is None:
            detail = serialize_patient_record(example, dataset, unit, reference_range)
        text = f'Example {index}:\nHistorical EHR Records:\n{detail}'
        if add_smart_logits:
            text += '\nExpert Model Probabilities: ' + expert_probabilities(example, dataset)
        label = example['label'] if 'label' in example else example['y']
        text += '\nOutcome Label: ' + format_icl_label(label, dataset)
        rendered.append(text)
    template = (USERPROMPT_FEW_SHOT_SMART_WITH_LOGITS if add_smart_logits else USERPROMPT_FEW_SHOT) if few_shot else USERPROMPT_ZERO_SHOT
    before, separator, after = template.partition('{TARGET_RECORD}')
    if not separator or '{TARGET_RECORD}' in after:
        raise ValueError('Prompt template requires exactly one target-record field')
    prefix = before.format(TASK_DESCRIPTION=TASK_DESCRIPTION[dataset],
                           RESPONSE_FORMAT=RESPONSE_FORMAT_ONLY_ANSWER[dataset],
                           EXAMPLE='\n\n'.join(rendered))
    detail = patient.get('detail')
    if detail is None:
        detail = serialize_patient_record(patient, dataset, unit, reference_range)
    if not isinstance(detail, str) or not detail.strip():
        raise ValueError('The target record must be nonempty text')
    start, end = len(prefix), len(prefix) + len(detail)
    suffix = after
    if add_smart_logits_for_test_example:
        suffix = '\nExpert Model Probabilities: ' + expert_probabilities(patient, dataset) + suffix
    prompt = prefix + detail + suffix
    return (prompt, start, end) if return_target_span else prompt
