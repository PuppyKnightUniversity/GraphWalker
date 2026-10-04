from typing import Dict, Any, List
import numpy as np
import torch
from prompt.EHR_prompt.mimic3.utils import mimic3_smooth_hourly_data

def transform_mimic4_readmission_ehr_to_detail_prompt(patient_example: Dict[str, Any]) -> str:
    X = patient_example['X']
    header = patient_example['header']
    X_str = X.astype(str)
    lines = []
    for fi, feature in enumerate(header):
        values = [val if val != '' else 'NaN' for val in X_str[:, fi]]
        lines.append(f"- {feature}: [{', '.join(values)}]")
    detail = "\n".join(lines)
    return detail


def mimic4_readmission_prompt_wrapper(patient_example: Dict[str, Any],
                                    is_few_shot: bool = False,
                                    icl_examples_list=None,
                                    inference_type: str = 'only_answer',
                                    unit: bool = False,
                                    reference_range: bool = False,
                                    smooth_hourly_data: bool = False,
                                    keep_last: bool = True,
                                    add_smart_logits: bool = False,
                                    add_smart_logits_for_test_example: bool = False) -> str:
    from prompt.EHR_prompt.common import build_ehr_prompt
    if smooth_hourly_data:
        raise ValueError('Apply temporal aggregation before EHR serialization')
    return build_ehr_prompt(
        patient_example, 'mimic4_readmission', icl_examples_list,
        is_few_shot=is_few_shot, inference_type=inference_type,
        unit=unit, reference_range=reference_range,
        add_smart_logits=add_smart_logits,
        add_smart_logits_for_test_example=add_smart_logits_for_test_example)
