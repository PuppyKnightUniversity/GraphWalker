from typing import Dict, Any, List
import numpy as np
import torch

def transform_tjh_mortality_ehr_to_detail_prompt(patient_example: Dict[str, Any]) -> str:
    '''
    Transform the ehr data into detail prompt for tjh mortality data
    Args:
        patient_example: Dict[str, Any]
            The ehr data of a patient
    Returns:
        detail: str
            The detail string of the ehr data
    '''
    X = patient_example['X']
    header = patient_example['header']
    
    # Convert to numpy array if not already
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    
    # Build detail lines for each feature
    detail_lines = []
    for fi, feature in enumerate(header):
        # Extract values for this feature across all time steps
        values = []
        for val in X[:, fi]:
            # Handle missing values: NaN, empty string, or None
            if isinstance(val, (int, float)) and np.isnan(val):
                values.append('NaN')
            elif isinstance(val, str) and (val == '' or val.lower() == 'nan' or val == 'None'):
                values.append('NaN')
            elif val is None:
                values.append('NaN')
            else:
                # Convert to string, preserving the original value
                values.append(str(val))
        detail_lines.append(f"- {feature}: [{', '.join(values)}]")
    
    detail = "\n".join(detail_lines)
    return detail


def tjh_mortality_prompt_wrapper(patient_example: Dict[str, Any],
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
        patient_example, 'tjh_mortality', icl_examples_list,
        is_few_shot=is_few_shot, inference_type=inference_type,
        unit=unit, reference_range=reference_range,
        add_smart_logits=add_smart_logits,
        add_smart_logits_for_test_example=add_smart_logits_for_test_example)
