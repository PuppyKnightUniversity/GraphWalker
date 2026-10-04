from typing import Dict, Any, List
import numpy as np
import re
import torch

def transform_mimic3_los_ehr_to_detail_prompt(patient_example: Dict[str, Any],
                                                    unit: bool = False,
                                                    reference_range: bool = False,
                                                    smooth_hourly_data: bool = True,
                                                    keep_last: bool = True) -> str:
    '''
    Transform the ehr data into detail prompt for mimic3 los data
    Args:
        patient_example: Dict[str, Any]
            The ehr data of a patient
        unit: bool
            Whether to include unit in the detail
        reference_range: bool
            Whether to include reference range in the detail
        smooth_hourly_data: bool
            Whether to smooth the hourly data
        keep_last: bool
            Whether to keep the last data point of each hour
    Returns:
        detail: str
            The detail string of the ehr data
    '''
    # predefine function1
    def format_EHR_detail(patient_data: np.ndarray, 
                          features: List[str], 
                          mask: np.ndarray,
                          unit: bool = False,
                          reference_range: bool = False,
                          dataset_name: str = 'mimic3_los') -> str:
        '''
        Format the ehr data into detail string, it will be called by prepare_prompt_for_patient_example

        Args:
            patient_data: np.ndarray
                The ehr data of a patient, numeric data
            features: List[str]
                The features of the ehr data
            mask: np.ndarray
                The mask of the ehr data, 1 for missing value, 0 for non-missing value
            unit: bool
                Whether to include unit in the detail
            reference_range: bool
                Whether to include reference range in the detail
            dataset_name: str
                The name of the dataset
        Returns:
            detail: str
                The detail string of the ehr data
        '''
        feature_values = {}
        # Define some categorical features with their possible values
        categorical_features_dict = {
            "Glascow coma scale eye opening": {
                1: "No Response",
                2: "To Pain",
                3: "To Speech",
                4: "Spontaneously",
            },
            "Glascow coma scale motor response": {
                1: "No Response",
                2: "Abnormal Extension",
                3: "Abnormal Flexion",
                4: "Flex-withdraws",
                5: "Localizes Pain",
                6: "Obeys Commands",
            },
            "Glascow coma scale verbal response": {
                1: "No Response",
                2: "Incomprehensible sounds",
                3: "Inappropriate Words",
                4: "Confused",
                5: "Oriented",
            },
        }

        for i, feature in enumerate(features):
            feature_values[feature] = []
            for visit_idx in range(patient_data.shape[0]):
                if mask[visit_idx, i] == 1 or not np.isfinite(patient_data[visit_idx, i]):
                    feature_values[feature].append('NaN')
                else:
                    value = patient_data[visit_idx, i]
                    if feature in categorical_features_dict:
                        if not np.isnan(value):
                            feature_values[feature].append(categorical_features_dict[feature].get(int(value), str(value)))
                        else:
                            feature_values[feature].append('NaN')
                    else:
                        feature_values[feature].append(f"{value}")
            
        # load unit and reference range
        import json
        from prompt.EHR_prompt.prompt_template import UNIT, REFERENCE_RANGE
        unit_values = dict(json.load(open(UNIT[dataset_name])))
        range_values = dict(json.load(open(REFERENCE_RANGE[dataset_name])))

        detail = ''
        for feature in features:
            unit_range = ''
            if unit or reference_range:
                unit_range = ' ('
                if unit:
                    unit_range += f'{unit_values.get(feature, "/")} '
                if reference_range:
                    unit_range += range_values.get(feature, '/')
                unit_range = unit_range.rstrip() + ')'
            detail += f"- {feature}{unit_range}: [{', '.join(feature_values[feature])}]\n"
        
        return detail.strip()

    # predefine function2
    def extract_leading_number(s):
        '''
        Extract the leading number from the string
        '''
        s = str(s)
        match = re.match(r'^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)', s)
        if match:
            return match.group(1) 
        return np.nan 
    # begin process patient example
    # smooth patient data
    if smooth_hourly_data:
        from prompt.EHR_prompt.mimic3.utils import mimic3_smooth_hourly_data
        patient_example = mimic3_smooth_hourly_data(patient_example, keep_last=keep_last)
    vectorized_extract = np.vectorize(extract_leading_number)
    X = patient_example['X']
    header = patient_example['header']
    record_times = X[:, 0].astype(str)
    feature_data = X[:, 1:].astype(str)
    feature_names = header[1:]    
    mask = (feature_data == '')

    numeric_data = np.full(feature_data.shape, np.nan, dtype=float)
    non_missing_indices = ~mask
    data_to_clean = feature_data[non_missing_indices]
    
    cleaned_data_str = vectorized_extract(data_to_clean)  
    cleaned_data_float = cleaned_data_str.astype(float)
    
    numeric_data[non_missing_indices] = cleaned_data_float
    
    # fomulate detail EHR prompt for one patient
    detail = format_EHR_detail(numeric_data, 
                               feature_names, 
                               mask, unit=unit, 
                               reference_range=reference_range, 
                               dataset_name='mimic3_los')
    
    return detail

def mimic3_los_prompt_wrapper(patient_example: Dict[str, Any],
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
        patient_example, 'mimic3_los', icl_examples_list,
        is_few_shot=is_few_shot, inference_type=inference_type,
        unit=unit, reference_range=reference_range,
        add_smart_logits=add_smart_logits,
        add_smart_logits_for_test_example=add_smart_logits_for_test_example)
