from typing import Dict, Any, List
import numpy as np
import re
import torch

def transform_mimic3_mortality_ehr_to_detail_prompt(patient_example: Dict[str, Any],
                                                    unit: bool = False,
                                                    reference_range: bool = False,
                                                    smooth_hourly_data: bool = True,
                                                    keep_last: bool = True) -> str:
    '''
    Transform the ehr data into detail prompt for mimic3 mortality data
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
                          dataset_name: str = 'mimic3_mortality') -> str:
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
                if mask[visit_idx, i] == 1:
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
                    unit_range += f'{unit_values[feature]} '
                if reference_range:
                    unit_range += range_values[feature]
                unit_range = unit_range.rstrip() + ')'
            detail += f"- {feature}{unit_range}: [{', '.join(feature_values[feature])}]\n"
        
        return detail.strip()

    # predefine function2
    def extract_leading_number(s):
        '''
        Extract the leading number from the string
        '''
        s = str(s)
        match = re.match(r'^\s*(-?\d+\.?\d*)', s)
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
                               dataset_name='mimic3_mortality')
    
    return detail


def mimic3_mortality_prompt_wrapper(patient_example: Dict[str, Any],
                                    is_few_shot: bool = False,
                                    icl_examples_list: List[Dict[str, Any]] = [],
                                    inference_type: str = 'only_answer',
                                    unit: bool = False,
                                    reference_range: bool = False,
                                    smooth_hourly_data: bool = True,
                                    keep_last: bool = True,
                                    add_smart_logits: bool = False,
                                    add_smart_logits_for_test_example: bool = True) -> str:
    '''
    Wrap the ehr data into prompt template for mimic3 mortality data
    
    Args:
        patient_example: Dict[str, Any] 
            The ehr data of a patient
        is_few_shot: bool
            Whether to use few-shot ICL 
        inference_type: str
            The type of inference
        icl_examples_list: List[str]
            The list of few-shot examples
        unit: bool
            Whether to include unit in the detail
        reference_range: bool
            Whether to include reference range in the detail
        smooth_hourly_data: bool
            Whether to smooth the hourly data
        keep_last: bool
            Whether to keep the last data point of each hour
    Returns:
        prompt: str
            The prompt contains patient ehr data
    '''

    # detail EHR prompt for one patient
    detail = patient_example['detail']
    if smooth_hourly_data:
        from prompt.EHR_prompt.mimic3.utils import mimic3_smooth_hourly_data
        patient_example = mimic3_smooth_hourly_data(patient_example, keep_last=keep_last)
    # load template
    if is_few_shot:
        if add_smart_logits:
            from prompt.EHR_prompt.prompt_template import USERPROMPT_FEW_SHOT_SMART_WITH_LOGITS as PROMPT_TEMPLATE
        else:
            from prompt.EHR_prompt.prompt_template import USERPROMPT_FEW_SHOT as PROMPT_TEMPLATE
    else:
        # zero-shot ICL
        from prompt.EHR_prompt.prompt_template import USERPROMPT_ZERO_SHOT as PROMPT_TEMPLATE
    
    from prompt.EHR_prompt.prompt_template import TASK_DESCRIPTION
    
    # load response format
    if inference_type == 'only_answer':
        from prompt.EHR_prompt.prompt_template import RESPONSE_FORMAT_ONLY_ANSWER as RESPONSE_FORMAT
    else:
        raise ValueError(f'we have not implemented the inference type for {inference_type}')

    # load few-shot examples
    if is_few_shot:
        # Each example contains the EHR detail and a binary prediction label (0/1)
        if add_smart_logits:
            def convert_logits_to_risk_value(smart_logits):
                '''
                Convert 2D logits array to risk value (probability of positive class)
                Args:
                    smart_logits: torch.Tensor or np.ndarray of shape [2]
                Returns:
                    risk_value: float, probability of positive class (class 1)
                '''
                # Convert to numpy if it's a torch tensor
                if isinstance(smart_logits, torch.Tensor):
                    logits_np = smart_logits.cpu().numpy()
                else:
                    logits_np = np.array(smart_logits)
                
                # Apply softmax to convert logits to probabilities
                exp_logits = np.exp(logits_np - np.max(logits_np))  # numerical stability
                probs = exp_logits / np.sum(exp_logits)
                
                # Return probability of positive class (index 1)
                return float(probs[1])
            
            example = '\n\n'.join([
                f"Example {i+1}:\n{icl_example['detail']}\nExpert Model Outputs:  {convert_logits_to_risk_value(icl_example['smart_logits']):.4f}\nLabel: {icl_example['label']}"
                for i, icl_example in enumerate(icl_examples_list)
            ])
        else:
            example = '\n\n'.join([
                f"Example {i+1}:\n{icl_example['detail']}\nLabel: {icl_example['label']}"
                for i, icl_example in enumerate(icl_examples_list)
            ])
    else:
        example = ''
        
    X = patient_example['X']
    record_times = X[:, 0].astype(float)

    prompt = PROMPT_TEMPLATE.format(
        LENGTH=len(record_times),
        RECORD_TIME_LIST=', '.join([f"{float(t):.2f}" for t in record_times]),
        DETAIL=detail,
        RESPONSE_FORMAT=RESPONSE_FORMAT['mimic3_mortality'],
        TASK_DESCRIPTION=TASK_DESCRIPTION['mimic3_mortality'],
        EXAMPLE=example,)
    

    # For ablation study, whether to add SMART model logits to the test example
    if add_smart_logits_for_test_example:
        prompt += f"\nExpert Model Outputs:  {convert_logits_to_risk_value(patient_example['smart_logits']):.4f}"

    return prompt