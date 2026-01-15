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
                                    icl_examples_list: List[Dict[str, Any]] = [],
                                    inference_type: str = 'only_answer',
                                    unit: bool = False,
                                    reference_range: bool = False,
                                    smooth_hourly_data: bool = False,
                                    keep_last: bool = True,
                                    add_smart_logits: bool = False,
                                    add_smart_logits_for_test_example: bool = True) -> str:
    '''
    Wrap the ehr data into prompt template for tjh mortality data
    
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
    # For TJH dataset, X does not contain time column as first column
    # X[:, 0] is Sex feature, not time
    # Use indices as time points (days from first measurement)
    num_time_steps = X.shape[0]
    record_times = np.arange(num_time_steps).astype(float)

    prompt = PROMPT_TEMPLATE.format(
        LENGTH=len(record_times),
        RECORD_TIME_LIST=', '.join([f"{float(t):.2f}" for t in record_times]),
        DETAIL=detail,
        RESPONSE_FORMAT=RESPONSE_FORMAT['tjh_mortality'],
        TASK_DESCRIPTION=TASK_DESCRIPTION['tjh_mortality'],
        EXAMPLE=example,)
    

    # For ablation study, whether to add SMART model logits to the test example
    if add_smart_logits_for_test_example:
        prompt += f"\nExpert Model Outputs:  {convert_logits_to_risk_value(patient_example['smart_logits']):.4f}"

    extra_info_prompt = '''

Important Scoring Guidelines for Mortality Risk Prediction:

Key Principles for Optimal Scoring:

1. **Conservative High-Risk Assessment (CRITICAL)**: Be extremely conservative when assigning high-risk scores (0.85-0.99). Only assign scores in this range when you have very high confidence based on overwhelming clinical evidence. High-risk scores should be reserved for cases with multiple severe, life-threatening indicators that clearly and unambiguously point to imminent mortality risk. When in doubt, err on the side of caution and assign a lower score.

2. **Precise Risk Calibration**: Assign risk scores that accurately reflect the patient's true mortality risk based on clinical evidence. High-risk patients (with severe indicators like critical lab values, organ dysfunction, or life-threatening conditions) should receive scores in the upper range (0.85-0.99) ONLY when evidence is overwhelming and unambiguous. Low-risk patients (with stable vitals, normal labs, improving status) should receive scores in the lower range (0.0001-0.01).

3. **Maximize Score Separation**: Create clear separation between risk levels. Avoid clustering scores in the middle range (0.3-0.7) unless evidence is genuinely ambiguous. The wider the separation between high-risk and low-risk scores, the better the AUPRC performance. However, maintain conservatism: do not assign high scores unless you are very confident.

4. **Evidence-Based Scoring with Conservative Bias**: Base your scores on the strength and clarity of clinical evidence. When multiple severe risk indicators are present AND you are very confident in their interpretation, assign higher scores (e.g., 0.9234, 0.8567). When indicators suggest low risk or are ambiguous, assign lower scores (e.g., 0.00023, 0.00145). When evidence is unclear or mixed, default to lower scores rather than high-risk scores.

5. **Account for Base Rate**: While most patients (90%+) have low risk, you should still identify and rank the truly high-risk patients accurately. However, maintain strict conservatism: only assign high-risk scores when you have very high confidence. Focus on relative ranking: ensure that patients with stronger evidence of mortality risk receive higher scores than those with weaker evidence, but do not assign high absolute scores unless evidence is overwhelming.

Remember: Your objective is to provide accurate risk scores that enable effective discrimination between high-risk and low-risk patients, optimizing the AUPRC metric through precise relative ranking. **Be extremely conservative with high-risk scores - only assign them when you are very confident based on overwhelming evidence.**\n

Your Answer:'''
    #prompt += extra_info_prompt
    return prompt