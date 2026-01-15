from typing import Dict, Any, List
import numpy as np
import torch
from prompt.EHR_prompt.mimic3.utils import mimic3_smooth_hourly_data

def transform_mimic4_mortality_ehr_to_detail_prompt(patient_example: Dict[str, Any]) -> str:
    X = patient_example['X']
    header = patient_example['header']
    X_str = X.astype(str)
    lines = []
    for fi, feature in enumerate(header):
        values = [val if val != '' else 'NaN' for val in X_str[:, fi]]
        lines.append(f"- {feature}: [{', '.join(values)}]")
    detail = "\n".join(lines)
    return detail

def mimic4_mortality_prompt_wrapper(patient_example: Dict[str, Any],
                                    is_few_shot: bool = False,
                                    icl_examples_list: List[Dict[str, Any]] = [],
                                    inference_type: str = 'only_answer',
                                    unit: bool = False,
                                    reference_range: bool = False,
                                    smooth_hourly_data: bool = False,
                                    keep_last: bool = True,
                                    add_smart_logits: bool = False,
                                    add_smart_logits_for_test_example: bool = False) -> str:
    X = patient_example['X']
    header = patient_example['header']
    t_len = int(patient_example['t'])

    if smooth_hourly_data:
        from prompt.EHR_prompt.mimic3.utils import mimic3_smooth_hourly_data
        patient_example = mimic3_smooth_hourly_data(patient_example, keep_last=keep_last)
        X = patient_example['X']
        t_len = X.shape[0]

    detail_lines = []
    X_str = X.astype(str)
    for i, feature in enumerate(header):
        values = [val if val != '' else 'NaN' for val in X_str[:, i]]
        detail_lines.append(f"- {feature}: [{', '.join(values)}]")
    detail = "\n".join(detail_lines)

    if is_few_shot:
        if add_smart_logits:
            from prompt.EHR_prompt.prompt_template import USERPROMPT_FEW_SHOT_SMART_WITH_LOGITS as PROMPT_TEMPLATE
        else:
            from prompt.EHR_prompt.prompt_template import USERPROMPT_FEW_SHOT as PROMPT_TEMPLATE
    else:
        from prompt.EHR_prompt.prompt_template import USERPROMPT_ZERO_SHOT as PROMPT_TEMPLATE
    from prompt.EHR_prompt.prompt_template import TASK_DESCRIPTION

    if inference_type == 'only_answer':
        from prompt.EHR_prompt.prompt_template import RESPONSE_FORMAT_ONLY_ANSWER as RESPONSE_FORMAT
    else:
        raise ValueError(f'we have not implemented the inference type for {inference_type}')

    if is_few_shot:
        if add_smart_logits:
            def convert_logits_to_risk_value(smart_logits):
                if isinstance(smart_logits, torch.Tensor):
                    logits_np = smart_logits.cpu().numpy()
                else:
                    logits_np = np.array(smart_logits)
                exp_logits = np.exp(logits_np - np.max(logits_np))
                probs = exp_logits / np.sum(exp_logits)
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

    record_times = np.arange(t_len).astype(float)
    prompt = PROMPT_TEMPLATE.format(
        LENGTH=len(record_times),
        RECORD_TIME_LIST=', '.join([f"{float(t):.2f}" for t in record_times]),
        DETAIL=detail,
        RESPONSE_FORMAT=RESPONSE_FORMAT['mimic3_mortality'],
        TASK_DESCRIPTION=TASK_DESCRIPTION['mimic3_mortality'],
        EXAMPLE=example,
    )

    if add_smart_logits_for_test_example:
        def convert_logits_to_risk_value(smart_logits):
            if isinstance(smart_logits, torch.Tensor):
                logits_np = smart_logits.cpu().numpy()
            else:
                logits_np = np.array(smart_logits)
            exp_logits = np.exp(logits_np - np.max(logits_np))
            probs = exp_logits / np.sum(exp_logits)
            return float(probs[1])
        prompt += f"\nExpert Model Outputs:  {convert_logits_to_risk_value(patient_example.get('smart_logits', [0.0, 1.0])):.4f}"

    extra_info_prompt = '''

Important Scoring Guidelines for Mortality Risk Prediction:

This task involves predicting mortality risk with significant class imbalance: over 90% of patients survive, while less than 10% have mortality outcomes. Your goal is to maximize AUPRC (Area Under the Precision-Recall Curve) by accurately ranking patients according to their true risk.

Key Principles for Optimal Scoring:

1. **Precise Risk Calibration**: Assign risk scores that accurately reflect the patient's true mortality risk based on clinical evidence. High-risk patients (with severe indicators like critical lab values, organ dysfunction, or life-threatening conditions) should receive scores in the upper range (0.85-0.99), while low-risk patients (with stable vitals, normal labs, improving status) should receive scores in the lower range (0.0001-0.01).

2. **Maximize Score Separation**: Create clear separation between risk levels. Avoid clustering scores in the middle range (0.3-0.7) unless evidence is genuinely ambiguous. The wider the separation between high-risk and low-risk scores, the better the AUPRC performance.

3. **Evidence-Based Scoring**: Base your scores on the strength and clarity of clinical evidence. When multiple severe risk indicators are present, assign higher scores (e.g., 0.9234, 0.8567). When indicators suggest low risk, assign lower scores (e.g., 0.00023, 0.00145). 

4. **Account for Base Rate**: While most patients (90%+) have low risk, you should still identify and rank the truly high-risk patients accurately. Focus on relative ranking: ensure that patients with stronger evidence of mortality risk receive higher scores than those with weaker evidence, regardless of the absolute score level.

Remember: Your objective is to provide accurate risk scores that enable effective discrimination between high-risk and low-risk patients, optimizing the AUPRC metric through precise relative ranking.\n

Your Answer:'''
    #prompt += extra_info_prompt
    return prompt