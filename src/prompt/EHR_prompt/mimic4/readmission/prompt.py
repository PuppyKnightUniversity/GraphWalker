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
                                     icl_examples_list: List[Dict[str, Any]] = [],
                                     inference_type: str = 'only_answer',
                                     unit: bool = False,
                                     reference_range: bool = False,
                                     smooth_hourly_data: bool = False,
                                     keep_last: bool = True,
                                     add_smart_logits: bool = False,
                                     add_smart_logits_for_test_example: bool = True) -> str:
    X = patient_example['X']
    header = patient_example['header']
    t_len = int(patient_example['t'])

    if smooth_hourly_data:
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
        RESPONSE_FORMAT=RESPONSE_FORMAT['mimic4_readmission'],
        TASK_DESCRIPTION=TASK_DESCRIPTION['mimic4_readmission'],
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

    return prompt