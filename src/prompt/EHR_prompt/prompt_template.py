"""Task instructions and shared historical-EHR prompt templates."""
from pathlib import Path


_MORTALITY_TASK = ("You are tasked with predicting whether the patient will die during the target visit, "
                   "using EHR records from preceding visits. The target visit is the patient's final visit, "
                   "and its records are excluded from the input.")
_LOS_TASK = ("You are tasked with predicting the patient's total hospital length of stay during the target "
             "visit, using EHR records from preceding visits. The target visit is the patient's final visit, "
             "and its records are excluded from the input.")
TASK_DESCRIPTION = {
    **{f'{dataset}_mortality': _MORTALITY_TASK for dataset in ('mimic3', 'mimic4', 'tjh')},
    **{f'{dataset}_los': _LOS_TASK for dataset in ('mimic3', 'mimic4', 'tjh')},
    'mimic4_readmission': (
        'You are tasked with predicting whether the patient will be readmitted to the ICU within 30 days '
        'after discharge from the target ICU stay, using EHR records from preceding visits. The target '
        "visit is the patient's final visit, and its records are excluded from the input."),
}

_MORTALITY_RESPONSE = '''\
Provide only a floating-point number between 0 and 1 representing the predicted probability of in-hospital mortality during the target visit. A higher value indicates a higher probability of death.
Do not provide any reasoning, explanation, or additional text. Output only the numerical value.
Example: 0.XX'''
_LOS_RESPONSE = '''\
Provide only a single letter (A, B, C, or D) representing the predicted length-of-stay category:
- A: Less than 3 days (< 3 days)
- B: 3 to 7 days (3 <= days <= 7)
- C: More than 7 and up to 14 days (7 < days <= 14)
- D: More than 14 days (> 14 days)
Do not provide any reasoning, explanation, or additional text. Output only the letter (A, B, C, or D).
Example: B'''
RESPONSE_FORMAT_ONLY_ANSWER = {
    **{f'{dataset}_mortality': _MORTALITY_RESPONSE for dataset in ('mimic3', 'mimic4', 'tjh')},
    **{f'{dataset}_los': _LOS_RESPONSE for dataset in ('mimic3', 'mimic4', 'tjh')},
    'mimic4_readmission': '''\
Provide only a floating-point number between 0 and 1 representing the predicted probability of a new ICU admission within this 30-day window.
Do not provide any reasoning, explanation, or additional text. Output only the numerical value.
Example: 0.XX''',
}

_RESOURCE_DIR = Path(__file__).resolve().parent
UNIT = {name: str(_RESOURCE_DIR / 'mimic3_unit.json')
        for name in ('mimic3_mortality', 'mimic3_los')}
REFERENCE_RANGE = {name: str(_RESOURCE_DIR / 'mimic3_range.json')
                   for name in ('mimic3_mortality', 'mimic3_los')}

_HISTORY_INTRO = '''\
You will be provided with longitudinal electronic health record (EHR) data from a patient's visits preceding the final visit. We refer to the final visit as the target visit. Records from the target visit and its outcome are not provided.
Each clinical feature is represented as a time-ordered sequence of measurements from the preceding visits. Missing values are denoted as NaN. Units and reference ranges are provided where applicable.
'''
USERPROMPT_ZERO_SHOT = _HISTORY_INTRO + '''\
Use the provided historical records to predict the specified outcome associated with the target visit.

Task Description:
{TASK_DESCRIPTION}

Instructions & Output Format:
{RESPONSE_FORMAT}

Target Patient's Historical Records:
{TARGET_RECORD}

Your Answer:'''

USERPROMPT_FEW_SHOT = _HISTORY_INTRO + '''\
You will also receive selected examples from other patients. Each example contains that patient's historical records preceding their own target visit, together with the known outcome associated with that visit. Use these examples and the target patient's historical records to predict the specified outcome for the target patient.

Task Description:
{TASK_DESCRIPTION}

Instructions & Output Format:
{RESPONSE_FORMAT}

Selected Patient Examples:
{EXAMPLE}

Target Patient's Historical Records:
{TARGET_RECORD}

Your Answer:'''

USERPROMPT_FEW_SHOT_SMART_WITH_LOGITS = USERPROMPT_FEW_SHOT.replace(
    'Selected Patient Examples:',
    'Expert model probabilities are provided as auxiliary estimates for the specified outcome.\n\n'
    'Selected Patient Examples:')
