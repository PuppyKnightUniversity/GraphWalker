"""Task instructions and shared historical-EHR prompt templates."""
from pathlib import Path


_MORTALITY_TASK = ("You are tasked with predicting whether the patient will die during this hospital admission. "
                   "Each visit is one calendar day of observations within this admission. The target visit is "
                   "its final observation day, whose measurements are excluded; use the preceding daily visits.")
_LOS_TASK = ("You are tasked with predicting the total hospital length of stay for this admission. Each visit is one "
             "calendar day of observations within this admission. The target visit is its final observation "
             "day, whose measurements are excluded; use the preceding daily visits.")
TASK_DESCRIPTION = {
    **{f'{dataset}_mortality': _MORTALITY_TASK for dataset in ('mimic3', 'mimic4', 'tjh')},
    **{f'{dataset}_los': _LOS_TASK for dataset in ('mimic3', 'mimic4', 'tjh')},
    'mimic4_readmission': (
        'You are tasked with predicting whether the patient will be readmitted to the ICU within 30 days '
        'after discharge from the index ICU stay (the last ICU stay of this hospital admission). '
        'Use the retained daily ICU observations from this admission through index ICU discharge, '
        'including the final observation day. No post-discharge or other-admission records are provided.'),
}

_MORTALITY_RESPONSE = '''\
Provide only a floating-point number between 0 and 1 representing the predicted probability of mortality during this hospital admission. A higher value indicates a higher probability of death.
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

HISTORY_INTRO = '''\
You will be provided with electronic health record (EHR) data from one hospital admission. A visit is one observed calendar day, not a separate admission. The final observation day is the target visit; its measurements and the admission outcome are not provided. The input contains preceding daily visits from the retained seven-day window. MIMIC records contain only ICU measurements; TJH records contain laboratory measurements. Records from other admissions are not included.
Each clinical feature is a time-ordered sequence of daily measurements. Missing values are denoted as NaN. Units and reference ranges are provided where applicable.
'''
READMISSION_INTRO = '''\
You will be provided with ICU electronic health record (EHR) data from one hospital admission. A visit is one observed calendar day, not a separate admission. The input contains daily observations from the retained seven-day window through discharge of the last ICU stay, including the final observation day. No post-discharge records, other-admission records, or readmission outcome are provided.
Each clinical feature is a time-ordered sequence of daily measurements. Missing values are denoted as NaN. Units and reference ranges are provided where applicable.
'''
USERPROMPT_ZERO_SHOT = HISTORY_INTRO + '''\
Use the provided historical records to predict the specified outcome associated with the target visit.

Task Description:
{TASK_DESCRIPTION}

Instructions & Output Format:
{RESPONSE_FORMAT}

Target Patient's Historical Records:
{TARGET_RECORD}

Your Answer:'''

USERPROMPT_FEW_SHOT = HISTORY_INTRO + '''\
You will also receive selected examples from other patients. Each example contains daily records from one hospital admission, using the same observation window and target-day inclusion rule as the target patient, together with the known task-specific outcome. Use these examples and the target patient's records to predict the specified outcome.

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
