SYSTEMPROMPT = {
    'mimic3_mortality': 'You are an experienced critical care physician working in an Intensive Care Unit (ICU), skilled in interpreting complex longitudinal patient data and predicting clinical outcomes.',
}

TASK_DESCRIPTION = {
    'mimic3_mortality': 'Your primary task is to assess the provided medical data and analyze the health records from ICU visits to determine the likelihood of the patient not surviving their hospital stay.',
}

# answer straightly without any thinking
RESPONSE_FORMAT_ONLY_ANSWER = {
    'mimic3_mortality': '''\
Provide only a floating-point number between 0 and 1 representing the predicted probability of mortality (higher value means higher likelihood of death).

Do not provide any reasoning, explanation, or additional text. Only output the numerical value.

Example: 0.XX''',
}

UNIT = {
    'mimic3_mortality': './prompt/mimic3_unit.json',
}

REFERENCE_RANGE = {
    'mimic3_mortality': './prompt/mimic3_range.json',
}

# zero-shot user prompt template
USERPROMPT_ZERO_SHOT = """\
I will provide you with longitudinal medical information for a patient. Each clinical feature is presented as a list of values, corresponding to these visits. Missing values are represented as `NaN`. Note that units and reference ranges are provided alongside relevant features.

PATIENT INFORMATION:
- Number of measurements: {LENGTH}
- Measurement times (hours from admission): [{RECORD_TIME_LIST}]

Your Task:
{TASK_DESCRIPTION}

Instructions & Output Format:
{RESPONSE_FORMAT}

{EXAMPLE}

Now, please analyze and predict for the following patient:

Clinical Features Over Time:
{DETAIL}"""

# FIXME: add few-shot ICL prompt template!
USERPROMPT_FEW_SHOT = """\
I will provide you with longitudinal medical information for a patient. Each clinical feature is presented as a list of values, corresponding to these visits. Missing values are represented as `NaN`. Note that units and reference ranges are provided alongside relevant features.

PATIENT INFORMATION:
- Number of measurements: {LENGTH}
- Measurement times (hours from admission): [{RECORD_TIME_LIST}]

Your Task:
{TASK_DESCRIPTION}

Instructions & Output Format:
{RESPONSE_FORMAT}

Here are some examples of patient data and their corresponding labels(1 means not surviving, 0 means surviving). You can use these examples to help you make your prediction.

{EXAMPLE}

Now, please analyze and predict for the following patient:

Clinical Features Over Time:
{DETAIL}"""

# Few-shot ICL prompt template with SMART model logits
USERPROMPT_FEW_SHOT_SMART_WITH_LOGITS = """\
I will provide you with longitudinal medical information for a patient. Each clinical feature is presented as a list of values, corresponding to these visits. Missing values are represented as `NaN`. Note that units and reference ranges are provided alongside relevant features.

PATIENT INFORMATION:
- Number of measurements: {LENGTH}
- Measurement times (hours from admission): [{RECORD_TIME_LIST}]

Your Task:
{TASK_DESCRIPTION}

Instructions & Output Format:
{RESPONSE_FORMAT}

Here are some examples of patient data and their corresponding labels(1 means not surviving, 0 means surviving). For each example, we also provide the results from an expert EHR analysis model  for your reference(0.XX means the probability of not surviving,higher value means higher likelihood of death). You can use these examples and the expert model's outputs to help you make your prediction.

{EXAMPLE}

Now, please analyze and predict for the following patient. **Important**: The expert model's outputs are provided only as reference values to assist your analysis. Do NOT simply copy or directly adopt these values. You should conduct your own independent analysis based on the clinical features, and make necessary corrections or adjustments to the expert model's outputs when appropriate.

Clinical Features Over Time:
{DETAIL}"""
