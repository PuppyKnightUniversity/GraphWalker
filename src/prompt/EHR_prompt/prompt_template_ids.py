"""
IDS-specific prompt templates for mimic3_mortality with enforced Reason + Answer outputs.
"""

IDS_RESPONSE_FORMAT = {
    'mimic3_mortality': """\
Respond with two lines:
Reason: <step-by-step clinical reasoning leading to the decision>
Answer: <probability between 0 and 1 for in-hospital mortality, higher means higher risk>"""
}

# Zero-shot CoT seed to generate the initial reasoning path
IDS_ZERO_SHOT_COT_TEMPLATE = """\
You are an experienced critical care physician. Analyze the longitudinal ICU measurements and reason step by step before providing the mortality risk.

PATIENT INFORMATION:
- Number of measurements: {LENGTH}
- Measurement times (hours from admission): [{RECORD_TIME_LIST}]

Your Task:
{TASK_DESCRIPTION}

Required Output Format:
{RESPONSE_FORMAT}

Now analyze and reason about the following patient:

Clinical Features Over Time:
{DETAIL}"""

# Few-shot prompt used in iterative IDS rounds
IDS_FEW_SHOT_TEMPLATE = """\
You are an experienced critical care physician. Analyze the longitudinal ICU measurements and reason step by step before providing the mortality risk.

PATIENT INFORMATION:
- Number of measurements: {LENGTH}
- Measurement times (hours from admission): [{RECORD_TIME_LIST}]

Your Task:
{TASK_DESCRIPTION}

Here are {EXAMPLE_NUM} retrieved demonstration patients and their outcomes (1 = did not survive, 0 = survived):

{EXAMPLE}

Required Output Format:
{RESPONSE_FORMAT}

Now analyze and reason about the following patient:

Clinical Features Over Time:
{DETAIL}"""
