QUESTION_DETAIL_TEMPLATE_CHINESE = """\
以下是一道{question_type}：{question}

选项：
{option}
"""

TASK_DESCRIPTION = {
    "cmb_exam_patient": "你是一个医疗领域的专家，非常擅长解答医学问题。你的任务是根据给定的医学问题，选择正确的选项。",
}

RESPONSE_FORMAT_ONLY_ANSWER = {
    "cmb_exam_patient": "请直接输出正确选项对应的字母，别的一个字也不要说，例如: A",
}


USERPROMPT_ZERO_SHOT_CHINESE = """\
    你的任务:
    {TASK_DESCRIPTION}

    指令和输出格式:
    {RESPONSE_FORMAT}
    
    {EXAMPLE}
    
    现在，请回答以下问题:
    {DETAIL}
"""

USERPROMPT_FEW_SHOT_CHINESE = """\
    你的任务:
    {TASK_DESCRIPTION}

    指令和输出格式:
    {RESPONSE_FORMAT}

    以下是一些例子，你可以使用这些例子来帮助你回答问题:
    {EXAMPLE}

    现在，请回答以下问题:
    {DETAIL}
"""