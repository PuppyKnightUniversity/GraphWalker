from typing import Dict, Any, List
import argparse
import numpy as np
import re
import torch


def text_prompt_wrapper_func(patient_example: Dict[str, Any],
                             dataset_name: str,
                             is_few_shot: bool = False,
                             icl_examples_list: List[Dict[str, Any]] = [],
                             inference_type: str = 'only_answer') -> str:
    detail = patient_example['detail']

    if is_few_shot:
        if dataset_name == 'cmb_exam_patient':
            from prompt.Text_prompt.prompt_template import USERPROMPT_FEW_SHOT_CHINESE as PROMPT_TEMPLATE
        else:
            raise ValueError(f'Unsupported prompt dataset: {dataset_name}')
    else:
        if dataset_name == 'cmb_exam_patient':
            from prompt.Text_prompt.prompt_template import USERPROMPT_ZERO_SHOT_CHINESE as PROMPT_TEMPLATE
        else:
            raise ValueError(f'Unsupported prompt dataset: {dataset_name}')

    from prompt.Text_prompt.prompt_template import TASK_DESCRIPTION

    if inference_type == 'only_answer':
        from prompt.Text_prompt.prompt_template import RESPONSE_FORMAT_ONLY_ANSWER as RESPONSE_FORMAT
    else:
        raise ValueError(f'Unsupported inference type: {inference_type}')

    if is_few_shot:
        if dataset_name == 'cmb_exam_patient':
            example = '\n\n'.join([
                f"例子 {i+1}:\n{icl_example['detail']}\n答案: {icl_example['label']}"
                for i, icl_example in enumerate(icl_examples_list)
            ])
        else:
            raise ValueError(f'Unsupported prompt dataset: {dataset_name}')
    else:
        example = ''

    prompt = PROMPT_TEMPLATE.format(
        DETAIL=detail,
        RESPONSE_FORMAT=RESPONSE_FORMAT[dataset_name],
        TASK_DESCRIPTION=TASK_DESCRIPTION[dataset_name],
        EXAMPLE=example,
    )

    return prompt

def train_val_test_dataset_prompt_wrapper(args,
                                          logger,
                                          train_dataset,
                                          val_dataset,
                                          test_dataset,
                                          is_few_shot: bool = False,
                                          max_tokens: int = 10000) -> List[str]:
    logger.processing_start("Prompt wrapping")
    assert args.dataset in ['medqa', 'cmb_exam_patient', 'cmb_clin']

    for split_name, dataset in zip(['train', 'val', 'test'], [train_dataset, val_dataset, test_dataset]):
        detail_all = []
        patient_num = len(dataset[dataset.keys()[0]])
        for i in range(patient_num):
            patient_example = {}
            for key in dataset.keys():
                patient_example[key] = dataset[key][i]
            detail = transform_clinical_text_to_detail_prompt(patient_example, dataset_name=args.dataset)
            detail_all.append(detail)

        dataset['detail'] = detail_all


    train_dataset, val_dataset, test_dataset = filter_detail_prompt_by_length(logger = logger, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, max_tokens=max_tokens)

    ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS = []
    if is_few_shot:
        from icl.select_icl_examples import FIND_ICL_EXAMPLES
        ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS = FIND_ICL_EXAMPLES(args,
                                                                    test_dataset,
                                                                    method=args.method,
                                                                    train_dataset=train_dataset,
                                                                    num_examples=args.icl_examples_num,
                                                                    logger=logger)


    prompt_all = []
    patient_num = len(test_dataset[test_dataset.keys()[0]])
    progress = logger.create_progress("Processing test patients prompt wrapping", patient_num)
    with progress:
        task = progress.add_task("Processing test patients prompt wrapping", total=patient_num)
        for i in range(patient_num):
            patient_example = {}
            for key in test_dataset.keys():
                patient_example[key] = test_dataset[key][i]

            ICL_EXAMPLES_LIST = ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS[i] if is_few_shot else []

            prompt = text_prompt_wrapper_func(patient_example,
                                              dataset_name=args.dataset,
                                              is_few_shot=is_few_shot,
                                              icl_examples_list=ICL_EXAMPLES_LIST,
                                              inference_type=args.inference_type,)
            prompt_all.append(prompt)
            progress.update(task, advance=1)

    test_dataset['data_prompt_fomat'] = prompt_all

    if is_few_shot:
        test_dataset['ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS'] = ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS

    logger.processing_complete("Processing test patients prompt wrapping")

    logger.show_message_example(test_dataset['data_prompt_fomat'][0], "Example Prompt")

    return train_dataset, val_dataset, test_dataset




def transform_clinical_text_to_detail_prompt(patient_example: Dict[str, Any],
                                             dataset_name: str) -> str:
    if dataset_name == 'cmb_exam_patient':
        question_key = 'question'
        option_key = 'option'
        question_type_key = 'question_type'
        from prompt.Text_prompt.prompt_template import QUESTION_DETAIL_TEMPLATE_CHINESE as QUESTION_DETAIL_TEMPLATE
    else:
        raise ValueError(f'Unsupported clinical text dataset: {dataset_name}')

    detail = ''
    question_type = patient_example[question_type_key]
    question = patient_example[question_key]
    option = '\n'.join([f"{key}. {patient_example[option_key][key]}" for key in patient_example[option_key].keys()])
    detail += QUESTION_DETAIL_TEMPLATE.format(question_type=question_type,
                                              question=question,
                                              option=option,
                                              ) + '\n'

    return detail



def filter_detail_prompt_by_length(logger, train_dataset, val_dataset, test_dataset, max_tokens: int = 10000) -> bool:

    logger.processing_complete("Processing test patients prompt wrapping")

    logger.processing_start(f"Filtering detail prompts longer than {max_tokens} tokens")

    try:
        import tiktoken
    except ImportError:
        logger.warning("tiktoken not found, falling back to word count approximation")
        tiktoken = None

    def count_tokens(text):
        if tiktoken is not None:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        else:
            return len(text.split())

    for split_name, dataset in zip(['train', 'val', 'test'], [train_dataset, val_dataset, test_dataset]):
        original_count = len(dataset['detail'])

        valid_indices = []
        for i, prompt in enumerate(dataset['detail']):
            token_count = count_tokens(prompt)
            if token_count <= max_tokens:
                valid_indices.append(i)

        filtered_count = len(valid_indices)

        for key in dataset.keys():
            if isinstance(dataset[key], list):
                dataset[key] = [dataset[key][i] for i in valid_indices]

        logger.info(f"{split_name}: filtered {original_count - filtered_count} samples "
                   f"({filtered_count}/{original_count} remaining)")

    logger.processing_complete(f"Filtering detail prompts longer than {max_tokens} tokens")

    logger.processing_complete("Prompt wrapping")

    return train_dataset, val_dataset, test_dataset



