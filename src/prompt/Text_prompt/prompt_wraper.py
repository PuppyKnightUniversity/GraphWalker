# This file is used to wrap the ehr data into prompt template
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
    '''
    Wrap the text data into prompt template
    Args:
        patient_example: Dict[str, Any]
            The text data of a patient
        dataset_name: str
            The name of the dataset
        is_few_shot: bool
            Whether to use few-shot ICL
        icl_examples_list: List[Dict[str, Any]]
            The list of few-shot examples
        inference_type: str
            The type of inference
    Returns:
        prompt: str
            The prompt contains text data
    '''
    detail = patient_example['detail']
    
    if is_few_shot:
        if dataset_name == 'cmb_exam_patient':
            from prompt.Text_prompt.prompt_template import USERPROMPT_FEW_SHOT_CHINESE as PROMPT_TEMPLATE
        else:
            raise ValueError(f'we have not implemented the prompt template for {dataset_name}')
    else:
        if dataset_name == 'cmb_exam_patient':
            from prompt.Text_prompt.prompt_template import USERPROMPT_ZERO_SHOT_CHINESE as PROMPT_TEMPLATE
        else:
            raise ValueError(f'we have not implemented the prompt template for {dataset_name}')
    
    from prompt.Text_prompt.prompt_template import TASK_DESCRIPTION
    
    if inference_type == 'only_answer':
        from prompt.Text_prompt.prompt_template import RESPONSE_FORMAT_ONLY_ANSWER as RESPONSE_FORMAT
    else:
        raise ValueError(f'we have not implemented the inference type for {inference_type}')
    
    if is_few_shot:
        if dataset_name == 'cmb_exam_patient':
            example = '\n\n'.join([
                f"例子 {i+1}:\n{icl_example['detail']}\n答案: {icl_example['label']}"
                for i, icl_example in enumerate(icl_examples_list)
            ])
        else:
            raise ValueError(f'we have not implemented the prompt template for {dataset_name}')
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
    '''
    Wrap the train/val/test dataset into prompt template
    
    Args:
        args: Arguments object containing dataset configuration
        logger: Logger object for logging progress
        train_dataset: Training dataset dictionary
        val_dataset: Validation dataset dictionary
        test_dataset: Test dataset dictionary
        is_few_shot: Whether to use few-shot learning
        max_tokens: Maximum token limit for filtering prompts (default: 4000)
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset) with filtered prompts
    '''
    # wrap prompt for train/val/test dataset
    logger.processing_start("Prompt wrapping")
    assert args.dataset in ['medqa', 'cmb_exam_patient', 'cmb_clin']
    
    # transform clinical text data into prompt named "detail"
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
    
    # TODO: caculate semantic embeddings here   

    # filter detail prompt by length
    train_dataset, val_dataset, test_dataset = filter_detail_prompt_by_length(logger = logger, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, max_tokens=max_tokens)
    
    # select ICL examples 
    ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS = []
    if is_few_shot:
        from icl.select_icl_examples import FIND_ICL_EXAMPLES
        ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS = FIND_ICL_EXAMPLES(args, 
                                                                    test_dataset, 
                                                                    method=args.method, 
                                                                    train_dataset=train_dataset, 
                                                                    num_examples=args.icl_examples_num,
                                                                    logger=logger)


    # wrap prompt for test dataset
    prompt_all = []
    patient_num = len(test_dataset[test_dataset.keys()[0]])
    progress = logger.create_progress("Processing test patients prompt wrapping", patient_num)
    with progress:
        task = progress.add_task("Processing test patients prompt wrapping", total=patient_num)
        # Begin to wrap prompt for each patient
        for i in range(patient_num):
            # build patient example
            patient_example = {}
            for key in test_dataset.keys():
                patient_example[key] = test_dataset[key][i]
            
            # extract ICL examples for the current test patient
            ICL_EXAMPLES_LIST = ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS[i] if is_few_shot else []
            
            # select ICL examples
            prompt = text_prompt_wrapper_func(patient_example,
                                              dataset_name=args.dataset,
                                              is_few_shot=is_few_shot, 
                                              icl_examples_list=ICL_EXAMPLES_LIST,
                                              inference_type=args.inference_type,)
            prompt_all.append(prompt)
            progress.update(task, advance=1)
    
    test_dataset['data_prompt_fomat'] = prompt_all
    
    # Save ICL examples list for conditional entropy computation
    if is_few_shot:
        test_dataset['ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS'] = ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS
    
    logger.processing_complete("Processing test patients prompt wrapping")    
    
    # print example of prompt
    logger.show_message_example(test_dataset['data_prompt_fomat'][0], "Example Prompt")
    
    return train_dataset, val_dataset, test_dataset




def transform_clinical_text_to_detail_prompt(patient_example: Dict[str, Any],
                                             dataset_name: str) -> str:
    '''
    Transform the clinical text data into detail prompt
    Args:
        patient_example: Dict[str, Any]
            The clinical text data of a patient
        dataset_name: str
            The name of the dataset
    Returns:
        detail: str
            The detail string of the clinical text data
    '''
    if dataset_name == 'cmb_exam_patient':
        question_key = 'question'
        option_key = 'option'
        question_type_key = 'question_type'
        from prompt.Text_prompt.prompt_template import QUESTION_DETAIL_TEMPLATE_CHINESE as QUESTION_DETAIL_TEMPLATE
    else:
        raise ValueError(f'we have not implemented the clinical text data for {dataset_name}')
    
    detail = ''
    # load question, option, answer
    question_type = patient_example[question_type_key]
    question = patient_example[question_key]
    option = '\n'.join([f"{key}. {patient_example[option_key][key]}" for key in patient_example[option_key].keys()])
    detail += QUESTION_DETAIL_TEMPLATE.format(question_type=question_type, 
                                              question=question, 
                                              option=option, 
                                              ) + '\n'
    
    return detail



def filter_detail_prompt_by_length(logger, train_dataset, val_dataset, test_dataset, max_tokens: int = 10000) -> bool:
    '''
    Filter the prompt by length
    Args:
        logger: Logger object for logging progress
        train_dataset: Training dataset dictionary
        val_dataset: Validation dataset dictionary
        test_dataset: Test dataset dictionary
        max_tokens: Maximum token limit for filtering prompts (default: 4000)
    Returns:
        bool: True if the prompt is filtered, False otherwise
    '''

    logger.processing_complete("Processing test patients prompt wrapping")
        
    # filter data which is too long
    logger.processing_start(f"Filtering detail prompts longer than {max_tokens} tokens")
    
    try:
        import tiktoken
    except ImportError:
        logger.warning("tiktoken not found, falling back to word count approximation")
        tiktoken = None
    
    def count_tokens(text):
        """Count tokens in text using tiktoken or word count as fallback"""
        if tiktoken is not None:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        else:
            # Fallback to approximate word count
            return len(text.split())
    
    for split_name, dataset in zip(['train', 'val', 'test'], [train_dataset, val_dataset, test_dataset]):
        original_count = len(dataset['detail'])
        
        # Count tokens for each prompt
        valid_indices = []
        for i, prompt in enumerate(dataset['detail']):
            token_count = count_tokens(prompt)
            if token_count <= max_tokens:
                valid_indices.append(i)
        
        filtered_count = len(valid_indices)
        
        # Filter all keys in the dataset
        for key in dataset.keys():
            if isinstance(dataset[key], list):
                dataset[key] = [dataset[key][i] for i in valid_indices]
        
        logger.info(f"{split_name}: filtered {original_count - filtered_count} samples "
                   f"({filtered_count}/{original_count} remaining)")
    
    logger.processing_complete(f"Filtering detail prompts longer than {max_tokens} tokens")

    logger.processing_complete("Prompt wrapping")
    
    return train_dataset, val_dataset, test_dataset

 

