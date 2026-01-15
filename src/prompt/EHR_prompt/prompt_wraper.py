# This file is used to wrap the ehr data into prompt template
from typing import Dict, Any, List, Tuple
import argparse
import numpy as np
import re
import torch


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


    # transform each EHR data into prompt named "detail", each dataset would have a "detail" key
    train_dataset, val_dataset, test_dataset = transform_ehr_to_detail_prompt(args, train_dataset, val_dataset, test_dataset)

    # Caculate different embeddings here   
    train_dataset, val_dataset, test_dataset = calculate_embeddings(args, train_dataset, val_dataset, test_dataset, logger)
    
    # filter detail prompt by length
    train_dataset, val_dataset, test_dataset = filter_detail_prompt_by_length(logger = logger, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, max_tokens=max_tokens)
    
    # select ICL examples 
    ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS = []
    if is_few_shot:
        from icl.select_icl_examples import FIND_ICL_EXAMPLES
        ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS = FIND_ICL_EXAMPLES(args, test_dataset, method=args.method, train_dataset=train_dataset, val_dataset=val_dataset, num_examples=args.icl_examples_num, logger=logger)

    # fomulate final prompt for inference
    test_dataset = formulate_final_prompt_for_inference(args, logger, test_dataset, ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS, is_few_shot)
    
    return train_dataset, val_dataset, test_dataset


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
        for key in list(dataset.keys()):
            value = dataset[key]
            # list 类型：逐个索引筛选
            if isinstance(value, list):
                dataset[key] = [value[i] for i in valid_indices]
            # torch.Tensor 类型（如 smart_embedding、semantic_embedding、smart_logits）
            elif torch.is_tensor(value):
                # 保留第一维与 detail 对齐
                dataset[key] = value[valid_indices]
        
        logger.info(f"{split_name}: filtered {original_count - filtered_count} samples "
                   f"({filtered_count}/{original_count} remaining)")
    
    logger.processing_complete(f"Filtering detail prompts longer than {max_tokens} tokens")

    logger.processing_complete("Prompt wrapping")
    
    return train_dataset, val_dataset, test_dataset

def transform_ehr_to_detail_prompt(args, train_dataset, val_dataset, test_dataset) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:

    for split_name, dataset in zip(['train', 'val', 'test'], [train_dataset, val_dataset, test_dataset]):
        detail_all = []
        for i in range(len(dataset['X'])):
            patient_example = {
                'X': dataset['X'][i],
                't': dataset['t'][i],
                'y': dataset['y'][i],
                'header': dataset['header'][i],
                'name': dataset['name'][i],
            }
            if args.dataset == 'mimic4_mortality':
                from prompt.EHR_prompt.mimic4.mortality.prompt import transform_mimic4_mortality_ehr_to_detail_prompt
                detail = transform_mimic4_mortality_ehr_to_detail_prompt(patient_example)
            elif args.dataset == 'tjh_mortality':
                from prompt.EHR_prompt.tjh.mortality.prompt import transform_tjh_mortality_ehr_to_detail_prompt
                detail = transform_tjh_mortality_ehr_to_detail_prompt(patient_example)
            elif args.dataset == 'mimic3_los':
                from prompt.EHR_prompt.mimic3.los.prompt import transform_mimic3_los_ehr_to_detail_prompt
                detail = transform_mimic3_los_ehr_to_detail_prompt(patient_example)
            elif args.dataset == 'mimic4_readmission':
                from prompt.EHR_prompt.mimic4.readmission.prompt import transform_mimic4_readmission_ehr_to_detail_prompt
                detail = transform_mimic4_readmission_ehr_to_detail_prompt(patient_example)
            elif args.dataset == 'mimic3_mortality':
                from prompt.EHR_prompt.mimic3.mortality.prompt import transform_mimic3_mortality_ehr_to_detail_prompt
                detail = transform_mimic3_mortality_ehr_to_detail_prompt(patient_example)
            else:
                raise ValueError(f'we have not implemented the transform_ehr_to_detail_prompt for {args.dataset}')
            detail_all.append(detail)
        dataset['detail'] = detail_all
    return train_dataset, val_dataset, test_dataset

def calculate_embeddings(args, train_dataset, val_dataset, test_dataset, logger) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    '''
    Calculate the embeddings for the train/val/test dataset
    Args:
        args: Arguments object containing dataset configuration
        train_dataset: Training dataset dictionary
        val_dataset: Validation dataset dictionary
        test_dataset: Test dataset dictionary
    Returns:
        Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]: The train/val/test dataset with embeddings
    '''
    # TODO: caculate different embeddings here   
    '''
        Use smart model to caculate embedding and psudo label
    '''
    if (
        args.method in ["llm_smart_embedding_topk", "graph_walker"]
        or args.llm_smart_embedding_topk_add_smart_logits
        or args.random_few_shot_add_smart_logits
        or args.embedding_model_name == 'smart'
    ):
        if 'data_smart' in train_dataset and 'data_smart' in val_dataset and 'data_smart' in test_dataset:
            from run.run_smart.smart_embedding import calculate_smart_embedding
            train_dataset, val_dataset, test_dataset = calculate_smart_embedding(args, train_dataset, val_dataset, test_dataset)
        else:
            logger.warning("SMART-adapted data not found; skipping SMART embedding")

    '''
        Calculate semantic embedding
    '''
    if args.method in ["llm_semantic_embedding_topk"] or args.embedding_model_name == 'qwen3-embedding-8b':
        from run.run_llm.semantic_embedding import calculate_semantic_embedding
        train_dataset, val_dataset, test_dataset = calculate_semantic_embedding(args, train_dataset, val_dataset, test_dataset)
    
    return train_dataset, val_dataset, test_dataset



def formulate_final_prompt_for_inference(args, logger, test_dataset, ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS, is_few_shot) -> Dict[str, Any]:


    # For ablation study, determine whether to add SMART model logits
    if args.llm_smart_embedding_topk_add_smart_logits or args.random_few_shot_add_smart_logits or args.graph_walker_add_smart_logits or args.cone_add_smart_logits:
        add_smart_logits = True
    else:
        add_smart_logits = False

    # For ablation study, determine whether to add SMART model logits to the test example
    if args.llm_smart_embedding_topk_add_smart_logits_for_test_example or args.random_few_shot_add_smart_logits_for_test_example or args.graph_walker_add_smart_logits_for_test_example:
        add_smart_logits_for_test_example = True
    else:
        add_smart_logits_for_test_example = False

    # wrap prompt for test dataset
    prompt_all = []
    patient_num = len(test_dataset['detail'])
    progress = logger.create_progress("Processing test patients prompt wrapping", patient_num)

    # wrap prompt for train/val/test dataset
    logger.processing_start("Prompt wrapping")
    if args.dataset == 'mimic3_mortality':
        from prompt.EHR_prompt.mimic3.mortality.prompt import mimic3_mortality_prompt_wrapper
        prompt_wraper_func = mimic3_mortality_prompt_wrapper
    elif args.dataset == 'mimic3_los':
        from prompt.EHR_prompt.mimic3.los.prompt import mimic3_los_prompt_wrapper
        prompt_wraper_func = mimic3_los_prompt_wrapper
    elif args.dataset == 'mimic4_mortality':
        from prompt.EHR_prompt.mimic4.mortality.prompt import mimic4_mortality_prompt_wrapper
        prompt_wraper_func = mimic4_mortality_prompt_wrapper
    elif args.dataset == 'mimic4_readmission':
        from prompt.EHR_prompt.mimic4.readmission.prompt import mimic4_readmission_prompt_wrapper
        prompt_wraper_func = mimic4_readmission_prompt_wrapper
    elif args.dataset == 'tjh_mortality':
        from prompt.EHR_prompt.tjh.mortality.prompt import tjh_mortality_prompt_wrapper
        prompt_wraper_func = tjh_mortality_prompt_wrapper
    else:
        raise ValueError(f'we have not implemented the prompt wrapper for {args.dataset}')

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
            prompt = prompt_wraper_func(patient_example, 
                                        is_few_shot=is_few_shot, 
                                        icl_examples_list=ICL_EXAMPLES_LIST,
                                        inference_type=args.inference_type, 
                                        unit=args.unit, 
                                        reference_range=args.reference_range,
                                        add_smart_logits=add_smart_logits,
                                        add_smart_logits_for_test_example=add_smart_logits_for_test_example,)
            prompt_all.append(prompt)
            progress.update(task, advance=1)
    
    test_dataset['data_prompt_fomat'] = prompt_all
    
    # Save ICL examples list for conditional entropy computation
    if is_few_shot:
        test_dataset['ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS'] = ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS
    
    logger.processing_complete("Processing test patients prompt wrapping")    
    
    # print example of prompt
    logger.show_message_example(test_dataset['data_prompt_fomat'][0], "Example Prompt")

    return test_dataset

