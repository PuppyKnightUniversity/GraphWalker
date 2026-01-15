# This file is used to wrap the ehr data into prompt template
from typing import Dict, Any, List
import argparse
import numpy as np
import re
import torch

def mimic3_smooth_hourly_data(patient_example: Dict[str, Any], keep_last: bool = True) -> Dict[str, Any]:
    '''
    Smooth the hourly data of the patient example,
    keep the last data point of each hour by default.
    if keep_last is False, keep the first data point of each hour.
    Args:
        patient_example: Dict[str, Any]
            The patient example
        keep_last: bool
            Whether to keep the last data point of each hour
    Returns:
        smoothed_patient_example: Dict[str, Any]
            The smoothed patient example
    '''
    from collections import defaultdict
    X = patient_example['X']
    record_times = X[:, 0].astype(float)
    feature_data = X[:, 1:]
    hourly_groups = defaultdict(list)

    for i, time in enumerate(record_times):
        hour = int(time)
        hourly_groups[hour].append((i, time, feature_data[i]))

    smoothed_indices = []
    smoothed_times = []
    smoothed_features = []
    
    for hour in sorted(hourly_groups.keys()):
        hour_data = hourly_groups[hour]
        
        if keep_last:
            # retain the last data point of each hour
            selected_idx, selected_time, selected_features = hour_data[-1]
        else:
            # retain the first data point of each hour
            selected_idx, selected_time, selected_features = hour_data[0]
        
        smoothed_indices.append(selected_idx)
        smoothed_times.append(selected_time)
        smoothed_features.append(selected_features)
    
    # build the smoothed data
    smoothed_X = np.column_stack([
        np.array(smoothed_times).reshape(-1, 1),
        np.array(smoothed_features)
    ])
    
    # create the smoothed patient example
    smoothed_patient_example = patient_example.copy()
    smoothed_patient_example['X'] = smoothed_X
    
    return smoothed_patient_example

def mimic3_mortality_prompt_wrapper(patient_example: Dict[str, Any],
                                    is_few_shot: bool = False,
                                    icl_examples_list: List[Dict[str, Any]] = [],
                                    inference_type: str = 'only_answer',
                                    unit: bool = False,
                                    reference_range: bool = False,
                                    smooth_hourly_data: bool = False,
                                    keep_last: bool = True,
                                    add_smart_logits: bool = False,
                                    add_smart_logits_for_test_example: bool = False) -> str:
    '''
    Wrap the ehr data into prompt template for mimic3 mortality data
    
    Args:
        patient_example: Dict[str, Any] 
            The ehr data of a patient
        is_few_shot: bool
            Whether to use few-shot ICL 
        inference_type: str
            The type of inference
        icl_examples_list: List[str]
            The list of few-shot examples
        unit: bool
            Whether to include unit in the detail
        reference_range: bool
            Whether to include reference range in the detail
        smooth_hourly_data: bool
            Whether to smooth the hourly data
        keep_last: bool
            Whether to keep the last data point of each hour
    Returns:
        prompt: str
            The prompt contains patient ehr data
    '''

    # detail EHR prompt for one patient
    detail = patient_example['detail']

    # load template
    if is_few_shot:
        if add_smart_logits:
            from prompt.prompt_template import USERPROMPT_FEW_SHOT_SMART_WITH_LOGITS as PROMPT_TEMPLATE
        else:
            from prompt.prompt_template import USERPROMPT_FEW_SHOT as PROMPT_TEMPLATE
    else:
        # zero-shot ICL
        from prompt.prompt_template import USERPROMPT_ZERO_SHOT as PROMPT_TEMPLATE
    
    from prompt.prompt_template import TASK_DESCRIPTION
    
    # load response format
    if inference_type == 'only_answer':
        from prompt.prompt_template import RESPONSE_FORMAT_ONLY_ANSWER as RESPONSE_FORMAT
    else:
        raise ValueError(f'we have not implemented the inference type for {inference_type}')

    # load few-shot examples
    if is_few_shot:
        # Each example contains the EHR detail and a binary prediction label (0/1)
        if add_smart_logits:
            def convert_logits_to_risk_value(smart_logits):
                '''
                Convert 2D logits array to risk value (probability of positive class)
                Args:
                    smart_logits: torch.Tensor or np.ndarray of shape [2]
                Returns:
                    risk_value: float, probability of positive class (class 1)
                '''
                # Convert to numpy if it's a torch tensor
                if isinstance(smart_logits, torch.Tensor):
                    logits_np = smart_logits.cpu().numpy()
                else:
                    logits_np = np.array(smart_logits)
                
                # Apply softmax to convert logits to probabilities
                exp_logits = np.exp(logits_np - np.max(logits_np))  # numerical stability
                probs = exp_logits / np.sum(exp_logits)
                
                # Return probability of positive class (index 1)
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
        
    X = patient_example['X']
    record_times = X[:, 0].astype(float)

    prompt = PROMPT_TEMPLATE.format(
        LENGTH=len(record_times),
        RECORD_TIME_LIST=', '.join([f"{float(t):.2f}" for t in record_times]),
        DETAIL=detail,
        RESPONSE_FORMAT=RESPONSE_FORMAT['mimic3_mortality'],
        TASK_DESCRIPTION=TASK_DESCRIPTION['mimic3_mortality'],
        EXAMPLE=example,)
    

    # For ablation study, whether to add SMART model logits to the test example
    if add_smart_logits_for_test_example:
        prompt += f"\nExpert Model Outputs:  {convert_logits_to_risk_value(patient_example['smart_logits']):.4f}"

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
    if args.dataset == 'mimic3_mortality':
        prompt_wraper_func = mimic3_mortality_prompt_wrapper
    else:
        raise ValueError(f'we have not implemented the prompt wrapper for {args.dataset}')
    
    
    # transform EHR data into prompt named "detail"
    for split_name, dataset in zip(['train', 'val', 'test'], [train_dataset, val_dataset, test_dataset]):
        detail_all = []
        patient_num = len(dataset['X'])
        for i in range(patient_num):
            patient_example = {}
            patient_example['X'] = dataset['X'][i]
            patient_example['t'] = dataset['t'][i]
            patient_example['y'] = dataset['y'][i]
            patient_example['header'] = dataset['header'][i]
            patient_example['name'] = dataset['name'][i]
            detail = transform_mimic3_mortality_ehr_to_detail_prompt(patient_example, 
                                                                     unit=args.unit, 
                                                                     reference_range=args.reference_range, 
                                                                     smooth_hourly_data=True, 
                                                                     keep_last=True)
            detail_all.append(detail)
            
        dataset['detail'] = detail_all
    
    # TODO: caculate different embeddings here   
    '''
        Use smart model to caculate embedding and psudo label
    '''
    if args.method == "llm_smart_embedding_topk" or "graph_walker" or args.llm_smart_embedding_topk_add_smart_logits or args.random_few_shot_add_smart_logits:
        from run.run_smart.smart_embedding import calculate_smart_embedding
        train_dataset, val_dataset, test_dataset = calculate_smart_embedding(args, train_dataset, val_dataset, test_dataset)

    # filter detail prompt by length
    train_dataset, val_dataset, test_dataset = filter_detail_prompt_by_length(logger = logger, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, max_tokens=max_tokens)
    

    # For ablation study, determine whether to add SMART model logits
    if args.llm_smart_embedding_topk_add_smart_logits or args.random_few_shot_add_smart_logits or args.graph_walker_add_smart_logits:
        add_smart_logits = True
    else:
        add_smart_logits = False

    # For ablation study, determine whether to add SMART model logits to the test example
    if args.llm_smart_embedding_topk_add_smart_logits_for_test_example or args.random_few_shot_add_smart_logits_for_test_example:
        add_smart_logits_for_test_example = True
    else:
        add_smart_logits_for_test_example = False

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
    patient_num = len(test_dataset['X'])
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
    
    return train_dataset, val_dataset, test_dataset




def transform_mimic3_mortality_ehr_to_detail_prompt(patient_example: Dict[str, Any],
                                                    unit: bool = False,
                                                    reference_range: bool = False,
                                                    smooth_hourly_data: bool = False,
                                                    keep_last: bool = True) -> str:
    '''
    Transform the ehr data into detail prompt for mimic3 mortality data
    Args:
        patient_example: Dict[str, Any]
            The ehr data of a patient
        unit: bool
            Whether to include unit in the detail
        reference_range: bool
            Whether to include reference range in the detail
        smooth_hourly_data: bool
            Whether to smooth the hourly data
        keep_last: bool
            Whether to keep the last data point of each hour
    Returns:
        detail: str
            The detail string of the ehr data
    '''
    # predefine function1
    def format_EHR_detail(patient_data: np.ndarray, 
                          features: List[str], 
                          mask: np.ndarray,
                          unit: bool = False,
                          reference_range: bool = False,
                          dataset_name: str = 'mimic3_mortality') -> str:
        '''
        Format the ehr data into detail string, it will be called by prepare_prompt_for_patient_example

        Args:
            patient_data: np.ndarray
                The ehr data of a patient, numeric data
            features: List[str]
                The features of the ehr data
            mask: np.ndarray
                The mask of the ehr data, 1 for missing value, 0 for non-missing value
            unit: bool
                Whether to include unit in the detail
            reference_range: bool
                Whether to include reference range in the detail
            dataset_name: str
                The name of the dataset
        Returns:
            detail: str
                The detail string of the ehr data
        '''
        feature_values = {}
        # Define some categorical features with their possible values
        categorical_features_dict = {
            "Glascow coma scale eye opening": {
                1: "No Response",
                2: "To Pain",
                3: "To Speech",
                4: "Spontaneously",
            },
            "Glascow coma scale motor response": {
                1: "No Response",
                2: "Abnormal Extension",
                3: "Abnormal Flexion",
                4: "Flex-withdraws",
                5: "Localizes Pain",
                6: "Obeys Commands",
            },
            "Glascow coma scale verbal response": {
                1: "No Response",
                2: "Incomprehensible sounds",
                3: "Inappropriate Words",
                4: "Confused",
                5: "Oriented",
            },
        }

        for i, feature in enumerate(features):
            feature_values[feature] = []
            for visit_idx in range(patient_data.shape[0]):
                if mask[visit_idx, i] == 1:
                    feature_values[feature].append('NaN')
                else:
                    value = patient_data[visit_idx, i]
                    if feature in categorical_features_dict:
                        if not np.isnan(value):
                            feature_values[feature].append(categorical_features_dict[feature].get(int(value), str(value)))
                        else:
                            feature_values[feature].append('NaN')
                    else:
                        feature_values[feature].append(f"{value}")
            
        # load unit and reference range
        import json
        from prompt.prompt_template import UNIT, REFERENCE_RANGE
        unit_values = dict(json.load(open(UNIT[dataset_name])))
        range_values = dict(json.load(open(REFERENCE_RANGE[dataset_name])))

        detail = ''
        for feature in features:
            unit_range = ''
            if unit or reference_range:
                unit_range = ' ('
                if unit:
                    unit_range += f'{unit_values[feature]} '
                if reference_range:
                    unit_range += range_values[feature]
                unit_range = unit_range.rstrip() + ')'
            detail += f"- {feature}{unit_range}: [{', '.join(feature_values[feature])}]\n"
        
        return detail.strip()

    # predefine function2
    def extract_leading_number(s):
        '''
        Extract the leading number from the string
        '''
        s = str(s)
        match = re.match(r'^\s*(-?\d+\.?\d*)', s)
        if match:
            return match.group(1) 
        return np.nan 
    # begin process patient example
    # smooth patient data
    if smooth_hourly_data:
        patient_example = mimic3_smooth_hourly_data(patient_example, keep_last=keep_last)
    vectorized_extract = np.vectorize(extract_leading_number)
    X = patient_example['X']
    header = patient_example['header']
    record_times = X[:, 0].astype(str)
    feature_data = X[:, 1:].astype(str)
    feature_names = header[1:]    
    mask = (feature_data == '')

    numeric_data = np.full(feature_data.shape, np.nan, dtype=float)
    non_missing_indices = ~mask
    data_to_clean = feature_data[non_missing_indices]
    
    cleaned_data_str = vectorized_extract(data_to_clean)  
    cleaned_data_float = cleaned_data_str.astype(float)
    
    numeric_data[non_missing_indices] = cleaned_data_float
    
    # fomulate detail EHR prompt for one patient
    detail = format_EHR_detail(numeric_data, 
                               feature_names, 
                               mask, unit=unit, 
                               reference_range=reference_range, 
                               dataset_name='mimic3_mortality')
    
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



