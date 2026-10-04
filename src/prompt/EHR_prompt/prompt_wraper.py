from typing import Dict, Any, Tuple


def train_val_test_dataset_prompt_wrapper(args,
                                          logger,
                                          train_dataset,
                                          val_dataset,
                                          test_dataset,
                                          is_few_shot: bool = False,
                                          max_tokens: int = 10000) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Serialize records, select demonstrations, and build prediction prompts."""
    train_dataset, val_dataset, test_dataset = transform_ehr_to_detail_prompt(args, train_dataset, val_dataset, test_dataset)

    validate_detail_prompt_lengths(logger, train_dataset, val_dataset, test_dataset,
                                   max_tokens=max_tokens)
    train_dataset, val_dataset, test_dataset = calculate_embeddings(
        args, train_dataset, val_dataset, test_dataset, logger)

    # select ICL examples
    ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS = []
    if is_few_shot:
        from icl.select_icl_examples import FIND_ICL_EXAMPLES
        ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS = FIND_ICL_EXAMPLES(args, test_dataset, method=args.method, train_dataset=train_dataset, val_dataset=val_dataset, num_examples=args.icl_examples_num, logger=logger)

    test_dataset = formulate_final_prompt_for_inference(args, logger, test_dataset, ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS, is_few_shot)

    return train_dataset, val_dataset, test_dataset


def validate_detail_prompt_lengths(logger, *datasets, max_tokens=10000):
    """Check the per-record budget without changing patient splits."""
    if max_tokens <= 0:
        raise ValueError('max_tokens must be positive')
    try:
        import tiktoken
        encoding = tiktoken.get_encoding('cl100k_base')
        count_tokens = lambda text: len(encoding.encode(text))
    except ImportError:
        logger.warning('tiktoken not found; record lengths use approximate word counts')
        count_tokens = lambda text: len(text.split())
    for split_index, dataset in enumerate(datasets):
        for index, detail in enumerate(dataset['detail']):
            length = count_tokens(detail)
            if length > max_tokens:
                raise ValueError(f'Record {index} in split {split_index} exceeds '
                                 f'--max_tokens_each_patient ({length} > {max_tokens}); '
                                 'increase the budget and model context length')


def transform_ehr_to_detail_prompt(args, train_dataset, val_dataset, test_dataset) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    from prompt.EHR_prompt.common import serialize_patient_record
    if getattr(args, 'ehr_protocol', 'historical_visits') != 'historical_visits':
        raise ValueError('Historical EHR prompts require --ehr_protocol historical_visits')
    for dataset in (train_dataset, val_dataset, test_dataset):
        dataset['detail'] = [
            serialize_patient_record(
                {key: values[i] for key, values in dataset.items()}, args.dataset,
                unit=args.unit, reference_range=args.reference_range)
            for i in range(len(dataset['X']))
        ]
    return train_dataset, val_dataset, test_dataset


def calculate_embeddings(args, train_dataset, val_dataset, test_dataset, logger) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Attach the selected encoder representations and optional expert logits."""
    if (
        args.method in ["llm_smart_embedding_topk"]
        or getattr(args, 'llm_smart_embedding_topk_add_smart_logits', False)
        or getattr(args, 'random_few_shot_add_smart_logits', False)
        or args.graph_walker_add_smart_logits
        or getattr(args, 'cone_add_smart_logits', False)
        or args.embedding_model_name == 'smart'
    ):
        if 'data_smart' in train_dataset and 'data_smart' in val_dataset and 'data_smart' in test_dataset:
            from run.run_smart.smart_embedding import calculate_smart_embedding
            train_dataset, val_dataset, test_dataset = calculate_smart_embedding(args, train_dataset, val_dataset, test_dataset)
        else:
            logger.warning("SMART-adapted data not found; skipping SMART embedding")

    if args.method in ["llm_semantic_embedding_topk"] or args.embedding_model_name == 'qwen3-embedding-8b':
        from run.run_llm.semantic_embedding import calculate_semantic_embedding
        train_dataset, val_dataset, test_dataset = calculate_semantic_embedding(args, train_dataset, val_dataset, test_dataset)

    return train_dataset, val_dataset, test_dataset



def formulate_final_prompt_for_inference(args, logger, test_dataset, ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS, is_few_shot) -> Dict[str, Any]:


    # For ablation study, determine whether to add SMART model logits
    if getattr(args, 'llm_smart_embedding_topk_add_smart_logits', False) or getattr(args, 'random_few_shot_add_smart_logits', False) or args.graph_walker_add_smart_logits or getattr(args, 'cone_add_smart_logits', False):
        add_smart_logits = True
    else:
        add_smart_logits = False

    # For ablation study, determine whether to add SMART model logits to the test example
    if getattr(args, 'llm_smart_embedding_topk_add_smart_logits_for_test_example', False) or getattr(args, 'random_few_shot_add_smart_logits_for_test_example', False) or args.graph_walker_add_smart_logits_for_test_example:
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
        raise ValueError(f'Unsupported prompt dataset: {args.dataset}')

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
