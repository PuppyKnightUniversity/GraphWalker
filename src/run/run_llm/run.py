from utils.logger import get_logger

def run_llm_inference_for_ICL(args, train_dataset, val_dataset, test_dataset, logger=None):
    # 1. wrap prompt
    if args.dataset in ['mimic3_mortality', 'mimic4_mortality', 'tjh_mortality', 'mimic3_los', 'mimic4_readmission']:
        from prompt.EHR_prompt.prompt_wraper import train_val_test_dataset_prompt_wrapper
    elif args.dataset in ['cmb_exam_patient']:
        from prompt.Text_prompt.prompt_wraper import train_val_test_dataset_prompt_wrapper
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    # zero-shot or few-shot
    if args.method == "graph_walker":
        is_few_shot = True
    else:
        is_few_shot = False
        
    train_dataset, val_dataset, test_dataset = train_val_test_dataset_prompt_wrapper(args, 
                                                                                     logger,
                                                                                     train_dataset, 
                                                                                     val_dataset, 
                                                                                     test_dataset, 
                                                                                     is_few_shot=is_few_shot,
                                                                                     max_tokens= args.max_tokens_each_patient)
    
    # 2. llm inference
    from llms.inference import llm_dataset_inference
    responses,logits = llm_dataset_inference(args, train_dataset, val_dataset, test_dataset, logger=logger)

    # 3. metrics evaluation
    from utils.llm_eval import llm_response_evaluation
    bootstrap_metrics = llm_response_evaluation(args, responses, logits, test_dataset, logger=logger)
    
    # 4. compute average conditional entropy on test set if requested
    if getattr(args, 'final_delta_H', False):
        from utils.conditional_entropy import compute_average_conditional_entropy
        ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS = test_dataset.get('ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS', None)
        entropy_metrics = compute_average_conditional_entropy(
            args,
            train_dataset,
            test_dataset,
            ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS=ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS,
            return_metrics=True,
            logger=logger
        )
        logger.info(f"Test entropy metrics: {entropy_metrics}")
        # Add to metrics if needed
        if isinstance(bootstrap_metrics, dict):
            bootstrap_metrics.update(entropy_metrics)
    
    return bootstrap_metrics

    
