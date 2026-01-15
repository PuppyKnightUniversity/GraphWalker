
def runexp(args):
    """
    Main function to run the experiment
    """
    import time
    from utils.logger import get_logger
    
    # Record start time
    start_time = time.time()
    
    # Create experiment info for automatic log file generation
    experiment_info = {
        'dataset': args.dataset,
        'model': getattr(args, 'llm_name', getattr(args, 'llm_local_path', 'unknown')),
        'method': args.method
    }
    
    # Initialize logger with automatic log file generation
    logger = get_logger("ICL-Experiment", experiment_info=experiment_info)
    
    if args.dataset in ['mimic4_los', 'mimic3_los', 'tjh_los']:
        task = 'length of stay prediction'
    elif args.dataset in ['mimic4_mortality', 'mimic3_mortality', 'tjh_mortality','tjh_mortality']:
        task = 'mortality prediction'
    elif args.dataset in ['mimic4_readmission']:
        task = 'readmission prediction'
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    # Display experiment start information
    experiment_args = {
        'dataset': args.dataset,
        'method': args.method,
        'task': task,  # You can make this configurable
        'toy_dataset': args.toy_dataset
    }
    logger.start_experiment(experiment_args)
    
    # prepare ICL dataset
    if args.dataset in ['mimic3_mortality', 'mimic3_los', 'mimic4_mortality', 'mimic4_los', 'mimic4_readmission', 'tjh_mortality', 'tjh_los']:
        # for ehr data
        from data.prepare_ehr_data import prepare_ehr_data
        train_dataset, val_dataset, test_dataset = prepare_ehr_data(args, logger)
    elif args.dataset in ['cmb_exam_patient', 'cmb_clin', 'medqa']:
        # TODO: for clinical text data
        from data.prepare_clinical_text_data import prepare_clinical_text_data
        train_dataset, val_dataset, test_dataset = prepare_clinical_text_data(args, logger)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    # Select and run method
    logger.info(f"Initializing method: [bold]{args.method}[/bold]")
    if args.method in ["llm_zero_shot", "graph_walker"]:
        from run.run_llm.run import run_llm_inference_for_ICL as run_method
    else:
        logger.error(f"Method {args.method} not supported")
        raise ValueError(f"Method {args.method} not supported")

    logger.info("Starting experiment execution...")
    run_method(args, train_dataset, val_dataset, test_dataset, logger)
    
    # Calculate and log total execution time
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.success(f"Experiment completed successfully! 总执行时间: {total_time:.2f} 秒")

