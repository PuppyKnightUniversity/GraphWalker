
def runexp(args):
    """
    Main function to run the experiment
    """
    from utils.utils import set_seed
    set_seed(args.seed)
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
    elif args.dataset in ['mimic4_mortality', 'mimic3_mortality', 'tjh_mortality']:
        task = 'mortality prediction'
    elif args.dataset in ['mimic4_readmission']:
        task = 'readmission prediction'
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    if args.method == 'graph_walker' and getattr(args, 'is_api', False):
        raise ValueError('GraphWalker scoring requires local token log-probabilities')
    if args.method in ['llm_zero_shot', 'graph_walker']:
        from run.run_llm.run import run_llm_inference_for_ICL as run_method
    elif args.method == 'llm_sft_train':
        from run.run_sft.run import run as run_method
    elif args.method == 'llm_sft_eval':
        from run.run_sft.sft_eval import run as run_method
    elif args.method == 'llm_sft_eval_vllm':
        from run.run_sft.sft_eval_vllm import run as run_method
    else:
        raise ValueError(f"Method {args.method} not supported")

    # Display experiment start information
    experiment_args = {
        'dataset': args.dataset,
        'method': args.method,
        'task': task,
        'toy_dataset': args.toy_dataset
    }
    logger.start_experiment(experiment_args)

    # prepare ICL dataset
    if args.dataset in ['mimic3_mortality', 'mimic3_los', 'mimic4_mortality', 'mimic4_los', 'mimic4_readmission', 'tjh_mortality', 'tjh_los']:
        # for ehr data
        from data.prepare_ehr_data import prepare_ehr_data
        train_dataset, val_dataset, test_dataset = prepare_ehr_data(args, logger)
    elif args.dataset in ['cmb_exam_patient', 'cmb_clin', 'medqa']:
        # Load clinical-text data.
        from data.prepare_clinical_text_data import prepare_clinical_text_data
        train_dataset, val_dataset, test_dataset = prepare_clinical_text_data(args, logger)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    # Select and run method
    logger.info(f"Initializing method: [bold]{args.method}[/bold]")
    logger.info("Starting experiment execution...")
    metrics = run_method(args, train_dataset, val_dataset, test_dataset, logger)
    if getattr(args, 'metrics_save_path', None):
        import hashlib
        import json
        from pathlib import Path
        import numpy as np
        def json_default(value):
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.tolist()
            raise TypeError(f'Unsupported result value: {type(value)}')
        record = {
            'config': vars(args),
            'test_size': len(test_dataset['y']),
            'test_ids_sha256': hashlib.sha256(json.dumps(
                test_dataset.get('name', []), default=json_default).encode()).hexdigest(),
            'metrics': metrics,
        }
        path = Path(args.metrics_save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, default=json_default) + '\n')

    # Calculate and log total execution time
    end_time = time.time()
    total_time = end_time - start_time

    logger.success(f"Experiment completed successfully! 总执行时间: {total_time:.2f} 秒")
    return metrics
