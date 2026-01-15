# In this file, we realize the main inference function for LLMs.

from utils.logger import get_logger

def llm_dataset_inference(args, 
                          train_dataset, 
                          val_dataset, 
                          test_dataset, 
                          logger=None)->list:
    """
    Inference the LLMs with the dataset.
    Args:
        args: the arguments.
        train_dataset: the train dataset.
        val_dataset: the val dataset.
        test_dataset: the test dataset.
        logger: the logger instance for progress display (optional).
    Returns:
        the list of responses.
    """
    if logger is None:
        logger = get_logger("LLM-Inference")

    # llm inference
    logger.processing_start("LLM inference")
    if args.is_api:
        assert args.use_vllm is False
        from llms.api_inference import inference as api_inference
        # Fetch labels only from 'y' if available in dataset
        if args.dataset in ['mimic3_mortality', 'mimic4_mortality', 'mimic3_los']:
            labels_list = test_dataset.get('y', None) if isinstance(test_dataset, dict) else None
        elif args.dataset in ['cmb_exam_patient']:
            labels_list = test_dataset.get('answer', None) if isinstance(test_dataset, dict) else None
        else:
            labels_list = None
        responses = api_inference(
            args.llm_name,
            test_dataset['data_prompt_fomat'],
            logger=logger,
            save_path=getattr(args, 'llm_responses_save_path', None),
            labels=labels_list,
        )
        logits = None        
    else:
        if args.use_vllm:
            from llms.vllm_inference import inference as vllm_inference
            # Fetch labels based on dataset type
            if args.dataset in ['mimic3_mortality', 'mimic4_mortality', 'mimic3_los']:
                labels_list = test_dataset.get('y', None) if isinstance(test_dataset, dict) else None
            elif args.dataset in ['cmb_exam_patient']:
                labels_list = test_dataset.get('answer', None) if isinstance(test_dataset, dict) else None
            else:
                labels_list = None

            return_logits = False
            if args.dataset in ['mimic3_los']:
                classification_options = ['A', 'B', 'C', 'D']  # 使用字母选项：A=<3天, B=3-7天, C=7-14天, D=>14天
                return_logits = True
            else:
                classification_options = None         

            responses, logits = vllm_inference(
                args,
                args.llm_local_path,
                test_dataset['data_prompt_fomat'],
                save_path=getattr(args, 'llm_responses_save_path', None),
                labels=labels_list,
                logger=logger,
                max_model_len=getattr(args, 'vllm_max_model_len', 16384),
                gpu_memory_utilization=getattr(args, 'vllm_gpu_memory_utilization', 0.85),
                vllm_batch_size=getattr(args, 'vllm_batch_size', 4),
                return_logits=return_logits,
                classification_options=classification_options,
                enable_thinking=getattr(args, 'vllm_enable_thinking', False),
            )
    logger.processing_complete("LLM inference")

    # print example of LLM response
    logger.show_message_example(responses[0], "Example LLM Response")
    if logits is not None and len(logits) > 0:
        # Convert logits dict to formatted string for display
        import json
        if isinstance(logits[0], dict):
            logits_str = json.dumps(logits[0], indent=2)
        else:
            logits_str = str(logits[0])
        logger.show_message_example(logits_str, "Example LLM Logits")    

    return responses, logits