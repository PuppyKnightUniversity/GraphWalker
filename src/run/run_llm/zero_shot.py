#from data.prepare_ehr_data import set_dataset_numerical_ehr
import pickle
from utils.logger import get_logger

def run_zero_shot(args, train_dataset, val_dataset, test_dataset, logger=None):
    '''
    Run the zero-shot method for LLMs.
    Args:
        args: the arguments.
        train_dataset: the train dataset.
        val_dataset: the val dataset.
        test_dataset: the test dataset.
        logger: the logger instance for progress display.
    Returns:
        the dictionary containing various evaluation metrics
    '''
    # 1. wrap prompt
    from prompt.prompt_wraper import train_val_test_dataset_prompt_wrapper
    train_dataset, val_dataset, test_dataset = train_val_test_dataset_prompt_wrapper(args, 
                                                                                     logger,
                                                                                     train_dataset, 
                                                                                     val_dataset, 
                                                                                     test_dataset, 
                                                                                     is_few_shot=False,  # NOTE: for zero-shot method, we do not use few-shot learning
                                                                                     max_tokens=10000)

    # 2. llm inference
    from llms.inference import llm_dataset_inference
    responses = llm_dataset_inference(args, train_dataset, val_dataset, test_dataset, logger=logger)

    # 3. metrics evaluation
    from utils.llm_eval import llm_response_evaluation
    bootstrap_metrics = llm_response_evaluation(args, responses, test_dataset, logger=logger)
    
    return bootstrap_metrics
    