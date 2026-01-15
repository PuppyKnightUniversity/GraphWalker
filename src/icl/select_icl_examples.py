from typing import List, Dict, Any

def FIND_ICL_EXAMPLES(args,
                      test_dataset,
                      method,
                      train_dataset,
                      val_dataset=None,
                      num_examples: int = 3,
                      logger=None) -> List[Dict[str, Any]]:
    '''
    Find the few-shot examples in the train dataset
    Args:
        args: the arguments
        test_dataset: the test dataset
        method: the method to find the few-shot examples
        train_dataset: the train dataset
        num_examples: the number of few-shot examples
    Returns:
        the few-shot examples list
    '''
    # select the few-shot examples in the train dataset
    if method != 'graph_walker':
        raise ValueError(f'we have not implemented the ICL method for {method}')
    
    from icl.method.graph_walker import select_graph_walker_examples
    ICL_EXAMPLES_LIST = select_graph_walker_examples(args, test_dataset, train_dataset, num_examples)
    
    return ICL_EXAMPLES_LIST
