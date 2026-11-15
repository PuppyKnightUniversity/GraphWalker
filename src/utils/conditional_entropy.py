from __future__ import annotations

from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer
from utils.logger import get_logger


def compute_average_conditional_entropy(
    args,
    train_dataset,
    test_dataset,
    ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS: Optional[List[List[Dict[str, Any]]]] = None,
    logger=None,
    return_metrics=False,
):
    if logger is None:
        logger = get_logger("ConditionalEntropy")
    
    if ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS is None:
        ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS = []
    logger.info("Loading vLLM model for conditional entropy computation...")
    vllm_model = _load_vllm_model(args, logger=logger)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    patient_num = len(test_dataset['detail'])
    conditional_entropies = []
    information_gains = []
    
    logger.info(f"Computing conditional entropy for {patient_num} test patients...")
    progress = logger.create_progress("Computing conditional entropy", patient_num)
    
    try:
        with progress:
            task = progress.add_task("Computing conditional entropy", total=patient_num)

            for i in range(patient_num):
                test_patient_example = {}
                for key in test_dataset.keys():
                    test_patient_example[key] = test_dataset[key][i]

                ICL_EXAMPLES_LIST = ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS[i] if i < len(ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS) else []

                example_node_indices = _convert_icl_examples_to_node_indices(
                    ICL_EXAMPLES_LIST, train_dataset
                )

                ce_loss = _compute_cross_entropy_for_examples(
                    args,
                    test_patient_example,
                    train_dataset,
                    example_node_indices,
                    vllm_model,
                    device,
                    getattr(args, 'graph_walker_parallel_batch_size_for_cal_greedy_score', 2)
                )

                zero_shot_ce = _compute_cross_entropy_for_examples(
                    args, test_patient_example, train_dataset, [], vllm_model, device,
                    getattr(args, 'graph_walker_parallel_batch_size_for_cal_greedy_score', 2),
                ) if example_node_indices else ce_loss
                conditional_entropies.append(ce_loss)
                information_gains.append(zero_shot_ce - ce_loss)
                progress.update(task, advance=1)

    finally:
        del vllm_model
        _release_vllm_model(logger=logger)
    
    avg_conditional_entropy = sum(conditional_entropies) / len(conditional_entropies) if conditional_entropies else float('inf')
    
    logger.success(f"Average conditional entropy on test set: {avg_conditional_entropy:.4f}")
    
    if return_metrics:
        return {
            'avg_conditional_entropy': avg_conditional_entropy,
            'avg_delta_H': sum(information_gains) / len(information_gains) if information_gains else float('nan'),
        }
    return avg_conditional_entropy


def _convert_icl_examples_to_node_indices(
    ICL_EXAMPLES_LIST: List[Dict[str, Any]],
    train_dataset: Dict[str, Any]
) -> List[int]:
    if not ICL_EXAMPLES_LIST:
        return []
    
    node_indices = []
    train_details = train_dataset['detail']
    train_labels = train_dataset['y']
    
    for icl_example in ICL_EXAMPLES_LIST:
        node_index = icl_example.get('node_index')
        if node_index is not None:
            if not 0 <= node_index < len(train_details):
                raise ValueError('ICL node index is outside the training set')
            if train_details[node_index] != icl_example['detail']:
                raise ValueError('ICL node index does not match the training record')
            node_indices.append(node_index)
            continue
        example_detail = icl_example.get('detail', '')
        example_label = icl_example.get('label', icl_example.get('y', None))
        
        found = False
        for idx in range(len(train_details)):
            if train_details[idx] == example_detail:
                if example_label is None or train_labels[idx] == example_label:
                    node_indices.append(idx)
                    found = True
                    break
        
        if not found:
            raise ValueError('Selected example was not found in the training set')
    
    return node_indices


def _load_vllm_model(args, logger=None) -> LLM:
    import os
    from vllm import LLM
    
    gpu_count = torch.cuda.device_count()
    tensor_parallel_size = min(gpu_count, 4)
    
    if logger:
        logger.info(f"Detected {gpu_count} GPUs, using {tensor_parallel_size} for tensor parallelism")
    
    if torch.cuda.is_available():
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / 1024**3
            if logger:
                logger.info(f"GPU {i} ({props.name}): {total_mem:.2f} GB total memory")
    
    model_path = args.llm_local_path
    max_model_len = getattr(args, 'vllm_max_model_len', 16384)
    gpu_memory_utilization = getattr(args, 'vllm_gpu_memory_utilization', 0.85)
    
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    vllm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "swap_space": 4
    }
    
    if logger:
        logger.info(f"Initializing vLLM with kwargs: {vllm_kwargs}")
    
    vllm_model = LLM(**vllm_kwargs)
    
    if logger:
        logger.success("vLLM model initialized successfully")
    
    return vllm_model


def _release_vllm_model(logger=None):
    from icl.method.graph_walker import _release_vllm_model as release
    release(logger=logger)


def _get_tokenizer(model_path: str, device: torch.device) -> AutoTokenizer:
    from icl.method.graph_walker import _metric_tokenizer_cache
    
    if model_path not in _metric_tokenizer_cache:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        _metric_tokenizer_cache[model_path] = tokenizer
    
    return _metric_tokenizer_cache[model_path]


def _build_prompt_from_examples(
    args, test_patient_example, train_dataset, example_node_indices: List[int]
) -> str:
    from icl.method.graph_walker import _build_prompt_from_examples as graph_walker_build_prompt
    return graph_walker_build_prompt(args, test_patient_example, train_dataset, example_node_indices)


def _compute_cross_entropy_for_examples(
    args, test_patient_example, train_dataset, example_node_indices: List[int],
    vllm_model: LLM, device: torch.device, parallel_batch_size_for_cal_greedy_score: int
) -> float:
    from icl.method.graph_walker import _compute_cross_entropy_for_examples as graph_walker_compute_ce
    return graph_walker_compute_ce(
        args, test_patient_example, train_dataset, example_node_indices,
        vllm_model, device, parallel_batch_size_for_cal_greedy_score
    )
