"""
计算test集合上平均条件熵的函数
根据当前选择的ICL样例，计算对于test的条件熵
计算方法和graph_walker保持一致，复用vllm加速的代码
"""
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from utils.logger import get_logger


def compute_average_conditional_entropy(
    args,
    train_dataset,
    test_dataset,
    ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS: Optional[List[List[Dict[str, Any]]]] = None,
    logger=None
) -> float:
    """
    计算test集合上平均条件熵
    
    Args:
        args: 参数对象
        train_dataset: 训练数据集
        test_dataset: 测试数据集
        ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS: ICL样例列表，如果为None，则从prompt中提取
        logger: 日志记录器
    
    Returns:
        平均条件熵（float）
    """
    if logger is None:
        logger = get_logger("ConditionalEntropy")
    
    # 如果ICL样例列表为空，直接返回错误
    if ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS is None:
        ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS = []
    # 加载vLLM模型
    logger.info("Loading vLLM model for conditional entropy computation...")
    vllm_model = _load_vllm_model(args, logger=logger)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 获取tokenizer
    model_path = args.llm_local_path
    tokenizer = _get_tokenizer(model_path, device)
    
    # 计算每个test样本的条件熵
    patient_num = len(test_dataset['detail'])
    conditional_entropies = []
    
    logger.info(f"Computing conditional entropy for {patient_num} test patients...")
    progress = logger.create_progress("Computing conditional entropy", patient_num)
    
    with progress:
        task = progress.add_task("Computing conditional entropy", total=patient_num)
        
        for i in range(patient_num):
            # 构建test patient example
            test_patient_example = {}
            for key in test_dataset.keys():
                test_patient_example[key] = test_dataset[key][i]
            
            # 获取当前test patient的ICL样例
            ICL_EXAMPLES_LIST = ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS[i] if i < len(ICL_EXAMPLES_LIST_FOR_ALL_TEST_PATIENTS) else []
            
            # 将ICL样例转换为node indices（如果是Dict格式，需要找到对应的索引）
            example_node_indices = _convert_icl_examples_to_node_indices(
                ICL_EXAMPLES_LIST, train_dataset
            )
            
            # 计算条件熵
            ce_loss = _compute_cross_entropy_for_examples(
                args,
                test_patient_example,
                train_dataset,
                example_node_indices,
                vllm_model,
                device,
                getattr(args, 'graph_walker_parallel_batch_size_for_cal_greedy_score', 2)
            )
            
            conditional_entropies.append(ce_loss)
            progress.update(task, advance=1)
    
    # 释放vLLM模型
    _release_vllm_model(vllm_model, logger=logger)
    
    # 计算平均条件熵
    avg_conditional_entropy = sum(conditional_entropies) / len(conditional_entropies) if conditional_entropies else float('inf')
    
    logger.success(f"Average conditional entropy on test set: {avg_conditional_entropy:.4f}")
    
    return avg_conditional_entropy


def _convert_icl_examples_to_node_indices(
    ICL_EXAMPLES_LIST: List[Dict[str, Any]],
    train_dataset: Dict[str, Any]
) -> List[int]:
    """
    将ICL样例列表转换为train_dataset中的节点索引
    
    Args:
        ICL_EXAMPLES_LIST: ICL样例列表，每个元素是一个字典，包含'detail'和'y'等字段
        train_dataset: 训练数据集
    
    Returns:
        节点索引列表
    """
    if not ICL_EXAMPLES_LIST:
        return []
    
    node_indices = []
    train_details = train_dataset['detail']
    train_labels = train_dataset['y']
    
    for icl_example in ICL_EXAMPLES_LIST:
        # 尝试通过detail和label匹配找到对应的索引
        # ICL样例可能使用'label'或'y'字段
        example_detail = icl_example.get('detail', '')
        example_label = icl_example.get('label', icl_example.get('y', None))
        
        # 在train_dataset中查找匹配的样本
        found = False
        for idx in range(len(train_details)):
            if train_details[idx] == example_detail:
                if example_label is None or train_labels[idx] == example_label:
                    node_indices.append(idx)
                    found = True
                    break
        
        if not found:
            # 如果找不到完全匹配的，尝试通过其他方式匹配
            # 这里可以根据实际情况调整匹配策略
            # 注意：这里不记录警告，因为可能ICL样例格式不同
            pass
    
    return node_indices


def _load_vllm_model(args, logger=None) -> LLM:
    """
    加载vLLM模型
    
    Args:
        args: 参数对象
        logger: 日志记录器
    
    Returns:
        vLLM模型实例
    """
    import os
    
    # 检测GPU数量
    gpu_count = torch.cuda.device_count()
    tensor_parallel_size = min(gpu_count, 4)  # 最多使用4个GPU
    
    if logger:
        logger.info(f"Detected {gpu_count} GPUs, using {tensor_parallel_size} for tensor parallelism")
    
    # 检查GPU内存
    if torch.cuda.is_available():
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / 1024**3  # GB
            if logger:
                logger.info(f"GPU {i} ({props.name}): {total_mem:.2f} GB total memory")
    
    # 检查模型路径是否存在
    model_path = args.llm_local_path
    max_model_len = getattr(args, 'vllm_max_model_len', 16384)
    gpu_memory_utilization = getattr(args, 'vllm_gpu_memory_utilization', 0.85)
    
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    # 初始化vLLM参数
    vllm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "swap_space": 4
    }
    
    # 初始化vLLM模型
    if logger:
        logger.info(f"Initializing vLLM with kwargs: {vllm_kwargs}")
    
    vllm_model = LLM(**vllm_kwargs)
    
    if logger:
        logger.success("vLLM model initialized successfully")
    
    return vllm_model


def _release_vllm_model(vllm_model: LLM, logger=None):
    """
    释放vLLM模型内存
    
    Args:
        vllm_model: vLLM模型实例
        logger: 日志记录器
    """
    if vllm_model is None:
        return
    
    try:
        if logger:
            logger.info("Releasing vLLM model memory...")
        
        # Delete the model object
        # vLLM will handle internal resource cleanup when the object is deleted
        del vllm_model
        
        # Force garbage collection to ensure Python objects are freed
        import gc
        gc.collect()
        
        # Clear GPU cache to free up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        if logger:
            logger.success("vLLM model memory released successfully")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to release vLLM model memory: {e}")
        else:
            print(f"Warning: Failed to release vLLM model memory: {e}")


def _get_tokenizer(model_path: str, device: torch.device) -> AutoTokenizer:
    """
    获取tokenizer（带缓存）
    
    Args:
        model_path: 模型路径
        device: 设备
    
    Returns:
        Tokenizer实例
    """
    # 使用全局缓存
    from icl.method.graph_walker_v7 import _metric_tokenizer_cache
    
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
    """
    从ICL样例和test patient构建prompt
    复用graph_walker中的实现
    
    Args:
        args: 参数对象
        test_patient_example: test patient样例
        train_dataset: 训练数据集
        example_node_indices: ICL样例的节点索引列表
    
    Returns:
        Prompt字符串
    """
    # 复用graph_walker中的实现
    from icl.method.graph_walker_v7 import _build_prompt_from_examples as graph_walker_build_prompt
    return graph_walker_build_prompt(args, test_patient_example, train_dataset, example_node_indices)


def _compute_cross_entropy_for_examples(
    args, test_patient_example, train_dataset, example_node_indices: List[int],
    vllm_model: LLM, device: torch.device, parallel_batch_size_for_cal_greedy_score: int
) -> float:
    """
    计算test patient在给定ICL样例下的交叉熵（条件熵）
    复用graph_walker中的实现
    
    Args:
        args: 参数对象
        test_patient_example: test patient样例
        train_dataset: 训练数据集
        example_node_indices: ICL样例的节点索引列表
        vllm_model: vLLM模型实例
        device: 设备
        parallel_batch_size_for_cal_greedy_score: 批处理大小
    
    Returns:
        交叉熵（条件熵）值
    """
    # 复用graph_walker中的实现
    from icl.method.graph_walker_v7 import _compute_cross_entropy_for_examples as graph_walker_compute_ce
    return graph_walker_compute_ce(
        args, test_patient_example, train_dataset, example_node_indices,
        vllm_model, device, parallel_batch_size_for_cal_greedy_score
    )

