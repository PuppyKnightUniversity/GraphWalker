# flake8: noqa
"""
Define arguments for scripts
"""

import argparse
from args.local_path_config import DATASET_PATH_MAP, LLM_PATH_MAP, EMBEDDING_MODEL_PATH_MAP

def parse_args():
    parser = argparse.ArgumentParser()

    '''
        Seed Settings
    '''
    parser.add_argument("--seed", type=int, default=3407, help="Seed Settings")

    '''
        Dataset Settings
    '''
    parser.add_argument("--dataset", type=str, default="mimic3_mortality", choices=['mimic3_mortality',
                                                                                    'mimic3_los',
                                                                                    'mimic4_mortality',
                                                                                    'mimic4_los',
                                                                                    'mimic4_readmission',
                                                                                    'tjh_mortality',
                                                                                    'tjh_los',
                                                                                    'cmb_exam_patient',
                                                                                    'cmb_clin',
                                                                                    'medqa'], help="Dataset to use")
    parser.add_argument("--dataset_path", type=str, default=None, help="Dataset path")
    parser.add_argument("--period_length", type=int, default=48, help="Period length for the data")
    parser.add_argument("--channel_info_path", type=str, default='../reference_code/SMART/data/resources/channel_info.json', help="Channel info path")
    parser.add_argument("--discretizer_config_path", type=str, default='../reference_code/SMART/data/resources/discretizer_config.json', help="Discretizer config path")
    parser.add_argument("--mid_data_dump_path", type=str, default='../mid_data', help="Dump path for the processed mid data")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train ratio for the data")
    parser.add_argument("--unit", action="store_true", default=True, help="Whether to include unit information in the prompt")
    parser.add_argument("--reference_range", action="store_true", default=True, help="Whether to include reference range information in the prompt")
    '''
        Method Settings
    '''
    parser.add_argument("--method", type=str, default="llm_zero_shot", choices=["llm_zero_shot",
                                                                                "graph_walker",
                                                                                "ehr_model_smart",
                                                                                "llm_sft_train",
                                                                                "llm_sft_eval",
                                                                                "llm_sft_eval_vllm",
                                                                                "llm_ids_iterative"])
    
    parser.add_argument("--max_tokens_each_patient", type=int, default=10000, help="Maximum tokens for each patient's prompt")
    parser.add_argument("--icl_examples_num", type=int, default=3, help="Number of few-shot examples to use")
    
    parser.add_argument("--inference_type", type=str, default="only_answer", choices=["only_answer"], help="Inference type, only suitable for llm.")
    parser.add_argument("--force_recompute_prompt_wrapper", action="store_true", default=False, help="Force re-wrapping prompts and ignore cached mid_data")
    '''
        LLM Settings
    '''
    parser.add_argument("--llm_name", type=str, default="qwen2-5-7b-instruct", choices=["qwen2-5-7b-instruct",
                                                                                        "qwen2-5-14b-instruct", 
                                                                                        "qwen2-5-72b-instruct", 
                                                                                        "qwen3-14b-instruct",
                                                                                        "qwen3-32b-instruct",
                                                                                        "llama-3.1-8b-instruct",
                                                                                        "ministral-3-14b-instruct",
                                                                                        'deepseek-r1-8b',
                                                                                        "deepseek-r1",
                                                                                        "xiaobei-32b",
                                                                                        "gpt-3.5-turbo",
                                                                                        "gpt-5"], help="LLM to use")
    parser.add_argument("--llm_local_path", type=str, default=None, help="LLM local path")
    parser.add_argument("--llm_adapter_path", type=str, default=None, help="LLM adapter path")
    parser.add_argument("--llm_responses_save_path", type=str, default=None, help="Directory to save individual LLM responses for checkpoint support")
    parser.add_argument("--is_api", action="store_true", default=False, help="Whether to use API to inference")

    '''
        VLLM Settings
    '''
    parser.add_argument("--use_vllm", action="store_true", default=False, help="Whether to use vllm to accelerate the inference")
    parser.add_argument("--vllm_max_model_len", type=int, default=16384, help="Maximum sequence length for vLLM (default: 16384). Can be increased if GPU memory allows (e.g., 32768, 65536)")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85, help="GPU memory utilization ratio for vLLM (default: 0.85). Lower this if you increase max_model_len and encounter OOM")
    parser.add_argument("--vllm_batch_size", type=int, default=4, help="Batch size for vLLM (default: 4)")
    parser.add_argument("--vllm_enable_thinking", action="store_true", default=False, help="Enable thinking mode for qwen3 models (default: False). When enabled, the model will generate thinking tokens before the final answer")
    parser.add_argument("--vllm_apply_chat_template", action="store_true", default=True, help="Apply chat template to the prompt, if True, the prompt will be converted to a chat template format, otherwise, the prompt will be used as is")
    '''
        SFT Training Settings
    '''
    parser.add_argument("--sft_max_seq_length", type=int, default=4096, help="Max sequence length for SFT")
    parser.add_argument("--sft_epochs", type=int, default=1, help="Training epochs for SFT")
    parser.add_argument("--sft_lr", type=float, default=1e-4, help="Learning rate for SFT")
    parser.add_argument("--sft_batch_size", type=int, default=1, help="Per-device train batch size for SFT")
    parser.add_argument("--sft_gradient_accumulation", type=int, default=8, help="Gradient accumulation steps for SFT")
    parser.add_argument("--sft_warmup_ratio", type=float, default=0.03, help="Warmup ratio for SFT")
    parser.add_argument("--sft_fp16", action="store_true", default=True, help="Use fp16 training for SFT")
    parser.add_argument("--sft_output_dir", type=str, default="../export/sft/", help="Output directory for SFT checkpoints")
    parser.add_argument("--sft_dry_run", action="store_true", default=False, help="Only build dataset and exit for SFT")
    parser.add_argument("--sft_use_lora", action="store_true", default=False, help="Use LoRA for SFT")
    parser.add_argument("--sft_lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--sft_lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--sft_lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--sft_lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", help="Comma-separated target modules for LoRA")
    parser.add_argument("--sft_gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing for SFT")
    parser.add_argument("--sft_deepspeed_zero3", action="store_true", default=True, help="Enable DeepSpeed ZeRO-3 for SFT")
    parser.add_argument("--sft_zero3_cpu_offload", action="store_true", default=False, help="Enable CPU offload for ZeRO-3")
    '''
        Embedding Model Settings
    '''
    parser.add_argument("--embedding_model_name", type=str, default=None, choices=["smart", "qwen3-embedding-8b"], help="Embedding model to use, if None, the embedding model will not be used")
    parser.add_argument("--embedding_model_train_from_scratch", action="store_true", default=False, help="Whether to train the embedding model from scratch, if True, the embedding model will be trained from scratch, otherwise, the embedding model will be loaded from the checkpoint")
    parser.add_argument("--embedding_model_path", type=str, default=None, help="Embedding model path")
    '''
        Smart Model Settings
    '''
    parser.add_argument('--smart_data_dropout', type=float, default=0.1)
    parser.add_argument('--smart_epochs', type=int, default=25)
    parser.add_argument('--smart_lr', type=float, default=1e-3)
    parser.add_argument('--smart_d_model', type=int, default=32)
    parser.add_argument('--smart_seed', type=int, default=3407) 
    parser.add_argument('--smart_batch_size', type=int, default=64)
    parser.add_argument('--smart_dropout', type=float, default=0.1)
    parser.add_argument('--smart_save_model', type=bool, default=True)
    parser.add_argument('--smart_local-rank', type=int, default=1)
    parser.add_argument('--smart_min_mask_ratio', type=float, default=0.)
    parser.add_argument('--smart_max_mask_ratio', type=float, default=0.75)
    parser.add_argument('--smart_e_layers', type=int, default=2)
    parser.add_argument('--smart_n_heads', type=int, default=4)
    parser.add_argument('--smart_input_dim', type=int, default=17)
    parser.add_argument('--smart_demo_dim', type=int, default=0)
    parser.add_argument('--smart_num_class', type=int, default=2)
    parser.add_argument('--smart_max_len', type=int, default=48)
    parser.add_argument('--smart_freeze_epochs', type=int, default=5)
    parser.add_argument("--smart_save_dir", type=str, default="../export/smart/", help="Save path for the checkpoints")

    '''
        Toy Dataset Settings
    '''
    parser.add_argument('--toy_dataset', action="store_true", default=False, help="Whether to use toy dataset, if True, the dataset will be toy dataset, otherwise, the dataset will be the original dataset")
    parser.add_argument('--toy_dataset_size_train', type=int, default=1600, help="Toy dataset size for train")
    parser.add_argument('--toy_dataset_size_val', type=int, default=200, help="Toy dataset size for val")
    parser.add_argument('--toy_dataset_size_test', type=int, default=200, help="Toy dataset size for test")
    parser.add_argument('--dist_main_port', type=int, default=None, help="Distributed main process port for accelerate/deepspeed to avoid conflicts")

    '''
        ICL method Settings
    '''
    # For method graph_walker
    parser.add_argument('--graph_walker_neighbor_num', type=int, default=3, help="Graph walker neighbor number, when building the graph, each node will be connected to the top-k most similar nodes")
    parser.add_argument('--graph_walker_parallel_batch_size_for_cal_greedy_score', type=int, default=2, help="Parallel batch size for calculating greedy score")
    parser.add_argument('--graph_walker_test_topk', type=int, default=3, help="Graph walker test top-k, when we add the test patient to the graph, we will connect it to the top-k most similar nodes")
    
    parser.add_argument('--graph_walker_n_clusters', type=int, default=10, help="Graph walker n clusters, when we use K-means to find patient cohorts, the number of clusters")
    parser.add_argument('--graph_walker_top_l_cohorts', type=int, default=3, help="Graph walker top-l cohorts, when we find the top-l candidate cohorts for the test patient, the number of cohorts")
    parser.add_argument('--graph_walker_top_k_per_cohort', type=int, default=2, help="Graph walker top-k per cohort, when we build the initial frontiers, the number of patients to select from each cohort")
    
    parser.add_argument('--graph_walker_leiden_resolution', type=float, default=0.9, help="Graph walker Leiden resolution parameter (higher = more clusters, lower = fewer clusters)")
    parser.add_argument('--graph_walker_mode', type=str, default='frontiers-lazy-greedy', choices=['random', 'frontiers-lazy-greedy'], help="Graph walker mode, when we select the patients from the cohorts, we can used different strategies to select the patients")
    parser.add_argument('--graph_walker_add_smart_logits', action="store_true", default=False, help="Whether to add SMART model logits to the graph walker examples, if True, the SMART model logits will be added to the graph walker examples, otherwise, the SMART model logits will not be added to the graph walker examples")
    parser.add_argument('--graph_walker_add_smart_logits_for_test_example', action="store_true", default=False, help="Whether to add SMART model logits to the test example, if True, the SMART model logits will be added to the test example, otherwise, the SMART model logits will not be added to the test example")
    
    # For conditional entropy computation
    parser.add_argument('--final_delta_H', action="store_true", default=False, help="Whether to compute average conditional entropy on test set")
    
    # Check if we're in a Jupyter notebook environment
    try:
        # Try to get the current module's name
        import sys
        if 'ipykernel' in sys.modules or 'IPython' in sys.modules:
            # We're in a Jupyter notebook, use default values
            args = parser.parse_args([])
        else:
            # We're in a regular script, parse command line arguments
            args = parser.parse_args()
    except:
        # Fallback: use default values
        args = parser.parse_args([])
    
    if args.dataset_path is None:
        dataset_key = args.dataset
        if dataset_key in DATASET_PATH_MAP:
            args.dataset_path = DATASET_PATH_MAP[dataset_key]
        else:
            raise ValueError(f"No dataset path mapping found for {dataset_key}")
            
    if args.llm_local_path is None:
        llm_key = args.llm_name
        if llm_key in LLM_PATH_MAP:
            args.llm_local_path = LLM_PATH_MAP[llm_key]
        else:
            raise ValueError(f"No LLM local path mapping found for {llm_key}")
    
    if args.embedding_model_path is None and args.embedding_model_name != "smart" and args.embedding_model_name is not None:
        embedding_model_key = args.embedding_model_name
        if embedding_model_key in EMBEDDING_MODEL_PATH_MAP:
            args.embedding_model_path = EMBEDDING_MODEL_PATH_MAP[embedding_model_key]
        else:
            raise ValueError(f"No embedding model path mapping found for {embedding_model_key}")
    
    # Assertion: if ICL examples don't have SMART logits, test example shouldn't have them either
    if not args.graph_walker_add_smart_logits and args.graph_walker_add_smart_logits_for_test_example:
        raise ValueError("Cannot add SMART logits to test example when ICL examples don't have SMART logits. Set --graph_walker_add_smart_logits to True if you want to add SMART logits to test example.")
    
    return args