CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --dataset mimic3_los \
    --llm_name qwen3-14b-instruct \
    --seed 3407 \
    --method graph_walker \
    --icl_examples_num 3 \
    --max_tokens_each_patient 10000 \
    --use_vllm \
    --llm_responses_save_path ./llm_responses/graph_walker_qwen3-14b-instruct-mimic3_los-period_length_24-graph_walker_neighbor_num_8-top_l_cohorts_2-top_k_per_cohort_3-n_clusters_10-run1_add_smart_logits-add_smart_logits_for_test_example_wo_cohorts_1_wo_greedy_1 \
    --toy_dataset \
    --toy_dataset_size_test 200 \
    --graph_walker_parallel_batch_size_for_cal_greedy_score 4 \
    --graph_walker_neighbor_num 8 \
    --vllm_max_model_len 20480 \
    --vllm_gpu_memory_utilization 0.65 \
    --vllm_batch_size 8 \
    --period_length 24 \
    --embedding_model_name smart \
    --graph_walker_top_l_cohorts 2 \
    --graph_walker_top_k_per_cohort 3 \
    --graph_walker_n_clusters 10 \
    --graph_walker_add_smart_logits \
#    --graph_walker_add_smart_logits_for_test_example \

