#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
: "${SMART_CHECKPOINT:?Set SMART_CHECKPOINT to the pretrained SMART encoder checkpoint}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" python main.py \
    --dataset "${DATASET:-mimic4_readmission}" \
    --llm_name qwen3-14b-instruct \
    --seed "${SEED:-3407}" \
    --data_split_seed 3407 \
    --method graph_walker \
    --icl_examples_num 3 \
    --max_tokens_each_patient 10000 \
    --use_vllm \
    --graph_walker_parallel_batch_size_for_cal_greedy_score 1 \
    --graph_walker_neighbor_num 8 \
    --vllm_max_model_len 20480 \
    --vllm_gpu_memory_utilization 0.85 \
    --period_length 24 \
    --embedding_model_name smart \
    --embedding_model_path "$SMART_CHECKPOINT" \
    --graph_walker_top_l_cohorts 3 \
    --graph_walker_top_k_per_cohort 3 \
    --graph_walker_leiden_resolution 1.0 \
    --metrics_save_path "../results/${DATASET:-mimic4_readmission}/qwen3-14b/seed-${SEED:-3407}.json" \
    "$@"
