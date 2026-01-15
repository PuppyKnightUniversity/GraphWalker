#!/bin/bash
export PYTHONPATH=$PYTHONPATH:./src

# Task 1: mimic3_mortality
echo "Running AdaCare on mimic3_mortality..."
python src/run/run_adacare/adacare_train.py \
    --dataset mimic3_mortality \
    --method ehr_model_adacare \
    --mid_data_dump_path ./mid_data \
    --adacare_save_dir ./export/adacare/ \
    --adacare_epochs 2 \
    --adacare_batch_size 64 \
    --adacare_lr 0.001

# Task 2: mimic3_los
echo "Running AdaCare on mimic3_los..."
python src/run/run_adacare/adacare_train.py \
    --dataset mimic3_los \
    --method ehr_model_adacare \
    --mid_data_dump_path ./mid_data \
    --adacare_save_dir ./export/adacare/ \
    --adacare_epochs 2 \
    --adacare_batch_size 64 \
    --adacare_lr 0.001

# Task 3: mimic4_readmission
echo "Running AdaCare on mimic4_readmission..."
python src/run/run_adacare/adacare_train.py \
    --dataset mimic4_readmission \
    --method ehr_model_adacare \
    --mid_data_dump_path ./mid_data \
    --adacare_save_dir ./export/adacare/ \
    --adacare_epochs 2 \
    --adacare_batch_size 64 \
    --adacare_lr 0.001
