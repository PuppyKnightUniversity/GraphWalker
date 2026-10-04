import math
import os
from typing import List, Dict

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import torch

from args.ehrbase_args import parse_args
from utils.logger import get_logger
from data.prepare_ehr_data import prepare_ehr_data
from prompt.EHR_prompt.common import build_ehr_prompt, format_icl_label
from prompt.EHR_prompt.prompt_wraper import transform_ehr_to_detail_prompt
from llms.prompt_format import format_model_prompt


def _build_zero_shot_prompt(example: Dict[str, any], dataset='mimic3_mortality') -> str:
    return build_ehr_prompt(example, dataset)


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, data_split: Dict[str, List], tokenizer, max_length: int, args=None):
        self.data_split = data_split
        self.tokenizer = tokenizer
        self.max_length = max_length
        from types import SimpleNamespace
        self.args = args or SimpleNamespace(dataset='mimic3_mortality', vllm_apply_chat_template=False)
        self.dataset = self.args.dataset

    def __len__(self):
        return len(self.data_split['X'])

    def __getitem__(self, idx):
        example = {key: values[idx] for key, values in self.data_split.items()}
        prompt_text = _build_zero_shot_prompt(example, self.dataset)
        prompt_text = format_model_prompt(self.args, self.tokenizer, prompt_text,
                                         enable_thinking=False)
        target_text = format_icl_label(self.data_split['y'][idx], self.dataset)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)['input_ids']
        target_ids = self.tokenizer(target_text, add_special_tokens=False)['input_ids']
        if self.tokenizer.eos_token_id is not None:
            target_ids.append(self.tokenizer.eos_token_id)
        if not target_ids or len(prompt_ids) + len(target_ids) > self.max_length:
            raise ValueError('SFT example exceeds max_length or has no answer tokens')
        input_ids = prompt_ids + target_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + target_ids

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


def _ensure_detail_fields(args, logger, train_data, val_data, test_data):
    return transform_ehr_to_detail_prompt(args, train_data, val_data, test_data)


def main():
    args = parse_args()
    logger = get_logger("SFT-Train", experiment_info={'dataset': args.dataset, 'method': 'llm_sft_train'})

    if args.sft_dry_run:
        import numpy as np
        header = ["Hours"] + [f"Feature{i}" for i in range(3)]
        def make_example(name, label):
            X = np.array([[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]])
            t = np.array([0.0, 1.0])
            return {
                'X': X,
                't': t,
                'y': label,
                'header': header,
                'name': name,
            }
        train_data = {k: [e[k] for e in [make_example('train_0', 0), make_example('train_1', 1)]] for k in ['X','t','y','header','name']}
        val_data = {k: [e[k] for e in [make_example('val_0', 0)]] for k in ['X','t','y','header','name']}
        test_data = {k: [e[k] for e in [make_example('test_0', 1)]] for k in ['X','t','y','header','name']}
        args.unit = False
        args.reference_range = False
        train_data, val_data, test_data = _ensure_detail_fields(args, logger, train_data, val_data, test_data)
        logger.info("SFT dry run: synthetic dataset prepared and detail prompts generated. Exiting.")
        return

    train_data, val_data, test_data = prepare_ehr_data(args, logger)
    train_data, val_data, test_data = _ensure_detail_fields(args, logger, train_data, val_data, test_data)

    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        Trainer,
        TrainingArguments,
        BitsAndBytesConfig,
    )

    model_path = args.llm_local_path if args.llm_local_path is not None else args.llm_name
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.sft_max_seq_length

    quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if args.sft_fp16 else None,
        trust_remote_code=True,
    )
    try:
        model.config.use_cache = False
    except Exception:
        pass
    if getattr(args, 'sft_gradient_checkpointing', True):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    if getattr(args, 'sft_use_lora', False):
        from peft import LoraConfig, get_peft_model, TaskType
        raw = getattr(args, 'sft_lora_target_modules', '')
        if not raw or raw.strip().lower() == 'auto':
            target_modules = ['q_proj','k_proj','v_proj','o_proj']
        else:
            target_modules = [m for m in raw.split(',') if m.strip() != '']
        lora_config = LoraConfig(
            r=args.sft_lora_r,
            lora_alpha=args.sft_lora_alpha,
            lora_dropout=args.sft_lora_dropout,
            target_modules=target_modules,
            bias='none',
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    train_dataset = SFTDataset(train_data, tokenizer, max_length=args.sft_max_seq_length, args=args)
    eval_dataset = SFTDataset(val_data, tokenizer, max_length=args.sft_max_seq_length, args=args)

    os.makedirs(args.sft_output_dir, exist_ok=True)

    total_steps = math.ceil(len(train_dataset) / (args.sft_batch_size * max(1, torch.cuda.device_count())))

    # Build TrainingArguments with a conservative set of kwargs for broad compatibility
    # build optional deepspeed zero-3 config
    ds_config = None
    if getattr(args, 'sft_deepspeed_zero3', True):
        ds_config = {
            "fp16": {"enabled": bool(getattr(args, 'sft_fp16', True))},
            "zero_optimization": {
                "stage": 3,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e8,
                "stage3_prefetch_bucket_size": 5e8,
                "stage3_param_persistence_threshold": 1e5,
            },
            "train_micro_batch_size_per_gpu": getattr(args, 'sft_batch_size', 1),
            "gradient_accumulation_steps": getattr(args, 'sft_gradient_accumulation', 1),
        }
        if getattr(args, 'sft_zero3_cpu_offload', False):
            ds_config["zero_optimization"]["offload_param"] = {"device": "cpu", "pin_memory": True}
            ds_config["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}

    training_args = TrainingArguments(
        output_dir=args.sft_output_dir,
        num_train_epochs=args.sft_epochs,
        per_device_train_batch_size=args.sft_batch_size,
        gradient_accumulation_steps=args.sft_gradient_accumulation,
        learning_rate=args.sft_lr,
        warmup_ratio=args.sft_warmup_ratio,
        logging_steps=max(1, total_steps // 50),
        fp16=args.sft_fp16,
        gradient_checkpointing=getattr(args, 'sft_gradient_checkpointing', True),
        ddp_find_unused_parameters=False,
        report_to=["none"],
        deepspeed=ds_config,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    logger.info("Start SFT training with HuggingFace Trainer...")
    trainer.train()
    logger.info("SFT training finished, saving model...")
    trainer.save_model(args.sft_output_dir)
    try:
        model.save_pretrained(args.sft_output_dir)
    except Exception:
        pass


if __name__ == "__main__":
    main()
