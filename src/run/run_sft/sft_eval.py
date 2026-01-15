import os
import torch
from args.ehrbase_args import parse_args
from utils.logger import get_logger
from data.prepare_ehr_data import prepare_ehr_data
from prompt.EHR_prompt.prompt_wraper import (
    transform_mimic3_mortality_ehr_to_detail_prompt,
    mimic3_mortality_prompt_wrapper,
)
from utils.llm_eval import llm_response_evaluation


def _ensure_detail(args, dataset):
    detail_all = []
    for i in range(len(dataset['X'])):
        patient_example = {
            'X': dataset['X'][i],
            't': dataset['t'][i],
            'y': dataset['y'][i],
            'header': dataset['header'][i],
            'name': dataset['name'][i],
        }
        detail = transform_mimic3_mortality_ehr_to_detail_prompt(
            patient_example,
            unit=args.unit,
            reference_range=args.reference_range,
            smooth_hourly_data=True,
            keep_last=True,
        )
        detail_all.append(detail)
    dataset['detail'] = detail_all
    return dataset


def _build_prompts(args, test_dataset):
    prompts = []
    for i in range(len(test_dataset['X'])):
        patient_example = {
            k: test_dataset[k][i] for k in test_dataset.keys()
        }
        prompt = mimic3_mortality_prompt_wrapper(
            patient_example,
            is_few_shot=False,
            icl_examples_list=[],
            inference_type=args.inference_type,
            unit=args.unit,
            reference_range=args.reference_range,
            add_smart_logits=False,
            add_smart_logits_for_test_example=False,
        )
        prompts.append(prompt)
    return prompts


def _load_model_and_tokenizer(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_path = args.llm_local_path if args.llm_local_path is not None else args.llm_name
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    adapter_path = args.llm_adapter_path or args.sft_output_dir
    if adapter_path and os.path.isdir(adapter_path):
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        except Exception:
            pass
    return model, tokenizer


def _generate_responses(model, tokenizer, prompts, max_new_tokens=16):
    responses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    for p in prompts:
        inputs = tokenizer(p, return_tensors='pt', truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(text.split(p)[-1].strip() if p in text else text.strip())
    return responses


def run(args, train_dataset, val_dataset, test_dataset, logger):
    test_dataset = _ensure_detail(args, test_dataset)
    prompts = _build_prompts(args, test_dataset)
    model, tokenizer = _load_model_and_tokenizer(args)
    responses = _generate_responses(model, tokenizer, prompts)
    metrics = llm_response_evaluation(args, responses, test_dataset, logger)
    logger.log_metrics(metrics, "SFT LoRA Evaluation Results")


def main():
    args = parse_args()
    logger = get_logger("SFT-Eval", experiment_info={'dataset': args.dataset, 'method': 'llm_sft_eval'})
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
        test_data = {k: [e[k] for e in [make_example('test_0', 1), make_example('test_1', 0)]] for k in ['X','t','y','header','name']}
        args.unit = False
        args.reference_range = False
        test_data = _ensure_detail(args, test_data)
        prompts = _build_prompts(args, test_data)
        responses = ["0.12", "0.85"]
        llm_response_evaluation(args, responses, test_data, logger)
        return
    train_data, val_data, test_data = prepare_ehr_data(args, logger)
    run(args, train_data, val_data, test_data, logger)


if __name__ == "__main__":
    main()