import os
from args.ehrbase_args import parse_args
from utils.logger import get_logger
from data.prepare_ehr_data import prepare_ehr_data
from prompt.EHR_prompt.prompt_wraper import (
    transform_mimic3_mortality_ehr_to_detail_prompt,
    mimic3_mortality_prompt_wrapper,
)
from utils.llm_eval import llm_response_evaluation
from llms.vllm_inference import inference as vllm_generate


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
        patient_example = {k: test_dataset[k][i] for k in test_dataset.keys()}
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


def run(args, train_dataset, val_dataset, test_dataset, logger):
    test_dataset = _ensure_detail(args, test_dataset)
    prompts = _build_prompts(args, test_dataset)
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        def count_tokens(s: str):
            return len(enc.encode(s))
    except Exception:
        def count_tokens(s: str):
            return len(s.split())
    max_allowed = getattr(args, "max_tokens_each_patient", 10000)
    valid_indices = [i for i, p in enumerate(prompts) if count_tokens(p) <= max_allowed]
    if len(valid_indices) < len(prompts):
        logger.info(f"Filtered {len(prompts) - len(valid_indices)} overlength prompts ({len(valid_indices)}/{len(prompts)})")
        prompts = [prompts[i] for i in valid_indices]
        for k in list(test_dataset.keys()):
            if isinstance(test_dataset[k], list):
                test_dataset[k] = [test_dataset[k][i] for i in valid_indices]
    model_path = args.llm_local_path if args.llm_local_path is not None else args.llm_name
    adapter_path = args.llm_adapter_path or args.sft_output_dir
    responses = vllm_generate(
        model_path=model_path,
        prompt_list=prompts,
        adapter_path=adapter_path,
        max_tokens=getattr(args, 'delta_max_tokens', 256),
        temperature=getattr(args, 'delta_temperature', 0.0),
        vllm_batch_size=getattr(args, 'delta_vllm_batch_size', 2),
        save_path=args.llm_responses_save_path,
        labels=[int(y) for y in test_dataset['y']],
        logger=logger,
    )
    metrics = llm_response_evaluation(args, responses, test_dataset, logger)
    logger.log_metrics(metrics, "SFT LoRA vLLM Evaluation Results")


def main():
    args = parse_args()
    logger = get_logger("SFT-Eval-vLLM", experiment_info={'dataset': args.dataset, 'method': 'llm_sft_eval_vllm'})
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
