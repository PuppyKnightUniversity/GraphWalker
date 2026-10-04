from args.ehrbase_args import parse_args
from utils.logger import get_logger
from data.prepare_ehr_data import prepare_ehr_data
from prompt.EHR_prompt.common import serialize_patient_record, build_ehr_prompt
from prompt.EHR_prompt.prompt_wraper import validate_detail_prompt_lengths
from utils.llm_eval import llm_response_evaluation
from llms.vllm_inference import inference as vllm_generate


def _ensure_detail(args, dataset):
    dataset['detail'] = [
        serialize_patient_record({k: values[i] for k, values in dataset.items()},
                                 args.dataset, unit=args.unit, reference_range=args.reference_range)
        for i in range(len(dataset['X']))]
    return dataset


def _build_prompts(args, test_dataset):
    return [build_ehr_prompt({k: values[i] for k, values in test_dataset.items()},
                             args.dataset, inference_type=args.inference_type)
            for i in range(len(test_dataset['X']))]


def run(args, train_dataset, val_dataset, test_dataset, logger):
    test_dataset = _ensure_detail(args, test_dataset)
    validate_detail_prompt_lengths(logger, test_dataset,
                                   max_tokens=getattr(args, 'max_tokens_each_patient', 10000))
    prompts = _build_prompts(args, test_dataset)
    model_path = args.llm_local_path if args.llm_local_path is not None else args.llm_name
    adapter_path = args.llm_adapter_path or args.sft_output_dir
    is_los = args.dataset.endswith('_los')
    responses, scores = vllm_generate(
        args=args,
        model_path=model_path,
        prompt_list=prompts,
        adapter_path=adapter_path,
        max_tokens=256,
        temperature=0.0,
        max_model_len=getattr(args, 'vllm_max_model_len', 16384),
        gpu_memory_utilization=getattr(args, 'vllm_gpu_memory_utilization', 0.85),
        vllm_batch_size=getattr(args, 'vllm_batch_size', 4),
        save_path=args.llm_responses_save_path,
        labels=[int(y) for y in test_dataset['y']],
        return_logits=is_los,
        classification_options=list('ABCD') if is_los else None,
        enable_thinking=False,
        logger=logger,
    )
    metrics = llm_response_evaluation(args, responses, scores, test_dataset, logger)
    logger.log_metrics(metrics, "SFT LoRA vLLM Evaluation Results")
    return metrics


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
        llm_response_evaluation(args, responses, None, test_data, logger)
        return
    train_data, val_data, test_data = prepare_ehr_data(args, logger)
    run(args, train_data, val_data, test_data, logger)


if __name__ == "__main__":
    main()
