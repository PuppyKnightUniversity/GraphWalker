import os
import torch
from args.ehrbase_args import parse_args
from utils.logger import get_logger
from data.prepare_ehr_data import prepare_ehr_data
from prompt.EHR_prompt.common import serialize_patient_record, build_ehr_prompt
from llms.prompt_format import format_model_prompt
from prompt.EHR_prompt.prompt_wraper import validate_detail_prompt_lengths
from utils.llm_eval import llm_response_evaluation


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


def _load_model_and_tokenizer(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_path = args.llm_local_path if args.llm_local_path is not None else args.llm_name
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    adapter_path = args.llm_adapter_path or args.sft_output_dir
    if adapter_path and os.path.isdir(adapter_path):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    elif adapter_path:
        raise FileNotFoundError(f'SFT adapter not found: {adapter_path}')
    return model, tokenizer


def _generate_responses(model, tokenizer, prompts, max_new_tokens=16):
    responses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    for p in prompts:
        inputs = tokenizer(p, return_tensors='pt', add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
        answer = outputs[0, inputs['input_ids'].shape[1]:]
        responses.append(tokenizer.decode(answer, skip_special_tokens=True).strip())
    return responses


def _score_los_options(model, tokenizer, prompts):
    """Normalize summed answer-token log-likelihoods over the four LOS labels."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    probabilities = []
    with torch.no_grad():
        for prompt in prompts:
            prefix = tokenizer.encode(prompt, add_special_tokens=False)
            if not prefix:
                raise ValueError('LOS prompts must be nonempty')
            scores = []
            for option in 'ABCD':
                answer = tokenizer.encode(option, add_special_tokens=False)
                if not answer:
                    raise ValueError('LOS answer labels must tokenize to nonempty sequences')
                ids = torch.tensor([prefix + answer], dtype=torch.long, device=device)
                logits = model(input_ids=ids).logits[0, len(prefix) - 1:-1].float()
                logprobs = torch.log_softmax(logits, dim=-1)
                observed = ids[0, len(prefix):]
                scores.append(logprobs.gather(1, observed[:, None]).sum())
            scores = torch.stack(scores)
            if not torch.isfinite(scores).all():
                raise ValueError('Nonfinite LOS option likelihoods')
            values = torch.softmax(scores, dim=0).cpu().tolist()
            probabilities.append(dict(zip('ABCD', values)))
    return probabilities


def run(args, train_dataset, val_dataset, test_dataset, logger):
    test_dataset = _ensure_detail(args, test_dataset)
    validate_detail_prompt_lengths(logger, test_dataset,
                                   max_tokens=getattr(args, 'max_tokens_each_patient', 10000))
    prompts = _build_prompts(args, test_dataset)
    model, tokenizer = _load_model_and_tokenizer(args)
    prompts = [format_model_prompt(args, tokenizer, p) for p in prompts]
    if args.dataset.endswith('_los'):
        scores = _score_los_options(model, tokenizer, prompts)
        responses = [max(row, key=row.get) for row in scores]
    else:
        scores = None
        responses = _generate_responses(model, tokenizer, prompts)
    metrics = llm_response_evaluation(args, responses, scores, test_dataset, logger)
    logger.log_metrics(metrics, "SFT LoRA Evaluation Results")
    return metrics


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
        llm_response_evaluation(args, responses, None, test_data, logger)
        return
    train_data, val_data, test_data = prepare_ehr_data(args, logger)
    run(args, train_data, val_data, test_data, logger)


if __name__ == "__main__":
    main()
