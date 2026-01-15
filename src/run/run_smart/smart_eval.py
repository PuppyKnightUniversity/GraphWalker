import os
import sys
import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Dict, List
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from models.smart import Encoder, Classifier
from utils.utils import set_seed, init_logging
from utils.llm_eval import evaluate_binary_model_with_bootstrap


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


def collate_fn(features: List[Dict[str, Any]]):
    batch = {}
    for key in features[0].keys():
        if key in ["x", "mask", "time"]:
            batch[key] = pad_sequence([torch.tensor(patient[key]) for patient in features], True)
        else:
            batch[key] = torch.tensor([patient[key] for patient in features])
    return batch


def _set_dataset_args(args):
    if args.dataset == 'mimic3_mortality':
        args.smart_input_dim = 17
        args.smart_demo_dim = 0
        args.smart_num_class = 2
        args.smart_max_len = 48
    elif args.dataset == 'mimic4_mortality':
        args.smart_input_dim = 44
        args.smart_demo_dim = 2
        args.smart_num_class = 2
        args.smart_max_len = 8
    elif args.dataset == 'mimic4_readmission':
        args.smart_input_dim = 44
        args.smart_demo_dim = 2
        args.smart_num_class = 2
        args.smart_max_len = 8
    elif args.dataset == 'tjh_mortality':
        args.smart_input_dim = 75
        args.smart_demo_dim = 2
        args.smart_num_class = 2
        args.smart_max_len = args.period_length
    elif args.dataset == 'tjh_los':
        args.smart_input_dim = 75
        args.smart_demo_dim = 2
        args.smart_num_class = 4
        args.smart_max_len = args.period_length
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")


def run_eval(args, test_dataset):
    os.makedirs(args.smart_save_dir, exist_ok=True)
    logger = logging.getLogger()
    init_logging(logger, args.smart_save_dir)
    logger.info(json.dumps(vars(args), indent=4))
    set_seed(args.seed)

    _set_dataset_args(args)

    test_dataset = CustomDataset(test_dataset['data_smart'])
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=args.smart_batch_size, sampler=test_sampler, collate_fn=collate_fn)

    encoder = Encoder(args).cuda()
    classifier = Classifier(args).cuda()

    ck_default = os.path.join(args.smart_save_dir, 'checkpoint-prc.pth')
    ck_fallback = os.path.join(args.smart_save_dir, 'checkpoint-mse.pth')
    ck_path = ck_default if os.path.exists(ck_default) else ck_fallback
    if not os.path.exists(ck_path):
        # also try project-root relative export path
        root_default = os.path.join('export', 'smart', args.dataset, 'checkpoint-prc.pth')
        root_fallback = os.path.join('export', 'smart', args.dataset, 'checkpoint-mse.pth')
        ck_path = root_default if os.path.exists(root_default) else root_fallback
    if not os.path.exists(ck_path):
        raise FileNotFoundError(f'Checkpoint not found in {args.smart_save_dir} or export/smart/{args.dataset}/')
    checkpoint = torch.load(ck_path)

    def _strip_module_prefix(sd):
        return { (k[7:] if k.startswith('module.') else k): v for k, v in sd.items() }

    enc_sd = checkpoint['encoder']
    enc_sd = _strip_module_prefix(enc_sd)
    encoder.load_state_dict(enc_sd, strict=True)
    if 'classifier' in checkpoint:
        cls_sd = checkpoint['classifier']
        cls_sd = _strip_module_prefix(cls_sd)
        classifier.load_state_dict(cls_sd, strict=True)
    encoder.eval()
    classifier.eval()

    score_list = []
    label_list = []
    with torch.no_grad():
        for batch in test_dataloader:
            for key in batch:
                batch[key] = batch[key].cuda()
            h = encoder(**batch)
            logits = classifier(h, **batch)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            score_list.extend(probs.cpu().tolist())
            label_list.extend(batch['labels'].cpu().tolist())

    results = evaluate_binary_model_with_bootstrap(
        score_list=score_list,
        label_list=label_list,
        n_bootstrap=1000,
        confidence_level=0.95,
        random_state=42,
        logger=logger,
    )

    out_path = os.path.join(args.smart_save_dir, 'eval_bootstrap.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved bootstrap evaluation to {out_path}")


if __name__ == "__main__":
    import pickle
    from args.ehrbase_args import parse_args
    args = parse_args()

    def _load_test_dataset(args):
        if args.dataset == 'mimic4_mortality':
            consolidated = os.path.join(args.mid_data_dump_path, 'mimic4_mortality', 'mimic4_mortality_smart.pkl')
            if os.path.exists(consolidated):
                obj = pickle.load(open(consolidated, 'rb'))
                if isinstance(obj, dict):
                    for k in ['test_dataset', 'test', 'data_test', 'test_data']:
                        if k in obj:
                            return obj[k]
                    # if file directly stores the test split
                    if 'data_smart' in obj:
                        return obj
                elif isinstance(obj, list):
                    return {'data_smart': obj}
                raise ValueError(f'Unsupported structure in {consolidated}')
            # fallback to seed-specific test split
            base = os.path.join(args.mid_data_dump_path, 'mimic4_mortality', 'seed' + str(args.seed))
            path = os.path.join(base, 'mimic4_mortality_test.pkl')
            return pickle.load(open(path, 'rb'))
        elif args.dataset == 'mimic3_mortality':
            base = os.path.join(args.mid_data_dump_path, 'mimic3_mortality', 'seed' + str(args.seed))
            path = os.path.join(base, 'mimic3_mortality_test.pkl')
            return pickle.load(open(path, 'rb'))
        elif args.dataset == 'tjh_mortality':
            base = os.path.join(args.mid_data_dump_path, 'tjh_mortality', 'seed' + str(args.seed))
            path = os.path.join(base, 'tjh_mortality_test.pkl')
            return pickle.load(open(path, 'rb'))
        elif args.dataset == 'tjh_los':
            base = os.path.join(args.mid_data_dump_path, 'tjh_los', 'seed' + str(args.seed))
            path = os.path.join(base, 'tjh_los_test.pkl')
            return pickle.load(open(path, 'rb'))
        else:
            raise ValueError(f'Dataset {args.dataset} not supported')

    test_dataset = _load_test_dataset(args)
    run_eval(args, test_dataset)
