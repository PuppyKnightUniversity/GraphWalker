'''
    Calculate the smart embedding for the train, val, and test dataset
'''
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
import torch
import random
from typing import List, Dict, Any
import os
from models.smart import Encoder, Classifier

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

    def dropout_data(self, drop_rate=0.1):
        for i in range(len(self.data)):
            for j in range(len(self.data[i]['x'])):
                for k in range(len(self.data[i]['x'][j])):
                    if self.data[i]['mask'][j][k] == 1 and random.random() < drop_rate:
                        self.data[i]['x'][j][k] = 0
                        self.data[i]['mask'][j][k] == 0


def collate_fn(features: List[Dict[str, Any]]):
    batch = {}
    for key in features[0].keys():
        if key in ["x", "mask", "time"]:
            batch[key] = pad_sequence([torch.tensor(patient[key]) for patient in features], True)
        else:
            batch[key] = torch.tensor([patient[key] for patient in features])
    return batch

def remove_module_prefix(state_dict):
    """
    Remove 'module.' prefix from state_dict keys if present.
    This is needed when loading models saved with DistributedDataParallel.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix (7 characters)
        else:
            new_state_dict[k] = v
    return new_state_dict

def calculate_smart_embedding(args, train_dataset, val_dataset, test_dataset):
    """
    Calculate SMART embeddings for train / val / test splits.
    
    NOTE:
        We must ensure that SMART model hyper-parameters (e.g. smart_input_dim)
        are exactly the same as those used during SMART training / finetuning,
        otherwise loading the checkpoint will raise size mismatch errors.
    """
    # Make sure SMART-related args are consistent with training configuration
    # Keep this logic in sync with `run_smart/smart_pretrain.py` and `smart_finetune.py`.
    if args.dataset == 'mimic3_mortality':
        args.smart_input_dim = 17
        args.smart_demo_dim = 0
        args.smart_num_class = 2
        args.smart_max_len = args.period_length
    elif args.dataset == 'mimic4_los':
        args.smart_input_dim = 44
        args.smart_demo_dim = 2
        args.smart_num_class = 4
        args.smart_max_len = args.period_length
    elif args.dataset == 'mimic4_mortality':
        args.smart_input_dim = 44
        args.smart_demo_dim = 2
        args.smart_num_class = 2
        args.smart_max_len = args.period_length
    elif args.dataset == 'mimic4_readmission':
        args.smart_input_dim = 44
        args.smart_demo_dim = 2
        args.smart_num_class = 2
        args.smart_max_len = args.period_length
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
    elif args.dataset == 'mimic3_los':
        args.smart_input_dim = 17
        args.smart_demo_dim = 0
        args.smart_num_class = 4
        args.smart_max_len = args.period_length
        if not hasattr(args, 'smart_max_len') or args.smart_max_len is None:
            args.smart_max_len = args.period_length
    else:
        raise ValueError(f"Dataset {args.dataset} not supported for SMART embedding calculation")        

    # load smart checkpoint
    checkpoint_path = os.path.join(args.smart_save_dir, 'checkpoint-prc.pth')
    if not os.path.exists(checkpoint_path):
        # fall back to project-root relative export path, similar to smart_eval.py
        root_default = os.path.join('export', 'smart', args.dataset, 'checkpoint-prc.pth')
        root_fallback = os.path.join('export', 'smart', args.dataset, 'checkpoint-mse.pth')
        checkpoint_path = root_default if os.path.exists(root_default) else root_fallback
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"SMART checkpoint not found in {args.smart_save_dir} or export/smart/{args.dataset}/. "
            f"You need to run SMART training/finetuning first to generate 'checkpoint-prc.pth'. ")

    checkpoint = torch.load(checkpoint_path)
    encoder = Encoder(args).cuda()
    classifier = Classifier(args).cuda()

    # Remove 'module.' prefix if present (from DistributedDataParallel)
    encoder_state_dict = remove_module_prefix(checkpoint['encoder'])
    classifier_state_dict = remove_module_prefix(checkpoint['classifier'])

    # If positional encoding table shape doesn't match (e.g. different smart_max_len),
    # drop it from checkpoint so we can use the current model's pos_table instead.
    pos_key = 'position_enc.pos_table'
    if pos_key in encoder_state_dict:
        try:
            model_pos_shape = encoder.position_enc.pos_table.shape
            ckpt_pos_shape = encoder_state_dict[pos_key].shape
            if model_pos_shape != ckpt_pos_shape:
                encoder_state_dict.pop(pos_key)
        except Exception:
            # If anything goes wrong, be conservative and still try to load without pos_table
            encoder_state_dict.pop(pos_key, None)

    # Load with strict=False so missing pos_table (or other intentionally skipped buffers)
    # won't cause a runtime error.
    encoder.load_state_dict(encoder_state_dict, strict=False)
    classifier.load_state_dict(classifier_state_dict)
    encoder.eval()
    classifier.eval()
    
    # dataset, sampler, dataloader
    custom_train_dataset = CustomDataset(train_dataset['data_smart'])
    custom_val_dataset = CustomDataset(val_dataset['data_smart'])
    custom_test_dataset = CustomDataset(test_dataset['data_smart'])
    
    train_sampler = SequentialSampler(custom_train_dataset)
    val_sampler = SequentialSampler(custom_val_dataset)
    test_sampler = SequentialSampler(custom_test_dataset)
    
    train_dataloader = DataLoader(custom_train_dataset, batch_size=args.smart_batch_size, sampler=train_sampler, collate_fn=collate_fn)
    val_dataloader = DataLoader(custom_val_dataset, batch_size=args.smart_batch_size, sampler=val_sampler, collate_fn=collate_fn)
    test_dataloader = DataLoader(custom_test_dataset, batch_size=args.smart_batch_size, sampler=test_sampler, collate_fn=collate_fn)
    
    # calculate smart embedding
    train_logits_all = []
    train_cls_token_all = []
    val_logits_all = []
    val_cls_token_all = []
    test_logits_all = []
    test_cls_token_all = []
    with torch.no_grad():
        for batch in train_dataloader:
            for key in batch:
                batch[key] = batch[key].cuda()
            h = encoder(**batch)
            logits, cls_token = classifier.get_logits_and_cls_token(h, **batch)
            train_logits_all.append(logits.cpu())
            train_cls_token_all.append(cls_token.cpu())
        for batch in val_dataloader:
            for key in batch:
                batch[key] = batch[key].cuda()
            h = encoder(**batch)
            logits, cls_token = classifier.get_logits_and_cls_token(h, **batch)
            val_logits_all.append(logits.cpu())
            val_cls_token_all.append(cls_token.cpu())
        for batch in test_dataloader:
            for key in batch:
                batch[key] = batch[key].cuda()
            h = encoder(**batch)
            logits, cls_token = classifier.get_logits_and_cls_token(h, **batch)
            test_logits_all.append(logits.cpu())
            test_cls_token_all.append(cls_token.cpu())
    
    train_logits = torch.cat(train_logits_all, dim=0)
    train_cls_token = torch.cat(train_cls_token_all, dim=0)
    val_logits = torch.cat(val_logits_all, dim=0)
    val_cls_token = torch.cat(val_cls_token_all, dim=0)
    test_logits = torch.cat(test_logits_all, dim=0)
    test_cls_token = torch.cat(test_cls_token_all, dim=0)
    
    train_dataset['smart_logits'] = train_logits
    train_dataset['smart_embedding'] = train_cls_token
    
    val_dataset['smart_logits'] = val_logits
    val_dataset['smart_embedding'] = val_cls_token
    
    test_dataset['smart_logits'] = test_logits
    test_dataset['smart_embedding'] = test_cls_token
    
    del encoder, classifier, checkpoint
    torch.cuda.empty_cache()  
    
    return train_dataset, val_dataset, test_dataset