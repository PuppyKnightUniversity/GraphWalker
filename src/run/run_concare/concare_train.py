import os
import json
import logging
import warnings
import numpy as np
from utils.utils import set_seed, distributed_init, init_logging
from utils.numerical.metrics import get_all_metrics, get_all_metrics_with_bootstrap, check_metric_is_better
from utils.numerical.loss import get_loss
from utils.numerical.sequence_handler import unpad_y
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
import random
from typing import Any, Dict, List
from models.concare import ConCare
import torch.distributed as dist

# Filter PyTorch DDP warnings
warnings.filterwarnings("ignore", message="You passed find_unused_parameters=true to DistributedDataParallel")


def log(logger, msg):
    if logger is not None:
        logger.info(msg)


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


def test(args, checkpoint_path, test_dataloader, logger, model, head, test_dataset, los_info=None):
    checkpoint_file = os.path.join(args.concare_save_dir, checkpoint_path)
    if not os.path.exists(checkpoint_file):
        log(logger, f"Checkpoint not found at {checkpoint_file}. Skipping test.")
        return {}
        
    checkpoint = torch.load(checkpoint_file)
    save_epoch = checkpoint['epoch']
    log(logger, "last saved model is in epoch {}".format(save_epoch))
    model.load_state_dict(checkpoint['model'])
    head.load_state_dict(checkpoint['head'])
    model.eval()
    head.eval()
    
    preds_all = []
    labels_all = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            for key in batch:
                batch[key] = batch[key].cuda()
            
            # Forward pass
            embedding, decov_loss = model(batch['x'], mask=batch['mask'])
            preds = head(embedding)
            
            preds_all.append(preds.cpu())
            labels_all.append(batch['labels'].cpu())

    preds = torch.cat(preds_all)
    labels = torch.cat(labels_all)

    # Determine task type
    if args.dataset in ['mimic3_los', 'mimic4_los', 'tjh_los']:
        if torch.is_floating_point(labels):
            task = 'los'
        else:
            task = 'multiclass'
    elif args.dataset in ['mimic3_mortality', 'mimic4_mortality', 'tjh_mortality']:
        task = 'mortality'
    elif args.dataset in ['mimic4_readmission']:
        task = 'readmission'
    else:
        raise ValueError(f"Unknown task for dataset {args.dataset}")

    # Log prediction statistics
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        log(logger, f"Prediction stats: min={preds.min():.4f}, max={preds.max():.4f}, mean={preds.mean():.4f}")

    metrics = get_all_metrics_with_bootstrap(preds, labels, task, los_info)
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        log(logger, "Test Metrics (with Bootstrap):")
        for k, v in metrics.items():
            if isinstance(v, dict) and 'mean' in v:
                log(logger, f"  {k}: {v['value']:.4f} ± {v['std']:.4f}")
            else:
                log(logger, f"  {k}: {v}")

    return metrics


def concare_train(args, train_dataset, val_dataset, test_dataset):

    # initialize the logger, seed, distributed
    # args.concare_save_dir is set in run.py or args
    
    # Construct absolute path with dataset subdirectory
    # Ensure we don't double append if called multiple times (though unlikely in this architecture)
    if not args.concare_save_dir.endswith(args.dataset):
        save_dir = os.path.abspath(os.path.join(args.concare_save_dir, args.dataset))
        args.concare_save_dir = save_dir
    else:
        args.concare_save_dir = os.path.abspath(args.concare_save_dir)
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if local_rank == 0:
        os.makedirs(args.concare_save_dir, exist_ok=True)
        logger = logging.getLogger()
        init_logging(logger, args.concare_save_dir)
        log(logger, f"Saving checkpoints to: {args.concare_save_dir}")
    else:
        logger = None
    
    log(logger, json.dumps(vars(args), indent=4))
    set_seed(args.seed)
    distributed_init(args)

    # Infer input dimensions from dataset
    first_sample = train_dataset['data_smart'][0]
    if 'x' in first_sample:
        # Check shape of x
        x_sample = first_sample['x']
        # x is usually list of lists or numpy array [Time, Feat]
        if hasattr(x_sample, 'shape'):
            args.concare_lab_dim = x_sample.shape[-1]
        elif isinstance(x_sample, list) and len(x_sample) > 0:
            args.concare_lab_dim = len(x_sample[0])
            
    if 'static' in first_sample:
        static_sample = first_sample['static']
        if hasattr(static_sample, 'shape'):
            args.concare_demo_dim = static_sample.shape[-1]
        elif isinstance(static_sample, list):
            args.concare_demo_dim = len(static_sample)
    else:
        args.concare_demo_dim = 0
        
    log(logger, f"Inferred dimensions: lab_dim={args.concare_lab_dim}, demo_dim={args.concare_demo_dim}")

    # Determine output dim
    if args.dataset in ['mimic3_los', 'mimic4_los', 'tjh_los']:
        # Check if labels are classification (int) or regression (float)
        # We checked earlier that mimic3_los has int labels [0,1,2,3]
        if 'labels' in first_sample:
            label_sample = first_sample['labels']
            if isinstance(label_sample, (int, np.integer)) or (hasattr(label_sample, 'dtype') and np.issubdtype(label_sample.dtype, np.integer)):
                 # It's likely classification
                 # We need to find number of classes. 
                 # Since we don't want to scan whole dataset, we might need to rely on prior knowledge or scan small part
                 # For mimic3_los, we know it's 4 classes.
                 # Let's do a quick scan of first 100 samples to guess max label
                 max_label = 0
                 for i in range(min(100, len(train_dataset['data_smart']))):
                     lbl = train_dataset['data_smart'][i]['labels']
                     if isinstance(lbl, (int, np.integer)):
                         max_label = max(max_label, int(lbl))
                     elif hasattr(lbl, 'item'):
                         max_label = max(max_label, int(lbl.item()))
                 
                 output_dim = max_label + 1
                 task = 'multiclass'
            else:
                 output_dim = 1 # Regression
                 task = 'los'
        else:
             output_dim = 1
             task = 'los'
    else:
        output_dim = 1 # Binary classification (sigmoid)
        task = 'mortality' if 'mortality' in args.dataset else 'readmission'
    
    log(logger, f"Task type: {task}, Output dim: {output_dim}")

    # Load datasets
    train_dataset = CustomDataset(train_dataset['data_smart'])
    val_dataset = CustomDataset(val_dataset['data_smart'])
    test_dataset = CustomDataset(test_dataset['data_smart'])
    
    log(logger, 'Dataset Loaded.')
    
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True, drop_last=True)
        val_sampler = SequentialSampler(val_dataset)
        test_sampler = SequentialSampler(test_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)
        test_sampler = SequentialSampler(test_dataset)
        
    train_dataloader = DataLoader(train_dataset, batch_size=args.concare_batch_size, sampler=train_sampler, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.concare_batch_size, sampler=val_sampler, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.concare_batch_size, sampler=test_sampler, collate_fn=collate_fn)

    # Initialize Model
    model = ConCare(
        lab_dim=args.concare_lab_dim,
        demo_dim=args.concare_demo_dim,
        hidden_dim=args.concare_hidden_dim,
        num_head=args.concare_num_head,
        dropout=args.concare_dropout
    ).cuda()
    
    head = nn.Sequential(
        nn.Linear(args.concare_hidden_dim, output_dim),
        nn.Dropout(0.0),
        nn.Sigmoid() if task not in ['los', 'multiclass'] else nn.Identity()
    ).cuda()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=local_rank, find_unused_parameters=True)
        head = torch.nn.parallel.DistributedDataParallel(head, device_ids=[args.gpu], output_device=local_rank, find_unused_parameters=True)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=args.concare_lr)
    
    # Training Loop
    best_metric_score = -float('inf') if task != 'los' else float('inf')
    best_metric_name = 'auprc' if task not in ['los', 'multiclass'] else ('mae' if task == 'los' else 'ma-ROC')
    
    # LOS info for unnormalization if needed
    los_info = None 
    
    for i in range(1, args.concare_epochs + 1):
        model.train()
        head.train()
        train_loss_accum = 0
        
        for step, batch in enumerate(train_dataloader, 1):
            for key in batch:
                batch[key] = batch[key].cuda()
            
            embedding, decov_loss = model(batch['x'], mask=batch['mask'])
            preds = head(embedding)
            
            # Calculate loss
            if task == 'los':
                loss = get_loss(preds.squeeze(), batch['labels'].float(), task)
            elif task == 'multiclass':
                # CrossEntropyLoss expects (N, C) preds and (N) long labels
                criterion = nn.CrossEntropyLoss()
                loss = criterion(preds, batch['labels'].long())
            else:
                loss = get_loss(preds.squeeze(), batch['labels'].float(), task)
                
            total_loss = loss + 10 * decov_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            train_loss_accum += total_loss.item() * batch['x'].shape[0]

        # Validation
        model.eval()
        head.eval()
        preds_all = []
        labels_all = []
        val_loss_accum = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                for key in batch:
                    batch[key] = batch[key].cuda()
                
                embedding, decov_loss = model(batch['x'], mask=batch['mask'])
                preds = head(embedding)
                
                if task == 'los':
                    loss = get_loss(preds.squeeze(), batch['labels'].float(), task)
                elif task == 'multiclass':
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(preds, batch['labels'].long())
                else:
                    loss = get_loss(preds.squeeze(), batch['labels'].float(), task)
                
                total_loss = loss + 10 * decov_loss
                val_loss_accum += total_loss.item() * batch['x'].shape[0]
                
                preds_all.append(preds.cpu())
                labels_all.append(batch['labels'].cpu())
        
        preds = torch.cat(preds_all)
        labels = torch.cat(labels_all)
        
        # Calculate metrics
        metrics = get_all_metrics(preds, labels, task, los_info={'los_std': 1.0, 'los_mean': 0.0}) 
        
        log(logger, f'Epoch {i}: Train Loss {train_loss_accum / len(train_dataset):.4f}, Val Loss {val_loss_accum / len(val_dataset):.4f}')
        
        if local_rank == 0:
            current_score = metrics[best_metric_name]
            is_better = False
            if task == 'los':
                if current_score < best_metric_score:
                    is_better = True
            else:
                if current_score > best_metric_score:
                    is_better = True
            
            if is_better:
                best_metric_score = current_score
                state = {
                    'model': model.state_dict(),
                    'head': head.state_dict(),
                    'epoch': i
                }
                log(logger, f'----- Save best model - {best_metric_name}: {current_score:.4f} -----')
                torch.save(state, os.path.join(args.concare_save_dir, 'checkpoint-best.pth'))

        if args.distributed:
            dist.barrier()

    # Test with best model
    if args.distributed:
        dist.barrier()
        
    test(args, 'checkpoint-best.pth', test_dataloader, logger, model, head, test_dataset, los_info={'los_std': 1.0, 'los_mean': 0.0})


if __name__ == "__main__":
    from args.ehrbase_args import parse_args
    import pickle
    
    args = parse_args()
    
    # Load data
    train_dataset = pickle.load(open(args.mid_data_dump_path + f'/{args.dataset}/seed' + str(args.seed) + f'/{args.dataset}_train.pkl', 'rb'))
    val_dataset = pickle.load(open(args.mid_data_dump_path + f'/{args.dataset}/seed' + str(args.seed) + f'/{args.dataset}_val.pkl', 'rb'))
    test_dataset = pickle.load(open(args.mid_data_dump_path + f'/{args.dataset}/seed' + str(args.seed) + f'/{args.dataset}_test.pkl', 'rb'))

    if getattr(args, 'toy_dataset', False):
        print("Using toy dataset...")
        # Dictionary slicing for toy dataset
        # data_smart is the key
        train_dataset = {'data_smart': train_dataset['data_smart'][:1600]}
        val_dataset = {'data_smart': val_dataset['data_smart'][:200]}
        test_dataset = {'data_smart': test_dataset['data_smart'][:200]}

    concare_train(args, train_dataset, val_dataset, test_dataset)
