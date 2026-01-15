import os
import json
import copy
import logging
import warnings
from utils.utils import set_seed, distributed_init, init_logging
from utils.metrics import print_metrics_binary, print_metrics_multilabel, print_metrics_regression
from utils.llm_eval import evaluate_binary_model_with_bootstrap
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
import random
from typing import Any, Dict, List
from models.smart import Encoder, Classifier
import torch.distributed as dist

# 屏蔽PyTorch DDP相关的警告
warnings.filterwarnings("ignore", message="You passed find_unused_parameters=true to DistributedDataParallel")

def random_masking(x, original_mask, min_mask_ratio, max_mask_ratio):
    """
    Perform per-sample random masking.
    """
    N, L, V = x.shape  # batch, length, var

    # Calculate mask ratios and lengths to keep for each sample in the batch
    mask_ratios = torch.rand(N, device=x.device) * \
        (max_mask_ratio - min_mask_ratio) + min_mask_ratio
    
    mask = torch.rand_like(x) < mask_ratios.view(-1, 1, 1)
    x = x * (~mask)  # True for reconstruction, False for original
    return x, original_mask * (~mask),  original_mask * mask


def test(args, checkpoint_path, test_dataloader, logger, encoder, classifier, criterion, test_dataset):
    checkpoint = torch.load(os.path.join(args.smart_save_dir, checkpoint_path))
    save_epoch = checkpoint['epoch']
    log(logger, "last saved model is in epoch {}".format(save_epoch))
    encoder.load_state_dict(checkpoint['encoder'])
    classifier.load_state_dict(checkpoint['classifier'])
    encoder.eval()
    classifier.eval()
    test_loss = 0
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for batch in test_dataloader:
            for key in batch:
                batch[key] = batch[key].cuda()
            h = encoder(**batch)
            preds = classifier(h, **batch)
            test_loss += criterion(preds, batch['labels']).item() * batch['x'].shape[0]
            preds_all.append(preds.cpu())
            labels_all.append(batch['labels'].cpu())
    if args.dataset == 'mimic3_los' or args.dataset == 'mimic4_los' or args.dataset == 'tjh_los':
        print_metrics = print_metrics_multilabel
        print_metrics(torch.cat(labels_all), torch.cat(preds_all), args.local_rank == 0)
    else:
        # For binary classification (mimic3_mortality), use bootstrap evaluation
        labels_all_tensor = torch.cat(labels_all)
        preds_all_tensor = torch.cat(preds_all)
        
        # Convert logits to probabilities (softmax and take positive class probability)
        if len(preds_all_tensor.shape) == 1:
            # If predictions are already probabilities or logits for single class
            probs = F.sigmoid(preds_all_tensor).numpy()
        else:
            # If predictions are logits for two classes, apply softmax and take positive class
            probs = F.softmax(preds_all_tensor, dim=1)[:, 1].numpy()
        
        labels_numpy = labels_all_tensor.numpy().astype(int)
        
        # Only run bootstrap evaluation on rank 0 to avoid duplicate output
        if args.local_rank == 0:
            log(logger, "Starting bootstrap evaluation...")
            bootstrap_results = evaluate_binary_model_with_bootstrap(
                score_list=probs.tolist(),
                label_list=labels_numpy.tolist(),
                n_bootstrap=1000,
                confidence_level=0.95,
                random_state=42,
                logger=logger
            )
            log(logger, "Bootstrap evaluation completed.")
        else:
            # Still run basic metrics for consistency
            print_metrics_binary(labels_all_tensor, preds_all_tensor, verbose=False)
    
    log(logger, 'Test Loss %.4f' % (test_loss / len(test_dataset)))

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

def smart_finetune(args, train_dataset, val_dataset, test_dataset):

    # initialize the logger, seed, distributed
    save_dir = os.path.join(args.smart_save_dir, args.dataset, 'smart')
    # 确保保存目录存在
    if args.local_rank == 0:
        os.makedirs(args.smart_save_dir, exist_ok=True)
    if args.local_rank == 0:
        logger = logging.getLogger()
        init_logging(logger, args.smart_save_dir if args.smart_save_model else None)
    else:
        logger = None
    log(logger, json.dumps(vars(args), indent=4))
    set_seed(args.seed)
    distributed_init(args)

    if args.dataset == 'mimic3_mortality':
        args.smart_input_dim = 17
        args.smart_demo_dim = 0
        args.smart_num_class = 2
        # Use period_length for smart_max_len to match the data time window
        args.smart_max_len = args.period_length
    elif args.dataset == 'mimic3_los':
        args.smart_input_dim = 17
        args.smart_demo_dim = 0
        args.smart_num_class = 4
        # Use period_length for smart_max_len to match the data time window
        args.smart_max_len = args.period_length
    elif args.dataset == 'mimic4_los':
        args.smart_input_dim = 44
        args.smart_demo_dim = 2
        args.smart_num_class = 4
    elif args.dataset == 'mimic4_mortality':
        args.smart_input_dim = 44
        args.smart_demo_dim = 2
        args.smart_num_class = 2
        # Use period_length for smart_max_len to match the data time window
        args.smart_max_len = args.period_length
    elif args.dataset == 'mimic4_readmission':
        args.smart_input_dim = 44
        args.smart_demo_dim = 2
        args.smart_num_class = 2
        # Use period_length for smart_max_len to match the data time window
        args.smart_max_len = args.period_length
    elif args.dataset == 'tjh_mortality':
        args.smart_input_dim = 75
        args.smart_demo_dim = 2
        args.smart_num_class = 2
        # Use period_length for smart_max_len to match the data time window
        args.smart_max_len = args.period_length
    elif args.dataset == 'tjh_los':
        args.smart_input_dim = 75
        args.smart_demo_dim = 2
        args.smart_num_class = 4
        # Use period_length for smart_max_len to match the data time window
        args.smart_max_len = args.period_length
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    # load the dataset 
    train_dataset = CustomDataset(train_dataset['data_smart'])
    val_dataset = CustomDataset(val_dataset['data_smart'])
    test_dataset = CustomDataset(test_dataset['data_smart'])
    if args.smart_data_dropout > 0:
        train_dataset.dropout_data(args.smart_data_dropout)
        val_dataset.dropout_data(args.smart_data_dropout)
        test_dataset.dropout_data(args.smart_data_dropout)
    log(logger, 'Dataset Loaded.')
    
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True, drop_last=True)
        val_sampler = SequentialSampler(val_dataset)
        test_sampler = SequentialSampler(test_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)
        test_sampler = SequentialSampler(test_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.smart_batch_size, sampler=train_sampler, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.smart_batch_size, sampler=val_sampler, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.smart_batch_size, sampler=test_sampler, collate_fn=collate_fn)

    encoder = Encoder(args).cuda()
    classifier = Classifier(args).cuda()
 
    if args.distributed:
        encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[args.gpu], output_device=args.local_rank, find_unused_parameters=True)
        classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[args.gpu], output_device=args.local_rank, find_unused_parameters=True)
        
    ema = [0.996, 1]
    ipe = len(train_dataloader)
    ipe_scale = 1.0
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*args.smart_epochs*ipe_scale)
                          for i in range(int(ipe*args.smart_epochs*ipe_scale)+1))
    
    param_groups = [
        {
            'params': encoder.parameters(),
        }, 
        {
            'params': classifier.parameters()
        }
    ]
    optimizer = torch.optim.Adam(param_groups, args.smart_lr)
    criterion = torch.nn.CrossEntropyLoss()
    if args.dataset == 'mimic_phenotyping':
        criterion = torch.nn.BCEWithLogitsLoss()
        print_metrics = print_metrics_multilabel
        save_metric = 'auc_macro'
    elif args.dataset in ['mimic3_los', 'mimic4_los', 'tjh_los']:
        criterion = torch.nn.CrossEntropyLoss()  # 多分类任务使用CrossEntropyLoss
        print_metrics = print_metrics_multilabel
        save_metric = 'auc_macro'
    else:
        print_metrics = print_metrics_binary
        save_metric = 'auprc'
    
    checkpoint = torch.load(os.path.join(args.smart_save_dir, 'checkpoint-mse.pth'))
    save_epoch = checkpoint['epoch']
    log(logger, "last saved model is in epoch {}".format(save_epoch))
    encoder.load_state_dict(checkpoint['encoder'])

    best_auc = 0
    best_prc = 0
    best_mse = 100
    for i in range(1, args.smart_epochs + 1):
        train_loss = 0
        val_loss = 0
        encoder.train()
        classifier.train()
        for step, batch in enumerate(train_dataloader, 1):
            for key in batch:
                batch[key] = batch[key].cuda()
            if i <= args.smart_freeze_epochs:
                with torch.no_grad():
                    h = encoder(**batch)
            else:
                h = encoder(**batch)
            preds = classifier(h, **batch)
            loss = criterion(preds, batch['labels'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch['x'].shape[0]

        encoder.eval()
        classifier.eval()
        preds_all = []
        labels_all = []
        with torch.no_grad():
            for batch in val_dataloader:
                for key in batch:
                    batch[key] = batch[key].cuda()
                h = encoder(**batch)
                preds = classifier(h, **batch)
                val_loss += criterion(preds, batch['labels']).item() * batch['x'].shape[0]
                preds_all.append(preds.cpu())
                labels_all.append(batch['labels'].cpu())
        metrics = print_metrics(torch.cat(labels_all), torch.cat(preds_all), args.local_rank == 0)
        log(logger, 'Epoch %d: Train Loss %.4f, Valid Loss %.4f' % (i, train_loss / len(train_dataset) * args.world_size, val_loss / len(val_dataset)))
        cur_mse = val_loss / len(val_dataset)
        if save_metric != 'mse':
            if metrics[save_metric] > best_prc:
                best_prc = metrics[save_metric]
                if args.local_rank == 0:
                    state = {
                        'encoder': encoder.state_dict(),
                        'classifier': classifier.state_dict(),
                        'epoch': i
                    }
                    log(logger, f'----- Save best model - {save_metric}: %.4f -----' % metrics[save_metric])
                    os.makedirs(args.smart_save_dir, exist_ok=True)
                    torch.save(state, os.path.join(args.smart_save_dir, 'checkpoint-prc.pth'))
        else:
            if metrics[save_metric] < best_mse:
                best_mse = metrics[save_metric]
                if args.local_rank == 0:
                    state = {
                        'encoder': encoder.state_dict(),
                        'classifier': classifier.state_dict(),
                        'epoch': i
                    }
                    log(logger, f'----- Save best model - {save_metric}: %.4f -----' % metrics[save_metric])
                    os.makedirs(args.smart_save_dir, exist_ok=True)
                    torch.save(state, os.path.join(args.smart_save_dir, 'checkpoint-prc.pth'))
        if args.distributed:
            dist.barrier()

    if args.distributed:
        dist.barrier()
    test(args, 'checkpoint-prc.pth', test_dataloader, logger, encoder, classifier, criterion, test_dataset)

if __name__ == "__main__":
    import pickle
    from args.ehrbase_args import parse_args
    args = parse_args()


    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if not hasattr(args, "local_rank"):
        args.local_rank = local_rank
    if not hasattr(args, "gpu"):
        args.gpu = local_rank
    if not hasattr(args, "smart_local_rank"):
        args.smart_local_rank = local_rank

    if not hasattr(args, "rank"):
        args.rank = rank
    if not hasattr(args, "world_size"):
        args.world_size = world_size


    train_dataset = pickle.load(open(args.mid_data_dump_path + f'/{args.dataset}/seed' + str(args.seed) + f'/{args.dataset}_train.pkl', 'rb'))
    val_dataset = pickle.load(open(args.mid_data_dump_path + f'/{args.dataset}/seed' + str(args.seed) + f'/{args.dataset}_val.pkl', 'rb'))
    test_dataset = pickle.load(open(args.mid_data_dump_path + f'/{args.dataset}/seed' + str(args.seed) + f'/{args.dataset}_test.pkl', 'rb'))
    
    # Toy dataset, for quick test
    if getattr(args, 'toy_dataset', False):
        print("Using toy dataset for quick testing...")
        train_dataset = {key: value[:1600] for key, value in train_dataset.items()}
        val_dataset = {key: value[:200] for key, value in val_dataset.items()}
        test_dataset = {key: value[:200] for key, value in test_dataset.items()}
        # Get dataset size for logging
        train_size = len(train_dataset.get('data_smart', train_dataset.get('X', [])))
        val_size = len(val_dataset.get('data_smart', val_dataset.get('X', [])))
        test_size = len(test_dataset.get('data_smart', test_dataset.get('X', [])))
        print(f"Toy dataset stats - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    smart_finetune(args, train_dataset, val_dataset, test_dataset)
