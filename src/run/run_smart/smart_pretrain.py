import os
import json
import copy
import logging
import warnings
from utils.utils import set_seed, distributed_init, init_logging
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
import random
from typing import Any, Dict, List
from models.smart import Encoder, EmbeddingDecoder
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


def test(args, checkpoint_path, test_dataloader, logger, encoder, predictor, target_encoder, criterion, test_dataset):
    checkpoint = torch.load(os.path.join(args.smart_save_dir, checkpoint_path))
    save_epoch = checkpoint['epoch']
    log(logger, "last saved model is in epoch {}".format(save_epoch))
    encoder.load_state_dict(checkpoint['encoder'])
    predictor.load_state_dict(checkpoint['predictor'])
    target_encoder.load_state_dict(checkpoint['target_encoder'])
    encoder.eval()
    predictor.eval()
    target_encoder.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            for key in batch:
                batch[key] = batch[key].cuda()
            with torch.no_grad():
                h = target_encoder(**batch)
            batch['labels'] = batch['x']
            batch['x'], batch['mask'], pretrain_mask = random_masking(batch['x'], batch['mask'], args.smart_min_mask_ratio, args.smart_max_mask_ratio)
            z = encoder(**batch)
            z = predictor(z)
            test_loss += criterion(z[:, :, 1:], h[:, :, 1:], pretrain_mask.permute(0, 2, 1).unsqueeze(-1).expand_as(z[:, :, 1:])).item() * batch['x'].shape[0]
    log(logger, 'Test Loss %.4f' % (test_loss / len(test_dataset)))


def smooth_l1_loss(pred, target, pad_mask, beta=1.0):
    diff = torch.abs(pred - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    combined_mask = pad_mask.bool()
    loss = (loss * combined_mask).sum() / (combined_mask.sum() + 1e-6)
    return loss

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

def smart_pretrain(args, train_dataset, val_dataset, test_dataset):

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
        # Use period_length for smart_max_len to match the data time window
        args.smart_max_len = args.period_length
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
    predictor = EmbeddingDecoder(args).cuda()
    target_encoder = copy.deepcopy(encoder)
    if args.distributed:
        encoder = torch.nn.parallel.DistributedDataParallel(encoder, static_graph=True, device_ids=[args.gpu], output_device=args.local_rank)
        predictor = torch.nn.parallel.DistributedDataParallel(predictor, static_graph=True, device_ids=[args.gpu], output_device=args.local_rank)
        target_encoder = torch.nn.parallel.DistributedDataParallel(target_encoder, device_ids=[args.gpu], output_device=args.local_rank, find_unused_parameters=True)
    for p in target_encoder.parameters():
        p.requires_grad = False
        
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
            'params': predictor.parameters()
        }
    ]
    optimizer = torch.optim.Adam(param_groups, args.smart_lr)
    criterion = smooth_l1_loss

    best_auc = 0
    best_prc = 0
    best_mse = 1
    for i in range(1, args.smart_epochs + 1):
        train_loss = 0
        val_loss = 0
        encoder.train()
        predictor.train()
        target_encoder.train()
        for step, batch in enumerate(train_dataloader, 1):
            for key in batch:
                batch[key] = batch[key].cuda()
            with torch.no_grad():
                h = target_encoder(**batch)
            batch['labels'] = batch['x']
            batch['x'], batch['mask'], pretrain_mask = random_masking(batch['x'], batch['mask'], args.smart_min_mask_ratio, args.smart_max_mask_ratio)
            z = encoder(**batch)
            z = predictor(z)
            loss = criterion(z[:, :, 1:], h[:, :, 1:], pretrain_mask.permute(0, 2, 1).unsqueeze(-1).expand_as(z[:, :, 1:]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                m = next(momentum_scheduler)
                for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                    param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
            train_loss += loss.item() * batch['x'].shape[0]

        encoder.eval()
        predictor.eval()
        target_encoder.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                for key in batch:
                    batch[key] = batch[key].cuda()
                with torch.no_grad():
                    h = target_encoder(**batch)
                batch['labels'] = batch['x']
                batch['x'], batch['mask'], pretrain_mask = random_masking(batch['x'], batch['mask'], args.smart_min_mask_ratio, args.smart_max_mask_ratio)
                z = encoder(**batch)
                z = predictor(z)
                val_loss += criterion(z[:, :, 1:], h[:, :, 1:], pretrain_mask.permute(0, 2, 1).unsqueeze(-1).expand_as(z[:, :, 1:])).item() * batch['x'].shape[0]
        log(logger, 'Epoch %d: Train Loss %.4f, Valid Loss %.4f' % (i, train_loss / len(train_dataset) * args.world_size, val_loss / len(val_dataset)))
        cur_mse = val_loss / len(val_dataset)
        if cur_mse < best_mse:
            best_mse = cur_mse
            if args.local_rank == 0:
                state = {
                    'encoder': encoder.state_dict(),
                    'predictor': predictor.state_dict(),
                    'target_encoder': target_encoder.state_dict(),
                    'epoch': i
                }
                log(logger, '----- Save best model - L1: %.4f -----' % cur_mse)
                # 确保保存目录存在
                os.makedirs(args.smart_save_dir, exist_ok=True)
                torch.save(state, os.path.join(args.smart_save_dir, 'checkpoint-mse.pth'))
        if args.distributed:
            dist.barrier()

    if args.distributed:
        dist.barrier()
    test(args, 'checkpoint-mse.pth', test_dataloader, logger, encoder, predictor, target_encoder, criterion, test_dataset)

if __name__ == "__main__":
    import pickle
    from args.ehrbase_args import parse_args
    args = parse_args()

    # --- 分布式 rank 兼容处理 ---
    # torchrun 提供的环境变量
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # 旧代码里用到 local_rank / gpu / smart_local_rank
    if not hasattr(args, "local_rank"):
        args.local_rank = local_rank
    if not hasattr(args, "gpu"):
        args.gpu = local_rank
    if not hasattr(args, "smart_local_rank"):
        args.smart_local_rank = local_rank

    # 旧代码里用到 rank / world_size
    if not hasattr(args, "rank"):
        args.rank = rank
    if not hasattr(args, "world_size"):
        args.world_size = world_size
    # ---------------------------------

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
    
    smart_pretrain(args, train_dataset, val_dataset, test_dataset)
