from run.run_smart.dims import resolve_smart_dims, validate_smart_checkpoint
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
                        self.data[i]['mask'][j][k] = 0


def collate_fn(features: List[Dict[str, Any]]):
    batch = {}
    for key in features[0].keys():
        if key in ["x", "mask", "time"]:
            batch[key] = pad_sequence([torch.tensor(patient[key]) for patient in features], True)
        else:
            batch[key] = torch.tensor([patient[key] for patient in features])
    return batch

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def calculate_smart_embedding(args, train_dataset, val_dataset, test_dataset):
    resolve_smart_dims(args, (train_dataset, val_dataset, test_dataset))

    if getattr(args, 'ehr_protocol', None) == 'admission_daily':
        args.smart_max_len = max(sample['lens'] for split in (train_dataset, val_dataset, test_dataset)
                                 for sample in split['data_smart'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, classifier = _load_embedding_models(args, device)
    for dataset in (train_dataset, val_dataset, test_dataset):
        loader = DataLoader(CustomDataset(dataset['data_smart']),
                            batch_size=args.smart_batch_size, shuffle=False,
                            collate_fn=collate_fn)
        embeddings, logits = [], []
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                h = encoder(**batch)
                embeddings.append(_encoder_cls_embedding(h).cpu())
                if classifier is not None:
                    logits.append(classifier(h, **batch).cpu())
        if not embeddings:
            raise ValueError('Cannot calculate SMART embeddings for an empty split')
        dataset['smart_embedding'] = torch.cat(embeddings, dim=0)
        if logits:
            dataset['smart_logits'] = torch.cat(logits, dim=0)
        else:
            dataset.pop('smart_logits', None)
    del encoder, classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return train_dataset, val_dataset, test_dataset


def _encoder_cls_embedding(hidden):
    return hidden[:, :, 0, :].reshape(hidden.shape[0], -1)


def _load_embedding_models(args, device):
    logits_requested = any(getattr(args, flag, False) for flag in (
        'llm_smart_embedding_topk_add_smart_logits', 'random_few_shot_add_smart_logits',
        'graph_walker_add_smart_logits', 'cone_add_smart_logits',
        'llm_smart_embedding_topk_add_smart_logits_for_test_example',
        'random_few_shot_add_smart_logits_for_test_example',
        'graph_walker_add_smart_logits_for_test_example',
    ))
    explicit_path = getattr(args, 'embedding_model_path', None)
    filename = 'checkpoint-prc.pth' if logits_requested else 'checkpoint-mse.pth'
    checkpoint_path = explicit_path or os.path.join(args.smart_save_dir, filename)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f'SMART checkpoint not found: {checkpoint_path}. Supply the pretrained '
            'encoder with --embedding_model_path; logits ablations require a '
            'supervised checkpoint containing a classifier.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    validate_smart_checkpoint(args, checkpoint, load_classifier=logits_requested)
    if 'classifier' in checkpoint and not logits_requested:
        raise ValueError('This checkpoint contains a supervised classifier. '
                         'Provide an encoder-only checkpoint, or enable the logits option.')
    encoder = Encoder(args).to(device)
    state = remove_module_prefix(checkpoint['encoder'])
    pos_key = 'position_enc.pos_table'
    if pos_key in state and state[pos_key].shape != encoder.position_enc.pos_table.shape:
        state[pos_key] = encoder.position_enc.pos_table
    encoder.load_state_dict(state, strict=True)
    encoder.requires_grad_(False).eval()
    classifier = None
    if logits_requested:
        if 'classifier' not in checkpoint:
            raise ValueError('SMART logits ablations require a supervised classifier checkpoint')
        classifier = Classifier(args).to(device)
        classifier.load_state_dict(remove_module_prefix(checkpoint['classifier']))
        classifier.requires_grad_(False).eval()
    return encoder, classifier
