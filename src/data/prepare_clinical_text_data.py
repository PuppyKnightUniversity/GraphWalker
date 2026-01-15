import json
import os
import pickle
import random
from typing import List, Dict, Any

def prepare_clinical_text_data_extract_from_raw(args, logger=None):
    '''
    Extract clinical text data from raw data
    '''
    raw_data_path = args.mid_data_dump_path + f'/{args.dataset}/{args.dataset}_raw.pkl'
    if os.path.exists(raw_data_path):
        if logger:
            logger.info(f'Loading {args.dataset} data from directory: {raw_data_path}')
        else:
            print(f'Loading {args.dataset} data from directory: {raw_data_path}')
        data_all = pickle.load(open(raw_data_path, 'rb'))
        return data_all
    else:
        if logger:
            logger.info(f'Extracting {args.dataset} data from raw data path: {args.dataset_path}')
        else:
            print(f'Extracting {args.dataset} data from raw data path: {args.dataset_path}')
        data_path = args.dataset_path + f'/{args.dataset}.jsonl'
        
        # Initialize data dictionary
        data_all = {}
        
        # Load JSONL file (each line is a JSON object)
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Get all keys from the first line to initialize the dictionary
        if len(lines) > 0:
            first_item = json.loads(lines[0])
            for key in first_item.keys():
                data_all[key] = []
        
        # Load all data
        for line in lines:
            item = json.loads(line.strip())
            for key in data_all.keys():
                data_all[key].append(item.get(key, None))
        
        # Dump the raw data
        if not os.path.exists(os.path.dirname(raw_data_path)):
            os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
        pickle.dump(data_all, open(raw_data_path, 'wb'))
        
        num_samples = len(data_all[list(data_all.keys())[0]]) if data_all else 0
        if logger:
            logger.info(f'Loaded {num_samples} samples from {args.dataset}')
        else:
            print(f'Loaded {num_samples} samples from {args.dataset}')
        return data_all

def prepare_clinical_text_data_train_val_test_split(args, data_all, logger=None):
    '''
    Split the data into train, val, and test sets
    '''
    split_data_path = args.mid_data_dump_path + f'/{args.dataset}/seed{args.seed}'
    if os.path.exists(split_data_path):
        train_data = pickle.load(open(split_data_path + f'/{args.dataset}_train.pkl', 'rb'))
        val_data = pickle.load(open(split_data_path + f'/{args.dataset}_val.pkl', 'rb'))
        test_data = pickle.load(open(split_data_path + f'/{args.dataset}_test.pkl', 'rb'))
        
        # Print sample sizes for loaded train/val/test sets
        first_key = list(train_data.keys())[0] if train_data else None
        if first_key:
            train_size = len(train_data[first_key])
            val_size = len(val_data[first_key])
            test_size = len(test_data[first_key])
            if logger:
                logger.info("Loaded existing data split:")
                logger.info(f"  Train samples: {train_size}")
                logger.info(f"  Val samples: {val_size}")
                logger.info(f"  Test samples: {test_size}")
            else:
                print("Loaded existing data split:")
                print(f"  Train samples: {train_size}")
                print(f"  Val samples: {val_size}")
                print(f"  Test samples: {test_size}")
        
        return train_data, val_data, test_data
    else:
        if logger:
            logger.info(f'Splitting {args.dataset} data into train, val, and test sets...')
        else:
            print(f'Splitting {args.dataset} data into train, val, and test sets...')
        train_ratio = args.train_ratio
        dump_path = args.mid_data_dump_path + f'/{args.dataset}/seed{args.seed}'
        if not os.path.exists(dump_path):
            os.makedirs(dump_path, exist_ok=True)
        
        # Get the number of samples (using the first key as reference)
        first_key = list(data_all.keys())[0]
        num_samples = len(data_all[first_key])
        data_index = list(range(num_samples))
        
        # Set random seed for reproducibility
        random.seed(args.seed)
        random.shuffle(data_index)
        
        train_num = int(num_samples * train_ratio)
        val_num = int(num_samples * ((1 - train_ratio) / 2))
        test_num = num_samples - train_num - val_num
        
        # Split data
        train_data = {}
        for idx in data_index[:train_num]:
            for key in data_all.keys():
                if key not in train_data:
                    train_data[key] = []
                train_data[key].append(data_all[key][idx])
        pickle.dump(train_data, open(dump_path + f'/{args.dataset}_train.pkl', 'wb'))
        
        val_data = {}
        for idx in data_index[train_num:train_num + val_num]:
            for key in data_all.keys():
                if key not in val_data:
                    val_data[key] = []
                val_data[key].append(data_all[key][idx])
        pickle.dump(val_data, open(dump_path + f'/{args.dataset}_val.pkl', 'wb'))
        
        test_data = {}
        for idx in data_index[train_num + val_num:]:
            for key in data_all.keys():
                if key not in test_data:
                    test_data[key] = []
                test_data[key].append(data_all[key][idx])
        pickle.dump(test_data, open(dump_path + f'/{args.dataset}_test.pkl', 'wb'))
        
        # Print sample sizes for train/val/test sets
        if logger:
            logger.info("Data split completed:")
            logger.info(f"  Train samples: {len(train_data[first_key])}")
            logger.info(f"  Val samples: {len(val_data[first_key])}")
            logger.info(f"  Test samples: {len(test_data[first_key])}")
            logger.info(f"  Total samples: {len(train_data[first_key]) + len(val_data[first_key]) + len(test_data[first_key])}")
        else:
            print("Data split completed:")
            print(f"  Train samples: {len(train_data[first_key])}")
            print(f"  Val samples: {len(val_data[first_key])}")
            print(f"  Test samples: {len(test_data[first_key])}")
            print(f"  Total samples: {len(train_data[first_key]) + len(val_data[first_key]) + len(test_data[first_key])}")
        
        return train_data, val_data, test_data

def prepare_clinical_text_data(args, logger=None):
    if logger is None:
        from utils.logger import get_logger
        logger = get_logger("Clinical-Text-DataPrep")
    assert args.dataset in ['cmb_exam_patient', 'cmb_clin', 'medqa'], f"Dataset {args.dataset} not supported in prepare_clinical_text_data"
    
    # Basic clinical text data preparation and wrap prompt
    logger.data_preparation_start(args.dataset)
    
    # Extract data from raw files
    data_all = prepare_clinical_text_data_extract_from_raw(args, logger)
    
    # Split into train, val, test
    train_data, val_data, test_data = prepare_clinical_text_data_train_val_test_split(args, data_all, logger)
    
    # Display data preparation completion
    first_key = list(train_data.keys())[0] if train_data else None
    if first_key:
        logger.data_preparation_complete(
            len(train_data[first_key]), 
            len(val_data[first_key]), 
            len(test_data[first_key])
        )
    
    # Toy dataset, for quick test
    if args.toy_dataset:
        logger.info("Using toy dataset for quick testing...")
        train_data = {key: value[:args.toy_dataset_size_train] for key, value in train_data.items()}
        val_data = {key: value[:args.toy_dataset_size_val] for key, value in val_data.items()}
        test_data = {key: value[:args.toy_dataset_size_test] for key, value in test_data.items()}
        
        first_key = list(train_data.keys())[0] if train_data else None
        if first_key:
            logger.toy_dataset_stats(
                len(train_data[first_key]), 
                len(val_data[first_key]), 
                len(test_data[first_key])
            )
    
    return train_data, val_data, test_data
