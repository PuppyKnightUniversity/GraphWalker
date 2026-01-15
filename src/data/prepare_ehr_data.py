# flake8: noqa
def prepare_ehr_data(args, logger=None):
    if logger is None:
        from utils.logger import get_logger
        logger = get_logger("EHR-DataPrep")
    
    # Basic ehr data preparation and wrap prompt
    logger.data_preparation_start(args.dataset)
    
    if args.dataset == "mimic3_mortality":
        from data.mimic3.prepare_mimic3_mortality import prepare
        train_data, val_data, test_data = prepare(args)
    elif args.dataset == "mimic3_los":
        from data.mimic3.prepare_mimic3_los import prepare
        train_data, val_data, test_data = prepare(args)
    elif args.dataset == "mimic4_mortality":
        from data.mimic4.prepare_mimic4_mortality import prepare
        train_data, val_data, test_data = prepare(args)
    elif args.dataset == "mimic4_los":
        from data.mimic4.prepare_mimic4_los import prepare
        train_data, val_data, test_data = prepare(args)
    elif args.dataset == "mimic4_readmission":
        from data.mimic4.prepare_mimic4_readmission import prepare
        train_data, val_data, test_data = prepare(args)
    elif args.dataset == "tjh_mortality":
        from data.tjh.prepare_tjh_mortality import prepare
        train_data, val_data, test_data = prepare(args)
    elif args.dataset == "tjh_los":
        from data.tjh.prepare_tjh_los import prepare
        train_data, val_data, test_data = prepare(args)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    # Display data preparation completion
    logger.data_preparation_complete(
        len(train_data['X']), 
        len(val_data['X']), 
        len(test_data['X'])
    )

    # Toy dataset, for quick test
    if args.toy_dataset:
        logger.info("Using toy dataset for quick testing...")
        train_data = {key: value[:args.toy_dataset_size_train] for key, value in train_data.items()}
        val_data = {key: value[:args.toy_dataset_size_val] for key, value in val_data.items()}
        test_data = {key: value[:args.toy_dataset_size_test] for key, value in test_data.items()}
        
        logger.toy_dataset_stats(
            len(train_data['X']), 
            len(val_data['X']), 
            len(test_data['X'])
        )
        
    return train_data, val_data, test_data

if __name__ == "__main__":
    from args.ehrbase_args import parse_args
    args = parse_args()
    from utils.utils import set_seed
    set_seed(args.seed)
    train_dataset, val_dataset, test_dataset = prepare_ehr_data(args)
    breakpoint()