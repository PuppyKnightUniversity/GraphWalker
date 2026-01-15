from data.prepare_ehr_data import set_dataset_numerical_ehr

def run_zero_shot(args):
    # load ehr dataset
    train_dataset_dict, val_dataset_dict, test_dataset_dict = set_dataset_numerical_ehr(args)
    
    pass