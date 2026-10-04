import os
import pickle
import numpy as np
import json
import re
from typing import List, Dict, Any
import pickle
import random
from tqdm import tqdm

def prepare_mimic3_mortality_extract_from_raw(args):
    '''
        Extract mimic3 mortality data from raw data
    '''
    period_length = args.period_length
    # Include period_length in cache file path to avoid cache conflicts
    raw_data_path = args.mid_data_dump_path + f'/mimic3_mortality/mimic3_mortality_raw_period{period_length}.pkl'
    if os.path.exists(raw_data_path):
        print('Loading mimic3 mortality data from directory: ', raw_data_path)
        patient_all = pickle.load(open(raw_data_path, 'rb'))
        return patient_all
    else:
        # Read the benchmark train, validation and test listfiles.
        print('Extracting mimic3 mortality data from raw data path: ', args.dataset_path)
        path = args.dataset_path
        print(f"period_length: {period_length}")

        patient_all = {}
        patient_all['X'] = []
        patient_all['t'] = []
        patient_all['y'] = []
        patient_all['header'] = []
        patient_all['name'] = []

        from data.mimic3.reader import InHospitalMortalityReader, read_chunk
        for mode in ['train', 'val', 'test']:
            # read data from raw mimic3 data
            reader = InHospitalMortalityReader(dataset_dir=os.path.join(path, 'train' if mode != 'test' else 'test'),
                    listfile=os.path.join(path, mode + '_listfile.csv'), period_length=period_length)
            N = reader.get_number_of_examples()
            # read data in chunks to accelerate
            ret = read_chunk(reader, N)
            data = ret["X"]
            ts = ret["t"]
            labels = ret["y"]
            header = ret["header"]
            names = ret["name"]
            # append all parts of data to one list
            patient_all['X'] += data
            patient_all['t'] += ts
            patient_all['y'] += labels
            patient_all['header'] += header
            patient_all['name'] += names
        # dump the raw data
        if not os.path.exists(raw_data_path):
            os.makedirs(os.path.dirname(raw_data_path))
        pickle.dump(patient_all, open(raw_data_path, 'wb'))

        return patient_all

def prepare_mimic3_mortality_for_smart(args, patient_all):
    '''
    This function is to adapt the raw data for SMART model
    '''
    period_length = args.period_length
    # Include period_length in cache file path to avoid cache conflicts
    smart_data_path = args.mid_data_dump_path + f'/mimic3_mortality/mimic3_mortality_smart_period{period_length}.pkl'
    print('Adapting mimic3 mortality data for SMART model...')
    # load channel info
    with open(args.channel_info_path) as f:
        series_channel_info = json.load(f)
    # load discretizer config
    with open(args.discretizer_config_path) as f:
        series_config = json.load(f)
        id_to_channel = series_config['id_to_channel']
        is_categorical_channel = series_config['is_categorical_channel']
        normal_values = series_config['normal_values']
        possible_values = series_config['possible_values']

    data_all = []
    mask_all = []
    label_all = []
    name_all = []

    data_smart = []
    for patient, name in tqdm(zip(patient_all['X'], patient_all['name']),
                            total=len(patient_all['X']),
                            desc="processing patients",
                            unit="patient"):
        data_patient = np.zeros(shape=(len(id_to_channel), period_length), dtype=np.float32)
        mask_patient = np.zeros(shape=(len(id_to_channel), period_length), dtype=np.float32)
        last_time = -1
        for row in patient:
            time = int(float(row[0]))
            if time == period_length:
                time -= 1
            if time > period_length:
                raise ValueError('This should not happen')
                break
            for index in range(len(row) - 1):
                value = row[index + 1]
                if value == '':
                    # continue
                    if mask_patient[index, time] == 0 and time - last_time > 0:
                        if last_time >= 0:
                            data_patient[index, last_time + 1:time + 1] = data_patient[index, last_time]
                        else:
                            if is_categorical_channel[id_to_channel[index]]:
                                data_patient[index, last_time + 1:time + 1] = series_channel_info[id_to_channel[index]]['values'][normal_values[id_to_channel[index]]]
                            else:
                                data_patient[index, last_time + 1:time + 1] = float(normal_values[id_to_channel[index]])
                else:
                    mask_patient[index, time] = 1
                    if is_categorical_channel[id_to_channel[index]]:
                        data_patient[index, time] = series_channel_info[id_to_channel[index]]['values'][value]
                    else:
                        data_patient[index, time] = float(value)
            last_time = time
        if last_time < period_length - 1:
            data_patient[:, last_time + 1:period_length] = data_patient[:, last_time, None]
        data_all.append(data_patient.transpose(-1, -2))
        mask_all.append(mask_patient.transpose(-1, -2))

    from data.smart_preprocessing import standardize_smart_data
    data_normalized = standardize_smart_data(args, patient_all, data_all, mask_all, 'mimic3_mortality')
    mask_all = [mask.tolist() for mask in mask_all]
    label_all = patient_all['y']
    name_all = patient_all['name']
    # NOTE: data_smart is a list of dicts, each dict corresponds to a patient
    x_len = [len(i) for i in data_normalized]
    for idx in range(len(patient_all['X'])):
        data_smart.append({
            "x": data_normalized[idx],
            "labels": label_all[idx],
            "lens": x_len[idx],
            "mask": mask_all[idx],
        })
    patient_all['data_smart'] = data_smart
    os.makedirs(os.path.dirname(smart_data_path), exist_ok=True)
    pickle.dump(patient_all, open(smart_data_path, 'wb'))
    return patient_all

def prepare_mimic3_mortality_train_val_test_split(args, patient_all):
    from data.splits import fixed_train_val_test_split
    return fixed_train_val_test_split(args, patient_all, 'mimic3_mortality')


def prepare(args):
    patient_all = prepare_mimic3_mortality_extract_from_raw(args)

    if args.method == "ehr_model_smart" or args.method == "llm_smart_embedding_topk" or args.method == "graph_walker" or args.embedding_model_name == 'smart':
        patient_all = prepare_mimic3_mortality_for_smart(args, patient_all)

    train_data, val_data, test_data = prepare_mimic3_mortality_train_val_test_split(args, patient_all)
    return train_data, val_data, test_data
