import os
import pickle
import numpy as np
import json
import re
from typing import List, Dict, Any
import pickle
import random
from tqdm import tqdm


class CustomBins:
    """Four LOS classes with boundaries at 3, 7 and 14 days."""
    nbins = 4


def get_bin_custom(x, nbins=4, one_hot=False):
    from data.labels import los_bin_from_hours
    if nbins != 4:
        raise ValueError('LOS classification uses four classes')
    index = los_bin_from_hours(x)
    return np.eye(4)[index] if one_hot else index


def prepare_mimic3_los_extract_from_raw(args):
    """Load observation windows and total hospital LOS labels."""
    import csv
    from pathlib import Path
    from data.labels import hospital_los_label
    from data.mimic3.reader import LengthOfStayReader, read_chunk

    metadata_path = Path(getattr(args, 'hospital_los_path', None) or
                         Path(args.dataset_path) / 'hospital_los.csv')
    if not metadata_path.is_file():
        raise FileNotFoundError(f'Hospital LOS metadata not found: {metadata_path}. Set --hospital_los_path.')
    labels = {}
    with metadata_path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            name = row['name']
            if name in labels:
                raise ValueError(f'Duplicate LOS metadata for {name}')
            labels[name] = hospital_los_label(row)
    patient_all = {key: [] for key in ('X', 't', 'y', 'header', 'name')}
    for mode in ('train', 'val', 'test'):
        reader = LengthOfStayReader(
            dataset_dir=os.path.join(args.dataset_path, 'test' if mode == 'test' else 'train'),
            listfile=os.path.join(args.dataset_path, mode + '_listfile.csv'))
        count = reader.get_number_of_examples()
        if not count:
            continue
        records = read_chunk(reader, count)
        for key in ('X', 't', 'header', 'name'):
            patient_all[key].extend(records[key])
        for name in records['name']:
            if name not in labels:
                raise ValueError(f'Missing hospital LOS metadata for {name}')
            patient_all['y'].append(labels[name])
    return patient_all


def prepare_mimic3_los_for_smart(args, patient_all):
    '''
    This function is to adapt the raw data for SMART model
    '''
    period_length = args.period_length
    # Store adapted inputs by observation window and LOS class count.
    smart_data_path = args.mid_data_dump_path + f'/mimic3_los/mimic3_los_smart_period{period_length}_binned_{CustomBins.nbins}classes.pkl'
    print('Adapting mimic3 length of stay data for SMART model...')
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
    for patient, name, t in tqdm(zip(patient_all['X'], patient_all['name'], patient_all['t']), total=len(patient_all['X']), desc="processing patients", unit="patient"):
        N_bins = min(int(t + 1 - 1e-6), period_length)
        data_patient = np.zeros(shape=(len(id_to_channel), N_bins), dtype=np.float32)
        mask_patient = np.zeros(shape=(len(id_to_channel), N_bins), dtype=np.float32)
        last_time = -1
        for row in patient:
            time = int(float(row[0]))
            if time == N_bins:
                time -= 1
            if time > N_bins:
                # raise ValueError('This should not happen')
                break
            for index in range(len(row) - 1):
                value = row[index + 1]
                if value == '':
                    if mask_patient[index, time] == 0 and time - last_time > 0:
                        # if last_time >= 0:
                        #     data_patient[index, last_time + 1:time + 1] = data_patient[index, last_time]
                        # else:
                        if is_categorical_channel[id_to_channel[index]]:
                            data_patient[index, last_time + 1:time + 1] = series_channel_info[id_to_channel[index]]['values'][normal_values[id_to_channel[index]]]
                        else:
                            data_patient[index, last_time + 1:time + 1] = float(normal_values[id_to_channel[index]])
                else:
                    mask_patient[index, time] += 1
                    if is_categorical_channel[id_to_channel[index]]:
                        data_patient[index, time] += series_channel_info[id_to_channel[index]]['values'][value]
                    else:
                        data_patient[index, time] += float(value)
            last_time = time
        data_patient = np.where(mask_patient > 0, data_patient / mask_patient, data_patient)
        mask_patient = np.where(mask_patient > 0, 1, 0)
        data_all.append(data_patient.transpose(-1, -2))
        mask_all.append(mask_patient.transpose(-1, -2))

    from data.smart_preprocessing import standardize_smart_data
    data_normalized = standardize_smart_data(args, patient_all, data_all, mask_all, 'mimic3_los')
    mask_all = [mask.tolist() for mask in mask_all]
    label_all = patient_all['y']
    name_all = patient_all['name']

    # NOTE: data_smart is a list of dicts, each dict corresponds to a patient
    x_len = [len(i) for i in data_normalized]
    for idx in range(len(patient_all['X'])):
        data_smart.append({
            "x": data_normalized[idx],
            "labels": label_all[idx],  # Use binned labels
            "lens": x_len[idx],
            "mask": mask_all[idx],
        })
    patient_all['data_smart'] = data_smart
    os.makedirs(os.path.dirname(smart_data_path), exist_ok=True)
    pickle.dump(patient_all, open(smart_data_path, 'wb'))
    return patient_all

def prepare_mimic3_los_train_val_test_split(args, patient_all):
    from data.splits import fixed_train_val_test_split
    return fixed_train_val_test_split(args, patient_all, 'mimic3_los')


def prepare(args):
    patient_all = prepare_mimic3_los_extract_from_raw(args)

    if args.method == "ehr_model_smart" or args.method == "llm_smart_embedding_topk" or args.method == "graph_walker":
        patient_all = prepare_mimic3_los_for_smart(args, patient_all)

    train_data, val_data, test_data = prepare_mimic3_los_train_val_test_split(args, patient_all)
    return train_data, val_data, test_data
