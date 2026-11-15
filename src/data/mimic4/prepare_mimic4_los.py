from data.smart_preprocessing import processed_feature_header
from data.labels import hospital_los_label
from data.splits import processed_patient_id
import os
import pickle
import numpy as np
import random

def prepare_mimic4_los_extract_from_raw(args):
    period_length = args.period_length
    raw_data_path = args.mid_data_dump_path + f'/mimic4_los/mimic4_los_raw_period{period_length}.pkl'
    base_path = os.path.join(args.dataset_path, 'processed', 'split')
    train_path = os.path.join(base_path, 'train_data.pkl')
    val_path = os.path.join(base_path, 'val_data.pkl')
    test_path = os.path.join(base_path, 'test_data.pkl')

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        raise FileNotFoundError(f"Missing processed split files under {base_path}")

    train_list = pickle.load(open(train_path, 'rb'))
    val_list = pickle.load(open(val_path, 'rb'))
    test_list = pickle.load(open(test_path, 'rb'))

    lab_features = None
    lab_feat_path = os.path.join(base_path, 'labtest_features.pkl')
    if os.path.exists(lab_feat_path):
        try:
            lab_features = pickle.load(open(lab_feat_path, 'rb'))
            if not isinstance(lab_features, (list, tuple)) or len(lab_features) == 0:
                lab_features = None
        except Exception:
            lab_features = None


    patient_all = { 'X': [], 'X_ts': [], 't': [], 'y': [], 'header': [], 'name': [], 'subject_id': [], 'missing_mask': [] }

    for split_name, data_list in [('train', train_list), ('val', val_list), ('test', test_list)]:
        for item in data_list:
            x = np.array(item['x_llm_ts'])
            header = processed_feature_header(x, lab_features)
            if patient_all['header'] and header != patient_all['header'][0]:
                raise ValueError('LLM feature schema differs across samples')
            x_ts = np.array(item['x_ts'])
            record_time = item.get('record_time', list(range(len(x))))
            y = hospital_los_label(item)
            pid = item['id']
            subject_id = processed_patient_id(item, getattr(args, 'patient_id_field', None))
            missing_mask = item['missing_mask']

            if len(record_time) != x.shape[0]:
                record_time = record_time[:x.shape[0]]


            patient_all['X'].append(x)
            patient_all['X_ts'].append(x_ts)
            patient_all['t'].append(len(record_time))
            patient_all['y'].append(y)
            patient_all['header'].append(header)
            patient_all['name'].append(pid)
            patient_all['subject_id'].append(subject_id)
            patient_all['missing_mask'].append(missing_mask)

    if not os.path.exists(raw_data_path):
        os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    pickle.dump(patient_all, open(raw_data_path, 'wb'))

    return patient_all

def prepare_mimic4_los_for_smart(args, patient_all):
    period_length = args.period_length
    smart_data_path = args.mid_data_dump_path + f'/mimic4_los/mimic4_los_smart_period{period_length}.pkl'
    print('Adapting mimic4 los data for SMART model...')

    expected_input_dim = getattr(args, 'smart_input_dim', None)

    data_smart = []
    for idx in range(len(patient_all['X'])):
        X_ts = patient_all['X_ts'][idx]

        if isinstance(X_ts, list):
            X_ts = np.array(X_ts)

        if X_ts.ndim != 2 or not all(X_ts.shape):
            raise ValueError(f"X_ts[{idx}] 应该是 2D 数组，但得到形状: {X_ts.shape}")

        if np.isinf(X_ts).any():
            raise ValueError('SMART features cannot contain infinite values')
        time_steps, feature_dim = X_ts.shape

        if expected_input_dim is None:
            expected_input_dim = feature_dim
        if feature_dim != expected_input_dim:
            raise ValueError(
                f"特征维度不匹配！期望 {expected_input_dim}，但得到 {feature_dim}。"
                f"请检查 X_ts[{idx}] 的形状: {X_ts.shape}"
            )

        if X_ts.dtype.kind == 'f':
            mask = (~np.isnan(X_ts)).astype(int)
        else:
            mask = np.ones_like(X_ts, dtype=int)

        x_list = np.where(mask, X_ts, 0).tolist()
        mask_list = mask.tolist()

        mask_list = [[int(m) for m in row] for row in mask_list]

        sample = {
            'x': x_list,
            'labels': patient_all['y'][idx],
            'lens': time_steps,
            'mask': mask_list,
        }
        data_smart.append(sample)

    patient_all['data_smart'] = data_smart

    pickle.dump(patient_all, open(smart_data_path, 'wb'))
    print(f'Successfully adapted {len(data_smart)} samples for SMART model')
    return patient_all

def prepare_mimic4_los_train_val_test_split(args, patient_all):
    from data.splits import fixed_train_val_test_split
    return fixed_train_val_test_split(args, patient_all, 'mimic4_los')


def prepare(args):
    patient_all = prepare_mimic4_los_extract_from_raw(args)
    if args.method == "ehr_model_smart" or args.embedding_model_name == 'smart':
        patient_all = prepare_mimic4_los_for_smart(args, patient_all)
    train_data, val_data, test_data = prepare_mimic4_los_train_val_test_split(args, patient_all)
    return train_data, val_data, test_data

