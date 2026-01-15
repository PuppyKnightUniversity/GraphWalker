import os
import pickle
import numpy as np
import random

def prepare_tjh_los_extract_from_raw(args):
    '''
        Extract tjh length of stay data from raw data
    '''
    period_length = args.period_length
    # Include period_length in cache file path to avoid cache conflicts
    raw_data_path = args.mid_data_dump_path + f'/tjh_los/tjh_los_raw_period{period_length}.pkl'
    if os.path.exists(raw_data_path):
        print('Loading tjh los data from directory: ', raw_data_path)
        patient_all = pickle.load(open(raw_data_path, 'rb'))
        return patient_all
    else:
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
                # TJH dataset has 73 labtest features, not 17
                if not isinstance(lab_features, (list, tuple)) or len(lab_features) == 0:
                    lab_features = None
            except Exception:
                lab_features = None

        def build_header(x_row: np.ndarray, lab_feats: list):
            sex_val = x_row[0]
            sex_str = None
            if sex_str is None:
                sex_str = 'Sex'

            if lab_feats is not None:
                lab_headers = list(lab_feats)
            else:
                # Default to 73 features if labtest_features.pkl is not available
                # This matches the actual number of features in TJH dataset
                lab_headers = [f"labtest_features[{i}]" for i in range(73)]
            return [sex_str, 'Age'] + lab_headers

        patient_all = {
            'X': [],
            'X_ts': [],
            't': [],
            'y': [],
            'header': [],
            'name': [],
            'missing_mask': []
        }

        all_sets = [('train', train_list), ('val', val_list), ('test', test_list)]
        for split_name, data_list in all_sets:
            for item in data_list:
                x = np.array(item['x_llm_ts'])
                x_ts = np.array(item['x_ts'])
                record_time = item.get('record_time', list(range(len(x))))
                y_raw = item['y_los']
                if isinstance(y_raw, (list, tuple)):
                    y = int(y_raw[0])
                else:
                    y = int(y_raw)
                pid = item['id']
                missing_mask = item['missing_mask']

                if len(record_time) != x.shape[0]:
                    # Prefer x length; ignore mismatch by trimming/padding record_time
                    record_time = record_time[:x.shape[0]]

                header = build_header(x[0], lab_features)

                patient_all['X'].append(x)  # for LLM
                patient_all['X_ts'].append(x_ts)  # for ML model
                patient_all['t'].append(len(record_time))
                patient_all['y'].append(y)
                patient_all['header'].append(header)
                patient_all['name'].append(pid)
                patient_all['missing_mask'].append(missing_mask)

        # dump the raw data
        if not os.path.exists(raw_data_path):
            os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
        pickle.dump(patient_all, open(raw_data_path, 'wb'))
        
        print(f'Total samples extracted: {len(patient_all["X"])}')
        
        return patient_all

def prepare_tjh_los_for_smart(args, patient_all):
    """
    将 tjh_los 数据适配为 SMART 模型输入格式
    
    注意维度对齐：
    - smart_input_dim = 75 (特征维度)
    - smart_demo_dim = 2 (人口统计学特征：Sex, Age)
    - 数据格式：x 应该是 (time_steps, 75) 的 list of lists
    """
    period_length = args.period_length
    smart_data_path = args.mid_data_dump_path + f'/tjh_los/tjh_los_smart_period{period_length}.pkl'
    if os.path.exists(smart_data_path):
        print('Loading adapted data for SMART model...')
        patient_all = pickle.load(open(smart_data_path, 'rb'))
        return patient_all
    else:
        print('Adapting tjh los data for SMART model...')
        
        # SMART 模型期望的输入维度
        expected_input_dim = 75  # smart_input_dim for tjh_los
        
        data_smart = []
        for idx in range(len(patient_all['X'])):
            X_ts = patient_all['X_ts'][idx]
            
            # 转换为 numpy array 以便检查维度
            if isinstance(X_ts, list):
                X_ts = np.array(X_ts)
            
            # 维度检查和验证
            if X_ts.ndim != 2:
                raise ValueError(f"X_ts[{idx}] 应该是 2D 数组，但得到形状: {X_ts.shape}")
            
            time_steps, feature_dim = X_ts.shape
            
            # 检查特征维度是否匹配
            if feature_dim != expected_input_dim:
                raise ValueError(
                    f"特征维度不匹配！期望 {expected_input_dim}，但得到 {feature_dim}。"
                    f"请检查 X_ts[{idx}] 的形状: {X_ts.shape}"
                )
            
            # 根据 X_ts 生成对应的 mask
            # 如果 X_ts 中有 NaN，则 mask 为 0，否则为 1
            if X_ts.dtype.kind == 'f':  # 浮点类型，可能有 NaN
                mask = (~np.isnan(X_ts)).astype(int)
            else:
                # 非浮点类型，假设所有值都是有效的
                mask = np.ones_like(X_ts, dtype=int)
            
            # 转换为 list 格式（smart 的 collate_fn 期望 list）
            x_list = X_ts.tolist()
            mask_list = mask.tolist()
            
            # 确保 mask 是整数类型（0 或 1）
            mask_list = [[int(m) for m in row] for row in mask_list]
            
            sample = {
                'x': x_list,  # (time_steps, 75) 的 list of lists
                'labels': patient_all['y'][idx],
                'lens': time_steps,
                'mask': mask_list,  # (time_steps, 75) 的 list of lists，与 x 形状一致
            }
            data_smart.append(sample)
        
        patient_all['data_smart'] = data_smart
        
        # Ensure directory exists before dumping
        os.makedirs(os.path.dirname(smart_data_path), exist_ok=True)
        pickle.dump(patient_all, open(smart_data_path, 'wb'))
        print(f'Successfully adapted {len(data_smart)} samples for SMART model')
        return patient_all

def prepare_tjh_los_train_val_test_split(args, patient_all):
    dump_path = os.path.join(args.mid_data_dump_path, f"tjh_los/seed{args.seed}")
    os.makedirs(dump_path, exist_ok=True)

    train_ratio = args.train_ratio
    patient_index = list(range(len(patient_all['X'])))
    random.seed(getattr(args, 'seed', 0))
    random.shuffle(patient_index)

    N = len(patient_index)
    train_num = int(N * train_ratio)
    val_num = int(N * ((1 - train_ratio) / 2))

    train_data = {}
    for idx in patient_index[:train_num]:
        for key in patient_all.keys():
            train_data.setdefault(key, []).append(patient_all[key][idx])
    pickle.dump(train_data, open(os.path.join(dump_path, 'tjh_los_train.pkl'), 'wb'))

    val_data = {}
    for idx in patient_index[train_num:train_num + val_num]:
        for key in patient_all.keys():
            val_data.setdefault(key, []).append(patient_all[key][idx])
    pickle.dump(val_data, open(os.path.join(dump_path, 'tjh_los_val.pkl'), 'wb'))

    test_data = {}
    for idx in patient_index[train_num + val_num:]:
        for key in patient_all.keys():
            test_data.setdefault(key, []).append(patient_all[key][idx])
    pickle.dump(test_data, open(os.path.join(dump_path, 'tjh_los_test.pkl'), 'wb'))

    print(f"Data split completed:")
    print(f"  Train samples: {len(train_data['X'])}")
    print(f"  Val samples: {len(val_data['X'])}")
    print(f"  Test samples: {len(test_data['X'])}")
    print(f"  Total samples: {len(train_data['X']) + len(val_data['X']) + len(test_data['X'])}")

    return train_data, val_data, test_data

def prepare(args):
    patient_all = prepare_tjh_los_extract_from_raw(args)
    if args.method == "ehr_model_smart" or args.embedding_model_name == 'smart':
        patient_all = prepare_tjh_los_for_smart(args, patient_all)
    train_data, val_data, test_data = prepare_tjh_los_train_val_test_split(args, patient_all)
    return train_data, val_data, test_data

