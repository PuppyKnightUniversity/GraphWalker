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
    """Custom bins for LOS classification based on days"""
    inf = 1e18
    # bins定义的是天数范围，实际使用时转换为小时
    # 第一类：< 3天
    # 第二类：3-7天
    # 第三类：7-14天
    # 第四类：> 14天
    bins = [(-inf, 3), (3, 7), (7, 14), (14, +inf)]
    nbins = len(bins)


def get_bin_custom(x, nbins=10, one_hot=False):
    """
    Convert continuous LOS value (in hours) to bin index
    
    Args:
        x: LOS value in hours (float)
        nbins: number of bins (default 10, 实际使用 CustomBins.nbins)
        one_hot: if True, return one-hot vector; else return bin index
    
    Returns:
        bin index (int) or one-hot vector (np.array)
    """
    # 将小时转换为天数进行比较
    x_days = x / 24.0
    
    # 使用 CustomBins.nbins 而不是传入的 nbins 参数
    actual_nbins = CustomBins.nbins
    
    for i in range(actual_nbins):
        a = CustomBins.bins[i][0]
        b = CustomBins.bins[i][1]
        # 处理边界情况
        if a == -CustomBins.inf:
            if x_days < b:
                if one_hot:
                    ret = np.zeros((actual_nbins,))
                    ret[i] = 1
                    return ret
                return i
        elif b == CustomBins.inf:
            if x_days >= a:
                if one_hot:
                    ret = np.zeros((actual_nbins,))
                    ret[i] = 1
                    return ret
                return i
        else:
            if a <= x_days < b:
                if one_hot:
                    ret = np.zeros((actual_nbins,))
                    ret[i] = 1
                    return ret
                return i
    
    # 如果超出范围，返回最后一个bin
    if one_hot:
        ret = np.zeros((actual_nbins,))
        ret[actual_nbins - 1] = 1
        return ret
    return actual_nbins - 1

def prepare_mimic3_los_extract_from_raw(args):
    '''
        Extract mimic3 length of stay data from raw data
    '''
    period_length = args.period_length
    # Include period_length in cache file path to avoid cache conflicts
    raw_data_path = args.mid_data_dump_path + f'/mimic3_los/mimic3_los_raw_period{period_length}.pkl'
    if os.path.exists(raw_data_path):
        print('Loading mimic3 los data from directory: ', raw_data_path)
        patient_all = pickle.load(open(raw_data_path, 'rb'))
        return patient_all
    else:
        # NOTE: it still have some problems about the data path, we need to fix it later
        # FIXME: check the data path
        print('Extracting mimic3 length of stay data from raw data path: ', args.dataset_path)
        path = args.dataset_path
        print(f"period_length: {period_length}")

        patient_all = {}
        patient_all['X'] = []
        patient_all['t'] = []
        patient_all['y'] = []
        patient_all['header'] = []
        patient_all['name'] = []

        from data.mimic3.reader import LengthOfStayReader, read_chunk
        for mode in ['train', 'val', 'test']:
            # read data from raw mimic3 data
            reader = LengthOfStayReader(dataset_dir=os.path.join(path, 'train' if mode != 'test' else 'test'),
                    listfile=os.path.join(path, mode + '_listfile.csv'),)
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
        
        # Apply CustomBins binning to labels (convert continuous hours to discrete bins)
        print("Applying CustomBins binning to LOS labels...")
        y_binned = []
        for label in patient_all['y']:
            bin_idx = get_bin_custom(label, nbins=CustomBins.nbins, one_hot=False)
            y_binned.append(int(bin_idx))
        patient_all['y'] = y_binned  # 直接用binned标签覆盖原来的连续值
        print(f"Label distribution after binning: {np.bincount(patient_all['y'])}")
        
        # dump the raw data
        if not os.path.exists(raw_data_path):
            os.makedirs(os.path.dirname(raw_data_path))
        pickle.dump(patient_all, open(raw_data_path, 'wb'))
        
        return patient_all

def prepare_mimic3_mortality_wrap_prompt(args, patient_all):
    '''
        Wrap prompt for mimic3 mortality data
    '''
    def format_input(patient_data: np.ndarray, features: List[str], mask: np.ndarray) -> str:
        feature_values = {}
        # Define some categorical features with their possible values
        categorical_features_dict = {
            "Glascow coma scale eye opening": {
                1: "No Response",
                2: "To Pain",
                3: "To Speech",
                4: "Spontaneously",
            },
            "Glascow coma scale motor response": {
                1: "No Response",
                2: "Abnormal Extension",
                3: "Abnormal Flexion",
                4: "Flex-withdraws",
                5: "Localizes Pain",
                6: "Obeys Commands",
            },
            "Glascow coma scale verbal response": {
                1: "No Response",
                2: "Incomprehensible sounds",
                3: "Inappropriate Words",
                4: "Confused",
                5: "Oriented",
            },
        }

        for i, feature in enumerate(features):
            values = []
            for visit_idx in range(patient_data.shape[0]):
                if mask[visit_idx, i] == 1:
                    values.append('NaN')
                else:
                    value = patient_data[visit_idx, i]
                    if feature in categorical_features_dict:
                        if not np.isnan(value):
                            values.append(categorical_features_dict[feature].get(int(value), str(value)))
                        else:
                            values.append('NaN')
                    else:
                        values.append(f"{value}")
            feature_values[feature] = f"[{', '.join(values)}]"

        detail = [f"- {feature}: {values_str}" for feature, values_str in feature_values.items()]
        
        return "\n".join(detail)


    def prepare_prompt_for_patient_example(patient_example: Dict[str, Any]) -> str:
        # TODO: 需要修改成适配 LOS 的 prompt
        def extract_leading_number(s):
            """
            从字符串中提取开头的数字
            """
            s = str(s)
            match = re.match(r'^\s*(-?\d+\.?\d*)', s)
            if match:
                return match.group(1) 
            return np.nan 
        
        vectorized_extract = np.vectorize(extract_leading_number)

        X = patient_example['X']
        header = patient_example['header']
        record_times = X[:, 0].astype(str)
        feature_data = X[:, 1:].astype(str)
        feature_names = header[1:]
        mask = (feature_data == '')
        
        numeric_data = np.full(feature_data.shape, np.nan, dtype=float)
        non_missing_indices = ~mask
        data_to_clean = feature_data[non_missing_indices]
        
        cleaned_data_str = vectorized_extract(data_to_clean)  
        cleaned_data_float = cleaned_data_str.astype(float)
        
        numeric_data[non_missing_indices] = cleaned_data_float

        detail = format_input(
            patient_data=numeric_data,
            features=feature_names,
            mask=mask
        )
        
        prompt_template = """\
    I will provide you with longitudinal medical information for a patient. Each clinical feature is presented as a list of values, corresponding to these visits. Missing values are represented as `NaN`. 

    PATIENT INFORMATION:
    - Number of measurements: {LENGTH}
    - Measurement times (hours from admission): [{RECORD_TIME_LIST}]

    Your task:
    Your primary task is to analyze the medical data to predict the length of stay (LOS) in the hospital. The LOS is defined as the number of days from admission to discharge, including any days spent in the ICU.

    Now, please analyze and predict for the following patient:

    Clinical Features Over Time:
    {DETAIL}"""

        prompt = prompt_template.format(
            LENGTH=len(record_times),
            RECORD_TIME_LIST=', '.join([f"{float(t):.2f}" for t in record_times]),
            DETAIL=detail
        )
        return prompt
    
    prompt_data_path = args.mid_data_dump_path + '/mortality_mimic3/mortality_mimic3_prompt.pkl'
    if os.path.exists(prompt_data_path):
        print('Loading wrapped prompt for mimic3 mortality data...')
        patient_all = pickle.load(open(prompt_data_path, 'rb'))
        return patient_all
    else:
        print('Wrapping prompt for mimic3 mortality data...')
        patient_num = len(patient_all['X'])
        # list of prompts for all patients
        prompt_all = []
        for i in tqdm(range(patient_num), desc="processing patients", unit="patient"):
            patient_example = {}
            patient_example['X'] = patient_all['X'][i]
            patient_example['t'] = patient_all['t'][i]
            patient_example['y'] = patient_all['y'][i]
            patient_example['header'] = patient_all['header'][i]
            patient_example['name'] = patient_all['name'][i]
            prompt = prepare_prompt_for_patient_example(patient_example)
            prompt_all.append(prompt)
        patient_all['data_prompt_fomat'] = prompt_all
        if not os.path.exists(os.path.dirname(prompt_data_path)):
            os.makedirs(os.path.dirname(prompt_data_path))
        pickle.dump(patient_all, open(prompt_data_path, 'wb'))
        return patient_all

def prepare_mimic3_los_for_smart(args, patient_all):
    '''
    This function is to adapt the raw data for SMART model
    '''
    period_length = args.period_length
    # Include period_length and binning info in cache file path to avoid cache conflicts
    # Use 'binned' suffix with number of classes to indicate that labels are binned for classification
    # This ensures cache is regenerated when number of classes changes (e.g., from 10 to 3)
    smart_data_path = args.mid_data_dump_path + f'/mimic3_los/mimic3_los_smart_period{period_length}_binned_{CustomBins.nbins}classes.pkl'
    if os.path.exists(smart_data_path):
        print('Loading adapted data for SMART model...')
        patient_all = pickle.load(open(smart_data_path, 'rb'))
        return patient_all
    else:
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
        
        data_all = np.array(data_all)
        mask_all = np.array(mask_all)
        data_all_concat = np.concatenate(data_all, axis=0)
        x_masked = np.ma.masked_array(data_all_concat, np.concatenate(mask_all, axis=0) == 0)
        mean = np.mean(x_masked, 0)
        std = np.std(x_masked, 0)
        data_normalized = np.where(mask_all == 1, (data_all - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1), 0)
        data_normalized = data_normalized.tolist()
        mask_all = mask_all.tolist()
        # 使用已经在prepare_mimic3_los_extract_from_raw中计算好的y（已经是binned标签）
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
        pickle.dump(patient_all, open(smart_data_path, 'wb'))
        return patient_all

def prepare_mimic3_los_train_val_test_split(args, patient_all):
    '''
        Split the data into train, val, and test sets
    '''
    period_length = args.period_length
    # Include period_length in cache file path to avoid cache conflicts
    split_data_path = args.mid_data_dump_path + f'/mimic3_los/seed{args.seed}_period{period_length}'
    train_file = split_data_path + '/mimic3_los_train.pkl'
    val_file = split_data_path + '/mimic3_los_val.pkl'
    test_file = split_data_path + '/mimic3_los_test.pkl'
    
    if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
        train_data = pickle.load(open(train_file, 'rb'))
        val_data = pickle.load(open(val_file, 'rb'))
        test_data = pickle.load(open(test_file, 'rb'))
        
        # Print sample sizes for loaded train/val/test sets
        print(f"Loaded existing data split:")
        print(f"  Train samples: {len(train_data['X'])}")
        print(f"  Val samples: {len(val_data['X'])}")
        print(f"  Test samples: {len(test_data['X'])}")
        print(f"  Total samples: {len(train_data['X']) + len(val_data['X']) + len(test_data['X'])}")
        
        return train_data, val_data, test_data
    else:
        print('Splitting mimic3 length of stay data into train, val, and test sets...')
        train_ratio = args.train_ratio
        dump_path = args.mid_data_dump_path + '/mimic3_los/seed' + str(args.seed)
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)
        
        patient_index = list(range(len(patient_all['X'])))
        # Set random seed for reproducibility
        random.seed(args.seed)
        random.shuffle(patient_index)
        x_len = [len(i) for i in patient_all['X']]

        train_num = int(len(patient_all['X']) * train_ratio)
        val_num = int(len(patient_all['X']) * ((1 - train_ratio) / 2))
        test_num = len(patient_all['X']) - train_num - val_num

        train_data = {}
        for idx in patient_index[: train_num]:
            for key in patient_all.keys():
                if key not in train_data:
                    train_data[key] = []
                train_data[key].append(patient_all[key][idx])
        pickle.dump(train_data, open(dump_path + '/mimic3_los_train.pkl', 'wb'))

        val_data = {}
        for idx in patient_index[train_num: train_num + val_num]:
            for key in patient_all.keys():
                if key not in val_data:
                    val_data[key] = []
                val_data[key].append(patient_all[key][idx])
        pickle.dump(val_data, open(dump_path + '/mimic3_los_val.pkl', 'wb'))

        test_data = {}
        for idx in patient_index[train_num + val_num:]:
            for key in patient_all.keys():
                if key not in test_data:
                    test_data[key] = []
                test_data[key].append(patient_all[key][idx])
        pickle.dump(test_data, open(dump_path + '/mimic3_los_test.pkl', 'wb'))

        # Print sample sizes for train/val/test sets
        print(f"Data split completed:")
        print(f"  Train samples: {len(train_data['X'])}")
        print(f"  Val samples: {len(val_data['X'])}")
        print(f"  Test samples: {len(test_data['X'])}")
        print(f"  Total samples: {len(train_data['X']) + len(val_data['X']) + len(test_data['X'])}")

        return train_data, val_data, test_data

def prepare(args):
    patient_all = prepare_mimic3_los_extract_from_raw(args)
    # patient_all = prepare_mimic3_mortality_wrap_prompt(args, patient_all)

    if args.method == "ehr_model_smart" or args.method == "llm_smart_embedding_topk" or args.method == "graph_walker":
        patient_all = prepare_mimic3_los_for_smart(args, patient_all)
        
    train_data, val_data, test_data = prepare_mimic3_los_train_val_test_split(args, patient_all)
    return train_data, val_data, test_data