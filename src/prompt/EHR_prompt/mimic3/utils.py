from typing import Dict, Any
import numpy as np

def mimic3_smooth_hourly_data(patient_example: Dict[str, Any], keep_last: bool = True) -> Dict[str, Any]:
    '''
    Smooth the hourly data of the patient example,
    keep the last data point of each hour by default.
    if keep_last is False, keep the first data point of each hour.
    Args:
        patient_example: Dict[str, Any]
            The patient example
        keep_last: bool
            Whether to keep the last data point of each hour
    Returns:
        smoothed_patient_example: Dict[str, Any]
            The smoothed patient example
    '''
    from collections import defaultdict
    X = patient_example['X']
    record_times = X[:, 0].astype(float)
    feature_data = X[:, 1:]
    hourly_groups = defaultdict(list)

    for i, time in enumerate(record_times):
        hour = int(time)
        hourly_groups[hour].append((i, time, feature_data[i]))

    smoothed_indices = []
    smoothed_times = []
    smoothed_features = []
    
    for hour in sorted(hourly_groups.keys()):
        hour_data = hourly_groups[hour]
        
        if keep_last:
            # retain the last data point of each hour
            selected_idx, selected_time, selected_features = hour_data[-1]
        else:
            # retain the first data point of each hour
            selected_idx, selected_time, selected_features = hour_data[0]
        
        smoothed_indices.append(selected_idx)
        smoothed_times.append(selected_time)
        smoothed_features.append(selected_features)
    
    # build the smoothed data
    smoothed_X = np.column_stack([
        np.array(smoothed_times).reshape(-1, 1),
        np.array(smoothed_features)
    ])
    
    # create the smoothed patient example
    smoothed_patient_example = patient_example.copy()
    smoothed_patient_example['X'] = smoothed_X
    
    return smoothed_patient_example