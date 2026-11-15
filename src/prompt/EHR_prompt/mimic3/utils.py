from typing import Dict, Any
import numpy as np

def mimic3_smooth_hourly_data(patient_example: Dict[str, Any], keep_last: bool = True) -> Dict[str, Any]:
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
            selected_idx, selected_time, selected_features = hour_data[-1]
        else:
            selected_idx, selected_time, selected_features = hour_data[0]
        
        smoothed_indices.append(selected_idx)
        smoothed_times.append(selected_time)
        smoothed_features.append(selected_features)
    
    smoothed_X = np.column_stack([
        np.array(smoothed_times).reshape(-1, 1),
        np.array(smoothed_features)
    ])
    
    smoothed_patient_example = patient_example.copy()
    smoothed_patient_example['X'] = smoothed_X
    
    return smoothed_patient_example