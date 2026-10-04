"""Train-fitted scaling for masked time series."""
import numpy as np

from data.splits import patient_split_indices


def standardize_smart_data(args, patient_all, values, masks, dataset):
    _, groups = patient_split_indices(args, patient_all, dataset)
    train = np.concatenate([values[i] for i in groups[0]])
    observed = np.concatenate([masks[i] for i in groups[0]]).astype(bool) & np.isfinite(train)
    count = observed.sum(axis=0)
    mean = np.where(observed, train, 0).sum(axis=0) / np.maximum(count, 1)
    var = np.where(observed, (train - mean) ** 2, 0).sum(axis=0) / np.maximum(count, 1)
    scale = np.where(var > 0, np.sqrt(var), 1)
    result = []
    for value, mask in zip(values, masks):
        scaled = np.where(np.asarray(mask).astype(bool) & np.isfinite(value), (value - mean) / scale, 0)
        if not np.isfinite(scaled).all():
            raise ValueError('Nonfinite normalized SMART features')
        result.append(scaled.astype(np.float32).tolist())
    return result
