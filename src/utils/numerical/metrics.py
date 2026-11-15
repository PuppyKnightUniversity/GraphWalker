import numpy as np
import torch
from sklearn import metrics as sklearn_metrics


def get_all_metrics(preds, labels, task, los_info):
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    if task in ["mortality", "readmission"]:
        if len(labels.shape) > 1 and labels.shape[-1] > 1:
            labels = labels[:, 0] if task == "mortality" else labels[:, 2]
        return get_binary_metrics(preds, labels)
    elif task == "los":
        if len(labels.shape) > 1 and labels.shape[-1] > 1:
            labels = labels[:, 1]
        return get_regression_metrics(reverse_los(preds, los_info), reverse_los(labels, los_info))
    elif task == "multiclass":
        return get_multiclass_metrics(preds, labels)
    elif task == "multitask":
        return get_binary_metrics(preds[:, 0], labels[:, 0]) | get_regression_metrics(
            reverse_los(preds[:, 1], los_info), reverse_los(labels[:, 1], los_info)
        )
    else:
        raise ValueError("Task not supported")


def get_all_metrics_with_bootstrap(preds, labels, task, los_info, n_bootstrap=1000, confidence_level=0.95, random_state=42):
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    if task in ["mortality", "readmission"]:
        if len(labels.shape) > 1 and labels.shape[-1] > 1:
            labels = labels[:, 0] if task == "mortality" else labels[:, 2]
        return get_binary_metrics_with_bootstrap(preds, labels, n_bootstrap, confidence_level, random_state)
    elif task == "los":
        if len(labels.shape) > 1 and labels.shape[-1] > 1:
            labels = labels[:, 1]
        return get_regression_metrics_with_bootstrap(reverse_los(preds, los_info), reverse_los(labels, los_info), n_bootstrap, confidence_level, random_state)
    elif task == "multiclass":
        return get_multiclass_metrics_with_bootstrap(preds, labels, n_bootstrap, confidence_level, random_state)
    elif task == "multitask":
        binary_res = get_binary_metrics_with_bootstrap(preds[:, 0], labels[:, 0], n_bootstrap, confidence_level, random_state)
        regression_res = get_regression_metrics_with_bootstrap(
            reverse_los(preds[:, 1], los_info), reverse_los(labels[:, 1], los_info), n_bootstrap, confidence_level, random_state
        )
        return {**binary_res, **regression_res}
    else:
        raise ValueError("Task not supported")


def reverse_los(y, los_info):
    if los_info is None:
        return y
    return y * los_info.get("los_std", 1.0) + los_info.get("los_mean", 0.0)


def minpse(preds, labels):
    precisions, recalls, _ = sklearn_metrics.precision_recall_curve(labels, preds)
    minpse_score = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    return minpse_score


def get_binary_metrics(preds, labels):
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    labels = labels.astype(int)
    
    try:
        auroc = sklearn_metrics.roc_auc_score(labels, preds)
    except ValueError:
        auroc = 0.0
        
    try:
        auprc = sklearn_metrics.average_precision_score(labels, preds)
    except ValueError:
        auprc = 0.0
        
    minpse_score = minpse(preds, labels)
    
    preds_binary = (preds > 0.5).astype(int)
    f1 = sklearn_metrics.f1_score(labels, preds_binary)
    accuracy = sklearn_metrics.accuracy_score(labels, preds_binary)

    precisions, recalls, thresholds = sklearn_metrics.precision_recall_curve(labels, preds)
    numerator = 2 * precisions * recalls
    denominator = precisions + recalls
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = np.where(denominator > 0, numerator / denominator, 0.0)
    best_f1 = np.max(f1_scores) if len(f1_scores) > 0 else 0.0

    return {
        "auroc": auroc,
        "auprc": auprc,
        "minpse": minpse_score,
        "f1": f1,
        "best_f1": best_f1,
        "accuracy": accuracy,
    }


def get_binary_metrics_with_bootstrap(preds, labels, n_bootstrap=1000, confidence_level=0.95, random_state=42):
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    labels = labels.astype(int)
    
    original_metrics = get_binary_metrics(preds, labels)
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(preds)
    bootstrap_results = {k: [] for k in original_metrics.keys()}
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_preds = preds[indices]
        boot_labels = labels[indices]
        
        if len(np.unique(boot_labels)) < 2:
            continue
            
        boot_metrics = get_binary_metrics(boot_preds, boot_labels)
        for k, v in boot_metrics.items():
            bootstrap_results[k].append(v)
            
    alpha = 1 - confidence_level
    percentile_lower = (alpha / 2) * 100
    percentile_upper = (1 - alpha / 2) * 100
    
    final_results = {}
    for metric_name, values in bootstrap_results.items():
        if not values:
            final_results[metric_name] = {
                'value': original_metrics[metric_name],
                'mean': np.nan, 'std': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan
            }
            continue
            
        values = np.array(values)
        final_results[metric_name] = {
            'value': original_metrics[metric_name],
            'mean': np.mean(values),
            'std': np.std(values),
            'ci_lower': np.percentile(values, percentile_lower),
            'ci_upper': np.percentile(values, percentile_upper)
        }
        
    return final_results


def get_regression_metrics(preds, labels):
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    mse = sklearn_metrics.mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    mae = sklearn_metrics.mean_absolute_error(labels, preds)
    r2 = sklearn_metrics.r2_score(labels, preds)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def get_regression_metrics_with_bootstrap(preds, labels, n_bootstrap=1000, confidence_level=0.95, random_state=42):
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    original_metrics = get_regression_metrics(preds, labels)
    
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = len(preds)
    bootstrap_results = {k: [] for k in original_metrics.keys()}
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_preds = preds[indices]
        boot_labels = labels[indices]
        
        boot_metrics = get_regression_metrics(boot_preds, boot_labels)
        for k, v in boot_metrics.items():
            bootstrap_results[k].append(v)

    alpha = 1 - confidence_level
    percentile_lower = (alpha / 2) * 100
    percentile_upper = (1 - alpha / 2) * 100
    
    final_results = {}
    for metric_name, values in bootstrap_results.items():
        if not values:
             final_results[metric_name] = {
                'value': original_metrics[metric_name],
                'mean': np.nan, 'std': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan
            }
             continue
             
        values = np.array(values)
        final_results[metric_name] = {
            'value': original_metrics[metric_name],
            'mean': np.mean(values),
            'std': np.std(values),
            'ci_lower': np.percentile(values, percentile_lower),
            'ci_upper': np.percentile(values, percentile_upper)
        }
        
    return final_results


def get_multiclass_metrics(preds, labels):
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    labels = labels.astype(int)
    
    if not np.allclose(np.sum(preds, axis=1), 1.0, atol=1e-5):
        from scipy.special import softmax
        preds = softmax(preds, axis=1)

    try:
        ma_roc = sklearn_metrics.roc_auc_score(labels, preds, average='macro', multi_class='ovr')
    except ValueError:
        ma_roc = 0.0
        
    try:
        mi_roc = sklearn_metrics.roc_auc_score(labels, preds, average='micro', multi_class='ovr')
    except ValueError:
        mi_roc = 0.0
        
    pred_classes = np.argmax(preds, axis=1)
    accuracy = sklearn_metrics.accuracy_score(labels, pred_classes)
    f1_macro = sklearn_metrics.f1_score(labels, pred_classes, average='macro')
    f1_micro = sklearn_metrics.f1_score(labels, pred_classes, average='micro')

    return {
        "ma-ROC": ma_roc,
        "mi-ROC": mi_roc,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro
    }


def get_multiclass_metrics_with_bootstrap(preds, labels, n_bootstrap=1000, confidence_level=0.95, random_state=42):
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    labels = labels.astype(int)
    
    original_metrics = get_multiclass_metrics(preds, labels)
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(preds)
    bootstrap_results = {k: [] for k in original_metrics.keys()}
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_preds = preds[indices]
        boot_labels = labels[indices]
        
        if len(np.unique(boot_labels)) < 2:
            continue
            
        boot_metrics = get_multiclass_metrics(boot_preds, boot_labels)
        for k, v in boot_metrics.items():
            bootstrap_results[k].append(v)
            
    alpha = 1 - confidence_level
    percentile_lower = (alpha / 2) * 100
    percentile_upper = (1 - alpha / 2) * 100
    
    final_results = {}
    for metric_name, values in bootstrap_results.items():
        if not values:
            final_results[metric_name] = {
                'value': original_metrics[metric_name],
                'mean': np.nan, 'std': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan
            }
            continue
            
        values = np.array(values)
        final_results[metric_name] = {
            'value': original_metrics[metric_name],
            'mean': np.mean(values),
            'std': np.std(values),
            'ci_lower': np.percentile(values, percentile_lower),
            'ci_upper': np.percentile(values, percentile_upper)
        }
        
    return final_results


def check_metric_is_better(cur_best, main_metric, score, task):
    if task == "los":
        if cur_best == {}:
            return True
        if score < cur_best[main_metric]:
            return True
        return False
    elif task == "multiclass":
        if cur_best == {}:
            return True
        if score > cur_best[main_metric]:
            return True
        return False
    elif task in ["mortality", "readmission", "multitask"]:
        if cur_best == {}:
            return True
        if score > cur_best[main_metric]:
            return True
        return False
    else:
        raise ValueError("Task not supported")
