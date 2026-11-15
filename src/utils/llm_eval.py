import re
import os
import json
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss,
                             confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from scipy import stats
import torch
import torch.nn.functional as F


def parse_llm_response_for_mimic3_mortality(response: str)->float:
    try:        
        response_cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        numbers = re.findall(r"(-?\d+\.?\d*)", response_cleaned)
        if len(numbers) == 0:
            print(f"No number found in the LLM response: {response[:200]}...")
            pred= -1.0
        else:
            for number in reversed(numbers):
                try:
                    pred = float(number)
                    if pred >= 0.0 and pred <= 1.0:
                        return pred
                except ValueError:
                    continue
            else:
                pred = float(numbers[0])
    except Exception as e:
        print(f"Error parsing LLM response: {response[:200]}...", f"Error: {e}")
        pred = -1.0
    return pred

def parse_llm_response_for_mimic3_los(response: str) -> str:
    if not response:
        return None

    response = response.strip().upper()
    
    match = re.match(r'^([A-D])', response)
    if match:
        return match.group(1)
    
    match = re.search(r'[\(\[（【]([A-D])[\)\]）】]', response)
    if match:
        return match.group(1)   
    match = re.search(r'[\(\[（【]([A-D])[\)\]）】]', response)
    if match:
        return match.group(1)
    
    match = re.search(r'(?:答案|选项|选择|answer|option|choice|category)[:：\s]*([A-D])', response, re.IGNORECASE)
    if match:
        return match.group(1)
    
    first_part = response[:50]
    match = re.search(r'\b([A-D])\b', first_part)
    if match:
        return match.group(1)
    
    return None 

def parse_llm_response_for_cmb_exam_patient(response: str) -> str:
    if not response:
        return None
    
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    response = response.strip().upper()
    
    match = re.match(r'^([A-Z])', response)
    if match:
        return match.group(1)
    
    match = re.search(r'[\(\[（【]([A-Z])[\)\]）】]', response)
    if match:
        return match.group(1)
    
    match = re.search(r'(?:答案|选项|选择|answer|option|choice)[:：\s]*([A-Z])', response, re.IGNORECASE)
    if match:
        return match.group(1)
    
    first_part = response[:50]
    match = re.search(r'\b([A-Z])\b', first_part)
    if match:
        return match.group(1)
    
    return None




def evaluate_model(score_list, label_list): 

    if len(score_list) != len(label_list):
        raise ValueError("Length of predicted scores and true labels must be the same")
    
    score_list = [1 - score for score in score_list]
    label_list = [1 - label for label in label_list]
    
    metrics = evaluate_binary_model(score_list, label_list)
    return metrics


def evaluate_and_save(score_list, label_list, save_path):
    metrics = evaluate_model(score_list, label_list)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def evaluate_binary_model(score_list, label_list, verbose=True, threshold=0.5):
    y_scores = np.array(score_list)
    y_true = np.array(label_list)
        
    if len(y_scores) != len(y_true):
        raise ValueError("Length of predicted scores and true labels must be the same")
    
    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)
    brier_score = brier_score_loss(y_true, y_scores)
    
    if not 0 <= threshold <= 1:
        raise ValueError("The classification threshold must be in [0, 1]")
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    best_f1 = f1_score(y_true, y_scores >= threshold, zero_division=0)
    best_threshold = threshold

    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    
    metrics = {
        'AUPRC': auprc,
        'AUROC': auroc,
        'F1 Score': best_f1,
        'MinPSE': minpse,
        'Brier Score': brier_score,
        'Threshold': best_threshold,
    }
    
    if verbose:
        print("Binary Classification Model Evaluation Results:")
        print(f"  PR AUC: {auprc:.4f}")
        print(f"  ROC AUC: {auroc:.4f}")
        print(f"  F1 Score: {best_f1:.4f}")
        print(f"  MinPSE: {minpse:.4f}")
        print(f"  Brier Score: {brier_score:.4f}")
        print(f"  Threshold: {best_threshold:.4f}")
    
    return metrics


def evaluate_model_with_threshold(score_list, label_list, threshold=0.5):

    if len(score_list) != len(label_list):
        raise ValueError("Length of predicted scores and true labels must be the same")
    
    score_list = [1 - score for score in score_list]
    label_list = [1 - label for label in label_list]
    
    metrics = evaluate_binary_model_with_threshold(score_list, label_list, threshold)
    return metrics


def evaluate_model_with_high_recall(score_list, label_list, recall_threshold=0.9): 

    if len(score_list) != len(label_list):
        raise ValueError("Length of predicted scores and true labels must be the same")
    
    score_list = [1 - score for score in score_list]
    label_list = [1 - label for label in label_list]
    
    precision_curve, recall_curve, threshold_curve = precision_recall_curve(label_list, score_list)
    
    high_recall_index = np.where(recall_curve >= recall_threshold)[0][-1]
    threshold = threshold_curve[high_recall_index]

    metrics = evaluate_binary_model_with_threshold(score_list, label_list, threshold)
    return metrics


def evaluate_binary_model_with_threshold(score_list, label_list, threshold): 
    y_scores = np.array(score_list)
    y_true = np.array(label_list)
    
    if len(y_scores) != len(y_true):
        raise ValueError("Length of predicted scores and true labels must be the same")
    
    y_pred = (y_scores >= threshold).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'PR AUC': pr_auc,
        'ROC AUC': roc_auc,
        'Confusion Matrix': {
            'True Negatives': tn,
            'False Positives': fp,
            'False Negatives': fn,
            'True Positives': tp
        },
    }
    
    print("Binary Classification Model Evaluation Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  PR AUC: {pr_auc:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    return metrics


def bootstrap_metrics(score_list, label_list, n_bootstrap=1000, confidence_level=0.95, random_state=None, threshold=0.5):
    rng = np.random.default_rng(random_state)
    
    y_scores = np.array(score_list)
    y_true = np.array(label_list)
    n_samples = len(y_scores)
    
    bootstrap_results = {
        'AUPRC': [],
        'AUROC': [],
        'F1_Score': [],
        'MinPSE': [],
        'Brier_Score': []
    }
    
    for i in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        boot_scores = y_scores[indices]
        boot_labels = y_true[indices]
        
        try:
            unique_labels = np.unique(boot_labels)
            if len(unique_labels) < 2:
                continue
            
            auroc = roc_auc_score(boot_labels, boot_scores)
            auprc = average_precision_score(boot_labels, boot_scores)
            brier_score = brier_score_loss(boot_labels, boot_scores)
            
            if np.isnan(auroc) or np.isnan(auprc) or np.isnan(brier_score):
                continue
            
            precisions, recalls, thresholds = precision_recall_curve(boot_labels, boot_scores)
            best_f1 = f1_score(boot_labels, boot_scores >= threshold, zero_division=0)
            minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
            
            if np.isnan(best_f1) or np.isnan(minpse):
                continue
            
            bootstrap_results['AUPRC'].append(auprc)
            bootstrap_results['AUROC'].append(auroc)
            bootstrap_results['F1_Score'].append(best_f1)
            bootstrap_results['MinPSE'].append(minpse)
            bootstrap_results['Brier_Score'].append(brier_score)
            
        except (ValueError, ZeroDivisionError):
            continue
    
    alpha = 1 - confidence_level
    percentile_lower = (alpha / 2) * 100
    percentile_upper = (1 - alpha / 2) * 100
    
    bootstrap_stats = {}
    for metric_name, values in bootstrap_results.items():
        if len(values) > 0:
            values = np.array(values)
            values = values[~np.isnan(values)]
            
            if len(values) > 0:
                bootstrap_stats[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'ci_lower': np.percentile(values, percentile_lower),
                    'ci_upper': np.percentile(values, percentile_upper),
                    'n_bootstrap': len(values)
                }
            else:
                bootstrap_stats[metric_name] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'median': np.nan,
                    'ci_lower': np.nan,
                    'ci_upper': np.nan,
                    'n_bootstrap': 0
                }
    
    return bootstrap_stats


def evaluate_binary_model_with_bootstrap(score_list, label_list, n_bootstrap=1000, confidence_level=0.95, random_state=None, logger=None, threshold=0.5):
    original_metrics = evaluate_binary_model(score_list, label_list, verbose=False, threshold=threshold)
    
    bootstrap_stats = bootstrap_metrics(score_list, label_list, n_bootstrap, confidence_level, random_state, threshold=threshold)
    
    combined_results = {}
    for metric_name in ['AUPRC', 'AUROC', 'F1 Score', 'MinPSE', 'Brier Score']:
        bootstrap_key = metric_name.replace(' ', '_')
        if bootstrap_key in bootstrap_stats:
            combined_results[metric_name] = {
                'value': original_metrics[metric_name],
                'bootstrap_mean': bootstrap_stats[bootstrap_key]['mean'],
                'bootstrap_std': bootstrap_stats[bootstrap_key]['std'],
                'bootstrap_median': bootstrap_stats[bootstrap_key]['median'],
                'ci_lower': bootstrap_stats[bootstrap_key]['ci_lower'],
                'ci_upper': bootstrap_stats[bootstrap_key]['ci_upper'],
                'n_bootstrap': bootstrap_stats[bootstrap_key]['n_bootstrap']
            }
    
    combined_results['Threshold'] = original_metrics['Threshold']
    
    print(f"Bootstrap samples: {n_bootstrap}, Confidence level: {confidence_level:.1%}")
    
    if logger:
        logger.info("=== Bootstrap Evaluation Results ===")
        logger.info(f"Bootstrap samples: {n_bootstrap}, Confidence level: {confidence_level:.1%}")
        
        for metric_name, metric_stats in combined_results.items():
            if metric_name != 'Threshold':
                original_val = metric_stats['value']
                bootstrap_mean = metric_stats['bootstrap_mean']
                bootstrap_std = metric_stats['bootstrap_std']
                ci_lower = metric_stats['ci_lower']
                ci_upper = metric_stats['ci_upper']
                
                logger.info(f"{metric_name}:")
                logger.info(f"  Original value: {original_val:.4f}")
                logger.info(f"  Bootstrap mean: {bootstrap_mean:.4f} ± {bootstrap_std:.4f}")
                logger.info(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            else:
                logger.info(f"{metric_name}: {metric_stats:.4f}")
    
    return combined_results


def bootstrap_accuracy(pred_list, label_list, n_bootstrap=1000, confidence_level=0.95, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    y_pred = np.array(pred_list)
    y_true = np.array(label_list)
    n_samples = len(y_pred)
    
    original_accuracy = accuracy_score(y_true, y_pred)
    
    bootstrap_accuracies = []
    
    for i in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_pred = y_pred[indices]
        boot_labels = y_true[indices]
        
        try:
            boot_accuracy = accuracy_score(boot_labels, boot_pred)
            if not np.isnan(boot_accuracy):
                bootstrap_accuracies.append(boot_accuracy)
        except (ValueError, ZeroDivisionError):
            continue
    
    alpha = 1 - confidence_level
    percentile_lower = (alpha / 2) * 100
    percentile_upper = (1 - alpha / 2) * 100
    
    if len(bootstrap_accuracies) > 0:
        bootstrap_accuracies = np.array(bootstrap_accuracies)
        bootstrap_stats = {
            'value': original_accuracy,
            'bootstrap_mean': np.mean(bootstrap_accuracies),
            'bootstrap_std': np.std(bootstrap_accuracies),
            'bootstrap_median': np.median(bootstrap_accuracies),
            'ci_lower': np.percentile(bootstrap_accuracies, percentile_lower),
            'ci_upper': np.percentile(bootstrap_accuracies, percentile_upper),
            'n_bootstrap': len(bootstrap_accuracies)
        }
    else:
        bootstrap_stats = {
            'value': original_accuracy,
            'bootstrap_mean': np.nan,
            'bootstrap_std': np.nan,
            'bootstrap_median': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_bootstrap': 0
        }
    
    return bootstrap_stats


def evaluate_multiple_choice_with_bootstrap(pred_list, label_list, n_bootstrap=1000, confidence_level=0.95, random_state=None, logger=None):
    original_accuracy = accuracy_score(label_list, pred_list)
    
    bootstrap_stats = bootstrap_accuracy(pred_list, label_list, n_bootstrap, confidence_level, random_state)
    
    combined_results = {
        'Accuracy': bootstrap_stats
    }
    
    print(f"Bootstrap samples: {n_bootstrap}, Confidence level: {confidence_level:.1%}")
    print(f"Accuracy: {original_accuracy:.4f}")
    if not np.isnan(bootstrap_stats['bootstrap_mean']):
        print(f"Bootstrap mean: {bootstrap_stats['bootstrap_mean']:.4f} ± {bootstrap_stats['bootstrap_std']:.4f}")
        print(f"95% CI: [{bootstrap_stats['ci_lower']:.4f}, {bootstrap_stats['ci_upper']:.4f}]")
    
    if logger:
        logger.info("=== Multiple Choice Evaluation Results (with Bootstrap) ===")
        logger.info(f"Bootstrap samples: {n_bootstrap}, Confidence level: {confidence_level:.1%}")
        logger.info(f"Accuracy: {original_accuracy:.4f}")
        if not np.isnan(bootstrap_stats['bootstrap_mean']):
            logger.info(f"Bootstrap mean: {bootstrap_stats['bootstrap_mean']:.4f} ± {bootstrap_stats['bootstrap_std']:.4f}")
            logger.info(f"95% CI: [{bootstrap_stats['ci_lower']:.4f}, {bootstrap_stats['ci_upper']:.4f}]")
            logger.info(f"Number of successful bootstrap samples: {bootstrap_stats['n_bootstrap']}")
    
    return combined_results


def bootstrap_metrics_multilabel(pred_list, label_list, n_bootstrap=1000, confidence_level=0.95, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    y_pred = np.array(pred_list)
    y_true = np.array(label_list)
    
    if len(y_true.shape) == 1:
        n_classes = y_pred.shape[1] if len(y_pred.shape) > 1 else 10
        y_true_onehot = np.zeros((len(y_true), n_classes))
        y_true_onehot[np.arange(len(y_true)), y_true.astype(int)] = 1
        y_true = y_true_onehot
    n_samples = len(y_pred)
    bootstrap_results = {
        'AUC_Micro': [],
        'AUC_Macro': [],
        'AUC_Weighted': [],
        'Macro_F1': []
    }     
    for i in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_pred = y_pred[indices]
        boot_labels = y_true[indices]
        
        try:
            auc_micro = roc_auc_score(boot_labels, boot_pred, average="micro")
            auc_macro = roc_auc_score(boot_labels, boot_pred, average="macro")
            auc_weighted = roc_auc_score(boot_labels, boot_pred, average="weighted")
            boot_pred_binary = (boot_pred >= 0.5).astype(int)
            macro_f1 = f1_score(boot_labels, boot_pred_binary, average="macro", zero_division=0)
            if np.isnan(auc_micro) or np.isnan(auc_macro) or np.isnan(auc_weighted) or np.isnan(macro_f1):
                continue        
            bootstrap_results['AUC_Micro'].append(auc_micro)
            bootstrap_results['AUC_Macro'].append(auc_macro)
            bootstrap_results['AUC_Weighted'].append(auc_weighted)
            bootstrap_results['Macro_F1'].append(macro_f1)
        except (ValueError, ZeroDivisionError) as e:
            continue
    
    alpha = 1 - confidence_level
    percentile_lower = (alpha / 2) * 100
    percentile_upper = (1 - alpha / 2) * 100
    
    bootstrap_stats = {}
    for metric_name, values in bootstrap_results.items():
        if len(values) > 0:
            values = np.array(values)
            values = values[~np.isnan(values)]
            
            if len(values) > 0:
                bootstrap_stats[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'ci_lower': np.percentile(values, percentile_lower),
                    'ci_upper': np.percentile(values, percentile_upper),
                    'n_bootstrap': len(values)
                }
            else:   
                bootstrap_stats[metric_name] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'median': np.nan,
                    'ci_lower': np.nan,
                    'ci_upper': np.nan,
                    'n_bootstrap': 0
                } 
        else:
            bootstrap_stats[metric_name] = {
                'mean': np.nan,
                'std': np.nan,
                'median': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'n_bootstrap': 0
            }
    return bootstrap_stats

def evaluate_multilabel_model_with_bootstrap(pred_list, label_list, n_bootstrap=1000, confidence_level=0.95, random_state=None, logger=None):
    if isinstance(pred_list, torch.Tensor):
        pred_numpy = pred_list.detach().cpu().numpy()
    elif isinstance(pred_list, np.ndarray):
        pred_numpy = pred_list
    else:
        pred_numpy = np.array(pred_list)
    
    if isinstance(label_list, torch.Tensor):
        label_numpy = label_list.detach().cpu().numpy()
    elif isinstance(label_list, np.ndarray):
        label_numpy = label_list
    else:
        label_numpy = np.array(label_list)
    if len(pred_numpy.shape) == 1:
        pred_tensor = torch.from_numpy(pred_numpy)
        if len(label_numpy.shape) == 1:
            n_classes = int(label_numpy.max()) + 1 if len(label_numpy) > 0 else pred_numpy.shape[0]
        else:
            n_classes = label_numpy.shape[1]
        if pred_numpy.shape[0] == n_classes:
            pred_numpy = pred_numpy.reshape(-1, n_classes)
            pred_tensor = torch.from_numpy(pred_numpy)
            pred_numpy = F.softmax(pred_tensor, dim=-1).numpy()
        else:
            pred_numpy = F.softmax(pred_tensor.unsqueeze(0), dim=-1).squeeze(0).numpy()
    else:
        pred_tensor = torch.from_numpy(pred_numpy)
        row_sums = pred_numpy.sum(axis=1) if len(pred_numpy.shape) > 1 else np.array([pred_numpy.sum()])
        is_probability = np.allclose(row_sums, 1.0, atol=0.01) and pred_numpy.min() >= 0.0 and pred_numpy.max() <= 1.0
        if not is_probability:
            pred_numpy = F.softmax(pred_tensor, dim=-1).numpy()
    
    if len(label_numpy.shape) == 1:
        n_classes = pred_numpy.shape[1] if len(pred_numpy.shape) > 1 else int(label_numpy.max()) + 1
        label_onehot = np.zeros((len(label_numpy), n_classes))
        label_onehot[np.arange(len(label_numpy)), label_numpy.astype(int)] = 1
        label_numpy = label_onehot
    auc_scores = roc_auc_score(label_numpy, pred_numpy, average=None)
    auc_micro = roc_auc_score(label_numpy, pred_numpy, average="micro")
    auc_macro = roc_auc_score(label_numpy, pred_numpy, average="macro")
    auc_weighted = roc_auc_score(label_numpy, pred_numpy, average="weighted")        
    pred_binary = (pred_numpy >= 0.5).astype(int)
    macro_f1 = f1_score(label_numpy, pred_binary, average="macro", zero_division=0)
    original_metrics = {
        'auc_scores': auc_scores,
        'auc_micro': auc_micro,
        'auc_macro': auc_macro,
        'auc_weighted': auc_weighted,
        'macro_f1': macro_f1
    }    
    bootstrap_stats = bootstrap_metrics_multilabel(pred_numpy, label_numpy, n_bootstrap, confidence_level, random_state)
    
    combined_results = {}
    metric_mapping = {
        'auc_micro': 'AUC_Micro',
        'auc_macro': 'AUC_Macro',
        'auc_weighted': 'AUC_Weighted',
        'macro_f1': 'Macro_F1'
    }
    for original_key, bootstrap_key in metric_mapping.items():
        if original_key in original_metrics and bootstrap_key in bootstrap_stats:
            combined_results[original_key] = {
                'value': original_metrics[original_key],
                'bootstrap_mean': bootstrap_stats[bootstrap_key]['mean'],
                'bootstrap_std': bootstrap_stats[bootstrap_key]['std'],
                'bootstrap_median': bootstrap_stats[bootstrap_key]['median'],
                'ci_lower': bootstrap_stats[bootstrap_key]['ci_lower'],
                'ci_upper': bootstrap_stats[bootstrap_key]['ci_upper'],
                'n_bootstrap': bootstrap_stats[bootstrap_key]['n_bootstrap']
            }
    if 'auc_scores' in original_metrics:
        combined_results['auc_scores'] = original_metrics['auc_scores']
    
    print(f"Bootstrap samples: {n_bootstrap}, Confidence level: {confidence_level:.1%}")
    if logger:
        logger.info("=== Multilabel Bootstrap Evaluation Results ===")
        logger.info(f"Bootstrap samples: {n_bootstrap}, Confidence level: {confidence_level:.1%}")
        
        for metric_name, metric_stats in combined_results.items():
            if metric_name != 'auc_scores' and isinstance(metric_stats, dict):
                original_val = metric_stats['value']
                bootstrap_mean = metric_stats['bootstrap_mean']
                bootstrap_std = metric_stats['bootstrap_std']
                ci_lower = metric_stats['ci_lower']
                ci_upper = metric_stats['ci_upper']
                
                logger.info(f"{metric_name}:")
                logger.info(f"  Original value: {original_val:.4f}")
                if not np.isnan(bootstrap_mean):
                    logger.info(f"  Bootstrap mean: {bootstrap_mean:.4f} ± {bootstrap_std:.4f}")
                    logger.info(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                else:
                    logger.info(f"  Bootstrap statistics: N/A (insufficient valid samples)")
    
    return combined_results



def llm_response_evaluation(args, responses, logits, test_dataset, logger=None):
    if args.dataset in ['mimic3_mortality', 'mimic4_mortality', 'tjh_mortality', 'mimic4_readmission']:
        predicions = [float(parse_llm_response_for_mimic3_mortality(response)) for response in responses]
        y_true = [int(test_dataset['y'][i]) for i in range(len(test_dataset['y']))]
        
        logger.info("Starting bootstrap evaluation...")
        bootstrap_metrics = evaluate_binary_model_with_bootstrap(score_list=predicions, label_list=y_true, n_bootstrap=1000, confidence_level=0.95, random_state=42, logger=logger, threshold=getattr(args, "classification_threshold", 0.5))
        logger.log_metrics(bootstrap_metrics, "LLM Response Evaluation Results (with Bootstrap)")
        return bootstrap_metrics
    elif args.dataset in ['mimic3_los']:
        
        if logits is None or len(logits) == 0:
            raise ValueError("Logits are required for mimic3_los evaluation but were not provided")
        
        n_samples = len(logits)
        n_classes = 4
        classification_options = ['A', 'B', 'C', 'D']
        letter_to_num = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        
        probs_list = []
        for i, values in enumerate(logits):
            if not isinstance(values, dict) or any(option not in values for option in classification_options):
                raise ValueError(f"Missing LOS option probabilities for sample {i}")
            probs = np.array([values[option] for option in classification_options], dtype=float)
            if not np.isfinite(probs).all() or (probs < 0).any() or probs.sum() <= 0:
                raise ValueError(f"Invalid LOS option probabilities for sample {i}")
            probs_list.append(probs / probs.sum())
        probs_array = np.array(probs_list)

        y_true = np.array(test_dataset['y'], dtype=int)
        
        if len(probs_array) != len(y_true):
            none_count = sum(1 for x in logits if x is None)
            dict_count = sum(1 for x in logits if isinstance(x, dict))
            raise ValueError(
                f"Mismatch between logits length ({len(probs_array)}) and labels length ({len(y_true)}). "
                f"Logits info: total={len(logits)}, None values={none_count}, dict values={dict_count}. "
                f"Please check if all samples were processed correctly."
            )

        logger.info("Starting bootstrap evaluation for multi-class classification (mimic3_los)...")
        logger.info(f"Number of samples: {len(probs_array)}")
        logger.info(f"Number of classes: {n_classes}")
        logger.info(f"Label distribution: {np.bincount(y_true)}")
        
        bootstrap_metrics = evaluate_multilabel_model_with_bootstrap(
            pred_list=probs_array.tolist(),
            label_list=y_true.tolist(),
            n_bootstrap=1000,
            confidence_level=0.95,
            random_state=42,
            logger=logger
        )
        logger.log_metrics(bootstrap_metrics, "LLM Response Evaluation Results (with Bootstrap)")
        return bootstrap_metrics

    elif args.dataset in ['cmb_exam_patient']:
        predictions = []
        for i, response in enumerate(responses):
            pred = parse_llm_response_for_cmb_exam_patient(response)
            if pred is None:
                if logger:
                    logger.warning(f"Failed to parse response for sample {i}: {response[:100]}...")
                else:
                    print(f"Warning: Failed to parse response for sample {i}: {response[:100]}...")
                predictions.append('X')
            else:
                predictions.append(pred)
        
        y_true = []
        label_key = 'answer'
        if label_key in test_dataset:
            y_true = [str(test_dataset[label_key][i]).strip().upper() for i in range(len(test_dataset[label_key]))]
        else:
            raise ValueError(f"test_dataset must contain '{label_key}' field for cmb_exam_patient dataset")
        
        valid_indices = [i for i, pred in enumerate(predictions) if pred != 'X']
        if len(valid_indices) < len(predictions):
            if logger:
                logger.warning(f"Filtered out {len(predictions) - len(valid_indices)} invalid predictions")
            else:
                print(f"Warning: Filtered out {len(predictions) - len(valid_indices)} invalid predictions")
        
        if len(valid_indices) == 0:
            raise ValueError("No valid predictions found after parsing")
        
        valid_predictions = [predictions[i] for i in valid_indices]
        valid_labels = [y_true[i] for i in valid_indices]
        
        if logger:
            logger.info("Starting bootstrap evaluation for multiple choice questions...")
        bootstrap_metrics = evaluate_multiple_choice_with_bootstrap(
            pred_list=valid_predictions, 
            label_list=valid_labels, 
            n_bootstrap=1000, 
            confidence_level=0.95, 
            random_state=42, 
            logger=logger
        )
        
        if logger:
            logger.log_metrics(bootstrap_metrics, "LLM Response Evaluation Results (with Bootstrap)")
        
        return bootstrap_metrics
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

if __name__ == '__main__':
    score_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    label_list = [0, 0, 1, 0, 0, 1, 1, 0, 1]
    
    print("=== Standard Evaluation ===")
    metrics = evaluate_binary_model(score_list, label_list)
    print("\n=== Bootstrap Evaluation ===")
    bootstrap_metrics = bootstrap_metrics(score_list, label_list, n_bootstrap=100)