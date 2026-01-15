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
    """
    Parse the LLM response for mimic3 mortality.
    Args:
        response: the response from the LLM.
    Returns:
        the predicted probability of mortality. (float)
    """
    try:        
        # Filter out <think></think> tags and their content if thinking mode is enabled
        # This handles both <think>...</think> and <think>\n...\n</think> patterns
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
    """
    Parse the LLM response for mimic3_los to extract the letter option (A, B, C, or D).
    Args:
        response: the response from the LLM.
    Returns:
        the predicted option letter (e.g., 'A', 'B', 'C', 'D'). Returns None if not found.
    """
    if not response:
        return None

    # Remove whitespace and convert to uppercase
    response = response.strip().upper()
    
    # Try to find a single letter option (A-D)
    # First, try to match a single letter at the start of the response
    match = re.match(r'^([A-D])', response)
    if match:
        return match.group(1)
    
    # Try to find a letter in parentheses or brackets: (A), [A], etc.
    match = re.search(r'[\(\[（【]([A-D])[\)\]）】]', response)
    if match:
        return match.group(1)   
    # Try to find a letter in parentheses or brackets: (A), [A], etc.
    match = re.search(r'[\(\[（【]([A-D])[\)\]）】]', response)
    if match:
        return match.group(1)
    
    # Try to find a letter after common prefixes
    match = re.search(r'(?:答案|选项|选择|answer|option|choice|category)[:：\s]*([A-D])', response, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Try to find any single letter (A-D) in the first few characters
    first_part = response[:50]
    match = re.search(r'\b([A-D])\b', first_part)
    if match:
        return match.group(1)
    
    return None 

def parse_llm_response_for_cmb_exam_patient(response: str) -> str:
    """
    Parse the LLM response for cmb_exam_patient multiple choice questions.
    Args:
        response: the response from the LLM.
    Returns:
        the predicted option letter (e.g., 'A', 'B', 'C', 'D'). Returns None if not found.
    """
    if not response:
        return None
    
    # Filter out <think></think> tags and their content if thinking mode is enabled
    # This handles both <think>...</think> and <think>\n...\n</think> patterns
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Remove whitespace and convert to uppercase
    response = response.strip().upper()
    
    # Try to find a single letter option (A-Z)
    # First, try to match a single letter at the start of the response
    match = re.match(r'^([A-Z])', response)
    if match:
        return match.group(1)
    
    # Try to find a letter in parentheses or brackets: (A), [A], etc.
    match = re.search(r'[\(\[（【]([A-Z])[\)\]）】]', response)
    if match:
        return match.group(1)
    
    # Try to find a letter after common prefixes
    match = re.search(r'(?:答案|选项|选择|answer|option|choice)[:：\s]*([A-Z])', response, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Try to find any single letter in the first few characters
    # Extract first 50 characters and look for a single letter
    first_part = response[:50]
    match = re.search(r'\b([A-Z])\b', first_part)
    if match:
        return match.group(1)
    
    return None




def evaluate_model(score_list, label_list): 
    """
    Evaluate the performance of a binary classification model by calculating various metrics
    and optionally plotting ROC and Precision-Recall curves.
    
    Parameters:
    score_list (list): List of predicted probabilities (scores between 0 and 1)
    label_list (list): List of true binary labels (0 or 1)
    
    Returns:
    dict: Dictionary containing various evaluation metrics
    """

    # Validate inputs
    if len(score_list) != len(label_list):
        raise ValueError("Length of predicted scores and true labels must be the same")
    
    # inverse score_list and label_list
    score_list = [1 - score for score in score_list]
    label_list = [1 - label for label in label_list]
    
    # Calculate metrics
    metrics = evaluate_binary_model(score_list, label_list)
    return metrics


def evaluate_and_save(score_list, label_list, save_path):
    metrics = evaluate_model(score_list, label_list)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def evaluate_binary_model(score_list, label_list, verbose=True): 
    """
    Evaluate the performance of a binary classification model by calculating various metrics
    and optionally plotting ROC and Precision-Recall curves.
    
    Parameters:
    score_list (list): List of predicted probabilities (scores between 0 and 1)
    label_list (list): List of true binary labels (0 or 1)
    verbose (bool): Whether to print results to console

    Returns:
    dict: Dictionary containing various evaluation metrics
    """
    # Convert to numpy arrays
    y_scores = np.array(score_list)
    y_true = np.array(label_list)
        
    # Validate inputs
    if len(y_scores) != len(y_true):
        raise ValueError("Length of predicted scores and true labels must be the same")
    
    # ===== Basic Metrics Calculation =====
    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)
    brier_score = brier_score_loss(y_true, y_scores)
    
    # ===== Best F1 Score and MinPSE Calculation =====
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)  # Avoid division by zero
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    # thresholds length = len(precisions)-1, 所以用条件判断一下
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
    
    # Calculate minpse: max of min(precision, recall) across all thresholds
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    
    # Construct results dictionary
    metrics = {
        'AUPRC': auprc,
        'AUROC': auroc,
        'F1 Score': best_f1,
        'MinPSE': minpse,
        'Brier Score': brier_score,
        'Threshold': best_threshold,
    }
    
    # Print evaluation results
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
    """
    Evaluate the performance of a binary classification model by calculating various metrics
    and optionally plotting ROC and Precision-Recall curves.
    
    Parameters:
    score_list (list): List of predicted probabilities (scores between 0 and 1)
    label_list (list): List of true binary labels (0 or 1)
    
    Returns:
    dict: Dictionary containing various evaluation metrics
    """

    # Validate inputs
    if len(score_list) != len(label_list):
        raise ValueError("Length of predicted scores and true labels must be the same")
    
    # inverse score_list and label_list
    score_list = [1 - score for score in score_list]
    label_list = [1 - label for label in label_list]
    
    # Calculate metrics at high recall threshold
    metrics = evaluate_binary_model_with_threshold(score_list, label_list, threshold)
    return metrics


def evaluate_model_with_high_recall(score_list, label_list, recall_threshold=0.9): 
    """
    Evaluate the performance of a binary classification model by calculating various metrics
    and optionally plotting ROC and Precision-Recall curves.
    
    Parameters:
    score_list (list): List of predicted probabilities (scores between 0 and 1)
    label_list (list): List of true binary labels (0 or 1)
    
    Returns:
    dict: Dictionary containing various evaluation metrics
    """

    # Validate inputs
    if len(score_list) != len(label_list):
        raise ValueError("Length of predicted scores and true labels must be the same")
    
    # inverse score_list and label_list
    score_list = [1 - score for score in score_list]
    label_list = [1 - label for label in label_list]
    
    # Calculate precision-recall curve
    precision_curve, recall_curve, threshold_curve = precision_recall_curve(label_list, score_list)
    
    # Find the threshold at high recall
    high_recall_index = np.where(recall_curve >= recall_threshold)[0][-1]
    threshold = threshold_curve[high_recall_index]

    # Calculate metrics at high recall threshold
    metrics = evaluate_binary_model_with_threshold(score_list, label_list, threshold)
    return metrics


def evaluate_binary_model_with_threshold(score_list, label_list, threshold): 
    """
    Evaluate the performance of a binary classification model by calculating various metrics
    and optionally plotting ROC and Precision-Recall curves.
    
    Parameters:
    score_list (list): List of predicted probabilities (scores between 0 and 1)
    label_list (list): List of true binary labels (0 or 1)
    threshold (float): Threshold for converting probabilities to binary predictions, default is 0.5
    plot_curves (bool): Whether to plot ROC and Precision-Recall curves
    
    Returns:
    dict: Dictionary containing various evaluation metrics
    """
    # Convert to numpy arrays
    y_scores = np.array(score_list)
    y_true = np.array(label_list)
    
    # Validate inputs
    if len(y_scores) != len(y_true):
        raise ValueError("Length of predicted scores and true labels must be the same")
    
    # Convert probabilities to binary predictions based on the threshold
    y_pred = (y_scores >= threshold).astype(int)
    
    # Calculate basic evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()
    
    # Construct results dictionary
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
    
    # Print evaluation results
    print("Binary Classification Model Evaluation Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  PR AUC: {pr_auc:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    return metrics


def bootstrap_metrics(score_list, label_list, n_bootstrap=1000, confidence_level=0.95, random_state=None):
    """
    Calculate bootstrap confidence intervals and statistics for binary classification metrics.
    
    Parameters:
    score_list (list): List of predicted probabilities (scores between 0 and 1)
    label_list (list): List of true binary labels (0 or 1)
    n_bootstrap (int): Number of bootstrap samples (default: 1000)
    confidence_level (float): Confidence level for intervals (default: 0.95)
    random_state (int): Random seed for reproducibility
    
    Returns:
    dict: Dictionary containing bootstrap statistics for each metric
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    y_scores = np.array(score_list)
    y_true = np.array(label_list)
    n_samples = len(y_scores)
    
    # Initialize arrays to store bootstrap results
    bootstrap_results = {
        'AUPRC': [],
        'AUROC': [],
        'F1_Score': [],
        'MinPSE': [],
        'Brier_Score': []
    }
    
    # Perform bootstrap sampling
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_scores = y_scores[indices]
        boot_labels = y_true[indices]
        
        # Calculate metrics for this bootstrap sample
        try:
            # Check if we have both classes in the bootstrap sample
            unique_labels = np.unique(boot_labels)
            if len(unique_labels) < 2:
                continue  # Skip if only one class
            
            # Basic metrics
            auroc = roc_auc_score(boot_labels, boot_scores)
            auprc = average_precision_score(boot_labels, boot_scores)
            brier_score = brier_score_loss(boot_labels, boot_scores)
            
            # Check for NaN values
            if np.isnan(auroc) or np.isnan(auprc) or np.isnan(brier_score):
                continue
            
            # F1 and MinPSE calculation
            precisions, recalls, thresholds = precision_recall_curve(boot_labels, boot_scores)
            f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
            best_f1 = np.max(f1_scores)
            minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
            
            # Check for NaN values in F1 and MinPSE
            if np.isnan(best_f1) or np.isnan(minpse):
                continue
            
            # Store results
            bootstrap_results['AUPRC'].append(auprc)
            bootstrap_results['AUROC'].append(auroc)
            bootstrap_results['F1_Score'].append(best_f1)
            bootstrap_results['MinPSE'].append(minpse)
            bootstrap_results['Brier_Score'].append(brier_score)
            
        except (ValueError, ZeroDivisionError):
            # Skip this bootstrap sample if it fails (e.g., only one class, division by zero)
            continue
    
    # Calculate statistics for each metric
    alpha = 1 - confidence_level
    percentile_lower = (alpha / 2) * 100
    percentile_upper = (1 - alpha / 2) * 100
    
    bootstrap_stats = {}
    for metric_name, values in bootstrap_results.items():
        if len(values) > 0:
            values = np.array(values)
            # Remove any remaining NaN values
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
                # If all values are NaN, set to NaN
                bootstrap_stats[metric_name] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'median': np.nan,
                    'ci_lower': np.nan,
                    'ci_upper': np.nan,
                    'n_bootstrap': 0
                }
    
    return bootstrap_stats


def evaluate_binary_model_with_bootstrap(score_list, label_list, n_bootstrap=1000, confidence_level=0.95, random_state=None, logger=None):
    """
    Evaluate binary classification model with bootstrap confidence intervals.
    
    Parameters:
    score_list (list): List of predicted probabilities (scores between 0 and 1)
    label_list (list): List of true binary labels (0 or 1)
    n_bootstrap (int): Number of bootstrap samples (default: 1000)
    confidence_level (float): Confidence level for intervals (default: 0.95)
    random_state (int): Random seed for reproducibility
    logger: Logger instance for detailed logging
    
    Returns:
    dict: Dictionary containing metrics with bootstrap statistics
    """
    # Get original metrics
    original_metrics = evaluate_binary_model(score_list, label_list, verbose=False)
    
    # Get bootstrap statistics
    bootstrap_stats = bootstrap_metrics(score_list, label_list, n_bootstrap, confidence_level, random_state)
    
    # Combine results
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
    
    # Add threshold information
    combined_results['Threshold'] = original_metrics['Threshold']
    
    # Print results with bootstrap statistics (simplified)
    print(f"Bootstrap samples: {n_bootstrap}, Confidence level: {confidence_level:.1%}")
    
    # Log detailed results to file if logger is provided
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
    """
    Calculate bootstrap confidence intervals for accuracy metric.
    
    Parameters:
    pred_list (list): List of predicted labels
    label_list (list): List of true labels
    n_bootstrap (int): Number of bootstrap samples (default: 1000)
    confidence_level (float): Confidence level for intervals (default: 0.95)
    random_state (int): Random seed for reproducibility
    
    Returns:
    dict: Dictionary containing bootstrap statistics for accuracy
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    y_pred = np.array(pred_list)
    y_true = np.array(label_list)
    n_samples = len(y_pred)
    
    # Calculate original accuracy
    original_accuracy = accuracy_score(y_true, y_pred)
    
    # Initialize array to store bootstrap results
    bootstrap_accuracies = []
    
    # Perform bootstrap sampling
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_pred = y_pred[indices]
        boot_labels = y_true[indices]
        
        # Calculate accuracy for this bootstrap sample
        try:
            boot_accuracy = accuracy_score(boot_labels, boot_pred)
            if not np.isnan(boot_accuracy):
                bootstrap_accuracies.append(boot_accuracy)
        except (ValueError, ZeroDivisionError):
            # Skip this bootstrap sample if it fails
            continue
    
    # Calculate statistics
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
        # If all bootstrap samples failed, set to NaN
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
    """
    Evaluate multiple choice questions with bootstrap confidence intervals.
    
    Parameters:
    pred_list (list): List of predicted option letters
    label_list (list): List of true option letters
    n_bootstrap (int): Number of bootstrap samples (default: 1000)
    confidence_level (float): Confidence level for intervals (default: 0.95)
    random_state (int): Random seed for reproducibility
    logger: Logger instance for detailed logging
    
    Returns:
    dict: Dictionary containing accuracy with bootstrap statistics
    """
    # Calculate original accuracy
    original_accuracy = accuracy_score(label_list, pred_list)
    
    # Get bootstrap statistics
    bootstrap_stats = bootstrap_accuracy(pred_list, label_list, n_bootstrap, confidence_level, random_state)
    
    # Combine results
    combined_results = {
        'Accuracy': bootstrap_stats
    }
    
    # Print results with bootstrap statistics
    print(f"Bootstrap samples: {n_bootstrap}, Confidence level: {confidence_level:.1%}")
    print(f"Accuracy: {original_accuracy:.4f}")
    if not np.isnan(bootstrap_stats['bootstrap_mean']):
        print(f"Bootstrap mean: {bootstrap_stats['bootstrap_mean']:.4f} ± {bootstrap_stats['bootstrap_std']:.4f}")
        print(f"95% CI: [{bootstrap_stats['ci_lower']:.4f}, {bootstrap_stats['ci_upper']:.4f}]")
    
    # Log detailed results to file if logger is provided
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
    """
    Calculate bootstrap confidence intervals and statistics for multilabel classification metrics.
    
    Parameters:
    pred_list (list or np.ndarray): List/array of predicted probabilities (shape: [n_samples, n_classes])
    label_list (list or np.ndarray): List/array of true labels (shape: [n_samples, n_classes] or [n_samples] for one-hot)
    n_bootstrap (int): Number of bootstrap samples (default: 1000)
    confidence_level (float): Confidence level for intervals (default: 0.95)
    random_state (int): Random seed for reproducibility
    
    Returns:
    dict: Dictionary containing bootstrap statistics for each metric
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    y_pred = np.array(pred_list)
    y_true = np.array(label_list)
    
    # Ensure y_true is 2D (one-hot encoded)
    if len(y_true.shape) == 1:
        # If labels are class indices, convert to one-hot
        n_classes = y_pred.shape[1] if len(y_pred.shape) > 1 else 10
        y_true_onehot = np.zeros((len(y_true), n_classes))
        y_true_onehot[np.arange(len(y_true)), y_true.astype(int)] = 1
        y_true = y_true_onehot
    n_samples = len(y_pred)
    # Initialize arrays to store bootstrap results
    bootstrap_results = {
        'AUC_Micro': [],
        'AUC_Macro': [],
        'AUC_Weighted': [],
        'Macro_F1': []
    }     
    # Perform bootstrap sampling
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_pred = y_pred[indices]
        boot_labels = y_true[indices]
        
        # Calculate metrics for this bootstrap sample
        try:
            # Calculate AUC scores
            auc_micro = roc_auc_score(boot_labels, boot_pred, average="micro")
            auc_macro = roc_auc_score(boot_labels, boot_pred, average="macro")
            auc_weighted = roc_auc_score(boot_labels, boot_pred, average="weighted")
            # Calculate Macro-F1: convert probabilities to binary predictions using threshold 0.5
            boot_pred_binary = (boot_pred >= 0.5).astype(int)
            macro_f1 = f1_score(boot_labels, boot_pred_binary, average="macro", zero_division=0)
            if np.isnan(auc_micro) or np.isnan(auc_macro) or np.isnan(auc_weighted) or np.isnan(macro_f1):
                continue        
            # Store results
            bootstrap_results['AUC_Micro'].append(auc_micro)
            bootstrap_results['AUC_Macro'].append(auc_macro)
            bootstrap_results['AUC_Weighted'].append(auc_weighted)
            bootstrap_results['Macro_F1'].append(macro_f1)
        except (ValueError, ZeroDivisionError) as e:
            # Skip this bootstrap sample if it fails
            continue
    
    # Calculate statistics for each metric
    alpha = 1 - confidence_level
    percentile_lower = (alpha / 2) * 100
    percentile_upper = (1 - alpha / 2) * 100
    
    bootstrap_stats = {}
    for metric_name, values in bootstrap_results.items():
        if len(values) > 0:
            values = np.array(values)
            # Remove any remaining NaN values
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
                # If all values are NaN, set to NaN
                bootstrap_stats[metric_name] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'median': np.nan,
                    'ci_lower': np.nan,
                    'ci_upper': np.nan,
                    'n_bootstrap': 0
                } 
        else:
            # If no bootstrap samples succeeded, set to NaN
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
    """
    Evaluate multilabel classification model with bootstrap confidence intervals.
    
    Parameters:
    pred_list (list or np.ndarray): List/array of predicted probabilities (shape: [n_samples, n_classes])
    label_list (list or np.ndarray): List/array of true labels (shape: [n_samples, n_classes] or [n_samples] for one-hot)
    n_bootstrap (int): Number of bootstrap samples (default: 1000)
    confidence_level (float): Confidence level for intervals (default: 0.95)
    random_state (int): Random seed for reproducibility
    logger: Logger instance for detailed logging
    
    Returns:
    dict: Dictionary containing metrics with bootstrap statistics
    """
    # Convert to numpy arrays
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
    # Process predictions: if logits, apply softmax; if already probabilities, use as is
    if len(pred_numpy.shape) == 1:
        # Single dimension - assume logits, apply softmax
        pred_tensor = torch.from_numpy(pred_numpy)
        # For single dimension, we need to know number of classes from labels
        if len(label_numpy.shape) == 1:
            n_classes = int(label_numpy.max()) + 1 if len(label_numpy) > 0 else pred_numpy.shape[0]
        else:
            n_classes = label_numpy.shape[1]
        # Reshape to (1, n_samples) then apply softmax
        if pred_numpy.shape[0] == n_classes:
            # Each element is a logit for a class, reshape to (n_samples, n_classes)
            pred_numpy = pred_numpy.reshape(-1, n_classes)
            pred_tensor = torch.from_numpy(pred_numpy)
            pred_numpy = F.softmax(pred_tensor, dim=-1).numpy()
        else:
            # Single value per sample, need to expand
            pred_numpy = F.softmax(pred_tensor.unsqueeze(0), dim=-1).squeeze(0).numpy()
    else:
        # Multi-dimensional - check if already probabilities or logits
        pred_tensor = torch.from_numpy(pred_numpy)
        # Check if values sum to ~1 per row (probabilities) or are logits
        row_sums = pred_numpy.sum(axis=1) if len(pred_numpy.shape) > 1 else np.array([pred_numpy.sum()])
        is_probability = np.allclose(row_sums, 1.0, atol=0.01) and pred_numpy.min() >= 0.0 and pred_numpy.max() <= 1.0
        if not is_probability:
            # Values are logits, apply softmax
            pred_numpy = F.softmax(pred_tensor, dim=-1).numpy()
        # else: already probabilities, use as is
    
    # Process labels: ensure one-hot encoding
    if len(label_numpy.shape) == 1:
        # Labels are class indices, convert to one-hot
        n_classes = pred_numpy.shape[1] if len(pred_numpy.shape) > 1 else int(label_numpy.max()) + 1
        label_onehot = np.zeros((len(label_numpy), n_classes))
        label_onehot[np.arange(len(label_numpy)), label_numpy.astype(int)] = 1
        label_numpy = label_onehot
    # Get original metrics using sklearn directly
    auc_scores = roc_auc_score(label_numpy, pred_numpy, average=None)
    auc_micro = roc_auc_score(label_numpy, pred_numpy, average="micro")
    auc_macro = roc_auc_score(label_numpy, pred_numpy, average="macro")
    auc_weighted = roc_auc_score(label_numpy, pred_numpy, average="weighted")        
    # Calculate Macro-F1: convert probabilities to binary predictions using threshold 0.5
    pred_binary = (pred_numpy >= 0.5).astype(int)
    macro_f1 = f1_score(label_numpy, pred_binary, average="macro", zero_division=0)
    original_metrics = {
        'auc_scores': auc_scores,
        'auc_micro': auc_micro,
        'auc_macro': auc_macro,
        'auc_weighted': auc_weighted,
        'macro_f1': macro_f1
    }    
    # Get bootstrap statistics
    bootstrap_stats = bootstrap_metrics_multilabel(pred_numpy, label_numpy, n_bootstrap, confidence_level, random_state)
    
    # Combine results
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
    # Add per-class AUC scores if available
    if 'auc_scores' in original_metrics:
        combined_results['auc_scores'] = original_metrics['auc_scores']
    
    # Print results with bootstrap statistics (simplified)
    print(f"Bootstrap samples: {n_bootstrap}, Confidence level: {confidence_level:.1%}")
    # Log detailed results to file if logger is provided
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
    """
    Evaluate the performance of LLM's responses by calculating various metrics
    Args:
        args: the arguments.
        responses: the list of responses from the LLM.
        logits: the list of logits from the LLM.
        test_dataset: the test dataset.
        logger: the logger instance for progress display (optional).
    Returns:
        the dictionary containing various evaluation metrics
    """
    if args.dataset in ['mimic3_mortality', 'mimic4_mortality', 'tjh_mortality', 'mimic4_readmission']:
        predicions = [float(parse_llm_response_for_mimic3_mortality(response)) for response in responses]  # extract answer from LLM responses
        y_true = [int(test_dataset['y'][i]) for i in range(len(test_dataset['y']))]
        
        # Evaluate with bootstrap
        logger.info("Starting bootstrap evaluation...")
        bootstrap_metrics = evaluate_binary_model_with_bootstrap(score_list=predicions, label_list=y_true, n_bootstrap=1000, confidence_level=0.95, random_state=42, logger=logger)
        logger.log_metrics(bootstrap_metrics, "LLM Response Evaluation Results (with Bootstrap)")
        return bootstrap_metrics
    elif args.dataset in ['mimic3_los']:
        # 多分类任务：将logits字典转换为概率矩阵
        # logits格式：每个元素是一个字典，键为'A', 'B', 'C', 'D'，值为概率
        # 需要转换为(n_samples, n_classes)的概率矩阵，其中类别0对应选项'A'，类别1对应选项'B'，以此类推
        
        if logits is None or len(logits) == 0:
            raise ValueError("Logits are required for mimic3_los evaluation but were not provided")
        
        n_samples = len(logits)
        n_classes = 4  # 4个类别
        classification_options = ['A', 'B', 'C', 'D']  # 使用字母选项
        # 字母到数字的映射：A->0, B->1, C->2, D->3
        letter_to_num = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        
        # 将logits字典列表转换为概率矩阵
        probs_list = []
        for i, logit_dict in enumerate(logits):
            if not isinstance(logit_dict, dict):
                if logger:
                    logger.warning(f"Sample {i}: logits is not a dict, got {type(logit_dict)}. Using uniform distribution.")
                else:
                    print(f"Warning: Sample {i}: logits is not a dict, got {type(logit_dict)}. Using uniform distribution.")
                # 使用均匀分布作为默认值
                probs = [1.0 / n_classes] * n_classes
            else:
                # 按照选项顺序提取概率
                probs = []
                for option in classification_options:
                    if option in logit_dict:
                        probs.append(float(logit_dict[option]))
                    else:
                        # 如果某个选项缺失，使用0作为默认值
                        if logger:
                            logger.warning(f"Sample {i}: option '{option}' not found in logits. Using 0.0.")
                        probs.append(0.0)
                # 归一化以确保概率和为1
                prob_sum = sum(probs)
                if prob_sum > 0:
                    probs = [p / prob_sum for p in probs]
                else:
                    # 如果所有概率都是0，使用均匀分布
                    probs = [1.0 / n_classes] * n_classes
            
            probs_list.append(probs)
        
        # 转换为numpy数组
        probs_array = np.array(probs_list)
        # 转换为numpy数组
        probs_array = np.array(probs_list)
        
        # 获取标签：test_dataset['y']中保存的已经是binned后的标签（0-3），直接使用
        y_true = np.array(test_dataset['y'], dtype=int)
        
        # 验证数据一致性
        if len(probs_array) != len(y_true):
            # 检查logits中None值的数量
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
        
        # 调用多分类评估函数（虽然名字是多标签，但可以处理多分类任务）
        bootstrap_metrics = evaluate_multilabel_model_with_bootstrap(
            pred_list=probs_array.tolist(),
            label_list=y_true.tolist(),
            n_bootstrap=1000,
            confidence_level=0.95,
            random_state=42,
            logger=logger
        )
        logger.log_metrics(bootstrap_metrics, "LLM Response Evaluation Results (with Bootstrap)")
        
    elif args.dataset in ['cmb_exam_patient']:
        # Parse LLM responses to extract option letters
        predictions = []
        for i, response in enumerate(responses):
            pred = parse_llm_response_for_cmb_exam_patient(response)
            if pred is None:
                if logger:
                    logger.warning(f"Failed to parse response for sample {i}: {response[:100]}...")
                else:
                    print(f"Warning: Failed to parse response for sample {i}: {response[:100]}...")
                # Use a default value or skip - for now, we'll use 'X' as invalid marker
                predictions.append('X')
            else:
                predictions.append(pred)
        
        # Extract true labels from test_dataset
        y_true = []
        label_key = 'answer'  # NOTE: in cmb_exam_patient dataset, the label is stored in the 'answer' key
        if label_key in test_dataset:
            y_true = [str(test_dataset[label_key][i]).strip().upper() for i in range(len(test_dataset[label_key]))]
        else:
            raise ValueError(f"test_dataset must contain '{label_key}' field for cmb_exam_patient dataset")
        
        # Filter out invalid predictions for accuracy calculation
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
        
        # Evaluate with bootstrap
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
    # Test data
    score_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    label_list = [0, 0, 1, 0, 0, 1, 1, 0, 1]
    
    print("=== Standard Evaluation ===")
    metrics = evaluate_binary_model(score_list, label_list)
    print("\n=== Bootstrap Evaluation ===")
    bootstrap_metrics = bootstrap_metrics(score_list, label_list, n_bootstrap=100)