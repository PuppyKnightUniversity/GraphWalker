import logging
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics


def print_metrics_multilabel(y_true, predictions, verbose=True):
    # 确保是torch tensor
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions)
    
    if len(y_true.shape) == 1:
        # 多分类任务：根据predictions的形状确定类别数
        if len(predictions.shape) == 2:
            num_classes = predictions.shape[1]
            y_true = F.one_hot(y_true.long(), num_classes)
            predictions = predictions.softmax(dim=-1)
        else:
            # 如果predictions也是1D，可能是多标签任务，使用sigmoid
            predictions = predictions.sigmoid()
    else:
        # 多标签任务：y_true已经是one-hot或multi-hot
        predictions = predictions.sigmoid()
    y_true = np.array(y_true)
    predictions = np.array(predictions)

    # 添加调试信息：检查标签和预测分布
    if verbose:
        # 如果是多分类（one-hot编码的y_true）
        if len(y_true.shape) == 2:
            y_true_classes = np.argmax(y_true, axis=1)
            pred_classes = np.argmax(predictions, axis=1)
            logging.info("Label distribution: {}".format(np.bincount(y_true_classes)))
            logging.info("Prediction distribution: {}".format(np.bincount(pred_classes, minlength=y_true.shape[1])))
            # 计算准确率
            accuracy = np.mean(y_true_classes == pred_classes)
            logging.info("Accuracy = {:.4f}".format(accuracy))
            # 打印混淆矩阵
            cm = metrics.confusion_matrix(y_true_classes, pred_classes)
            logging.info("Confusion Matrix:\n{}".format(cm))
            # 计算每个类别的精确率、召回率和F1分数
            precision_per_class = metrics.precision_score(y_true_classes, pred_classes, average=None, zero_division=0)
            recall_per_class = metrics.recall_score(y_true_classes, pred_classes, average=None, zero_division=0)
            f1_per_class = metrics.f1_score(y_true_classes, pred_classes, average=None, zero_division=0)
            logging.info("Per-class Precision: {}".format([f"{p:.4f}" for p in precision_per_class]))
            logging.info("Per-class Recall: {}".format([f"{r:.4f}" for r in recall_per_class]))
            logging.info("Per-class F1-Score: {}".format([f"{f:.4f}" for f in f1_per_class]))
        else:
            # 多标签情况
            logging.info("y_true shape: {}, predictions shape: {}".format(y_true.shape, predictions.shape))

    auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
    ave_auc_micro = metrics.roc_auc_score(y_true, predictions,
                                          average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, predictions,
                                          average="macro")
    ave_auc_weighted = metrics.roc_auc_score(y_true, predictions,
                                             average="weighted")

    if verbose:
        logging.info("auc_micro = {:.4f}".format(ave_auc_micro))
        logging.info("auc_macro = {:.4f}".format(ave_auc_macro))
        logging.info("auc_weighted = {:.4f}".format(ave_auc_weighted))
        if len(y_true.shape) == 2:
            logging.info("Per-class AUC: {}".format(auc_scores))

    return {"auc_scores": auc_scores,
            "auroc": ave_auc_micro,
            "auc_macro": ave_auc_macro,
            "auc_weighted": ave_auc_weighted}


def print_metrics_binary(y_true, predictions, verbose=True):
    predictions = np.array(predictions)
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose(
            (1, 0))

    cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))
    if verbose:
        # logging.info("confusion matrix:")
        logging.info(cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
    auroc = metrics.roc_auc_score(y_true, predictions[:, 1])

    (precisions, recalls,
     thresholds) = metrics.precision_recall_curve(y_true, predictions[:, 1])
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    f1_score = 2 * prec1 * rec1 / (prec1 + rec1)
    if verbose:
        logging.info("AUC of ROC = {:.4f}".format(auroc))
        logging.info("AUC of PRC = {:.4f}".format(auprc))
        logging.info("min(+P, Se) = {:.4f}".format(minpse))
        logging.info("f1_score = {:.4f}".format(f1_score))

    return {
        "auroc": auroc,
        "auprc": auprc,
        "minpse": minpse,
        "f1_score": f1_score
    }


def print_metrics_regression(y_true, predictions, verbose=True):
    predictions = np.array(predictions)
    y_true = np.array(y_true)

    mse = metrics.mean_squared_error(y_true, predictions)
    mae = metrics.mean_absolute_error(y_true, predictions)

    if verbose:
        logging.info("MSE = {:.4f}".format(mse))
        logging.info("MAE = {:.4f}".format(mae))

    return {"mse": mse,
            "mae": mae}
    