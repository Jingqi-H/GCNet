from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np
import torch


def hist_info(n_cl, pred, gt):
    """
    :param n_cl:
    :param pred: [bs, h, w]
    :param gt: [bs, h, w]
    :return:
    """
    assert (pred.shape == gt.shape)
    k = (gt >= 0) & (gt < n_cl)
    labeled = np.sum(k)
    correct = np.sum((pred[k] == gt[k]))

    return np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int),
                       minlength=n_cl ** 2).reshape(n_cl,
                                                    n_cl), labeled, correct


def compute_score(hist, correct, labeled):
    """
    :param hist:
    :param correct:
    :param labeled:
    :return:
    """
    np.seterr(divide='ignore', invalid='ignore')
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))  # .sum(1) 行求和， .sum(0) 列求和
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    freq = hist.sum(1) / hist.sum()
    freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
    mean_pixel_acc = correct / labeled

    return iu, mean_IU, mean_IU_no_back, mean_pixel_acc


def metrics_score_multi(y_true, y_pred, average='macro'):
    """
    :param y_true: [bs]
    :param y_pred: [bs, num_class]
    :param average:
    :return:
    """
    pred_probs, pred_labels = torch.max(torch.from_numpy(y_pred), dim=1)
    acc_ = metrics.accuracy_score(y_true, pred_labels)
    recall_ = metrics.recall_score(y_true, pred_labels, average=average)
    precision_ = metrics.precision_score(y_true, pred_labels, average=average)
    auc_ = metrics.roc_auc_score(y_true, y_pred, multi_class="ovr",
                                 average=average)
    f1_ = metrics.f1_score(y_true, pred_labels, average=average)
    return acc_, recall_, precision_, auc_, f1_


def pred_prob2pred_label(outputs):
    pred_label = torch.max(outputs.cpu(), dim=1)[1]
    pred_prob = torch.softmax(outputs.cpu(), dim=1)
    return pred_prob, pred_label
