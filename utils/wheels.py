import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix

from utils.visulize import plot_lr, plot_confusion_matrix, plot_roc, plot_confusion_matrix_inPrediction


def save_list2np(list_, file_name):
    a = np.array(list_)
    np.save(file_name + '_a.npy', a)


def load_np2list(np_file):
    a = np.load(np_file)
    a = a.tolist()
    return a


def save_intermediate(net, save_interme, CP,
                      k, val_gt_labels_, val_pred_labels,
                      val_pred_probs, cla, test_pred_probs, test_gt_labels,
                      test_pred_labels, tr_result, va_result, te_result):
    torch.save(net.state_dict(), os.path.join(save_interme, 'CP' + str(CP) + 'K' + str(k + 1) + '.pth'))

    tr_result.to_csv(os.path.join(save_interme, 'Train' + 'K' + str(k + 1) + '.csv'))
    va_result.to_csv(os.path.join(save_interme, 'Val' + 'K' + str(k + 1) + '.csv'))
    te_result.to_csv(os.path.join(save_interme, 'Test' + 'K' + str(k + 1) + '.csv'))

    val_cm = confusion_matrix(val_gt_labels_, val_pred_labels, labels=None, sample_weight=None)
    val_con_mat = plot_confusion_matrix(val_cm, _classes=cla)
    plt.savefig(os.path.join(save_interme, 'CP' + str(CP) + 'confusion_matrix_val_k' + str(k + 1)))
    val_roc = plot_roc(val_pred_probs, val_gt_labels_)
    plt.savefig(os.path.join(save_interme, 'CP' + str(CP) + 'roc_val_k' + str(k + 1)))

    test_cm = confusion_matrix(test_gt_labels, test_pred_labels, labels=None, sample_weight=None)
    test_con_mat = plot_confusion_matrix(test_cm, _classes=cla)
    plt.savefig(os.path.join(save_interme, 'CP' + str(CP) + 'confusion_matrix_test_k' + str(k + 1)))
    test_roc = plot_roc(test_pred_probs, test_gt_labels)
    plt.savefig(os.path.join(save_interme, 'CP' + str(CP) + 'roc_test_k' + str(k + 1)))


def save_best(k, net, val_gt_labels_, val_pred_labels, val_pred_probs, cla,
              save_best_dir, save_dir, test_gt_labels=None, test_pred_probs=None, test_pred_labels=None):
    # val_pred_probs, val_pred_labels, val_gt_labels_
    val_cm = confusion_matrix(val_gt_labels_, val_pred_labels, labels=None, sample_weight=None)
    plot_confusion_matrix_inPrediction(val_cm)
    plt.savefig(os.path.join(save_best_dir, 'confusion_matrix2_val_k' + str(k + 1)))
    val_con_mat = plot_confusion_matrix(val_cm, _classes=cla)
    plt.savefig(os.path.join(save_best_dir, 'confusion_matrix_val_k' + str(k + 1)))
    val_roc = plot_roc(val_pred_probs, val_gt_labels_)
    plt.savefig(os.path.join(save_best_dir, 'roc_val_k' + str(k + 1)))

    # confusion_matrix
    # test_cm = confusion_matrix(test_gt_labels, test_pred_labels, labels=None, sample_weight=None)
    # test_con_mat = plot_confusion_matrix(test_cm, _classes=cla)
    # plt.savefig(os.path.join(save_best_dir, 'confusion_matrix_test_k' + str(k + 1)))
    # roc
    # test_roc = plot_roc(test_pred_probs, test_gt_labels)
    # plt.savefig(os.path.join(save_best_dir, 'roc_test_k' + str(k + 1)))

    # model
    torch.save(net.state_dict(), os.path.join(save_dir, 'best_linear_model_K' + str(k + 1) + '.pth'))


def save_k_final_results(net, save_dir, save_display_dir,
                         k, lr_epoch, val_gt_labels_, val_pred_labels,
                         val_pred_probs, cla, test_pred_probs=None, test_gt_labels=None,
                         test_pred_labels=None):
    # torch.save(net.state_dict(), os.path.join(save_dir, 'final_linear_model_K' + str(k + 1) + '.pth'))

    plot_lr(lr_epoch)
    plt.savefig(os.path.join(save_display_dir, 'lr_' + 'K' + str(k + 1) + '.png'))

    # val_pred_probs, val_pred_labels, val_gt_labels_
    val_cm = confusion_matrix(val_gt_labels_, val_pred_labels, labels=None, sample_weight=None)
    plot_confusion_matrix_inPrediction(val_cm)
    plt.savefig(os.path.join(save_display_dir, 'confusion_matrix2_val_k' + str(k + 1)))
    plot_confusion_matrix(val_cm, _classes=cla)
    plt.savefig(os.path.join(save_display_dir, 'confusion_matrix_val_k' + str(k + 1)))

    plot_roc(val_pred_probs, val_gt_labels_)
    plt.savefig(os.path.join(save_display_dir, 'roc_val_k' + str(k + 1)))

    # test_pred_probs, test_pred_labels, test_gt_labels
    # test_cm = confusion_matrix(test_gt_labels, test_pred_labels, labels=None, sample_weight=None)
    # test_con_mat = plot_confusion_matrix(test_cm, _classes=cla)
    # plt.savefig(os.path.join(save_display_dir, 'confusion_matrix_test_k' + str(k + 1)))
    # test_roc = plot_roc(test_pred_probs, test_gt_labels)
    # plt.savefig(os.path.join(save_display_dir, 'roc_test_k' + str(k + 1)))


def pre2mask(pre):

    batch_size = pre.shape[0]

    tensor_list = []
    for b in range(batch_size):
        pre_b = pre[b]
        pre_b = torch.softmax(pre_b, dim=0)
        max_val, max_index = torch.max(pre_b, dim=0)

        tensor_list.append(max_index)

    mask = torch.stack(tensor_list)
    return mask


def init_dir(date):
    """
    :param phase:
    :param date:
    :return:
    """
    save_dir = os.path.join('results/', date)
    mkfile(save_dir)
    save_intermediate_dir = os.path.join(save_dir, 'intermediate')
    mkfile(save_intermediate_dir)
    save_display_dir = os.path.join(save_dir, 'display')
    mkfile(save_display_dir)
    save_csv_dir = os.path.join(save_dir, 'csv')
    mkfile(save_csv_dir)
    save_best_dir = os.path.join(save_dir, 'best')
    mkfile(save_best_dir)
    save_pre_img = os.path.join(save_dir, 'display_output')
    return save_dir, save_intermediate_dir, save_display_dir, save_csv_dir, save_best_dir, save_pre_img


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, base_lr, total_niters, lr_power):
    lr = lr_poly(base_lr, i_iter, total_niters, lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr


def s2t(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return h, m, s


def gain_index(k, seg, total_size, indices):
    trll = 0
    trlr = k * seg
    vall = trlr
    valr = k * seg + seg
    trrl = valr
    trrr = total_size

    print("train indices: [%d,%d),[%d,%d), val indices: [%d,%d)"
          % (trll, trlr, trrl, trrr, vall, valr))
    train_indices = indices[trll:trlr] + indices[trrl:trrr]
    val_indices = indices[vall:valr]

    return train_indices, val_indices


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


def load_data(feature, label, name):
    """

    :param feature:
    :param label:
    :param name:
    :return:
    """
    _feature = pd.read_csv(feature, header=None)
    _label = pd.read_csv(label, header=None)
    _names = pd.read_csv(name, header=None)

    feature_list = _feature.values.tolist()
    label_list = _label.values.tolist()
    name_list = _names.values.tolist()

    feature_array = np.array(feature_list)  # (982, 512)
    class_array = np.squeeze(np.array(label_list))  # (982, 1)
    name_array = np.squeeze(np.array(name_list))  # (982, 1)

    return feature_array, class_array, name_array
