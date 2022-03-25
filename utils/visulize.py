import numpy as np
from itertools import cycle
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
import os
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from PIL import Image
import torch


def tensor2img(ten):
    img = ten.numpy()
    colors = [(0, 0, 0), (255, 0, 0), (0, 128, 0), (255, 255, 0), (0, 0, 128)]
    num_classes = 5
    seg_img = np.zeros((np.shape(img)[0], np.shape(img)[1], 3))
    for c in range(num_classes):
        seg_img[:, :, 0] += ((img[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((img[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((img[:, :] == c) * (colors[c][2])).astype('uint8')

    image = Image.fromarray(np.uint8(seg_img)).resize((2100, 700), Image.NEAREST)
    # image = Image.fromarray(np.uint8(seg_img))
    return image


def show_image(prediction, save_dir, name, label):
    tensor = prediction.cpu().clone()
    max_val, max_index = torch.max(torch.softmax(tensor, dim=1), dim=1)

    plt.cla()
    plt.close('all')
    plt.figure()
    for i, ten in enumerate(max_index):
        # print(i, ten.shape)
        img = tensor2img(ten)
        img.save(os.path.join(save_dir, label + name[i].split('.')[0] + '.png'))
        # plt.imshow(img)
        # plt.show()
    return img


def plot_lr(lr_list):
    # print(lr_list)
    x = []
    for i in range(len(lr_list)):
        x.append(i + 1)
    plt.figure(figsize=(20, 10))
    plt.plot(x, lr_list, label='loss', linewidth=2.0)

    plt.legend()
    plt.xlabel('num', fontsize=24)
    plt.ylabel('loss', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.title('name', fontsize=24)
    plt.grid(linestyle='-.')

    # plt.savefig('./test.png')
    # plt.show()


def plot_loss_acc(csv_file):
    """
    :param csv_file:
    :return:
    """
    plt.cla()
    plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(30, 10))
    plot_name = ['loss', 'acc']
    plot_phase = [['train_loss', 'val_loss'], ['train_acc@1', 'val_acc@1']]
    for i in range(len(plot_name)):
        for j in range(len(plot_phase[0])):
            axes[i].plot(csv_file[plot_phase[i][j]])
        axes[i].grid(linestyle=":")
        axes[i].tick_params(labelsize=30)
        axes[i].set_xlabel('Epoch', fontsize=30)
        axes[i].set_ylabel(plot_name[i], fontsize=30)
        axes[i].legend(plot_phase[i], fontsize=20, loc='upper left')

    return fig


def plot_confusion_matrix_inPrediction(cm, _classes=None):
    """
    :param cm:
    :param _classes:
    :return:
    """
    if _classes == None:
        classes = ['N1', 'N2', 'N3', 'N4', 'W']
    else:
        classes = _classes

    plt.cla()
    plt.close('all')
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    # print(cm)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        # print(y_val, x_val)
        c = cm[y_val][x_val]
        cc = c / cm[y_val].sum()
        if c > 0.001:
            plt.text(x_val, y_val, "%.2f" % (cc), color='black', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=0)
    plt.yticks(xlocations, classes)
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')

    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    return plt


def plot_confusion_matrix(cm, _classes=None):
    if _classes == None:
        classes = ['N1', 'N2', 'N3', 'N4', 'W']
    else:
        classes = _classes

    plt.cla()
    plt.close('all')
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.0f" % (c,), color='black', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    # plt.xticks(xlocations, classes, rotation=45)
    plt.xticks(xlocations, classes, rotation=0)
    plt.yticks(xlocations, classes)
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    return plt


def plot_roc(y_score, y_test):
    y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    n_classes = y_test.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.cla()
    plt.close('all')
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    colors = cycle(['gold', '#1E90FF', '#FF6347', '#9370DB', '#228B22'])
    classes = ['N1', 'N2', 'N3', 'N4', 'W']
    lw = 2
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of {0} (area = {1:0.2f})'
                       ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    return plt


if __name__ == '__main__':
    prediction = torch.rand([2, 5, 4, 4])
    save_dir = './today/predict_img'
    name = ['a', 'b']
    show_image(prediction, save_dir + '/EP' + str(2), name)
