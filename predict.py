import json
import time, datetime
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from torchvision import transforms
from sklearn import metrics

from models.gcnet import GCNet
from dataset.MyImageFolder import StandarImageFolder
from utils.wheels import s2t, mkfile
from utils.metrics import pred_prob2pred_label, metrics_score_multi
from utils.visulize import plot_confusion_matrix, plot_roc, plot_confusion_matrix_inPrediction, \
    show_image
from config.config import BaseConfig


class DisplayOutput(object):
    def __init__(self, root):
        self.root = root

    def __call__(self, data_img, data_pre, name=None):
        data_img = data_img.cpu().clone()
        data_pre = data_pre.cpu().clone()

        plt.cla()
        plt.close('all')
        figure, axes = plt.subplots(data_pre.shape[0], 3, figsize=(6, data_pre.shape[0]))
        figure.tight_layout()
        for i in range(data_pre.shape[0]):
            img1_ten = data_img[i]
            pre1_ten = data_pre[i]
            img_show = Image.fromarray(np.uint8(self.img_ten2arr(img1_ten)), mode='RGB')
            pre_show = self.mask_ten2color(pre1_ten)

            axes[0].imshow(img_show)
            axes[0].axis('off')

            axes[1].imshow(pre_show)
            axes[1].axis('off')

            axes[2].imshow(img_show, alpha=1)
            axes[2].imshow(pre_show, alpha=0.4)
            axes[2].axis('off')

        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)

        if not name is None:
            plt.savefig(os.path.join(self.root + '/fig', name + '.png'))
        return pre_show, figure

    def mask_ten2color(self, ten):
        img = ten.numpy()
        colors = [(0, 0, 0), (255, 0, 0), (0, 128, 0), (255, 255, 0), (0, 0, 128)]
        num_classes = 5
        seg_img = np.zeros((np.shape(img)[0], np.shape(img)[1], 3))
        for c in range(num_classes):
            seg_img[:, :, 0] += ((img[:, :] == c) * (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((img[:, :] == c) * (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((img[:, :] == c) * (colors[c][2])).astype('uint8')

        #     image = Image.fromarray(np.uint8(seg_img)).resize((2100, 700), Image.NEAREST)
        image = Image.fromarray(np.uint8(seg_img))
        return image

    def img_ten2arr(self, input_image, imtype=np.uint8):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if not isinstance(input_image, np.ndarray):
            if isinstance(input_image, torch.Tensor):
                image_tensor = input_image.data
            else:
                return input_image
            image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
            if image_numpy.shape[0] == 1:  # grayscale to RGB
                image_numpy = np.tile(image_numpy, (3, 1, 1))
            for i in range(len(mean)):
                image_numpy[i] = image_numpy[i] * std[i] + mean[i]
            image_numpy = image_numpy * 255
            image_numpy = np.transpose(image_numpy, (1, 2, 0))
        else:
            image_numpy = input_image
        return image_numpy.astype(imtype)


model_weight_path = './results/GCNet.pth'
save_display_dir = os.path.join('./results/predictor', model_weight_path.split('/')[-3])
mkfile(save_display_dir)
save_pre_path = './results/predictor/predict_img'
mkfile(save_pre_path)
save_fig_dir = './results/predictor/fig'
mkfile(save_fig_dir)

start_time = time.time()
print('model:', model_weight_path.split('/')[-3])
print("start time:", datetime.datetime.now())
parser = BaseConfig(
    os.path.join("./config/", "config.yaml"))
args = parser.get_args()

# create model
# load model weights  build_lane_network
net = GCNet(parser).cuda()
# for name, param in net.named_parameters():
#     if param.requires_grad:
#         print(name)
# parm = {}
# for name, parameters in net.named_parameters():
#     if name == 'binary_seg.0.bias':
#         print(name, ':', parameters.size())
#         print(name, ':', parameters)
#         parm[name] = parameters.cpu().detach().numpy()

data_transform = transforms.Compose([
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_data = StandarImageFolder(root=os.path.join(args.img_path, 'data_test'),
                               transform=data_transform)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

net.eval()
dd = DisplayOutput(root=save_fig_dir)

missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)
if len(missing_keys) or len(unexpected_keys):
    print('missing_keys, unexpected_keys:', missing_keys, unexpected_keys)

pred_probs, pred_labels, gt_labels = 0, 0, 0
val_acc, recall, precision, auc, f1 = 0, 0, 0, 0, 0
with torch.no_grad():
    pred_label_all, pred_prob_all, gt_label = [], [], []
    pred_prob, pred_label = 0, 0
    for step, [img, label, name] in enumerate(test_loader, start=0):
        resnet_y, feature, em = net(img.cuda())
        pred_prob, pred_label = pred_prob2pred_label(resnet_y)

        # show_image(prediction=feature, save_dir=save_pre_path, name=name,
        #            label=str(label.item()) + str(pred_label.item()))
        # max_val, max_index = torch.max(torch.softmax(feature, dim=1), dim=1)
        # pp, ff = dd(data_img=img, data_pre=max_index)
        # ff.savefig(save_fig_dir + '/' + str(label.item()) + str(pred_label.item()) + name[0].split('.')[0] + '.png')

        pred_prob_all.append(pred_prob)
        pred_label_all.append(pred_label)
        gt_label.append(label)
        # break

pred_probs = np.concatenate(pred_prob_all)
pred_labels = np.concatenate(pred_label_all)
gt_labels = np.concatenate(gt_label)
val_acc, recall, precision, auc, f1 = metrics_score_multi(gt_labels, pred_probs)
print('acc:{:.2f} | recall:{:.2f} | precision:{:.2f} | auc:{:.2f} | f1:{:.2f}'.format(val_acc * 100,
                                                                                      recall * 100,
                                                                                      precision * 100,
                                                                                      auc * 100, f1 * 100))

# cm = metrics.confusion_matrix(gt_labels, pred_labels, labels=None, sample_weight=None)
# print(cm)
# plot_confusion_matrix(cm)
# # plt.savefig(os.path.join(save_display_dir, 'test_confusion_matrix' + '.pdf'), format="pdf", bbox_inches='tight', pad_inches=0)
# plt.savefig(os.path.join(save_display_dir, 'test_confusion_matrix' + '.png'))
# plot_confusion_matrix_inPrediction(cm)
# # plt.savefig(os.path.join(save_display_dir, 'test_confusion_matrix2' + '.pdf'), format="pdf", bbox_inches='tight', pad_inches=0)
# plt.savefig(os.path.join(save_display_dir, 'test_confusion_matrix2' + '.png'))
# plot_roc(pred_probs, gt_labels)
# # plt.savefig(os.path.join(save_display_dir, 'test_roc' + '.pdf'), format="pdf", bbox_inches='tight', pad_inches=0)
# plt.savefig(os.path.join(save_display_dir, 'test_roc' + '.png'))

print("\nEnd time:", datetime.datetime.now())
h, m, s = s2t(time.time() - start_time)
print("Using Time: %02dh %02dm %02ds" % (h, m, s))
