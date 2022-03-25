import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


class DisplayOutput(object):
    def __init__(self, root):
        self.root = root

    def __call__(self, data_img, data_mask, data_pre, name=None):
        data_img = data_img.cpu().clone()
        data_mask = data_mask.cpu().clone()
        data_pre = data_pre.cpu().clone()

        plt.cla()
        plt.close('all')
        figure, axes = plt.subplots(data_mask.shape[0], 4, figsize=(8, data_mask.shape[0]))
        figure.tight_layout()
        for i in range(data_mask.shape[0]):
            img1_ten = data_img[i]
            gt1_ten = data_mask[i]
            pre1_ten = data_pre[i]
            img_show = Image.fromarray(np.uint8(self.img_ten2arr(img1_ten)), mode='RGB')
            gt_show = self.mask_ten2color(gt1_ten)
            pre_show = self.mask_ten2color(pre1_ten)

            axes[i][0].imshow(img_show)
            axes[i][0].axis('off')
            axes[i][1].imshow(gt_show)
            axes[i][1].axis('off')

            axes[i][2].imshow(pre_show)
            axes[i][2].axis('off')

            axes[i][3].imshow(img_show, alpha=1)
            axes[i][3].imshow(pre_show, alpha=0.4)
            axes[i][3].axis('off')

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

        image = Image.fromarray(np.uint8(seg_img))
        return image

    def img_ten2arr(self, input_image, imtype=np.uint8):
        """
        from https://www.cnblogs.com/wanghui-garcia/p/11393076.html
        """
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if not isinstance(input_image, np.ndarray):
            if isinstance(input_image, torch.Tensor):
                image_tensor = input_image.data
            else:
                return input_image
            image_numpy = image_tensor.cpu().float().numpy()
            if image_numpy.shape[0] == 1:
                image_numpy = np.tile(image_numpy, (3, 1, 1))
            for i in range(len(mean)):
                image_numpy[i] = image_numpy[i] * std[i] + mean[i]
            image_numpy = image_numpy * 255
            image_numpy = np.transpose(image_numpy, (1, 2, 0))
        else:
            image_numpy = input_image
        return image_numpy.astype(imtype)
