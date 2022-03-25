from utils.wheels import gain_index
from dataset import augumentation
import torch.utils.data as DATA
from PIL import Image
import torchvision.transforms.functional as tf
import random
import torch
import numpy as np
import os
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
from dataset.augumentation import AddPepperNoise


def k_fold_loader(i, seg, indices, tr_dataset, batch_size):
    train_indices, val_indices = gain_index(i, seg, len(tr_dataset), indices)
    print('train_indices\n', train_indices)
    print('val_indices\n', val_indices)

    train_sampler = DATA.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = DATA.sampler.SubsetRandomSampler(val_indices)

    tr_len, val_len = len(train_sampler), len(valid_sampler)
    # print(train_len, val_len)
    print("train data: {} | val data: {}".format(tr_len, val_len))
    print()

    tr_loader = DATA.DataLoader(tr_dataset,
                                drop_last=True,
                                batch_size=batch_size,
                                sampler=train_sampler,
                                num_workers=16)
    val_loader = DATA.DataLoader(tr_dataset,
                                 batch_size=1,
                                 sampler=valid_sampler,
                                 drop_last=False,
                                 num_workers=16)

    return tr_len, tr_loader, val_loader


class StandarImageFolder(ImageFolder):

    def __init__(self, root, transform=None):
        super(StandarImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        label = self.imgs[index][1]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label, path.split("/")[-1]


# class ImgMaskTransform(object):
#
#     def __init__(self, PepperNoise_p=0.95, img_size=(256, 512)):
#         self.img_standar_transform = transforms.Compose([
#             AddPepperNoise(PepperNoise_p, p=0.5),
#             transforms.Resize(img_size),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#
#         self.hflip = False
#
#     def __call__(self, img):
#         p1 = random.random()
#
#         if p1 > 0.5:
#             img = tf.hflip(img)
#             self.hflip = True
#         img = self.img_standar_transform(img)
#
#         return img, self.hflip
#

class Modified_Metric_Learning_ImageFolder(ImageFolder):

    def __init__(self, root, mask_root, transform=None):
        super(Modified_Metric_Learning_ImageFolder, self).__init__(root, transform)
        self.mask_folder = mask_root

    def __getitem__(self, index):
        pp = random.random()

        img_path = self.imgs[index][0]
        img_name, img_class = img_path.split("/")[-1], img_path.split("/")[-2]
        mask_path = self.mask_folder + '/' + img_class + '/' + img_name[:-4] + ".png"

        img = self.loader(img_path)

        if self.transform is not None:
            if pp > 0.5:
                img = tf.hflip(img)
            img = self.transform(img)

        if os.path.exists(mask_path):
            # print(mask_path)
            seg_label_mask = '/' + img_class + '/' + img_name[:-4] + ".png"
        else:
            seg_label_mask = 'None'

        cla_label = self.imgs[index][1]

        output = {
            "name": img_name,
            "img": img,
            "cla_label": cla_label,
            "seg_label_mask": seg_label_mask,
            "is_hflip": pp,
            "mask_path": mask_path,
        }
        return output

    def get_seg_label_batch(self, img_size, mask_index, hflip):

        seg_label_list = []
        embedding_mask = []
        for k in range(len(mask_index)):
            seg_label_path = mask_index[k]
            mask_is_hflip = hflip[k].item()
            if seg_label_path != 'None':
                seg_label = Image.open(self.mask_folder + seg_label_path)

                if mask_is_hflip > 0.5:
                    seg_label = tf.hflip(seg_label)

                # seg_label = Image.fromarray(np.uint8(seg_label)).resize((img_size[1], img_size[0]), Image.NEAREST)
                seg_label = transforms.Resize(img_size)(seg_label)

                seg_label_ = torch.from_numpy(np.array(seg_label))
                seg_label_list.append(seg_label_)
                embedding_mask.append(True)
            else:
                embedding_mask.append(False)
        try:
            seg_label_ten = torch.stack(seg_label_list).cuda()
        except:
            seg_label_ten = None
        return seg_label_ten, embedding_mask
