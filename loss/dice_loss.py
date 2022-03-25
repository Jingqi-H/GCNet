import torch.nn as nn
import torch
from torch.nn import functional as F


class MultiDiceLoss(object):
    def __init__(self, num_classes=5, class_weights=None):
        """
        :param num_classes:
        :param class_weights:
        """
        super(MultiDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights

    def __call__(self, outputs, targets, seg_mask=None, is_negative=False):
        """
        ouputs: NxCxHxW (should before softmax)
        targets: NxHxW
        :param outputs: net(img)--> [bs, num_instance, h, w]
        :param targets: --> [bs, h, w]
        :return:
        """
        loss_dice = 0
        smooth = 1.
        outputs = F.softmax(outputs, dim=1)
        if is_negative:
            outputs = 1 - outputs
            outputs = torch.clamp(outputs, 1e-7, 1.0)
        for cls in range(self.num_classes):
            jaccard_target = (targets == cls).float()
            jaccard_output = outputs[:, cls]

            if not seg_mask is None:
                jaccard_target = jaccard_target[seg_mask]
                jaccard_output = jaccard_output[seg_mask]
                # print(jaccard_target, jaccard_output)
            if self.class_weights is not None:
                w = self.class_weights[cls]
            else:
                w = 1.
            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()
            #                loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
            loss_dice += w * (1 - (2. * intersection + smooth) / (union + smooth))
            # three kinds of loss formulas: (1) 1 - iou (2) -iou (3) -torch.log(iou)
        return loss_dice / self.num_classes


if __name__ == '__main__':
    # gt = torch.rand([1, 5, 4, 4]).cuda()
    pre = torch.randn([2, 5, 4, 4]).cuda()
    # _, gt = torch.max(gt, dim=1)
    print(pre.shape)

    gt = torch.tensor([[[4, 0, 0, 3],
                        [4, 4, 4, 3],
                        [4, 0, 1, 1],
                        [1, 1, 4, 0]],

                       [[4, 0, 0, 3],
                        [4, 4, 4, 3],
                        [4, 0, 1, 1],
                        [1, 1, 4, 0]]
                       ], device='cuda:0')
    print(gt.shape)

    mask = torch.tensor([[[True, False, False, False],
                          [False, False, False, False],
                          [True, False, True, False],
                          [False, True, False, False]],

                         [[True, False, True, True],
                          [False, True, False, False],
                          [False, False, True, False],
                          [False, True, False, False]]], device='cuda:0')

    loss = MultiDiceLoss(num_classes=5)
    a = loss(outputs=pre, targets=gt)
    b = loss(outputs=pre, targets=gt, seg_mask=mask)
    print('mask=none: {}\nmask!=none: {}'.format(a, b))

    # cc = torch.unique(gt).cpu().numpy()
    # print(cc)
    # for cla in cc:
    #     print(cla)
