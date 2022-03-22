import torch
import numpy as np
import os

from utils.metrics import pred_prob2pred_label
from loss.loss_total import caculate_all_loss
from utils.visulize import show_image
from utils.wheels import mkfile
from utils.display_output import DisplayOutput


def val_test(per_epoch,
             pseudo,
             net,
             val_dataset,
             data_loader,
             disc_loss,
             seg_loss,
             cla_loss,
             p_seg,
             p_discriminative,
             p_cla,
             save_pre,
             is_val=True):
    net.eval()
    total_loss, total_correct_1, total_correct_2, total_num = 0.0, 0.0, 0.0, 0
    five_iou, total_each_acc, total_each_iou, total_iou_no_back = [], [], [], []

    img_tensor, mask_tensor, pre_tensor = [], [], []
    ff_name = ''
    dd = DisplayOutput(root=save_pre + '/EP' + str(per_epoch))
    with torch.no_grad():
        pred_label_all, pred_prob_all, gt_label = [], [], []
        pred_prob, pred_label = 0, 0
        total_num_mask = 0
        for step, data in enumerate(data_loader, start=0):
            # print(data['seg_label_mask'])
            image, cla_label = data['img'].cuda(), data['cla_label'].cuda()  # data['name']是list
            # print(data['name'])
            # print(data['cla_label'].type(), data['cla_label'])

            resnet_y, feature, em = net(image)

            seg_label, _mask = val_dataset.get_seg_label_batch(img_size=(data['img'].shape[2],
                                                                         data['img'].shape[3]),
                                                               mask_index=data['seg_label_mask'],
                                                               hflip=data['is_hflip'])

            _seg_loss, _cla_loss, dict_dis_loss, num_mask, val_seg_acc = caculate_all_loss(epoch=per_epoch,
                                                                                           using_pseudo=pseudo[0],
                                                                                           pseudo_threshold=pseudo[1],
                                                                                           pseudo_ep=pseudo[2],
                                                                                           seg_label=seg_label,
                                                                                           cla_label=cla_label,
                                                                                           seg_pred=feature,
                                                                                           seg_pred_after_mask=feature[
                                                                                               _mask],
                                                                                           em_after_mask=em[_mask],
                                                                                           cla_pred_y=resnet_y,
                                                                                           disc_loss_func=disc_loss,
                                                                                           seg_loss_func=seg_loss,
                                                                                           cla_loss_func=cla_loss)

            if num_mask != 0:
                total_each_iou.append(val_seg_acc[1])
                total_iou_no_back.append(val_seg_acc[2])
                total_each_acc.append(val_seg_acc[3])
                five_iou.append(val_seg_acc[0])

            loss = p_seg * _seg_loss + p_discriminative * dict_dis_loss[
                'loss'] + p_cla * _cla_loss
            total_num_mask += num_mask
            pred_prob, pred_label = pred_prob2pred_label(resnet_y)
            pred_prob_all.append(pred_prob)
            pred_label_all.append(pred_label)
            gt_label.append(cla_label.cpu())

            total_num += image.size(0)
            total_loss += loss.item() * image.size(0)
            prediction = torch.argsort(resnet_y, dim=-1, descending=True)
            total_correct_1 += torch.sum(
                (prediction[:, 0:1] == cla_label.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_2 += torch.sum(
                (prediction[:, 0:2] == cla_label.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            if per_epoch % 20 == 0:
                save_pre_dir = save_pre + '/pre_all' + '/EP' + str(per_epoch)
                mkfile(save_pre_dir)
                show_image(prediction=feature, save_dir=save_pre_dir, name=data['name'],
                           label=str(data['cla_label'].item()))
            if not seg_label is None:
                ff_name += '_' + str(data['cla_label'].item()) + str(pred_label.item()) + data['name'][0].split('.')[0]
                img_tensor.append(data['img'][_mask])
                mask_tensor.append(seg_label)
                max_val, max_index = torch.max(torch.softmax(feature[_mask], dim=1), dim=1)
                pre_tensor.append(max_index)

        pp, ff = dd(data_img=torch.cat(img_tensor, dim=0), data_mask=torch.cat(mask_tensor, dim=0),
                    data_pre=torch.cat(pre_tensor, dim=0))
        save_fig_dir = save_pre + '/fig'
        mkfile(save_fig_dir)
        ff.savefig(save_fig_dir + '/EP' + str(per_epoch) + ff_name + '.png')

        print('[EVAL]\n{} Loss: {:.4f} ACC@1: {:.2f}% ACC@2: {:.2f}%'
              .format('Val' if is_val else 'Test', total_loss / total_num,
                      total_correct_1 / total_num * 100, total_correct_2 / total_num * 100))
        print('mean_pixel_acc: {:.4f} | mean_IU: {:.4f} | mean_IU_no_back : {:.4f}'.format(
            np.mean(total_each_acc), np.mean(total_each_iou), np.mean(total_iou_no_back)))

        pred_probs = np.concatenate(pred_prob_all)
        pred_labels = np.concatenate(pred_label_all)
        gt_labels = np.concatenate(gt_label)

        dict_ = {
            "val_loss": torch.true_divide(total_loss, total_num),
            "val_acc_1": torch.true_divide(total_correct_1, total_num),
            "val_acc_2": torch.true_divide(total_correct_2, total_num),
            "val_pred_probs": pred_probs,
            "val_pred_labels": pred_labels,
            'val_gt_labels': gt_labels,
            "five_iou": np.stack(five_iou),
            "mean_pixel_acc": np.mean(total_each_acc),
            "mean_IU": np.mean(total_each_iou),
            "mean_IU_no_back": np.mean(total_iou_no_back),
            "predict_figure": ff,
        }
    return dict_


def test(net, data_loader, criterion, is_val=False):
    net.eval()
    total_loss, total_correct_1, total_correct_2, total_num = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        pred_label_all, pred_prob_all, gt_label = [], [], []
        pred_prob, pred_label = 0, 0
        for step, [data, label, name] in enumerate(data_loader, start=0):
            # print('in test:', data.shape)
            resnet_y, _, _ = net(data.cuda())
            loss = criterion(resnet_y, label.cuda())

            pred_prob, pred_label = pred_prob2pred_label(resnet_y)
            pred_prob_all.append(pred_prob)
            pred_label_all.append(pred_label)
            gt_label.append(label)

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(resnet_y, dim=-1, descending=True)  # 从大到小排列返回索引
            total_correct_1 += torch.sum(
                (prediction[:, 0:1] == label.cuda().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_2 += torch.sum(
                (prediction[:, 0:2] == label.cuda().unsqueeze(dim=-1)).any(dim=-1).float()).item()

        print('{} Loss: {:.4f} ACC@1: {:.2f}% ACC@2: {:.2f}%'
              .format('Val' if is_val else 'Test', total_loss / total_num,
                      total_correct_1 / total_num * 100, total_correct_2 / total_num * 100))

        pred_probs = np.concatenate(pred_prob_all)
        pred_labels = np.concatenate(pred_label_all)
        gt_labels = np.concatenate(gt_label)

    return total_loss / total_num, total_correct_1 / total_num, total_correct_2 / total_num, pred_probs, pred_labels, gt_labels
