import torch
import numpy as np

from utils.wheels import adjust_learning_rate
from loss.loss_total import caculate_all_loss
from utils.metrics import compute_score, hist_info


def pseudo_ups(f_pass, model, inputs, kappa_n, tau_n, tau_p, kappa_p, uncertainty, temp_nl):
    """
    from https://github.com/nayeemrizve/ups
    :param f_pass:
    :param model:
    :param inputs:
    :param kappa_n:
    :param tau_n:
    :param tau_p:
    :param kappa_p:
    :param no_uncertainty:
    :param temp_nl:
    :return: pseudo_target, selected_p, selected_n
    """

    out_prob = []
    out_prob_nl = []

    if uncertainty:
        enable_dropout(model)
    else:
        f_pass = 1

    with torch.no_grad():
        for _ in range(f_pass):
            _, output, _ = model(inputs)
            # print(outputs.shape)
            # print(outputs['seg_pre'])
            # print(torch.softmax(outputs['seg_pre'], dim=1))
            # print(torch.sigmoid(outputs['seg_pre']))
            out_prob.append(torch.softmax(output, dim=1))
            out_prob_nl.append(
                torch.softmax(output / temp_nl, dim=1))
            # out_prob.append(torch.softmax(outputs, dim=1))
            # out_prob_nl.append(
            #     torch.softmax(outputs / temp_nl, dim=1))
    out_prob = torch.stack(out_prob)
    out_prob_nl = torch.stack(out_prob_nl)  # torch.Size([f_pass, 2, 5, 128, 256])
    out_std = torch.std(out_prob, dim=0)  # torch.Size([2, 5, 128, 256])
    out_std_nl = torch.std(out_prob_nl, dim=0)
    out_prob = torch.mean(out_prob, dim=0)  # torch.Size([2, 5, 128, 256])
    out_prob_nl = torch.mean(out_prob_nl, dim=0)

    max_value, max_idx = torch.max(out_prob, dim=1)
    max_value_nl, max_idx_nl = torch.max(out_prob_nl, dim=1)
    min_value, min_idx = torch.min(out_prob_nl, dim=1)

    max_std = out_std.view(-1, 5).gather(1, max_idx.view(-1, 1)).reshape(max_idx.shape)
    max_std_nl = out_std_nl.view(-1, 5).gather(1, max_idx_nl.view(-1, 1)).reshape(max_idx_nl.shape)
    # print('max_value', max_value.shape)
    # print('max_std', max_std.shape)
    # print('max_idx_nl', max_idx_nl.shape)
    # print('max_std_nl', max_std_nl.shape)

    # selecting positive/negative pseudo-labels
    if uncertainty:  # use uncertainty in the pesudo-label selection, default true
        selected_idx = (max_value >= tau_p) * (max_std.squeeze(1) < kappa_p)
        selected_idx_nl = (max_value_nl < tau_n) * (max_std_nl.squeeze(1) < kappa_n)
    else:
        selected_idx = max_value >= tau_p
        selected_idx_nl = max_value_nl < tau_n
    return min_idx, max_idx, selected_idx, selected_idx_nl


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def train(args,
          per_epoch,
          net,
          train_dataset,
          data_loader,
          train_optimizer,
          disc_loss_func,
          seg_loss_func,
          cla_loss_func):
    net.train()
    total_niters = args.max_epoch * len(data_loader)
    lr_list = []
    adjust_lr = 0.0
    total_loss, total_correct_1, total_correct_2, total_num = 0.0, 0.0, 0.0, 0

    per_loss, per_seg_loss, per_cla_loss = [], [], []
    tt_seg, tt_disc = [], []
    per_discriminative_loss, per_var_loss, per_dist_loss, per_reg_loss = [], [], [], []
    total_each_acc, total_each_iou, total_iou_no_back = [], [], []

    total_num = 0
    t_result = {}
    # print('EP:', per_epoch)
    for step, data in enumerate(data_loader, start=0):
        image, cla_label = data['img'].cuda(), data['cla_label'].cuda()

        current_idx = (per_epoch - 1) * len(data_loader) + step
        train_optimizer.zero_grad()
        adjust_lr = adjust_learning_rate(train_optimizer, current_idx, base_lr=args.learning_rate,
                                         total_niters=total_niters,
                                         lr_power=0.9)

        resnet_y, feature, em = net(image)

        seg_label, _mask = train_dataset.get_seg_label_batch(img_size=(data['img'].shape[2],
                                                                       data['img'].shape[3]),
                                                             mask_index=data['seg_label_mask'],
                                                             hflip=data['is_hflip'])

        num_seg_label, acc = 0, []
        if not seg_label is None:
            num_seg_label = seg_label.shape[0]
        if seg_label is None and args.using_pseudo and per_epoch >= args.start_pseudo_epoch:
            # print('i am using pseudo.')
            net.eval()
            pseudo_target_n, pseudo_target_p, mask_p, mask_n = pseudo_ups(f_pass=10, model=net, inputs=image,
                                                                          kappa_n=args.kappa_n, tau_n=args.tau_n,
                                                                          tau_p=args.tau_p, kappa_p=args.kappa_p,
                                                                          uncertainty=args.uncertainty, temp_nl=2)
            net.train()
            seg_loss_p = seg_loss_func(feature, pseudo_target_p, mask_p)
            if args.negative_pseudo:
                seg_loss_n = seg_loss_func(feature, pseudo_target_p, mask_n, is_negative=True)
            else:
                seg_loss_n = torch.tensor(0, dtype=feature.dtype, device=feature.device)
            if mask_p.float().sum() == 0 and mask_n.float().sum() == 0:
                seg_loss = torch.tensor(0, dtype=feature.dtype, device=feature.device)
            elif mask_p.float().sum() != 0 and mask_n.float().sum() != 0:
                seg_loss = (seg_loss_p + seg_loss_n) / 2
            else:
                seg_loss = seg_loss_p + seg_loss_n
        elif not seg_label is None:
            # print('i have gt label.')
            seg_loss = seg_loss_func(feature[_mask], seg_label.to(dtype=torch.int64))

            _, _seg_pred_after_mask = torch.max(feature[_mask], dim=1)
            a, b, c = hist_info(n_cl=5, pred=_seg_pred_after_mask.cpu().detach().numpy(),
                                gt=seg_label.cpu().detach().numpy())
            acc = compute_score(a, c, b)  # acc = each_iu, mean_IU, mean_IU_no_back, mean_pixel_acc
            # print('segloss:{}\nacc:\n{}'.format(seg_loss, acc))
        else:
            # print('i am not using pseudo and not calculating seg loss.')
            seg_loss = torch.tensor(0, dtype=feature.dtype, device=feature.device)
        disc_loss = disc_loss_func(embedding=em[_mask], segLabel=seg_label)
        cla_loss = cla_loss_func(resnet_y, cla_label)

        loss = args.p_seg * seg_loss + args.p_disc * disc_loss[
            'loss'] + args.p_cla * cla_loss
        loss.backward()
        train_optimizer.step()
        lr_list.append(adjust_lr)

        tt_seg.append(args.p_seg * seg_loss.item())
        tt_disc.append(args.p_disc * disc_loss['loss'].item())

        per_loss.append(loss.item())
        per_cla_loss.append(cla_loss.item())
        # print('num_mask:', num_mask)
        if num_seg_label != 0:
            per_discriminative_loss.append(args.p_disc * disc_loss['loss'].item())
            per_var_loss.append(args.p_var * disc_loss['loss_var'].item())
            # print('###:', dict_dis_loss['loss_dist'])
            per_dist_loss.append(args.p_dist * disc_loss['loss_dist'].item())
            per_reg_loss.append(args.p_reg * disc_loss['loss_reg'].item())
            per_seg_loss.append(args.p_seg * seg_loss.item())

            total_each_iou.append(acc[1])
            total_iou_no_back.append(acc[2])
            total_each_acc.append(acc[3])

        total_num += image.size(0)
        prediction = torch.argsort(resnet_y, dim=-1, descending=True)
        total_correct_1 += torch.sum((prediction[:, 0:1] == cla_label.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        total_correct_2 += torch.sum((prediction[:, 0:2] == cla_label.unsqueeze(dim=-1)).any(dim=-1).float()).item()

    print('[TRAIN]\nvar loss: {:.4f} | dist loss: {:.4f} | reg loss: {:.4f}'.format(np.mean(per_var_loss),
                                                                                    np.mean(per_dist_loss),
                                                                                    np.mean(per_reg_loss)))

    print(
        'total Loss: {:.4f} | seg loss: {:.4f}/{:.4f} | discriminative loss: {:.4f}/{:.4f} | cla loss: {:.4f}'.format(
            np.mean(per_loss),
            np.mean(per_seg_loss),
            np.mean(tt_seg),
            np.mean(per_discriminative_loss),
            np.mean(tt_disc),
            np.mean(per_cla_loss)))

    print('mean_pixel_acc: {:.4f} | mean_IU: {:.4f} | mean_IU_no_back : {:.4f}'.format(
        np.mean(total_each_acc), np.mean(total_each_iou), np.mean(total_iou_no_back)))
    print('ACC@1: {:.2f}% ACC@2: {:.2f}%'.format(torch.true_divide(total_correct_1, total_num) * 100,
                                                 torch.true_divide(total_correct_2, total_num) * 100))

    t_result['lr'] = lr_list
    t_result['loss'] = np.mean(per_loss)
    t_result['acc_1'] = torch.true_divide(total_correct_1, total_num)
    t_result['acc_2'] = torch.true_divide(total_correct_2, total_num)
    t_result['loss_seg'] = np.mean(tt_seg)

    t_result['loss_var'] = np.mean(per_var_loss)
    t_result['loss_dist'] = np.mean(per_dist_loss)
    t_result['loss_reg'] = np.mean(per_reg_loss)
    t_result['loss_em'] = np.mean(per_discriminative_loss)

    t_result['loss_cla'] = np.mean(per_cla_loss)
    return t_result
