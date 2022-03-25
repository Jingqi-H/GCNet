import torch
from utils.wheels import pre2mask
from utils.metrics import compute_score, hist_info
from utils.pseudo_label import pseudo_label


def caculate_all_loss(epoch,
                      using_pseudo,
                      pseudo_threshold,
                      pseudo_ep,
                      seg_label,
                      cla_label,
                      seg_pred,
                      seg_pred_after_mask,
                      em_after_mask,
                      cla_pred_y,
                      disc_loss_func,
                      seg_loss_func,
                      cla_loss_func):
    num_seg_label, acc = 0, []

    if not seg_label is None:
        num_seg_label = seg_label.shape[0]

    seg_loss = torch.tensor(0, dtype=seg_pred.dtype, device=seg_pred.device)
    if not seg_label is None:
        seg_loss = seg_loss_func(seg_pred_after_mask, seg_label.to(dtype=torch.int64))

        _, _seg_pred_after_mask = torch.max(seg_pred_after_mask, dim=1)
        a, b, c = hist_info(n_cl=5, pred=_seg_pred_after_mask.cpu().detach().numpy(),
                            gt=seg_label.cpu().detach().numpy())
        # print('@\n{}@\n{}@\n{}'.format(a, b, c))
        acc = compute_score(a, c, b)  # acc = each_iu, mean_IU, mean_IU_no_back, mean_pixel_acc

    else:
        # print(seg_label)
        if using_pseudo and epoch >= pseudo_ep:
            # print('using pseudo_label.')
            # seg_pseudo_label = pre2mask(seg_pred)
            # seg_loss = seg_loss_func(outputs=seg_pred, targets=seg_pseudo_label)
            seg_pseudo_label, seg_loss_mask = pseudo_label(seg_pred, threshold=pseudo_threshold)
            seg_loss = seg_loss_func(outputs=seg_pred, targets=seg_pseudo_label, seg_mask=seg_loss_mask)

    dict_loss = disc_loss_func(embedding=em_after_mask, segLabel=seg_label)
    # print(cla_pred_y.shape, cla_label.cuda().shape)
    c_loss = cla_loss_func(cla_pred_y, cla_label.cuda())

    return seg_loss, c_loss, dict_loss, num_seg_label, acc


def pseudo_ups(f_pass, model, inputs, kappa_n, tau_n, tau_p, kappa_p, uncertainty, temp_nl):
    """
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
            # outputs = torch.rand([2, 5, 128, 256])
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


