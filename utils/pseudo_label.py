import torch


def pseudo_label(y_weak_preds, threshold=0.7):
    y_weak_probas = torch.softmax(y_weak_preds, dim=1).detach()
    max_y_weak_probas, y_pseudo = torch.max(y_weak_probas, dim=1)
    unsup_loss_mask = (max_y_weak_probas >= threshold)
    return y_pseudo, unsup_loss_mask