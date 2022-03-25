import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


class DiscriminativeLoss_klintan(_Loss):
    def __init__(self, parser):
        """
        20210220
        refer to https://github.com/klintan/pytorch-lanenet/tree/master/lanenet/model
        :param embed_dim: no use in this class function
        :param delta_v:
        :param delta_d:
        :param norm:
        :param scale_var:
        :param scale_dist:
        :param scale_reg:
        """
        super(DiscriminativeLoss_klintan, self).__init__(reduction='mean')
        self.parser = parser
        self.args = self.parser.get_args()

        self.num_instance = self.args.num_instance
        self.delta_var = self.args.delta_v
        self.delta_dist = self.args.delta_d
        self.norm = 2
        self.scale_var = self.args.p_var
        self.scale_dist = self.args.p_dist
        self.scale_reg = self.args.p_reg
        print('Discriminative loss is Klintan.')
        assert self.norm in [1, 2]

    def forward(self, embedding, segLabel):
        # _assert_no_grad(target)
        if segLabel is not None:
            var_loss, dist_loss, reg_loss = self._discriminative_loss(embedding, segLabel)
        else:
            var_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
            dist_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
            reg_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)

        loss = var_loss * self.scale_var + dist_loss * self.scale_dist + reg_loss * self.scale_reg

        output = {
            "loss_var": var_loss,
            "loss_dist": dist_loss,
            "loss_reg": reg_loss,
            "loss": loss
        }

        return output

    def _discriminative_loss(self, embedding, seg_gt):
        batch_size = embedding.shape[0]
        embed_dim = embedding.shape[1]

        var_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        dist_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        reg_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)

        for b in range(batch_size):
            embedding_b = embedding[b]  # (embed_dim, H, W)
            seg_gt_b = seg_gt[b]

            labels = torch.unique(seg_gt_b)
            labels = labels[labels != 0]
            num_lanes = len(labels)
            if num_lanes == 0:
                # please refer to issue here: https://github.com/harryhan618/LaneNet/issues/12
                _nonsense = embedding.sum()
                _zero = torch.zeros_like(_nonsense)
                var_loss = var_loss + _nonsense * _zero
                dist_loss = dist_loss + _nonsense * _zero
                reg_loss = reg_loss + _nonsense * _zero
                continue

            centroid_mean = []
            for lane_idx in labels:
                seg_mask_i = (seg_gt_b == lane_idx)
                if not seg_mask_i.any():
                    continue
                embedding_i = embedding_b[:, seg_mask_i]

                mean_i = torch.mean(embedding_i, dim=1)
                centroid_mean.append(mean_i)
                # print('mean_i:', mean_i.shape)
                # print('embedding_i:', embedding_i.shape)

                # ---------- var_loss -------------
                # print('torch.norm:', torch.norm(embedding_i.permute(1, 0) - mean_i, dim=1).shape)
                var_loss = var_loss + torch.mean(F.relu(
                    torch.norm(embedding_i.permute(1, 0) - mean_i, dim=1) - self.delta_var) ** 2) / num_lanes
            centroid_mean = torch.stack(centroid_mean)  # (n_lane, embed_dim)

            if num_lanes > 1:
                centroid_mean1 = centroid_mean.reshape(-1, 1, embed_dim)
                centroid_mean2 = centroid_mean.reshape(1, -1, embed_dim)
                dist = torch.norm(centroid_mean1 - centroid_mean2, dim=2)  # shape (num_lanes, num_lanes)

                # diagonal elements are 0, now mask above delta_d
                # dist = dist + torch.eye(num_lanes, dtype=dist.dtype,
                #                         device=dist.device) * 2 * self.delta_dist
                # dist_loss = dist_loss + torch.sum(F.relu(-dist + self.delta_dist) ** 2) / (
                #         num_lanes * (num_lanes - 1)) / 2
                margin = 2 * self.delta_dist * (1.0 - torch.eye(num_lanes, dtype=dist.dtype, device=dist.device))
                dist_loss = dist_loss + torch.sum(F.relu(-dist + margin) ** 2) / (
                        num_lanes * (num_lanes - 1)) / 2

            # reg_loss is not used in original paper
            reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1))

        var_loss = var_loss / batch_size
        dist_loss = dist_loss / batch_size
        reg_loss = reg_loss / batch_size
        return var_loss, dist_loss, reg_loss