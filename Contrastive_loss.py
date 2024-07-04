"""
Code taken from ---
Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
It also supports the unsupervised contrastive loss in SimCLR
"""
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def init_prototypes(net, eval_loader, device):
    net.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for _, (inputs, labels, _,_) in enumerate(eval_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            features = net(inputs, forward_pass='proj')
            all_features.append(features)
            all_labels.append(labels)
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    net.init_prototypes(all_features, all_labels)

@torch.no_grad()
def noise_correction(pt_outputs, dm_outputs, labels, indices, device):
    pt_outputs, dm_outputs, labels, indices = (pt_outputs.to(device), dm_outputs.to(device),
                                               labels.to(device), indices.to(device))
    # noise cleaning for clustering
    alpha = 0.5
    soft_labels = alpha * F.softmax(pt_outputs, dim=1) + (1 - alpha) * F.softmax(dm_outputs, dim=1)

    # assign a new pseudo label
    max_score, hard_label = soft_labels.max(1)
    correct_idx = max_score > 0.8
    labels[correct_idx] = hard_label[correct_idx]

    return labels

@torch.no_grad()
def build_mask_step(outputs, k, labels, device):
    outputs, labels = outputs.to(device), labels.to(device)

    tops = torch.zeros_like(outputs, device=device)
    if k == 0:
        topk = torch.topk(outputs, 1, dim=1)[1]
        # make the topk of the outputs to be 1, others to be 0
        tops = torch.scatter(tops, 1, topk, 1)
    else:
        topk = torch.topk(outputs, k, dim=1)[1]
        # make the topk of the outputs to be 1, others to be 0
        tops = torch.scatter(tops, 1, topk, 1)

        tops = torch.scatter(tops, 1, labels.unsqueeze(dim=1), 1)

    neg_samples = torch.ones(len(outputs), len(outputs), dtype=torch.float, device=device)

    # conflict matrix, where conflict[i][j]==0 means the i-th and j-th class do not have overlap topk,
    # can be used as negative pairs
    conflicts = torch.matmul(tops, tops.t())
    # negative pairs: (conflicts == 0) or (conflicts != 0 and neg_samples == 0)
    neg_samples = neg_samples * conflicts
    # make a mask metrix, where neg_samples==0, the mask is -1 (negative pairs), otherwise 0 (neglect pairs)
    mask = torch.where(neg_samples == 0, -1, 0)
    # make the diagonal of the mask to be 1 (positive pairs)
    mask = torch.where(torch.eye(len(outputs), device=device) == 1, 1, mask)
    return mask


class SemiLoss1(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self):
        super(SemiLoss1, self).__init__()

    def linear_rampup(self, lambda_u, current, warm_up, rampup_length=16):
        current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
        return lambda_u * float(current)

    def forward(self, outputs_x, targets_x, outputs_u, targets_u, lambda_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, self.linear_rampup(lambda_u, epoch, warm_up)


class NegEntropy1(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def entropy(x, input_as_probabilities):
    """
    Helper function to compute the entropy over the batch

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """
    EPS = 1e-8

    if input_as_probabilities:
        x_ = torch.clamp(x, min=EPS)
        b = x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)

    if len(b.size()) == 2:  # Sample-wise entropy
        return -b.sum(dim=1).mean()
    elif len(b.size()) == 1:  # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))


class InfoNCELoss(nn.Module):
    def __init__(self, temperature, batch_size, world_size=1, flat=False, n_views=2):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.world_size = world_size
        self.flat = flat
        self.n_views = n_views

    def forward(self, features):
        labels = torch.cat([torch.arange(len(features)/2) for _ in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()   # bool方阵，batch*batch
        labels = labels.to(features.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.n_views * self.batch_size * self.world_size, self.n_views * self.batch_size * self.world_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device) # 默认返回二维数组，对角为1
        labels = labels[~mask].view(labels.shape[0], -1)    # bsz*(bsz-1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)   # bsz*(bsz-1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # labels一致的标记出来

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1) # labels不一致的标记出来

        if self.flat:
            logits = (negatives - torch.sum(positives, dim=1).view(labels.shape[0], -1)) / self.temperature
            labels = torch.zeros_like(logits).to(features.device)
            # logits = (negatives - positives) / self.temperature
            # labels = torch.zeros(positives.shape[0], dtype=torch.long).to(features.device)
            # labels[:, :self.n_views - 1] = 1
        else:
            logits = torch.cat([positives, negatives], dim=1) / self.temperature
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)
        return logits, labels


class PLRLoss(nn.Module):
    def __init__(self, temperature=1, base_temperature=1, flat=False):
        super(PLRLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.flat = flat

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        Args:
            features: shape of [bsz, n_views, ...]
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz],
                mask_{i,j}=1 if sample j has the same class as sample i,
                mask_{i,j}=-1 if sample j has different classes with sample i,
                mask_{i,j}=0 if sample j is not used for contrast.
                Can be asymmetric.
        Returns:
            A loss scalar.
        """

        def _get_loss(_mask, _anchor_dot_contrast):
            # create negative_mask from mask where mask_{i,j}==-1
            negative_mask = torch.eq(_mask, -1).float().to(device)
            # create positive_mask from mask where mask_{i,j}==1
            positive_mask = torch.eq(_mask, 1).float().to(device)
            # create neglect_mask where mask_{i,j}==0
            neglect_mask = torch.eq(_mask, 0).float().to(device)
            non_neglect_mask = 1 - neglect_mask

            # compute logits for non-neglect cases
            _anchor_dot_contrast *= non_neglect_mask

            if self.flat:
                # follow FlatNCE in https://arxiv.org/abs/2107.01152
                # filter out no negative case samples, which lead to nan loss
                has_negative = torch.nonzero(negative_mask.sum(1)).squeeze(1)
                negative_mask = negative_mask[has_negative]
                logits = (_anchor_dot_contrast - torch.sum(_anchor_dot_contrast * positive_mask, dim=1,
                                                           keepdim=True)) / self.temperature
                logits = logits[has_negative]

                exp_logits = torch.exp(logits) * negative_mask
                v = torch.log(exp_logits.sum(1, keepdim=True))
                loss_implicit = torch.exp(v - v.detach())  # all equal to 1

                # loss_explicit = - torch.log(1 / (1 + torch.sum(exp_logits, dim=1, keepdim=True)))  # just for watching
                # loss = loss_implicit.mean() - 1 + loss_explicit.mean().detach()
                _loss = loss_implicit.mean()
            else:
                # compute logits for non-neglect cases
                _anchor_dot_contrast = torch.div(_anchor_dot_contrast, self.temperature)
                # for numerical stability
                logits_max, _ = torch.max(_anchor_dot_contrast, dim=1, keepdim=True)
                logits = _anchor_dot_contrast - logits_max.detach()

                # compute log_prob
                exp_logits = torch.exp(logits) * non_neglect_mask
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

                # compute mean of log-likelihood over positive
                mean_log_prob_pos = (positive_mask * log_prob).sum(1) / positive_mask.sum(1)

                # loss
                _loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
                _loss = _loss.mean()
            return _loss

        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            mask = torch.where(mask == 0, -1, mask)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            mask = torch.where(mask == 0, -1, mask)
        else:
            pass

        features = F.normalize(features, dim=-1)
        # noise-tolerant contrastive loss
        anchor_count = contrast_count = features.shape[1]
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        mask = mask * (1 - torch.eye(batch_size * anchor_count, dtype=torch.float32).to(device))

        anchor_feature = contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T)
        loss = _get_loss(mask, anchor_dot_contrast)

        return loss


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.05, contrast_mode='all',
                 base_temperature=0.05):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        anchor_dot_contrast[anchor_dot_contrast==float('inf')] = 1

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits[logits==float('inf')] = 1


        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        mask[mask==float('inf')] = 1

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        log_prob[log_prob==float('inf')] = 1

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
