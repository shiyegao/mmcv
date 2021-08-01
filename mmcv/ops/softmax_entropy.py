import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def info_max(x: torch.Tensor) -> torch.Tensor:
    loss_ent = x.softmax(1)
    loss_ent = -torch.sum(loss_ent * torch.log(loss_ent + 1e-5), dim=1).mean(0)
    loss_div = x.softmax(1).mean(0)
    loss_div = torch.sum(loss_div * torch.log(loss_div + 1e-5))
    return loss_ent + loss_div


class SoftmaxEntropy(nn.Module):
    """Softmax entropy

    Args:
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(SoftmaxEntropy, self).__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                cls_score,
                gt_label,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = softmax_entropy(cls_score)
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return self.loss_weight * loss


class InfoMax(nn.Module):
    """Softmax entropy

    Args:
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(InfoMax, self).__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                cls_score,
                gt_label,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = info_max(cls_score)
        if reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'mean':
            loss = loss.mean()
        return self.loss_weight * loss