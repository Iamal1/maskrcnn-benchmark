import torch
import numpy as np
import torch.nn.functional as F

def balanced_l1_loss(pred,
                     target,
                     weight=None,
                     alpha=0.5,
                     gamma=1.5,
                     beta=1.0,
                     avg_factor=None,
                     reduction='none'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0

    diff = torch.abs(pred - target)
    b = np.e**(gamma / alpha) - 1
    loss = torch.where(
        diff < beta,
        alpha / b * (b * diff + 1) * torch.log(b * diff / beta + 1) -
        alpha * diff, gamma * diff + gamma / b - alpha * beta)

    #NOTE not supported in torch 0.4.0
    reduction = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    # in smooth_l1 it uses mean(), assume no diff
    if reduction == 1:
        loss = loss.sum() / pred.numel()
    elif reduction == 2:
        loss = loss.sum()

    # not needed.
    # if weight is not None:
    #     loss = torch.sum(loss * weight)[None]
    #     if avg_factor is None:
    #         avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6

    # if avg_factor is not None:
    #     loss /= avg_factor

    return loss