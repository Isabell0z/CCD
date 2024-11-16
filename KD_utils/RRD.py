import torch


def compute_RRD_loss(pred_t, pred_s, K):
    _, interested_idx = torch.topk(pred_t, K, dim=1)
    mask = torch.zeros_like(pred_t).to(pred_t.device)
    mask.scatter_(1, interested_idx, 1)
    interested_r = pred_s * mask
    uninterested_r = pred_s * (1 - mask)

    above = torch.sum(interested_r, dim=1, keepdims=True)

    below_1 = interested_r.flip(-1).exp().cumsum(dim=1)
    below_2 = uninterested_r.exp().sum(dim=1, keepdims=True)
    below = (below_1 + below_2).log().sum(dim=1, keepdims=True)

    return -(above - below).sum()
