import numpy as np
import torch
import torch.nn.functional as F

# loss functions to be compared


def batch_select(mat, idx):
    mask = torch.arange(mat.size(1)).expand_as(mat).to(
        mat.device, dtype=torch.long)
    mask = (mask == idx.view(-1, 1))
    return torch.masked_select(mat, mask)


def unique_and_padding(mat, padding_idx, dim=-1):
    """Conducts unique operation along dim and pads to the same length."""
    samples, _ = torch.sort(mat, dim=dim)
    samples_roll = torch.roll(samples, -1, dims=dim)
    samples_diff = samples - samples_roll
    samples_diff[:,
                 -1] = 1  # deal with the edge case that there is only one unique sample in a row
    samples_mask = torch.bitwise_not(samples_diff == 0)  # unique mask
    samples *= samples_mask.to(dtype=samples.dtype)
    samples += (1 - samples_mask.to(dtype=samples.dtype)) * padding_idx
    samples, _ = torch.sort(samples, dim=dim)
    # shrink size to max unique length
    samples = torch.unique(samples, dim=dim)
    return samples


def loss_ce(logits, a, unused_p=None, unused_ranker_logits=None):
    """Cross entropy."""
    return -torch.mean(batch_select(F.log_softmax(logits, dim=1), a))


def loss_ips(logits,
             a,
             p,
             unused_ranker_logits=None,
             upper_limit=100,
             lower_limit=0.01):
    """IPS loss (one-stage)."""
    importance_weight = batch_select(F.softmax(logits.detach(), dim=1), a) / p
    importance_weight = torch.where(
        importance_weight > lower_limit, importance_weight,
        lower_limit * torch.ones_like(importance_weight))
    importance_weight = torch.where(
        importance_weight < upper_limit, importance_weight,
        upper_limit * torch.ones_like(importance_weight))
    importance_weight /= torch.mean(importance_weight)
    return -torch.mean(
        batch_select(F.log_softmax(logits, dim=1), a) * importance_weight)


def loss_2s(logits,
            a,
            p,
            ranker_logits,
            slate_sample_size=100,
            slate_size=30,
            temperature=np.e,
            alpha=1e-5,
            upper_limit=100,
            lower_limit=0.01):
    """Two stage loss."""
    num_logits = logits.size(1)
    rls = ranker_logits.detach()
    probs = F.softmax(logits.detach() / temperature, dim=1)
    log_probs = F.log_softmax(logits, dim=1)

    rls = torch.cat(
        [
            rls,
            torch.Tensor([float("-inf")]).to(rls.device).view(1, 1).expand(
                rls.size(0), 1)
        ],
        dim=1)
    log_probs = torch.cat(
        [log_probs,
         torch.zeros(log_probs.size(0), 1).to(log_probs.device)],
        dim=1)

    importance_weight = batch_select(F.softmax(logits.detach(), dim=1), a) / p
    importance_weight = torch.where(
        importance_weight > lower_limit, importance_weight,
        lower_limit * torch.ones_like(importance_weight))
    importance_weight = torch.where(
        importance_weight < upper_limit, importance_weight,
        upper_limit * torch.ones_like(importance_weight))
    importance_weight /= torch.mean(importance_weight)

    log_action_res = []
    sampled_slate_res = []
    for i in range(probs.size(0)):
        samples = torch.multinomial(
            probs[i], slate_sample_size * slate_size, replacement=True).view(
                slate_sample_size, slate_size)
        samples = torch.cat(
            [samples, a[i].view(1, 1).expand(samples.size(0), 1)], dim=1)
        samples = unique_and_padding(samples, num_logits)
        rp = F.softmax(
            F.embedding(samples, rls[i].view(-1, 1)).squeeze(-1), dim=1)
        lp = torch.sum(
            F.embedding(samples, log_probs[i].view(-1, 1)).squeeze(-1),
            dim=1) - log_probs[i, a[i]]
        sampled_slate_res.append(
            torch.mean(importance_weight[i] * rp[samples == a[i]] * lp))
        log_action_res.append(
            torch.mean(importance_weight[i] * rp[samples == a[i]]))
    loss = -torch.mean(
        torch.stack(log_action_res) * batch_select(log_probs, a))
    loss += -torch.mean(torch.stack(sampled_slate_res)) * alpha
    return loss
