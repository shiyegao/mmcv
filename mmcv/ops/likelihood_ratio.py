import torch
import torch.jit
import torch.nn.functional as F

@torch.jit.script
def soft_likelihood_ratio(logits: torch.Tensor) -> torch.Tensor:
    delta = 1  # 0.025 set in paper
    logits = logits.double()  # we need double-precision here

    # Shift logits to stabilize log-sum-exp computation
    logits_max = logits.max(dim=1, keepdim=True)[0]
    logits_shifted = logits - logits_max

    # Compute exponentiated logits and their sum once
    logit_exp = torch.exp(logits_shifted)
    logit_exp_sum = torch.sum(logit_exp, dim=1, keepdim=True)

    # Vectorized computation of the logsumexp terms (excluding the respective class)
    logsumexp = torch.log(logit_exp_sum - logit_exp + 1e-50)

    # Compute expected probability ratio loss
    loss = -delta * torch.sum(F.softmax(logits, dim=1) * (logits_shifted - logsumexp), dim=1)
    return loss


@torch.jit.script
def hard_likelihood_ratio(logits: torch.Tensor) -> torch.Tensor:
    delta = 1  # 0.025 set in paper
    topk = torch.topk(logits, k=logits.shape[1], dim=1).values
    loss = -delta * (topk[:, 0] - torch.logsumexp(topk[:, 1:], dim=1))
    return loss

