"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np

import torch as th
import torch.nn.functional as F

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    # print(logvar2.shape)
    # temp1 = 0.5 * (-1.0 + logvar2 - logvar1 + th.exp(logvar1 - logvar2))
    # print(f'const = {temp1.mean()}, coef={(th.exp(-logvar2) * 0.5).mean()}, mse={((mean1 - mean2) ** 2).mean().item()}')

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

def gaussian_density(x, *, means, log_scales):
    from torch.distributions import Normal
    normal_dist = Normal(means, log_scales.exp())
    logp = normal_dist.log_prob(x)
    return logp 


def discretized_text_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    print(x.shape, means.shape)
    # assert x.shape == means.shape == log_scales.shape
    print(x, means) 
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs




def rounding_losses(hat_x0, tokens, emb_matrix, tau=0.1, eps=1e-8):
    """
    hat_x0: (B, L, d) predicted embeddings (x0_pred from the model)
    tokens: (B, L) ground-truth token ids
    emb_matrix: (V, d) vocabulary embedding matrix
    tau: temperature for softmax

    Returns: L_anchor, L_qCE
    """
    B, L, d = hat_x0.shape
    V, dE = emb_matrix.shape
    assert d == dE, "Embedding dim mismatch"

    h = hat_x0.reshape(-1, d)   # (N, d)
    t = tokens.reshape(-1)      # (N,)

    # normalize for cosine similarity
    h_norm = F.normalize(h, dim=1)
    E_norm = F.normalize(emb_matrix, dim=1)

    # logits = cosine similarities (N,V)
    logits = h_norm @ E_norm.T

    # ---- Anchor MSE ----
    nearest_idx = logits.argmax(dim=1)   # (N,)
    anchor_targets = emb_matrix[nearest_idx].detach()  # stop-grad
    L_anchor = F.mse_loss(h, anchor_targets)

    # ---- Quantization-aware CE ----
    logits_scaled = logits / (tau + eps)
    L_qCE = F.cross_entropy(logits_scaled, t, ignore_index=-100)

    return L_anchor, L_qCE
