"""
Created on: 2024, May 10th
Author: 'Cecover' on GitHub

Title: Scalable Diffusion Models with Transformers
Framework used: PyTorch
Code credits:
    - https://github.com/chuanyangjin/fast-DiT/blob/main/diffusion/diffusion_utils.py
    - https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
    - https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
    - https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
"""

import torch
import numpy as np


def kl_score(mean1, logvar1, mean2, logvar2) -> torch.Tensor:
    """
    Compute the KL divergence between two gaussian distributions.
    The shapes are automatically broadcasted, thus we can compare batches to scalars.
    """

    tensor = None

    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break

    assert tensor is not None, "At least one argument must be a Tensor!"

    # Force variances to be Tensors.
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.Tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + (logvar2 - logvar1)
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) ** torch.exp(-logvar1 - logvar2)
    )


def approximate_std_cdf(x):
    """
    Approximating CDF of a standard Gaussian.
    """

    return 0.5 * (
        1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
    )


def continuous_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Computes the log-likelihood of a continuous Gaussian distribution.
    """

    centered_x = x - means
    inverse_std = torch.exp(-log_scales)
    normalized_x = centered_x * inverse_std
    log_probabilities = torch.distributions.Normal(
        torch.zeros_like(x), torch.ones_like(x)
    ).log_prob(normalized_x)

    return log_probabilities


def discrete_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Computes the log-likelihood of a Gaussian distribution, while discretizing to a given image
    """

    assert x.shape == means.shape == log_scales.shape

    centered_x = x - means
    inverse_std = torch.exp(-log_scales)

    plus_in = inverse_std * (centered_x + 1.0 / 255.0)
    cdf_plus = approximate_std_cdf(plus_in)

    min_in = inverse_std * (centered_x - 1.0 / 255.0)
    cdf_min = approximate_std_cdf(min_in)

    cdf_delta = cdf_plus - cdf_min

    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))

    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(
            x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))
        ),
    )

    assert log_probs.shape == x.shape

    return log_probs
