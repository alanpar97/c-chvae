"""Heterogeneous log-likelihoods for the C-CHVAE decoder.

For each variable type the decoder produces raw parameters ``theta``;
the helpers below convert ``theta`` into the *natural* parameters of
the corresponding distribution (applying the affine reparameterization
that pulls in the data normalization stats), and provide a single
``log_prob`` and a ``sample`` operation per type.

This file is the PyTorch port of the original ``Loglik.py`` module. The
training-time and test-time variants from the original code are merged:
in eager mode there is no need for two implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical, Normal, Poisson

from cchvae.preprocessing import FeatureSpec

EPS = 1e-6


@dataclass
class LikelihoodOutput:
    """Result bundle returned by every likelihood call."""

    log_prob: Tensor  # shape (batch,)
    params: tuple[Tensor, ...]
    samples: Tensor  # shape (batch, dim)


class Likelihood(Protocol):
    """A callable taking ``(data, theta, norm)`` and returning :class:`LikelihoodOutput`."""

    def __call__(
        self,
        data: Tensor,
        theta: Tensor | tuple[Tensor, ...],
        normalization: tuple[float, float],
        spec: FeatureSpec,
    ) -> LikelihoodOutput: ...


def _as_tensor_pair(norm: tuple[float, float], device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    mean = torch.as_tensor(norm[0], device=device, dtype=dtype)
    var = torch.as_tensor(norm[1], device=device, dtype=dtype).clamp(min=EPS)
    return mean, var


def _safe_positive(t: Tensor, lo: float = EPS, hi: float = 1e6) -> Tensor:
    """Map an unbounded tensor to a strictly positive, finite range.

    Replaces NaN/Inf with finite values *before* clamping so that the
    resulting tensor is always inside ``[lo, hi]``.
    """
    return torch.nan_to_num(F.softplus(t), nan=lo, posinf=hi, neginf=lo).clamp(min=lo, max=hi)


def loglik_real(
    data: Tensor,
    theta: tuple[Tensor, Tensor],
    normalization: tuple[float, float],
    spec: FeatureSpec,
) -> LikelihoodOutput:
    """Gaussian likelihood with affine reparameterization."""
    est_mean, est_logvar = theta
    data_mean, data_var = _as_tensor_pair(normalization, data.device, data.dtype)

    est_var = _safe_positive(est_logvar, lo=EPS, hi=1.0)

    est_mean = torch.sqrt(data_var) * est_mean + data_mean
    est_var = data_var * est_var

    dist = Normal(est_mean, torch.sqrt(est_var))
    log_p_x = dist.log_prob(data).sum(dim=-1)
    return LikelihoodOutput(log_prob=log_p_x, params=(est_mean, est_var), samples=dist.rsample())


def loglik_pos(
    data: Tensor,
    theta: tuple[Tensor, Tensor],
    normalization: tuple[float, float],
    spec: FeatureSpec,
) -> LikelihoodOutput:
    """Log-normal likelihood: model log(1+x) as Gaussian."""
    est_mean, est_logvar = theta
    data_mean, data_var = _as_tensor_pair(normalization, data.device, data.dtype)

    data_log = torch.log1p(data.clamp(min=0.0))
    est_var = _safe_positive(est_logvar, lo=EPS, hi=1.0)
    est_mean = torch.sqrt(data_var) * est_mean + data_mean
    est_var = data_var * est_var

    dist = Normal(est_mean, torch.sqrt(est_var))
    log_p_x = dist.log_prob(data_log).sum(dim=-1) - data_log.sum(dim=-1)
    samples = torch.exp(dist.rsample()) - 1.0
    return LikelihoodOutput(log_prob=log_p_x, params=(est_mean, est_var), samples=samples)


def loglik_count(
    data: Tensor,
    theta: Tensor,
    normalization: tuple[float, float],
    spec: FeatureSpec,
) -> LikelihoodOutput:
    """Poisson likelihood. ``theta`` is a single rate logit."""
    est_lambda = _safe_positive(theta, lo=EPS, hi=1e6)
    dist = Poisson(est_lambda)
    log_p_x = dist.log_prob(data.clamp(min=0.0)).sum(dim=-1)
    samples = dist.sample()
    return LikelihoodOutput(log_prob=log_p_x, params=(est_lambda,), samples=samples)


def loglik_cat(
    data: Tensor,
    theta: Tensor,
    normalization: tuple[float, float],
    spec: FeatureSpec,
) -> LikelihoodOutput:
    """Categorical likelihood with softmax over ``dim`` classes."""
    log_pi = theta
    log_p_x = (data * F.log_softmax(log_pi, dim=-1)).sum(dim=-1)

    probs = F.softmax(log_pi, dim=-1)
    sampled_idx = Categorical(probs=probs).sample()
    samples = F.one_hot(sampled_idx, num_classes=spec.dim).to(log_pi.dtype)
    return LikelihoodOutput(log_prob=log_p_x, params=(log_pi,), samples=samples)


def loglik_ordinal(
    data: Tensor,
    theta: tuple[Tensor, Tensor],
    normalization: tuple[float, float],
    spec: FeatureSpec,
) -> LikelihoodOutput:
    """Cumulative-logit ordinal likelihood (thermometer-encoded data)."""
    partition_param, mean_param = theta
    batch_size = data.shape[0]

    mean_value = mean_param.view(-1, 1)
    theta_values = torch.cumsum(_safe_positive(partition_param, lo=EPS, hi=1e6), dim=1)
    sigmoid_est_mean = torch.sigmoid(theta_values - mean_value)

    ones = torch.ones((batch_size, 1), device=data.device, dtype=data.dtype)
    zeros = torch.zeros((batch_size, 1), device=data.device, dtype=data.dtype)
    mean_probs = torch.cat([sigmoid_est_mean, ones], dim=1) - torch.cat([zeros, sigmoid_est_mean], dim=1)
    mean_probs = mean_probs.clamp(min=EPS, max=1.0)

    # data is thermometer-encoded; convert to one-hot of the highest "on" level.
    true_level = data.sum(dim=1).long().clamp(min=1) - 1
    true_one_hot = F.one_hot(true_level, num_classes=spec.dim).to(data.dtype)

    log_p_x = torch.log((mean_probs * true_one_hot).sum(dim=-1).clamp(min=EPS))

    sampled_level = Categorical(probs=mean_probs).sample()
    sampled_idx = (sampled_level + 1).clamp(max=spec.dim)
    arange = torch.arange(spec.dim, device=data.device).view(1, -1)
    samples = (arange < sampled_idx.view(-1, 1)).to(data.dtype)

    return LikelihoodOutput(log_prob=log_p_x, params=(mean_probs,), samples=samples)


LIKELIHOODS: dict[str, Likelihood] = {  # type: ignore[dict-item]
    "real": loglik_real,
    "pos": loglik_pos,
    "count": loglik_count,
    "cat": loglik_cat,
    "ordinal": loglik_ordinal,
}
