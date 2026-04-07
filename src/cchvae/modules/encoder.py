"""Encoder for the C-CHVAE: ``x \u2192 (s, z)``.

The encoder defines the variational posterior

.. math::

    q(s, z \\mid x, x_c) = q(s \\mid x, x_c)\\, q(z \\mid s, x, x_c)

where ``s`` is a Gumbel-softmax categorical and ``z`` is a Gaussian whose
parameters are obtained as a *factorized product of Gaussians*: one
Gaussian per free feature plus a unit-variance prior, all combined via
the standard product-of-experts formula. ``x_c`` denotes the optional
conditioning (immutable) features.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cchvae.preprocessing import FeatureSchema

WEIGHT_STD = 0.05


def _make_linear(in_features: int, out_features: int) -> nn.Linear:
    """Build a Linear layer matching the original ``stddev=0.05`` initializer."""
    layer = nn.Linear(in_features, out_features)
    nn.init.normal_(layer.weight, std=WEIGHT_STD)
    nn.init.zeros_(layer.bias)
    return layer


class Encoder(nn.Module):
    """Conditional categorical-Gaussian encoder.

    Parameters
    ----------
    schema : FeatureSchema
        Feature schema describing free and conditional columns.
    z_dim : int
        Continuous latent dimensionality.
    s_dim : int
        Number of categorical mixture components.
    """

    def __init__(self, schema: FeatureSchema, z_dim: int, s_dim: int) -> None:
        super().__init__()
        self.schema = schema
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.free_widths: list[int] = [s.dim for s in schema.free_specs]
        self.cond_widths: list[int] = [s.dim for s in schema.conditional_specs]
        self.x_dim = sum(self.free_widths)
        self.x_cond_dim = sum(self.cond_widths)

        # q(s | x, x_c)
        self.s_logits = _make_linear(self.x_dim + self.x_cond_dim, s_dim)

        # q(z | s, x_d, x_c) -- one Gaussian per free feature
        self.z_means = nn.ModuleList()
        self.z_logvars = nn.ModuleList()
        for width in self.free_widths:
            in_dim = width + s_dim + self.x_cond_dim
            self.z_means.append(_make_linear(in_dim, z_dim))
            self.z_logvars.append(_make_linear(in_dim, z_dim))

        # p(z | s)
        self.prior_z_mean = _make_linear(s_dim, z_dim)

    def forward(
        self,
        x_free: Tensor,
        x_cond: Tensor,
        tau: float,
    ) -> dict[str, Tensor | tuple[Tensor, Tensor]]:
        """Encode a batch.

        Parameters
        ----------
        x_free : Tensor
            Encoded free features, shape ``(batch, x_dim)``.
        x_cond : Tensor
            Encoded conditional features, shape ``(batch, x_cond_dim)``.
            Pass an empty tensor of width 0 if there are no conditional
            features.
        tau : float
            Gumbel-softmax temperature.

        Returns
        -------
        dict
            ``{"s": samples_s, "z": samples_z, "q_s": logits, "q_z": (mean, logvar), "p_z": (mean, logvar)}``.
        """
        batch_size = x_free.shape[0]
        s_input = torch.cat([x_free, x_cond], dim=1) if self.x_cond_dim else x_free
        s_logits = self.s_logits(s_input)
        samples_s = F.gumbel_softmax(s_logits, tau=tau, hard=False)

        # Per-feature factorized Gaussians + unit prior at the end.
        means: list[Tensor] = []
        logvars: list[Tensor] = []
        cursor = 0
        for i, width in enumerate(self.free_widths):
            block = x_free[:, cursor : cursor + width]
            cursor += width
            inputs = [block, samples_s]
            if self.x_cond_dim:
                inputs.append(x_cond)
            joined = torch.cat(inputs, dim=1)
            means.append(self.z_means[i](joined))
            logvars.append(self.z_logvars[i](joined))

        zero = torch.zeros((batch_size, self.z_dim), device=x_free.device, dtype=x_free.dtype)
        means.append(zero)
        logvars.append(zero)

        means_stack = torch.stack(means, dim=0)  # (n_experts, batch, z_dim)
        logvars_stack = torch.stack(logvars, dim=0)

        # Product of Gaussians: precisions add.
        log_var_joint = -torch.logsumexp(-logvars_stack, dim=0)
        mean_joint = torch.exp(log_var_joint) * (means_stack * torch.exp(-logvars_stack)).sum(dim=0)

        eps = torch.randn_like(mean_joint)
        samples_z = mean_joint + torch.exp(0.5 * log_var_joint) * eps

        # Prior p(z | s): mean depends on s, log variance fixed at 0.
        prior_mean = self.prior_z_mean(samples_s)
        prior_logvar = torch.zeros_like(prior_mean)

        return {
            "s": samples_s,
            "z": samples_z,
            "q_s": s_logits,
            "q_z": (mean_joint, log_var_joint),
            "p_z": (prior_mean, prior_logvar),
        }
