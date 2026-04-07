"""End-to-end C-CHVAE model: encoder + decoder + ELBO + training loop."""

from __future__ import annotations

import logging
import math
import time
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from cchvae.likelihoods import LIKELIHOODS, LikelihoodOutput
from cchvae.modules.decoder import Decoder
from cchvae.modules.encoder import Encoder
from cchvae.preprocessing import FeatureSchema

logger = logging.getLogger("cchvae")


class CCHVAEModel(nn.Module):
    """The complete conditional heterogeneous VAE.

    Parameters
    ----------
    schema : FeatureSchema
        Feature schema (free + conditional columns).
    normalization_params : list[tuple[float, float]]
        ``(mean, var)`` per *free* feature, in schema order. These are
        baked into the affine reparameterization in the likelihoods.
    z_dim : int
        Continuous latent dimensionality.
    y_dim : int
        Per-feature width of the intermediate representation.
    s_dim : int
        Number of categorical mixture components.
    reconstruction_weight : float, default 1.0
        Multiplier on the reconstruction term in the ELBO. The original
        TF code used a magic ``1.20`` here; we expose it as a parameter
        and default to the standard ELBO.
    """

    def __init__(
        self,
        schema: FeatureSchema,
        normalization_params: list[tuple[float, float]],
        z_dim: int = 2,
        y_dim: int = 5,
        s_dim: int = 3,
        reconstruction_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if len(normalization_params) != len(schema.free_specs):
            raise ValueError(
                f"normalization_params has length {len(normalization_params)} "
                f"but schema has {len(schema.free_specs)} free features."
            )
        self.schema = schema
        self.normalization_params = normalization_params
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.s_dim = s_dim
        self.reconstruction_weight = reconstruction_weight

        self.encoder = Encoder(schema=schema, z_dim=z_dim, s_dim=s_dim)
        self.decoder = Decoder(schema=schema, z_dim=z_dim, y_dim=y_dim)

        # Per-column standardization buffers used to normalize encoder inputs.
        # Categorical/ordinal columns (already in [0, 1]) are passed through
        # unchanged; numeric columns are log1p'd (for pos/count) and then
        # standardized using the stats computed by the Preprocessor.
        free_dim = sum(s.dim for s in schema.free_specs)
        norm_mean = torch.zeros(free_dim)
        norm_std = torch.ones(free_dim)
        is_log = torch.zeros(free_dim, dtype=torch.bool)
        cursor = 0
        for spec, (m, v) in zip(schema.free_specs, normalization_params, strict=True):
            if spec.type in ("real", "pos", "count"):
                norm_mean[cursor] = float(m)
                norm_std[cursor] = float(v) ** 0.5
                if spec.type in ("pos", "count"):
                    is_log[cursor] = True
            cursor += spec.dim
        self.register_buffer("_norm_mean", norm_mean)
        self.register_buffer("_norm_std", norm_std.clamp(min=1e-6))
        self.register_buffer("_norm_log", is_log)

    def _normalize_for_encoder(self, x_free: Tensor) -> Tensor:
        """Apply log1p (where applicable) + standardization to encoder inputs."""
        log_mask = cast("Tensor", self._norm_log)
        mean = cast("Tensor", self._norm_mean)
        std = cast("Tensor", self._norm_std)
        x = torch.where(log_mask, torch.log1p(x_free.clamp(min=0.0)), x_free)
        return (x - mean) / std

    def forward(self, x_free: Tensor, x_cond: Tensor, tau: float) -> dict:
        """Forward pass returning everything needed to compute the ELBO."""
        x_norm = self._normalize_for_encoder(x_free)
        enc = self.encoder(x_norm, x_cond, tau=tau)
        thetas, _ = self.decoder(enc["z"])
        likelihoods = self._evaluate_likelihoods(x_free, thetas)
        return {**enc, "thetas": thetas, "likelihoods": likelihoods}

    def _evaluate_likelihoods(self, x_free: Tensor, thetas: list) -> list[LikelihoodOutput]:
        """Run the per-feature likelihoods on the encoded data."""
        cursor = 0
        outputs: list[LikelihoodOutput] = []
        for i, spec in enumerate(self.schema.free_specs):
            block = x_free[:, cursor : cursor + spec.dim]
            cursor += spec.dim
            likelihood = LIKELIHOODS[spec.type]
            outputs.append(likelihood(block, thetas[i], self.normalization_params[i], spec))
        return outputs

    def elbo(self, fwd: dict) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute the ELBO terms.

        Returns
        -------
        elbo : Tensor
            Scalar evidence lower bound (mean over the batch).
        loss_reconstruction : Tensor
            Per-row reconstruction log-likelihood, shape ``(batch,)``.
        kl_z : Tensor
            Per-row KL on ``z``, shape ``(batch,)``.
        kl_s : Tensor
            Per-row KL on ``s``, shape ``(batch,)``.
        """
        log_p_x = torch.stack([lik.log_prob for lik in fwd["likelihoods"]], dim=0).sum(dim=0)

        log_pi = fwd["q_s"]
        pi = torch.softmax(log_pi, dim=-1)
        log_pi_norm = torch.log_softmax(log_pi, dim=-1)
        kl_s = (pi * log_pi_norm).sum(dim=-1) + math.log(self.s_dim)

        mean_p, logvar_p = fwd["p_z"]
        mean_q, logvar_q = fwd["q_z"]
        kl_z = (
            0.5
            * (
                torch.exp(logvar_q - logvar_p) + (mean_p - mean_q).pow(2) / torch.exp(logvar_p) - logvar_q + logvar_p
            ).sum(dim=-1)
            - 0.5 * self.z_dim
        )

        elbo = (self.reconstruction_weight * log_p_x - kl_z - kl_s).mean()
        return elbo, log_p_x, kl_z, kl_s

    def encode(self, x_free: Tensor, x_cond: Tensor) -> Tensor:
        """Return the deterministic posterior mean of ``z`` for inference."""
        with torch.no_grad():
            x_norm = self._normalize_for_encoder(x_free)
            enc = self.encoder(x_norm, x_cond, tau=1e-3)
            mean_q, _ = enc["q_z"]
            return mean_q

    def decode(self, z: Tensor) -> list[LikelihoodOutput]:
        """Decode ``z`` to per-feature likelihoods (no data dependency)."""
        thetas, _ = self.decoder(z)
        # Use a zero placeholder for data; we only need params/samples downstream.
        cursor = 0
        outputs = []
        for i, spec in enumerate(self.schema.free_specs):
            placeholder = torch.zeros((z.shape[0], spec.dim), device=z.device, dtype=z.dtype)
            cursor += spec.dim
            outputs.append(LIKELIHOODS[spec.type](placeholder, thetas[i], self.normalization_params[i], spec))
        return outputs

    def fit(
        self,
        x_free: np.ndarray,
        x_cond: np.ndarray,
        epochs: int = 80,
        batch_size: int = 100,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        """Train the VAE on encoded data.

        Parameters
        ----------
        x_free : np.ndarray
            Encoded free features, shape ``(n, free_encoded_dim)``.
        x_cond : np.ndarray
            Encoded conditional features, shape ``(n, cond_encoded_dim)``.
        epochs : int
            Number of training epochs.
        batch_size : int
            Mini-batch size.
        learning_rate : float
            Adam learning rate.
        device : str
            ``"cpu"`` or ``"cuda"``.
        verbose : bool
            Log per-epoch ELBO/KL summaries.

        Returns
        -------
        history : dict[str, list[float]]
            Per-epoch averages of ``elbo``, ``log_p_x``, ``kl_z``, ``kl_s``.
        """
        torch_device = torch.device(device)
        self.to(torch_device)
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        x_free_t = torch.as_tensor(x_free, dtype=torch.float32, device=torch_device)
        x_cond_t = torch.as_tensor(x_cond, dtype=torch.float32, device=torch_device)
        n_samples = x_free_t.shape[0]
        n_batches = max(n_samples // batch_size, 1)

        history: dict[str, list[float]] = {"elbo": [], "log_p_x": [], "kl_z": [], "kl_s": []}
        start = time.time()

        for epoch in range(epochs):
            tau = max(1.0 - 0.001 * epoch, 1e-3)
            perm = torch.randperm(n_samples, device=torch_device)
            x_free_shuf = x_free_t[perm]
            x_cond_shuf = x_cond_t[perm]

            avg_loss = 0.0
            avg_kl_z = 0.0
            avg_kl_s = 0.0
            avg_elbo = 0.0

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                xf = x_free_shuf[start_idx:end_idx]
                xc = x_cond_shuf[start_idx:end_idx]
                if xf.shape[0] == 0:
                    continue

                fwd = self(xf, xc, tau=tau)
                elbo, log_p_x, kl_z, kl_s = self.elbo(fwd)
                loss = -elbo

                if not torch.isfinite(loss):
                    # Skip the batch entirely if numerical issues produced NaN/Inf.
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                optimizer.step()

                avg_loss += float(log_p_x.mean().item())
                avg_kl_z += float(kl_z.mean().item())
                avg_kl_s += float(kl_s.mean().item())
                avg_elbo += float(elbo.item())

            avg_loss /= n_batches
            avg_kl_z /= n_batches
            avg_kl_s /= n_batches
            avg_elbo /= n_batches
            history["elbo"].append(avg_elbo)
            history["log_p_x"].append(avg_loss)
            history["kl_z"].append(avg_kl_z)
            history["kl_s"].append(avg_kl_s)

            if verbose:
                logger.info(
                    "Epoch %3d  time=%6.1fs  log_p_x=%.4f  KL_z=%.4f  KL_s=%.4f  ELBO=%.4f",
                    epoch,
                    time.time() - start,
                    avg_loss,
                    avg_kl_z,
                    avg_kl_s,
                    avg_elbo,
                )

        self.eval()
        return history
