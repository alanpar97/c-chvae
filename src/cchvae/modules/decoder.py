"""Decoder for the C-CHVAE: ``z \u2192 y \u2192 \u03b8`` per heterogeneous feature.

The decoder is a deterministic function that maps a latent code ``z`` to
an intermediate representation ``y`` (one shared linear layer), splits
``y`` into per-feature blocks, and produces the natural parameters
``theta`` of each variable's likelihood via a small per-feature head.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from cchvae.preprocessing import FeatureSchema, FeatureSpec

WEIGHT_STD = 0.05


def _make_linear(in_features: int, out_features: int) -> nn.Linear:
    layer = nn.Linear(in_features, out_features)
    nn.init.normal_(layer.weight, std=WEIGHT_STD)
    nn.init.zeros_(layer.bias)
    return layer


class Decoder(nn.Module):
    """Heterogeneous decoder.

    Parameters
    ----------
    schema : FeatureSchema
        Feature schema describing free features.
    z_dim : int
        Latent dimensionality.
    y_dim : int
        Width of the per-feature ``y`` partition (the total ``y`` width is
        ``y_dim * len(free_specs)``).
    """

    def __init__(self, schema: FeatureSchema, z_dim: int, y_dim: int) -> None:
        super().__init__()
        self.schema = schema
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.free_specs = schema.free_specs
        self.y_total = y_dim * len(self.free_specs)

        # Shared z -> y layer.
        self.z_to_y = _make_linear(z_dim, self.y_total)

        # Per-feature theta heads. Each head is a small ModuleDict so we can
        # store the (potentially multiple) parameters that each likelihood
        # needs (e.g. real has mean+logvar; ordinal has partition+mean).
        self.theta_heads = nn.ModuleList()
        for spec in self.free_specs:
            self.theta_heads.append(self._make_head(spec, y_dim))

    @staticmethod
    def _make_head(spec: FeatureSpec, y_dim: int) -> nn.ModuleDict:
        head = nn.ModuleDict()
        if spec.type in ("real", "pos"):
            head["mean"] = _make_linear(y_dim, spec.dim)
            head["logvar"] = _make_linear(y_dim, spec.dim)
        elif spec.type == "count":
            head["lambda"] = _make_linear(y_dim, spec.dim)
        elif spec.type == "cat":
            # Constrain identifiability by zeroing the first logit.
            head["logits"] = _make_linear(y_dim, spec.dim - 1)
        elif spec.type == "ordinal":
            head["partition"] = _make_linear(y_dim, spec.dim - 1)
            head["mean"] = _make_linear(y_dim, 1)
        else:  # pragma: no cover - guarded by schema validation
            raise ValueError(f"Unknown feature type: {spec.type}")
        return head

    def forward(self, z: Tensor) -> tuple[list[Tensor | tuple[Tensor, ...]], Tensor]:
        """Map ``z`` to per-feature theta and the intermediate ``y``.

        Returns
        -------
        thetas : list
            One element per free feature; either a single Tensor (count, cat)
            or a tuple of Tensors (real/pos: mean+logvar; ordinal: partition+mean).
        y : Tensor
            The intermediate representation, useful for diagnostics.
        """
        batch_size = z.shape[0]
        y = self.z_to_y(z)
        y_blocks = y.view(batch_size, len(self.free_specs), self.y_dim)

        thetas: list[Tensor | tuple[Tensor, ...]] = []
        for i, spec in enumerate(self.free_specs):
            block = y_blocks[:, i, :]
            head: nn.ModuleDict = self.theta_heads[i]  # type: ignore[assignment]
            if spec.type in ("real", "pos"):
                thetas.append((head["mean"](block), head["logvar"](block)))
            elif spec.type == "count":
                thetas.append(head["lambda"](block))
            elif spec.type == "cat":
                logits_partial = head["logits"](block)
                zeros = torch.zeros((batch_size, 1), device=z.device, dtype=z.dtype)
                thetas.append(torch.cat([zeros, logits_partial], dim=1))
            elif spec.type == "ordinal":
                thetas.append((head["partition"](block), head["mean"](block)))

        return thetas, y
