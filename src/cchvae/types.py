"""Common type aliases used across the cchvae package."""

from __future__ import annotations

from numbers import Real
from typing import Literal

type FeatureType = Literal["real", "pos", "count", "cat", "ordinal"]
"""Heterogeneous feature types supported by the C-CHVAE likelihood model.

- ``"real"``: continuous Gaussian
- ``"pos"``: positive continuous (log-normal)
- ``"count"``: non-negative integer (Poisson)
- ``"cat"``: nominal categorical (softmax)
- ``"ordinal"``: ordered categorical (cumulative logit)
"""

VALID_FEATURE_TYPES: tuple[FeatureType, ...] = ("real", "pos", "count", "cat", "ordinal")

type Prediction = Real | str
type Predictions = Prediction | list[Real] | list[str]
