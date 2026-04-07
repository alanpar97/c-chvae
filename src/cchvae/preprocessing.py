"""Schema inference, encoding, and normalization for heterogeneous tabular data.

The :class:`Preprocessor` turns a user-facing :class:`pandas.DataFrame` into the
flat ``float32`` matrix that the C-CHVAE network expects, and back again.

For each column we apply:

============ ============================================================
Type         Transform
============ ============================================================
``real``     Identity (raw float). Standardization stats stored.
``pos``      ``log(1+x)``. Stats computed on log-space.
``count``    ``log(1+x)``. Stats computed on log-space.
``cat``      One-hot encoding (``dim`` columns).
``ordinal``  Thermometer encoding (``dim`` columns).
============ ============================================================

The ``mean``/``var`` returned per column are *only* used inside the
likelihood (the affine reparameterization on top of the network output);
the network itself receives raw normalized values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype, is_string_dtype

from cchvae.errors import CCHVAEValueError
from cchvae.types import VALID_FEATURE_TYPES, FeatureType


@dataclass
class FeatureSpec:
    """Description of a single column in the schema.

    Attributes
    ----------
    name : str
        Original column name.
    type : FeatureType
        One of ``real``, ``pos``, ``count``, ``cat``, ``ordinal``.
    dim : int
        Encoded width: 1 for ``real/pos/count``, ``n_categories`` for
        ``cat``/``ordinal``.
    categories : list | None
        Sorted unique categories for ``cat``/``ordinal`` columns; ``None``
        otherwise.
    """

    name: str
    type: FeatureType
    dim: int
    categories: list | None = None


@dataclass
class FeatureSchema:
    """Ordered collection of :class:`FeatureSpec` describing a DataFrame.

    Attributes
    ----------
    specs : list[FeatureSpec]
        One spec per column, in the order the model sees them.
    immutable_columns : list[str]
        Names of columns that the counterfactual search may not change.
    """

    specs: list[FeatureSpec]
    immutable_columns: list[str] = field(default_factory=list)

    @property
    def column_names(self) -> list[str]:
        """Original (un-encoded) column names in schema order."""
        return [s.name for s in self.specs]

    @property
    def encoded_dim(self) -> int:
        """Total width of the encoded matrix."""
        return sum(s.dim for s in self.specs)

    @property
    def free_specs(self) -> list[FeatureSpec]:
        """Specs for mutable (non-immutable) columns."""
        return [s for s in self.specs if s.name not in self.immutable_columns]

    @property
    def conditional_specs(self) -> list[FeatureSpec]:
        """Specs for immutable (conditional) columns."""
        return [s for s in self.specs if s.name in self.immutable_columns]


class Preprocessor:
    """Fit-transform / inverse-transform helper for heterogeneous tabular data.

    Parameters
    ----------
    schema : FeatureSchema
        Schema describing every column to encode.

    Notes
    -----
    Normalization statistics for ``real``/``pos``/``count`` columns are
    computed during :meth:`fit` and exposed via
    :attr:`normalization_params` (one ``[mean, var]`` pair per *free*
    feature, in schema order). They feed the affine reparameterization
    inside the heterogeneous likelihoods.
    """

    def __init__(self, schema: FeatureSchema) -> None:
        self.schema = schema
        self._fitted = False
        self._free_norm: list[tuple[float, float]] = []
        self._cond_norm: list[tuple[float, float]] = []

    @property
    def normalization_params(self) -> list[tuple[float, float]]:
        """``[(mean, var), ...]`` for every *free* feature, in schema order."""
        self._check_fitted()
        return list(self._free_norm)

    @property
    def conditional_normalization_params(self) -> list[tuple[float, float]]:
        """``[(mean, var), ...]`` for every *conditional* feature."""
        self._check_fitted()
        return list(self._cond_norm)

    def fit(self, df: pd.DataFrame) -> Preprocessor:
        """Compute normalization statistics from ``df``."""
        self._validate_columns(df)

        self._free_norm = [self._fit_one(df, s) for s in self.schema.free_specs]
        self._cond_norm = [self._fit_one(df, s) for s in self.schema.conditional_specs]
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Encode ``df`` to ``(free_matrix, cond_matrix)`` float32 arrays.

        Returns
        -------
        free : np.ndarray
            Encoded mutable features, shape ``(n, free_encoded_dim)``.
        cond : np.ndarray
            Encoded immutable features, shape ``(n, cond_encoded_dim)``. May
            have width 0 if no immutable columns are declared.
        """
        self._check_fitted()
        self._validate_columns(df)

        free_blocks = [self._encode_one(df[s.name], s) for s in self.schema.free_specs]
        cond_blocks = [self._encode_one(df[s.name], s) for s in self.schema.conditional_specs]

        free = (
            np.concatenate(free_blocks, axis=1).astype(np.float32, copy=False)
            if free_blocks
            else np.empty((len(df), 0), dtype=np.float32)
        )
        cond = (
            np.concatenate(cond_blocks, axis=1).astype(np.float32, copy=False)
            if cond_blocks
            else np.empty((len(df), 0), dtype=np.float32)
        )
        return free, cond

    def inverse_transform_free(self, encoded: np.ndarray) -> pd.DataFrame:
        """Decode an encoded *free*-feature matrix back to a DataFrame."""
        self._check_fitted()
        return self._inverse_transform(encoded, self.schema.free_specs)

    def split_columns(self, encoded: np.ndarray, which: str = "free") -> list[np.ndarray]:
        """Slice an encoded matrix into per-feature blocks.

        Parameters
        ----------
        encoded : np.ndarray
            Encoded matrix to split.
        which : {"free", "conditional"}
            Which schema slice to use for the column widths.
        """
        if which == "free":
            specs = self.schema.free_specs
        elif which == "conditional":
            specs = self.schema.conditional_specs
        else:
            raise CCHVAEValueError(
                f"Unknown 'which' value: {which!r}.",
                param="which",
                hint="Pass 'free' or 'conditional'.",
                source="Preprocessor.split_columns",
            )

        blocks = []
        cursor = 0
        for s in specs:
            blocks.append(encoded[:, cursor : cursor + s.dim])
            cursor += s.dim
        return blocks

    def _fit_one(self, df: pd.DataFrame, spec: FeatureSpec) -> tuple[float, float]:
        """Compute (mean, var) for one column. Categorical columns return (0, 1)."""
        col = df[spec.name].to_numpy()
        if spec.type == "real":
            mean = float(np.mean(col))
            var = float(np.clip(np.var(col), 1e-6, np.inf))
        elif spec.type in ("pos", "count"):
            log_col = np.log(1.0 + col.astype(np.float64))
            mean = float(np.mean(log_col))
            var = float(np.clip(np.var(log_col), 1e-6, np.inf))
        else:
            mean, var = 0.0, 1.0
        return (mean, var)

    def _encode_one(self, series: pd.Series, spec: FeatureSpec) -> np.ndarray:
        """Encode one column to its dense float32 block."""
        n = len(series)
        if spec.type in ("real", "pos", "count"):
            return series.to_numpy(dtype=np.float32).reshape(n, 1)

        if spec.type == "cat":
            assert spec.categories is not None
            cat_to_idx = {c: i for i, c in enumerate(spec.categories)}
            indices = series.map(cat_to_idx)
            if indices.isna().any():
                missing = sorted(set(series[indices.isna()]))
                raise CCHVAEValueError(
                    f"Unknown category value(s) in column {spec.name!r}.",
                    config={"missing": missing, "known": spec.categories},
                    param=spec.name,
                    hint="Refit the preprocessor on data containing every category.",
                    source="Preprocessor._encode_one",
                )
            out = np.zeros((n, spec.dim), dtype=np.float32)
            out[np.arange(n), indices.to_numpy(dtype=np.int64)] = 1.0
            return out

        if spec.type == "ordinal":
            assert spec.categories is not None
            cat_to_idx = {c: i for i, c in enumerate(spec.categories)}
            indices = series.map(cat_to_idx)
            if indices.isna().any():
                missing = sorted(set(series[indices.isna()]))
                raise CCHVAEValueError(
                    f"Unknown ordinal value(s) in column {spec.name!r}.",
                    config={"missing": missing, "known": spec.categories},
                    param=spec.name,
                    hint="Refit the preprocessor on data containing every ordinal level.",
                    source="Preprocessor._encode_one",
                )
            idx = indices.to_numpy(dtype=np.int64)
            # Thermometer encoding: ones up to and including the level.
            arange = np.arange(spec.dim).reshape(1, -1)
            out = (arange <= idx.reshape(-1, 1)).astype(np.float32)
            return out

        raise CCHVAEValueError(  # pragma: no cover - guarded by schema validation
            f"Unknown feature type {spec.type!r}.",
            param="type",
            source="Preprocessor._encode_one",
        )

    def _inverse_transform(self, encoded: np.ndarray, specs: list[FeatureSpec]) -> pd.DataFrame:
        """Decode an encoded matrix into a DataFrame using ``specs``."""
        cursor = 0
        out: dict[str, np.ndarray] = {}
        for s in specs:
            block = encoded[:, cursor : cursor + s.dim]
            cursor += s.dim
            if s.type == "real":
                out[s.name] = block[:, 0]
            elif s.type in ("pos", "count"):
                # The decoder produces values already in the data domain
                # (likelihoods do the affine + exp inversion). We just clip
                # negatives that come from sampling noise on 'pos'.
                values = block[:, 0]
                if s.type == "count":
                    values = np.maximum(np.round(values), 0).astype(np.int64)
                else:
                    values = np.maximum(values, 0.0)
                out[s.name] = values
            elif s.type == "cat":
                assert s.categories is not None
                idx = np.argmax(block, axis=1)
                out[s.name] = np.array([s.categories[i] for i in idx])
            elif s.type == "ordinal":
                assert s.categories is not None
                # Thermometer: highest index where the value is "on".
                level = np.clip(block.sum(axis=1).round().astype(np.int64) - 1, 0, s.dim - 1)
                out[s.name] = np.array([s.categories[i] for i in level])
        return pd.DataFrame(out)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [s.name for s in self.schema.specs if s.name not in df.columns]
        if missing:
            raise CCHVAEValueError(
                f"DataFrame is missing required columns: {missing}",
                config={"expected": self.schema.column_names, "received": list(df.columns)},
                param="df",
                hint="Pass a DataFrame containing all columns the schema was built on.",
                source="Preprocessor._validate_columns",
            )

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise CCHVAEValueError(
                "Preprocessor has not been fitted yet.",
                source="Preprocessor",
                hint="Call .fit(df) before .transform(df).",
            )


def infer_schema(
    df: pd.DataFrame,
    feature_types: dict[str, FeatureType] | None = None,
    immutable_features: list[str] | None = None,
) -> FeatureSchema:
    """Build a :class:`FeatureSchema` from a DataFrame plus optional overrides.

    Inference rules
    ---------------
    - ``object`` / ``category`` / ``bool`` \u2192 ``cat``
    - integer dtype \u2192 ``count``
    - float dtype \u2192 ``real``
    - User-supplied ``feature_types`` always wins.
    """
    feature_types = feature_types or {}
    immutable_features = immutable_features or []

    for name, ftype in feature_types.items():
        if ftype not in VALID_FEATURE_TYPES:
            raise CCHVAEValueError(
                f"Unknown feature type {ftype!r} for column {name!r}.",
                config={"valid": list(VALID_FEATURE_TYPES)},
                param="feature_types",
                hint=f"Use one of {VALID_FEATURE_TYPES}.",
                source="infer_schema",
            )
    unknown_overrides = [c for c in feature_types if c not in df.columns]
    if unknown_overrides:
        raise CCHVAEValueError(
            f"feature_types keys not in DataFrame columns: {unknown_overrides}",
            config={"columns": list(df.columns)},
            param="feature_types",
            hint="Make sure every override key matches a column in the DataFrame.",
            source="infer_schema",
        )
    unknown_immutable = [c for c in immutable_features if c not in df.columns]
    if unknown_immutable:
        raise CCHVAEValueError(
            f"immutable_features not in DataFrame columns: {unknown_immutable}",
            config={"columns": list(df.columns)},
            param="immutable_features",
            hint="Each immutable feature name must match a DataFrame column.",
            source="infer_schema",
        )

    specs: list[FeatureSpec] = []
    for col in df.columns:
        if col in feature_types:
            ftype: FeatureType = feature_types[col]
        else:
            series = df[col]
            if (
                is_bool_dtype(series)
                or is_string_dtype(series)
                or series.dtype == object
                or str(series.dtype) == "category"
            ):
                ftype = "cat"
            elif is_integer_dtype(series):
                ftype = "count"
            elif is_float_dtype(series):
                ftype = "real"
            else:
                raise CCHVAEValueError(
                    f"Cannot infer feature type for column {col!r} (dtype={series.dtype}).",
                    param="feature_types",
                    hint=f"Specify it explicitly via feature_types={{'{col}': '<type>'}}.",
                    source="infer_schema",
                )

        if ftype in ("cat", "ordinal"):
            categories = sorted(df[col].dropna().unique().tolist())
            dim = len(categories)
            if dim < 2:
                raise CCHVAEValueError(
                    f"Categorical column {col!r} has fewer than 2 unique values.",
                    config={"n_unique": dim},
                    param=col,
                    hint="Drop the column or supply data with at least 2 categories.",
                    source="infer_schema",
                )
            specs.append(FeatureSpec(name=col, type=ftype, dim=dim, categories=categories))
        else:
            specs.append(FeatureSpec(name=col, type=ftype, dim=1))

    return FeatureSchema(specs=specs, immutable_columns=list(immutable_features))
