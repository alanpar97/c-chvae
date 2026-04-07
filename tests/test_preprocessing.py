"""Tests for schema inference and the Preprocessor round-trip."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cchvae import CCHVAEValueError, Preprocessor, infer_schema


def _make_df(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, size=50),
            "income": rng.uniform(10_000, 200_000, size=50),
            "score": rng.normal(0.0, 1.0, size=50),
            "region": rng.choice(["N", "S", "E", "W"], size=50),
        }
    )


def test_infer_schema_defaults() -> None:
    df = _make_df()
    schema = infer_schema(df)
    by_name = {s.name: s for s in schema.specs}
    assert by_name["age"].type == "count"
    assert by_name["income"].type == "real"
    assert by_name["score"].type == "real"
    assert by_name["region"].type == "cat"
    assert by_name["region"].dim == 4


def test_infer_schema_override_types_and_immutable() -> None:
    df = _make_df()
    schema = infer_schema(
        df,
        feature_types={"income": "pos"},
        immutable_features=["region"],
    )
    assert next(s for s in schema.specs if s.name == "income").type == "pos"
    assert schema.immutable_columns == ["region"]
    assert {s.name for s in schema.free_specs} == {"age", "income", "score"}
    assert [s.name for s in schema.conditional_specs] == ["region"]


def test_preprocessor_roundtrip_cat_and_numeric() -> None:
    df = _make_df()
    schema = infer_schema(df, feature_types={"income": "pos"}, immutable_features=["region"])
    pre = Preprocessor(schema).fit(df)
    free, cond = pre.transform(df)

    assert free.shape[0] == len(df)
    assert cond.shape[0] == len(df)
    # free: age(1) + income(1) + score(1) = 3
    assert free.shape[1] == 3
    # cond: region(4)
    assert cond.shape[1] == 4

    # Categorical block is valid one-hot
    assert np.allclose(cond.sum(axis=1), 1.0)


def test_preprocessor_rejects_unknown_override() -> None:
    df = _make_df()
    with pytest.raises(CCHVAEValueError):
        infer_schema(df, feature_types={"missing_col": "real"})


def test_preprocessor_rejects_bad_feature_type() -> None:
    df = _make_df()
    with pytest.raises(CCHVAEValueError):
        infer_schema(df, feature_types={"age": "not_a_type"})  # type: ignore[dict-item]


def test_inverse_transform_free_preserves_categories() -> None:
    df = _make_df()
    schema = infer_schema(df, immutable_features=["region"])
    pre = Preprocessor(schema).fit(df)
    free, _ = pre.transform(df)
    decoded = pre.inverse_transform_free(free)
    assert set(decoded.columns) == {"age", "income", "score"}
    # Integer "age" column survives round-trip as integer
    assert (decoded["age"] == df["age"]).all()
