"""Unit tests for the Counterfactual value object."""

from __future__ import annotations

import pandas as pd
import pytest

from cchvae import CCHVAEError, CCHVAEValueError, Counterfactual


def test_counterfactual_with_series_input() -> None:
    original = pd.Series({"age": 30, "income": 50000.0})
    cf = pd.Series({"age": 35, "income": 50000.0})

    result = Counterfactual(
        original_instance=original,
        counterfactual_instance=cf,
        original_prediction=0,
        counterfactual_prediction=1,
    )

    assert isinstance(result.original_instance, pd.DataFrame)
    assert isinstance(result.counterfactuals, pd.DataFrame)
    assert len(result) == 1
    assert result.highlighted_counterfactuals.loc[0, "income"] == "-"
    assert result.highlighted_counterfactuals.loc[0, "age"] == 35
    assert "0 \u2192 1" in result.highlighted_counterfactuals.loc[0, "Prediction"]


def test_counterfactual_with_multiple_rows() -> None:
    original = pd.Series({"a": 1.0, "b": 2.0})
    cfs = pd.DataFrame({"a": [3.0, 4.0], "b": [2.0, 2.0]})

    result = Counterfactual(
        original_instance=original,
        counterfactual_instance=cfs,
        original_prediction=0,
        counterfactual_prediction=[1, 1],
    )
    assert len(result) == 2
    assert (result.highlighted_counterfactuals["b"] == "-").all()


def test_counterfactual_rejects_mismatched_columns() -> None:
    original = pd.Series({"a": 1.0, "b": 2.0})
    cf = pd.Series({"a": 1.0, "c": 2.0})
    with pytest.raises(CCHVAEError):
        Counterfactual(original, cf, 0, 1)


def test_counterfactual_rejects_empty() -> None:
    original = pd.Series({"a": 1.0})
    cf = pd.DataFrame({"a": []})
    with pytest.raises(CCHVAEValueError):
        Counterfactual(original, cf, 0, 1)


def test_counterfactual_rejects_mismatched_prediction_list() -> None:
    original = pd.Series({"a": 1.0})
    cf = pd.DataFrame({"a": [2.0, 3.0]})
    with pytest.raises(CCHVAEValueError):
        Counterfactual(original, cf, 0, [1])
