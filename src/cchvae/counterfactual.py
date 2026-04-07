"""The :class:`Counterfactual` value object.

A :class:`Counterfactual` instance bundles a single original instance with the
counterfactual(s) that flip its prediction, plus a "highlighted" view that
shows only the changed feature values.
"""

from __future__ import annotations

from numbers import Real

import pandas as pd

from cchvae.errors import CCHVAEError, CCHVAEValueError
from cchvae.types import Prediction, Predictions


class Counterfactual:
    """A counterfactual explanation for a single original instance.

    A single :class:`Counterfactual` corresponds to one original instance and
    one *or more* counterfactual rows generated for it.

    Parameters
    ----------
    original_instance : pd.Series | pd.DataFrame
        The original instance for which counterfactuals were generated.
    counterfactual_instance : pd.Series | pd.DataFrame
        The generated counterfactual instance(s). Must have the same columns
        as ``original_instance``.
    original_prediction : Prediction
        The model's prediction for ``original_instance``.
    counterfactual_prediction : Predictions
        The model's prediction(s) for the counterfactual rows. Either a
        single scalar (broadcast to every row) or a list with one entry per
        counterfactual row.
    """

    def __init__(
        self,
        original_instance: pd.Series | pd.DataFrame,
        counterfactual_instance: pd.Series | pd.DataFrame,
        original_prediction: Prediction,
        counterfactual_prediction: Predictions,
    ) -> None:
        (
            self._original_instance,
            self._counterfactuals,
            self._original_prediction,
            self._counterfactual_prediction,
        ) = self._validate_init(
            original_instance,
            counterfactual_instance,
            original_prediction,
            counterfactual_prediction,
        )

        self._highlighted_counterfactuals = self._create_highlighted_counterfactuals()

    @property
    def original_instance(self) -> pd.DataFrame:
        """The original instance as a one-row DataFrame."""
        return self._original_instance

    @property
    def counterfactuals(self) -> pd.DataFrame:
        """The generated counterfactual instance(s) as a DataFrame."""
        return self._counterfactuals

    @property
    def highlighted_counterfactuals(self) -> pd.DataFrame:
        """A DataFrame where unchanged columns are replaced with ``"-"``.

        Also appends a ``"Prediction"`` column showing the
        ``original \u2192 counterfactual`` flip per row.
        """
        return self._highlighted_counterfactuals

    @property
    def original_prediction(self) -> Prediction:
        """The model's prediction for the original instance."""
        return self._original_prediction

    @property
    def counterfactual_prediction(self) -> Predictions:
        """The model's prediction(s) for the counterfactual rows."""
        return self._counterfactual_prediction

    def __len__(self) -> int:
        return len(self._counterfactuals)

    def __repr__(self) -> str:
        return (
            f"Counterfactual(n={len(self)}, "
            f"original_prediction={self._original_prediction!r}, "
            f"columns={list(self._counterfactuals.columns)})"
        )

    def _create_highlighted_counterfactuals(self) -> pd.DataFrame:
        """Build the "what changed" view of the counterfactual rows."""
        highlighted = self._counterfactuals.copy()

        for col in highlighted.columns:
            original_value = self._original_instance.iloc[0][col]
            highlighted[col] = highlighted[col].where(highlighted[col] != original_value, "-")

        prediction_cells: list[str]
        if isinstance(self._counterfactual_prediction, list):
            prediction_cells = [
                f"{self._original_prediction} \u2192 {pred}" for pred in self._counterfactual_prediction
            ]
        else:
            prediction_cells = [f"{self._original_prediction} \u2192 {self._counterfactual_prediction}"] * len(
                highlighted
            )

        highlighted["Prediction"] = prediction_cells
        return highlighted

    def _validate_init(
        self,
        original_instance: pd.Series | pd.DataFrame,
        counterfactual_instance: pd.Series | pd.DataFrame,
        original_prediction: Prediction,
        counterfactual_prediction: Predictions,
    ) -> tuple[pd.DataFrame, pd.DataFrame, Prediction, Predictions]:
        """Validate inputs and normalize Series \u2192 DataFrame."""
        if not isinstance(original_instance, pd.Series | pd.DataFrame):
            raise CCHVAEValueError(
                "Original Instance must be a pandas Series or DataFrame.",
                config={"type": type(original_instance).__name__},
                param="original_instance",
                hint="Ensure the original instance is a pandas Series or DataFrame.",
                source="Counterfactual._validate_init",
            )

        if not isinstance(counterfactual_instance, pd.Series | pd.DataFrame):
            raise CCHVAEValueError(
                "Counterfactual Instance must be a pandas Series or DataFrame.",
                config={"type": type(counterfactual_instance).__name__},
                param="counterfactual_instance",
                hint="Ensure the counterfactual instance is a pandas Series or DataFrame.",
                source="Counterfactual._validate_init",
            )

        if counterfactual_instance.empty:
            raise CCHVAEValueError(
                "Counterfactual Instance is empty.",
                param="counterfactual_instance",
                hint="Drop unsuccessful instances before creating a Counterfactual object.",
                source="Counterfactual._validate_init",
            )

        if not _is_prediction(original_prediction):
            raise CCHVAEValueError(
                "Original Prediction must be an int, float, or str.",
                config={"type": type(original_prediction).__name__},
                param="original_prediction",
                hint="Ensure the original prediction is a valid type (int, float, or str).",
                source="Counterfactual._validate_init",
            )

        if not _is_predictions(counterfactual_prediction):
            raise CCHVAEValueError(
                "Counterfactual Prediction must be an int, float, str, or a list of those types.",
                config={"type": type(counterfactual_prediction).__name__},
                param="counterfactual_prediction",
                hint="Ensure the counterfactual prediction is a valid type (int, float, str, list).",
                source="Counterfactual._validate_init",
            )

        if isinstance(counterfactual_prediction, list):
            if not counterfactual_prediction:
                raise CCHVAEValueError(
                    "Counterfactual Prediction list cannot be empty.",
                    param="counterfactual_prediction",
                    hint="Ensure the counterfactual prediction list contains at least one element.",
                    source="Counterfactual._validate_init",
                )

            n_counterfactuals = len(counterfactual_instance)
            if len(counterfactual_prediction) != n_counterfactuals:
                raise CCHVAEValueError(
                    "Counterfactual Prediction list length must match number of counterfactual rows.",
                    config={
                        "n_rows": n_counterfactuals,
                        "len_predictions": len(counterfactual_prediction),
                    },
                    param="counterfactual_prediction",
                    hint="Provide one prediction per counterfactual row.",
                    source="Counterfactual._validate_init",
                )

            first_type = type(counterfactual_prediction[0])
            if not all(isinstance(pred, first_type) for pred in counterfactual_prediction):
                raise CCHVAEValueError(
                    "All elements in Counterfactual Prediction list must be of the same type.",
                    config={"types": [type(pred).__name__ for pred in counterfactual_prediction]},
                    param="counterfactual_prediction",
                    hint="Ensure all predictions in the list are of the same type (int, float, or str).",
                    source="Counterfactual._validate_init",
                )

        self._validate_counterfactuals(original_instance, counterfactual_instance)

        if isinstance(original_instance, pd.Series):
            original_instance = original_instance.to_frame().T

        if isinstance(counterfactual_instance, pd.Series):
            counterfactual_instance = counterfactual_instance.to_frame().T

        return original_instance, counterfactual_instance, original_prediction, counterfactual_prediction

    @staticmethod
    def _validate_counterfactuals(
        original_instance: pd.Series | pd.DataFrame,
        counterfactual_instance: pd.Series | pd.DataFrame,
    ) -> None:
        """Ensure original and counterfactual share the same column names."""
        if isinstance(original_instance, pd.Series):
            original_columns = list(original_instance.index)
        else:
            original_columns = list(original_instance.columns)

        if isinstance(counterfactual_instance, pd.Series):
            counter_columns = list(counterfactual_instance.index)
        else:
            counter_columns = list(counterfactual_instance.columns)

        if set(counter_columns) != set(original_columns):
            raise CCHVAEError(
                "Counterfactual instance must have the same columns as the original instance.",
                config={"expected": original_columns, "received": counter_columns},
                param="columns",
                hint="Ensure the counterfactual generator preserves feature names.",
                source="Counterfactual._validate_counterfactuals",
            )

        if len(counter_columns) != len(original_columns):
            raise CCHVAEError(
                "Counterfactual Instance must have the same number of columns as the original instance.",
                config={"expected": len(original_columns), "received": len(counter_columns)},
                param="columns",
                hint="Ensure the counterfactual generator preserves the number of features.",
                source="Counterfactual._validate_counterfactuals",
            )


def _is_prediction(value: object) -> bool:
    """Check if *value* is a valid scalar prediction (Real or str, but not bool)."""
    if isinstance(value, bool):
        return False
    return isinstance(value, Real | str)


def _is_predictions(value: object) -> bool:
    """Check if *value* is a valid Predictions (scalar or list of homogeneous predictions)."""
    if _is_prediction(value):
        return True
    if isinstance(value, list):
        return all(_is_prediction(item) for item in value)
    return False
