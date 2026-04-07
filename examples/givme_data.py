"""Helper to fetch and preprocess the Give Me Some Credit dataset from OpenML."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

OPENML_DATA_ID = 46468

FREE_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
]
CONDITIONAL_COLUMNS = ["age", "NumberOfDependents"]

_COUNT_FEATURES = {
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "age",
    "NumberOfDependents",
}


def load_give_me_credit() -> tuple[pd.DataFrame, np.ndarray]:
    """Fetch the Give Me Some Credit dataset from OpenML, clean it, and return (X, y).

    Downloads on first call; subsequent calls use sklearn's local cache.

    Returns
    -------
    df : pd.DataFrame
        Features with conditional columns first (age, NumberOfDependents),
        then free columns. ~90k rows after dropping rows with missing values.
    y : np.ndarray of int
        Binary target: 1 = serious delinquency (default), 0 = no default.
    """
    bunch = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True, parser="auto")
    df = bunch.frame.copy()

    target_col = bunch.target.name if hasattr(bunch.target, "name") else "SeriousDlqin2yrs"
    y_raw = df.pop(target_col)

    # Drop rows with any NaN (MonthlyIncome and NumberOfDependents have missings)
    mask = df[FREE_COLUMNS + CONDITIONAL_COLUMNS].notna().all(axis=1)
    df = df.loc[mask].reset_index(drop=True)
    y = y_raw.loc[mask].astype(int).to_numpy()

    # Order columns: conditional first, then free
    df = df[CONDITIONAL_COLUMNS + FREE_COLUMNS]

    for col in _COUNT_FEATURES:
        df[col] = df[col].astype(int)

    return df, y
