"""End-to-end "Give Me Some Credit" example for the cchvae package.

Fetches the GiveMeSomeCredit dataset from OpenML (ID 46468), trains a
random-forest classifier, and generates counterfactuals for a few applicants
predicted to default.

Run from the project root:

.. code-block:: bash

    uv run python examples/givme_example.py
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from cchvae import CCHVAE, Counterfactual

from givme_data import CONDITIONAL_COLUMNS, FREE_COLUMNS, load_give_me_credit

logging.basicConfig(level=logging.INFO, format="%(message)s")

FEATURE_TYPES = {
    "RevolvingUtilizationOfUnsecuredLines": "pos",
    "NumberOfTime30-59DaysPastDueNotWorse": "count",
    "DebtRatio": "pos",
    "MonthlyIncome": "pos",
    "NumberOfOpenCreditLinesAndLoans": "count",
    "NumberOfTimes90DaysLate": "count",
    "NumberRealEstateLoansOrLines": "count",
    "NumberOfTime60-89DaysPastDueNotWorse": "count",
    "age": "count",
    "NumberOfDependents": "count",
}

# Subsample: the full dataset has ~120k rows; 5k is plenty for a demo.
SUBSAMPLE = 5_000


def main() -> None:
    df, y = load_give_me_credit()

    rng = np.random.default_rng(619)
    if len(df) > SUBSAMPLE:
        idx = rng.choice(len(df), size=SUBSAMPLE, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
        y = y[idx]

    x_train, x_test, y_train, _ = train_test_split(df, y, test_size=0.2, random_state=619)
    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=619, n_jobs=-1)
    clf.fit(x_train, y_train)
    print(f"Train accuracy: {clf.score(x_train, y_train):.3f}")

    preds = clf.predict(x_test)
    denied = x_test[preds == 1].head(3).reset_index(drop=True)
    if denied.empty:
        print("No defaulters in test set \u2014 nothing to explain.")
        return

    explainer = CCHVAE(
        classifier=clf,
        background_data=x_train,
        target_class=0,
        immutable_features=CONDITIONAL_COLUMNS,
        feature_types=FEATURE_TYPES,
        latent_dim=2,
        intermediate_dim=5,
        categorical_latent_dim=3,
        epochs=20,
        batch_size=128,
        verbose=True,
    )

    result = explainer.generate_counterfactuals(
        denied,
        num_counterfactuals=3,
        search_samples=500,
        max_iterations=100,
        step_size=0.5,
    )

    explanations = [result] if isinstance(result, Counterfactual) else result
    for i, cf in enumerate(explanations):
        print(f"\n=== Instance {i} ({len(cf)} counterfactuals) ===")
        print(cf.highlighted_counterfactuals.to_string(index=False))


if __name__ == "__main__":
    main()
