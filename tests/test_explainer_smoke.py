"""End-to-end smoke test for the CCHVAE explainer on a synthetic dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cchvae import CCHVAE, Counterfactual


def _make_dataset(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 70, size=n)
    income = rng.uniform(20_000, 150_000, size=n)
    score = rng.normal(0, 1, size=n)
    region = rng.choice(["N", "S", "E", "W"], size=n)

    # Positive class is more likely for high income + high score
    logit = 0.00003 * (income - 80_000) + 1.2 * score
    prob = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(size=n) < prob).astype(int)

    df = pd.DataFrame({"age": age, "income": income, "score": score, "region": region})
    return df, y


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cchvae_generates_counterfactuals() -> None:
    df, y = _make_dataset()
    clf = LogisticRegression(max_iter=500).fit(pd.get_dummies(df, columns=["region"]), y)

    # Wrap the classifier so .predict expects a DataFrame with the original columns.
    class Wrapped:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return clf.predict(
                pd.get_dummies(X, columns=["region"]).reindex(
                    columns=pd.get_dummies(df, columns=["region"]).columns, fill_value=0
                )
            )

    wrapped = Wrapped()

    explainer = CCHVAE(
        classifier=wrapped,
        background_data=df,
        target_class=1,
        immutable_features=["region"],
        feature_types={"income": "pos"},
        latent_dim=2,
        intermediate_dim=4,
        categorical_latent_dim=2,
        epochs=10,
        batch_size=64,
        verbose=False,
    )

    # Pick a row the classifier predicts as class 0 (so we try to flip to 1).
    preds = wrapped.predict(df)
    negatives = df[preds == 0].head(3).reset_index(drop=True)
    assert len(negatives) >= 1

    result = explainer.generate_counterfactuals(
        negatives,
        num_counterfactuals=2,
        search_samples=200,
        max_iterations=40,
        step_size=0.5,
    )

    if isinstance(result, Counterfactual):
        result = [result]
    assert all(isinstance(r, Counterfactual) for r in result)
    # At least one instance has counterfactuals.
    assert any(len(r) > 0 for r in result)
    # Immutable feature was preserved.
    for r in result:
        for _, row in r.counterfactuals.iterrows():
            assert row["region"] == r.original_instance.iloc[0]["region"]
