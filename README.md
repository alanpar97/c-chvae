# C-CHVAE

A clean PyTorch implementation of **C-CHVAE** (Pawelczyk et al., WWW 2020):
*Learning Model-Agnostic Counterfactual Explanations for Tabular Data*.

## Setup

Counterfactual explanations identify the smallest change to an input that flips a
classifier's prediction in a desired direction — e.g. turning "loan rejected" into
"loan awarded", or "high cardiovascular risk" into "low risk". C-CHVAE produces
counterfactuals that are **proximate** (not local outliers) and **connected** to
regions of substantial data density (close to correctly classified observations).
Together, these two requirements are known as **counterfactual faithfulness**.

## Intuition

C-CHVAE embeds counterfactual search into a data-density approximator — a
variational autoencoder. The encoder maps the original tabular data into a
lower-dimensional, real-valued, dense representation `z`, which defines the
neighbourhood to search. C-CHVAE then perturbs `z → z + δ` and decodes the
perturbed latent back into feature space. For small perturbations, the decoder
produces a plausible counterfactual that is passed to the pretrained classifier
to check whether the prediction has flipped.

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/). Clone the
repository and install the package into a local virtual environment:

```bash
git clone https://github.com/alanpar97/c-chvae
cd c-chvae
uv sync
```

This creates a `.venv/` in the project root with `cchvae` and all its
dependencies installed. Activate it with `source .venv/bin/activate`, or run
commands with `uv run python ...`.

## Quick start

`CCHVAE` takes any sklearn-style classifier together with a background
`pandas.DataFrame` and returns `Counterfactual` objects for new instances.

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from cchvae import CCHVAE

# 1. A background dataset and a classifier you want to explain.
X_train: pd.DataFrame = ...  # your training features
y_train = ...                # your training labels

clf = RandomForestClassifier().fit(X_train, y_train)

# 2. Build the explainer. If no pretrained VAE is passed, one is trained here.
explainer = CCHVAE(
    classifier=clf,
    background_data=X_train,
    immutable_features=["age", "sex"],          # columns that must not change
    feature_types={"income": "pos"},            # optional overrides; rest is inferred
    z_dim=2,
    epochs=50,
)

# 3. Explain a single instance (or a whole DataFrame of instances).
instance = X_train.iloc[[0]]
result = explainer.generate_counterfactuals(instance, n_counterfactuals=3)

print(result.original_prediction, "→", result.counterfactual_prediction)
print(result.highlighted_counterfactuals)   # only the columns that changed
```

For a full end-to-end example on the *Give Me Some Credit* dataset, see
[`examples/givme_example.py`](examples/givme_example.py).

## Bibtex

```bibtex
@inproceedings{pawelczyk_learning2019,
  author    = {Pawelczyk, Martin and Broelemann, Klaus and Kasneci, Gjergji},
  title     = {Learning Model-Agnostic Counterfactual Explanations for Tabular Data},
  year      = {2020},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  booktitle = {Proceedings of The Web Conference 2020},
  pages     = {3126--3132},
  numpages  = {7},
  keywords  = {Transparency, Counterfactual explanations, Interpretability},
  location  = {Taipei, Taiwan},
  series    = {WWW '20}
}
```
