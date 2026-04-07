"""Latent-space hypersphere shell search for counterfactual generation.

For a single original instance encoded as ``z_0``, we sample
``search_samples`` random perturbations ``\u0394z`` uniformly distributed on
the L^p hypersphere shell ``[l, h)``, decode them to data space, run them
through the user's classifier, and keep those that hit the target class.
If too few candidates are collected, we double the shell ``[l, h) \u2192 [h, h+step)``
and try again, up to ``max_iterations`` shells.

This is a faithful port of the search loop in the original
``Sampling.py`` (lines 201\u2013325), restructured as a pure function.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from cchvae.modules.vae import CCHVAEModel
from cchvae.preprocessing import Preprocessor

logger = logging.getLogger("cchvae")


@dataclass
class SearchResult:
    """Result of a single instance's counterfactual search.

    Attributes
    ----------
    counterfactuals_encoded : np.ndarray
        Encoded counterfactual rows in *free* feature space, shape
        ``(k_found, free_encoded_dim)``. ``k_found`` may be less than
        ``num_counterfactuals`` if the search exhausted its budget.
    z_counterfactuals : np.ndarray
        Latent codes of the kept counterfactuals.
    z_original : np.ndarray
        Latent code of the original instance.
    """

    counterfactuals_encoded: np.ndarray
    z_counterfactuals: np.ndarray
    z_original: np.ndarray


def latent_shell_search(
    model: CCHVAEModel,
    preprocessor: Preprocessor,
    classifier,
    target_class,
    x_free_row: np.ndarray,
    x_cond_row: np.ndarray,
    background_free: np.ndarray,
    num_counterfactuals: int = 5,
    search_samples: int = 1000,
    max_iterations: int = 500,
    step_size: float = 0.5,
    norm: int = 2,
    device: str = "cpu",
) -> SearchResult:
    """Search for counterfactuals for a single instance.

    Parameters
    ----------
    model : CCHVAEModel
        Trained C-CHVAE model.
    preprocessor : Preprocessor
        Fitted preprocessor (used for inverse-transform).
    classifier : object
        Object exposing ``.predict(np.ndarray)``. The classifier is
        called on the *original feature-space* DataFrame columns
        (``cond ++ free``).
    target_class : int | str
        Desired prediction value.
    x_free_row : np.ndarray
        Encoded free features for the single instance, shape ``(1, x_dim)``.
    x_cond_row : np.ndarray
        Encoded conditional features, shape ``(1, x_cond_dim)``.
    background_free : np.ndarray
        Encoded free features of the background dataset; used only to fit
        a scaler for the L1 distance ranking inside the shell.
    num_counterfactuals : int
        Number of counterfactuals to return per instance (``k``).
    search_samples : int
        Perturbations sampled per shell expansion.
    max_iterations : int
        Maximum number of shell expansions before giving up.
    step_size : float
        Width of each shell expansion in latent space.
    norm : int
        L^p norm used to measure latent perturbation magnitude.
    device : str
        Torch device.

    Returns
    -------
    SearchResult
    """
    torch_device = torch.device(device)
    model.eval()

    free_dim = x_free_row.shape[1]
    z_dim = model.z_dim

    scaler = StandardScaler().fit(background_free)
    scaled_original = scaler.transform(x_free_row)

    x_free_t = torch.as_tensor(x_free_row, dtype=torch.float32, device=torch_device)
    x_cond_t = torch.as_tensor(x_cond_row, dtype=torch.float32, device=torch_device)

    z_original = model.encode(x_free_t, x_cond_t).detach().cpu().numpy()

    z_replicated = np.repeat(z_original, search_samples, axis=0)
    cond_replicated = np.repeat(x_cond_row, search_samples, axis=0)
    scaled_original_replicated = np.repeat(scaled_original, search_samples, axis=0)

    accepted_x = np.zeros((num_counterfactuals, free_dim), dtype=np.float32)
    accepted_z = np.zeros((num_counterfactuals, z_dim), dtype=np.float32)
    n_accepted = 0

    low = 0.0
    high = low + step_size

    for _ in range(max_iterations):
        delta_z = np.random.randn(search_samples, z_dim)
        radii = np.random.rand(search_samples) * (high - low) + low
        norms = np.linalg.norm(delta_z, ord=norm, axis=1)
        norms = np.where(norms == 0, 1.0, norms)
        delta_z = delta_z * (radii / norms).reshape(-1, 1)

        z_tilde = z_replicated + delta_z

        with torch.no_grad():
            z_tilde_t = torch.as_tensor(z_tilde, dtype=torch.float32, device=torch_device)
            decoded = model.decode(z_tilde_t)

        # Build the candidate matrix in *encoded* free space using each
        # likelihood's "expected value" (mean) for downstream prediction.
        cursor = 0
        candidate_blocks: list[np.ndarray] = []
        for spec, lik in zip(model.schema.free_specs, decoded, strict=True):
            params = lik.params
            if spec.type in ("real", "pos"):
                # mean is the first param; for 'pos' the affine reparam already
                # mapped it back to log space, so undo log1p.
                mean = params[0].detach().cpu().numpy()
                if spec.type == "pos":
                    mean = np.expm1(mean)
                    mean = np.maximum(mean, 0.0)
                candidate_blocks.append(mean)
            elif spec.type == "count":
                rate = params[0].detach().cpu().numpy()
                candidate_blocks.append(rate)
            elif spec.type == "cat":
                logits = params[0].detach().cpu().numpy()
                idx = np.argmax(logits, axis=1)
                one_hot = np.zeros_like(logits)
                one_hot[np.arange(len(idx)), idx] = 1.0
                candidate_blocks.append(one_hot)
            elif spec.type == "ordinal":
                probs = params[0].detach().cpu().numpy()
                level = np.argmax(probs, axis=1)
                arange = np.arange(spec.dim).reshape(1, -1)
                thermo = (arange <= level.reshape(-1, 1)).astype(np.float32)
                candidate_blocks.append(thermo)
            cursor += spec.dim

        x_tilde = np.concatenate(candidate_blocks, axis=1).astype(np.float32)

        # Predict on candidates: rebuild a DataFrame in original column order.
        df_candidates = _candidates_to_dataframe(preprocessor, x_tilde, cond_replicated)
        y_tilde = np.asarray(classifier.predict(df_candidates))

        scaled_tilde = scaler.transform(x_tilde)
        l1_dist = np.sum(np.abs(scaled_tilde - scaled_original_replicated), axis=1)

        hits = np.where(y_tilde == target_class)[0]

        if len(hits) == 0:
            low = high
            high = low + step_size
            continue

        order = hits[np.argsort(l1_dist[hits])]
        for j in order:
            if n_accepted >= num_counterfactuals:
                break
            accepted_x[n_accepted] = x_tilde[j]
            accepted_z[n_accepted] = z_tilde[j]
            n_accepted += 1

        if n_accepted >= num_counterfactuals:
            break

        low = high
        high = low + step_size

    return SearchResult(
        counterfactuals_encoded=accepted_x[:n_accepted],
        z_counterfactuals=accepted_z[:n_accepted],
        z_original=z_original,
    )


def _candidates_to_dataframe(preprocessor: Preprocessor, x_free: np.ndarray, x_cond: np.ndarray):
    """Decode encoded candidates back to a user-facing DataFrame for the classifier."""
    free_df = preprocessor.inverse_transform_free(x_free)
    if preprocessor.schema.conditional_specs:
        cond_df = preprocessor._inverse_transform(x_cond, preprocessor.schema.conditional_specs)
        # Reassemble in original column order.
        merged = {**{c: cond_df[c] for c in cond_df.columns}, **{c: free_df[c] for c in free_df.columns}}
        import pandas as pd

        return pd.DataFrame(merged)[preprocessor.schema.column_names]
    return free_df[preprocessor.schema.column_names]
