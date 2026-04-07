"""The :class:`CCHVAE` public explainer class."""

from __future__ import annotations

import logging
import random
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
import torch

from cchvae.counterfactual import Counterfactual
from cchvae.errors import CCHVAEError, CCHVAEValueError
from cchvae.modules.vae import CCHVAEModel
from cchvae.preprocessing import Preprocessor, infer_schema
from cchvae.search import latent_shell_search
from cchvae.types import FeatureType

logger = logging.getLogger("cchvae")


@runtime_checkable
class ClassifierProtocol(Protocol):
    """Minimal classifier interface used by :class:`CCHVAE`."""

    def predict(self, X) -> np.ndarray: ...  # noqa: N803


class CCHVAE:
    """Counterfactual explanations for tabular classifiers via a heterogeneous VAE.

    Parameters
    ----------
    classifier : ClassifierProtocol
        Any object exposing ``.predict(DataFrame) -> array``. Predictions on
        the original feature space are used both to determine the original
        prediction of each instance and to validate counterfactual candidates.
    background_data : pd.DataFrame
        Reference data used to (a) infer the feature schema, (b) fit the
        :class:`Preprocessor`, and (c) train the VAE if no pretrained one
        is supplied.
    target_class : int | str
        Desired prediction value for counterfactuals (the "flip-to" class).
    immutable_features : list[str] | None, default ``None``
        Columns the counterfactual search may not change.
    feature_types : dict[str, FeatureType] | None, default ``None``
        Per-column overrides for type inference. Keys must match
        ``background_data.columns``.
    latent_dim : int, default ``2``
        Dimensionality of the continuous latent ``z``.
    intermediate_dim : int, default ``5``
        Per-feature width of the intermediate ``y`` representation.
    categorical_latent_dim : int, default ``3``
        Number of categorical mixture components ``s``.
    learning_rate : float, default ``1e-3``
        Adam learning rate (training only).
    epochs : int, default ``80``
        Number of training epochs (training only).
    batch_size : int, default ``100``
        Mini-batch size (training only).
    pretrained_vae : CCHVAEModel | None, default ``None``
        If provided, training is skipped and this model is used directly.
        Its schema and dims must match the inferred schema.
    device : str, default ``"cpu"``
        Torch device.
    random_state : int | None, default ``619``
        Seed for ``random``, NumPy, and PyTorch.
    verbose : bool, default ``False``
        Forwarded to ``CCHVAEModel.fit``.

    Raises
    ------
    CCHVAEValueError
        If any constructor argument fails validation.
    """

    def __init__(
        self,
        classifier: ClassifierProtocol,
        background_data: pd.DataFrame,
        target_class: int | str,
        immutable_features: list[str] | None = None,
        feature_types: dict[str, FeatureType] | None = None,
        latent_dim: int = 2,
        intermediate_dim: int = 5,
        categorical_latent_dim: int = 3,
        learning_rate: float = 1e-3,
        epochs: int = 80,
        batch_size: int = 100,
        pretrained_vae: CCHVAEModel | None = None,
        device: str = "cpu",
        random_state: int | None = 619,
        verbose: bool = False,
    ) -> None:
        self._validate_init(
            classifier=classifier,
            background_data=background_data,
            target_class=target_class,
            immutable_features=immutable_features,
            latent_dim=latent_dim,
            intermediate_dim=intermediate_dim,
            categorical_latent_dim=categorical_latent_dim,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
        )

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
            torch.manual_seed(random_state)

        self.classifier = classifier
        self.background_data = background_data.reset_index(drop=True)
        self.target_class = target_class
        self.device = device
        self.verbose = verbose

        self.schema = infer_schema(
            background_data,
            feature_types=feature_types,
            immutable_features=immutable_features,
        )
        self.preprocessor = Preprocessor(self.schema).fit(self.background_data)

        bg_free, bg_cond = self.preprocessor.transform(self.background_data)
        self._background_free_encoded = bg_free

        if pretrained_vae is None:
            self.model = CCHVAEModel(
                schema=self.schema,
                normalization_params=self.preprocessor.normalization_params,
                z_dim=latent_dim,
                y_dim=intermediate_dim,
                s_dim=categorical_latent_dim,
            )
            self.model.fit(
                x_free=bg_free,
                x_cond=bg_cond,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
                verbose=verbose,
            )
        else:
            if pretrained_vae.z_dim != latent_dim:
                raise CCHVAEValueError(
                    "pretrained_vae latent_dim does not match the requested latent_dim.",
                    config={"expected": latent_dim, "received": pretrained_vae.z_dim},
                    param="pretrained_vae",
                    hint="Pass a model that was built with the same latent_dim.",
                    source="CCHVAE.__init__",
                )
            if [s.name for s in pretrained_vae.schema.specs] != [s.name for s in self.schema.specs]:
                raise CCHVAEValueError(
                    "pretrained_vae schema columns do not match the inferred schema.",
                    param="pretrained_vae",
                    hint="Use the same background_data the VAE was trained on.",
                    source="CCHVAE.__init__",
                )
            self.model = pretrained_vae.to(torch.device(device))
            self.model.eval()

    def generate_counterfactuals(
        self,
        instances: pd.DataFrame,
        num_counterfactuals: int = 5,
        search_samples: int = 1000,
        max_iterations: int = 500,
        step_size: float = 0.5,
        norm: int = 2,
    ) -> Counterfactual | list[Counterfactual]:
        """Generate counterfactuals for one or more instances.

        Parameters
        ----------
        instances : pd.DataFrame
            One or more rows to explain. Must contain the same columns as
            the ``background_data`` passed to the constructor.
        num_counterfactuals : int, default ``5``
            Number of counterfactuals to return per instance.
        search_samples : int, default ``1000``
            Latent perturbations sampled per shell expansion.
        max_iterations : int, default ``500``
            Max shell expansions before giving up on a single instance.
        step_size : float, default ``0.5``
            Width of each shell expansion.
        norm : int, default ``2``
            L^p norm for the latent shell.

        Returns
        -------
        Counterfactual | list[Counterfactual]
            A single :class:`Counterfactual` if ``instances`` had one row,
            otherwise a list (one entry per *successful* instance).

        Raises
        ------
        CCHVAEValueError
            If ``instances`` is malformed.
        CCHVAEError
            If no counterfactuals could be found for any instance.
        """
        self._validate_instances(instances)
        was_single = len(instances) == 1
        instances = instances.reset_index(drop=True)

        x_free, x_cond = self.preprocessor.transform(instances)
        original_predictions = np.asarray(self.classifier.predict(instances[self.schema.column_names]))

        results: list[Counterfactual] = []
        for i in range(len(instances)):
            row_free = x_free[i : i + 1]
            row_cond = x_cond[i : i + 1] if x_cond.size else x_cond.reshape(1, 0)

            search = latent_shell_search(
                model=self.model,
                preprocessor=self.preprocessor,
                classifier=self.classifier,
                target_class=self.target_class,
                x_free_row=row_free,
                x_cond_row=row_cond,
                background_free=self._background_free_encoded,
                num_counterfactuals=num_counterfactuals,
                search_samples=search_samples,
                max_iterations=max_iterations,
                step_size=step_size,
                norm=norm,
                device=self.device,
            )

            if search.counterfactuals_encoded.shape[0] == 0:
                logger.warning("No counterfactuals found for instance %d.", i)
                continue

            cf_df = self._decoded_counterfactuals(search.counterfactuals_encoded, instances.iloc[i])
            cf_predictions = np.asarray(self.classifier.predict(cf_df[self.schema.column_names])).tolist()

            results.append(
                Counterfactual(
                    original_instance=instances.iloc[i],
                    counterfactual_instance=cf_df,
                    original_prediction=_unwrap_prediction(original_predictions[i]),
                    counterfactual_prediction=[_unwrap_prediction(p) for p in cf_predictions],
                )
            )

        if not results:
            raise CCHVAEError(
                "No counterfactuals were found for any instance.",
                config={"n_instances": len(instances)},
                param="instances",
                hint=("Try increasing search_samples, max_iterations, or step_size, or relax the target_class."),
                source="CCHVAE.generate_counterfactuals",
            )

        if was_single and len(results) == 1:
            return results[0]
        return results

    def _decoded_counterfactuals(self, encoded: np.ndarray, original_row: pd.Series) -> pd.DataFrame:
        """Inverse-transform encoded CFs back to a DataFrame, restoring immutables."""
        free_df = self.preprocessor.inverse_transform_free(encoded)
        for col in self.schema.immutable_columns:
            free_df[col] = original_row[col]
        return free_df[self.schema.column_names]

    def _validate_init(
        self,
        *,
        classifier: object,
        background_data: object,
        target_class: object,
        immutable_features: object,
        latent_dim: int,
        intermediate_dim: int,
        categorical_latent_dim: int,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        device: str,
    ) -> None:
        if not isinstance(classifier, ClassifierProtocol):
            raise CCHVAEValueError(
                "classifier must implement a .predict(DataFrame) method.",
                config={"type": type(classifier).__name__},
                param="classifier",
                hint="Pass any sklearn-style classifier (e.g. RandomForestClassifier).",
                source="CCHVAE._validate_init",
            )
        if not isinstance(background_data, pd.DataFrame):
            raise CCHVAEValueError(
                "background_data must be a pandas DataFrame.",
                config={"type": type(background_data).__name__},
                param="background_data",
                source="CCHVAE._validate_init",
            )
        if background_data.empty:
            raise CCHVAEValueError(
                "background_data must contain at least one row.",
                param="background_data",
                source="CCHVAE._validate_init",
            )
        if not isinstance(target_class, int | str):
            raise CCHVAEValueError(
                "target_class must be an int or str.",
                config={"type": type(target_class).__name__},
                param="target_class",
                source="CCHVAE._validate_init",
            )
        if immutable_features is not None and not isinstance(immutable_features, list):
            raise CCHVAEValueError(
                "immutable_features must be a list of column names or None.",
                config={"type": type(immutable_features).__name__},
                param="immutable_features",
                source="CCHVAE._validate_init",
            )
        for name, value in [
            ("latent_dim", latent_dim),
            ("intermediate_dim", intermediate_dim),
            ("categorical_latent_dim", categorical_latent_dim),
            ("epochs", epochs),
            ("batch_size", batch_size),
        ]:
            if not isinstance(value, int) or value < 1:
                raise CCHVAEValueError(
                    f"{name} must be a positive int.",
                    config={"value": value},
                    param=name,
                    source="CCHVAE._validate_init",
                )
        if not isinstance(learning_rate, float) or learning_rate <= 0:
            raise CCHVAEValueError(
                "learning_rate must be a positive float.",
                config={"value": learning_rate},
                param="learning_rate",
                source="CCHVAE._validate_init",
            )
        if device not in ("cpu", "cuda") and not device.startswith("cuda:"):
            raise CCHVAEValueError(
                "device must be 'cpu', 'cuda', or 'cuda:<index>'.",
                config={"value": device},
                param="device",
                source="CCHVAE._validate_init",
            )

    def _validate_instances(self, instances: object) -> None:
        if not isinstance(instances, pd.DataFrame):
            raise CCHVAEValueError(
                "instances must be a pandas DataFrame.",
                config={"type": type(instances).__name__},
                param="instances",
                source="CCHVAE.generate_counterfactuals",
            )
        if instances.empty:
            raise CCHVAEValueError(
                "instances must contain at least one row.",
                param="instances",
                source="CCHVAE.generate_counterfactuals",
            )
        missing = [c for c in self.schema.column_names if c not in instances.columns]
        if missing:
            raise CCHVAEValueError(
                f"instances is missing columns: {missing}",
                config={"expected": self.schema.column_names, "received": list(instances.columns)},
                param="instances",
                source="CCHVAE.generate_counterfactuals",
            )


def _unwrap_prediction(value):
    """Cast numpy scalars to plain int / float / str so Counterfactual accepts them."""
    if isinstance(value, np.generic):
        return value.item()
    return value
