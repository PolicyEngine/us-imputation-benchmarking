"""Quantile Random Forest (QRF) utilities.

This module provides specialized functionality for Quantile Random Forest implementations.
It builds on scikit-learn's random forest to enable quantile predictions.
"""

import logging
import os
import pickle
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from pydantic import validate_call
from quantile_forest import RandomForestQuantileRegressor

from microimpute.config import RANDOM_STATE, VALIDATE_CONFIG


class QRF:
    categorical_columns: Optional[List[str]] = None
    encoded_columns: Optional[List[str]] = None
    output_columns: Optional[List[str]] = None

    def __init__(
        self, seed: int = RANDOM_STATE, file_path: Optional[str] = None
    ) -> None:
        """Initialize Quantile Random Forest.

        Args:
            seed: Random seed for reproducibility.
            file_path: Path to a pickled model file to load.

        Raises:
            FileNotFoundError: If file_path is provided but doesn't exist
            RuntimeError: If file loading fails
        """
        self.logger = logging.getLogger(__name__)
        self.logger.debug(
            f"Initializing QRF with seed={seed}, file_path={file_path}"
        )
        self.seed = seed
        self.qrf = None

        if file_path is not None:
            try:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(
                        f"Model file not found: {file_path}"
                    )

                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                self.seed = data["seed"]
                self.categorical_columns = data["categorical_columns"]
                self.encoded_columns = data["encoded_columns"]
                self.output_columns = data["output_columns"]
                self.qrf = data["qrf"]
                self.logger.info(
                    f"Successfully loaded QRF model from {file_path}"
                )
            except (pickle.PickleError, KeyError) as e:
                self.logger.error(
                    f"Failed to load model from {file_path}: {str(e)}"
                )
                raise RuntimeError(
                    f"Failed to load QRF model from {file_path}"
                ) from e

    @validate_call(config=VALIDATE_CONFIG)
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **qrf_kwargs: Any) -> None:
        """Fit the Quantile Random Forest model.

        Args:
            X: Feature DataFrame.
            y: Target DataFrame.
            **qrf_kwargs: Additional keyword arguments to pass
                to RandomForestQuantileRegressor.

        Raises:
            ValueError: If X or y are empty or have incompatible shapes
            RuntimeError: If model fitting fails
        """
        self.logger.debug(
            f"Fitting QRF with X shape {X.shape}, y shape {y.shape}"
        )

        # Validate inputs
        if len(X) != len(y):
            self.logger.error(
                f"Shape mismatch: X has {len(X)} rows, y has {len(y)} rows"
            )
            raise ValueError(
                f"X and y must have same number of rows, got {len(X)} and {len(y)}"
            )

        try:
            self.categorical_columns = X.select_dtypes(
                include=["object"]
            ).columns
            self.logger.debug(
                f"Categorical columns: {list(self.categorical_columns)}"
            )

            X = pd.get_dummies(
                X, columns=self.categorical_columns, drop_first=True
            )
            self.encoded_columns = X.columns
            self.output_columns = y.columns

            self.logger.info(
                f"Creating QRF with seed={self.seed} and {len(self.encoded_columns)} features"
            )
            self.qrf = RandomForestQuantileRegressor(
                random_state=self.seed, **qrf_kwargs
            )

            self.logger.info("Fitting QRF model...")
            self.qrf.fit(X, y)
            self.logger.info("QRF model fitted successfully")

        except Exception as e:
            self.logger.error(f"Failed to fit QRF model: {str(e)}")
            raise RuntimeError("Failed to fit QRF model") from e

    @validate_call(config=VALIDATE_CONFIG)
    def predict(
        self,
        X: pd.DataFrame,
        count_samples: int = 10,
        mean_quantile: float = 0.5,
    ) -> pd.DataFrame:
        """Make predictions with the Quantile Random Forest model.

        Args:
            X: Feature DataFrame.
            count_samples: Number of quantile samples to generate.
            mean_quantile: Target quantile for predictions (between 0 and 1).

        Returns:
            pd.DataFrame: DataFrame with predicted values, with the same columns as the
                         target variable used during training.

        Raises:
            ValueError: If X is empty, model isn't fitted, or parameters are invalid
            RuntimeError: If prediction fails
        """
        self.logger.debug(
            f"Predicting with QRF for {len(X)} samples, "
            f"count_samples={count_samples}, mean_quantile={mean_quantile}"
        )

        # Validate inputs
        if self.qrf is None:
            self.logger.error("QRF model has not been fitted")
            raise ValueError("Model must be fitted before prediction")

        if count_samples <= 0:
            self.logger.error(f"Invalid count_samples: {count_samples}")
            raise ValueError("count_samples must be positive")

        if mean_quantile <= 0 or mean_quantile >= 1:
            self.logger.error(f"Invalid mean_quantile: {mean_quantile}")
            raise ValueError(
                "mean_quantile must be between 0 and 1 (exclusive)"
            )

        try:
            if (
                self.categorical_columns is None
                or self.encoded_columns is None
            ):
                self.logger.error(
                    "Model not properly initialized with required attributes"
                )
                raise ValueError("Model not properly initialized")

            self.logger.debug(
                "Encoding categorical features with one-hot encoding"
            )
            X = pd.get_dummies(
                X, columns=self.categorical_columns, drop_first=True
            )

            # Check for missing columns
            missing_cols = set(self.encoded_columns) - set(X.columns)
            if missing_cols:
                self.logger.warning(
                    f"Missing columns in input data: {missing_cols}"
                )
                for col in missing_cols:
                    X[col] = 0  # Add missing columns with zeros

            # Ensure columns are in the same order as during training
            X = X[self.encoded_columns]

            self.logger.debug(
                f"Making predictions with {count_samples} quantile samples"
            )
            pred = self.qrf.predict(
                X, quantiles=list(np.linspace(0, 1, count_samples))
            )

            self.logger.debug(
                f"Generating beta distribution with a={mean_quantile/(1-mean_quantile)}"
            )
            random_generator = np.random.default_rng(self.seed)
            a = mean_quantile / (1 - mean_quantile)
            input_quantiles = (
                random_generator.beta(a, 1, size=len(X)) * count_samples
            )
            input_quantiles = input_quantiles.astype(int)

            # Handle edge cases to prevent index errors
            input_quantiles = np.clip(input_quantiles, 0, count_samples - 1)

            self.logger.debug(
                f"Extracting predictions from shape {pred.shape}"
            )
            if len(pred.shape) == 2:
                predictions = pred[np.arange(len(pred)), input_quantiles]
            else:
                predictions = pred[np.arange(len(pred)), :, input_quantiles]

            self.logger.info(f"Generated predictions for {len(X)} samples")
            return pd.DataFrame(predictions, columns=self.output_columns)

        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise RuntimeError(
                "Failed to generate predictions with QRF model"
            ) from e

    @validate_call(config=VALIDATE_CONFIG)
    def save(self, path: str) -> None:
        """Save the model to disk.

        Args:
            path: File path to save the pickled model.

        Raises:
            ValueError: If the model hasn't been fitted
            RuntimeError: If saving fails
        """
        self.logger.debug(f"Saving QRF model to {path}")

        # Validate model state
        if self.qrf is None:
            self.logger.error("Cannot save unfitted model")
            raise ValueError("Model must be fitted before saving")

        try:
            # Ensure directory exists
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                self.logger.debug(f"Creating directory: {directory}")
                os.makedirs(directory, exist_ok=True)

            self.logger.debug("Serializing model data")
            with open(path, "wb") as f:
                pickle.dump(
                    {
                        "seed": self.seed,
                        "categorical_columns": self.categorical_columns,
                        "encoded_columns": self.encoded_columns,
                        "output_columns": self.output_columns,
                        "qrf": self.qrf,
                    },
                    f,
                )
            self.logger.info(f"Successfully saved QRF model to {path}")

        except (OSError, pickle.PickleError) as e:
            self.logger.error(f"Failed to save model to {path}: {str(e)}")
            raise RuntimeError(f"Failed to save QRF model to {path}") from e
