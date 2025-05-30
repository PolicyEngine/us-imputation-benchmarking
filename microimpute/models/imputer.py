"""Abstract base classes for imputation models.

This module defines the core architecture for imputation models in MicroImpute.
It provides two abstract base classes:
1. Imputer - For model initialization and fitting
2. ImputerResults - For storing fitted models and making predictions

All model implementations should extend these classes to ensure a consistent interface.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
from pydantic import validate_call

from microimpute.config import RANDOM_STATE, VALIDATE_CONFIG


class Imputer(ABC):
    """
    Abstract base class for fitting imputation models.

    All imputation models should inherit from this class and implement
    the required methods.
    """

    def __init__(self, seed: Optional[int] = RANDOM_STATE) -> None:
        """Initialize the imputer model."""
        self.predictors: Optional[List[str]] = None
        self.imputed_variables: Optional[List[str]] = None
        self.seed = seed
        self.logger = logging.getLogger(__name__)

    @validate_call(config=VALIDATE_CONFIG)
    def _validate_data(self, data: pd.DataFrame, columns: List[str]) -> None:
        """Validate that all required columns are in the data.

        Args:
            data: DataFrame to validate
            columns: Column names that should be present

        Raises:
            ValueError: If any columns are missing from the data or if data is empty
        """
        if data is None or data.empty:
            raise ValueError("Data must not be None or empty")

        missing_columns: Set[str] = set(columns) - set(data.columns)
        if missing_columns:
            error_msg = f"Missing columns in data: {missing_columns}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        string_columns = [
            col
            for col in data.columns
            if data[col].dtype == "object" or data[col].dtype == "string"
        ]
        if string_columns:
            error_msg = f"String columns detected: {string_columns}. All columns must be numeric."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    @validate_call(config=VALIDATE_CONFIG)
    def fit(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        weight_col: Optional[Union[str, np.array]] = None,
        **kwargs: Any,
    ) -> Any:  # Returns ImputerResults
        """Fit the model to the training data.

        Args:
            X_train: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            weight_col: Optional name of the column or column array containing
                sampling weights. When provided, `X_train` will be sampled with
                replacement using this column as selection probabilities
                before fitting the model.
            **kwargs: Additional model-specific parameters.

        Returns:
            The fitted model instance.

        Raises:
            ValueError: If input data is invalid or missing required columns.
            RuntimeError: If model fitting fails.
            NotImplementedError: If method is not implemented by subclass.
        """
        try:
            # Validate data
            self._validate_data(X_train, predictors + imputed_variables)

            for variable in imputed_variables:
                if variable in predictors:
                    error_msg = (
                        f"Variable '{variable}' is both in the predictors and imputed "
                        "variables list. Please ensure they are distinct."
                    )
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
        except Exception as e:
            raise ValueError(f"Invalid input data for model: {str(e)}") from e

        if weight_col is not None:
            if isinstance(weight_col, str):
                if weight_col not in X_train.columns:
                    raise ValueError(
                        f"Weight column '{weight_col}' not found in training data"
                    )
                weights = X_train[weight_col]
            else:
                weights = weight_col
            if (weights <= 0).any():
                raise ValueError("Weights must be positive")
            weights_normalized = weights / weights.sum()
            X_train = X_train.sample(
                n=len(X_train),
                replace=True,
                weights=weights_normalized,
                random_state=self.seed,
            ).reset_index(drop=True)

        # Save predictors and imputed variables
        self.predictors = predictors
        self.imputed_variables = imputed_variables

        # Defer actual training to subclass with all parameters
        fitted_model = self._fit(
            X_train, predictors, imputed_variables, **kwargs
        )
        return fitted_model

    @abstractmethod
    @validate_call(config=VALIDATE_CONFIG)
    def _fit(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        **kwargs: Any,
    ) -> None:
        """Actual model-fitting logic (overridden in method subclass).

        Args:
            X_train: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            **kwargs: Additional model-specific parameters.

        Raises:
            ValueError: If specific model parameters are invalid.
            RuntimeError: If model fitting fails.
        """
        raise NotImplementedError("Subclasses must implement `_fit`")


class ImputerResults(ABC):
    """
    Abstract base class representing a fitted model for imputation.

    All imputation models should inherit from this class and implement
    the required methods.

    predict() can only be called once the model is fitted in an
    ImputerResults instance.
    """

    def __init__(
        self,
        predictors: List[str],
        imputed_variables: List[str],
        seed: int,
    ):
        self.predictors = predictors
        self.imputed_variables = imputed_variables
        self.seed = seed
        self.logger = logging.getLogger(__name__)

    @validate_call(config=VALIDATE_CONFIG)
    def _validate_quantiles(
        self,
        quantiles: Optional[List[float]],
    ) -> None:
        """Validate that all provided quantiles are valid.

        Args:
            quantiles: List of quantiles to validate

        Raises:
            ValueError: If passed quantiles are not in the correct format
        """
        if quantiles is not None:
            if not isinstance(quantiles, list):
                self.logger.error(
                    f"quantiles must be a list, got {type(quantiles)}"
                )
                raise ValueError(
                    f"quantiles must be a list, got {type(quantiles)}"
                )

            invalid_quantiles = [q for q in quantiles if not 0 <= q <= 1]
            if invalid_quantiles:
                self.logger.error(
                    f"Invalid quantiles (must be between 0 and 1): {invalid_quantiles}"
                )
                raise ValueError(
                    f"All quantiles must be between 0 and 1, got {invalid_quantiles}"
                )

    @validate_call(config=VALIDATE_CONFIG)
    def predict(
        self,
        X_test: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> Dict[float, pd.DataFrame]:
        """Predict imputed values at specified quantiles.

        Will validate that quantiles passed are in the correct format.

        Args:
            X_test: DataFrame containing the test data.
            quantiles: List of quantiles to predict. If None, uses random quantile.
            **kwargs: Additional model-specific parameters.

        Returns:
            Dictionary mapping quantiles to imputed values.

        Raises:
            ValueError: If input data is invalid.
            RuntimeError: If imputation fails.
        """
        try:
            # Validate quantiles
            self._validate_quantiles(quantiles)
        except Exception as quantile_error:
            raise ValueError(
                f"Invalid quantiles: {str(quantile_error)}"
            ) from quantile_error

        # Defer actual imputations to subclass with all parameters
        imputations = self._predict(X_test, quantiles, **kwargs)
        return imputations

    @abstractmethod
    @validate_call(config=VALIDATE_CONFIG)
    def _predict(
        self, X_test: pd.DataFrame, quantiles: Optional[List[float]] = None
    ) -> Dict[float, pd.DataFrame]:
        """Predict imputed values at specified quantiles.

        Args:
            X_test: DataFrame containing the test data.
            quantiles: List of quantiles to predict. If None, uses random quantile.

        Returns:
            Dictionary mapping quantiles to imputed values.

        Raises:
            RuntimeError: If imputation fails.
            NotImplementedError: If method is not implemented by subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement the predict method"
        )
