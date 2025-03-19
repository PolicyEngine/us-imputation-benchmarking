from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

from us_imputation_benchmarking.utils.logging_utils import get_logger


class Imputer(ABC):
    """
    Abstract base class for imputation models.

    All imputation models should inherit from this class and implement
    the required methods.
    """

    def __init__(self, random_seed=None) -> None:
        """Initialize the imputer model."""
        self.predictors: Optional[List[str]] = None
        self.imputed_variables: Optional[List[str]] = None
        self.seed = random_seed
        self.logger = get_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    def _validate_data(
        self, data: pd.DataFrame, columns: List[str], context: str
    ) -> None:
        """Validate that all required columns are in the data.

        Args:
            data: DataFrame to validate
            columns: Column names that should be present
            context: Context string for error messages (e.g., 'training', 'prediction')

        Raises:
            ValueError: If any columns are missing from the data
        """
        missing_columns: Set[str] = set(columns) - set(data.columns)
        if missing_columns:
            error_msg = f"Missing columns in {context} data: {missing_columns}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        **kwargs: Any,
    ) -> "Imputer":
        """Fit the model to the training data.

        Args:
            X_train: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            **kwargs: Additional keyword arguments for specific model implementations.

        Returns:
            The fitted model instance.

        Raises:
            ValueError: If input data is invalid or missing required columns.
            RuntimeError: If model fitting fails.
        """
        pass

    @abstractmethod
    def predict(
        self, X_test: pd.DataFrame, quantiles: Optional[List[float]] = None
    ) -> Dict[float, Union[np.ndarray, pd.DataFrame]]:
        """Predict imputed values at specified quantiles.

        Args:
            X_test: DataFrame containing the test data.
            quantiles: List of quantiles to predict. If None, uses random quantile.

        Returns:
            Dictionary mapping quantiles to predicted values.

        Raises:
            ValueError: If model is not fitted or input data is invalid.
            RuntimeError: If prediction fails.
        """
        pass
