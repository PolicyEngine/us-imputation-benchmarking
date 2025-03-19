from typing import Any, Collection, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm

from us_imputation_benchmarking.models.imputer import Imputer
from us_imputation_benchmarking.utils.logging_utils import get_logger


class QuantReg(Imputer):
    """
    Quantile Regression model for imputation.

    This model uses statsmodels' QuantReg implementation to
    directly predict specific quantiles.
    """

    def __init__(self) -> None:
        """Initialize the Quantile Regression model."""
        super().__init__()
        self.models: Dict[float, Any] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        quantiles: Optional[List[float]] = None,
    ) -> "QuantReg":
        """Fit the Quantile Regression model to the training data.

        Args:
            X_train: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            quantiles: List of quantiles to fit models for.

        Returns:
            The fitted model instance.

        Raises:
            ValueError: If input data is invalid or missing required columns.
            ValueError: If any quantile is outside the [0, 1] range.
            RuntimeError: If model fitting fails.
        """
        try:
            # Validate input data
            self._validate_data(X_train, predictors + imputed_variables, "training")

            # Validate quantiles if provided
            if quantiles:
                invalid_quantiles = [q for q in quantiles if not 0 <= q <= 1]
                if invalid_quantiles:
                    error_msg = (
                        f"Quantiles must be between 0 and 1, got: {invalid_quantiles}"
                    )
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                self.logger.info(
                    f"Fitting QuantReg models for {len(quantiles)} quantiles: {quantiles}"
                )
            else:
                self.logger.info(
                    "No quantiles provided, will fit a single random quantile"
                )

            self.predictors = predictors
            self.imputed_variables = imputed_variables

            Y = X_train[imputed_variables]
            X_with_const = sm.add_constant(X_train[predictors])
            self.logger.info(
                f"Prepared training data with {len(X_train)} samples, {len(predictors)} predictors"
            )

            if quantiles:
                for q in quantiles:
                    self.logger.info(f"Fitting quantile regression for q={q}")
                    self.models[q] = sm.QuantReg(Y, X_with_const).fit(q=q)
                    self.logger.info(f"Model for q={q} fitted successfully")
            else:
                q = np.random.uniform(0, 1)
                self.logger.info(f"Fitting quantile regression for random q={q:.4f}")
                self.models[q] = sm.QuantReg(Y, X_with_const).fit(q=q)
                self.logger.info(f"Model for q={q:.4f} fitted successfully")

            self.logger.info(f"QuantReg has {len(self.models)} fitted models")
            return self

        except Exception as e:
            self.logger.error(f"Error fitting QuantReg model: {str(e)}")
            raise RuntimeError(f"Failed to fit QuantReg model: {str(e)}") from e

    def predict(
        self, X_test: pd.DataFrame, quantiles: Optional[List[float]] = None
    ) -> Dict[float, np.ndarray]:
        """Predict values at specified quantiles using the Quantile Regression model.

        Args:
            X_test: DataFrame containing the test data.
            quantiles: List of quantiles to predict. If None, uses the quantiles
                from training.

        Returns:
            Dictionary mapping quantiles to predicted values.

        Raises:
            ValueError: If a requested quantile was not fitted during training.
            ValueError: If model is not fitted or input data is invalid.
            RuntimeError: If prediction fails.
        """
        try:
            # Check if model is fitted
            if not self.models:
                error_msg = "No models have been fitted yet. Call fit() first."
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Validate predictors
            if self.predictors is None or self.imputed_variables is None:
                error_msg = "Model not properly initialized with predictors and imputed variables"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Validate input data
            self._validate_data(X_test, self.predictors, "prediction")

            # Process quantiles parameter
            if quantiles is None:
                quantiles = list(self.models.keys())
                self.logger.info(
                    f"Using {len(quantiles)} quantiles from fitted models: {quantiles}"
                )
            else:
                self.logger.info(f"Predicting at {len(quantiles)} requested quantiles")

            imputations: Dict[float, np.ndarray] = {}
            X_test_with_const = sm.add_constant(X_test[self.predictors])
            self.logger.info(f"Prepared test data with {len(X_test)} samples")

            # Predict for each quantile
            for q in quantiles:
                if q not in self.models:
                    error_msg = f"Model for quantile {q} not fitted. Available quantiles: {list(self.models.keys())}"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)

                self.logger.info(f"Predicting with model for q={q}")
                imputation = self.models[q].predict(X_test_with_const)
                imputations[q] = imputation

            self.logger.info(f"Completed predictions for {len(quantiles)} quantiles")
            return imputations

        except ValueError as e:
            # Re-raise value errors directly
            raise e
        except Exception as e:
            self.logger.error(f"Error in QuantReg prediction: {str(e)}")
            raise RuntimeError(
                f"Failed to predict with QuantReg model: {str(e)}"
            ) from e
