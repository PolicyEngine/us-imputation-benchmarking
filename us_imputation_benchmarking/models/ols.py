from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from us_imputation_benchmarking.models.imputer import Imputer


class OLS(Imputer):
    """
    Ordinary Least Squares regression model for imputation.

    This model predicts different quantiles by assuming normally
    distributed residuals.
    """

    def __init__(self) -> None:
        """Initialize the OLS model."""
        super().__init__()
        self.model = None
        self.predictors: Optional[List[str]] = None
        self.imputed_variables: Optional[List[str]] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
    ) -> "OLS":
        """Fit the OLS model to the training data.

        Args:
            X_train: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.

        Returns:
            The fitted model instance.

        Raises:
            ValueError: If input data is invalid or missing required columns.
            RuntimeError: If model fitting fails.
        """
        try:
            # Validate input data
            self._validate_data(X_train, predictors + imputed_variables, "training")

            self.predictors = predictors
            self.imputed_variables = imputed_variables
            self.logger.info(f"Fitting OLS model with {len(predictors)} predictors")

            Y = X_train[imputed_variables]
            X_with_const = sm.add_constant(X_train[predictors])

            self.model = sm.OLS(Y, X_with_const).fit()
            self.logger.info(
                f"OLS model fitted successfully, R-squared: {self.model.rsquared:.4f}"
            )
            return self
        except Exception as e:
            self.logger.error(f"Error fitting OLS model: {str(e)}")
            raise RuntimeError(f"Failed to fit OLS model: {str(e)}") from e

    def predict(
        self, X_test: pd.DataFrame, quantiles: Optional[List[float]] = None
    ) -> Dict[float, np.ndarray]:
        """Predict values at specified quantiles using the OLS model.

        Args:
            X_test: DataFrame containing the test data.
            quantiles: List of quantiles to predict.

        Returns:
            Dictionary mapping quantiles to predicted values.

        Raises:
            ValueError: If model is not fitted or input data is invalid.
            RuntimeError: If prediction fails.
        """
        try:
            if self.model is None:
                error_msg = "Model must be fitted before prediction"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Validate input data
            self._validate_data(X_test, self.predictors, "prediction")

            imputations: Dict[float, np.ndarray] = {}
            X_test_with_const = sm.add_constant(X_test[self.predictors])

            if quantiles:
                self.logger.info(
                    f"Predicting at {len(quantiles)} quantiles: {quantiles}"
                )
                for q in quantiles:
                    imputation = self._predict_quantile(X_test_with_const, q)
                    imputations[q] = imputation
            else:
                q = np.random.uniform(0, 1)
                self.logger.info(f"Predicting at random quantile: {q:.4f}")
                imputation = self._predict_quantile(X_test_with_const, q)
                imputations[q] = imputation

            return imputations
        except ValueError as e:
            # Re-raise ValueError for specific error types
            raise e
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise RuntimeError(f"Failed to predict with OLS model: {str(e)}") from e

    def _predict_quantile(self, X: pd.DataFrame, q: float) -> np.ndarray:
        """Predict values at a specified quantile.

        Args:
            X: Feature matrix with constant.
            q: Quantile to predict.

        Returns:
            Array of predicted values at the specified quantile.

        Raises:
            ValueError: If quantile value is outside [0, 1] range.
            RuntimeError: If prediction fails.
        """
        try:
            if not 0 <= q <= 1:
                error_msg = f"Quantile must be between 0 and 1, got {q}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            mean_pred = self.model.predict(X)
            se = np.sqrt(self.model.scale)

            return mean_pred + norm.ppf(q) * se
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            self.logger.error(f"Error predicting at quantile {q}: {str(e)}")
            raise RuntimeError(f"Failed to predict at quantile {q}: {str(e)}") from e
