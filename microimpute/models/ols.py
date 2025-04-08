"""Ordinary Least Squares regression model for imputation."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import validate_call
from scipy.stats import norm

from microimpute.config import VALIDATE_CONFIG
from microimpute.models.imputer import Imputer, ImputerResults


class OLSResults(ImputerResults):
    """
    Fitted OLS instance ready for imputation.
    """

    def __init__(
        self,
        model: "OLS",
        predictors: List[str],
        imputed_variables: List[str],
    ) -> None:
        """Initialize the OLS results.

        Args:
            model: Fitted OLS model.
            predictors: List of predictor variable names.
            imputed_variables: List of imputed variable names.
        """
        super().__init__(predictors, imputed_variables)
        self.model = model

    @validate_call(config=VALIDATE_CONFIG)
    def _predict(
        self, X_test: pd.DataFrame, quantiles: Optional[List[float]] = None
    ) -> Dict[float, pd.DataFrame]:
        """Predict values at specified quantiles using the OLS model.

        Args:
            X_test: DataFrame containing the test data.
            quantiles: List of quantiles to predict.

        Returns:
            Dictionary mapping quantiles to predicted values.

        Raises:
            RuntimeError: If prediction fails.
        """
        try:
            # Create output dictionary with results
            imputations: Dict[float, pd.DataFrame] = {}

            X_test_with_const = sm.add_constant(X_test[self.predictors])

            if quantiles:
                self.logger.info(
                    f"Predicting at {len(quantiles)} quantiles: {quantiles}"
                )
                for q in quantiles:
                    imputation = self._predict_quantile(X_test_with_const, q)
                    imputations[q] = pd.DataFrame(imputation)
            else:
                q = np.random.uniform(0, 1)
                self.logger.info(f"Predicting at random quantile: {q:.3f}")
                imputation = self._predict_quantile(X_test_with_const, q)
                imputations[q] = pd.DataFrame(imputation)
            return imputations

        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise RuntimeError(
                f"Failed to predict with OLS model: {str(e)}"
            ) from e

    @validate_call(config=VALIDATE_CONFIG)
    def _predict_quantile(self, X: pd.DataFrame, q: float) -> np.ndarray:
        """Predict values at a specified quantile.

        Args:
            X: Feature matrix with constant.
            q: Quantile to predict.

        Returns:
            Array of predicted values at the specified quantile.

        Raises:
            RuntimeError: If prediction fails.
        """
        try:
            mean_pred = self.model.predict(X)
            se = np.sqrt(self.model.scale)

            return mean_pred + norm.ppf(q) * se
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            self.logger.error(f"Error predicting at quantile {q}: {str(e)}")
            raise RuntimeError(
                f"Failed to predict at quantile {q}: {str(e)}"
            ) from e


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
        self.logger.debug("Initializing OLS imputer")

    @validate_call(config=VALIDATE_CONFIG)
    def _fit(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
    ) -> OLSResults:
        """Fit the OLS model to the training data.

        Args:
            X_train: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.

        Returns:
            The fitted model instance.

        Raises:
            RuntimeError: If model fitting fails.
        """
        try:
            self.logger.info(
                f"Fitting OLS model with {len(predictors)} predictors"
            )

            Y = X_train[imputed_variables]
            X_with_const = sm.add_constant(X_train[predictors])

            self.model = sm.OLS(Y, X_with_const).fit()
            self.logger.info(
                f"OLS model fitted successfully, R-squared: {self.model.rsquared:.4f}"
            )
            return OLSResults(
                model=self.model,
                predictors=predictors,
                imputed_variables=imputed_variables,
            )
        except Exception as e:
            self.logger.error(f"Error fitting OLS model: {str(e)}")
            raise RuntimeError(f"Failed to fit OLS model: {str(e)}") from e
