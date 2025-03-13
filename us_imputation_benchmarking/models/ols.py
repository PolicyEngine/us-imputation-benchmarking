import statsmodels.api as sm
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import List, Dict, Union, Optional
from us_imputation_benchmarking.models.imputer import Imputer


class OLS(Imputer):
    """
    Ordinary Least Squares regression model for imputation.

    This model predicts different quantiles by assuming normally
    distributed residuals.
    """
    def __init__(self):
        """Initialize the OLS model."""
        super().__init__()
        self.model = None
        self.predictors: Optional[List[str]] = None
        self.imputed_variables: Optional[List[str]] = None

    def fit(
        self,
        X: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
    ) -> "OLS":
        """Fit the OLS model to the training data.

        Args:
            X: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.

        Returns:
            The fitted model instance.
        """
        self.predictors = predictors
        self.imputed_variables = imputed_variables

        Y = X[imputed_variables]
        X_with_const = sm.add_constant(X[predictors])

        self.model = sm.OLS(Y, X_with_const).fit()
        return self

    def predict(
        self, test_X: pd.DataFrame, 
        quantiles: Optional[List[float]] = None
    ) -> Dict[float, np.ndarray]:
        """Predict values at specified quantiles using the OLS model.

        Args:
            test_X: DataFrame containing the test data.
            quantiles: List of quantiles to predict.

        Returns:
            Dictionary mapping quantiles to predicted values.
        """
        imputations: Dict[float, np.ndarray] = {}
        test_X_with_const = sm.add_constant(test_X[self.predictors])

        if quantiles:
            for q in quantiles:
                imputation = self._predict_quantile(test_X_with_const, q)
                imputations[q] = imputation
        else: 
            q = np.random.uniform(0, 1)
            imputation = self._predict_quantile(test_X_with_const, q)
            imputations[q] = imputation

        return imputations

    def _predict_quantile(self, X: pd.DataFrame, q: float) -> np.ndarray:
        """Predict values at a specified quantile.

        Args:
            X: Feature matrix with constant.
            q: Quantile to predict.

        Returns:
            Array of predicted values at the specified quantile.
        """
        mean_pred = self.model.predict(X)
        se = np.sqrt(self.model.scale)
        return mean_pred + norm.ppf(q) * se
