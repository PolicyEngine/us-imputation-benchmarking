import statsmodels.api as sm
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Collection, Any
from us_imputation_benchmarking.models.imputer import Imputer


class QuantReg(Imputer):
    """
    Quantile Regression model for imputation.

    This model uses statsmodels' QuantReg implementation to
    directly predict specific quantiles.
    """
    def __init__(self):
        """Initialize the Quantile Regression model."""
        super().__init__()
        self.models: Dict[float, Any] = {}

    def fit(
        self,
        X: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        quantiles: Optional[List[float]] = None
    ) -> "QuantReg":
        """Fit the Quantile Regression model to the training data.

        Args:
            X: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            quantiles: List of quantiles to fit models for.

        Returns:
            The fitted model instance.
        """
        self.predictors = predictors
        self.imputed_variables = imputed_variables

        Y = X[imputed_variables]
        X_with_const = sm.add_constant(X[predictors])

        if quantiles:
            for q in quantiles:
                self.models[q] = sm.QuantReg(Y, X_with_const).fit(q=q)
        else:
            q = np.random.uniform(0,1)
            self.models[q] = sm.QuantReg(Y, X_with_const).fit(q=q)

        return self

    def predict(
        self, test_X: pd.DataFrame, quantiles: Optional[List[float]] = None
    ) -> Dict[float, np.ndarray]:
        """Predict values at specified quantiles using the Quantile Regression model.

        Args:
            test_X: DataFrame containing the test data.
            quantiles: List of quantiles to predict. If None, uses the quantiles
                from training.

        Returns:
            Dictionary mapping quantiles to predicted values.

        Raises:
            ValueError: If a requested quantile was not fitted during training.
        """
        imputations: Dict[float, np.ndarray] = {}
        test_X_with_const = sm.add_constant(test_X[self.predictors])

        if quantiles is None:
            quantiles = list(self.models.keys())

        for q in quantiles:
            if q not in self.models:
                raise ValueError(
                    f"Model for quantile {q} not fitted. Available quantiles: {list(self.models.keys())}"
                )

            imputation = self.models[q].predict(test_X_with_const)
            imputations[q] = imputation

        return imputations
