"""Tests for the OLS (Ordinary Least Squares) imputation model.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from us_imputation_benchmarking.comparisons.data import preprocess_data
from us_imputation_benchmarking.config import QUANTILES
from us_imputation_benchmarking.evaluations import *
from us_imputation_benchmarking.models.ols import OLS

# Test Method on iris dataset
iris_data = load_iris()
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

predictors = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)"]
imputed_variables = ["petal width (cm)"]

iris_df = iris_df[predictors + imputed_variables]


def test_ols_cross_validation(
    data: pd.DataFrame = iris_df,
    predictors: List[str] = predictors,
    imputed_variables: List[str] = imputed_variables,
    quantiles: List[float] = QUANTILES,
) -> None:
    """
    Test the OLS model on a specific dataset.

    Args:
            data: DataFrame with the dataset of interest.
            predictors: List of predictor variables.
            imputed_variables: List of variables to impute.
            quantiles: List of quantiles to predict.
    """
    ols_results = cross_validate_model(
        OLS, data, predictors, imputed_variables
    )

    ols_results.to_csv("ols_results.csv")

    assert not ols_results.isna().any().any()

    plot_train_test_performance(
        ols_results, save_path="ols_train_test_performance.png"
    )


def test_ols_example(
    data: pd.DataFrame = iris_df,
    predictors: List[str] = predictors,
    imputed_variables: List[str] = imputed_variables,
    quantiles: List[float] = QUANTILES,
) -> None:
    """
    Example of how to use the OLS imputer model.

    This example demonstrates:
    - Initializing an OLS model
    - Fitting the model to training data
    - Predicting quantiles on test data
    - How OLS models assume normally distributed residuals

    Args:
        data: DataFrame with the dataset to use.
        predictors: List of predictor column names.
        imputed_variables: List of target column names.
        quantiles: List of quantiles to predict.
    """
    X_train, X_test = preprocess_data(data)

    # Initialize OLS model
    model = OLS()

    # Fit the model
    fitted_model = model.fit(X_train, predictors, imputed_variables)

    # Predict at multiple quantiles
    predictions: Dict[float, pd.DataFrame] = fitted_model.predict(
        X_test, quantiles
    )

    # Check structure of predictions
    assert isinstance(predictions, dict)
    assert set(predictions.keys()) == set(quantiles)

    # Demonstrate how OLS uses normal distribution assumption
    median_pred = predictions[0.5]
    q10_pred = predictions[0.1]
    q90_pred = predictions[0.9]

    # The difference between q90 and median should approximately equal
    # the difference between median and q10 for OLS (symmetric distribution)
    upper_diff = q90_pred - median_pred
    lower_diff = median_pred - q10_pred

    # Allow some numerical error
    np.testing.assert_allclose(
        upper_diff.mean(),
        lower_diff.mean(),
        rtol=0.1,
        err_msg="OLS should have symmetric quantile predictions around the median",
    )
