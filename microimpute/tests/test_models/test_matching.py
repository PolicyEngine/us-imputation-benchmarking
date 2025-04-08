"""Tests for the Statistical Matching imputation model."""

from typing import Dict, List

import pandas as pd
from sklearn.datasets import load_iris

from microimpute.comparisons.data import preprocess_data
from microimpute.config import QUANTILES
from microimpute.evaluations import *
from microimpute.models.matching import Matching

# Test Method on iris dataset
iris_data = load_iris()
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

predictors = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)"]
imputed_variables = ["petal width (cm)"]

iris_df = iris_df[predictors + imputed_variables]


def test_matching_cross_validation(
    data: pd.DataFrame = iris_df,
    predictors: List[str] = predictors,
    imputed_variables: List[str] = imputed_variables,
    quantiles: List[float] = QUANTILES,
) -> None:
    """
    Test the Matching model on a specific dataset.

    Args:
            data: DataFrame with the dataset of interest.
            predictors: List of predictor variables.
            imputed_variables: List of variables to impute.
            quantiles: List of quantiles to predict.
    """
    matching_results = cross_validate_model(
        Matching, data, predictors, imputed_variables
    )

    matching_results.to_csv("matching_results.csv")

    assert not matching_results.isna().any().any()

    plot_train_test_performance(
        matching_results, save_path="matching_train_test_performance.png"
    )


def test_matching_example_use(
    data: pd.DataFrame = iris_df,
    predictors: List[str] = predictors,
    imputed_variables: List[str] = imputed_variables,
    quantiles: List[float] = QUANTILES,
) -> None:
    """
    Example of how to use the Statistical Matching imputer model.

    This example demonstrates:
    - Initializing a Matching model
    - Fitting the model to donor data
    - Predicting values for recipient data
    - How matching uses nearest neighbors for imputation

    Args:
        data: DataFrame with the dataset of interest.
        predictors: List of predictor variables.
        imputed_variables: List of variables to impute.
        quantiles: List of quantiles to predict.
    """
    X_train, X_test = preprocess_data(data)

    # Initialize Matching model
    model = Matching()

    # Fit the model (stores donor data)
    fitted_model = model.fit(X_train, predictors, imputed_variables)

    # Predict for the test data
    # For matching, quantiles don't have the same meaning as in regression
    # The same matched value is used for all quantiles
    test_quantiles: List[float] = [0.5]  # Just one quantile for simplicity
    predictions: Dict[float, pd.DataFrame] = fitted_model.predict(
        X_test, test_quantiles
    )

    # Check structure of predictions
    assert isinstance(predictions, dict)
    assert 0.5 in predictions

    # Check that predictions are pandas DataFrame for matching model
    assert isinstance(predictions[0.5], pd.DataFrame)
