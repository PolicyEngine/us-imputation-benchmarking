"""
Test module for the Imputer abstract class and its model implementations.

This module demonstrates the compatibility and interchangeability of different
imputer models thanks to the common Imputer interface.
"""

from typing import Type

import pandas as pd
import pytest
from sklearn.datasets import load_iris

from us_imputation_benchmarking.comparisons.data import preprocess_data
from us_imputation_benchmarking.config import QUANTILES
from us_imputation_benchmarking.models import *


@pytest.fixture
def iris_data() -> pd.DataFrame:
    """Create a dataset from the Iris dataset for testing.

    Returns:
        A DataFrame with the Iris dataset.
    """
    # Load the Iris dataset
    iris = load_iris()

    # Create DataFrame with feature names
    data = pd.DataFrame(iris.data, columns=iris.feature_names)

    predictors = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)"]
    imputed_variables = ["petal width (cm)"]

    data = data[predictors + imputed_variables]

    return data


# Define all imputer model classes to test
ALL_IMPUTER_MODELS = [OLS, QuantReg, QRF, Matching]


# Parametrize tests to run for each model
@pytest.mark.parametrize(
    "model_class", ALL_IMPUTER_MODELS, ids=lambda cls: cls.__name__
)
def test_init_signatures(model_class: Type[Imputer]) -> None:
    """Test that all models can be initialized without required arguments.

    Args:
        model_class: The model class to test
    """
    # Check that we can initialize the model without errors
    model = model_class()
    assert (
        model.predictors is None
    ), f"{model_class.__name__} should initialize predictors as None"
    assert (
        model.imputed_variables is None
    ), f"{model_class.__name__} should initialize imputed_variables as None"


@pytest.mark.parametrize(
    "model_class", ALL_IMPUTER_MODELS, ids=lambda cls: cls.__name__
)
def test_fit_predict_interface(
    model_class: Type[Imputer], iris_data: pd.DataFrame
) -> None:
    """Test the fit and predict methods for each model.
    Demonstrating models can be interchanged through the Imputer interface.

    Args:
        model_class: The model class to test
        iris_data: DataFrame with sample data
    """
    quantiles = QUANTILES
    predictors = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)"]
    imputed_variables = ["petal width (cm)"]

    X_train, X_test = preprocess_data(iris_data)

    # Test with specified quantiles

    # Initialize the model
    model = model_class()

    # Fit the model
    if model_class.__name__ == "QuantReg":
        # For QuantReg, we need to explicitly fit the quantiles
        fitted_model = model.fit(
            X_train, predictors, imputed_variables, quantiles=quantiles
        )
    else:
        fitted_model = model.fit(X_train, predictors, imputed_variables)

    # Check that the model stored the variable names
    assert model.predictors == predictors
    assert model.imputed_variables == imputed_variables
    assert fitted_model.predictors == predictors
    assert fitted_model.imputed_variables == imputed_variables

    # Predict with explicit quantiles
    predictions = fitted_model.predict(X_test, quantiles)

    # Check prediction format
    assert isinstance(
        predictions, dict
    ), f"{model_class.__name__} predict should return a dictionary"
    assert set(predictions.keys()).issubset(set(quantiles)), (
        f"{model_class.__name__} predict should return keys in the "
        f"specified quantiles"
    )

    # Check prediction shape
    for q, pred in predictions.items():
        assert pred.shape[0] == len(X_test)

    # Test with default quantiles (None)

    # Initialize the model
    model_default_q = model_class()

    # Fit the model
    fitted_default_model = model_default_q.fit(
        X_train, predictors, imputed_variables
    )

    default_predictions = fitted_default_model.predict(X_test)
    assert isinstance(default_predictions, dict), (
        f"{model_class.__name__} predict should return a dictionary even with "
        f"default quantiles"
    )
