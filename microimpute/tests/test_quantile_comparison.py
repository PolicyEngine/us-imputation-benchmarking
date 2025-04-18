"""Tests for the end-to-end quantile loss comparison workflow.

This module tests the complete workflow of:
1. Preparing data
2. Training different imputation models
3. Generating predictions
4. Comparing models using quantile loss metrics
5. Visualizing the results
"""

from typing import List, Type

import pandas as pd
from sklearn.datasets import load_diabetes

from microimpute.comparisons import *
from microimpute.config import RANDOM_STATE
from microimpute.models import *


def test_quantile_comparison_diabetes() -> None:
    """Test the end-to-end quantile loss comparison workflow."""
    diabetes_data = load_diabetes()
    diabetes_df = pd.DataFrame(
        diabetes_data.data, columns=diabetes_data.feature_names
    )

    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1", "s4"]

    diabetes_df = diabetes_df[predictors + imputed_variables]
    X_train, X_test = preprocess_data(diabetes_df)

    Y_test: pd.DataFrame = X_test[imputed_variables]

    model_classes: List[Type[Imputer]] = [QRF, OLS, QuantReg, Matching]
    method_imputations = get_imputations(
        model_classes, X_train, X_test, predictors, imputed_variables
    )

    loss_comparison_df = compare_quantile_loss(
        Y_test, method_imputations, imputed_variables
    )

    assert not loss_comparison_df.isna().any().any()

    loss_comparison_df.to_csv("diabetes_comparison_results.csv")

    plot_loss_comparison(
        loss_comparison_df, save_path="diabetes_loss_comparison.jpg"
    )


def test_quantile_comparison_cps() -> None:
    """Test the end-to-end quantile loss comparison workflow on the cps data set."""
    X_train, X_test, PREDICTORS, IMPUTED_VARIABLES = prepare_scf_data(
        full_data=False, years=2019
    )
    # Shrink down the data by sampling
    X_train = X_train.sample(frac=0.01, random_state=RANDOM_STATE)
    X_test = X_test.sample(frac=0.01, random_state=RANDOM_STATE)

    Y_test: pd.DataFrame = X_test[IMPUTED_VARIABLES]

    model_classes: List[Type[Imputer]] = [QRF, OLS, QuantReg, Matching]
    method_imputations = get_imputations(
        model_classes, X_train, X_test, PREDICTORS, IMPUTED_VARIABLES
    )

    loss_comparison_df = compare_quantile_loss(
        Y_test, method_imputations, IMPUTED_VARIABLES
    )

    assert not loss_comparison_df.isna().any().any()

    loss_comparison_df.to_csv("cps_comparison_results.csv")

    plot_loss_comparison(
        loss_comparison_df, save_path="cps_loss_comparison.jpg"
    )
