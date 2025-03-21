import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import List, Dict, Type, Any, Union, Optional, Tuple, Callable
from us_imputation_benchmarking.comparisons.quantile_loss import quantile_loss
from us_imputation_benchmarking.config import QUANTILES, RANDOM_STATE
from us_imputation_benchmarking.models.quantreg import QuantReg


def cross_validate_model(
    model_class: Type,
    data: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    quantiles: Optional[List[float]] = QUANTILES,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Perform cross-validation for an imputation model.

    Args:
        model_class: Model class to evaluate (e.g., QRF, OLS, QuantReg, Matching).
        data: Full dataset to split into training and testing folds.
        predictors: Names of columns to use as predictors.
        imputed_variables: Names of columns to impute.
        quantiles: List of quantiles to evaluate. Defaults to standard set if None.
        n_splits: Number of cross-validation folds.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with train and test rows, quantiles as columns, and average loss values
    """

    test_results = {q: [] for q in quantiles}
    train_results = {q: [] for q in quantiles}
    train_y_values = []
    test_y_values = []

    # Set up k-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, test_idx in kf.split(data):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

        # Store actual values for this fold
        train_y = train_data[imputed_variables].values
        test_y = test_data[imputed_variables].values
        train_y_values.append(train_y)
        test_y_values.append(test_y)

        # Instantiate the model
        model = model_class()

        # Handle different model fitting requirements
        if model_class == QuantReg:
            model.fit(train_data, predictors, imputed_variables, quantiles)
        else:
            model.fit(train_data, predictors, imputed_variables)

        # Get predictions for this fold
        fold_test_imputations = model.predict(test_data, quantiles)
        fold_train_imputations = model.predict(train_data, quantiles)

        for q in quantiles:
            test_results[q].append(fold_test_imputations[q])
            train_results[q].append(fold_train_imputations[q])

    avg_test_losses = {q: [] for q in quantiles}
    avg_train_losses = {q: [] for q in quantiles}
    for k in range(len(test_y_values)):
        for q in quantiles:
            # Flatten arrays for easier calculation
            test_y_flat = test_y_values[k].flatten()
            train_y_flat = train_y_values[k].flatten()
            test_pred_flat = test_results[q][k].values.flatten()
            train_pred_flat = train_results[q][k].values.flatten()

            # Calculate the loss for this fold and quantile
            test_loss = quantile_loss(q, test_y_flat, test_pred_flat)
            train_loss = quantile_loss(q, train_y_flat, train_pred_flat)

            # Store the mean loss
            avg_test_losses[q].append(test_loss.mean())
            avg_train_losses[q].append(train_loss.mean())

    # Calculate the average loss across all folds for each quantile
    final_test_losses = {
        q: np.mean(losses) for q, losses in avg_test_losses.items()
    }
    final_train_losses = {
        q: np.mean(losses) for q, losses in avg_train_losses.items()
    }

    # Create DataFrame with quantiles as columns
    final_results = pd.DataFrame(
        [final_train_losses, final_test_losses], index=["train", "test"]
    )

    # Generate summary statistics
    train_mean = final_results.loc["train"].mean()
    test_mean = final_results.loc["test"].mean()
    train_test_ratio = train_mean / test_mean

    print(f"\nPerformance Summary for {model_class.__name__}:")
    print(f"Average Train Loss: {train_mean:.6f}")
    print(f"Average Test Loss: {test_mean:.6f}")
    print(f"Train/Test Ratio: {train_test_ratio:.6f}")

    return final_results
