from typing import List, Optional, Type

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from us_imputation_benchmarking.comparisons.quantile_loss import quantile_loss
from us_imputation_benchmarking.config import QUANTILES, RANDOM_STATE
from us_imputation_benchmarking.models.quantreg import QuantReg
from us_imputation_benchmarking.utils.logging_utils import get_logger

log = get_logger(__name__)


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
        model_class: Model class to evaluate (e.g., QRF, OLS, QuantReg,
                   Matching).
        data: Full dataset to split into training and testing folds.
        predictors: Names of columns to use as predictors.
        imputed_variables: Names of columns to impute.
        quantiles: List of quantiles to evaluate. Defaults to standard
            set if None.
        n_splits: Number of cross-validation folds.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with train and test rows, quantiles as columns, and average
        loss values

    Raises:
        ValueError: If input data is invalid or missing required columns.
        RuntimeError: If cross-validation fails.
    """
    try:
        # Input validation
        if not isinstance(data, pd.DataFrame):
            error_msg = f"data must be a pandas DataFrame, got {type(data).__name__}"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Validate predictor and imputed variable columns exist
        missing_predictors = [col for col in predictors if col not in data.columns]
        if missing_predictors:
            error_msg = f"Missing predictor columns: {missing_predictors}"
            log.error(error_msg)
            raise ValueError(error_msg)

        missing_imputed = [col for col in imputed_variables if col not in data.columns]
        if missing_imputed:
            error_msg = f"Missing imputed variable columns: {missing_imputed}"
            log.error(error_msg)
            raise ValueError(error_msg)

        if quantiles:
            invalid_quantiles = [q for q in quantiles if not 0 <= q <= 1]
            if invalid_quantiles:
                error_msg = (
                    f"Invalid quantiles (must be between 0 and 1): {invalid_quantiles}"
                )
                log.error(error_msg)
                raise ValueError(error_msg)

        # Set up results containers
        test_results = {q: [] for q in quantiles}
        train_results = {q: [] for q in quantiles}
        train_y_values = []
        test_y_values = []

        log.info(
            f"Starting {n_splits}-fold cross-validation for {model_class.__name__}"
        )
        log.info(f"Evaluating at {len(quantiles)} quantiles: {quantiles}")

        # Set up k-fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # Perform cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(data)):
            log.info(f"Processing fold {fold_idx+1}/{n_splits}")

            # Split data for this fold
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]

            # Store actual values for this fold
            train_y = train_data[imputed_variables].values
            test_y = test_data[imputed_variables].values
            train_y_values.append(train_y)
            test_y_values.append(test_y)

            try:
                # Instantiate the model
                log.info(f"Initializing {model_class.__name__} model")
                model = model_class()

                # Handle different model fitting requirements
                if model_class == QuantReg:
                    log.info(f"Fitting QuantReg model with explicit quantiles")
                    model.fit(train_data, predictors, imputed_variables, quantiles)
                else:
                    log.info(f"Fitting {model_class.__name__} model")
                    model.fit(train_data, predictors, imputed_variables)

                # Get predictions for this fold
                log.info(f"Generating predictions for train and test data")
                fold_test_imputations = model.predict(test_data, quantiles)
                fold_train_imputations = model.predict(train_data, quantiles)

                # Store results for each quantile
                for q in quantiles:
                    test_results[q].append(fold_test_imputations[q])
                    train_results[q].append(fold_train_imputations[q])

                log.info(f"Fold {fold_idx+1} completed successfully")

            except Exception as fold_error:
                log.error(f"Error in fold {fold_idx+1}: {str(fold_error)}")
                raise RuntimeError(
                    f"Failed during fold {fold_idx+1}: {str(fold_error)}"
                ) from fold_error

        # Calculate loss metrics
        log.info("Computing loss metrics across all folds")
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

                log.debug(
                    f"Fold {k+1}, q={q}: Train loss = {train_loss.mean():.6f}, Test loss = {test_loss.mean():.6f}"
                )

        # Calculate the average loss across all folds for each quantile
        log.info("Calculating final average metrics")
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

        log.info(f"Cross-validation completed for {model_class.__name__}")
        log.info(f"Average Train Loss: {train_mean:.6f}")
        log.info(f"Average Test Loss: {test_mean:.6f}")
        log.info(f"Train/Test Ratio: {train_test_ratio:.6f}")

        return final_results

    except ValueError as e:
        # Re-raise validation errors directly
        raise e
    except Exception as e:
        log.error(f"Error during cross-validation: {str(e)}")
        raise RuntimeError(f"Cross-validation failed: {str(e)}") from e
