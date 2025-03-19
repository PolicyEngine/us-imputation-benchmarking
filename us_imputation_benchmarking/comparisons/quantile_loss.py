from typing import Dict, List, Union

import numpy as np
import pandas as pd

from us_imputation_benchmarking.config import QUANTILES
from us_imputation_benchmarking.utils.logging_utils import get_logger

# Create logger for this module
log = get_logger(__name__)


def quantile_loss(q: float, y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Calculate the quantile loss.

    Args:
        q: Quantile to be evaluated, e.g., 0.5 for median.
        y: True value.
        f: Fitted or predicted value.

    Returns:
        Array of quantile losses.
    """
    e = y - f
    return np.maximum(q * e, (q - 1) * e)


def compute_quantile_loss(
    test_y: np.ndarray, imputations: np.ndarray, q: float
) -> np.ndarray:
    """Compute quantile loss for given true values and imputations.

    Args:
        test_y: Array of true values.
        imputations: Array of predicted/imputed values.
        q: Quantile value.

    Returns:
        np.ndarray: Element-wise quantile loss values between true values and predictions.

    Raises:
        ValueError: If q is not between 0 and 1.
        ValueError: If test_y and imputations have different shapes.
    """
    try:
        # Validate quantile value
        if not 0 <= q <= 1:
            error_msg = f"Quantile must be between 0 and 1, got {q}"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Validate input dimensions
        if len(test_y) != len(imputations):
            error_msg = (
                f"Length mismatch: test_y has {len(test_y)} elements, "
                f"imputations has {len(imputations)} elements"
            )
            log.error(error_msg)
            raise ValueError(error_msg)

        log.debug(f"Computing quantile loss for q={q} with {len(test_y)} samples")
        losses = quantile_loss(q, test_y, imputations)
        mean_loss = np.mean(losses)
        log.debug(f"Quantile loss at q={q}: mean={mean_loss:.6f}")

        return losses

    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        log.error(f"Error computing quantile loss: {str(e)}")
        raise RuntimeError(f"Failed to compute quantile loss: {str(e)}") from e


quantiles_legend: List[str] = [str(int(q * 100)) + "th percentile" for q in QUANTILES]


def compare_quantile_loss(
    test_y: pd.DataFrame,
    method_imputations: Dict[str, Dict[float, Union[np.ndarray, pd.DataFrame]]],
) -> pd.DataFrame:
    """Compare quantile loss across different imputation methods.

    Args:
        test_y: DataFrame containing true values.
        method_imputations: Nested dictionary mapping method names
            to dictionaries mapping quantiles to imputation values.

    Returns:
        pd.DataFrame: Results dataframe with columns 'Method',
            'Percentile', and 'Loss' containing the mean quantile
            loss for each method and percentile.

    Raises:
        ValueError: If input data formats are invalid.
        RuntimeError: If comparison operation fails.
    """
    try:
        # Input validation
        if not isinstance(test_y, pd.DataFrame):
            error_msg = (
                f"test_y must be a pandas DataFrame, got {type(test_y).__name__}"
            )
            log.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(method_imputations, dict):
            error_msg = f"method_imputations must be a dictionary, got {type(method_imputations).__name__}"
            log.error(error_msg)
            raise ValueError(error_msg)

        if not method_imputations:
            error_msg = "method_imputations dictionary is empty"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Log comparison operation details
        log.info(
            f"Comparing quantile loss for {len(method_imputations)} methods: {list(method_imputations.keys())}"
        )
        log.info(f"Using {len(QUANTILES)} quantiles: {QUANTILES}")
        log.info(f"True values shape: {test_y.shape}")

        # Initialize empty dataframe with method names, quantile, and loss columns
        results_df: pd.DataFrame = pd.DataFrame(
            columns=["Method", "Percentile", "Loss"]
        )

        # Process each method and quantile
        for method, imputation in method_imputations.items():
            for quantile in QUANTILES:
                log.debug(f"Computing loss for {method} at quantile {quantile}")

                # Validate that the quantile exists in the imputation results
                if quantile not in imputation:
                    error_msg = f"Quantile {quantile} not found in imputations for method {method}"
                    log.error(error_msg)
                    raise ValueError(error_msg)

                # Flatten arrays for computation
                test_values = test_y.values.flatten()
                pred_values = imputation[quantile].values.flatten()

                # Compute loss
                q_loss = compute_quantile_loss(
                    test_values,
                    pred_values,
                    quantile,
                )

                # Create new row and add to results
                new_row = {
                    "Method": method,
                    "Percentile": str(int(quantile * 100)) + "th percentile",
                    "Loss": q_loss.mean(),
                }

                log.debug(
                    f"Mean loss for {method} at q={quantile}: {q_loss.mean():.6f}"
                )

                results_df = pd.concat(
                    [results_df, pd.DataFrame([new_row])], ignore_index=True
                )

        return results_df

    except ValueError as e:
        # Re-raise validation errors
        raise e
    except Exception as e:
        log.error(f"Error in quantile loss comparison: {str(e)}")
        raise RuntimeError(f"Failed to compare quantile loss: {str(e)}") from e
