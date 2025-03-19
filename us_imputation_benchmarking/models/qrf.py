from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from us_imputation_benchmarking.config import RANDOM_STATE
from us_imputation_benchmarking.models.imputer import Imputer
from us_imputation_benchmarking.utils import qrf


class QRF(Imputer):
    """
    Quantile Random Forest model for imputation.

    This model uses a Quantile Random Forest to predict quantiles.
    The underlying QRF implementation is from utils.qrf.
    """

    def __init__(self) -> None:
        """Initialize the QRF model.

        The random seed is set through the RANDOM_STATE constant from config.
        """
        super().__init__()
        self.qrf = qrf.QRF(seed=RANDOM_STATE)

    def fit(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        **qrf_kwargs: Any,
    ) -> "QRF":
        """Fit the QRF model to the training data.

        Args:
            X_train: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            **qrf_kwargs: Additional keyword arguments to pass to QRF.

        Returns:
            The fitted model instance.

        Raises:
            ValueError: If input data is invalid or missing required columns.
            RuntimeError: If model fitting fails.
        """
        try:
            # Validate input data
            self._validate_data(X_train, predictors + imputed_variables, "training")

            self.predictors = predictors
            self.imputed_variables = imputed_variables

            self.logger.info(
                f"Fitting QRF model with {len(predictors)} predictors and "
                f"optional parameters: {qrf_kwargs}"
            )

            # Extract training data
            X = X_train[predictors]
            y = X_train[imputed_variables]

            # Fit the QRF model
            self.qrf.fit(X, y, **qrf_kwargs)

            self.logger.info(
                f"QRF model fitted successfully with {len(X)} training samples"
            )
            return self

        except Exception as e:
            self.logger.error(f"Error fitting QRF model: {str(e)}")
            raise RuntimeError(f"Failed to fit QRF model: {str(e)}") from e

    def predict(
        self, X_test: pd.DataFrame, quantiles: Optional[List[float]] = None
    ) -> Dict[float, np.ndarray]:
        """Predict values at specified quantiles using the QRF model.

        Args:
            X_test: DataFrame containing the test data.
            quantiles: List of quantiles to predict.

        Returns:
            Dictionary mapping quantiles to predicted values.

        Raises:
            ValueError: If model is not fitted or input data is invalid.
            RuntimeError: If prediction fails.
        """
        try:
            # Validate that model is fitted
            if self.predictors is None or self.imputed_variables is None:
                error_msg = "Model must be fitted before prediction"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Validate input data
            self._validate_data(X_test, self.predictors, "prediction")

            imputations: Dict[float, np.ndarray] = {}

            if quantiles:
                self.logger.info(
                    f"Predicting at {len(quantiles)} quantiles: {quantiles}"
                )
                for q in quantiles:
                    if not 0 <= q <= 1:
                        error_msg = f"Quantile must be between 0 and 1, got {q}"
                        self.logger.error(error_msg)
                        raise ValueError(error_msg)

                    imputation = self.qrf.predict(
                        X_test[self.predictors], mean_quantile=q
                    )
                    imputations[q] = imputation
            else:
                q = np.random.uniform(0, 1)
                self.logger.info(f"Predicting at random quantile: {q:.4f}")
                imputation = self.qrf.predict(X_test[self.predictors], mean_quantile=q)
                imputations[q] = imputation

            self.logger.info(f"QRF predictions completed for {len(X_test)} samples")
            return imputations

        except ValueError as e:
            # Re-raise validation errors directly
            raise e
        except Exception as e:
            self.logger.error(f"Error during QRF prediction: {str(e)}")
            raise RuntimeError(f"Failed to predict with QRF model: {str(e)}") from e
