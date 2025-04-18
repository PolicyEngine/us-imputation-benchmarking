"""Quantile Random Forest imputation model."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import validate_call

from microimpute.config import RANDOM_STATE, VALIDATE_CONFIG
from microimpute.models.imputer import Imputer, ImputerResults
from microimpute.utils import qrf


class QRFResults(ImputerResults):
    """
    Fitted QRF instance ready for imputation.
    """

    def __init__(
        self,
        models: Dict[str, "QRF"],
        predictors: List[str],
        imputed_variables: List[str],
    ) -> None:
        """Initialize the QRF results.

        Args:
            model: Fitted QRF model.
            predictors: List of column names used as predictors.
            imputed_variables: List of column names to be imputed.
        """
        super().__init__(predictors, imputed_variables)
        self.models = models

    @validate_call(config=VALIDATE_CONFIG)
    def _predict(
        self, X_test: pd.DataFrame, quantiles: Optional[List[float]] = None
    ) -> Dict[float, pd.DataFrame]:
        """Predict values at specified quantiles using the QRF model.

        Args:
            X_test: DataFrame containing the test data.
            quantiles: List of quantiles to predict.

        Returns:
            Dictionary mapping quantiles to predicted values.

        Raises:
            RuntimeError: If prediction fails.
        """
        try:
            # Create output dictionary with results
            imputations: Dict[float, pd.DataFrame] = {}

            if quantiles:
                self.logger.info(
                    f"Predicting at {len(quantiles)} quantiles: {quantiles}"
                )
                for q in quantiles:
                    imputed_df = pd.DataFrame()
                    for variable in self.imputed_variables:
                        model = self.models[variable]
                        imputed_df[variable] = model.predict(
                            X_test[self.predictors], mean_quantile=q
                        )
                    imputations[q] = imputed_df
            else:
                q = np.random.uniform(0, 1)
                self.logger.info(f"Predicting at random quantile: {q:.4f}")
                imputed_df = pd.DataFrame()
                for variable in self.imputed_variables:
                    model = self.models[variable]
                    imputed_df = model.predict(
                        X_test[self.predictors], mean_quantile=q
                    )
                imputations[q] = imputed_df

            self.logger.info(
                f"QRF predictions completed for {len(X_test)} samples"
            )
            return imputations

        except Exception as e:
            self.logger.error(f"Error during QRF prediction: {str(e)}")
            raise RuntimeError(
                f"Failed to predict with QRF model: {str(e)}"
            ) from e


class QRF(Imputer):
    """
    Quantile Random Forest model for imputation.

    This model uses a Quantile Random Forest to predict quantiles.
    The underlying QRF implementation is from utils.qrf.
    """

    def __init__(self, random_seed: int = RANDOM_STATE) -> None:
        """Initialize the QRF model.

        The random seed is set through the RANDOM_STATE constant from config.

        Args:
            random_seed: Seed for replicability.
        """
        super().__init__()
        self.seed = random_seed
        self.models = {}
        self.logger.debug("Initializing QRF imputer")

    @validate_call(config=VALIDATE_CONFIG)
    def _fit(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        **qrf_kwargs: Any,
    ) -> QRFResults:
        """Fit the QRF model to the training data.

        Args:
            X_train: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            **qrf_kwargs: Additional keyword arguments to pass to QRF.

        Returns:
            The fitted model instance.

        Raises:
            RuntimeError: If model fitting fails.
        """
        try:
            self.logger.info(
                f"Fitting QRF model with {len(predictors)} predictors and "
                f"optional parameters: {qrf_kwargs}"
            )

            # Extract training data
            X = X_train[predictors]
            # Initialize and fit a QRF model for each variable
            for variable in imputed_variables:
                model = qrf.QRF(seed=self.seed)
                y = pd.DataFrame(X_train[variable])
                # Fit the QRF model
                model.fit(X, y, **qrf_kwargs)

                self.logger.info(
                    f"QRF model fitted successfully with {len(X)} training samples"
                )

                self.models[variable] = model
            return QRFResults(
                models=self.models,
                predictors=predictors,
                imputed_variables=imputed_variables,
            )
        except Exception as e:
            self.logger.error(f"Error fitting QRF model: {str(e)}")
            raise RuntimeError(f"Failed to fit QRF model: {str(e)}") from e
