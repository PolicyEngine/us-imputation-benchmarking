"""Statistical matching imputation model using hot deck methods."""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import validate_call
from rpy2.robjects import pandas2ri

from microimpute.config import VALIDATE_CONFIG
from microimpute.models.imputer import Imputer, ImputerResults
from microimpute.utils.statmatch_hotdeck import nnd_hotdeck_using_rpy2

MatchingHotdeckFn = Callable[
    [
        Optional[pd.DataFrame],
        Optional[pd.DataFrame],
        Optional[List[str]],
        Optional[List[str]],
    ],
    Tuple[pd.DataFrame, pd.DataFrame],
]


class MatchingResults(ImputerResults):
    """
    Fitted Matching instance ready for imputation.
    """

    def __init__(
        self,
        matching_hotdeck: MatchingHotdeckFn,
        donor_data: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
    ) -> None:
        """Initialize the matching model.

        Args:
            matching_hotdeck: Function that performs the hot deck matching.
            donor_data: DataFrame containing the donor data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
        """
        super().__init__(predictors, imputed_variables)
        self.matching_hotdeck = matching_hotdeck
        self.donor_data = donor_data

    @validate_call(config=VALIDATE_CONFIG)
    def _predict(
        self, X_test: pd.DataFrame, quantiles: Optional[List[float]] = None
    ) -> Dict[float, pd.DataFrame]:
        """Predict imputed values using the matching model.

        Args:
            X_test: DataFrame containing the recipient data.
            quantiles: List of quantiles to predict.

        Returns:
            Dictionary mapping quantiles to imputed values.

        Raises:
            ValueError: If model is not properly set up or
                input data is invalid.
            RuntimeError: If matching or prediction fails.
        """
        try:
            self.logger.info(
                f"Performing matching for {len(X_test)} recipient records"
            )

            # Create a copy to avoid modifying the input
            try:
                self.logger.debug("Creating copy of test data")
                X_test_copy = X_test.copy()

                # Drop imputed variables if they exist in test data
                if any(
                    col in X_test.columns for col in self.imputed_variables
                ):
                    self.logger.debug(
                        f"Dropping imputed variables from test data: {self.imputed_variables}"
                    )
                    X_test_copy.drop(
                        self.imputed_variables,
                        axis=1,
                        inplace=True,
                        errors="ignore",
                    )
            except Exception as copy_error:
                self.logger.error(
                    f"Error preparing test data: {str(copy_error)}"
                )
                raise RuntimeError(
                    "Failed to prepare test data for matching"
                ) from copy_error

            # Perform the matching
            try:
                self.logger.info("Calling R-based hot deck matching function")
                fused0, fused1 = self.matching_hotdeck(
                    receiver=X_test_copy,
                    donor=self.donor_data,
                    matching_variables=self.predictors,
                    z_variables=self.imputed_variables,
                )
            except Exception as matching_error:
                self.logger.error(
                    f"Error in hot deck matching: {str(matching_error)}"
                )
                raise RuntimeError(
                    "Hot deck matching failed"
                ) from matching_error

            # Convert R objects to pandas DataFrame
            try:
                self.logger.debug("Converting R result to pandas DataFrame")
                fused0_pd = pandas2ri.rpy2py(fused0)

                # Verify imputed variables exist in the result
                missing_imputed = [
                    var
                    for var in self.imputed_variables
                    if var not in fused0_pd.columns
                ]
                if missing_imputed:
                    self.logger.error(
                        f"Imputed variables missing from matching result: {missing_imputed}"
                    )
                    raise ValueError(
                        f"Matching failed to produce these variables: {missing_imputed}"
                    )

                self.logger.info(
                    f"Matching completed, fused dataset has {len(fused0_pd)} records"
                )
            except Exception as convert_error:
                self.logger.error(
                    f"Error converting matching results: {str(convert_error)}"
                )
                raise RuntimeError(
                    "Failed to process matching results"
                ) from convert_error

            # Create output dictionary with results
            imputations: Dict[float, pd.DataFrame] = {}

            try:
                if quantiles:
                    self.logger.info(
                        f"Creating imputations for {len(quantiles)} quantiles"
                    )
                    # For each quantile, return a DataFrame with all imputed variables
                    for q in quantiles:
                        imputed_df = pd.DataFrame()
                        for variable in self.imputed_variables:
                            self.logger.debug(
                                f"Adding result for imputed variable {variable} at quantile {q}"
                            )
                            imputed_df[variable] = fused0_pd[variable]
                        imputations[q] = imputed_df
                else:
                    # If no quantiles specified, use a default one
                    q = 0.5
                    self.logger.info(
                        f"Creating imputation for default quantile {q}"
                    )
                    imputed_df = pd.DataFrame()
                    for variable in self.imputed_variables:
                        imputed_df[variable] = fused0_pd[variable]
                    imputations[q] = imputed_df

                # Verify output shapes
                for q, df in imputations.items():
                    self.logger.debug(
                        f"Imputation result for q={q}: shape={df.shape}"
                    )
                    if len(df) != len(X_test):
                        self.logger.warning(
                            f"Result shape mismatch: expected {len(X_test)} rows, got {len(df)}"
                        )

                return imputations
            except Exception as output_error:
                self.logger.error(
                    f"Error creating output imputations: {str(output_error)}"
                )
                raise RuntimeError(
                    "Failed to create output imputations"
                ) from output_error

        except ValueError as e:
            # Re-raise validation errors directly
            raise e
        except Exception as e:
            self.logger.error(f"Error during matching prediction: {str(e)}")
            raise RuntimeError(f"Failed to perform matching: {str(e)}") from e


class Matching(Imputer):
    """
    Statistical matching model for imputation using nearest neighbor distance
    hot deck method.

    This model uses R's StatMatch package through rpy2 to perform nearest
    neighbor distance hot deck matching for imputation.
    """

    def __init__(
        self, matching_hotdeck: MatchingHotdeckFn = nnd_hotdeck_using_rpy2
    ) -> None:
        """Initialize the matching model.

        Args:
            matching_hotdeck: Function that performs the hot deck matching.

        Raises:
            ValueError: If matching_hotdeck is not callable
        """
        super().__init__()
        self.logger.debug("Initializing Matching imputer")

        # Validate input
        if not callable(matching_hotdeck):
            self.logger.error("matching_hotdeck must be a callable function")
            raise ValueError("matching_hotdeck must be a callable function")

        self.matching_hotdeck = matching_hotdeck
        self.donor_data: Optional[pd.DataFrame] = None

    @validate_call(config=VALIDATE_CONFIG)
    def _fit(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
    ) -> MatchingResults:
        """Fit the matching model by storing the donor data and variable names.

        Args:
            X_train: DataFrame containing the donor data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.

        Returns:
            The fitted model instance.

        Raises:
            ValueError: If matching cannot be set up.
        """
        try:
            self.donor_data = X_train.copy()

            self.logger.info(
                f"Matching model ready with {len(X_train)} donor records"
            )
            self.logger.info(f"Using predictors: {predictors}")
            self.logger.info(
                f"Targeting imputed variables: {imputed_variables}"
            )

            return MatchingResults(
                matching_hotdeck=self.matching_hotdeck,
                donor_data=self.donor_data,
                predictors=predictors,
                imputed_variables=imputed_variables,
            )
        except Exception as e:
            self.logger.error(f"Error setting up matching model: {str(e)}")
            raise ValueError(
                f"Failed to set up matching model: {str(e)}"
            ) from e
