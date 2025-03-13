from us_imputation_benchmarking.utils.statmatch_hotdeck import (
    nnd_hotdeck_using_rpy2,
)
import pandas as pd
import numpy as np
import logging
from rpy2.robjects import pandas2ri
from typing import List, Dict, Optional, Callable, Tuple, Any
from us_imputation_benchmarking.models.imputer import Imputer
import random


log = logging.getLogger(__name__)


class Matching(Imputer):
    """
    Statistical matching model for imputation using nearest neighbor distance hot deck method.

    This model uses R's StatMatch package through rpy2 to perform nearest neighbor
    distance hot deck matching for imputation.
    """
    def __init__(self, matching_hotdeck: Callable = nnd_hotdeck_using_rpy2):
        """Initialize the matching model.

        Args:
            matching_hotdeck: Function that performs the hot deck matching.
        """
        super().__init__()
        self.matching_hotdeck = matching_hotdeck
        self.donor_data: Optional[pd.DataFrame] = None

    def fit(
        self,
        X: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
    ) -> "Matching":
        """Fit the matching model by storing the donor data and variable names.

        Args:
            X: DataFrame containing the donor data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.

        Returns:
            The fitted model instance.
        """
        self.donor_data = X.copy()
        self.predictors = predictors
        self.imputed_variables = imputed_variables
        return self

    def predict(
        self, test_X: pd.DataFrame, 
        quantiles: Optional[List[float]] = None
    ) -> Dict[float, pd.DataFrame]:
        """Predict imputed values using the matching model.

        Args:
            test_X: DataFrame containing the recipient data.
            quantiles: List of quantiles to predict.

        Returns:
            Dictionary mapping quantiles to imputed values.
        """
        imputations: Dict[float, pd.DataFrame] = {}
        test_X_copy = test_X.copy()
        test_X_copy.drop(
            self.imputed_variables, axis=1, inplace=True, errors="ignore"
        )

        fused0, fused1 = self.matching_hotdeck(
            receiver=test_X_copy,
            donor=self.donor_data,
            matching_variables=self.predictors,
            z_variables=self.imputed_variables
        )

        fused0_pd = pandas2ri.rpy2py(fused0)

        if quantiles: 
            for q in quantiles:
                imputations[q] = fused0_pd[self.imputed_variables]
        else: 
            q = np.random.uniform(0,1)
            imputations[q] = fused0_pd[self.imputed_variables]

        return imputations
