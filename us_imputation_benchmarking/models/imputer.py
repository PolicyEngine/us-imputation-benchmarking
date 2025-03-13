from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Union


class Imputer(ABC):
    """
    Abstract base class for imputation models.
    
    All imputation models should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, random_seed=None) -> None:
        """Initialize the imputer model."""
        self.predictors: Optional[List[str]] = None
        self.imputed_variables: Optional[List[str]] = None
        self.seed = random_seed
    
    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        **kwargs: Any,
    ) -> "Imputer":
        """Fit the model to the training data.
        
        Args:
            X: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            **kwargs: Additional keyword arguments for specific model implementations.
            
        Returns:
            The fitted model instance.
        """
        pass
    
    @abstractmethod
    def predict(
        self, 
        test_X: pd.DataFrame, 
        quantiles: Optional[List[float]] = None
    ) -> Dict[float, Union[np.ndarray, pd.DataFrame]]:
        """Predict imputed values at specified quantiles.
        
        Args:
            test_X: DataFrame containing the test data.
            quantiles: List of quantiles to predict. If None, uses random quantile.
            
        Returns:
            Dictionary mapping quantiles to predicted values.
        """
        pass