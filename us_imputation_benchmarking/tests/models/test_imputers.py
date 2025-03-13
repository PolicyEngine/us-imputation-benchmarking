"""
Test module for the Imputer abstract class and its model implementations.

This module demonstrates the compatibility and interchangeability of different
imputer models thanks to the common Imputer interface.
"""

import pytest
import pandas as pd
import numpy as np
from typing import List, Type, Dict, Any
from sklearn.datasets import load_iris

from us_imputation_benchmarking.models.imputer import Imputer
from us_imputation_benchmarking.models.ols import OLS
from us_imputation_benchmarking.models.qrf import QRF
from us_imputation_benchmarking.models.quantreg import QuantReg
from us_imputation_benchmarking.models.matching import Matching
from us_imputation_benchmarking.config import RANDOM_STATE


class TestImputerInterface:
    """
    Test class for verifying compatibility with the Imputer abstract base class.
    
    These tests verify that all model implementations correctly adhere to the Imputer
    interface and can be used interchangeably.
    """
    
    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create a dataset from the Iris dataset for testing.
        
        Returns:
            A DataFrame with the Iris dataset.
        """
        # Load the Iris dataset
        iris = load_iris()
        
        # Create DataFrame with feature names
        data = pd.DataFrame(iris.data, columns=iris.feature_names)
        
        # Add target variable (sepal length) as the variable to impute
        data['target'] = iris.target
        
        return data
    
    @pytest.fixture
    def train_test_data(self, sample_data: pd.DataFrame):
        """Split data into training and test sets.
        
        Args:
            sample_data: DataFrame with Iris data.
            
        Returns:
            Tuple of (train_data, test_data)
        """
        # Split data into train and test
        train_data = sample_data.sample(frac=0.8, random_state=RANDOM_STATE)
        test_data = sample_data.drop(train_data.index)
        
        return train_data, test_data
    
    @pytest.fixture
    def model_classes(self) -> List[Type[Imputer]]:
        """Get list of all imputer model classes.
        
        Returns:
            List of Imputer subclasses
        """
        return [OLS, QuantReg, QRF, Matching]
    
    def test_inheritance(self, model_classes: List[Type[Imputer]]):
        """Test that all model classes inherit from Imputer.
        
        Args:
            model_classes: List of model classes to test
        """
        for model_class in model_classes:
            # Check that the class is a subclass of Imputer
            assert issubclass(model_class, Imputer), f"{model_class.__name__} should inherit from Imputer"
    
    def test_init_signatures(self, model_classes: List[Type[Imputer]]):
        """Test that all models can be initialized without required arguments.
        
        Args:
            model_classes: List of model classes to test
        """
        for model_class in model_classes:
            # Check that we can initialize the model without errors
            model = model_class()
            assert model.predictors is None, f"{model_class.__name__} should initialize predictors as None"
            assert model.imputed_variables is None, f"{model_class.__name__} should initialize imputed_variables as None"
    
    def test_fit_predict_interface(self, model_classes: List[Type[Imputer]], train_test_data):
        """Test the fit and predict methods of all models to ensure they follow the interface.
        
        Args:
            model_classes: List of model classes to test
            train_test_data: Tuple of (train_data, test_data)
        """
        train_data, test_data = train_test_data
        predictors = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']
        imputed_variables = ['petal width (cm)']
        quantiles = [0.25, 0.5, 0.75]
        
        for model_class in model_classes:
            # Initialize the model
            model = model_class()
            
            # Fit the model
            if model_class.__name__ == 'QuantReg':
                # For QuantReg, we need to explicitly fit the quantiles
                model.fit(train_data, predictors, imputed_variables, quantiles=quantiles)
            else:
                model.fit(train_data, predictors, imputed_variables)
            
            # Check that the model stored the variable names
            assert model.predictors == predictors
            assert model.imputed_variables == imputed_variables
            
            # Predict with explicit quantiles
            predictions = model.predict(test_data, quantiles)
            
            # Check prediction format
            assert isinstance(predictions, dict), f"{model_class.__name__} predict should return a dictionary"
            assert set(predictions.keys()).issubset(set(quantiles)), f"{model_class.__name__} predict should return keys in the specified quantiles"
            
            # Check prediction shape
            for q, pred in predictions.items():
                if isinstance(pred, np.ndarray):
                    assert len(pred) == len(test_data)
                elif isinstance(pred, pd.DataFrame):
                    assert pred.shape[0] == len(test_data)
            
            # Test with default quantiles (None)
            default_predictions = model.predict(test_data)
            assert isinstance(default_predictions, dict), f"{model_class.__name__} predict should return a dictionary even with default quantiles"
            
    def test_model_interchangeability(self, model_classes: List[Type[Imputer]], train_test_data):
        """Test that models can be used interchangeably through the Imputer interface.
        
        Args:
            model_classes: List of model classes to test
            train_test_data: Tuple of (train_data, test_data)
        """
        train_data, test_data = train_test_data
        predictors = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']
        imputed_variables = ['petal width (cm)']
        quantiles = [0.5]  # Just use median for simplicity
        
        def run_imputation_workflow(imputer: Imputer, fit_quantiles: bool = False) -> Dict[float, Any]:
            """Generic function that works with any Imputer implementation.
            
            Args:
                imputer: An instance of an Imputer subclass
                fit_quantiles: Whether to explicitly fit quantiles for QuantReg
                
            Returns:
                Dictionary of predictions
            """
            # Fit the model
            if fit_quantiles:
                imputer.fit(train_data, predictors, imputed_variables, quantiles=quantiles)
            else:
                imputer.fit(train_data, predictors, imputed_variables)
            
            # Make predictions
            return imputer.predict(test_data, quantiles)
        
        # Run the same workflow with different model implementations
        for model_class in model_classes:
            model = model_class()
            # For QuantReg, we need to explicitly fit the quantiles
            if model_class.__name__ == 'QuantReg':
                predictions = run_imputation_workflow(model, fit_quantiles=True)
            else:
                predictions = run_imputation_workflow(model)
            
            # Basic validation of predictions
            assert len(predictions) > 0, f"{model_class.__name__} should return at least one prediction"
            
            # For QuantReg, check that we have the specific quantile
            if model_class.__name__ == 'QuantReg':
                assert 0.5 in predictions
                
            # Check that prediction values are not None
            for q, pred in predictions.items():
                assert pred is not None, f"{model_class.__name__} predictions should not be None"
            
    def test_method_docstrings(self, model_classes: List[Type[Imputer]]):
        """Test that all models have proper docstrings for their methods.
        
        Args:
            model_classes: List of model classes to test
        """
        for model_class in model_classes:
            # Check class docstring
            assert model_class.__doc__, f"{model_class.__name__} is missing a class docstring"
            
            # Check method docstrings
            assert model_class.fit.__doc__, f"{model_class.__name__}.fit is missing a docstring"
            assert model_class.predict.__doc__, f"{model_class.__name__}.predict is missing a docstring"
            
            # Basic content check for docstrings
            assert "fit" in model_class.fit.__doc__.lower()
            assert "predict" in model_class.predict.__doc__.lower()