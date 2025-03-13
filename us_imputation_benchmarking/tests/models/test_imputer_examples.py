"""
Test module with examples of using different imputer models.

This module provides practical examples of how to use each imputer model,
demonstrating their similarities and differences within the common interface.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Type
from sklearn.datasets import load_iris

from us_imputation_benchmarking.models.imputer import Imputer
from us_imputation_benchmarking.models.ols import OLS
from us_imputation_benchmarking.models.qrf import QRF
from us_imputation_benchmarking.models.quantreg import QuantReg
from us_imputation_benchmarking.models.matching import Matching
from us_imputation_benchmarking.config import RANDOM_STATE


class TestImputerExamples:
    """
    Examples of how to use each imputer model implementation.
    
    This class includes detailed examples for each model implementation,
    showing how to initialize, fit, and predict with each model type.
    """
    
    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """
        Load the Iris dataset for testing.
        
        This function loads the scikit-learn Iris dataset, which is suitable for
        demonstrating different imputation models due to its well-known structure
        with numerical features.
        
        Returns:
            A DataFrame with the Iris dataset.
        """
        # Load the Iris dataset
        iris = load_iris()
        
        # Create DataFrame with feature names
        data = pd.DataFrame(iris.data, columns=iris.feature_names)
        
        # Add target class as a feature for completeness
        data['species'] = iris.target
        
        return data
    
    @pytest.fixture
    def train_test_data(self, sample_data: pd.DataFrame):
        """
        Split data into training and test sets.
        
        Args:
            sample_data: DataFrame with Iris data.
            
        Returns:
            Tuple of (train_data, test_data)
        """
        # Split data into train and test
        train_data = sample_data.sample(frac=0.8, random_state=RANDOM_STATE)
        test_data = sample_data.drop(train_data.index)
        
        return train_data, test_data

    def test_ols_example(self, train_test_data):
        """
        Example of how to use the OLS imputer model.
        
        This example demonstrates:
        - Initializing an OLS model
        - Fitting the model to training data
        - Predicting quantiles on test data
        - How OLS models assume normally distributed residuals
        
        Args:
            train_test_data: Tuple of (train_data, test_data)
        """
        train_data, test_data = train_test_data
        predictors = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']
        imputed_variables = ['petal width (cm)']
        
        # Initialize OLS model
        model = OLS()
        
        # Fit the model
        model.fit(train_data, predictors, imputed_variables)
        
        # Predict at multiple quantiles
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        predictions = model.predict(test_data, quantiles)
        
        # Check structure of predictions
        assert isinstance(predictions, dict)
        assert set(predictions.keys()) == set(quantiles)
        
        # Demonstrate how OLS uses normal distribution assumption
        median_pred = predictions[0.5]
        q10_pred = predictions[0.1]
        q90_pred = predictions[0.9]
        
        # The difference between q90 and median should approximately equal
        # the difference between median and q10 for OLS (symmetric distribution)
        upper_diff = q90_pred - median_pred
        lower_diff = median_pred - q10_pred
        
        # Allow some numerical error
        np.testing.assert_allclose(
            upper_diff.mean(), 
            lower_diff.mean(), 
            rtol=0.1,
            err_msg="OLS should have symmetric quantile predictions around the median"
        )

    def test_quantreg_example(self, train_test_data):
        """
        Example of how to use the Quantile Regression imputer model.
        
        This example demonstrates:
        - Initializing a QuantReg model
        - Fitting the model to specific quantiles
        - Predicting quantiles on test data
        - How QuantReg can capture non-symmetric distributions
        
        Args:
            train_test_data: Tuple of (train_data, test_data)
        """
        train_data, test_data = train_test_data
        predictors = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']
        imputed_variables = ['petal width (cm)']
        
        # Initialize QuantReg model
        model = QuantReg()
        
        # Fit the model to specific quantiles
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        model.fit(train_data, predictors, imputed_variables, quantiles=quantiles)
        
        # Predict at the fitted quantiles
        predictions = model.predict(test_data)
        
        # Check structure of predictions
        assert isinstance(predictions, dict)
        assert set(predictions.keys()) == set(quantiles)
        
        # The advantage of QuantReg is that it can capture asymmetric distributions
        # No assertion here since asymmetry depends on the data

    def test_qrf_example(self, train_test_data):
        """
        Example of how to use the Quantile Random Forest imputer model.
        
        This example demonstrates:
        - Initializing a QRF model
        - Fitting the model with optional hyperparameters
        - Predicting quantiles on test data
        - How QRF can capture complex nonlinear relationships
        
        Args:
            train_test_data: Tuple of (train_data, test_data)
        """
        train_data, test_data = train_test_data
        predictors = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']
        imputed_variables = ['petal width (cm)']
        
        # Initialize QRF model
        model = QRF()
        
        # Fit the model with RF hyperparameters
        model.fit(
            train_data, 
            predictors, 
            imputed_variables,
            n_estimators=100,  # Number of trees
            min_samples_leaf=5  # Min samples in leaf nodes
        )
        
        # Predict at multiple quantiles
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        predictions = model.predict(test_data, quantiles)
        
        # Check structure of predictions
        assert isinstance(predictions, dict)
        assert set(predictions.keys()) == set(quantiles)
        
        # QRF should capture nonlinear relationships
        # We'd need more complex tests to verify this in detail

    def test_matching_example(self, train_test_data):
        """
        Example of how to use the Statistical Matching imputer model.
        
        This example demonstrates:
        - Initializing a Matching model
        - Fitting the model to donor data
        - Predicting values for recipient data
        - How matching uses nearest neighbors for imputation
        
        Args:
            train_test_data: Tuple of (train_data, test_data)
        """
        train_data, test_data = train_test_data
        predictors = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']
        imputed_variables = ['petal width (cm)']
        
        # Initialize Matching model
        model = Matching()
        
        # Fit the model (stores donor data)
        model.fit(train_data, predictors, imputed_variables)
        
        # Predict for the test data
        # For matching, quantiles don't have the same meaning as in regression
        # The same matched value is used for all quantiles
        quantiles = [0.5]  # Just use one quantile for simplicity
        predictions = model.predict(test_data, quantiles)
        
        # Check structure of predictions
        assert isinstance(predictions, dict)
        assert 0.5 in predictions
        
        # Check that predictions are pandas DataFrame for matching model
        assert isinstance(predictions[0.5], pd.DataFrame)

    def test_compare_all_models(self, train_test_data):
        """
        Compare predictions from all model types using the common Imputer interface.
        
        This example demonstrates:
        - How to use all models interchangeably
        - Collecting and comparing predictions from different models
        - Potential visualization of results
        
        Args:
            train_test_data: Tuple of (train_data, test_data)
        """
        train_data, test_data = train_test_data
        predictors = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']
        imputed_variables = ['petal width (cm)']
        quantiles = [0.25, 0.5, 0.75]
        
        # Initialize all models
        models = {
            'OLS': OLS(),
            'QuantReg': QuantReg(),
            'QRF': QRF(),
            'Matching': Matching()
        }
        
        # Fit and predict with all models
        all_predictions = {}
        for name, model in models.items():
            # All models use the same fit/predict interface thanks to the Imputer parent class
            if name == 'QuantReg':
                # QuantReg needs to be fitted with specific quantiles
                model.fit(train_data, predictors, imputed_variables, quantiles=quantiles)
            else:
                model.fit(train_data, predictors, imputed_variables)
            
            predictions = model.predict(test_data, quantiles)
            all_predictions[name] = predictions
        
        # Compare median predictions from each model
        median_predictions = {}
        for name, predictions in all_predictions.items():
            if isinstance(predictions[0.5], pd.DataFrame):
                # For matching model
                median_predictions[name] = predictions[0.5]['petal width (cm)'].values
            else:
                # For regression-based models
                median_predictions[name] = predictions[0.5]
        
        # Check that we have predictions from all models
        assert len(median_predictions) == len(models)
        
        # All predictions should have the same length
        for name, preds in median_predictions.items():
            assert len(preds) == len(test_data)
        
        # Here we could add code to create visualizations
        # But we'll skip that to keep the test simple
        
    def test_generic_imputer_function(self, train_test_data):
        """
        Example of a generic function that works with any Imputer implementation.
        
        This demonstrates how to write code that is agnostic to the specific
        imputer model being used, thanks to the common interface.
        
        Args:
            train_test_data: Tuple of (train_data, test_data)
        """
        train_data, test_data = train_test_data
        
        def fit_and_evaluate_imputer(
            imputer: Imputer,
            train_data: pd.DataFrame,
            test_data: pd.DataFrame,
            predictors: List[str],
            imputed_variables: List[str],
            quantiles: List[float]
        ) -> Dict[str, Any]:
            """
            Generic function that works with any Imputer implementation.
            
            Args:
                imputer: Any instance of an Imputer subclass
                train_data: Training data
                test_data: Test data
                predictors: Predictor column names
                imputed_variables: Target column names
                quantiles: Quantiles to predict
                
            Returns:
                Dictionary with evaluation results
            """
            # Fit the model
            imputer.fit(train_data, predictors, imputed_variables)
            
            # Make predictions
            predictions = imputer.predict(test_data, quantiles)
            
            # Example evaluation: calculate mean absolute error at median
            if isinstance(predictions[0.5], pd.DataFrame):
                # For matching model
                median_preds = predictions[0.5][imputed_variables[0]].values
            else:
                # For regression-based models
                median_preds = predictions[0.5]
                
            actuals = test_data[imputed_variables[0]].values
            mae = np.mean(np.abs(median_preds - actuals))
            
            return {
                'model_type': type(imputer).__name__,
                'predictions': predictions,
                'median_mae': mae
            }
        
        # Use the generic function with different imputer implementations
        predictors = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']
        imputed_variables = ['petal width (cm)']
        quantiles = [0.25, 0.5, 0.75]
        
        # Test with OLS imputer
        ols_results = fit_and_evaluate_imputer(
            OLS(), train_data, test_data, predictors, imputed_variables, quantiles
        )
        
        # Test with QRF imputer
        qrf_results = fit_and_evaluate_imputer(
            QRF(), train_data, test_data, predictors, imputed_variables, quantiles
        )
        
        # Both should return results with the same structure
        assert 'median_mae' in ols_results
        assert 'median_mae' in qrf_results
        
        # This demonstrates how different imputers can be used interchangeably
        # in the same analysis pipeline