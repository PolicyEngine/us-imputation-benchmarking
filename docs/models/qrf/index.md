# Quantile Regression Forests

The `QRF` model uses an ensemble of decision trees to predict different quantiles of the target variable distribution.

## How It Works
- Implements a Quantile Random Forest algorithm from the utils.qrf module
- Builds an ensemble of decision trees on bootstrapped data samples
- Each tree predicts the target variable using a subset of features
- Quantiles are estimated from the distribution of predictions across trees

## Key Features
- Non-parametric approach that can capture complex non-linear relationships
- Provides uncertainty estimates through prediction intervals
- Can handle high-dimensional data and interactions between variables
- Can capture heteroscedasticity (varying variance across the input space)
- Typically more accurate than linear models for complex relationships
