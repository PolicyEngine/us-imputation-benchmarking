# Ordinary Least Squares Linear Regression

The `OLS` model uses linear regression to predict missing values based on relationships between predictor and target variables.

## How It Works
- Fits a linear regression model using statsmodels OLS implementation
- Assumes normally distributed residuals to generate different quantiles
- For prediction, computes the mean prediction and adds a quantile-specific offset

## Key Features
- Simple parametric approach with fast training and prediction
- Assumes linear relationships between variables
- Models uncertainty by assuming normal distribution of residuals
- Uses the inverse normal CDF (norm.ppf) to generate quantile-specific predictions
