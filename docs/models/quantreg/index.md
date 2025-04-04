# Quantile Regression

The `QuantReg` model directly models specific quantiles of the target variable distribution.

## How It Works
- Uses statsmodels' QuantReg implementation
- Fits separate models for each requested quantile
- Minimizes asymmetrically weighted absolute residuals
- During prediction, applies the appropriate model for each requested quantile

## Key Features
- Allows direct modeling of conditional quantiles without distributional assumptions
- Can capture heteroscedasticity (varying variance across the input space)
- Provides a complete picture of the conditional distribution of the target
