# Benchmarking Different Imputation Methods

This documentation describes how the MicroImpute package allows you to compare different imputation methods using quantile loss metrics.

## Overview

The benchmarking functionality enables you to:

1. Compare multiple imputation models on the same dataset
2. Evaluate performance across different quantiles
3. Visualize the results to identify the best-performing methods
4. Make data-driven decisions about which imputation approach to use

## Benchmarking Process

The typical benchmarking workflow involves these steps:

```python
from us_imputation_benchmarking.comparisons import (
    prepare_scf_data, get_imputations, compare_quantile_loss, plot_loss_comparison
)
from us_imputation_benchmarking.models import QRF, OLS, QuantReg, Matching

# 1. Prepare data
X_train, X_test, PREDICTORS, IMPUTED_VARIABLES = prepare_scf_data(years=2019)
Y_test = X_test[IMPUTED_VARIABLES]

# 2. Define models to compare
model_classes = [QRF, OLS, QuantReg, Matching]

# 3. Generate imputations with each model
method_imputations = get_imputations(
    model_classes, X_train, X_test, PREDICTORS, IMPUTED_VARIABLES
)

# 4. Compare performance using quantile loss
loss_comparison_df = compare_quantile_loss(Y_test, method_imputations)

# 5. Visualize the results
plot_loss_comparison(loss_comparison_df, save_path="loss_comparison.png")
```

## Data Preparation

The `prepare_scf_data()` function processes Survey of Consumer Finances data for benchmarking:

- Downloads data from specified survey years
- Selects relevant predictor and target variables
- Normalizes features
- Splits data into training and testing sets

You can use your own data preparation process as long as you provide properly formatted training and testing datasets.

## Imputation Generation

The `get_imputations()` function handles:

- Training each model on the same training data
- Generating predictions at specified quantiles
- Organizing results in a consistent format for comparison

The function returns a nested dictionary structure:

```
{
    "ModelName1": {
        0.1: DataFrame of predictions at 10th percentile,
        0.5: DataFrame of predictions at 50th percentile,
        0.9: DataFrame of predictions at 90th percentile
    },
    "ModelName2": {
        0.1: DataFrame of predictions at 10th percentile,
        ...
    },
    ...
}
```

## Quantile Loss Calculation

The package evaluates imputation quality using quantile loss:

1. `quantile_loss()` function implements the standard quantile loss formulation:
   - L(y, f, q) = max(q * (y - f), (q - 1) * (y - f))
   - This penalizes under-prediction more heavily for higher quantiles and over-prediction more heavily for lower quantiles

2. `compute_quantile_loss()` calculates element-wise losses between true and imputed values

3. `compare_quantile_loss()` evaluates multiple methods across different quantiles, returning a structured DataFrame with columns:
   - Method: The name of the imputation model
   - Percentile: The quantile being evaluated (e.g., "10th percentile")
   - Loss: The average quantile loss value

## Visualization

The `plot_loss_comparison()` function creates bar charts comparing loss across quantiles for different methods:

- Groups results by model and quantile
- Uses color coding to distinguish between models
- Supports saving plots as both static images and interactive HTML files

## Example Visualization

The generated visualization provides a clear comparison of model performance:

- X-axis shows different quantiles (e.g., 10th, 25th, 50th percentiles)
- Y-axis shows the average quantile loss (lower is better)
- Grouped bars allow easy comparison between methods at each quantile

## Extending the Benchmarking Framework

To benchmark your own custom imputation models:

1. Implement your model by extending the `Imputer` abstract base class (refer to the [implement-new-model.ipynb](../models/imputer/implement-new-model.ipynb) file for more details)
2. Include your model class in the `model_classes` list
3. Run the benchmarking process as described above

All models that implement the `Imputer` interface can be seamlessly integrated into the benchmarking framework.

## Best Practices
For effective benchmarking:

- Compare models on multiple datasets to ensure robustness
- Evaluate performance across different quantiles, not just the median
- Consider computational requirements alongside statistical performance
- Use cross-validation for more reliable performance estimates

## Advanced Usage

For more advanced benchmarking scenarios, and validation of individual impuation methods, the package also supports:

- Cross-validation evaluation (`cross_validate_model()`)
- Train-test performance comparisons (`plot_train_test_performance()`)
- Custom quantile sets for targeted evaluation