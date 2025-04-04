"""MicroImpute Package

A package for benchmarking different imputation methods using microdata.
"""

__version__ = "0.1.0"

# Import data handling functions
from us_imputation_benchmarking.comparisons.data import (
    prepare_scf_data,
    preprocess_data,
)
from us_imputation_benchmarking.comparisons.imputations import get_imputations
from us_imputation_benchmarking.comparisons.plot import plot_loss_comparison

# Import comparison utilities
from us_imputation_benchmarking.comparisons.quantile_loss import (
    compare_quantile_loss,
    compute_quantile_loss,
    quantile_loss,
)

# Main configuration
from us_imputation_benchmarking.config import (
    PLOT_CONFIG,
    QUANTILES,
    RANDOM_STATE,
    VALIDATE_CONFIG,
)

# Import evaluation modules
from us_imputation_benchmarking.evaluations.cross_validation import (
    cross_validate_model,
)
from us_imputation_benchmarking.evaluations.train_test_performance import (
    plot_train_test_performance,
)

# Import main models and utilities
from us_imputation_benchmarking.models import (
    OLS,
    QRF,
    Imputer,
    ImputerResults,
    Matching,
    QuantReg,
)
