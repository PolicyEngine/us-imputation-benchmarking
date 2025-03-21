from us_imputation_benchmarking.comparisons.data import preprocess_data
from us_imputation_benchmarking.comparisons.imputations import get_imputations
from us_imputation_benchmarking.comparisons.quantile_loss import (
    compare_quantile_loss,
)
from us_imputation_benchmarking.comparisons.plot import plot_loss_comparison
from us_imputation_benchmarking.models.qrf import QRF
from us_imputation_benchmarking.models.ols import OLS
from us_imputation_benchmarking.models.quantreg import QuantReg
from us_imputation_benchmarking.config import RANDOM_STATE
from us_imputation_benchmarking.models.matching import Matching
from us_imputation_benchmarking.evaluations.cross_validation import (
    cross_validate_model,
)
from us_imputation_benchmarking.evaluations.train_test_performance import (
    plot_train_test_performance,
)


def test_quantile_comparison():
    X, X_test, PREDICTORS, IMPUTED_VARIABLES = preprocess_data(
        full_data=False, years=2019
    )
    # Shrink down the data by sampling
    X = X.sample(frac=0.01, random_state=RANDOM_STATE)
    X_test = X_test.sample(frac=0.01, random_state=RANDOM_STATE)

    Y_test = X_test[IMPUTED_VARIABLES]
    data, PREDICTORS, IMPUTED_VARIABLES = preprocess_data(full_data=True)
    data = data.sample(frac=0.01, random_state=RANDOM_STATE)

    model_classes = [QRF, OLS, QuantReg, Matching]
    method_imputations = get_imputations(
        model_classes, X, X_test, PREDICTORS, IMPUTED_VARIABLES
    )

    loss_comparison_df = compare_quantile_loss(Y_test, method_imputations)

    assert not loss_comparison_df.isna().any().any()

    plot_loss_comparison(loss_comparison_df, save_path="loss_comparison.png")

    qrf_results = cross_validate_model(
        QRF, data, PREDICTORS, IMPUTED_VARIABLES
    )
    qrf_results.to_csv("qrf_results.csv")

    assert not qrf_results.isna().any().any()

    #plot_train_test_performance(qrf_results, save_path="train_test_performance.png")
