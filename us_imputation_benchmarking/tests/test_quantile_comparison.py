from us_imputation_benchmarking.comparisons.data import prepare_scf_data
from us_imputation_benchmarking.comparisons.imputations import get_imputations
from us_imputation_benchmarking.comparisons.plot import plot_loss_comparison
from us_imputation_benchmarking.comparisons.quantile_loss import \
    compare_quantile_loss
from us_imputation_benchmarking.config import RANDOM_STATE
from us_imputation_benchmarking.models.matching import Matching
from us_imputation_benchmarking.models.ols import OLS
from us_imputation_benchmarking.models.qrf import QRF
from us_imputation_benchmarking.models.quantreg import QuantReg


def test_quantile_comparison():
    X_train, X_test, PREDICTORS, IMPUTED_VARIABLES = prepare_scf_data(
        full_data=False, years=2019
    )
    # Shrink down the data by sampling
    X_train = X_train.sample(frac=0.01, random_state=RANDOM_STATE)
    X_test = X_test.sample(frac=0.01, random_state=RANDOM_STATE)

    Y_test = X_test[IMPUTED_VARIABLES]

    model_classes = [QRF, OLS, QuantReg, Matching]
    method_imputations = get_imputations(
        model_classes, X_train, X_test, PREDICTORS, IMPUTED_VARIABLES
    )

    loss_comparison_df = compare_quantile_loss(Y_test, method_imputations)

    assert not loss_comparison_df.isna().any().any()

    # plot_loss_comparison(loss_comparison_df, save_path="loss_comparison.png")
