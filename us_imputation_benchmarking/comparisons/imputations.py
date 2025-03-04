from models import qrf, ols, quantreg, matching, gradient_boosting, random_forests
from sklearn.model_selection import KFold
import numpy as np


def get_imputations(methods, X, test_X, predictors, imputed_variables, data=None) -> dict:

    QUANTILES = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    method_imputations = dict()
    for method in methods:
        method_imputations[method] = dict()

    # Cross-validation option
    if data:
        for method in methods:
            for q in QUANTILES:
                method_imputations[method][q] = []
        K = 5
        kf = KFold(n_splits=K, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(data):
            X, test_X = data.iloc[train_idx], data.iloc[test_idx]
            # QRF
            if "QRF" in methods:
                imputations = qrf.impute_qrf(X, test_X, predictors, imputed_variables, QUANTILES)
                for q in QUANTILES:
                    method_imputations["QRF"][q].append(imputations[q])

            # Matching
            if "Matching" in methods:   
                imputations = matching.impute_matching(X, test_X, predictors, imputed_variables, QUANTILES)
                for q in QUANTILES:
                    method_imputations["Matching"][q].append(imputations[q])

            # OLS
            if "OLS" in methods:
                imputations = ols.impute_ols(X, test_X, predictors, imputed_variables, QUANTILES)
                for q in QUANTILES:
                    method_imputations["OLS"][q].append(imputations[q])

            # QuantReg
            if "QuantReg" in methods:
                imputations = quantreg.impute_quantreg(X, test_X, predictors, imputed_variables, QUANTILES)
                for q in QUANTILES:
                    method_imputations["QuantReg"][q].append(imputations[q])
        for method in methods:
            for q in QUANTILES:
                method_imputations[method][q] = np.mean(method_imputations[method][q])
    
    else:
        # QRF
        if "QRF" in methods:
            imputations = qrf.impute_qrf(X, test_X, predictors, imputed_variables, QUANTILES)
            method_imputations["QRF"] = imputations

        # Matching
        if "Matching" in methods:   
            imputations = matching.impute_matching(X, test_X, predictors, imputed_variables, QUANTILES)
            method_imputations["Matching"] = imputations

        # OLS
        if "OLS" in methods:
            imputations = ols.impute_ols(X, test_X, predictors, imputed_variables, QUANTILES)
            method_imputations["OLS"] = imputations

        # QuantReg
        if "QuantReg" in methods:
            imputations = quantreg.impute_quantreg(X, test_X, predictors, imputed_variables, QUANTILES)
            method_imputations["QuantReg"] = imputations

    return method_imputations