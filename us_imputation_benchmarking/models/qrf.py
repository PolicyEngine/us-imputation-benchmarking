from policyengine_us_data.utils import QRF
import numpy as np


def impute_qrf(X, test_X, predictors, imputed_variables, quantiles):
    imputations = {}
    qrf = QRF()
    qrf.fit(X[predictors], X[imputed_variables]) 
    for q in quantiles:
        imputation = qrf.predict(test_X[predictors], mean_quantile = q)
        imputations[q] = imputation
    
    return imputations