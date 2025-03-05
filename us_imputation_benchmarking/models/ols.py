import statsmodels.api as sm
import numpy as np
from scipy.stats import norm

def ols_quantile(m, X, q):
    # m: OLS model.
    # X: X matrix.
    # q: Quantile.
    #
    # Set alpha based on q. Vectorized for different values of q.
    mean_pred = m.predict(X)
    se = np.sqrt(m.scale)
    return mean_pred + norm.ppf(q) * se

def impute_ols(X, test_X, predictors, imputed_variables, quantiles):
    imputations = {}
    Y = X[imputed_variables]    
    # add constant for OLS
    X = sm.add_constant(X[predictors])
    test_X = sm.add_constant(test_X[predictors])
    ols = sm.OLS(Y, X).fit()
    for q in quantiles: 
        imputation = ols_quantile(ols, test_X, q)
        imputations[q] = imputation
    return imputations