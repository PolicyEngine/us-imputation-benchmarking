import statsmodels.api as sm

def impute_quantreg(X, test_X, predictors, imputed_variables, quantiles):
    imputations = {}
    Y = X[imputed_variables]
    # add constant for QuantReg
    X = sm.add_constant(X[predictors])
    test_X = sm.add_constant(test_X[predictors])
    for q in quantiles:
        quantreg = sm.QuantReg(Y, X).fit(q=q)
        imputation = quantreg.predict(test_X)
        imputations[q] = imputation
    return imputations