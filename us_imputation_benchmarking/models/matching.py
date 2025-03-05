import pandas as pd
import numpy as np
import logging
import os
import pkg_resources
import rpy2
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
# Enable R-Python DataFrame and array conversion
pandas2ri.activate()
numpy2ri.activate()
utils = importr('utils')
utils.chooseCRANmirror(ind=1)
StatMatch = importr("StatMatch")

log = logging.getLogger(__name__)

"""
data.rec: A matrix or data frame that plays the role of recipient in the statistical matching application.

data.don: A matrix or data frame that that plays the role of donor in the statistical matching application.

mtc.ids: A matrix with two columns. Each row must contain the name or the index of the recipient record (row) 
          in data.don and the name or the index of the corresponding donor record (row) in data.don. Note that 
          this type of matrix is returned by the functions NND.hotdeck, RANDwNND.hotdeck, rankNND.hotdeck, and 
          mixed.mtc.

z.vars: A character vector with the names of the variables available only in data.don that should be “donated” to data.rec.
networth

dup.x: Logical. When TRUE the values of the matching variables in data.don are also “donated” to data.rec. 
        The names of the matching variables have to be specified with the argument match.vars. To avoid confusion, 
        the matching variables added to data.rec are renamed by adding the suffix “don”. By default dup.x=FALSE.

match.vars: A character vector with the names of the matching variables. It has to be specified only when dup.x=TRUE. 
All other vars (no need to specify)     
"""

def nnd_hotdeck_using_rpy2(receiver = None, donor = None, matching_variables = None,
        z_variables = None, donor_classes = None):
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri

    assert receiver is not None and donor is not None
    assert matching_variables is not None

    pandas2ri.activate()
    StatMatch = importr("StatMatch")

    if isinstance(donor_classes, str):
        assert donor_classes in receiver, 'Donor class not present in receiver'
        assert donor_classes in donor, 'Donor class not present in donor'

    try:
        if donor_classes:
            out_NND = StatMatch.NND_hotdeck(
                data_rec = receiver,
                data_don = donor,
                match_vars = pd.Series(matching_variables),
                don_class = pd.Series(donor_classes)
                )
        else:
            out_NND = StatMatch.NND_hotdeck(
                data_rec = receiver,
                data_don = donor,
                match_vars = pd.Series(matching_variables),
                # don_class = pd.Series(donor_classes)
                )
            
    except Exception as e:
        print(1)
        print(receiver)
        print(2)
        print(donor)
        print(3)
        print(pd.Series(matching_variables))
        print(e)

    # Convert out_NND[0] to a NumPy array and reshape to 2 columns
    # To address shape errors when using hotdeck for mtc_ids
    mtc_ids_array = np.array(out_NND[0])
    mtc_ids_2d = mtc_ids_array.reshape(-1, 2)
    # Convert the 2D NumPy array to an R matrix using py2rpy
    mtc_ids = ro.conversion.py2rpy(mtc_ids_2d)

    print("mtc_ids:", mtc_ids)
    print("mtc_ids type:", type(mtc_ids))

    # create synthetic data.set, without the
    # duplication of the matching variables

    fused_0 = StatMatch.create_fused(
            data_rec = receiver,
            data_don = donor,
            mtc_ids = mtc_ids,
            z_vars = pd.Series(z_variables)
            )

    # create synthetic data.set, with the "duplication"
    # of the matching variables

    fused_1 = StatMatch.create_fused(
            data_rec = receiver,
            data_don = donor,
            mtc_ids = mtc_ids,
            z_vars = pd.Series(z_variables),
            dup_x = False,
            match_vars = pd.Series(matching_variables)
            )

    return fused_0, fused_1

def impute_matching(X, test_X, predictors, imputed_variables, quantiles, 
                    matching_hotdeck = nnd_hotdeck_using_rpy2):
    imputations = {}
    test_X_dup = test_X.copy()
    test_X.drop(imputed_variables, axis=1, inplace=True, errors='ignore') # in case the variable is already missing 

    fused0, fused1 = matching_hotdeck(receiver = test_X, 
                           donor = X, 
                           matching_variables = predictors,
                           z_variables = imputed_variables, 
                           donor_classes = None)

    print(fused0.colnames)
    fused0_pd = pandas2ri.rpy2py(fused0)
    print(fused0_pd[imputed_variables])
    
    for q in quantiles:
        imputations[q] = fused0_pd[imputed_variables]

    return imputations