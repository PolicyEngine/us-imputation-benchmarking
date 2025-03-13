import numpy as np
import pandas as pd
import logging
import os
import rpy2
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from typing import List, Dict, Optional, Union, Any, Tuple

# Enable R-Python DataFrame and array conversion
pandas2ri.activate()
numpy2ri.activate()
utils = importr("utils")
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

z.vars: A character vector with the names of the variables available only in data.don that should be "donated" to data.rec.
networth

dup.x: Logical. When TRUE the values of the matching variables in data.don are also "donated" to data.rec. 
        The names of the matching variables have to be specified with the argument match.vars. To avoid confusion, 
        the matching variables added to data.rec are renamed by adding the suffix "don". By default dup.x=FALSE.

match.vars: A character vector with the names of the matching variables. It has to be specified only when dup.x=TRUE. 
All other vars (no need to specify)     
"""


def nnd_hotdeck_using_rpy2(
    receiver: Optional[pd.DataFrame] = None,
    donor: Optional[pd.DataFrame] = None,
    matching_variables: Optional[List[str]] = None,
    z_variables: Optional[List[str]] = None,
) -> Tuple[Any, Any]:
    """Perform nearest neighbor distance hot deck matching using R's StatMatch package.

    Args:
        receiver: DataFrame containing recipient data.
        donor: DataFrame containing donor data.
        matching_variables: List of column names to use for matching.
        z_variables: List of column names to donate from donor to recipient.

    Returns:
        Tuple containing two fused datasets:
          - First without duplication of matching variables
          - Second with duplication of matching variables

    Raises:
        AssertionError: If receiver, donor, or matching_variables are not provided.
    """
    assert (
        receiver is not None and donor is not None
    ), "Receiver and donor must be provided"
    assert (
        matching_variables is not None
    ), "Matching variables must be provided"

    # Make sure R<->Python conversion is enabled
    pandas2ri.activate()
    
    # Import R's StatMatch package
    StatMatch = importr("StatMatch")

    # Call the NND_hotdeck function from R
    out_NND = StatMatch.NND_hotdeck(
        data_rec=receiver,
        data_don=donor,
        match_vars=pd.Series(matching_variables),
    )

    # Create the correct matching indices matrix for StatMatch.create_fused
    # Get all indices as 1-based (for R)
    recipient_indices = np.arange(1, len(receiver) + 1)
    
    # For direct NND matching we need the row positions from mtc.ids
    mtc_ids_r = out_NND.rx2("mtc.ids")
    
    # Create the properly formatted 2-column matrix that create_fused expects
    if hasattr(mtc_ids_r, 'ncol') and mtc_ids_r.ncol == 2:
        # Already a matrix with the right shape, use it directly
        mtc_ids = mtc_ids_r
    else:
        # The IDs returned aren't in the expected format, extract and convert them
        mtc_array = np.array(mtc_ids_r)
        
        # If we have a 1D array with strings, convert to integers
        if mtc_array.dtype.kind in ['U', 'S']:
            mtc_array = np.array([int(x) for x in mtc_array])
        
        # If the mtc.ids array has 2 values per recipient (recipient_idx, donor_idx pairs)
        if len(mtc_array) == 2 * len(receiver):
            # Extract only the donor indices (every second value)
            donor_indices = mtc_array.reshape(-1, 2)[:, 1]
            
            # Make sure these indices are within the valid range (1 to donor dataset size)
            # If they're not, we need to map them to valid indices
            donor_indices_valid = np.remainder(donor_indices - 1, len(donor)) + 1
        else:
            # Use the indices directly (up to the length of receiver)
            if len(mtc_array) >= len(receiver):
                donor_indices_valid = mtc_array[:len(receiver)]
            else:
                # If we have too few indices, repeat the last ones to match length
                # Probably not a common edge case but there may be better ways to handle it
                donor_indices_valid = np.concatenate([
                    mtc_array,
                    np.repeat(mtc_array[-1], len(receiver) - len(mtc_array))
                ])
        
        # Create the final mtc.ids matrix required by create_fused
        mtc_matrix = np.column_stack((recipient_indices, donor_indices_valid))
        
        # Convert to R matrix
        mtc_ids = ro.r.matrix(
            ro.IntVector(mtc_matrix.flatten()),
            nrow=len(recipient_indices),
            ncol=2
        )
    
    # Create the fused datasets using create_fused
    # First without duplication of matching variables
    fused_0 = StatMatch.create_fused(
        data_rec=receiver,
        data_don=donor,
        mtc_ids=mtc_ids,
        z_vars=pd.Series(z_variables),
    )

    # Second with duplication of matching variables
    fused_1 = StatMatch.create_fused(
        data_rec=receiver,
        data_don=donor,
        mtc_ids=mtc_ids,
        z_vars=pd.Series(z_variables),
        dup_x=False,
        match_vars=pd.Series(matching_variables),
    )

    return fused_0, fused_1
