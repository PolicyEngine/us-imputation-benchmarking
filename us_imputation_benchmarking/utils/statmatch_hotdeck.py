import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import rpy2
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr

from us_imputation_benchmarking.utils.logging_utils import get_logger

# Enable R-Python DataFrame and array conversion
pandas2ri.activate()
numpy2ri.activate()
utils = importr("utils")
utils.chooseCRANmirror(ind=1)
StatMatch = importr("StatMatch")


log = get_logger(__name__)

"""
data.rec: A matrix or data frame that plays the role of recipient in the 
         statistical matching application.

data.don: A matrix or data frame that that plays the role of donor in the 
         statistical matching application.

mtc.ids: A matrix with two columns. Each row must contain the name or the index 
         of the recipient record (row) in data.don and the name or the index of 
         the corresponding donor record (row) in data.don. Note that this type 
         of matrix is returned by the functions NND.hotdeck, RANDwNND.hotdeck, 
         rankNND.hotdeck, and mixed.mtc.

z.vars: A character vector with the names of the variables available only in 
        data.don that should be "donated" to data.rec.
networth

dup.x: Logical. When TRUE the values of the matching variables in data.don are 
       also "donated" to data.rec. The names of the matching variables have to 
       be specified with the argument match.vars. To avoid confusion, the 
       matching variables added to data.rec are renamed by adding the suffix 
       "don". By default dup.x=FALSE.

match.vars: A character vector with the names of the matching variables. It has 
           to be specified only when dup.x=TRUE. All other vars (no need to 
           specify)
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
        ValueError: If receiver, donor, or matching_variables are not provided.
        ValueError: If matching_variables are not found in either dataset.
        ValueError: If z_variables are not found in donor dataset.
        RuntimeError: If R operations or statistical matching fails.
    """
    try:
        # Validate inputs
        if receiver is None or donor is None:
            error_msg = "Receiver and donor must be provided"
            log.error(error_msg)
            raise ValueError(error_msg)

        if matching_variables is None or not matching_variables:
            error_msg = "Matching variables must be provided and cannot be empty"
            log.error(error_msg)
            raise ValueError(error_msg)

        if z_variables is None or not z_variables:
            error_msg = "z_variables must be provided and cannot be empty"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Validate that matching variables exist in both datasets
        missing_in_receiver = [
            v for v in matching_variables if v not in receiver.columns
        ]
        missing_in_donor = [v for v in matching_variables if v not in donor.columns]

        if missing_in_receiver:
            error_msg = f"Matching variables missing in receiver: {missing_in_receiver}"
            log.error(error_msg)
            raise ValueError(error_msg)

        if missing_in_donor:
            error_msg = f"Matching variables missing in donor: {missing_in_donor}"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Validate that z_variables exist in donor
        missing_z = [v for v in z_variables if v not in donor.columns]
        if missing_z:
            error_msg = f"Z variables missing in donor: {missing_z}"
            log.error(error_msg)
            raise ValueError(error_msg)

        log.info(
            f"Performing matching with {len(receiver)} recipient and {len(donor)} donor records"
        )
        log.info(f"Using matching variables: {matching_variables}")
        log.info(f"Donating variables: {z_variables}")

        # Make sure R<->Python conversion is enabled
        pandas2ri.activate()

        try:
            # Import R's StatMatch package
            StatMatch = importr("StatMatch")
        except Exception as r_import_error:
            log.error(f"Failed to import R's StatMatch package: {str(r_import_error)}")
            raise RuntimeError(
                f"Error importing R package: {str(r_import_error)}"
            ) from r_import_error

        try:
            # Call the NND_hotdeck function from R
            log.info("Calling R's NND_hotdeck function")
            out_NND = StatMatch.NND_hotdeck(
                data_rec=receiver,
                data_don=donor,
                match_vars=pd.Series(matching_variables),
            )
            log.info("NND_hotdeck completed successfully")
        except Exception as nnd_error:
            log.error(f"Error in R's NND_hotdeck function: {str(nnd_error)}")
            raise RuntimeError(
                f"Statistical matching failed: {str(nnd_error)}"
            ) from nnd_error

        try:
            # Create the correct matching indices matrix for StatMatch.create_fused
            # Get all indices as 1-based (for R)
            recipient_indices = np.arange(1, len(receiver) + 1)

            # For direct NND matching we need the row positions from mtc.ids
            mtc_ids_r = out_NND.rx2("mtc.ids")

            # Create the properly formatted 2-column matrix that create_fused expects
            if hasattr(mtc_ids_r, "ncol") and mtc_ids_r.ncol == 2:
                log.info("Using pre-formatted mtc.ids matrix")
                # Already a matrix with the right shape, use it directly
                mtc_ids = mtc_ids_r
            else:
                log.info("Converting mtc.ids to proper format")
                # The IDs returned aren't in the expected format, extract and convert them
                mtc_array = np.array(mtc_ids_r)

                # If we have a 1D array with strings, convert to integers
                if mtc_array.dtype.kind in ["U", "S"]:
                    log.info("Converting string indices to integers")
                    mtc_array = np.array([int(x) for x in mtc_array])

                # If the mtc.ids array has 2 values per recipient (recipient_idx, donor_idx pairs)
                if len(mtc_array) == 2 * len(receiver):
                    log.info("Processing paired indices")
                    # Extract only the donor indices (every second value)
                    donor_indices = mtc_array.reshape(-1, 2)[:, 1]

                    # Make sure these indices are within the valid range (1 to donor dataset size)
                    # If they're not, we need to map them to valid indices
                    donor_indices_valid = (
                        np.remainder(donor_indices - 1, len(donor)) + 1
                    )
                else:
                    log.info(f"Processing {len(mtc_array)} unpaired indices")
                    # Use the indices directly (up to the length of receiver)
                    if len(mtc_array) >= len(receiver):
                        donor_indices_valid = mtc_array[: len(receiver)]
                    else:
                        # If we have too few indices, repeat the last ones to match length
                        log.warning(
                            f"Too few matching indices ({len(mtc_array)}) for "
                            f"{len(receiver)} recipients. Repeating the last index."
                        )
                        donor_indices_valid = np.concatenate(
                            [
                                mtc_array,
                                np.repeat(
                                    mtc_array[-1], len(receiver) - len(mtc_array)
                                ),
                            ]
                        )

                # Create the final mtc.ids matrix required by create_fused
                mtc_matrix = np.column_stack((recipient_indices, donor_indices_valid))

                # Convert to R matrix
                mtc_ids = ro.r.matrix(
                    ro.IntVector(mtc_matrix.flatten()),
                    nrow=len(recipient_indices),
                    ncol=2,
                )
        except Exception as matrix_error:
            log.error(f"Error processing matching indices: {str(matrix_error)}")
            raise RuntimeError(
                f"Failed to process matching indices: {str(matrix_error)}"
            ) from matrix_error

        try:
            # Create the fused datasets using create_fused
            log.info(
                "Creating first fused dataset (without duplicating matching variables)"
            )
            # First without duplication of matching variables
            fused_0 = StatMatch.create_fused(
                data_rec=receiver,
                data_don=donor,
                mtc_ids=mtc_ids,
                z_vars=pd.Series(z_variables),
            )

            log.info(
                "Creating second fused dataset (with duplicating matching variables)"
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

            log.info("Fusion completed successfully")
            return fused_0, fused_1

        except Exception as fusion_error:
            log.error(f"Error creating fused datasets: {str(fusion_error)}")
            raise RuntimeError(
                f"Failed to create fused datasets: {str(fusion_error)}"
            ) from fusion_error

    except ValueError as ve:
        # Re-raise validation errors
        raise ve
    except Exception as e:
        # Catch any other unexpected errors
        log.error(f"Unexpected error in statistical matching: {str(e)}")
        raise RuntimeError(
            f"Statistical matching failed with unexpected error: {str(e)}"
        ) from e
