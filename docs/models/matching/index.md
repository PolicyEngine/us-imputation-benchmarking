# Statistical Matching

The `Matching` model performs imputation using nearest neighbor distance hot deck matching. It uses donor data to impute missing values in recipient data by matching records based on similarities in predictor variables.

## How It Works
- Uses R's StatMatch package through rpy2 to perform nearest neighbor distance hot deck matching
- Stores donor data and variable names during fitting
- During prediction, matches the test data (recipients) with similar records in the donor data
- Transfers values from donor to recipient for the imputed variables

## Key Features
- Non-parametric approach that doesn't assume a specific distribution
- Preserves the empirical distribution of the imputed variables
- Handles complex relationships between variables
- Returns actual observed values rather than modeled estimates
