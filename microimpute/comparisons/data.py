"""Data preparation utilities for imputation benchmarking.

This module provides functions for acquiring, preprocessing, and splitting data for imputation benchmarking.
It includes utilities for downloading Survey of Consumer Finances
(SCF) data, normalizing features, and creating train-test splits with consistent parameters.
"""

import io
import logging
import zipfile
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from pydantic import validate_call
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from microimpute.config import (
    RANDOM_STATE,
    TEST_SIZE,
    TRAIN_SIZE,
    VALID_YEARS,
    VALIDATE_CONFIG,
)

logger = logging.getLogger(__name__)


@validate_call(config=VALIDATE_CONFIG)
def scf_url(year: int) -> str:
    """Return the URL of the SCF summary microdata zip file for a year.

    Args:
        year: Year of SCF summary microdata to retrieve.

    Returns:
        URL of summary microdata zip file for the given year.

    Raises:
        ValueError: If the year is not in VALID_YEARS.
    """
    logger.debug(f"Generating SCF URL for year {year}")

    if year not in VALID_YEARS:
        logger.error(
            f"Invalid SCF year: {year}. Valid years are {VALID_YEARS}"
        )
        raise ValueError(
            f"The SCF is not available for {year}. Valid years are {VALID_YEARS}"
        )

    url = f"https://www.federalreserve.gov/econres/files/scfp{year}s.zip"
    logger.debug(f"Generated URL: {url}")
    return url


@validate_call(config=VALIDATE_CONFIG)
def _load(
    years: Optional[Union[int, List[int]]] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load Survey of Consumer Finances data for specified years and columns.

    Args:
        years: Year or list of years to load data for.
        columns: List of column names to load.

    Returns:
        DataFrame containing the requested data.

    Raises:
        ValueError: If no Stata files are found in the downloaded zip
            or invalid parameters
        RuntimeError: If there's a network error or a problem processing
            the downloaded data
    """

    logger.info(f"Loading SCF data with years={years}")

    try:
        # Identify years for download
        if years is None:
            years = VALID_YEARS
            logger.warning(f"Using default years: {years}")

        if isinstance(years, int):
            years = [years]

        # Validate all years are valid
        invalid_years = [year for year in years if year not in VALID_YEARS]
        if invalid_years:
            logger.error(f"Invalid years specified: {invalid_years}")
            raise ValueError(
                f"Invalid years: {invalid_years}. Valid years are {VALID_YEARS}"
            )

        all_data: List[pd.DataFrame] = []

        for year in tqdm(years):
            logger.info(f"Processing data for year {year}")
            try:
                # Download zip file
                logger.debug(f"Downloading SCF data for year {year}")
                url = scf_url(year)
                try:
                    response = requests.get(url, timeout=60)
                    response.raise_for_status()  # Raise an error for bad responses
                except requests.exceptions.RequestException as e:
                    logger.error(
                        f"Network error downloading SCF data for year {year}: {str(e)}"
                    )
                    raise RuntimeError(
                        f"Failed to download SCF data for year {year}"
                    ) from e

                # Process zip file
                try:
                    logger.debug("Creating zipfile from downloaded content")
                    z = zipfile.ZipFile(io.BytesIO(response.content))

                    # Find the .dta file in the zip
                    dta_files: List[str] = [
                        f for f in z.namelist() if f.endswith(".dta")
                    ]
                    if not dta_files:
                        logger.error(
                            f"No Stata files found in zip for year {year}"
                        )
                        raise ValueError(
                            f"No Stata files found in zip for year {year}"
                        )

                    logger.debug(f"Found Stata files: {dta_files}")

                    # Read the Stata file
                    try:
                        logger.debug(f"Reading Stata file: {dta_files[0]}")
                        with z.open(dta_files[0]) as f:
                            df = pd.read_stata(
                                io.BytesIO(f.read()), columns=columns
                            )
                            logger.debug(
                                f"Read DataFrame with shape {df.shape}"
                            )

                        # Ensure 'wgt' is included
                        if (
                            columns is not None
                            and "wgt" not in df.columns
                            and "wgt" not in columns
                        ):
                            logger.debug("Re-reading with 'wgt' column added")
                            # Re-read to include weights
                            with z.open(dta_files[0]) as f:
                                cols_with_weight: List[str] = list(
                                    set(columns) | {"wgt"}
                                )
                                df = pd.read_stata(
                                    io.BytesIO(f.read()),
                                    columns=cols_with_weight,
                                )
                                logger.debug(
                                    f"Re-read DataFrame with shape {df.shape}"
                                )
                    except Exception as e:
                        logger.error(
                            f"Error reading Stata file for year {year}: {str(e)}"
                        )
                        raise RuntimeError(
                            f"Failed to process Stata file for year {year}"
                        ) from e

                except zipfile.BadZipFile as e:
                    logger.error(f"Bad zip file for year {year}: {str(e)}")
                    raise RuntimeError(
                        f"Downloaded zip file is corrupt for year {year}"
                    ) from e

                # Add year column
                df["year"] = year
                logger.info(
                    f"Successfully processed data for year {year}, shape: {df.shape}"
                )
                all_data.append(df)

            except Exception as e:
                logger.error(f"Error processing year {year}: {str(e)}")
                raise

        # Combine all years
        logger.debug(f"Combining data from {len(all_data)} years")
        if len(all_data) > 1:
            result = pd.concat(all_data)
            logger.info(
                f"Combined data from {len(years)} years, final shape: {result.shape}"
            )
            return result
        else:
            logger.info(
                f"Returning data for single year, shape: {all_data[0].shape}"
            )
            return all_data[0]

    except Exception as e:
        logger.error(f"Error in _load: {str(e)}")
        raise


@validate_call(config=VALIDATE_CONFIG)
def prepare_scf_data(
    full_data: bool = False, years: Optional[Union[int, List[int]]] = None
) -> Union[
    Tuple[pd.DataFrame, List[str], List[str], dict],  # when full_data=True
    Tuple[
        pd.DataFrame, pd.DataFrame, List[str], List[str], dict
    ],  # when full_data=False
]:
    """Preprocess the Survey of Consumer Finances data for model training and testing.

    Args:
        full_data: Whether to return the complete dataset without splitting.
        years: Year or list of years to load data for.

    Returns:
        Different tuple formats depending on the value of full_data:
          - If full_data=True: (data, predictor_columns, imputed_columns, dummy_info)
          - If full_data=False: (train_data, test_data,
                predictor_columns, imputed_columns, dummy_info)

        Where dummy_info is a dictionary with information about dummy variables created from string columns.

    Raises:
        ValueError: If required columns are missing from the data
        RuntimeError: If data processing fails
    """
    logger.info(
        f"Preparing SCF data with full_data={full_data}, years={years}"
    )

    try:
        # Load the raw data
        logger.debug("Loading SCF data")
        data = _load(years=years)

        # Define columns needed for analysis
        # predictors shared with cps data
        PREDICTORS: List[str] = [
            "hhsex",  # sex of head of household
            "age",  # age of respondent
            "married",  # marital status of respondent
            "kids",  # number of children in household
            "educ",  # highest level of education
            "race",  # race of respondent
            "income",  # total annual income of household
            "wageinc",  # income from wages and salaries
            "bussefarminc",  # income from business, self-employment or farm
            "intdivinc",  # income from interest and dividends
            "ssretinc",  # income from social security and retirement accounts
            "lf",  # labor force status
        ]

        IMPUTED_VARIABLES: List[str] = ["networth"]

        # Validate that all required columns exist in the data
        missing_columns = [
            col
            for col in PREDICTORS + IMPUTED_VARIABLES
            if col not in data.columns
        ]
        if missing_columns:
            logger.error(
                f"Required columns missing from SCF data: {missing_columns}"
            )
            raise ValueError(
                f"Required columns missing from SCF data: {missing_columns}"
            )

        logger.debug(
            f"Selecting {len(PREDICTORS)} predictors and {len(IMPUTED_VARIABLES)} imputation variables"
        )
        data = data[PREDICTORS + IMPUTED_VARIABLES]
        logger.debug(f"Data shape after column selection: {data.shape}")

        if full_data:
            logger.info("Processing full dataset without splitting")
            data, dummy_info = preprocess_data(data, full_data=True)
            logger.info(
                f"Returning full processed dataset with shape {data.shape}"
            )
            return data, PREDICTORS, IMPUTED_VARIABLES, dummy_info
        else:
            logger.info("Splitting data into train and test sets")
            X_train, X_test, dummy_info = preprocess_data(data)
            logger.info(
                f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}"
            )
            return X_train, X_test, PREDICTORS, IMPUTED_VARIABLES, dummy_info

    except Exception as e:
        logger.error(f"Error in prepare_scf_data: {str(e)}")
        raise RuntimeError(f"Failed to prepare SCF data: {str(e)}") from e


@validate_call(config=VALIDATE_CONFIG)
def preprocess_data(
    data: pd.DataFrame,
    full_data: Optional[bool] = False,
    train_size: Optional[float] = TRAIN_SIZE,
    test_size: Optional[float] = TEST_SIZE,
    random_state: Optional[int] = RANDOM_STATE,
    normalize: Optional[bool] = False,
) -> Union[
    Tuple[pd.DataFrame, dict],  # when full_data=True
    Tuple[pd.DataFrame, pd.DataFrame, dict],  # when full_data=False
]:
    """Preprocess the data for model training and testing.

    Args:
        data: DataFrame containing the data to preprocess.
        full_data: Whether to return the complete dataset without splitting.
        train_size: Proportion of the dataset to include in the train split.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Random seed for reproducibility.
        normalize: Whether to normalize the data.

    Returns:
        Different tuple formats depending on the value of full_data:
          - If full_data=True: (data, dummy_info)
          - If full_data=False: (X_train, X_test, dummy_info)

        Where dummy_info is a dictionary mapping original columns to their resulting dummy columns

    Raises:
        ValueError: If data is empty or invalid
        RuntimeError: If data preprocessing fails
    """

    logger.debug(
        f"Preprocessing data with shape {data.shape}, full_data={full_data}"
    )

    # Initialize dummy information dictionary
    dummy_info = {
        "original_dtypes": {},
        "column_mapping": {},
        "original_categories": {},
    }

    try:
        if data.empty:
            raise ValueError("Data must not be None or empty")
        # Check for missing values
        missing_count = data.isna().sum().sum()
        if missing_count > 0:
            logger.warning(f"Data contains {missing_count} missing values")

        # Transform boolean and categorical columns to numerical format
        logger.debug(
            "Converting boolean and categorical columns to numerical format"
        )
        try:
            # Identify boolean columns and convert them to strings
            bool_columns = [
                col
                for col in data.columns
                if (
                    pd.api.types.is_bool_dtype(data[col])
                    or (
                        pd.api.types.is_integer_dtype(data[col])
                        and set(data[col].unique()) == {0, 1}
                    )
                    or (
                        pd.api.types.is_float_dtype(data[col])
                        and set(data[col].unique()) == {0.0, 1.0}
                    )
                )
            ]

            if bool_columns:
                logger.info(
                    f"Found {len(bool_columns)} boolean columns to convert: {bool_columns}"
                )
                for col in bool_columns:
                    dummy_info["original_dtypes"][col] = (
                        "bool",
                        data[col].dtype,
                    )
                    # For boolean columns, map the column to itself since we don't create dummies
                    dummy_info["column_mapping"][col] = [col]
                    data[col] = data[col].astype("float64")

            # Identify string and object columns (excluding already processed booleans)
            string_columns = [
                col
                for col in data.columns
                if (
                    pd.api.types.is_string_dtype(data[col])
                    or pd.api.types.is_object_dtype(data[col])
                )
                and col not in bool_columns
            ]

            # Identify numeric columns that represent categorical data
            numeric_categorical_columns = [
                col
                for col in data.columns
                if pd.api.types.is_numeric_dtype(data[col])
                and data[col].nunique()
                < 10  # Parse as category if unique count < 10
                and col
                not in bool_columns  # Exclude already processed boolean columns
            ]

            if numeric_categorical_columns:
                logger.warning(
                    f"Found {len(numeric_categorical_columns)} numeric columns with unique values < 10, treating as categorical: {numeric_categorical_columns}. Converting to numeric variable."
                )
                for col in numeric_categorical_columns:
                    dummy_info["original_categories"][col] = [
                        float(i) for i in data[col].unique().tolist()
                    ]
                    dummy_info["original_dtypes"][col] = (
                        "numeric categorical",
                        data[col].dtype,
                    )
                    data[col] = data[col].astype("float64")

            if string_columns:
                logger.info(
                    f"Found {len(string_columns)} categorical columns to convert: {string_columns}"
                )

                # Store original categories and dtypes for categorical columns
                for col in string_columns:
                    dummy_info["original_dtypes"][col] = (
                        "categorical",
                        data[col].dtype,
                    )
                    dummy_info["original_categories"][col] = (
                        data[col].unique().tolist()
                    )

            if string_columns:
                # Use pandas get_dummies to create one-hot encoded features
                dummy_data = pd.get_dummies(
                    data[string_columns], dtype="float64"
                )
                for col in dummy_data.columns:
                    dummy_data[col] = dummy_data[col].astype("float64")
                logger.debug(
                    f"Created {dummy_data.shape[1]} dummy variables from {len(string_columns)} string columns"
                )

                # Create mapping from original columns to their resulting dummy columns
                for orig_col in string_columns:
                    # Find all dummy columns that came from this original column
                    related_dummies = [
                        col
                        for col in dummy_data.columns
                        if col.startswith(f"{orig_col}_")
                    ]
                    dummy_info["column_mapping"][orig_col] = (
                        related_dummies
                        if len(related_dummies) > 0
                        else [orig_col]
                    )

                # Drop original string columns and join the dummy variables
                numeric_data = data.drop(columns=string_columns)
                logger.debug(
                    f"Removed original string columns, data shape: {numeric_data.shape}"
                )

                # Combine numeric columns with dummy variables
                data = pd.concat([numeric_data, dummy_data], axis=1)
                for col in data.columns:
                    data[col] = data[col].astype("float64")
                logger.info(
                    f"Data shape after dummy variable conversion: {data.shape}"
                )
        except Exception as e:
            logger.error(f"Error during string column conversion: {str(e)}")
            raise RuntimeError(
                "Failed to convert string columns to dummy variables"
            ) from e

        if normalize:
            logger.debug("Normalizing data")
            try:
                mean = data.mean(axis=0)
                std = data.std(axis=0)

                # Check for constant columns (std=0)
                constant_cols = std[std == 0].index.tolist()
                if constant_cols:
                    logger.warning(
                        f"Found constant columns (std=0): {constant_cols}"
                    )
                    # Handle constant columns by setting std to 1 to avoid division by zero
                    for col in constant_cols:
                        std[col] = 1

                # Apply normalization
                data = (data - mean) / std
                logger.debug("Data normalized successfully")

                # Store normalization parameters
                normalization_params = {
                    col: {"mean": mean[col], "std": std[col]}
                    for col in data.columns
                }

                logger.debug(
                    f"Normalization parameters: {normalization_params}"
                )

            except Exception as e:
                logger.error(f"Error during data normalization: {str(e)}")
                raise RuntimeError("Failed to normalize data") from e

        if full_data and normalize:
            logger.info("Returning full preprocessed dataset")
            return data, dummy_info, normalization_params
        elif full_data:
            logger.info("Returning full preprocessed dataset")
            return data, dummy_info
        else:
            logger.debug(
                f"Splitting data with train_size={train_size}, test_size={test_size}"
            )
            try:
                X_train, X_test = train_test_split(
                    data,
                    test_size=test_size,
                    train_size=train_size,
                    random_state=random_state,
                )
                logger.info(
                    f"Data split into train ({X_train.shape}) and test ({X_test.shape}) sets"
                )
                if normalize:
                    return (
                        X_train,
                        X_test,
                        dummy_info,
                        normalization_params,
                    )
                else:
                    return X_train, X_test, dummy_info

            except Exception as e:
                logger.error(f"Error during train-test split: {str(e)}")
                raise RuntimeError(
                    "Failed to split data into train and test sets"
                ) from e

    except Exception as e:
        logger.error(f"Error in preprocess_data: {str(e)}")
        raise


@validate_call(config=VALIDATE_CONFIG)
def postprocess_imputations(
    imputations: Dict[float, pd.DataFrame], dummy_info: Dict[str, Any]
) -> Dict[float, pd.DataFrame]:
    """Convert imputed bool and categorical dummy variables back to original data types.

    This function reverses the encoding applied by preprocess_data,
    converting dummy variables back to their original boolean or categorical forms.
    For numeric categorical variables, values are rounded to the nearest valid category.

    Args:
        imputations: Dictionary mapping quantiles to DataFrames of imputed values
        dummy_info: Dictionary containing information about dummy variable mappings
            and original data types

    Returns:
        Dictionary mapping quantiles to DataFrames with original data types restored

    Raises:
        ValueError: If dummy_info is missing required information
        RuntimeError: If conversion back to original types fails
    """
    logger.debug(
        f"Post-processing {len(imputations)} quantile imputations with dummy_info keys: {dummy_info.keys()}"
    )

    try:
        processed_imputations = {}

        for quantile, df in imputations.items():
            logger.debug(
                f"Processing quantile {quantile} with shape {df.shape}"
            )
            df_processed = df.copy()

            for orig_col, dummy_cols in dummy_info.get(
                "column_mapping", {}
            ).items():
                if orig_col in dummy_info.get("original_dtypes", {}):
                    orig_dtype_info = dummy_info["original_dtypes"][orig_col]

                    # Extract dtype category and original pandas dtype
                    if (
                        isinstance(orig_dtype_info, tuple)
                        and len(orig_dtype_info) == 2
                    ):
                        dtype_category, original_pandas_dtype = orig_dtype_info
                    else:
                        # Fallback for old format
                        logger.warning(
                            f"Unexpected dtype format for {orig_col}: {orig_dtype_info}"
                        )
                        continue

                    # Check if this variable was imputed based on its type
                    is_imputed = False
                    if dtype_category in ["bool", "numeric categorical"]:
                        # For bool and numeric categorical, check if original column is present
                        is_imputed = orig_col in df_processed.columns
                    elif dtype_category == "categorical":
                        # For regular categorical, check if dummy columns are present
                        available_dummies = [
                            col
                            for col in dummy_cols
                            if col in df_processed.columns
                        ]
                        is_imputed = len(available_dummies) > 0

                    if not is_imputed:
                        logger.debug(
                            f"Skipping {orig_col} - not in imputed variables"
                        )
                        continue

                    logger.debug(
                        f"Converting {orig_col} back to {dtype_category} with original dtype {original_pandas_dtype}"
                    )

                    if dtype_category == "bool":
                        # Convert back to boolean from float (>0.5 threshold for discretization)
                        df_processed[orig_col] = df_processed[orig_col] > 0.5
                        # Convert to original boolean dtype
                        df_processed[orig_col] = df_processed[orig_col].astype(
                            original_pandas_dtype
                        )
                        logger.debug(
                            f"Converted {orig_col} back to boolean type {original_pandas_dtype}"
                        )

                    # Handle numeric categorical columns - round to nearest valid category
                    elif dtype_category == "numeric categorical":
                        categories = dummy_info["original_categories"][
                            orig_col
                        ]
                        logger.debug(
                            f"Rounding {orig_col} to nearest categories: {categories}"
                        )

                        def round_to_nearest_category(value):
                            if pd.isna(value):
                                return categories[
                                    0
                                ]  # Default to first category for NaN
                            return min(
                                categories,
                                key=lambda x: abs(float(x) - float(value)),
                            )

                        df_processed[orig_col] = df_processed[orig_col].apply(
                            round_to_nearest_category
                        )

                        # Convert back to original numeric type
                        try:
                            df_processed[orig_col] = df_processed[
                                orig_col
                            ].astype(original_pandas_dtype)
                            logger.debug(
                                f"Converted {orig_col} back to numeric categorical type: {original_pandas_dtype}"
                            )
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                f"Could not convert {orig_col} to {original_pandas_dtype}: {e}"
                            )
                            # Fallback: ensure it's numeric
                            df_processed[orig_col] = pd.to_numeric(
                                df_processed[orig_col], errors="coerce"
                            )
                            # Fill any NaN with first category
                            df_processed[orig_col] = df_processed[
                                orig_col
                            ].fillna(categories[0])
                            # Try to convert to int if all values are integers
                            if (df_processed[orig_col] % 1 == 0).all():
                                try:
                                    df_processed[orig_col] = df_processed[
                                        orig_col
                                    ].astype("int64")
                                except:
                                    pass  # Keep as float if int conversion fails

                    # Handle regular categorical columns that use dummy variables
                    elif dtype_category == "categorical":
                        # Find available dummy columns
                        available_dummies = [
                            col
                            for col in dummy_cols
                            if col in df_processed.columns
                        ]

                        if len(available_dummies) > 0:
                            logger.debug(
                                f"Converting dummy columns back to categorical {orig_col}"
                            )

                            categories = dummy_info["original_categories"][
                                orig_col
                            ]

                            dummy_subset = df_processed[available_dummies]

                            # Find the dummy column with highest value for each row
                            max_idx = dummy_subset.idxmax(axis=1)

                            # Create mapping from dummy columns to their categories
                            category_mapping = {}
                            for cat in categories:
                                dummy_name = f"{orig_col}_{cat}"
                                if dummy_name in available_dummies:
                                    category_mapping[dummy_name] = cat

                            df_processed[orig_col] = max_idx.map(
                                category_mapping
                            )

                            nan_mask = df_processed[orig_col].isna()
                            if nan_mask.any():
                                # For rows with NaN, find the category with highest average probability
                                avg_probs = dummy_subset[nan_mask].mean(axis=0)
                                if len(avg_probs) > 0:
                                    best_dummy = avg_probs.idxmax()
                                    default_category = category_mapping.get(
                                        best_dummy, categories[0]
                                    )
                                else:
                                    default_category = categories[0]
                                df_processed.loc[nan_mask, orig_col] = (
                                    default_category
                                )
                                logger.warning(
                                    f"Some values could not be mapped for {orig_col}, using default: {default_category}"
                                )

                            # Convert to original categorical type if needed
                            try:
                                if original_pandas_dtype != "object":
                                    df_processed[orig_col] = df_processed[
                                        orig_col
                                    ].astype(original_pandas_dtype)
                                    logger.debug(
                                        f"Converted {orig_col} back to categorical type: {original_pandas_dtype}"
                                    )
                            except (ValueError, TypeError) as e:
                                logger.warning(
                                    f"Could not convert {orig_col} to {original_pandas_dtype}: {e}"
                                )

                            # Drop the dummy columns
                            df_processed = df_processed.drop(
                                columns=available_dummies
                            )
                            logger.debug(
                                f"Converted dummy columns back to categorical {orig_col}"
                            )
                        else:
                            logger.warning(
                                f"No dummy columns found for categorical variable {orig_col}"
                            )

            processed_imputations[quantile] = df_processed
            logger.debug(
                f"Processed quantile {quantile}, final shape: {df_processed.shape}"
            )

        logger.info(
            f"Successfully post-processed {len(processed_imputations)} quantile imputations"
        )
        return processed_imputations

    except Exception as e:
        logger.error(f"Error in postprocess_imputations: {str(e)}")
        raise RuntimeError(
            f"Failed to post-process imputations: {str(e)}"
        ) from e
