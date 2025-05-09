"""
Pipeline for autoimputation of missing values in a dataset.
This module integrates all steps necessary for method selection and imputation of missing values.
"""

import logging
import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Type

import joblib
import pandas as pd
from pydantic import validate_call
from rpy2.robjects import pandas2ri
from tqdm.auto import tqdm

from microimpute.comparisons import *
from microimpute.comparisons.data import preprocess_data
from microimpute.config import (
    QUANTILES,
    RANDOM_STATE,
    TRAIN_SIZE,
    VALIDATE_CONFIG,
)
from microimpute.evaluations import cross_validate_model
from microimpute.models import *

log = logging.getLogger(__name__)


@validate_call(config=VALIDATE_CONFIG)
def autoimpute(
    donor_data: pd.DataFrame,
    receiver_data: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    models: Optional[List[Type]] = None,
    quantiles: Optional[List[float]] = QUANTILES,
    hyperparameters: Optional[Dict[str, Dict[str, Any]]] = None,
    tune_hyperparameters: Optional[bool] = False,
    random_state: Optional[int] = RANDOM_STATE,
    train_size: Optional[float] = TRAIN_SIZE,
    k_folds: Optional[int] = 5,
    verbose: Optional[bool] = False,
) -> tuple[dict[float, pd.DataFrame], "Imputer", pd.DataFrame]:
    """Automatically select and apply the best imputation model.

    This function evaluates multiple imputation methods using cross-validation
    to determine which performs best on the provided donor data, then applies
    the winning method to impute values in the receiver data.

    Args:
        donor_data : Dataframe containing both predictor and target variables
            used  to train models
        receiver_data : Dataframe containing predictor variables where imputed
            values will be generated
        predictors : List of column names of predictor variables used to
            predict imputed variables
        imputed_variables : List of column names of variables to be imputed in
            the receiver data
        models : List of imputer model classes to compare.
            If None, uses [QRF, OLS, QuantReg, Matching]
        quantiles : List of quantiles to predict for each imputed variable.
            Uses default QUANTILES if not passed.
        hyperparameters : Dictionary of hyperparameters for specific models,
            with model names as keys. Defaults to None and uses default model hyperparameters then.
        tune_hyperparameters : Whether to tune hyperparameters for the models.
            Defaults to False.
        random_state : Random seed for reproducibility
        train_size : Proportion of data to use for training in preprocessing
        k_folds : Number of folds for cross-validation. Defaults to 5.
        verbose : Whether to print detailed logs. Defaults to False.

    Returns:
        A tuple containing:
        - Dictionary mapping quantiles to DataFrames of imputed values
        - The fitted imputation model (best performing)
        - DataFrame with cross-validation performance metrics for all evaluated models

    Raises:
        ValueError: If inputs are invalid (e.g., invalid quantiles, missing columns)
        RuntimeError: For unexpected errors during imputation
    """
    # Set up logging level based on verbose parameter
    log_level = logging.INFO if verbose else logging.WARNING
    log.setLevel(log_level)
    warnings.filterwarnings("ignore")

    # Set up parallel processing
    n_jobs: Optional[int] = -1

    # Create a progress tracking system
    if verbose:
        main_progress = tqdm(total=4, desc="AutoImputation progress")
        main_progress.set_description("Input validation")

    # Step 0: Input validation
    try:
        # Validate quantiles if provided
        if quantiles:
            invalid_quantiles = [q for q in quantiles if not 0 <= q <= 1]
            if invalid_quantiles:
                error_msg = f"Invalid quantiles (must be between 0 and 1): {invalid_quantiles}"
                log.error(error_msg)
                raise ValueError(error_msg)

        # Validate that predictor and imputed variable columns exist in donor data
        missing_predictors_donor = [
            col for col in predictors if col not in donor_data.columns
        ]
        if missing_predictors_donor:
            error_msg = f"Missing predictor columns in donor data: {missing_predictors_donor}"
            log.error(error_msg)
            raise ValueError(error_msg)

        missing_predictors_receiver = [
            col for col in predictors if col not in receiver_data.columns
        ]
        if missing_predictors_receiver:
            error_msg = f"Missing predictor columns in reciver data: {missing_predictors_receiver}"
            log.error(error_msg)
            raise ValueError(error_msg)

        missing_imputed_donor = [
            col for col in imputed_variables if col not in donor_data.columns
        ]
        if missing_imputed_donor:
            error_msg = f"Missing imputed variable columns in donor data: {missing_imputed_donor}"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Validate that predictor columns exist in receiver data (imputed variables may not be present in receiver data)
        missing_predictors_receiver = [
            col for col in predictors if col not in receiver_data.columns
        ]
        if missing_predictors_receiver:
            error_msg = f"Missing predictor columns in test data: {missing_predictors_receiver}"
            log.error(error_msg)
            raise ValueError(error_msg)

        log.info(
            f"Generating imputations to impute from {len(donor_data)} donor data to {len(receiver_data)} receiver data for variables {imputed_variables} with predictors {predictors}. "
        )

        if (hyperparameters is not None) and (tune_hyperparameters == True):
            error_msg = "Cannot specify both model_hyperparams and request to automatically tune hyperparameters, please select one or the other."
            log.error(error_msg)
            raise ValueError(error_msg)

        # Step 1: Data preparation
        if verbose:
            log.info("Preprocessing data...")
            main_progress.update(1)
            main_progress.set_description("Data preparation")

        # If imputed variables are in receiver data, remove them
        receiver_data = receiver_data.drop(
            columns=imputed_variables, errors="ignore"
        )

        training_data = donor_data.copy()
        imputing_data = receiver_data.copy()

        training_data[predictors], dummy_info = preprocess_data(
            training_data[predictors],
            full_data=True,
            train_size=train_size,
            test_size=(1 - train_size),
        )
        training_data[imputed_variables], dummy_info, normalizing_params = (
            preprocess_data(
                training_data[imputed_variables],
                full_data=True,
                train_size=train_size,
                test_size=(1 - train_size),
                normalizing_features=True,
            )
        )
        imputing_data, dummy_info = preprocess_data(
            imputing_data[predictors],
            full_data=True,
            train_size=train_size,
            test_size=(1 - train_size),
        )

        if dummy_info:
            # Retrieve new predictors and imputed variables after processed data
            for orig_col, dummy_cols in dummy_info.items():
                if orig_col in predictors:
                    predictors.remove(orig_col)
                    predictors.extend(dummy_cols)
                elif orig_col in imputed_variables:
                    imputed_variables.remove(orig_col)
                    imputed_variables.extend(dummy_cols)

        # Step 2: Imputation with each method
        if verbose:
            main_progress.update(1)
            main_progress.set_description("Model evaluation")

        if not models:
            # If no models are provided, use default models
            model_classes: List[Type[Imputer]] = [QRF, OLS, QuantReg, Matching]
        else:
            model_classes = models

        if hyperparameters:
            model_names = [
                model_class.__name__ for model_class in model_classes
            ]
            for model_name, model_params in hyperparameters.items():
                if model_name in model_names:
                    # Update the model class with the provided hyperparameters
                    if model_name == "QRF":
                        log.info(
                            f"Using hyperparameters for QRF: {model_params}"
                        )
                    elif model_name == "Matching":
                        log.info(
                            f"Using hyperparameters for Matching: {model_params}"
                        )
                else:
                    log.info(
                        f"None of the hyperparameters provided are relevant for the supported models: {model_names}. Using default hyperparameters."
                    )

        method_test_losses = {}
        log.info(
            "Hyperparameter tuning and cross-validation for model comparisson in progress... "
        )

        def evaluate_model(
            model: Type[Imputer],
            data: pd.DataFrame,
            predictors: List[str],
            imputed_variables: List[str],
            quantiles: List[float],
            k_folds: Optional[int] = 5,
            random_state: Optional[bool] = RANDOM_STATE,
            tune_hyperparams: Optional[bool] = True,
            hyperparameters: Optional[Dict[str, Any]] = None,
        ) -> tuple[str, pd.DataFrame]:
            """Evaluate a single imputation model with cross-validation.

            Args:
                model: The imputation model class to evaluate
                data: The dataset to use for evaluation
                predictors: List of predictor column names
                imputed_variables: List of columns to impute
                quantiles: List of quantiles to evaluate
                k_folds: Number of cross-validation folds
                random_state: Random seed for reproducibility
                tune_hyperparams: Whether to tune hyperparameters
                hyperparameters: Optional model-specific hyperparameters

            Returns:
                Tuple containing model name and cross-validation results DataFrame
            """
            model_name = model.__name__
            log.info(f"Evaluating {model_name}...")

            # For Matching model using R, we need to activate converters in this thread
            if model_name == "Matching":
                # Explicitly activate pandas-to-R conversion for this thread
                from rpy2.robjects import numpy2ri, pandas2ri

                pandas2ri.activate()
                numpy2ri.activate()

            return model_name, cross_validate_model(
                model_class=model,
                data=data,
                predictors=predictors,
                imputed_variables=imputed_variables,
                quantiles=quantiles,
                n_splits=k_folds,
                random_state=random_state,
                model_hyperparams=hyperparameters,
            )

        # Special handling for models that use rpy2
        # Use sequential processing for Matching model to avoid thread context issues
        has_matching = any(
            model.__name__ == "Matching" for model in model_classes
        )
        if has_matching and n_jobs != 1:
            log.info(
                "Using sequential processing (n_jobs=1) because Matching model is present"
            )
            n_jobs = 1

        parallel_tasks = []
        for model in model_classes:
            parallel_tasks.append(
                (
                    model,
                    training_data,
                    predictors,
                    imputed_variables,
                    quantiles,
                    k_folds,
                    RANDOM_STATE,
                    tune_hyperparameters,
                    hyperparameters,
                )
            )

        # Execute in parallel
        results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(lambda args: evaluate_model(*args))(task)
            for task in tqdm(parallel_tasks, desc="Evaluating models")
        )

        # Process results
        for model_name, cv_result in results:
            method_test_losses[model_name] = cv_result.loc["test"]

        method_results_df = pd.DataFrame.from_dict(
            method_test_losses, orient="index"
        )

        # Step 3: Compare imputation methods
        log.info(f"Comparing across {model_classes} methods. ")

        if verbose:
            main_progress.update(1)
            main_progress.set_description("Model selection")

        # add a column called "mean_loss" with the average loss across quantiles
        method_results_df["mean_loss"] = method_results_df.mean(axis=1)

        # Step 4: Select best method
        best_method = method_results_df["mean_loss"].idxmin()
        best_row = method_results_df.loc[best_method]

        log.info(
            f"The method with the lowest average loss is {best_method}, with an average loss across variables and quantiles of {best_row['mean_loss']}. "
        )

        # Step 5: Generate imputations with the best method on the receiver data
        log.info(
            f"Generating imputations using the best method: {best_method} on the receiver data. "
        )

        if verbose:
            main_progress.update(1)
            main_progress.set_description("Imputation")

        models_dict = {model.__name__: model for model in model_classes}
        chosen_model = models_dict[best_method]

        # Initialize the model
        model = chosen_model()
        imputation_q = 0.5  # this can be an input parameter, or if unspecified will default to a random quantile
        # Fit the model
        if best_method == "QuantReg":
            # For QuantReg, we need to explicitly fit the quantile
            fitted_model = model.fit(
                training_data,
                predictors,
                imputed_variables,
                quantiles=[imputation_q],
            )
        else:
            fitted_model = model.fit(
                training_data, predictors, imputed_variables
            )

        # Predict with explicit quantiles
        imputations = fitted_model.predict(
            imputing_data, quantiles=[imputation_q]
        )

        # Unnormalize the imputations
        mean = pd.Series(
            {col: p["mean"] for col, p in normalizing_params.items()}
        )
        std = pd.Series(
            {col: p["std"] for col, p in normalizing_params.items()}
        )
        unnormalized_imputations = {}
        for q, df in imputations.items():
            cols = df.columns  # the imputed variables
            df_unnorm = df.mul(std[cols], axis=1)  # × std
            df_unnorm = df_unnorm.add(mean[cols], axis=1)  # + mean
            unnormalized_imputations[q] = df_unnorm

        log.info(
            f"Imputation generation completed for {len(receiver_data)} samples using the best method: {best_method} and the median quantile. "
        )

        if verbose:
            main_progress.set_description("Complete")
            main_progress.close()

        median_imputations = unnormalized_imputations[
            0.5
        ]  # this may not work if we change the value of imputation_q
        # Add the imputed variables to the receiver data
        try:
            missing_imputed_vars = []
            for var in imputed_variables:
                if var in median_imputations.columns:
                    receiver_data[var] = median_imputations[var]
                else:
                    missing_imputed_vars.append(var)
                    log.warning(
                        f"Imputed variable {var} not found in the imputations. "
                    )
        except KeyError as e:
            error_msg = f"Missing imputed variable in the imputations: {e}"
            log.error(error_msg)
            raise ValueError(error_msg)

        return (
            unnormalized_imputations,
            receiver_data,
            fitted_model,
            method_results_df,
        )

    except ValueError as e:
        # Re-raise validation errors directly
        raise e
    except Exception as e:
        log.error(f"Unexpected error during autoimputation: {str(e)}")
        raise RuntimeError(f"Failed to generate imputations: {str(e)}") from e
