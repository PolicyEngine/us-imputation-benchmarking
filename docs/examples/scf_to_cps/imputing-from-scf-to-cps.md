# Imputing wealth from the SCF to the CPS

This notebook demonstrates how to use the `microimpute` package and specifically the `autoimpute` function to impute wealth variables from the Survey of Consumer Finances to the Current Population Survey.

The Survey of Consumer Finances (SCF) is a triennial survey conducted by the Federal Reserve that collects detailed information on U.S. families' balance sheets, income, and demographic characteristics, with a special focus on wealth measures. The Current Population Survey (CPS) is a monthly survey conducted by the Census Bureau that provides comprehensive data on the labor force, employment, unemployment, and demographic characteristics, but lacks detailed wealth information.

By using `microimpute`, wealth information can be transfered from the SCF to the CPS, enabling economic analyses that require both detailed labor market and wealth data.

```python
import io
import logging
import zipfile
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import validate_call
from tqdm import tqdm
import warnings

from microimpute.config import (
    VALIDATE_CONFIG, VALID_YEARS, PLOT_CONFIG
)
from microimpute.comparisons import *
from microimpute.visualizations import *

logger = logging.getLogger(__name__)
```

## Loading and preparing the SCF and CPS datasets

The first step in the imputation process involves acquiring and harmonizing the two datasets. Extracting data from the SCF and the CPS, and then processing it to ensure the variables are compatible for imputation are crucial pre-processing steps for successful imputation. This involves identifying predictor variables that exist in both data sets and can meaningfully predict wealth, as well as ensuring they are named and encoded identically. 

```python 
@validate_call(config=VALIDATE_CONFIG)
def scf_url(year: int, VALID_YEARS: List[int] = VALID_YEARS) -> str:
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
def load_scf(
    years: Optional[Union[int, List[int]]] = VALID_YEARS,
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
```

```python
predictors = [
    "hhsex",  # sex of head of household
    "age",  # age of respondent (hoh)
    "married",  # marital status of respondent (hoh)
    "kids",  # number of children in household
    "race",  # race of respondent (hoh)
    "income",  # total annual income of household
    "wageinc",  # income from wages and salaries
    "bussefarminc",  # income from business, self-employment or farm
    "intdivinc",  # income from interest and dividends
    "ssretinc",  # income from social security and retirement accounts
]

imputed_variables = ["networth"]

scf_data = load_scf(2022)
scf_data = scf_data[predictors + imputed_variables]
scf_marital_mapping = {
    1: 1,  # Married - civilian spouse present -> Married
    2: 5,  # Living with partner -> Never Married
    3: 2,  # Separated -> Separated
    4: 3,  # Divorced -> Divorced
    5: 4,  # Widowed -> Widowed
    6: 5,  # Never Married -> Never Married
}

# Apply the mapping to recode the marital status values
scf_data["married"] = np.vectorize(scf_marital_mapping.get)(
    scf_data["married"]
)

file_path = "/Users/movil1/Desktop/PYTHONJOBS/PolicyEngine/microimpute/docs/examples/data/Census CPS 2022.h5"

household_df = pd.read_hdf(file_path, key="household")
household_df = household_df.set_index("H_SEQ")
person_df = pd.read_hdf(file_path, key="person")
person_df = person_df.set_index("PH_SEQ")
cps_data = pd.DataFrame()
cps_data["household_id"] = household_df.index
person_df["is_household_head"] = person_df.P_SEQ == 1
cps_data["hhsex"] = (
    person_df[person_df.is_household_head].loc[household_df.index].A_SEX.values
)
cps_data["age"] = (
    person_df[person_df.is_household_head].loc[household_df.index].A_AGE.values
)
cps_data["married"] = (
    person_df[person_df.is_household_head]
    .loc[household_df.index]
    .A_MARITL.values
)
person_df["is_child"] = person_df.A_AGE < 18
cps_data["kids"] = (
    person_df.groupby(person_df.index)
    .is_child.sum()
    .loc[household_df.index]
    .values
)
cps_data["race"] = (
    person_df[person_df.is_household_head]
    .loc[household_df.index]
    .PRDTRACE.values
)
cps_data["income"] = household_df.HTOTVAL.values
cps_data["wageinc"] = household_df.HWSVAL.values
cps_data["bussefarminc"] = (
    household_df.HSEVAL.values + household_df.HFRVAL.values
)
cps_data["intdivinc"] = (
    household_df.HINTVAL.values + household_df.HDIVVAL.values
)
cps_data["ssretinc"] = (
    household_df.HSSVAL.values
    + household_df.HSSIVAL.values
    + household_df.HPENVAL.values
)

cps_marital_mapping = {
    1: 1,  # Married - civilian spouse present -> Married
    2: 1,  # Married - AF spouse present -> Married
    3: 1,  # Married - spouse absent -> Married
    4: 4,  # Widowed -> Widowed
    5: 3,  # Divorced -> Divorced
    6: 2,  # Separated -> Separated
    7: 5,  # Never married -> Never married
}

# Apply the mapping to recode the marital status values
cps_data["married"] = np.vectorize(cps_marital_mapping.get)(
    cps_data["married"]
)

cps_race_mapping = {
    1: 1,  # White only -> WHITE
    2: 2,  # Black only -> BLACK/AFRICAN-AMERICAN
    3: 5,  # American Indian, Alaskan Native only -> AMERICAN INDIAN/ALASKA NATIVE
    4: 4,  # Asian only -> ASIAN
    5: 6,  # Hawaiian/Pacific Islander only -> NATIVE HAWAIIAN/PACIFIC ISLANDER
    6: 7,  # White-Black -> OTHER
    7: 7,  # White-AI -> OTHER
    8: 7,  # White-Asian -> OTHER
    9: 7,  # White-HP -> OTHER
    10: 7,  # Black-AI -> OTHER
    11: 7,  # Black-Asian -> OTHER
    12: 7,  # Black-HP -> OTHER
    13: 7,  # AI-Asian -> OTHER
    14: 7,  # AI-HP -> OTHER
    15: 7,  # Asian-HP -> OTHER
    16: 7,  # White-Black-AI -> OTHER
    17: 7,  # White-Black-Asian -> OTHER
    18: 7,  # White-Black-HP -> OTHER
    19: 7,  # White-AI-Asian -> OTHER
    20: 7,  # White-AI-HP -> OTHER
    21: 7,  # White-Asian-HP -> OTHER
    22: 7,  # Black-AI-Asian -> OTHER
    23: 7,  # White-Black-AI-Asian -> OTHER
    24: 7,  # White-AI-Asian-HP -> OTHER
    25: 7,  # Other 3 race comb. -> OTHER
    26: 7,  # Other 4 or 5 race comb. -> OTHER
}

# Apply the mapping to recode the race values
cps_data["race"] = np.vectorize(cps_race_mapping.get)(cps_data["race"])
```

## Running wealth imputation with autoimpute

After harmonizing the two datasets, the `autoimpute` function from `microimpute` can be used to transfer wealth information from the SCF to the CPS. This powerful function streamlines the imputation process by automating hyperparameter tuning, method selection, validation, and application.

Behind the scenes, `autoimpute` evaluates multiple statistical approaches, including Quantile Random Forest, Ordinary Least Squares, Quantile Regression, and Statistical Matching. It performs cross-validation to determine which method most accurately captures the relationship between the predictor variables and wealth measures in the SCF data. The function then applies the best-performing method to generate synthetic wealth values for CPS households.

By enabling hyperparameter tuning, the function can optimize each method's parameters, further improving imputation accuracy. This automated approach saves considerable time and effort compared to manually testing different imputation strategies, while ensuring the selection of the most appropriate method for this specific imputation task.

```python
warnings.filterwarnings("ignore")

# Run the autoimpute process
imputations, imputed_data, fitted_model, method_results_df = autoimpute(
    donor_data=scf_data,
    receiver_data=cps_data,
    predictors=predictors,
    imputed_variables=imputed_variables,
    tune_hyperparameters=True,  # enable automated hyperparameter tuning
)

print(
    f"Shape of receiver data before imputation: {cps_data.shape} \nShape of receiver data after imputation: {imputed_data.shape}"
)
```

## Comparing method performance

The method comparison plot below shows how different imputation methods performed across various quantiles. Lower quantile loss values indicate better performance. 

```python
# Extract the quantiles used in the evaluation
quantiles = [q for q in method_results_df.columns if isinstance(q, float)]

comparison_viz = method_comparison_results(
    data=method_results_df,
    metric_name="Test Quantile Loss",
    data_format="wide",
)
fig = comparison_viz.plot(
    title="Autoimpute Method Comparison",
    show_mean=True,
)
```

![png](./scf_to_cps_loss.png)

## Evaluating wealth imputations

To assess the imputation results, a comparison the distribution of wealth in the original SCF data with the imputed values in the CPS allow examining how well the imputation preserves important characteristics of the wealth distribution, such as its shape, central tendency, and dispersion.

Wealth distributions are typically highly skewed, with a long right tail representing a small number of households with very high net worth. A successful imputation should preserve this characteristic skewness while maintaining realistic values across the entire distribution. Examining both the raw distributions and log-transformed versions of wealth values can better capture important information for evaluation.

```python
def plot_networth_distributions(
    scf_data: pd.DataFrame,
    imputed_data: pd.DataFrame,
) -> go.Figure:
    """Plot the distribution of net worth in SCF and imputed CPS data.

    Args:
        scf_data: DataFrame containing SCF data.
        imputed_data: DataFrame containing imputed CPS data.

    Returns:
        Plotly figure object.
    """
    logger.info("Plotting net worth distributions")

    # 0. Check for missing values
    if scf_data['networth'].isnull().any():
        logger.warning("SCF data contains missing values in 'networth' column")
    if imputed_data['networth'].isnull().any():
        logger.warning("Imputed data contains missing values in 'networth' column")    
    
    # 1. Compute medians & means
    scf_median = scf_data['networth'].median()
    scf_mean   = scf_data['networth'].mean()
    cps_median = imputed_data['networth'].median()
    cps_mean   = imputed_data['networth'].mean()

    # 2. Compute upper_limit & true max counts
    upper_limit = np.percentile(scf_data['networth'], 99.5)
    scf_counts, _ = np.histogram(scf_data['networth'], bins=150, range=(0,upper_limit))
    cps_counts, _ = np.histogram(imputed_data['networth'], bins=150, range=(0,upper_limit))
    max_scf = scf_counts.max()
    max_cps = cps_counts.max()

    # 3. Build subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            "Distribution of Networth in SCF Data (Original)",
            "Distribution of Imputed Networth in CPS Data"
        ],
        vertical_spacing=0.15
    )

    # 4. Add histograms
    fig.add_trace(
        go.Histogram(
            x=scf_data['networth'],
            nbinsx=150,
            opacity=0.7,
            marker_color='blue',
            showlegend=False
        ), row=1, col=1
    )
    fig.add_trace(
        go.Histogram(
            x=imputed_data['networth'],
            nbinsx=150,
            opacity=0.7,
            marker_color='purple',
            showlegend=False
        ), row=2, col=1
    )

    # 5. Add full-height median/mean lines for SCF
    fig.add_trace(
        go.Scatter(
            x=[scf_median, scf_median],
            y=[0, 20000],
            mode='lines',
            line=dict(color='blue',   dash='dash'),
            name=f"Median: ${scf_median:,.0f}",
            legendgroup='scf'
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=[scf_mean, scf_mean],
            y=[0, 20000],
            mode='lines',
            line=dict(color='blue', dash='dot'),
            name=f"Mean: ${scf_mean:,.0f}",
            legendgroup='scf'
        ), row=1, col=1
    )

    # 6. Add full-height median/mean lines for CPS
    fig.add_trace(
        go.Scatter(
            x=[cps_median, cps_median],
            y=[0, 38000],
            mode='lines',
            line=dict(color='purple',   dash='dash'),
            name=f"Median: ${cps_median:,.0f}",
            legendgroup='cps'
        ), row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=[cps_mean, cps_mean],
            y=[0, 38000],
            mode='lines',
            line=dict(color='purple', dash='dot'),
            name=f"Mean: ${cps_mean:,.0f}",
            legendgroup='cps'
        ), row=2, col=1
    )

    # 7. Final layout tweaks
    fig.update_layout(
        title_text="Comparison of Original and Imputed Networth Distributions",
        height=PLOT_CONFIG["height"],
        width =PLOT_CONFIG["width"],
        legend_tracegroupgap=180
    )

    # 8. Zoom x-axis to drop extreme outliers
    fig.update_xaxes(range=[0, upper_limit], row=1, col=1, title_text="Networth")
    fig.update_xaxes(range=[0, upper_limit], row=2, col=1, title_text="Networth")
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)

    return fig
```

![png](./scf_to_cps_networth_distribution.png)

```python
def plot_log_transformed_distributions(
    scf_data: pd.DataFrame,
    imputed_data: pd.DataFrame,
) -> go.Figure:
    """Plot the log-transformed distribution of net worth in SCF and imputed CPS data.
    
    Args:
        scf_data: DataFrame containing SCF data.
        imputed_data: DataFrame containing imputed CPS data.

    Returns:
        Plotly figure object.
    """ 
    # Create a log transformation function that handles negative values
    def safe_log(x):
        # For negative values, take log of absolute value and negate
        # For zero, replace with a small positive value
        sign = np.sign(x)
        log_x = np.log10(np.maximum(np.abs(x), 1e-10))
        return sign * log_x

    # Create log-transformed data
    scf_log = safe_log(scf_data['networth'])
    cps_log = safe_log(imputed_data['networth'])

    # Calculate statistics for log-transformed data
    scf_log_median = np.median(scf_log)
    cps_log_median = np.median(cps_log)
    scf_log_mean = np.mean(scf_log)
    cps_log_mean = np.mean(cps_log)

    # Create a single plot with both distributions
    fig = go.Figure()

    # Add histograms for both datasets
    fig.add_trace(
        go.Histogram(
            x=scf_log,
            nbinsx=60,
            opacity=0.7,
            name="SCF Log Networth",
            marker_color='blue'
        )
    )

    fig.add_trace(
        go.Histogram(
            x=cps_log,
            nbinsx=60,
            opacity=0.7,
            name="CPS Imputed Log Networth",
            marker_color='purple'
        )
    )

    # Add vertical lines for medians
    fig.add_trace(
        go.Scatter(
            x=[scf_log_median, scf_log_median],
            y=[0, 10000],
            mode="lines",
            line=dict(color="blue", width=2, dash="dash"),
            name=f"SCF Median: ${10**scf_log_median:,.0f}"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[cps_log_median, cps_log_median],
            y=[0, 10000],
            mode="lines",
            line=dict(color="purple", width=2, dash="dash"),
            name=f"CPS Median: ${10**cps_log_median:,.0f}"
        )
    )

    # Add vertical lines for means
    fig.add_trace(
        go.Scatter(
            x=[scf_log_mean, scf_log_mean],
            y=[0, 10000],
            mode="lines",
            line=dict(color="blue", width=2, dash="dot"),
            name=f"SCF Mean: ${10**scf_log_mean:,.0f}"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[cps_log_mean, cps_log_mean],
            y=[0, 10000],
            mode="lines",
            line=dict(color="purple", width=2, dash="dot"),
            name=f"CPS Mean: ${10**cps_log_mean:,.0f}"
        )
    )

    # Update layout with improved titles and labels
    fig.update_layout(
        title="Log-Transformed Networth Distribution Comparison",
        xaxis_title="Log10 of Networth",
        yaxis_title="Frequency",
        height=PLOT_CONFIG["height"],
        width=PLOT_CONFIG["width"], 
        barmode='overlay',
        bargap=0.1,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1,
            orientation="v",
            xanchor="left",
            yanchor="top"
        )
    )

    # Add tick labels showing the actual dollar values
    tick_values = [-6, -4, -2, 0, 2, 4, 6, 8]
    tick_labels = ['$' + format(10**x if x >= 0 else -10**abs(x), ',.0f') for x in tick_values]
    fig.update_xaxes(
        tickvals=tick_values,
        ticktext=tick_labels
    )

    return fig
```

![png](./scf_to_cps_networth_log_transform.png)

The logarithmic transformation provides a clearer view of the wealth distribution across its entire range. By logarithmically scaling the data, the extreme values are compressed while expanding the visibility of differences in the lower and middle portions of the distribution.

This transformation is particularly valuable for wealth data, where values can span many orders of magnitude. The plot above, shows how closely the imputed CPS wealth distribution matches the original SCF distribution in terms of shape and central tendency. The vertical lines marking the mean and median values help gauge how these statistical properties have been preserved through the imputation process.