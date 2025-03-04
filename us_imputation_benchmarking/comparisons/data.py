import microdf as mdf
import scf
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Union


VALID_YEARS = [
    1989,
    1992,
    1995,
    1998,
    2001,
    2004,
    2007,
    2010,
    2013,
    2016,
    2019,
]

def scf_url(year: int) -> str:
    """ Returns the URL of the SCF summary microdata zip file for a year.

    :param year: Year of SCF summary microdata to retrieve.
    :type year: int
    :return: URL of summary microdata zip file for the given year.
    :rtype: str
    """
    assert year in VALID_YEARS, "The SCF is not available for " + str(year)
    return (
        "https://www.federalreserve.gov/econres/files/scfp"
        + str(year)
        + "s.zip"
    )


def load_single_scf(year: int, columns: list) -> pd.DataFrame:
    """ Loads SCF summary microdata for a given year and set of columns.

    :param year: Year of SCF summary microdata to retrieve.
    :type year: int
    :param columns: List of columns. The weight column `wgt` is always
        returned. Defaults to all columns in the summary dataset.
    :type columns: list
    :return: SCF summary microdata for the given year.
    :rtype: pd.DataFrame
    """
    # Add wgt to all returns.
    if columns is not None:
        columns = list(set(columns) | set(["wgt"]))
    return mdf.read_stata_zip(scf_url(year), columns=columns)


def load(
    years: list = VALID_YEARS,
    columns: list = None,
    as_microdataframe: bool = False,
) -> Union[pd.DataFrame, mdf.MicroDataFrame]:
    """ Loads SCF summary microdata for a set of years and columns.

    :param years: Year(s) to load SCF data for. Can be a list or single number.
        Defaults to all available years, starting with 1989.
    :type years: list
    :param columns: List of columns. The weight column `wgt` is always returned.
    :type columns: list
    :param as_microdataframe: Whether to return as a MicroDataFrame with
        weight set, defaults to False.
    :type as_microdataframe: bool
    :return: SCF summary microdata for the set of years.
    :rtype: Union[pd.DataFrame, mdf.MicroDataFrame]
    """
    # Make cols a list if a single column is passed.
    if columns is not None:
        columns = mdf.listify(columns)
    # If years is a single year rather than a list, don't use a loop.
    if isinstance(years, int):
        res = load_single_scf(years, columns)
    # Otherwise append to a list within a loop, and concatenate.
    else:
        scfs = []
        for year in years:
            tmp = load_single_scf(year, columns)
            tmp["year"] = year
            scfs.append(tmp)
        res = pd.concat(scfs)
    # Return as a MicroDataFrame or DataFrame.
    if as_microdataframe:
        return mdf.MicroDataFrame(res, weights="wgt")
    return res

def preprocess_data(full_data=False):
    data = load([VALID_YEARS[-1]])

    # predictors shared with cps data

    PREDICTORS = ["hhsex",      # sex of head of household
                "age",          # age of respondent
                "married",      # marital status of respondent
                "kids",         # number of children in household
                "educ",         # highest level of education
                "race",         # race of respondent 
                "income",       # total annual income of household  
                "wageinc",      # income from wages and salaries
                "bussefarminc", # income from business, self-employment or farm
                "intdivinc",    # income from interest and dividends
                "ssretinc",     # income from social security and retirement accounts
                "lf",           # labor force status
                ]   

    IMPUTED_VARIABLES = ["networth"] # some property also captured in cps data (HPROP_VAL)

    data = data[PREDICTORS + IMPUTED_VARIABLES]
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data = (data - mean) / std

    if full_data:
        return data, PREDICTORS, IMPUTED_VARIABLES
    else:
        X, test_X = train_test_split(data, test_size=0.2, train_size=0.8, random_state=42)
        return X, test_X, PREDICTORS, IMPUTED_VARIABLES