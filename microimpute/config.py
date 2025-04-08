"""
Configuration module for MicroImpute.

This module centralizes all constants and configuration parameters used across
the package.
"""

from typing import Any, Dict, List

from pydantic import ConfigDict

# Define a configuration for pydantic validation that allows
# arbitrary types like pd.DataFrame
VALIDATE_CONFIG = ConfigDict(arbitrary_types_allowed=True)

# Data configuration
VALID_YEARS: List[int] = [
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

TRAIN_SIZE: float = 0.8
TEST_SIZE: float = 0.2

# Analysis configuration
QUANTILES: List[float] = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]

# Random state for reproducibility
RANDOM_STATE: int = 42

# Model parameters
DEFAULT_MODEL_PARAMS: Dict[str, Dict[str, Any]] = {
    "qrf": {},
    "quantreg": {},
    "ols": {},
    "matching": {},
}

# Plotting configuration
PLOT_CONFIG: Dict[str, Any] = {
    "width": 1000,
    "height": 600,
    "colors": {},
}
