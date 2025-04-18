"""
Configuration module for MicroImpute.

This module centralizes all constants and configuration parameters used across
the package.
"""

from typing import Any, Dict, List

import numpy as np
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
QUANTILES: List[float] = [round(q, 2) for q in np.arange(0.05, 1.00, 0.05)]

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
    "width": 750,
    "height": 600,
    "colors": {},
}
