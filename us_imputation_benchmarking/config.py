"""
Configuration module for US imputation benchmarking.

This module centralizes all constants and configuration parameters used across
the package.
"""
from typing import Dict, List, Any


# Data configuration
VALID_YEARS: List[int] = [1989, 1992, 1995, 1998, 2001, 2004, 2007, 2010, 2013, 2016, 2019]

# Analysis configuration
QUANTILES: List[float] = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]

# Random state for reproducibility
RANDOM_STATE: int = 42

# Model parameters
DEFAULT_MODEL_PARAMS: Dict[str, Dict[str, Any]] = {
    "qrf": {},
    "quantreg": {},
    "ols": {},
    "matching": {}
}

# Plotting configuration
PLOT_CONFIG: Dict[str, Any] = {
    "width": 1000,
    "height": 600,
    "colors": {},
}
