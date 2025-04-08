"""
Logging utilities for the MicroImpute package.

This module provides centralized logging configuration and utilities
for consistent error handling across the codebase.
"""

import logging
import sys
from typing import Optional

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Define logging levels for different environments
DEBUG_LEVEL = logging.DEBUG
INFO_LEVEL = logging.INFO
WARN_LEVEL = logging.WARNING
ERROR_LEVEL = logging.ERROR


def configure_logging(
    level: int = INFO_LEVEL,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    log_file: Optional[str] = None,
) -> None:
    """Configure global logging settings.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Format string for log messages
        date_format: Format string for timestamps
        log_file: Optional file path to write logs
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
