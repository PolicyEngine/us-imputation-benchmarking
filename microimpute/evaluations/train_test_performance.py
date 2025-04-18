"""Train-test performance visualization for imputation models.

This module provides functions for visualizing the performance of imputation models
on both training and test datasets. It helps identify overfitting and compare
model performance across different quantiles.
"""

import logging
import os
from typing import Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from pydantic import validate_call

from microimpute.config import PLOT_CONFIG, VALIDATE_CONFIG

logger = logging.getLogger(__name__)


@validate_call(config=VALIDATE_CONFIG)
def plot_train_test_performance(
    results: pd.DataFrame,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (PLOT_CONFIG["width"], PLOT_CONFIG["height"]),
) -> go.Figure:
    """Plot the performance comparison between training and testing
        sets across quantiles.

    Args:
        results: DataFrame with train and test rows, quantiles
            as columns, and loss values.
        title: Custom title for the plot. If None, a default title is used.
        save_path: Path to save the plot. If None, the plot is displayed.
        figsize: Figure size as (width, height) in pixels.

    Returns:
        Plotly figure object

    Raises:
        ValueError: If results DataFrame is invalid or missing required indices
        RuntimeError: If plot creation or saving fails
    """
    logger.debug(
        f"Creating train-test performance plot from results shape {results.shape}"
    )

    # Validate inputs
    required_indices = ["train", "test"]
    available_indices = results.index.tolist()
    missing_indices = [
        idx for idx in required_indices if idx not in available_indices
    ]

    if missing_indices:
        logger.warning(
            f"Missing indices in results DataFrame: {missing_indices}"
        )
        logger.info(f"Available indices: {available_indices}")

    try:
        # Convert column names to strings if they are not already
        logger.debug("Converting column names to strings")
        results.columns = [str(col) for col in results.columns]

        # Default title if none provided
        if title is None:
            title = "Average Quantile Loss: Train vs Test"
            logger.debug(f"Using default title: '{title}'")
        else:
            logger.debug(f"Using custom title: '{title}'")

        # Create a new figure
        logger.debug("Creating Plotly figure")
        fig = go.Figure()

        # Add bars for training data if present
        if "train" in results.index:
            logger.debug("Adding training data bars")
            fig.add_trace(
                go.Bar(
                    x=results.columns,
                    y=results.loc["train"],
                    name="Train",
                    marker_color="rgba(0, 128, 0, 0.7)",  # green with transparency
                )
            )

        # Add bars for test data if present
        if "test" in results.index:
            logger.debug("Adding test data bars")
            fig.add_trace(
                go.Bar(
                    x=results.columns,
                    y=results.loc["test"],
                    name="Test",
                    marker_color="rgba(255, 0, 0, 0.7)",  # red with transparency
                )
            )

        # Update layout with title, axis labels, etc.
        logger.debug("Updating plot layout")
        fig.update_layout(
            title=title,
            xaxis_title="Quantile",
            yaxis_title="Average Quantile Loss",
            barmode="group",
            width=figsize[0],
            height=figsize[1],
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            margin=dict(l=50, r=50, t=80, b=50),
        )

        # Add grid lines
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)"
        )

        # Save or show the plot
        if save_path:
            try:
                logger.info(f"Saving plot to {save_path}")

                # Ensure directory exists
                save_dir = os.path.dirname(save_path)
                if save_dir and not os.path.exists(save_dir):
                    logger.debug(f"Creating directory: {save_dir}")
                    os.makedirs(save_dir, exist_ok=True)

                # Save as image
                fig.write_image(save_path)

                # Also save HTML version for interactive viewing
                html_path = save_path.replace(".jpg", ".html")
                fig.write_html(html_path)

                logger.info(f"Plot saved to {save_path} and {html_path}")
            except Exception as e:
                logger.error(f"Error saving train-test plot: {str(e)}")
                raise RuntimeError(
                    f"Failed to save plot to {save_path}"
                ) from e

        logger.debug("Train-test performance plot created successfully")
        return fig

    except Exception as e:
        logger.error(f"Error creating train-test performance plot: {str(e)}")
        raise RuntimeError(
            f"Failed to create train-test performance plot: {str(e)}"
        ) from e
