"""Visualization utilities for imputation results comparison.

It supports creating bar charts for comparing quantile loss
across different imputation methods and quantiles.
"""

import logging
import os
from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic import validate_call

from microimpute.config import PLOT_CONFIG, QUANTILES, VALIDATE_CONFIG

logger = logging.getLogger(__name__)


@validate_call(config=VALIDATE_CONFIG)
def plot_loss_comparison(
    loss_comparison_df: pd.DataFrame,
    quantiles: List[float] = QUANTILES,
    save_path: Optional[str] = None,
) -> go.Figure:
    """Plot a bar chart comparing quantile losses across different methods.

    Args:
        loss_comparison_df: DataFrame containing loss comparison data.
        quantiles: List of quantile values (e.g. [0.05, 0.1, ...]).
        save_path: Path to save the plot. If None, the plot is displayed.

    Returns:
        Plotly figure object

    Raises:
        ValueError: If input DataFrame is invalid or missing required columns
        RuntimeError: If plot creation or saving fails
    """
    logger.debug(
        f"Creating loss comparison plot with DataFrame of shape "
        f"{loss_comparison_df.shape}"
    )

    # Validate inputs
    required_columns = ["Method", "Imputed Variable", "Percentile", "Loss"]
    missing_columns = [
        col
        for col in required_columns
        if col not in loss_comparison_df.columns
    ]
    if missing_columns:
        logger.error(
            f"Missing required columns for plotting: {missing_columns}"
        )
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    if quantiles is None or len(quantiles) == 0:
        logger.warning(
            "Empty quantiles list provided, using default QUANTILES"
        )
        quantiles = QUANTILES

    try:
        logger.debug("Creating bar chart with plotly express")
        df_avg = loss_comparison_df[
            loss_comparison_df["Imputed Variable"] == "average"
        ]
        fig = px.bar(
            df_avg,
            x="Percentile",
            y="Loss",
            color="Method",
            barmode="group",
            title="Test Loss Across Quantiles for Different Imputation Methods",
            labels={
                "Percentile": "Percentiles",
                "Loss": "Average Test Quantile Loss",
            },
        )

        logger.debug("Updating plot layout")
        # Update layout for better appearance
        fig.update_layout(
            title_font_size=14,
            xaxis_title_font_size=12,
            yaxis_title_font_size=12,
            legend_title="Method",
            height=PLOT_CONFIG["height"],
            width=PLOT_CONFIG["width"],
        )

        # Add grid lines on y-axis
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

                # Also save as HTML for interactive viewing
                html_path = save_path.replace(".jpg", ".html")
                fig.write_html(html_path)

                logger.info(f"Plot saved to {save_path} and {html_path}")
            except Exception as e:
                logger.error(f"Error saving plot: {str(e)}")
                raise RuntimeError(
                    f"Failed to save plot to {save_path}"
                ) from e

        logger.debug("Plot creation completed successfully")
        return fig

    except Exception as e:
        logger.error(f"Error creating loss comparison plot: {str(e)}")
        raise RuntimeError(
            f"Failed to create loss comparison plot: {str(e)}"
        ) from e
