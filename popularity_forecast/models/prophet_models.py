"""Prophet forecasting module for time series prediction.

This module provides a wrapper around Facebook's Prophet forecasting model,
offering a simple interface for time series forecasting with prediction intervals.
"""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet


class ProphetForecaster:
    """Wrapper for Facebook's Prophet time series forecasting model.

    This class provides a simplified interface to the Prophet model for time series forecasting,
    with support for prediction intervals and visualization.
    """

    def __init__(
        self, seasonality_mode="multiplicative", interval_width=0.95, changepoint_prior_scale=0.05
    ):
        """
        Initialize a Prophet forecaster with customizable parameters.

        Args:
            seasonality_mode (str): 'multiplicative' or 'additive'
            interval_width (float): Confidence interval width (0 to 1)
            changepoint_prior_scale (float): Flexibility of the trend
        """
        self.seasonality_mode = seasonality_mode
        self.interval_width = interval_width
        self.changepoint_prior_scale = changepoint_prior_scale
        self.model = None

    def _prepare_data(self, df, target_col, date_col=None):
        """
        Prepare data for Prophet model.

        Args:
            df (pd.DataFrame): Input data
            target_col (str): Column name for the target variable
            date_col (str, optional): Column name for the date. If None, try to infer from data

        Returns:
            pd.DataFrame: DataFrame with 'ds' and 'y' columns
        """
        # Make a copy to avoid modifying the original dataframe
        prophet_df = df.copy()

        # Set the date column
        if date_col and date_col in prophet_df.columns:
            date_column = date_col
        else:
            # Try to infer date column
            date_candidates = [
                col
                for col in prophet_df.columns
                if any(
                    date_keyword in col.lower()
                    for date_keyword in ["date", "time", "day", "month", "year"]
                )
            ]

            if date_candidates:
                date_column = date_candidates[0]
            else:
                # If no date column found, create one with sequential dates
                prophet_df["date"] = pd.date_range(
                    start=datetime.now().replace(day=1), periods=len(prophet_df), freq="MS"
                )
                date_column = "date"

        # Rename columns to match Prophet requirements
        prophet_df = prophet_df.rename(columns={date_column: "ds", target_col: "y"})

        # Ensure ds is in datetime format
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

        # Select only required columns
        prophet_df = prophet_df[["ds", "y"]]

        return prophet_df

    def forecast(self, df, target_col, prediction_length=2, date_col=None, eval_mode=False):
        """
        Forecast using Prophet model.

        Args:
            df (pd.DataFrame): Input data
            target_col (str): Column name for the target variable
            prediction_length (int): Number of steps to predict
            date_col (str, optional): Column name for the date
            eval_mode (bool): If True, use last prediction_length rows as ground truth

        Returns:
            predictions (dict): Dictionary containing forecast results
            ground_truth (np.ndarray or None): Ground truth if eval_mode is True, else None
            metrics (dict or None): Evaluation metrics if eval_mode is True, else None
        """
        # Handle evaluation mode
        if eval_mode:
            train_df = df.iloc[:-prediction_length].copy()
            test_df = df.iloc[-prediction_length:].copy()
            ground_truth = pd.to_numeric(test_df[target_col], errors="coerce").dropna().values
        else:
            train_df = df.copy()
            ground_truth = None

        # Prepare data for Prophet
        prophet_df = self._prepare_data(train_df, target_col, date_col)

        # Create and fit the model
        self.model = Prophet(
            seasonality_mode=self.seasonality_mode,
            interval_width=self.interval_width,
            changepoint_prior_scale=self.changepoint_prior_scale,
        )

        self.model.fit(prophet_df)

        # Create future dataframe
        freq = self._infer_frequency(prophet_df)
        future = self.model.make_future_dataframe(periods=prediction_length, freq=freq)

        # Make predictions
        forecast = self.model.predict(future)

        # Extract predictions for the forecast period
        predictions = {
            "mean": forecast["yhat"].values[-prediction_length:],
            "lower": forecast["yhat_lower"].values[-prediction_length:],
            "upper": forecast["yhat_upper"].values[-prediction_length:],
            "dates": forecast["ds"].values[-prediction_length:],
        }

        # Calculate metrics if in evaluation mode
        metrics = None
        if eval_mode and ground_truth is not None:
            mae = np.mean(np.abs(predictions["mean"] - ground_truth))
            rmse = np.sqrt(np.mean((predictions["mean"] - ground_truth) ** 2))
            mape = np.mean(np.abs((predictions["mean"] - ground_truth) / ground_truth)) * 100
            metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}

        return predictions, ground_truth, metrics

    def _infer_frequency(self, df):
        """
        Infer the frequency of the time series data.

        Args:
            df (pd.DataFrame): DataFrame with 'ds' column

        Returns:
            str: Pandas frequency string
        """
        if len(df) <= 1:
            return "D"  # Default to daily if not enough data

        # Sort by date
        df = df.sort_values("ds")

        # Calculate differences between consecutive dates
        date_diffs = df["ds"].diff().dropna()

        # Get the most common difference
        if date_diffs.empty:
            return "D"

        median_diff = date_diffs.median()
        days = median_diff.days

        if days == 0:
            return "H"  # Hourly
        elif 1 <= days <= 2:
            return "D"  # Daily
        elif 6 <= days <= 8:
            return "W"  # Weekly
        elif 28 <= days <= 31:
            return "MS"  # Monthly (start)
        elif 90 <= days <= 92:
            return "QS"  # Quarterly (start)
        elif 365 <= days <= 366:
            return "YS"  # Yearly (start)
        else:
            return "D"  # Default to daily

    def plot_forecast(self, df, target_col, prediction_length=2, date_col=None, figsize=(12, 6)):
        """
        Plot the forecast results.

        Args:
            df (pd.DataFrame): Input data
            target_col (str): Column name for the target variable
            prediction_length (int): Number of steps to predict
            date_col (str, optional): Column name for the date
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Prepare data and fit model if not already done
        if self.model is None:
            prophet_df = self._prepare_data(df, target_col, date_col)
            self.model = Prophet(
                seasonality_mode=self.seasonality_mode,
                interval_width=self.interval_width,
                changepoint_prior_scale=self.changepoint_prior_scale,
            )
            self.model.fit(prophet_df)

        # Create future dataframe
        freq = self._infer_frequency(self._prepare_data(df, target_col, date_col))
        future = self.model.make_future_dataframe(periods=prediction_length, freq=freq)

        # Make predictions
        forecast = self.model.predict(future)

        # Create plot
        plt.figure(figsize=figsize)
        fig = self.model.plot(forecast)
        plt.title("Prophet Forecast")
        plt.xlabel("Date")
        plt.ylabel(target_col)

        # Add components plot
        plt.figure(figsize=figsize)
        self.model.plot_components(forecast)
        plt.tight_layout()

        return fig
