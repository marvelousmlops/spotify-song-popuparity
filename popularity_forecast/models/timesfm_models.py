"""TimesFM forecasting module for time series prediction.

This module provides a wrapper around Google's TimesFM foundation model,
offering a simple interface for time series forecasting with quantile predictions.
"""

import numpy as np
import pandas as pd
import timesfm


class TimesFMForecaster:
    """Wrapper for Google's TimesFM time series foundation model.

    This class provides a simplified interface to the TimesFM model for time series forecasting,
    with support for both TimesFM 1.0 (200M) and 2.0 (500M) models.
    """

    def __init__(
        self,
        model_version="2.0",
        backend="gpu",
        per_core_batch_size=32,
        horizon_len=128,
        context_len=2048,
    ):
        """
        Initialize a TimesFM forecaster with customizable parameters.

        Args:
            model_version (str): Version of TimesFM model to use: "1.0" or "2.0"
            backend (str): "cpu" or "gpu"
            per_core_batch_size (int): Batch size per core
            horizon_len (int): Horizon length for forecasting
            context_len (int): Context length for the model (max 512 for 1.0, max 2048 for 2.0)
        """
        self.model_version = model_version
        self.backend = backend
        self.per_core_batch_size = per_core_batch_size
        self.horizon_len = horizon_len

        # Set model parameters based on version
        if model_version == "1.0":
            self.num_layers = None  # Default for 1.0 model
            self.context_len = min(512, context_len)  # Max 512 for 1.0
            self.repo_id = "google/timesfm-1.0-200m-pytorch"
            self.use_positional_embedding = None  # Default for 1.0
        else:  # 2.0
            self.num_layers = 50  # Required for 2.0 model
            self.context_len = min(2048, context_len)  # Max 2048 for 2.0
            self.repo_id = "google/timesfm-2.0-500m-pytorch"
            self.use_positional_embedding = False  # Required for 2.0

        # Initialize model
        self.model = None

    def _init_model(self):
        """Initialize the TimesFM model if not already initialized."""
        if self.model is None:
            print(f"Initializing TimesFM {self.model_version} model...")

            # Set up model parameters
            hparams_kwargs = {
                "backend": self.backend,
                "per_core_batch_size": self.per_core_batch_size,
                "horizon_len": self.horizon_len,
                "context_len": self.context_len,
            }

            # Add version-specific parameters
            if self.model_version == "2.0":
                hparams_kwargs["num_layers"] = self.num_layers
                hparams_kwargs["use_positional_embedding"] = self.use_positional_embedding

            # Initialize the model
            self.model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(**hparams_kwargs),
                checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=self.repo_id),
            )
            print("TimesFM model initialized successfully.")

    def _determine_frequency(self, df, date_col=None):
        """
        Determine the frequency category (0, 1, or 2) based on the data.

        Args:
            df (pd.DataFrame): Input data
            date_col (str): Date column name

        Returns:
            int: Frequency category (0, 1, or 2)
        """
        if date_col and date_col in df.columns:
            # Try to infer frequency from datetime column
            try:
                # Convert to datetime if not already
                dates = pd.to_datetime(df[date_col])

                # Calculate median time difference
                diff = pd.Series(dates).diff().median()
                days = diff.days if hasattr(diff, "days") else 0

                # Determine frequency category
                if days < 1:  # Sub-daily
                    return 0  # High frequency
                elif 1 <= days <= 7:  # Daily to weekly
                    return 0  # High frequency
                elif 7 < days <= 31:  # Weekly to monthly
                    return 1  # Medium frequency
                else:  # > Monthly
                    return 2  # Low frequency
            except Exception as e:
                # Default to high frequency if inference fails
                print(f"Warning: Could not infer frequency from dates: {e}")
                return 0
        else:
            # Default to high frequency if no date column
            return 0

    def forecast(
        self, df, target_col, prediction_length=2, date_col=None, eval_mode=False, freq=None
    ):
        """
        Forecast using TimesFM model.

        Args:
            df (pd.DataFrame): Input data
            target_col (str): Column name for the target variable
            prediction_length (int): Number of steps to predict
            date_col (str, optional): Column name for the date
            eval_mode (bool): If True, use last prediction_length rows as ground truth
            freq (int, optional): Frequency category (0, 1, or 2). If None, will be inferred.

        Returns:
            predictions (dict): Dictionary containing forecast results
            ground_truth (np.ndarray or None): Ground truth if eval_mode is True, else None
            metrics (dict or None): Evaluation metrics if eval_mode is True, else None
        """
        # Initialize model if not already done
        self._init_model()

        # Handle evaluation mode
        if eval_mode:
            train_df = df.iloc[:-prediction_length].copy()
            test_df = df.iloc[-prediction_length:].copy()
            ground_truth = pd.to_numeric(test_df[target_col], errors="coerce").dropna().values
        else:
            train_df = df.copy()
            ground_truth = None

        # Extract target series
        target_series = pd.to_numeric(train_df[target_col], errors="coerce").dropna().values

        # Determine frequency if not provided
        if freq is None:
            freq = self._determine_frequency(df, date_col)
            print(f"Inferred frequency category: {freq} (0=high, 1=medium, 2=low)")

        # Prepare input for TimesFM
        forecast_input = [target_series]
        frequency_input = [freq]

        # Make prediction
        point_forecast, quantile_forecast = self.model.forecast(
            forecast_input,
            freq=frequency_input,
        )

        # Extract predictions
        predictions = {
            "mean": point_forecast[0],
            "quantiles": quantile_forecast[0] if quantile_forecast is not None else None,
            "quantile_levels": (
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                if quantile_forecast is not None
                else None
            ),
        }

        # Calculate metrics if in evaluation mode
        metrics = None
        if eval_mode and ground_truth is not None:
            # Use only the first prediction_length values from predictions
            # TimesFM has a minimum horizon length (usually 128), but we only need prediction_length
            predictions_subset = predictions["mean"][:prediction_length]

            # Calculate metrics using the subset
            mae = np.mean(np.abs(predictions_subset - ground_truth))
            rmse = np.sqrt(np.mean((predictions_subset - ground_truth) ** 2))
            mape = np.mean(np.abs((predictions_subset - ground_truth) / ground_truth)) * 100
            metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}

        return predictions, ground_truth, metrics

    def forecast_with_covariates(
        self,
        df,
        target_col,
        prediction_length=2,
        date_col=None,
        eval_mode=False,
        freq=None,
        static_cat_cols=None,
        static_num_cols=None,
        dynamic_cat_cols=None,
        dynamic_num_cols=None,
    ):
        """
        Forecast using TimesFM model with covariates.

        Args:
            df (pd.DataFrame): Input data
            target_col (str): Column name for the target variable
            prediction_length (int): Number of steps to predict
            date_col (str, optional): Column name for the date
            eval_mode (bool): If True, use last prediction_length rows as ground truth
            freq (int, optional): Frequency category (0, 1, or 2). If None, will be inferred.
            static_cat_cols (list): List of static categorical covariate column names
            static_num_cols (list): List of static numerical covariate column names
            dynamic_cat_cols (list): List of dynamic categorical covariate column names
            dynamic_num_cols (list): List of dynamic numerical covariate column names

        Returns:
            predictions (dict): Dictionary containing forecast results
            ground_truth (np.ndarray or None): Ground truth if eval_mode is True, else None
            metrics (dict or None): Evaluation metrics if eval_mode is True, else None
        """
        # This is a placeholder for the covariates functionality
        # The actual implementation would require more complex data preparation
        # and would use the forecast_with_covariates method from TimesFM

        print("Warning: Covariate support is experimental and requires additional setup.")
        print("Using standard forecast method without covariates.")

        return self.forecast(df, target_col, prediction_length, date_col, eval_mode, freq)
