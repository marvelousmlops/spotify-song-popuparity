"""Chronos forecasting module for time series prediction.

This module provides a wrapper around Amazon's Chronos forecasting model,
offering a simple interface for time series forecasting with quantile predictions.
"""

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline


class ChronosForecaster:
    """Wrapper for Amazon's Chronos time series forecasting model.

    This class provides a simplified interface to the Chronos model for time series forecasting,
    with support for quantile predictions and evaluation metrics.
    """

    def __init__(self, model_name="amazon/chronos-t5-small", device="cpu", dtype=torch.bfloat16):
        """Initialize the Chronos forecaster.

        Args:
            model_name (str): Name of the Chronos model to use.
            device (str): Device to run the model on ('cpu' or 'cuda').
            dtype (torch.dtype): Data type for model weights.
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=dtype,
        )

    def forecast(
        self, df, target_col, prediction_length=2, eval_mode=False, quantile_levels=[0.1, 0.5, 0.9]
    ):
        """Forecast future values using the Chronos model.

        Args:
            df (pd.DataFrame): Input data.
            target_col (str): Name of the target column.
            prediction_length (int): Number of steps to predict.
            eval_mode (bool): If True, use last prediction_length rows as ground truth.
            quantile_levels (list): Quantile levels for prediction intervals.

        Returns:
            predictions (dict): Dictionary containing mean and quantile predictions.
            ground_truth (np.ndarray or None): Ground truth if eval_mode is True, else None.
            metrics (dict or None): Evaluation metrics if eval_mode is True, else None.
        """
        target_series = pd.to_numeric(df[target_col], errors="coerce").dropna().values
        if eval_mode:
            context = target_series[:-prediction_length]
            ground_truth = target_series[-prediction_length:]
        else:
            context = target_series
            ground_truth = None

        # Create tensor directly from the context data
        context_tensor = torch.tensor(context, dtype=torch.float32)

        # Use predict_quantiles instead of predict
        quantiles, mean = self.pipeline.predict_quantiles(
            context=context_tensor,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
        )

        # Convert tensors to numpy arrays
        mean_np = mean.detach().cpu().numpy()
        quantiles_np = quantiles.detach().cpu().numpy()

        # Create predictions dictionary
        predictions = {
            "mean": mean_np,
            "quantiles": quantiles_np,
            "quantile_levels": quantile_levels,
        }

        metrics = None
        if eval_mode and ground_truth is not None:
            # Use mean predictions for metrics calculation
            mae = np.mean(np.abs(mean_np[0] - ground_truth))
            rmse = np.sqrt(np.mean((mean_np[0] - ground_truth) ** 2))
            mape = np.mean(np.abs((mean_np[0] - ground_truth) / ground_truth)) * 100
            metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}

        return predictions, ground_truth, metrics
