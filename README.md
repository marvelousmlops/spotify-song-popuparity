# Spotify Popularity Forecasting

A comprehensive toolkit for time series forecasting of Spotify song popularity using multiple state-of-the-art models. This project provides a unified interface for running forecasts using Chronos T5, Prophet, and TimesFM models, making it easy to compare their performance on song popularity prediction tasks.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Organization](#data-organization)
- [Usage](#usage)
  - [Collecting Song Popularity Data](#collecting-song-popularity-data)
  - [Running Forecasts](#running-forecasts)
  - [Model Options](#model-options)
- [Model Comparison](#model-comparison)
- [Conclusion](#conclusion)

## Features

- Support for three powerful time series forecasting models:
  - **Chronos T5**: Amazon's time series forecasting model based on T5 transformer architecture
  - **Prophet**: Facebook's decomposable time series forecasting model
  - **TimesFM**: Google's Time Series Foundation Model
- Spotify API integration for collecting song popularity data
- Historical popularity data collection and simulation
- Unified interface for comparing multiple forecasting models
- Automatic evaluation metrics calculation (MAE, RMSE, MAPE)
- Interactive visualizations with Plotly
- Flexible data handling with automatic preprocessing

## Installation

### Prerequisites

- Python 3.11+
- uv (recommended for dependency management)

### Setup with uv

```bash
# Clone the repository
git clone https://github.com/yourusername/spotify-song-popularity.git
cd spotify-song-popularity

# Create a virtual environment
uv venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
uv sync
```

### Spotify API Credentials

To use the Spotify API integration, you need to set up your credentials:

1. Create a Spotify Developer account at [developer.spotify.com](https://developer.spotify.com/)
2. Create a new application to get your Client ID and Client Secret
3. Create a `.env` file in the project root with the following content:

```
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

## Project Structure

```
popularity_forecast/
├── models/                  # Model implementations
│   ├── chronos_models.py    # Chronos T5 model wrapper
│   ├── prophet_models.py    # Prophet model wrapper
│   └── timesfm_models.py    # TimesFM model wrapper
├── utils/
│   └── spotify_api.py       # Spotify API integration
├── simple_forecast.py       # Main forecasting script with visualization
├── song_popularity_history.py # Script to collect song popularity data
└── model_comparison.py      # Script for comparing model performance
```

### Key Components

- **Spotify API Integration**: The `spotify_api.py` module provides a clean interface to the Spotify Web API for retrieving song metadata and popularity scores.

- **Model Wrappers**: Each forecasting model has a dedicated wrapper class that provides a consistent interface for training and prediction.

- **Data Collection**: The `song_popularity_history.py` script collects current popularity scores and generates historical data.

- **Forecasting**: The `simple_forecast.py` script runs multiple models on the same data and compares their performance.

- **Visualization**: Interactive Plotly visualizations show predictions alongside actual popularity scores.

### Manual Setup with pip

```bash
# Clone the repository
git clone https://github.com/yourusername/spotify-song-popularity.git
cd spotify-song-popularity

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install python-dotenv pandas numpy plotly scikit-learn
pip install git+https://github.com/amazon-science/chronos-forecasting.git
pip install prophet timesfm
```

## Data Organization

The project uses the following data directory structure:

```
data/
├── spotify_popularity/
│   ├── output/    # CSV files with song popularity history
│   └── plots/     # Interactive HTML visualizations
```

## Usage

### Collecting Song Popularity Data

To collect popularity data for a song:

```bash
python -m popularity_forecast.song_popularity_history "Song Name" --artist "Artist Name" --days 90
```

This will:
1. Search for the song on Spotify
2. Retrieve its current popularity score
3. Generate simulated historical data for the past 90 days
4. Save the data to a CSV file in `data/spotify_popularity/output/`

### Running Forecasts

To run forecasts using all available models:

```bash
python -m popularity_forecast.simple_forecast "Song Name" --days 10
```

This will:
1. Load the song's popularity history
2. Use the last 10 days as a test set
3. Train each model on the remaining data
4. Generate predictions for the test period
5. Calculate performance metrics
6. Create an interactive visualization

### Model Options

You can run specific models using the `--models` flag:

```bash
# Run only Chronos T5
python -m popularity_forecast.simple_forecast "Song Name" --models chronos

# Run Prophet and TimesFM
python -m popularity_forecast.simple_forecast "Song Name" --models prophet timesfm

# Run all models (default)
python -m popularity_forecast.simple_forecast "Song Name" --models all
```

## Model Comparison

The project includes a comprehensive framework for comparing the performance of different forecasting models on song popularity data.

### Performance Metrics

Models are evaluated using standard time series metrics:

- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual values
- **RMSE (Root Mean Squared Error)**: Square root of the average squared differences
- **MAPE (Mean Absolute Percentage Error)**: Average percentage difference

### Example Results

Here are some example results from running the models on "Buffalo Stance" by Neneh Cherry:

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| Chronos T5 | 3.68 | 5.19 | 8.70% |
| Prophet | 5.88 | 7.52 | 14.01% |
| TimesFM | 4.63 | 7.63 | 9.84% |

In this case, Chronos T5 outperformed the other models across all metrics.

### Visualization

The forecasting results are visualized in an interactive HTML report that includes:

1. Time series plot showing training data, actual values, and predictions
2. Performance metrics table
3. Hover information for detailed data points

The visualization makes it easy to compare model performance and identify patterns in the data.

## Conclusion

This project demonstrates the application of state-of-the-art time series forecasting models to predict Spotify song popularity. By comparing Chronos T5, Prophet, and TimesFM on the same data, we can evaluate their strengths and weaknesses for this specific use case.

The results consistently show that Chronos T5 tends to outperform the other models for song popularity prediction, likely due to its transformer-based architecture that can capture complex patterns in the data.

Feel free to experiment with different songs and model configurations to see how the predictions vary. The interactive visualizations make it easy to compare model performance and identify interesting patterns in song popularity trends.
