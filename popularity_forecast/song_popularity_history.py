#!/usr/bin/env python
"""
Song Popularity History Generator

This script generates a CSV file with popularity scores for a given song over the past 90 days.
Since Spotify API only provides current popularity scores, this script:
1. Checks if we have historical data for the song
2. If not, simulates historical data based on the current popularity score with realistic variations
"""

import os
import sys
import csv
import random
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from popularity_forecast.utils.spotify_api import SpotifyAPI, load_credentials_from_env

def load_credentials():
    """Load Spotify API credentials from environment variables."""
    return load_credentials_from_env()

def get_song_id(spotify, song_name, artist_name=None):
    """
    Get Spotify track ID for a song.
    
    Args:
        spotify: SpotifyAPI instance
        song_name: Name of the song
        artist_name: Optional artist name to refine search
        
    Returns:
        dict: Track data including ID and current popularity
    """
    track_data = spotify.search_track(song_name, artist_name)
    if not track_data:
        print(f"No track found for: {song_name}")
        if artist_name:
            print(f"Try searching without specifying the artist: {artist_name}")
        return None
    
    print(f"Found track: {track_data['name']} by {track_data['artist']}")
    print(f"Current popularity score: {track_data['popularity']}/100")
    
    return track_data

def check_existing_history(track_id):
    """
    Check if we have existing history data for this track.
    
    Args:
        track_id: Spotify track ID
        
    Returns:
        pd.DataFrame or None: Existing history data if available
    """
    history_dir = "data/spotify_popularity/history"
    os.makedirs(history_dir, exist_ok=True)
    
    history_file = f"{history_dir}/{track_id}.csv"
    if os.path.exists(history_file):
        try:
            df = pd.read_csv(history_file)
            if len(df) > 0:
                print(f"Found existing history data with {len(df)} entries.")
                return df
        except Exception as e:
            print(f"Error reading existing history: {str(e)}")
    
    return None

def generate_simulated_history(track_data, days=90):
    """
    Generate simulated popularity history based on current score.
    
    Args:
        track_data: Track data including current popularity
        days: Number of days of history to generate
        
    Returns:
        pd.DataFrame: Simulated history data
    """
    current_popularity = track_data['popularity']
    current_date = datetime.now()
    
    # Create date range for the past 90 days
    dates = [(current_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
    dates.reverse()  # Oldest to newest
    
    # Generate realistic popularity variations
    # Use a random walk with constraints to create realistic variations
    # Convert track ID to a numeric seed by summing character codes
    seed = sum(ord(c) for c in track_data['id']) % 10000
    np.random.seed(seed)  # Use track ID as seed for reproducibility
    
    # Base popularity around current value with some variation
    base_popularity = max(min(current_popularity, 95), 5)  # Constrain between 5-95 to allow movement
    
    # Generate random walk
    changes = np.random.normal(0, 1.5, days)  # Small daily changes
    
    # Add some trend and seasonality
    trend = np.linspace(-5, 5, days)  # Slight trend
    seasonality = 3 * np.sin(np.linspace(0, 6*np.pi, days))  # Weekly pattern
    
    # Combine components
    raw_popularity = base_popularity + np.cumsum(changes) + trend + seasonality
    
    # Ensure values stay within 0-100 range and end at current popularity
    popularity_scores = np.clip(raw_popularity, 0, 100).tolist()
    
    # Adjust the last value to match current popularity
    popularity_scores[-1] = current_popularity
    
    # Round to integers
    popularity_scores = [round(score) for score in popularity_scores]
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'popularity': popularity_scores,
        'id': track_data['id'],
        'name': track_data['name'],
        'artist': track_data['artist']
    })
    
    return df

def save_history_csv(df, track_id, song_name):
    """
    Save history data to CSV files.
    
    Args:
        df: DataFrame with history data
        track_id: Spotify track ID
        song_name: Name of the song for the output filename
        
    Returns:
        tuple: Paths to the saved files
    """
    history_dir = "data/spotify_popularity/history"
    output_dir = "data/spotify_popularity/output"
    os.makedirs(history_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the base song name (remove any remaster/version info)
    base_song_name = song_name.split(' - ')[0].strip()
    
    # Clean song name for filename
    clean_name = "".join(c if c.isalnum() else "_" for c in base_song_name)
    
    # Save to history file (for future use)
    history_file = f"{history_dir}/{track_id}.csv"
    df.to_csv(history_file, index=False)
    
    # Save to named output file
    output_file = f"{output_dir}/{clean_name}.csv"
    df.to_csv(output_file, index=False)
    
    return history_file, output_file

def main():
    """Main function to generate song popularity history."""
    parser = argparse.ArgumentParser(description="Generate song popularity history CSV")
    parser.add_argument("song_name", help="Name of the song")
    parser.add_argument("--artist", help="Artist name (optional)")
    parser.add_argument("--days", type=int, default=90, help="Number of days of history (default: 90)")
    args = parser.parse_args()
    
    # Load credentials and initialize Spotify API
    client_id, client_secret = load_credentials()
    spotify = SpotifyAPI(client_id, client_secret)
    
    # Get song ID
    track_data = get_song_id(spotify, args.song_name, args.artist)
    if not track_data:
        return
    
    track_id = track_data['id']
    
    # Check for existing history
    history_df = check_existing_history(track_id)
    
    # If no history or not enough data points, generate simulated history
    if history_df is None or len(history_df) < args.days:
        print(f"Generating simulated popularity history for the past {args.days} days...")
        history_df = generate_simulated_history(track_data, args.days)
        print("Note: This is simulated data based on the current popularity score.")
    
    # Save to CSV files
    history_file, output_file = save_history_csv(history_df, track_id, track_data['name'])
    
    print(f"\nPopularity history saved to: {output_file}")
    print(f"Data points: {len(history_df)}")
    print(f"Date range: {history_df['date'].iloc[0]} to {history_df['date'].iloc[-1]}")
    print(f"Popularity range: {history_df['popularity'].min()} to {history_df['popularity'].max()}")
    
    # Display sample of the data
    print("\nSample data (first 5 and last 5 entries):")
    pd.set_option('display.max_columns', None)
    print(pd.concat([history_df.head(), history_df.tail()]))

if __name__ == "__main__":
    main()
