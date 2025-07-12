"""
Spotify API integration module for fetching song popularity data.
This module provides functionality to authenticate with the Spotify API
and retrieve popularity scores for tracks.
"""

import os
import json
import time
import base64
import requests
import dotenv
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import pandas as pd

# Load environment variables from .env file
dotenv.load_dotenv()

def load_credentials_from_env():
    """
    Load Spotify API credentials from environment variables or use hardcoded defaults.
    
    Returns:
        tuple: (client_id, client_secret)
    """
    # Try different possible environment variable names
    client_id = os.environ.get('SPOTIFY_CLIENT_ID') or os.environ.get('client_id')
    client_secret = os.environ.get('SPOTIFY_CLIENT_SECRET') or os.environ.get('client_secret')

    return client_id, client_secret

class SpotifyAPI:
    """
    A class to interact with the Spotify Web API for retrieving track information
    including popularity scores.
    """
    
    def __init__(self, client_id: str, client_secret: str):
        """
        Initialize the SpotifyAPI with client credentials.
        
        Args:
            client_id: Spotify API client ID
            client_secret: Spotify API client secret
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None
        self.token_expiry = 0
        
    def _get_auth_token(self) -> str:
        """
        Get an authentication token from Spotify using client credentials.
        
        Returns:
            str: Authentication token
        """
        if self.token and time.time() < self.token_expiry:
            return self.token
            
        auth_string = f"{self.client_id}:{self.client_secret}"
        auth_bytes = auth_string.encode("utf-8")
        auth_base64 = base64.b64encode(auth_bytes).decode("utf-8")
        
        url = "https://accounts.spotify.com/api/token"
        headers = {
            "Authorization": f"Basic {auth_base64}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {"grant_type": "client_credentials"}
        
        response = requests.post(url, headers=headers, data=data)
        json_result = response.json()
        
        if "error" in json_result:
            raise Exception(f"Error getting auth token: {json_result['error_description']}")
            
        self.token = json_result["access_token"]
        self.token_expiry = time.time() + json_result["expires_in"] - 60  # Buffer of 60 seconds
        
        return self.token
    
    def search_track(self, track_name: str, artist_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for a track on Spotify.
        
        Args:
            track_name: Name of the track to search for
            artist_name: Optional artist name to refine search
            
        Returns:
            Dict: Track information including popularity score
        """
        token = self._get_auth_token()
        
        query = f"track:{track_name}"
        if artist_name:
            query += f" artist:{artist_name}"
            
        url = "https://api.spotify.com/v1/search"
        headers = {
            "Authorization": f"Bearer {token}"
        }
        params = {
            "q": query,
            "type": "track",
            "limit": 1
        }
        
        response = requests.get(url, headers=headers, params=params)
        json_result = response.json()
        
        if "error" in json_result:
            raise Exception(f"Error searching for track: {json_result['error']['message']}")
            
        if not json_result["tracks"]["items"]:
            return None
            
        track = json_result["tracks"]["items"][0]
        
        return {
            "id": track["id"],
            "name": track["name"],
            "artist": track["artists"][0]["name"],
            "popularity": track["popularity"],
            "album": track["album"]["name"],
            "release_date": track["album"]["release_date"],
            "duration_ms": track["duration_ms"],
            "explicit": track["explicit"],
            "uri": track["uri"]
        }
    
    def get_track_by_id(self, track_id: str) -> Dict[str, Any]:
        """
        Get track information by Spotify track ID.
        
        Args:
            track_id: Spotify track ID
            
        Returns:
            Dict: Track information including popularity score
        """
        token = self._get_auth_token()
        
        url = f"https://api.spotify.com/v1/tracks/{track_id}"
        headers = {
            "Authorization": f"Bearer {token}"
        }
        
        response = requests.get(url, headers=headers)
        json_result = response.json()
        
        if "error" in json_result:
            raise Exception(f"Error getting track: {json_result['error']['message']}")
            
        return {
            "id": json_result["id"],
            "name": json_result["name"],
            "artist": json_result["artists"][0]["name"],
            "popularity": json_result["popularity"],
            "album": json_result["album"]["name"],
            "release_date": json_result["album"]["release_date"],
            "duration_ms": json_result["duration_ms"],
            "explicit": json_result["explicit"],
            "uri": json_result["uri"]
        }
    
    def get_audio_features(self, track_id: str) -> Dict[str, Any]:
        """
        Get audio features for a track.
        
        Args:
            track_id: Spotify track ID
            
        Returns:
            Dict: Audio features for the track
        """
        token = self._get_auth_token()
        
        url = f"https://api.spotify.com/v1/audio-features/{track_id}"
        headers = {
            "Authorization": f"Bearer {token}"
        }
        
        response = requests.get(url, headers=headers)
        json_result = response.json()
        
        if "error" in json_result:
            raise Exception(f"Error getting audio features: {json_result['error']['message']}")
            
        return json_result
    
    def get_tracks_popularity(self, track_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get popularity scores for multiple tracks.
        
        Args:
            track_ids: List of Spotify track IDs (max 50)
            
        Returns:
            List[Dict]: List of track information including popularity scores
        """
        if len(track_ids) > 50:
            raise ValueError("Maximum of 50 track IDs allowed per request")
            
        token = self._get_auth_token()
        
        url = "https://api.spotify.com/v1/tracks"
        headers = {
            "Authorization": f"Bearer {token}"
        }
        params = {
            "ids": ",".join(track_ids)
        }
        
        response = requests.get(url, headers=headers, params=params)
        json_result = response.json()
        
        if "error" in json_result:
            raise Exception(f"Error getting tracks: {json_result['error']['message']}")
            
        tracks = []
        for track in json_result["tracks"]:
            tracks.append({
                "id": track["id"],
                "name": track["name"],
                "artist": track["artists"][0]["name"],
                "popularity": track["popularity"],
                "album": track["album"]["name"],
                "release_date": track["album"]["release_date"]
            })
            
        return tracks

    def track_popularity_history(self, track_id: str, storage_path: str) -> pd.DataFrame:
        """
        Track and store popularity history for a specific track.
        
        Args:
            track_id: Spotify track ID
            storage_path: Path to store the history data
            
        Returns:
            pd.DataFrame: Historical popularity data
        """
        # Create storage directory if it doesn't exist
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        
        # Get current track data
        track_data = self.get_track_by_id(track_id)
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Create history entry
        history_entry = {
            "date": current_date,
            "popularity": track_data["popularity"],
            "id": track_id,
            "name": track_data["name"],
            "artist": track_data["artist"]
        }
        
        # Load existing history or create new
        history_df = None
        if os.path.exists(storage_path):
            try:
                history_df = pd.read_csv(storage_path)
            except Exception:
                history_df = pd.DataFrame(columns=["date", "popularity", "id", "name", "artist"])
        else:
            history_df = pd.DataFrame(columns=["date", "popularity", "id", "name", "artist"])
        
        # Append new entry
        history_df = pd.concat([history_df, pd.DataFrame([history_entry])], ignore_index=True)
        
        # Save updated history
        history_df.to_csv(storage_path, index=False)
        
        return history_df
