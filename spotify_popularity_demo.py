#!/usr/bin/env python
"""
Spotify Popularity Score Demo

This script demonstrates how to use the Spotify API to get popularity scores for tracks.
"""

import os
from popularity_forecast.utils.spotify_api import SpotifyAPI

def main():
    """Main function to demonstrate Spotify API popularity score retrieval."""
    # Load credentials
    client_id = "ae9a3b52c044417f8e3a04068d82e13b"
    client_secret = "aead7f726fec4dcd81fc9199d393b6f5"
    
    # Initialize Spotify API
    spotify = SpotifyAPI(client_id, client_secret)
    
    # Search for a track
    print("Searching for 'Bohemian Rhapsody' by 'Queen'...")
    track_data = spotify.search_track("Bohemian Rhapsody", "Queen")
    
    if track_data:
        print("\nTrack Information:")
        print(f"Name: {track_data['name']}")
        print(f"Artist: {track_data['artist']}")
        print(f"Popularity: {track_data['popularity']}/100")
        print(f"Album: {track_data['album']}")
        print(f"Release Date: {track_data['release_date']}")
        print(f"Track ID: {track_data['id']}")
    else:
        print("No track found.")
    
    # Compare multiple popular songs
    print("\n\nComparing popularity of multiple tracks...")
    tracks_to_compare = [
        {"name": "Bohemian Rhapsody", "artist": "Queen"},
        {"name": "Stairway to Heaven", "artist": "Led Zeppelin"},
        {"name": "Sweet Child O Mine", "artist": "Guns N Roses"},
        {"name": "Billie Jean", "artist": "Michael Jackson"},
        {"name": "Smells Like Teen Spirit", "artist": "Nirvana"}
    ]
    
    results = []
    for track in tracks_to_compare:
        result = spotify.search_track(track["name"], track["artist"])
        if result:
            results.append(result)
            print(f"Found: {result['name']} by {result['artist']} (Popularity: {result['popularity']}/100)")
    
    if results:
        print("\nPopularity Ranking:")
        for i, track in enumerate(sorted(results, key=lambda x: x['popularity'], reverse=True)):
            print(f"{i+1}. {track['name']} by {track['artist']}: {track['popularity']}/100")

    # Try a current hit song
    print("\n\nChecking popularity of a recent hit...")
    recent_hit = spotify.search_track("Houdini", "Dua Lipa")
    if recent_hit:
        print(f"'{recent_hit['name']}' by {recent_hit['artist']} has a popularity score of {recent_hit['popularity']}/100")

    # Try a classic song
    print("\nChecking popularity of a classic song...")
    classic = spotify.search_track("Yesterday", "The Beatles")
    if classic:
        print(f"'{classic['name']}' by {classic['artist']} has a popularity score of {classic['popularity']}/100")

if __name__ == "__main__":
    main()
