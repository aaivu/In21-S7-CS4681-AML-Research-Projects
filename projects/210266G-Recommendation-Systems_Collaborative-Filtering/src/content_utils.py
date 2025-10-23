import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

def process_content_features(data_path, item_map):
    print("Processing content features from music.csv...")
    music_df = pd.read_csv(f"{data_path}/music.csv")

    item_ids = list(item_map.keys())
    item_indices = list(item_map.values())
    
    aligned_df = pd.DataFrame({'track_id': item_ids}, index=item_indices)
    aligned_df = aligned_df.merge(music_df, on='track_id', how='left').sort_index()

    numerical_cols = [
        'year', 'duration_ms', 'danceability', 'energy', 'key', 'loudness',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo'
    ]
    for col in numerical_cols:
        aligned_df[col].fillna(aligned_df[col].mean(), inplace=True)
    
    scaler = MinMaxScaler()
    numerical_features = scaler.fit_transform(aligned_df[numerical_cols])

    aligned_df['genre'].fillna('unknown', inplace=True)
    genre_features = pd.get_dummies(aligned_df['genre'], prefix='genre', dummy_na=False)
    
    # Ensure all possible genres from the full dataset are columns
    # This prevents errors if a genre is missing in the aligned subset
    all_genres = pd.get_dummies(music_df['genre'].fillna('unknown'), prefix='genre', dummy_na=False)
    for col in all_genres.columns:
        if col not in genre_features.columns:
            genre_features[col] = 0
    genre_features = genre_features[all_genres.columns] # Ensure consistent order
    
    feature_matrix = np.concatenate([numerical_features, genre_features.values], axis=1)
    
    print(f"Created content feature matrix with shape: {feature_matrix.shape}")
    return torch.FloatTensor(feature_matrix)