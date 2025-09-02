import streamlit as st 
import pandas as pd
import pickle
from difflib import get_close_matches
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

CLIENT_ID = "fc02c949f3b34f09b7874b6ee3d8aad7"
CLIENT_SECRET = "db76050e5529419bb9bfe06d3bb3bb41"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET
))

with open('track_df.pkl','rb') as f:
    track_df = pickle.load(f)
    
with open('matrix.pkl','rb') as f:
    similarity_matrix = pickle.load(f)

def recommend(song_name, top_n=10):
    song_name_clean = song_name.lower().strip()
    
    matches = track_df[track_df['track_name_clean'] == song_name_clean]
    if matches.empty:
        all_names = track_df['track_name_clean'].tolist()
        close = get_close_matches(song_name_clean, all_names, n=1, cutoff=0.6)
        if not close:
            return f"Song '{song_name}' not found in the database."
        matches = track_df[track_df['track_name_clean'] == close[0]]
    
    index = matches.index[0]
    
    if hasattr(similarity_matrix, 'shape') and len(similarity_matrix.shape) > 1:
        sim_scores = cosine_similarity(similarity_matrix[index], similarity_matrix).flatten()
        top_indices = np.argsort(sim_scores)[-top_n-1:-1][::-1]
    else:
        same_genre = track_df[track_df['track_genre'] == track_df.iloc[index]['track_genre']]
        top_indices = same_genre.sample(min(top_n, len(same_genre))).index
    
    return track_df.iloc[top_indices][["track_name", "artists", "track_genre", "popularity"]]

def get_album_cover(track_name, artist):
    query = f"{track_name} {artist}"
    results = sp.search(q=query, limit=1, type="track")
    if results["tracks"]["items"]:
        return results["tracks"]["items"][0]["album"]["images"][0]["url"]
    return None

# ------------------------
# Streamlit UI
# ------------------------
st.title("Song Recommendation System")

typed_song = st.text_input("🔍 Type a song name (or leave blank to use dropdown):")

select_song_name = st.selectbox(
    "Or pick from the list:",
    track_df['track_name'].values
)

song_choice = typed_song if typed_song.strip() else select_song_name

if st.button("Recommend"):
    recommendation = recommend(song_choice)
    st.subheader(f"Recommended songs for: **{song_choice}**")
    
    if isinstance(recommendation, str):
        st.error(recommendation)
    else:
        num_cols = 3
        for i in range(0, len(recommendation), num_cols):
            cols = st.columns(num_cols)
            for j, col in enumerate(cols):
                if i+j < len(recommendation):
                    row = recommendation.iloc[i+j]
                    cover_url = get_album_cover(row['track_name'], row['artists'])
                    if cover_url:
                        col.image(cover_url, width=150)
                    col.markdown(f"**{row['track_name']}**  \nby {row['artists']}  \nGenre: {row['track_genre']}  \nPopularity: {row['popularity']}")