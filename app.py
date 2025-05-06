import streamlit as st
import pandas as pd
import numpy as np

# Load the cleaned Spotify dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Spotify.csv", sep=';')
    df.columns = df.columns.str.strip()
    return df

df = load_data()

st.title("🎵 Spotify Song Recommender")
st.markdown("Answer a few quick questions and get a song you'll probably love — plus similar suggestions.")

# 1. Genre selection
genre = st.selectbox("🎧 What genre do you enjoy?", sorted(df['top genre'].unique()))

# 2. Tempo
tempo_pref = st.radio("⏱ Do you prefer fast or slow songs?", ["Fast", "Slow"])

# 3. Energy
energy_pref = st.radio("⚡ Do you like upbeat or calm music?", ["Upbeat", "Calm"])

# 4. Instrumental or lyrics
instrument_pref = st.radio("🎼 Do you prefer instrumental or lyrical songs?", ["Instrumental", "With Lyrics"])

# 5. Mood
mood = st.selectbox("😊 What mood are you in?", ["Happy", "Sad", "Romantic", "Chill", "Energetic"])

# Map user input to expected audio features
def preference_to_target():
    target = {}
    
    target['bpm'] = 130 if tempo_pref == "Fast" else 80
    target['energy'] = 75 if energy_pref == "Upbeat" else 35
    target['acousticness'] = 70 if instrument_pref == "Instrumental" else 20
    target['speechiness'] = 10 if instrument_pref == "Instrumental" else 50
    mood_valence = {
        "Happy": 80,
        "Sad": 20,
        "Romantic": 50,
        "Chill": 40,
        "Energetic": 70
    }
    target['valence'] = mood_valence[mood]
    
    return target

# Compute similarity score
def compute_similarity(song, target):
    score = 0
    # Weighted differences
    score += abs(song['bpm'] - target['bpm']) * 0.3
    score += abs(song['energy'] - target['energy']) * 0.3
    score += abs(song['acousticness'] - target['acousticness']) * 0.2
    score += abs(song['speechiness'] - target['speechiness']) * 0.1
    score += abs(song['valence'] - target['valence']) * 0.3
    return score

if st.button("🎯 Recommend a Song"):
    # Step 1: Filter songs by genre
    genre_songs = df[df['top genre'].str.lower() == genre.lower()]
    
    if genre_songs.empty:
        st.warning("No songs found in that genre. Try another!")
    else:
        target_features = preference_to_target()

        # Step 2: Compute similarity score for each song
        genre_songs['similarity_score'] = genre_songs.apply(
            lambda row: compute_similarity(row, target_features), axis=1
        )

        # Step 3: Sort songs by score (lower = more similar)
        top_songs = genre_songs.sort_values(by='similarity_score').reset_index(drop=True)

        # Step 4: Show top match
        top_song = top_songs.iloc[0]
        st.success("🎵 Your top recommended song is:")
        st.write(f"**{top_song['title']}** by *{top_song['artist']}*")
        
        # Step 5: Show similar songs
        st.markdown("### 🎶 You might also like:")
        for i in range(1, 6):
            row = top_songs.iloc[i]
            st.write(f"- {row['title']} by {row['artist']}")
