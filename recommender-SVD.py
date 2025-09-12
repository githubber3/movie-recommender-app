import pandas as pd
from surprise import Dataset, SVD, Reader
import streamlit as st
import requests
import os

# --- Caching Functions for ml-20m Data ---
@st.cache_data(show_spinner=False)
def load_movie_data():
    """Load ml-20m movies.csv into a Pandas DataFrame."""
    movies_df = pd.read_csv(os.path.join("ml-20m", "movies.csv"))
    return movies_df

@st.cache_data(show_spinner=False)
def load_rating_data():
    """Load ml-20m ratings.csv into a Pandas DataFrame."""
    ratings_df = pd.read_csv(os.path.join("ml-20m", "ratings.csv"))
    return ratings_df

@st.cache_data(show_spinner=False)
def load_links_data():
    """Load ml-20m links.csv into a Pandas DataFrame."""
    links_df = pd.read_csv(os.path.join("ml-20m", "links.csv"))
    links_df['tmdbId'] = links_df['tmdbId'].fillna(0).astype(int)
    return links_df


@st.cache_data(show_spinner=False)
def load_merged_movies_data():
    """
    Load and merge movies.csv with links.csv into one DataFrame.
    """
    movies_df = load_movie_data()
    links_df = load_links_data()
    merged_df = pd.merge(movies_df, links_df, on='movieId', how='left')
    return merged_df


# --- TMDB API Interaction ---

# In recommender.py

# ... (other code) ...

# --- TMDB API Interaction ---
@st.cache_data(ttl=3600)
def get_movie_details(tmdb_id):
    """Fetches movie details (poster, overview) from TMDB API using tmdbId."""
    if not tmdb_id or tmdb_id == 0:
        return None  # Return None instead of empty details

    tmdb_api_key = st.secrets.get("tmdb_api_key")
    if not tmdb_api_key:
        st.warning("TMDb API key not found in secrets.")
        return None

    base_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    params = {"api_key": tmdb_api_key, "language": "en-US"}

    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 404:
            # Silent fail for missing movies
            return None
        response.raise_for_status()
        data = response.json()
        return {
            "poster_path": data.get("poster_path"),
            "overview": data.get("overview", "Description not available.")
        }
    except requests.exceptions.RequestException as e:
        st.warning(f"TMDb fetch failed for tmdbId {tmdb_id}: {e}")
        return None



@st.cache_resource(show_spinner="Training SVD model...")
def load_and_train_svd_model():
    """Load data and train the SVD model, caching the model itself."""
    ratings_df = load_rating_data()
    
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    algo = SVD()
    algo.fit(trainset)
    
    return algo, trainset

# --- Recommendation Function ---

import re

def extract_year_from_title(title):
    match = re.search(r'\((\d{4})\)', title)
    return match.group(1) if match else "Unknown"

def get_movie_recommendations(movie_title, n=5):
    ratings_df = load_rating_data()
    merged_movies_df = load_merged_movies_data()
    movies_df = merged_movies_df[['movieId', 'title']].drop_duplicates()

    # Compute average ratings
    average_ratings = ratings_df.groupby('movieId')['rating'].mean().reset_index()
    average_ratings.columns = ['movieId', 'avg_rating']

    # Merge ratings with movie metadata
    merged_movies_df = pd.merge(merged_movies_df, average_ratings, on='movieId', how='left')

    # Find movieId for the selected title
    movie_row = movies_df[movies_df['title'] == movie_title]
    if movie_row.empty:
        return f"Movie '{movie_title}' not found in the dataset."
    movie_id = movie_row['movieId'].iloc[0]

    # Get a user who rated this movie
    relevant_users = ratings_df[ratings_df['movieId'] == movie_id]['userId'].tolist()
    if not relevant_users:
        return f"No ratings found for '{movie_title}'. Unable to generate recommendations."
    user_id = relevant_users[0]

    # Load model and generate predictions
    algo, trainset = load_and_train_svd_model()
    rated_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    predictions = [
        algo.predict(user_id, item_id)
        for item_id in movies_df['movieId']
        if item_id not in rated_movie_ids
    ]

    predictions.sort(key=lambda x: x.est, reverse=True)

    # Build list of recommended movies
    recommended_movies_with_details = []
    count = 0

    for pred in predictions:
        if count >= n:
            break

        movie_info = merged_movies_df[merged_movies_df['movieId'] == pred.iid]
        if movie_info.empty:
            continue
        movie_info = movie_info.iloc[0]

        title = movie_info['title']
        tmdb_id = movie_info['tmdbId']
        avg_rating = movie_info.get('avg_rating')

        details = get_movie_details(tmdb_id)
        if details is None:
            continue

        # Fallback: try to extract year from title if release_date not from TMDb
        release_year = details.get('release_date') or extract_year_from_title(title)

        recommended_movies_with_details.append({
            "title": title,
            "poster_path": details.get("poster_path"),
            "overview": details.get("overview"),
            "vote_average": round(avg_rating, 1) if avg_rating else None,
            "release_date": release_year
        })

        count += 1

    return recommended_movies_with_details
