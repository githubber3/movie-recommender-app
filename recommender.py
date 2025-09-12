import pandas as pd
from surprise import Dataset, SVD, Reader
import streamlit as st
import requests
import os
import re
import zipfile
import requests


def download_and_extract_ml_20m():
    folder_name = "ml-20m"
    zip_file = "ml-20m.zip"

    if not os.path.exists(folder_name):
        # Replace with your actual Google Drive file ID
        file_id = "1yJXGy0oHO4FboOj5j105QSxh9XrrQ1Hm"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"

        print("Downloading ml-20m.zip from Google Drive...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(zip_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print("Extracting ml-20m.zip...")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(".")

        os.remove(zip_file)
        print("ml-20m dataset is ready.")

download_and_extract_ml_20m()

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

def extract_year_from_title(title):
    match = re.search(r'\((\d{4})\)', title)
    return match.group(1) if match else "Unknown"

@st.cache_data(show_spinner=False)
def get_top_rated_movies(n=5):
    """Returns the top-rated movies by average rating."""
    ratings_df = load_rating_data()
    movies_df = load_merged_movies_data()

    avg_ratings = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    avg_ratings.columns = ['movieId', 'avg_rating', 'rating_count']

    # Filter out movies with very few ratings (e.g., < 50)
    filtered = avg_ratings[avg_ratings['rating_count'] >= 50]

    merged = pd.merge(filtered, movies_df, on='movieId', how='inner').sort_values(by='avg_rating', ascending=False)

    recommendations = []
    for _, row in merged.head(n).iterrows():
        details = get_movie_details(row['tmdbId'])
        if not details:
            continue

        recommendations.append({
            "title": row['title'],
            "poster_path": details.get("poster_path"),
            "overview": details.get("overview"),
            "avg_rating": round(row['avg_rating'], 2),
            "release_date": details.get("release_date")
        })

    return recommendations

def get_movie_recommendations(movie_title=None, n=5, method="Personalized"):
    ratings_df = load_rating_data()
    merged_movies_df = load_merged_movies_data()

    # Compute average ratings and count
    average_ratings = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    average_ratings.columns = ['movieId', 'avg_rating', 'rating_count']

    # Merge ratings with metadata
    merged = pd.merge(merged_movies_df, average_ratings, on='movieId', how='left')

    if method == "Top Rated":
        # Optional filter: only include movies with 50+ ratings
        if method == "Top Rated":
            return get_top_rated_movies(n)


    else:
        # Personalized (SVD-based) recommendation
        movies_df = merged_movies_df[['movieId', 'title']].drop_duplicates()
        movie_row = movies_df[movies_df['title'] == movie_title]
        if movie_row.empty:
            return f"Movie '{movie_title}' not found in the dataset."

        movie_id = movie_row['movieId'].iloc[0]
        relevant_users = ratings_df[ratings_df['movieId'] == movie_id]['userId'].tolist()
        if not relevant_users:
            return f"No ratings found for '{movie_title}'. Unable to generate recommendations."

        user_id = relevant_users[0]
        algo, trainset = load_and_train_svd_model()
        rated_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
        predictions = [
            algo.predict(user_id, item_id)
            for item_id in movies_df['movieId']
            if item_id not in rated_movie_ids
        ]
        predictions.sort(key=lambda x: x.est, reverse=True)

        recommended_movies_with_details = []
        count = 0
        for pred in predictions:
            if count >= n:
                break

            movie_info = merged[merged['movieId'] == pred.iid]

            if movie_info.empty:
                continue
            movie_info = movie_info.iloc[0]

            tmdb_id = movie_info['tmdbId']
            details = get_movie_details(tmdb_id)
            if details is None:
                continue

            release_year = details.get('release_date') or extract_year_from_title(movie_info['title'])

            recommended_movies_with_details.append({
                "title": movie_info['title'],
                "poster_path": details.get("poster_path"),
                "overview": details.get("overview"),
                "vote_average": round(movie_info['avg_rating'], 1) if pd.notna(movie_info['avg_rating']) else None,
                "release_date": release_year
            })

            count += 1

        return recommended_movies_with_details
