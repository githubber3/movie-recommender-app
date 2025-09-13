import pandas as pd
from surprise import Dataset, SVD, Reader
import streamlit as st
import requests
import os
import re
import zipfile
import gdown

# --- Ensure dataset is available ---
@st.cache_data(show_spinner="Downloading dataset...")
def ensure_data_available():
    download_and_extract_ml_20m()

def download_and_extract_ml_20m():
    zip_file = "ml-20m.zip"
    data_folder = "ml-20m"

    if not os.path.exists(data_folder):
        url = "https://drive.google.com/uc?id=1yJXGy0oHO4FboOj5j105QSxh9XrrQ1Hm"

        st.info("Downloading dataset...")
        gdown.download(url, zip_file, quiet=False)

        if not os.path.exists(zip_file):
            raise FileNotFoundError("Download failed: ml-20m.zip not found.")

        try:
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(".")
        except zipfile.BadZipFile:
            raise RuntimeError("Downloaded file is not a valid ZIP archive")

        os.remove(zip_file)


# --- Caching Functions for ml-20m Data ---
@st.cache_data
def load_movie_data():
    return pd.read_csv(os.path.join("ml-20m", "movies.csv"))

@st.cache_data
def load_rating_data():
    return pd.read_csv(os.path.join("ml-20m", "ratings.csv"))

@st.cache_data
def load_links_data():
    links_df = pd.read_csv(os.path.join("ml-20m", "links.csv"))
    links_df['tmdbId'] = links_df['tmdbId'].fillna(0).astype(int)
    return links_df

@st.cache_data
def load_merged_movies_data():
    return pd.merge(load_movie_data(), load_links_data(), on='movieId', how='left')


# --- TMDB API Interaction ---
@st.cache_data(ttl=3600)
def get_movie_details(tmdb_id):
    if not tmdb_id or tmdb_id == 0:
        return None

    tmdb_api_key = st.secrets.get("tmdb_api_key")
    if not tmdb_api_key:
        st.warning("TMDb API key not found in secrets.")
        return None

    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            params={"api_key": tmdb_api_key, "language": "en-US"}
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        return {
            "poster_path": data.get("poster_path"),
            "overview": data.get("overview", "Description not available."),
            "release_date": data.get("release_date")
        }
    except requests.exceptions.RequestException as e:
        st.warning(f"TMDb fetch failed for tmdbId {tmdb_id}: {e}")
        return None


# --- Train SVD (on subsample) ---
@st.cache_resource(show_spinner="Training SVD model...")
def load_and_train_svd_model():
    ratings_df = load_rating_data()

    # 🔁 Try with 100_000 first, then scale up gradually
    sample_size = 100_000
    if len(ratings_df) > sample_size:
        ratings_df = ratings_df.sample(sample_size, random_state=42)

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

@st.cache_data
def get_top_rated_movies(n=5):
    ratings_df = load_rating_data()
    movies_df = load_merged_movies_data()

    avg_ratings = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    avg_ratings.columns = ['movieId', 'avg_rating', 'rating_count']
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
    ensure_data_available()
    ratings_df = load_rating_data()
    merged_movies_df = load_merged_movies_data()

    if method == "Top Rated":
        return get_top_rated_movies(n)

    # Personalized
    movies_df = merged_movies_df[['movieId', 'title']].drop_duplicates()
    movie_row = movies_df[movies_df['title'] == movie_title]

    if movie_row.empty:
        st.warning(f"Movie '{movie_title}' not found.")
        return []

    movie_id = movie_row['movieId'].iloc[0]
    relevant_users = ratings_df[ratings_df['movieId'] == movie_id]['userId'].tolist()

    if not relevant_users:
        st.warning(f"No ratings found for '{movie_title}'.")
        return []

    try:
        algo, trainset = load_and_train_svd_model()
    except Exception as e:
        st.error(f"Model training failed: {e}")
        return []

    user_id = relevant_users[0]
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

        movie_info = merged_movies_df[merged_movies_df['movieId'] == pred.iid]
        if movie_info.empty:
            continue

        movie_info = movie_info.iloc[0]
        tmdb_id = movie_info['tmdbId']
        details = get_movie_details(tmdb_id)
        if not details:
            continue

        recommended_movies_with_details.append({
            "title": movie_info['title'],
            "poster_path": details.get("poster_path"),
            "overview": details.get("overview"),
            "release_date": details.get("release_date") or extract_year_from_title(movie_info['title'])
        })

        count += 1

    return recommended_movies_with_details
