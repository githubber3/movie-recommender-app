import pandas as pd
from surprise import Dataset, SVD, Reader
import streamlit as st

# --- Data loading and caching ---
@st.cache_data
def load_movie_data():
    """Load movie titles and cache the result."""
    return pd.read_csv('ml-1m/movies.csv')

@st.cache_data
def load_rating_data():
    """Load ratings and cache the result."""
    return pd.read_csv('ml-1m/ratings.csv')

@st.cache_resource
def load_and_train_svd_model():
    """Load data and train the SVD model, caching the model itself."""
    ratings_df = load_rating_data()
    
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['UserID', 'MovieID', 'Rating']], reader)
    trainset = data.build_full_trainset()
    
    algo = SVD()
    algo.fit(trainset)
    
    return algo, trainset

# --- Recommendation Function ---
def get_movie_recommendations(movie_title, n=5):
    """
    Returns a list of movie titles recommended based on the input movie title.
    """
    
    movies_df = load_movie_data()
    ratings_df = load_rating_data()
    algo, trainset = load_and_train_svd_model()
    
    try:
        movie_id = movies_df[movies_df['Title'] == movie_title]['MovieID'].iloc[0]
    except IndexError:
        return ["Movie not found in the dataset."]
        
    relevant_users = ratings_df[ratings_df['MovieID'] == movie_id]['UserID'].tolist()
    
    if not relevant_users:
        return [f"No ratings found for '{movie_title}'. Unable to generate recommendations."]
    
    all_movie_ids = movies_df['MovieID'].tolist()
    user_id_to_predict = relevant_users[0]  # Using the first relevant user for simplicity
    
    predictions = []
    rated_movie_ids = ratings_df[ratings_df['UserID'] == user_id_to_predict]['MovieID'].tolist()
    
    for item_id in all_movie_ids:
        if item_id not in rated_movie_ids:
            predictions.append(algo.predict(user_id_to_predict, item_id))
            
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    top_n_recs = [pred.iid for pred in predictions[:n]]
    
    recommended_titles = movies_df[movies_df['MovieID'].isin(top_n_recs)]['Title'].tolist()
    
    return recommended_titles

if __name__ == '__main__':
    sample_movie = "Toy Story (1995)"
    print(f"Recommendations for {sample_movie}:")
    recs = get_movie_recommendations(sample_movie)
    if isinstance(recs, list):
        for r in recs:
            print(f"- {r}")
    else:
        print(recs)
