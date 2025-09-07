import pandas as pd
from surprise import Dataset, SVD, Reader
from surprise.model_selection import train_test_split
from collections import defaultdict
import streamlit as st

# --- Data loading ---
@st.cache_resource
def load_and_train_model():
    """Load the dataset and train the SVD model, caching the model itself"""
    data = Dataset.load_builtin('ml-100k')
    trainset, _ = train_test_split(data, test_size=0.2)
    algo = SVD()
    algo.fit(trainset)
    return algo

@st.cache_data
def load_movie_data():
    """Load movie titles and cache the result."""
    return pd.read_csv('ml-1m/movies.csv')

movies_df = load_movie_data()

@st.cache_data
def load_rating_data():
    return pd.read_csv('ml-1m/ratings.csv')

ratings_df = load_rating_data()

# --- Train the SVD model (You only need to do this once) ---
# Surprise Reader object expects columns to be in this order: ['user', 'item', 'rating']
# reader = Reader(rating_scale=(1, 5))
# data = Dataset.load_from_df(ratings_df[['UserID', 'MovieID', 'Rating']], reader)
# trainset, _ = train_test_split(data, test_size=0.01) # Use a small test size for simplicity
# algo = SVD()
# algo.fit(trainset)

# --- Recommendation Function ---
def get_movie_recommendations(movie_title, n=5):
    """
    Returns a list of movie titles recommended based on the input movie title.
    This is s simplified approach based on user-item relationship
    """

    # Find the movie ID for the given title
    try:
        movie_id = movies_df[movies_df['Title'] == movie_title]['MovieID'].iloc[0]
    except IndexError:
        return ["Movie not found."]
    
    # Get a list of users who rated this movie highly (e.g., > 4)
    relevant_users = ratings_df[ratings_df['MovieID'] == movie_id]
    top_raters = relevant_users[relevant_users['Rating'] >= 4]['UserID'].tolist()

    if not top_raters:
        return ["Not enough data for this movie."]
    
    # Create a defaultdict to store movie ratings by users
    all_recs = defaultdict(list)

    # For each top rater, find other movies they rated
    for user_id in top_raters:
        user_ratings = ratings_df[ratings_df['UserID'] == user_id]
        
        # Exclude the movie they just rated
        user_ratings = user_ratings[user_ratings['MovieID'] != movie_id]

        # Get predictions for the movies they rated highly
        for _, row in user_ratings.iterrows():
            item_id = row['MovieID']
            # Make a prediction for this user/item pair
            predicted_rating = algo.predict(user_id, item_id).est
            all_recs[item_id].append(predicted_rating)

    # Compute the average predicted rating for each movie and get the top N
    avg_recs = {item_id: sum(ratings) / len(ratings) for item_id, ratings in all_recs.items()}
    top_n_recs = sorted(avg_recs.items(), key=lambda x: x[1], reverse=True)[:n]

    # Get the movie titles for the recommended MovieIDs
    recommended_movie_ids = [item[0] for item in top_n_recs]
    reommended_titles = movies_df[movies_df['MovieID'].isin(recommended_movie_ids)]['Title'].tolist()

    return reommended_titles

if __name__ == '__main__':
    # Test the recommender funtion
    sample_movie = "Toy Story (1995)"
    print(f"Recommendations for {sample_movie}:")
    recs = get_movie_recommendations(sample_movie)
    for r in recs:
        print(f"- {r}")
            