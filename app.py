import streamlit as st
import pandas as pd
from recommender import get_movie_recommendations, load_movie_data

# --- App configuration ---
st.set_page_config(layout="wide")
st.title("Movie Recommender App")

# --- Session state management ---
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None

# --- Data loading and caching ---
@st.cache_data
def get_movie_options():
    """Load movie titles and cache the result."""
    movies_df = load_movie_data()
    return sorted(movies_df['Title'].tolist())

movie_options = get_movie_options()

# --- Layout and widgets ---
col1, col2 = st.columns()

with col1:
    st.header("Select a Movie")
    selected_movie = st.selectbox("Choose a movie", movie_options, key="movie_select")

with col2:
    st.header("Recommendations")
    if st.button("Get Recommendations"):
        with st.spinner('Generating recommendations...'):
            st.session_state.recommendations = get_movie_recommendations(selected_movie)
            st.session_state.selected_movie = selected_movie

# --- Display recommendations ---
if st.session_state.recommendations is not None:
    # Use the movie from session state to avoid stale display
    st.write(f"### Recommendations for **{st.session_state.selected_movie}**:")

    # Handle different return types
    if isinstance(st.session_state.recommendations, list):
        if st.session_state.recommendations:  # Check if the list is not empty
            for movie in st.session_state.recommendations:
                st.write(f"- **{movie}**")
        else:
            st.info("No recommendations found for this movie.")
    else:
        # Handle error messages returned by the recommender
        st.error(st.session_state.recommendations)
