import streamlit as st
from recommender import (
    get_movie_recommendations,
    load_movie_data,
    ensure_data_available
)

# --- Ensure dataset is ready ---
ensure_data_available()

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
    movies_df = load_movie_data()
    return sorted(movies_df['title'].tolist())

movie_options = get_movie_options()

# --- Method toggle ---
def on_method_change():
    method = st.session_state.get("rec_method", "Personalized")
    if method == "Top Rated":
        st.session_state.recommendations = get_movie_recommendations(None, method="Top Rated")
        st.session_state.selected_movie = None

rec_method = st.radio(
    "Recommendation Method:",
    ["Personalized", "Top Rated"],
    key="rec_method",
    horizontal=True,
    on_change=on_method_change
)

# --- Layout and widgets ---
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Select a Movie")
    if rec_method == "Personalized":
        selected_movie = st.selectbox(
            "Select a movie to get recommendations for",
            movie_options,
            key="movie_select"
        )
    else:
        selected_movie = None

with col2:
    st.header("Recommendations")
    if rec_method == "Personalized":
        if st.button("Get Recommendations", key="get_recs_btn"):
            with st.spinner('Generating recommendations...'):
                st.session_state.recommendations = get_movie_recommendations(
                    selected_movie,
                    method="Personalized"
                )
                st.session_state.selected_movie = selected_movie

# --- Display Recommendations ---
if st.session_state.recommendations:
    st.subheader("Top Recommendations:")
    recs = st.session_state.recommendations
    cols = st.columns(len(recs))

    for col, rec in zip(cols, recs):
        with col:
            st.markdown(f"**{rec['title']}**")
            if rec["poster_path"]:
                st.image(f"https://image.tmdb.org/t/p/w200{rec['poster_path']}")
            else:
                st.write("Poster not available")
            st.caption(f"{rec['release_date']}")
            st.write(rec["overview"])
