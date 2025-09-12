import streamlit as st
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
    return sorted(movies_df['title'].tolist())

movie_options = get_movie_options()


def on_method_change():
    method = st.session_state.get("rec_method", "Personalized")
    movie = st.session_state.get("movie_select")

    # Only auto-fetch recommendations for Top Rated
    if method == "Top Rated":
        st.session_state.recommendations = get_movie_recommendations(
            None,
            method="Top Rated"
        )
        st.session_state.selected_movie = None

# Radio with callback
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

    # Only show the button in Personalized mode
    if rec_method == "Personalized":
        if st.button("Get Recommendations", key="get_recs_btn"):
            with st.spinner('Generating recommendations...'):
                st.session_state.recommendations = get_movie_recommendations(
                    selected_movie,
                    method="Personalized"
                )
                st.session_state.selected_movie = selected_movie




# --- Display recommendations ---
if st.session_state.recommendations is not None:
    st.write(f"### Recommendations for **{st.session_state.selected_movie}**:")

    if isinstance(st.session_state.recommendations, list) and st.session_state.recommendations:
        movies = st.session_state.recommendations
        posters_per_row = 5

        # Custom CSS for hover tooltip
        st.markdown("""
            <style>
            .movie-container {
                display: inline-block;
                text-align: center;
                margin: 10px;
                position: relative;
                width: 150px;
            }
            .movie-poster {
                width: 150px;
                height: auto;
                border-radius: 8px;
                transition: transform 0.2s;
            }
            .movie-container:hover .movie-poster {
                transform: scale(1.05);
            }
            .tooltip {
                visibility: hidden;
                background-color: rgba(0, 0, 0, 0.8);
                color: #fff;
                text-align: left;
                border-radius: 5px;
                padding: 10px;
                position: absolute;
                z-index: 1;
                bottom: 110%;
                left: 50%;
                transform: translateX(-50%);
                width: 250px;
                opacity: 0;
                transition: opacity 0.3s;
                font-size: 0.8rem;
            }
            .movie-container:hover .tooltip {
                visibility: visible;
                opacity: 1;
            }
            .movie-title {
                font-weight: bold;
                margin-top: 5px;
                font-size: 0.9rem;
            }
            .movie-meta {
                font-size: 0.8rem;
                color: #666;
            }
            </style>
        """, unsafe_allow_html=True)

        # Build grid layout
        for i in range(0, len(movies), posters_per_row):
            cols = st.columns(posters_per_row)
            for j in range(posters_per_row):
                if i + j < len(movies):
                    movie = movies[i + j]
                    with cols[j]:
                        # Safely extract and sanitize values
                        title = movie.get('title', 'No Title')
                        poster_path = movie.get('poster_path')
                        poster_url = f"https://image.tmdb.org/t/p/w500/{poster_path}" if poster_path else ""

                        overview = movie.get('overview', 'No overview available.')
                        overview = overview.replace('"', '&quot;').replace("'", "&#39;")  # Sanitize for HTML

                        release_date = movie.get('release_date') or "Unknown"

                        # Get rating from known keys and sanitize
                        rating = movie.get('vote_average') or movie.get('rating')
                        if rating is not None:
                            try:
                                stars = f"{'⭐️' * int(round(float(rating)))} ({float(rating):.1f})"
                            except:
                                stars = f"Rating: {rating}"
                        else:
                            stars = "Rating: N/A"

                        # Final HTML block
                        html = f"""
                            <div class="movie-container">
                                <img src="{poster_url}" class="movie-poster"/>
                                <div class="tooltip">{overview}</div>
                                <div class="movie-title">{title}</div>
                                <div class="movie-meta">
                                    {stars}<br>
                                    {release_date}
                                </div>
                            </div>
                        """
                        st.markdown(html, unsafe_allow_html=True)
