import streamlit as st
import pandas as pd
from recommender import get_movie_recommendations # import the recommender funtion

# Pass a string argument to st.title()
st.title("Movie Recommender App")

# Load the movie titles from the CSV file
movies_df = pd.read_csv('ml-1m/movies.csv')
movie_options = sorted(movies_df['Title'].tolist())

# Create a selectbox with all the available movie titles
selected_movie = st.selectbox("Select a movie", movie_options)

# Create the button
if st.button("Get Recommendations"):
    # This code block is executed only when the button is clicked.
    with st.spinner('Generating recommendations...'):
        # Call the get_movie_recommendations function
        recommendations = get_movie_recommendations(selected_movie)

    st.write(f"### Recommendations for **{selected_movie}**:")   

    # Display the recommendations
    if isinstance(recommendations, list):
        for movie in recommendations:
            st.write(f"- **{movie}**")

    else:
        st.write(recommendations)
