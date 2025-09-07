import streamlit as st
import pandas as pd
from recommender import get_movie_recommendations # import the recommender funtion

# Use session state to store the recommendations
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

# Set the page to wide layout for better column spacing
st.set_page_config(layout="wide")    

# Pass a string argument to st.title()
st.title("Movie Recommender App")

# Load the movie titles from the CSV file
movies_df = pd.read_csv('ml-1m/movies.csv')
movie_options = sorted(movies_df['Title'].tolist())

# Create two columns with a specified width ratio
col1, col2 = st.columns([2, 1])

# Create a selectbox with all the available movie titles
# Plase the movie selector in the first , thinner column
with col1:
    st.header("Select a Movie")
    selected_movie = st.selectbox("Choose a movie", movie_options)

# Place the recommendations in the second, wider column
with col2:
    st.header("Recommendations")
    # Create the button
    if st.button("Get Recommendations"):
    # This code block is executed only when the button is clicked.
         with st.spinner('Generating recommendations...'):
        # Call the get_movie_recommendations function
            st.session_state.recommendations = get_movie_recommendations(selected_movie)

   # st.write(f"### Recommendations for **{selected_movie}**:")   

    # Display the recommendations if they exist in the session state
    if st.session_state.recommendations is not None:
        st.write(f"### Recommendations for  **{selected_movie}**:")

        # Check if the result is a list and display it
        if isinstance(st.session_state.recommendations, list):
            for movie in st.session_state.recommendations:
                st.write(f"- **{movie}**")
        else:
            # Handle the case where recommendations is not a list
            st.write(st.session_state.recommendations)
