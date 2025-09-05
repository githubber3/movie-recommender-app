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



# Function to get recommendations (replace with your actual logic)
# def get_recommendations(movie_title):
#     # This is a placeholder. Your actual recommendation engine would go here.
#     # It would take the selected movie and return a list of recommended titles.
#     if movie_title == "The Matrix":
#         return ["Dark City", "Blade Runner", "Ghost in the Shell"]
#     elif movie_title == "Inception":
#         return ["Shutter Island", "Memento", "Tenet"]
#     else:
#         return ["No recommendations found."]



# Create a list of movies to display in the select box
# movie_options = ["The Matrix", "Inception", "Interstellar", "Pulp Fiction"]

# Call st.selectbox() with a label and options
# selected_movie = st.selectbox("Select a movie", movie_options)

# The 'selected_movie' variable will now hold the user's selection
# st.write("You selected:", selected_movie)

# if st.button("Get Recommendations"):
#     # This code will run when the button is clicked.
#     st.write(f"### Recommendations for {selected_movie}:")
#     # Your recommendation logic goes here.

# # Get the recommendations
#     recommendations = get_recommendations(selected_movie)

#     # Display the recommendations using a loop or st.write
#     for movie in recommendations:
#         st.write(f"- {movie}")    
