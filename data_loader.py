import pandas as pd

# Define the paths to the dataset files
movies_filepath = 'ml-1m/movies.dat'
ratings_filepath = 'ml-1m/ratings.dat'

# Define column names based on the dataset's README
movies_cols = ['MovieID', 'Title', 'Genres']
ratings_cols = ['UserID', 'MovieID', 'Rating', 'Timestamp']

# Load the movies data
movies_df = pd.read_csv(
    movies_filepath,
    sep='::',
    engine='python',
    header=None,
    names=movies_cols,
    encoding='latin-1'
)

# Load the ratings data
ratings_df = pd.read_csv(
    ratings_filepath,
    sep='::',
    engine='python',
    header=None,
    names=ratings_cols,
    encoding='latin-1'
)

# Pring the first few rows to verify the data was loaded correctly
print("Movies Data:")
print(movies_df.head())
print("\nRatings Data:")
print(ratings_df.head())

# Save the DataFrames to a more standard format for easier use later
movies_df.to_csv('ml-1m/movies.csv', index=False)
ratings_df.to_csv('ml-1m/ratings.csv', index=False)

print("\nSaved .dat files to .csv format.")