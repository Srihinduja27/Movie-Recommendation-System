import pandas as pd

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

print("Movies Shape:", movies.shape)
print("Ratings Shape:", ratings.shape)

print("\nMovies Dataset")
print(movies.head())

print("\nRatings Dataset")
print(ratings.head())
