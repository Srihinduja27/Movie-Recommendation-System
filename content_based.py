import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("data/movies.csv")

# Handle missing values
movies["genres"] = movies["genres"].fillna("")

# Convert genres into numerical vectors
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

# Similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Movie title -> index mapping
indices = pd.Series(
    movies.index,
    index=movies["title"]
).drop_duplicates()

# Recommendation function
def content_recommend(movie_title):

    idx = indices[movie_title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return movies["title"].iloc[movie_indices]
if __name__ == "__main__":
    import pickle
    with open("models/cosine_sim.pkl", "wb") as f:
        pickle.dump(cosine_sim, f)
    print("Cosine similarity Matrix Saved Successfully")