import pickle
from content_based import movies, cosine_sim, indices

# Load trained SVD model
with open("models/svd_model.pkl", "rb") as f:
    svd_model = pickle.load(f)

def hybrid_recommend(movie_name):

    idx = indices[movie_name]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:31]

    hybrid_scores = []

    for movie_idx, content_score in sim_scores:

        movie_id = movies.iloc[movie_idx]["movieId"]

        collab_score = svd_model.predict(
            1,
            movie_id
        ).est

        final_score = (
            0.4 * content_score
            +
            0.6 * collab_score
        )

        hybrid_scores.append(
            (movie_idx, final_score)
        )

    hybrid_scores = sorted(
        hybrid_scores,
        key=lambda x: x[1],
        reverse=True
    )

    movie_indices = [
        x[0]
        for x in hybrid_scores[:10]
    ]

    return list(
        movies["title"].iloc[movie_indices]
    )

if __name__ == "__main__":
    print(
        hybrid_recommend(
            "Toy Story (1995)"
        )
    )