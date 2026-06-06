import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD

# Load ratings data
ratings = pd.read_csv("data/ratings.csv")

# Surprise format
reader = Reader(rating_scale=(0.5, 5.0))

data = Dataset.load_from_df(
    ratings[["userId", "movieId", "rating"]],
    reader
)

# Train model
trainset = data.build_full_trainset()

model = SVD()

model.fit(trainset)

print("Collaborative Filtering Model Trained Successfully!")
import pickle
if __name__ == "__main__":
    model.fit(trainset)

    with open("models/svd_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("SVD Model Saved Successfully!")