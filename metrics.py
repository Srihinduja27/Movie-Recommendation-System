# metrics.py
import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def precision_at_k(recommended_items, relevant_items, k=10):
    """
    recommended_items: list of movieIds predicted
    relevant_items: set of movieIds user actually liked (rating >= threshold)
    """
    recommended_items = recommended_items[:k]

    if len(recommended_items) == 0:
        return 0

    hits = 0
    for item in recommended_items:
        if item in relevant_items:
            hits += 1

    return hits / k
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

from metrics import rmse

# Load data
ratings = pd.read_csv("data/ratings.csv")

reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2)

# Train SVD
model = SVD()
model.fit(trainset)

# Predictions
predictions = model.test(testset)

# RMSE (built-in Surprise)
print("📊 SVD RMSE:", accuracy.rmse(predictions))

# Extra manual RMSE (optional demo)
y_true = [pred.r_ui for pred in predictions]
y_pred = [pred.est for pred in predictions]

print("📊 Manual RMSE:", rmse(y_true, y_pred))
from collections import defaultdict
import numpy as np

def get_top_n(predictions, n=10):
    top_n = defaultdict(list)

    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = [iid for iid, _ in user_ratings[:n]]

    return top_n


# Build predictions
top_n = get_top_n(predictions, n=10)

# Evaluate Precision@K
precisions = []

for uid, recommended in top_n.items():
    # actual liked movies (ground truth)
    user_ratings = ratings[ratings["userId"] == uid]

    relevant = set(user_ratings[user_ratings["rating"] >= 4]["movieId"])

    if len(relevant) == 0:
        continue

    hits = len(set(recommended) & relevant)

    precisions.append(hits / 10)

print("📊 Precision@10:", np.mean(precisions))
def evaluate_hybrid(test_users, hybrid_recommend_func):
    precisions = []

    for user in test_users:
        recommended = hybrid_recommend_func(user)[:10]

        # assume you define liked movies
        relevant = ratings[
        (ratings["userId"] == uid) &
        (ratings["rating"] >= 4)
        ]["movieId"].tolist()

        hits = len(set(recommended) & set(relevant))
        precisions.append(hits / 10)

    return np.mean(precisions)