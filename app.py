import streamlit as st
import pandas as pd
import json

from hybrid import hybrid_recommend
from metrics import rmse
from surprise import accuracy

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Hybrid Movie System", page_icon="🎬", layout="wide")

st.title("🎬 Movie Recommendation System")
st.markdown("---")

# ---------------- LOAD DATA (IMPORTANT FIRST) ----------------
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

# ---------------- DATA STATS ----------------
total_movies = movies["movieId"].nunique()

all_genres_set = set(
    genre for g in movies["genres"].dropna().str.split("|") for genre in g
)
total_genres = len(all_genres_set)
avg_rating_dataset = ratings["rating"].mean()

st.metric(
    "⭐ Average Dataset Rating",
    round(avg_rating_dataset, 2)
)

# ---------------- SHOW DATA OVERVIEW ----------------
col1, col2 = st.columns(2)

with col1:
    st.metric("🎬 Total Movies", total_movies)

with col2:
    st.metric("🎭 Total Genres", total_genres)

st.markdown("---")

# ---------------- SESSION STATE ----------------
if "favourites" not in st.session_state:
    try:
        with open("favourites.json", "r") as f:
            st.session_state.favourites = json.load(f)
    except:
        st.session_state.favourites = []

def save_favourites():
    with open("favourites.json", "w") as f:
        json.dump(st.session_state.favourites, f)

def add_fav(movie):
    if movie not in st.session_state.favourites:
        st.session_state.favourites.append(movie)
        save_favourites()

def remove_fav(movie):
    if movie in st.session_state.favourites:
        st.session_state.favourites.remove(movie)
        save_favourites()

# ---------------- PROCESS MOVIES ----------------
movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")
movies["year"] = movies["year"].fillna("Unknown")

movie_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()
movie_ratings.columns = ["movieId", "avg_rating"]

movies = pd.merge(movies, movie_ratings, on="movieId", how="left")
movies["avg_rating"] = movies["avg_rating"].fillna(0)

# ---------------- EVALUATION DASHBOARD ----------------
st.subheader("📊 Model Evaluation Dashboard")

if st.button("🚀 Run Evaluation"):

    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import train_test_split
    from collections import defaultdict
    import numpy as np

    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    trainset, testset = train_test_split(data, test_size=0.2)

    model = SVD()
    model.fit(trainset)

    predictions = model.test(testset)

    svd_rmse = accuracy.rmse(predictions, verbose=False)

    y_true = [p.r_ui for p in predictions]
    y_pred = [p.est for p in predictions]

    manual_rmse = rmse(y_true, y_pred)

    # ---------------- PRECISION@10 ----------------
    top_n = defaultdict(list)

    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid in top_n:
        top_n[uid].sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = [iid for iid, _ in top_n[uid][:10]]

    precisions = []

    for uid, recs in top_n.items():

        relevant = ratings[
            (ratings["userId"] == uid) &
            (ratings["rating"] >= 4)
        ]["movieId"].tolist()

        if len(relevant) == 0:
            continue

        hits = len(set(recs) & set(relevant))
        precisions.append(hits / 10)

    precision_at_10 = np.mean(precisions)

    st.success("Evaluation Completed ✅")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📉 SVD RMSE", round(svd_rmse, 4))

    with col2:
        st.metric("📉 Manual RMSE", round(manual_rmse, 4))

    with col3:
        st.metric("🎯 Precision@10", round(precision_at_10, 4))

# ---------------- FILTER UI ----------------
st.subheader("🎯 Filters")

all_genres = sorted(list(all_genres_set))

col1, col2, col3 = st.columns(3)

with col1:
    selected_genres = st.multiselect("🎭 Genres", all_genres)

with col2:
    year_options = ["All"] + sorted(movies["year"].unique().tolist())
    year_input = st.selectbox("📅 Year", year_options)

with col3:
    rating_filter = st.selectbox("⭐ Rating", ["All", "Above 3", "Above 4", "Above 4.5"])

col4, col5 = st.columns(2)

with col4:
    num_movies = st.slider("🔢 Number of Movies", 5, 30, 10)

with col5:
    sort_by = st.selectbox("📊 Sort By", ["Rating (High-Low)", "Title (A-Z)"])


selected_movie = st.selectbox(
    "🎬 Search / Select Movie",
    ["All Movies"] + sorted(movies["title"].tolist())
)


st.markdown("---")

# ---------------- FILTERING ----------------
filtered_movies = movies.copy()

if selected_movie != "All Movies":
    filtered_movies = filtered_movies[
        filtered_movies["title"] == selected_movie
    ]
if selected_genres:
    filtered_movies = filtered_movies[
        filtered_movies["genres"].apply(lambda x: any(g in x for g in selected_genres))
    ]

if year_input != "All":
    filtered_movies = filtered_movies[filtered_movies["year"] == year_input]

if rating_filter == "Above 3":
    filtered_movies = filtered_movies[filtered_movies["avg_rating"] >= 3]
elif rating_filter == "Above 4":
    filtered_movies = filtered_movies[filtered_movies["avg_rating"] >= 4]
elif rating_filter == "Above 4.5":
    filtered_movies = filtered_movies[filtered_movies["avg_rating"] >= 4.5]

if sort_by == "Rating (High-Low)":
    filtered_movies = filtered_movies.sort_values(by="avg_rating", ascending=False)
else:
    filtered_movies = filtered_movies.sort_values(by="title", ascending=True)


# ---------------- RECOMMEND ----------------
if st.button("🚀 Get Recommendations"):

    if filtered_movies.empty:
        st.warning("No movies found 😢")
        st.stop()

    final_movies = filtered_movies.head(min(num_movies, len(filtered_movies)))

    st.success(f"Showing {len(final_movies)} Movies")

    cols = st.columns(2)

    for i, (_, row) in enumerate(final_movies.iterrows()):

        with cols[i % 2]:

            st.markdown(f"""
            <div style="padding:15px;border-radius:10px;background:#262730;color:white;margin:10px;">
            🎬 {row['title']}<br><br>
            🎭 {row['genres']}<br>
            📅 {row['year']}<br>
            ⭐ {round(row['avg_rating'],2)}
            </div>
            """, unsafe_allow_html=True)
            st.caption(
    "🎯 Recommended based on Hybrid Filtering (SVD + Content-Based Similarity)"
)

            st.button(
                "⭐ Add to Favourites",
                key=f"fav_{row['movieId']}",
                on_click=add_fav,
                args=(row["title"],)
            )
st.markdown("---")
st.subheader("⭐ Top Rated Movies")

top_movies = movies.sort_values(
    by="avg_rating",
    ascending=False
).head(10)

for _, row in top_movies.iterrows():
    st.write(
        f"🎬 {row['title']} | ⭐ {round(row['avg_rating'],2)}"
    )

# ---------------- FAVOURITES ----------------
st.markdown("---")
st.subheader("💖 Your Favourites")

if len(st.session_state.favourites) == 0:
    st.info("No favourites yet 😢")

else:
    for fav in st.session_state.favourites:

        col1, col2 = st.columns([4, 1])

        with col1:
            st.write("💖", fav)

        with col2:
            st.button(
                "❌ Remove",
                key=f"rm_{fav}",
                on_click=remove_fav,
                args=(fav,)
            )