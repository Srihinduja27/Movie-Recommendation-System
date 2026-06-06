🎬 Hybrid Movie Recommendation System
A machine learning-based movie recommendation system that combines Collaborative Filtering and Content-Based Filtering to generate highly accurate and personalized movie suggestions.

##Project Overview##
This project builds a hybrid recommendation engine that improves recommendation quality by combining:
 *User behavior (ratings)
 *Movie content features (genres, tags, similarity)
It solves the limitations of single recommendation approaches by blending both techniques.

##Features##
 1.Personalized movie recommendations
 2.Hybrid filtering (Content + Collaborative)
 3.Movie similarity computation using cosine similarity
 4.SVD-based collaborative filtering model
 5.Fast prediction pipeline
 6.Modular project structure

##Tech Stack##
1.Python
2.Pandas / NumPy
3.Scikit-learn
4.Surprise (SVD model)
5.Streamlit
6.Pickle (model storage)

##Project Structure##
Hybrid-Movie-Recommendation-System/
│
├── app.py
├── preprocess.py
├── content_based.py
├── collaborative.py
├── hybrid.py
├── metrics.py
├── test.py
│
├── models/
│   ├── cosine_sim.pkl        (not uploaded to GitHub)
│   ├── svd_model.pkl         (not uploaded to GitHub)
│
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│
├── __pycache__/              (ignored in .gitignore)
│
├── requirements.txt
├── .gitignore
└── README.md

##Workflow##
1.Data preprocessing (movies + ratings)
2.Content-based similarity calculation
3.Collaborative filtering using SVD
4.Hybrid score generation
5.Final recommendation output

Recommended Movies:
1. Inception
2. Interstellar
3. The Dark Knight
4. The Matrix

##Key Learnings##
1.Hybrid recommendation systems
2.Feature engineering for text-based similarity
3.Matrix factorization (SVD)
4.Model serialization using Pickle
5.Modular ML project design
⚠️ Note
Large model files (.pkl) are excluded from GitHub due to size limits.
They can be regenerated using preprocess.py.
