🎬 Hybrid Movie Recommendation System
A machine learning-based movie recommendation system that combines Collaborative Filtering and Content-Based Filtering to generate highly accurate and personalized movie suggestions.

##Project Overview##
This project builds a hybrid recommendation engine that improves recommendation quality by combining:
 *User behavior (ratings)
 *Movie content features (genres, tags, similarity)
It solves the limitations of single recommendation approaches by blending both techniques.

##Features##
 Personalized movie recommendations
 Hybrid filtering (Content + Collaborative)
 Movie similarity computation using cosine similarity
 SVD-based collaborative filtering model
 Fast prediction pipeline
 Modular project structure

##Tech Stack##
Python
Pandas / NumPy
Scikit-learn
Surprise (SVD model)
Streamlit
Pickle (model storage)

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
Hybrid recommendation systems
Feature engineering for text-based similarity
Matrix factorization (SVD)
Model serialization using Pickle
Modular ML project design
⚠️ Note
Large model files (.pkl) are excluded from GitHub due to size limits.
They can be regenerated using preprocess.py.
