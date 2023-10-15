from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

app = Flask(__name__)

def get_movie_recommendations(movie_name, movies_data, feature_vectors):
    list_of_all_titles = movies_data['title'].tolist()

    # Calculate cosine similarity
    similarity = cosine_similarity(feature_vectors)

    # Find close matches
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not find_close_match:
        return []  # No close match found

    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))

    # Sort similar movies
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    suggested_movies = []

    # Generate movie recommendations
    i = 1
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        if i < 15:
            suggested_movies.append(title_from_index)
            i += 1

    return suggested_movies

@app.route("/")
def hello_world():
    return render_template('index.html', suggested_movies=[])
# Load movie data and preprocess it here
movies_data = pd.read_csv('movies.csv')
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# Data preprocessing
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine features
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        movie_name = request.form['enter'].lower()
        app.logger.info(f"Received request for movie: {movie_name}")

        suggested_movies = get_movie_recommendations(movie_name, movies_data, feature_vectors)
        app.logger.info(f"Recommendations for movie {movie_name}: {suggested_movies}")
        return render_template('result.html', suggested_movies=suggested_movies)

if __name__ == "__main__":
    app.run(debug=True)
    logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')