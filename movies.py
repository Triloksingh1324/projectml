# import numpy as np
# import pandas as pd
# import difflib
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import os

# os.chdir('C:\\Users\\hp\\Project')
# movies_data =pd.read_csv('movies.csv')
# movies_data.head()
# selected_features = ['genres','keywords','tagline','cast','director']
# print(selected_features)
# for feature in selected_features:
#   movies_data[feature] = movies_data[feature].fillna('')
# combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
# vectorizer = TfidfVectorizer()
# feature_vectors = vectorizer.fit_transform(combined_features)
# movie_name = input(' Enter your favourite movie name : ')
# list_of_all_titles = movies_data['title'].tolist()

# similarity = cosine_similarity(feature_vectors)
# find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
# close_match = find_close_match[0]
# index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
# similarity_score = list(enumerate(similarity[index_of_the_movie]))

# sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

# suggested_movies = []
# print('Movies suggested for you : \n')
# i = 1
# for movie in sorted_similar_movies:
#   index = movie[0]
#   title_from_index = movies_data[movies_data.index==index]['title'].values[0]
#   if (i<30):
#     suggested_movies.append(title_from_index)
#     i+=1
#     os.chdir('D:\\projectml')
#     try:
#         pickle.dump(suggested_movies, open('model.pkl', 'wb'))
#     except Exception as e:
#          print("Error:", e)
#     model=pickle.load(open('model.pkl','rb'))



import numpy as np
import pandas as pd
import difflib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie data
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

def get_movie_recommendations(movie_name):
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
        if i < 30:
            suggested_movies.append(title_from_index)
            i += 1
    
    return suggested_movies



