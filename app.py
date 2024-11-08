import gc
import os
import gdown
from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import pickle
from flask import Flask


app = Flask(__name__)

API_KEY = '3acc39754e9014eca3c0799f663b5785'
BASE_URL = 'https://api.themoviedb.org/3'

data_loaded = False
df = None
cosine_sim = None
nb_sentiment_model = None
svm_sentiment_model = None
vectorizer = None
vectorizer_svm = None

@app.before_request
def load_data():
    global df, cosine_sim, nb_sentiment_model, svm_sentiment_model, vectorizer, vectorizer_svm, data_loaded

    if not data_loaded:
        # URL of the file in Google Drive (for the dataset)
        file_url_movies = 'https://drive.google.com/uc?id=18gEjCisVUR5GuFCgyuCR_FFG2LBJDbvR&export=download'

        # URL of the file for the svm_sentiment_model.pkl
        file_url_svm_model = 'https://drive.google.com/uc?id=1j7O1sc9k1Law5vYRLZzjTxQL3VrNpEVZ&export=download'

        # Use gdown to download the movies dataset
        gdown.download(file_url_movies, 'new_movies.csv', quiet=False)

        # Download the svm sentiment model
        gdown.download(file_url_svm_model, 'svm_sentiment_model.pkl', quiet=False)

        # Load the DataFrame
        dtype_dict = {
            'id': 'int32',
            'title': 'str',
            'concat': 'str',
        }

        df = pd.read_csv('new_movies.csv', low_memory=False, encoding='utf-8', nrows=10000, dtype=dtype_dict, usecols=['id', 'title', 'concat'])

        # Check if the DataFrame is empty
        if df.empty:
            raise ValueError("The DataFrame is empty, cannot proceed with TF-IDF transformation.")

        # Clean 'concat' column (remove NaN values)
        df = df[df['concat'].notna()]

        # TF-IDF Vectorization
        tfidf = TfidfVectorizer(analyzer='word', stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(df['concat'])

        # Ensure the tfidf_matrix is not empty
        if tfidf_matrix.shape[0] == 0:
            raise ValueError("TF-IDF matrix is empty, check the input data.")

        # Initialize cosine similarity matrix
        chunk_size = 500
        n_chunks = len(df) // chunk_size + 1
        cosine_sim = np.zeros((len(df), len(df)), dtype='float32')

        # Loop through the data in chunks
        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(df))
            chunk = tfidf_matrix[start:end]

            # Skip empty chunks
            if chunk.shape[0] == 0:
                print(f"Warning: Empty chunk at index {i}, skipping...")
                continue

            try:
                # Compute cosine similarity for this chunk
                cosine_sim[start:end] = cosine_similarity(chunk, tfidf_matrix)
            except ValueError as e:
                print(f"Error in computing cosine similarity for chunk {i}: {e}")
                continue

        # Load models
        with open('nb_sentiment_model.pkl', 'rb') as f:
            nb_sentiment_model = pickle.load(f)
        with open('review_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('svm_sentiment_model.pkl', 'rb') as f:
            svm_sentiment_model = pickle.load(f)
        with open('review_vectorizer_SVM.pkl', 'rb') as f:
            vectorizer_svm = pickle.load(f)

        # Clear unnecessary objects to free memory
        gc.collect()

        # Set the flag to True to indicate data has been loaded
        data_loaded = True


# Analyze the sentiment of a single review by the sentiment analysis model
def analyze_review_sentiment(review_text):
    try:
        # Transform the review text using the vectorizer
        review_vector = vectorizer.transform([review_text])
        # Predict sentiment
        prediction = nb_sentiment_model.predict(review_vector)[0]
        return 'POSITIVE' if prediction == 1 else 'NEGATIVE'
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return 'UNKNOWN'

# Analyze the sentiment of a single review by support vector machines model
def analyze_review_sentiment_by_svm(review_text):
    try:
        # Transform the review text using the vectorizer
        review_vector = vectorizer_svm.transform([review_text])
        # Predict sentiment
        prediction = svm_sentiment_model.predict(review_vector)[0]
        return 'POSITIVE' if prediction == 1 else 'NEGATIVE'
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return 'UNKNOWN'


def get_recommendations(title):
    if title not in df['title'].values:
        print(f"Movie '{title}' not found in the dataset.")
        return 'Sorry! try another movie name'

    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # higher 5 results
    movie_indices = [i[0] for i in sim_scores]
    recommended_movies = []

    for movie_index in movie_indices:
        movie_id = df['title'].iloc[movie_index]
        movie_details = get_movie_details(movie_id)

        if movie_details:
            # Round the rating to one decimal place
            vote_average = movie_details.get('vote_average', 0)
            rounded_vote_average = round(vote_average, 1) if isinstance(vote_average, (int, float)) else vote_average

            recommended_movies.append({
                'title': movie_details.get('title', ''),
                "poster_path": movie_details.get('poster_path', ''),
                "overview": movie_details.get('overview', ''),
                "rating": movie_details.get('rating', ''),
                "release_date": movie_details.get('release_date', ''),
                "vote_average": rounded_vote_average,
                "vote_count": movie_details.get('vote_count', ''),
            })
    return recommended_movies


def get_movie_details(movie_name):
    url = f"{BASE_URL}/search/movie?api_key={API_KEY}&query={movie_name}"
    print(f"Fetching URL: {url}")
    response = requests.get(url)
    print(f"Response Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            movie_details = data['results'][0]
            print(f"Movie details: {movie_details}")
            return movie_details
    print("No results found.")
    return None


def get_movie_reviews(movie_id):
    # Get movie reviews and analyze their sentiment
    url = f"{BASE_URL}/movie/{movie_id}/reviews?api_key={API_KEY}"
    print(f"Fetching Reviews URL: {url}")
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        reviews = data.get('results', [])

        # Check if there are reviews
        if not reviews:
            return []

        # Analyze sentiment for each review
        for review in reviews:
            # Extract the review content
            review_text = review.get('content', '')
            # Add sentiment analysis to each review
            review['sentiment'] = analyze_review_sentiment_by_svm(review_text)

            # Confidence score
            review['confidence'] = 0.8 if len(review_text) > 100 else 0.5

        print(f"Processed {len(reviews)} reviews with sentiment analysis")
        return reviews

    print("No reviews found.")
    return []


@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        movie_title = request.form['movie_title']
        recommendations = get_recommendations(movie_title)

    return render_template('index.html', recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)
