from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import pickle


app = Flask(__name__)

API_KEY = '3acc39754e9014eca3c0799f663b5785'
BASE_URL = 'https://api.themoviedb.org/3'

# Load the data
df = pd.read_csv('new_movies.csv', low_memory=False, encoding='utf-8', nrows=10000)

# Set up the TF-IDF
tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['concat'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Load the sentiment analysis model and vectorizer
nb_sentiment_model = pickle.load(open('nb_sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('review_vectorizer.pkl', 'rb'))

#Load the support vector machines model and vectorizer
svm_sentiment_model = pickle.load(open('svm_sentiment_model.pkl', 'rb'))
vectorizer_svm = pickle.load(open('review_vectorizer_SVM.pkl', 'rb'))


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

 #Analyze the sentiment of a single review by support vector machines model
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
    #Get movie reviews and analyze their sentiment
    url = f"{BASE_URL}/movie/{movie_id}/reviews?api_key={API_KEY}"
    print(f"Fetching Reviews URL: {url}")
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        reviews = data.get('results', [])

        # Analyze sentiment for each review
        for review in reviews:
            # Extract the review content
            review_text = review.get('content', '')
            # Add sentiment analysis to each review
            review['sentiment'] = analyze_review_sentiment_by_svm(review_text)
            print( review['sentiment'])

            # Confidence score
            review['confidence'] = 0.8 if len(review_text) > 100 else 0.5

        print(f"Processed {len(reviews)} reviews with sentiment analysis")
        return reviews

    print("No reviews found.")
    return []


@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    error_message = None
    movie_details = None
    reviews = []

    if request.method == 'POST':
        movie_name = request.form.get('movie_name')
        if movie_name:
            recommendations = get_recommendations(movie_name)
            if isinstance(recommendations, str):
                error_message = recommendations
                recommendations = []
            else:
                movie_details = get_movie_details(movie_name)
                if movie_details:
                    movie_id = movie_details['id']
                    # Get reviews with sentiment analysis
                    reviews = get_movie_reviews(movie_id)


    print("Recommendations:", recommendations)
    print(type(recommendations))

    return render_template('index.html',
                           recommendations=recommendations,
                           movie_details=movie_details,
                           reviews=reviews,
                           error_message=error_message)


@app.route('/suggest_movies', methods=['GET'])
def suggest_movies():
    partial_title = request.args.get('query', '').strip()
    if partial_title:
        matching_movies = df[df['title'].str.contains(partial_title, case=False)]['title'].head(5).tolist()
        return jsonify(matching_movies)
    return jsonify([])


if __name__ == '__main__':
    app.run(debug=True)