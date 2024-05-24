import os
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle
import difflib
from langdetect import detect, lang_detect_exception
from googletrans import Translator
import logging
from logging import StreamHandler

app = Flask(__name__)

# Set up logging
app.logger.addHandler(StreamHandler())
app.logger.setLevel(logging.ERROR)

# Initialize vectorizer
vectorizer = None

# Load cached data
try:
    with open('feature_vectors.pkl', 'rb') as f:
        feature_vectors = pickle.load(f)
    with open('original_product_names.pkl', 'rb') as f:
        original_product_names = pickle.load(f)
    with open('original_product_images.pkl', 'rb') as f:
        original_product_images = pickle.load(f)
    ann_index = NearestNeighbors(algorithm='auto', metric='cosine')
    ann_index.fit(feature_vectors)
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))  # Load vectorizer from cache
    print("Loaded existing ANN index")
except:
    print("Creating new ANN index")
    combined_features = []
    original_product_names = []
    original_product_images = []
    chunksize = 10000

    # Load data in chunks and preprocess
    for dataset_path, selected_features in [
        ('mens_westernwear.csv', ['Name', 'Image']),
        ('women_footwear.csv', ['Name', 'Image']),
        ('women_westernwear.csv', ['Name', 'Image']),
        ('BigBasket2.csv', ['Name', 'Image']),
        ('applicants.csv', ['Name', 'Image']),
        ('BigBasket3.csv', ['Name', 'Image']),
        ('electronics_product1.csv', ['Name', 'Image']),
        ('electronics_product2.csv', ['Name', 'Image']),
        ('amz_ca_total_products.csv', ['Name', 'Image'])
    ]:
        for chunk in pd.read_csv(dataset_path, chunksize=chunksize):
            for feature in selected_features:
                chunk[feature] = chunk[feature].fillna('')
            combined_features.extend(chunk['Name'].str.lower() + ' ' + chunk['Image'])
            original_product_names.extend(chunk['Name'])
            original_product_images.extend(chunk['Image'])

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Use unigrams and bigrams
    feature_vectors = vectorizer.fit_transform(combined_features)

    # Build approximate nearest neighbor index
    ann_index = NearestNeighbors(algorithm='auto', metric='cosine')
    ann_index.fit(feature_vectors)

    # Cache feature vectors, product names, product images, and vectorizer
    with open('feature_vectors.pkl', 'wb') as f:
        pickle.dump(feature_vectors, f)
    with open('original_product_names.pkl', 'wb') as f:
        pickle.dump(original_product_names, f)
    with open('original_product_images.pkl', 'wb') as f:
        pickle.dump(original_product_images, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Review_of system.html')
def project_overview():
    return render_template('Review_of system.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        if vectorizer is None:
            return "Error: Vectorizer not initialized."

        query = request.form['product_name']  # Get the input query

        # Detect the language of the query
        query_lang = detect(query)

        # Translate the query to English if not in English
        translator = Translator()
        if query_lang != 'en':
            query = translator.translate(query, dest='en').text.lower()
        else:
            query = query.lower()  # Convert to lowercase if already in English

        # Find products that match the query
        matching_products = []
        preprocessed_names = [name.lower() for name in original_product_names]
        matches = [name for name in preprocessed_names if query in name]
        matching_indices = [preprocessed_names.index(match) for match in matches]
        matching_products = [(original_product_names[idx], original_product_images[idx]) for idx in matching_indices]

        if matching_products:
            # Calculate similarity scores for matching products
            combined_features = [name.lower() + ' ' + image_url for name, image_url in matching_products]
            query_vector = vectorizer.transform(combined_features)
            distances, indices = ann_index.kneighbors(query_vector, n_neighbors=20)
            similarity_scores = 1 - distances.ravel()

            # Sort matching products by similarity scores
            sorted_products = sorted(zip(matching_products, similarity_scores), key=lambda x: x[1], reverse=True)

            # Get top 20 recommendations
            recommended_products = [product_info for product_info, _ in sorted_products[:20]]

            return render_template('recommendations.html', product_name=query.capitalize(), recommended_products=recommended_products)
        else:
            # Suggest similar product names using difflib
            close_matches = difflib.get_close_matches(query, original_product_names, n=10, cutoff=0.6)
            suggested_products = [(name, '') for name in close_matches]
            return render_template('suggestions.html', product_name=query.capitalize(), suggested_products=suggested_products)

    except lang_detect_exception as e:
        app.logger.error(f"Language detection error: {str(e)}")
        return "An error occurred during language detection. Please try again later."
    except TranslatorError as e:
        app.logger.error(f"Translation error: {str(e)}")
        return "An error occurred during translation. Please try again later."
    except Exception as e:
        app.logger.error(f"Error in recommend function: {str(e)}")
        return "An error occurred. Please try again later."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
