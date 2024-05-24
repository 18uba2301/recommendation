import os
import pickle
import difflib
import logging
from logging import StreamHandler
from functools import lru_cache
from flask import Flask, render_template, request

from langdetect import detect, lang_detect_exception
from googletrans import Translator

app = Flask(__name__)

# Set up logging
app.logger.addHandler(StreamHandler())
app.logger.setLevel(logging.ERROR)

# Load cached data
try:
    with open('preprocessed_data.pkl', 'rb') as f:
        preprocessed_data = pickle.load(f)
except FileNotFoundError:
    preprocessed_data = preprocess_data()
    with open('preprocessed_data.pkl', 'wb') as f:
        pickle.dump(preprocessed_data, f)

product_names, product_images, preprocessed_names = preprocessed_data

@lru_cache(maxsize=1024)
def translate_text(text, src_lang, dest_lang='en'):
    translator = Translator()
    return translator.translate(text, src=src_lang, dest=dest_lang).text.lower()

def preprocess_data():
    chunksize = 10000
    product_names = []
    product_images = []
    preprocessed_names = []

    for dataset_path, selected_features in [
        ('mens_westernwear.csv', ['Name', 'Image']),
        ('women_footwear.csv', ['Name', 'Image']),
        ('women_westernwear.csv', ['Name', 'Image']),
        ('BigBasket2.csv', ['Name', 'Image']),
        ('applicants.csv', ['Name', 'Image']),
        ('BigBasket3.csv', ['Name', 'Image']),
        ('electronics_product1.csv', ['Name', 'Image']),
        ('electronics_product2.csv', ['Name', 'Image'])
       
    ]:
        for chunk in pd.read_csv(dataset_path, chunksize=chunksize):
            for feature in selected_features:
                chunk[feature] = chunk[feature].fillna('')
            product_names.extend(chunk['Name'])
            product_images.extend(chunk['Image'])
            preprocessed_names.extend([name.lower() for name in chunk['Name']])

    return product_names, product_images, preprocessed_names

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Review_of system.html')
def project_overview():
    return render_template('Review_of system.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        query = request.form['product_name']  # Get the input query

        # Detect the language of the query
        query_lang = detect(query)

        # Translate the query to English if not in English
        if query_lang != 'en':
            query = translate_text(query, query_lang)
        else:
            query = query.lower()  # Convert to lowercase if already in English

        # Find products that match the query
        matches = [name for name in preprocessed_names if query in name]
        matching_indices = [preprocessed_names.index(match) for match in matches]
        matching_products = [(product_names[idx], product_images[idx]) for idx in matching_indices]

        if matching_products:
            # Calculate similarity scores for matching products
            similarity_scores = [difflib.SequenceMatcher(None, query, name.lower()).ratio() for name in [p[0] for p in matching_products]]

            # Sort matching products by similarity scores
            sorted_products = sorted(zip(matching_products, similarity_scores), key=lambda x: x[1], reverse=True)

            # Get top 20 recommendations
            recommended_products = [product_info for product_info, _ in sorted_products[:20]]

            return render_template('recommendations.html', product_name=query.capitalize(), recommended_products=recommended_products)
        else:
            # Suggest similar product names using difflib
            close_matches = difflib.get_close_matches(query, product_names, n=10, cutoff=0.6)
            suggested_products = [(name, '') for name in close_matches]
            return render_template('suggestions.html', product_name=query.capitalize(), suggested_products=suggested_products)

    except lang_detect_exception as e:
        app.logger.error(f"Language detection error: {str(e)}")
        return "An error occurred during language detection. Please try again later."
    except Exception as e:
        app.logger.error(f"Error in recommend function: {str(e)}")
        return "An error occurred. Please try again later."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
