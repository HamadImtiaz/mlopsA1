from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
import logging

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

# Load the dataset
df = pd.read_excel("C:\\Users\\Dell\\Downloads\\reviews_dataset.xlsx")
df = df.rename(columns={"category": "Category", "district": "District"})

# Encoding the categorical variables
encoder = OrdinalEncoder()
categorical_features = ["District", "Category"]
x = df[categorical_features].values
y = df["Place Name"].values
x = encoder.fit_transform(x)

# Train the model
model = RandomForestClassifier(random_state=1)
model.fit(x, y)

# Function to calculate sentiment
def calculate_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 1  # Positive
    elif analysis.sentiment.polarity == 0:
        return 0  # Neutral
    else:
        return -1  # Negative

# Sentiment analysis endpoint
@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()
        review = data.get('review')
        if review is None:
            return jsonify({'error': 'Review not provided'}), 400
        sentiment = calculate_sentiment(review)
        return jsonify({'sentiment': sentiment})
    except Exception as e:
        app.logger.error(f"Error in analyze_sentiment: {str(e)}")
        return jsonify({'error': 'An error occurred during sentiment analysis'}), 500

# Recommendation endpoint
@app.route('/recommend', methods=['POST'])
def recommend_places():
    try:
        data = request.get_json()
        district = data.get('district')
        category = data.get('category')
        if not district or not category:
            return jsonify({'error': 'District or category not provided'}), 400
        
        new_instance = encoder.transform([[district, category]])
        probabilities = model.predict_proba(new_instance)
        top_5_places = np.argsort(probabilities[0])[-5:][::-1]
        unique_place_names_array = df["Place Name"].unique()
        place_names = unique_place_names_array.tolist()
        recommended_places = [place_names[i] for i in top_5_places]
        return jsonify({'recommended_places': recommended_places})
    except Exception as e:
        app.logger.error(f"Error in recommend_places: {str(e)}")
        return jsonify({'error': 'An error occurred during recommendation'}),500

        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
