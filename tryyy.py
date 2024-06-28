from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load data from CSV file
df = pd.read_csv('hospitals.csv')

@app.route('/')
def index():
    return render_template('index.html', recommendations=[])

@app.route('/recommend', methods=['POST'])
def recommend():
    selected_city = request.form.get('city')
    selected_specialty = request.form.get('specialty')

    # Filter hospitals based on the selected city and specialty
    city_filtered_df = df[df['location'] == selected_city]
    specialty_filtered_df = city_filtered_df[city_filtered_df['specialty'] == selected_specialty]

    if specialty_filtered_df.empty:
        message = f"No hospitals found with the selected specialty ({selected_specialty}) in {selected_city}."
        return render_template('index.html', recommendations=[], message=message)
    
    # Combine relevant features into a single text column for TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1)
    tfidf_matrix = tfidf_vectorizer.fit_transform(specialty_filtered_df['description'])

    # Compute cosine similarity between hospitals
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Function to get hospital recommendations based on cosine similarities
    def get_recommendations(hospital_id, cosine_similarities, limit=5):
        if cosine_similarities.shape[0] == 1:
            # If there's only one hospital, return its details
            return [specialty_filtered_df.iloc[0].to_dict()]

        # Sort hospitals based on similarity scores
        sim_scores = list(enumerate(cosine_similarities[hospital_id]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        limit = min(max(len(sim_scores) - 1, 1), limit)
        sim_scores = sim_scores[1:limit + 1]

        # Get indices of recommended hospitals
        hospital_indices = [i[0] for i in sim_scores]

        # Return details of recommended hospitals
        return specialty_filtered_df.iloc[hospital_indices].to_dict(orient='records')

    user_hospital_id = specialty_filtered_df.iloc[0]['hospital_id']
    recommendations = get_recommendations(user_hospital_id, cosine_similarities)

    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
