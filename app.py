import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load data & model
df = pd.read_csv('movie_metadata.csv')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Transform content column
tfidf_matrix = tfidf.transform(df['content'])

# Streamlit UI
st.title("🎥 Movie Recommendation System 🍿")
st.write("Enter a movie description or keywords to get recommendations! ✨")

# Input from user
query = st.text_input("🔍 Enter your search:")

if query:
    # Transform query using the same vectorizer
    query_vec = tfidf.transform([query])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Get top 5 similar movie indices
    top_indices = similarity_scores.argsort()[-5:][::-1]

    st.subheader("📌 Top Recommendations:")
    for idx in top_indices:
        movie = df.iloc[idx]
        st.markdown(f"""
        **🎬 Name:** {movie['name']}  
        **🎭 Genre:** {movie['genre']}  
        **📅 Year:** {movie['year']}  
        **⭐ Rating:** {movie['rating']}
        ---
        """)
