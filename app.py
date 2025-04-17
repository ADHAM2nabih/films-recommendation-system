import streamlit as st
import joblib
import faiss
import pandas as pd

# Load the saved recommendation model (Pipeline)
recommendation_system = joblib.load('recommendation_system_pipeline.joblib')

# Set up the Streamlit interface
st.title("Movie Recommendation System 🎥🍿")
st.write("Enter a movie title or a similar text to get recommendations. 🤔")

# User input field
user_query = st.text_input("Enter your search query here 🔍:")

if user_query:
    # Call the recommendation system to get top recommendations based on the user query
    top_recommendations = recommendation_system.recommend(user_query, top_n=5)

    # Display the recommendations for the user
    st.write("Here are the top recommendations based on your query: 🎬✨")
    
    # Load the movie metadata
    df = pd.read_csv('movie_metadata.csv')
    
    # Display the recommended movies based on the indices returned from FAISS
    for idx in top_recommendations[0]:
        movie = df.iloc[idx]
        st.write(f"**{movie['name']}** ({movie['year']}) - Rating: {movie['rating']} ⭐ - Genre: {movie['genre']} 🎭")
