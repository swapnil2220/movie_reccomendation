import streamlit as st
import pandas as pd
import pickle
import requests

st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# --- Caching data for faster performance ---
@st.cache_data
def load_data():
    # Replace with your actual Google Drive file IDs
    movies_url = "https://drive.google.com/uc?export=download&id=13bUWqw5bWNNVD8B1Ne3SVS_ryOTM87XO"
    credits_url = "https://drive.google.com/uc?export=download&id=1Ru8Unf-s34iB0ghoeByqnK4lkBOmJXQS"

    movies = pd.read_csv(movies_url)
    credits = pd.read_csv(credits_url)
    return movies, credits

# --- Caching model loading ---
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

@st.cache_resource
def load_movie_dict():
    return pickle.load(open("movie_dict.pkl", "rb"))

# --- Load datasets and models ---
st.write("Loading dataset and model, please wait...")
movies, credits = load_data()
model = load_model()
movie_dict = load_movie_dict()

# Convert dict to dataframe for easy lookup
movies_df = pd.DataFrame(movie_dict)

# --- Recommendation logic ---
def recommend(movie):
    movie_index = movies_df[movies_df['title'] == movie].index[0]
    distances = model[movie_index]
    recommended_movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    for i in recommended_movie_list:
        recommended_movies.append(movies_df.iloc[i[0]].title)
    return recommended_movies

# --- Streamlit UI ---
st.title("🎥 Movie Recommendation System")

selected_movie = st.selectbox(
    "Select a movie to get recommendations:",
    movies_df['title'].values
)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    st.subheader("Recommended Movies:")
    for movie in recommendations:
        st.write("✅", movie)
