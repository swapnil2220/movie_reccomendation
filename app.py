import streamlit as st
import pandas as pd
import pickle
import requests
import os

st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# --- Caching dataset ---
@st.cache_data(show_spinner=True)
def load_data():
    movies_url = "https://drive.google.com/uc?export=download&id=13bUWqw5bWNNVD8B1Ne3SVS_ryOTM87XO"
    credits_url = "https://drive.google.com/uc?export=download&id=1Ru8Unf-s34iB0ghoeByqnK4lkBOmJXQS"

    movies = pd.read_csv(movies_url)
    credits = pd.read_csv(credits_url)
    return movies, credits

# --- Caching model loading ---
@st.cache_resource
def load_model():
    model_url = "https://drive.google.com/uc?export=download&id=1k8dLK-6MHuAf6quBWMujOJ56_9wfkHv0"
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        r = requests.get(model_url)
        with open(model_path, "wb") as f:
            f.write(r.content)
    return pickle.load(open(model_path, "rb"))

@st.cache_resource
def load_movie_dict():
    dict_url = "https://drive.google.com/uc?export=download&id=1O8KmAF7ARWoJLxO2MTDVgWoVDYcCCNdU"
    dict_path = "movie_dict.pkl"
    if not os.path.exists(dict_path):
        r = requests.get(dict_url)
        with open(dict_path, "wb") as f:
            f.write(r.content)
    return pickle.load(open(dict_path, "rb"))

# --- Load everything ---
movies, credits = load_data()
model = load_model()
movie_dict = load_movie_dict()
movies_df = pd.DataFrame(movie_dict)

# --- Recommendation function ---
def recommend(movie):
    movie_index = movies_df[movies_df['title'] == movie].index[0]
    distances = model[movie_index]
    recommended_movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = []
    for i in recommended_movie_list:
        recommended_movies.append(movies_df.iloc[i[0]].title)
    return recommended_movies

# --- Streamlit UI ---
st.title("🎥 Movie Recommendation System")

selected_movie = st.selectbox("Select a movie to get recommendations:", movies_df['title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.subheader("Recommended Movies:")
    for movie in recommendations:
        st.write("➡", movie)
