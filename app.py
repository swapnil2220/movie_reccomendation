import streamlit as st
import pickle
import pandas as pd

# Load the processed movie data
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('model.pkl', 'rb'))

# Streamlit App Title
st.title("🎬 Movie Recommendation System")

# Dropdown for selecting a movie
selected_movie = st.selectbox(
    "Select a movie:",
    movies['title'].values
)

# Recommendation function
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

# Button to display recommendations
if st.button("Recommend Movies"):
    recommendations = recommend(selected_movie)
    st.subheader("Top 5 Recommended Movies:")
    for movie in recommendations:
        st.write(f"🎥 {movie}")
