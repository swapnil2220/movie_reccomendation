# Movie Recommendation System

A content-based movie recommendation system built using **Python**, **Pandas**, **Scikit-Learn**, and **Streamlit**.  
The system recommends the top 5 movies similar to the one selected by the user, based on movie metadata and cosine similarity.

---

## Features

- Recommends the top 5 movies based on content similarity.
- Uses cosine similarity on movie overviews, genres, and other metadata.
- Built with TMDB dataset for high-quality movie data.
- Interactive **Streamlit** web application for easy usage.
- Fast and lightweight — works locally without external API calls.

---

## Tech Stack

- **Programming Language:** Python
- **Libraries & Frameworks:** Pandas, NumPy, Scikit-Learn, Pickle, Streamlit
- **Dataset:** TMDB 5000 Movies & Credits Dataset
- **Deployment Platform:** Streamlit Cloud (optional)

---

## Dataset

We are using the **TMDB 5000 Movies Dataset**, which includes:
- `tmdb_5000_movies.csv` — movie details, overviews, and genres.
- `tmdb_5000_credits.csv` — cast and crew information.

Dataset Source: [Kaggle - TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

---

## Project Structure

movie-recommender/
│── app.py # Streamlit app
│── model.pkl # Cosine similarity matrix
│── movie_dict.pkl # Processed movie metadata
│── model_builder.py # Script to build similarity model
│── tmdb_5000_movies.csv # TMDB movies dataset
│── tmdb_5000_credits.csv # TMDB credits dataset
│── requirements.txt # Python dependencies
│── README.md # Project documentation
│── .gitignore # Ignore unnecessary files