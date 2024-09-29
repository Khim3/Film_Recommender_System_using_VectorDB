import streamlit as st
import numpy as np
import pandas as pd
# page layout
st.set_page_config(page_title="Movies Recommender System", page_icon="🎬", layout="wide")

def main():
    

    # Load the data
    # movies = pd.read_csv("data/movies.csv")

    # # Create a dropdown menu for selecting a movie
    # selected_movie = st.selectbox("Select a movie", movies["title"])

    # # Display the selected movie
    # st.write("You selected the movie:", selected_movie)

    # # Display the movie's genres
    # genres = movies[movies["title"] == selected_movie]["genres"].values[0]
    # st.write("Genres:", genres)

    # # Display the movie's poster
    # poster = f"https://m.media-amazon.com/images/M/{movies[movies['title'] == selected_movie]['imdbId'].values[0]}.jpg"
    # st.image(poster, caption=selected_movie, use_column_width=True)

    # # Get the movie's index
    # movie_index = movies[movies["title"] == selected_movie].index[0]

    # # Load the similarity matrix
    # similarity_matrix = np.load("data/similarity_matrix.npy")

    # # Get the top 5 similar movies
    # similar_movies = list(enumerate(similarity_matrix[movie_index]))
    # similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    # similar_movies = similar_movies[1:6]

    # # Display the top 5 similar movies
    # st.write("Top 5 similar movies:")
    # for i, (movie, similarity) in enumerate(similar_movies):
    #     st.write(f"{i + 1}. {movies.iloc[movie]['title']} (Similarity: {similarity:.2f})")

    st.title("Movies Recommender System")
    st.sidebar.title("Settings")
    # file upload
    st.sidebar.subheader("Upload your own movie dataset")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        movies = pd.read_csv(uploaded_file)
        st.write(movies)
    else:
        st.sidebar.write("Please upload a CSV file.")
    # embedding button
    if st.sidebar.button("Embedding"):
        st.sidebar.write("Embedding")
    # chat input film plot
    st.subheader("Chat input film plot")
    film_plot = st.text_area("Enter a film plot")
    if st.button("Submit"):
        st.write("Buoi Xuan Quoc")
    # create a  credits button
    credits =st.sidebar.button("Credits")
    if credits:
        st.sidebar.write("""Nguyen Huy Bao - ITDSIU21076 
                         Nguyen Nhat Khiem - ITDSIU21091 
                         Trinh Binh Gold - ITDSIU21103""")
if __name__ == "__main__":
    main()










