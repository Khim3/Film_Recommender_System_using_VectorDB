import streamlit as st
import numpy as np
import pandas as pd
import pymongo
from sentence_transformers import SentenceTransformer
# page layout
st.set_page_config(page_title="Movies Recommender System", page_icon="🎬", layout="wide")

#model = SentenceTransformer('thenlper/gte-large')

# Correct function to create embeddings for a chosen column in the dataframe
def create_embedding(df, chosen_column):
    # Ensure the chosen column exists in the dataframe
    if chosen_column not in df.columns:
        raise ValueError(f"Column '{chosen_column}' not found in the dataframe")
    # Check for valid text input in the chosen column and create embeddings
    #df['embedding'] = df[chosen_column].apply(lambda text: model.encode(text).tolist() if isinstance(text, str) and text.strip() else [])
    st.sidebar.write (f'Quoc has chosen {chosen_column} and created embeddings')
    connect_mongodb()
    


def main():

    st.title("Movies Recommender System")
    st.sidebar.title("Settings")
    
    # file upload
    st.sidebar.subheader("Upload your own movie dataset")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        movies = pd.read_csv(uploaded_file)
        st.write(movies)
        st.sidebar.subheader("List of all columns in the dataset")
        #st.sidebar.selectbox("Select a column", movies.columns.tolist())
        chosen_column = st.sidebar.selectbox("Select a column", movies.columns.tolist())
        st.sidebar.write(f"Chosen column: {chosen_column}")
    else:
        st.sidebar.write("Please upload a CSV file.")
    
    # button to create embeddings
    if st.sidebar.button("Create Embeddings") and uploaded_file:
        create_embedding(movies, chosen_column)
    # button to connect to MongoDB
    if st.sidebar.button("Connect to MongoDB"):
        connect_mongodb()
    # create a  credits button
    credits =st.sidebar.button("Credits")
    if credits:
         st.sidebar.markdown("""
        **Nguyen Huy Bao** - ITDSIU21076  
        **Nguyen Nhat Khiem** - ITDSIU21091  
        **Trinh Binh Gold** - ITDSIU21103
        """)
         
    # chat input film plot
    st.subheader("Let's find the movie you are looking for!")
    film_plot = st.text_area("Enter a film plot")
    if st.button("Submit"):
        st.write("Buoi Xuan Quoc")
         
         
def connect_mongodb():
    # MongoDB credentials (replace these with your actual username and password)
    username = "khim3"  # Replace with your MongoDB username
    password = "RugewX0f7wVuaJ08"  # Replace with your MongoDB password

    # MongoDB URI
    mongo_uri = f'mongodb+srv://{username}:{password}@cluster0.c6lq9.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'

    # Attempt to connect to MongoDB
    try:
        client = pymongo.MongoClient(mongo_uri)
        # Check if the connection is established by calling the 'server_info' method
        client.server_info()  # This command will throw an exception if the connection is not established
        st.sidebar.success("Successfully connected to MongoDB!")
        return client
    except pymongo.errors.ConnectionFailure as e:
        st.sidebar.error(f"Failed to connect to MongoDB: {e}")
        return None
if __name__ == "__main__":
    main()










