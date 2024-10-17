import streamlit as st
import pandas as pd
import pymongo
import torch
import os
from sentence_transformers import SentenceTransformer
from pymongo.operations import SearchIndexModel
from pymongo.collection import Collection

torch.cuda.empty_cache()

# Page layout
st.set_page_config(page_title="Movies Recommender System", page_icon="🎬", layout="wide")

# Initialize the SentenceTransformer model
model = SentenceTransformer('thenlper/gte-large')

# Function to create embeddings for a chosen column in the dataframe
def create_embedding(df, chosen_column):
    embedding_column = chosen_column + '_embedding'
    df[embedding_column] = df[chosen_column].apply(lambda text: model.encode(text).tolist() if isinstance(text, str) and text.strip() else [])
    return df  # Return the updated dataframe

# Function to connect to MongoDB
def connect_mongodb():
    username = "khim3"
    password = "RugewX0f7wVuaJ08"
    mongo_uri = f'mongodb+srv://{username}:{password}@cluster0.c6lq9.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
    try:
        client = pymongo.MongoClient(mongo_uri)
        client.server_info()  # Test connection
        return client
    except pymongo.errors.ConnectionFailure as e:
        st.sidebar.error(f"Failed to connect to MongoDB: {e}")
        return None

# Function to create or access a MongoDB database and collection
def create_database(client, db_name, collection_name):
    if db_name in client.list_database_names():
        db = client[db_name]
        if collection_name in db.list_collection_names():
            collection = db[collection_name]  # Assign the existing collection
            upload_required = False
        else:
            collection = db[collection_name]  # Create a new collection if it does not exist
            upload_required = True
    else:
        db = client[db_name]
        collection = db[collection_name]
        upload_required = True

    return db, collection, upload_required

# Function to create a search index with a specific configuration
def create_search_index(client, db_name, collection_name, field_name, num_dimensions=1024):
    try:
        database = client[db_name]
        collection = database[collection_name]
        search_index_model = SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "numDimensions": num_dimensions,
                        "path": field_name,
                        "similarity": "cosine"
                    }
                ]
            },
            name="vector_search_index",
            type="vectorSearch"
        )
        result = collection.create_search_index(model=search_index_model)
        st.sidebar.success("Search index created successfully!")
    except Exception as e:
        st.sidebar.info(f"Search index already exists !!")

# Function to create an embedding for a text query
def create_embedding_query(text: str) -> list[float]:
    if not text or not text.strip():
        return []
    try:
        embedding = model.encode(text)
        return embedding.tolist()
    except Exception as e:
        st.error(f"Error creating embedding: {e}")
        return []

# Function to perform vector search in MongoDB
def vector_search(user_query, collection: Collection, field):
    query_embedding = create_embedding_query(user_query)
    if not query_embedding:
        st.error("Invalid query or empty embedding")
        return 'Invalid query'
    
    vector_search_stage = {
        "$vectorSearch": {
            "index": "vector_search_index",
            "queryVector": query_embedding,
            "path": field,
            "numCandidates": 10,
            "limit": 5
        }    
    }
    unset_stage = {
        '$unset': field
    }
    
    project_stage = {
        '$project': {
            '_id': 0,
            'fullplot': 1,
            'title': 1,
            'director': 1,
            'countries': 1,
            'genres': 1,
            'score': {
                '$meta': 'vectorSearchScore'
                }
            
        }
    }
    
    pipeline = [vector_search_stage, unset_stage, project_stage]

    try:
        results = collection.aggregate(pipeline)
        return list(results)
    except Exception as e:
        st.error(f"Search index already exists!!")
        return []

# Function to display search results
def get_search_result(query, collection, field):
    if collection is None:
        st.error("Collection is not defined. Please upload data first.")
        return ""
    results = vector_search(query, collection, field)
    search_result = ''
    if results:
        for result in results:
            search_result += f"Title: {result.get('title', 'N/A')}\n"
            search_result += f"Countries: {result.get('countries', 'N/A')}\n"
            search_result += f"Genres: {result.get('genres', 'N/A')}\n"
            search_result += f"Plot: {result.get('fullplot', 'N/A')}\n"
            
            score = result.get('score', None)
            if score is not None:
                search_result += f"Score: {score:.3f}\n\n"
            else:
                search_result += "Score: N/A\n"
    else:
        search_result = "No results found or query invalid."
    
    return search_result

def display_search_results(results):
    for result in results:
        st.write(f"**Title:** {result.get('title', 'N/A')}")
        st.write(f"**Countries:** {result.get('countries', 'N/A')}")
        st.write(f"**Genres:** {result.get('genres', 'N/A')}")
        st.write(f"**Plot:** {result.get('fullplot', 'N/A')}")

        score = result.get('score', None)
        if score is not None:
            st.write(f"**Score:** {score:.3f}")
        else:
            st.write("**Score:** N/A")

        st.write("---")

# Main function for Streamlit app
def main():
    # Initialize session state variables for the collection and field_name
    if 'collection' not in st.session_state:
        st.session_state.collection = None
    if 'field_name' not in st.session_state:
        st.session_state.field_name = None

    st.title("Movies Recommender System")
    st.write("### Welcome to the Movies Recommender System!")

    st.sidebar.title("Settings")
    st.sidebar.subheader("Upload your own movie dataset")

    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Dataset", df.head(10))
        file_name = os.path.splitext(uploaded_file.name)[0]
        st.sidebar.subheader("List of All Columns in the Dataset")
        chosen_column = st.sidebar.selectbox("Select a column for embedding", df.columns.tolist())
        st.sidebar.write(f"Chosen column: {chosen_column}")
        collection_name = file_name + '_' + chosen_column + "_collection"
        field_name = chosen_column + '_embedding'

        if st.sidebar.button("Process and Upload to MongoDB"):
            df = create_embedding(df, chosen_column)
            client = connect_mongodb()
            if client:
                db, collection, upload_required = create_database(client, file_name, collection_name)

                if upload_required:
                    collection.delete_many({})
                    collection.insert_many(df.to_dict('records'))
                    st.sidebar.success("Data successfully uploaded to MongoDB!")

                create_search_index(client, db_name=file_name, collection_name=collection_name, field_name=field_name, num_dimensions=1024)

                # Save collection and field_name to session state
                st.session_state.collection = collection
                st.session_state.field_name = field_name

    user_query = st.text_input("Enter your query here")

    # Retrieve collection and field_name from session state
    collection = st.session_state.collection
    field_name = st.session_state.field_name

    # Ensure collection is defined before performing the search
    if user_query and collection is not None and st.button("Search"):
        results = vector_search(user_query, collection, field_name)
        if results:
            display_search_results(results)
    #  if user_query and collection is not None and st.button("Search"):
    #     information = get_search_result(user_query, collection, field_name)
    #     combined_info = f'### Query: {user_query}\n\n{information}'
    #     st.write(combined_info)

    if st.sidebar.button("Credits"):
        st.sidebar.markdown("""
        **Nguyen Huy Bao** - ITDSIU21076  
        **Nguyen Nhat Khiem** - ITDSIU21091  
        **Trinh Binh Gold** - ITDSIU21103
        """)

if __name__ == "__main__":
    main()
