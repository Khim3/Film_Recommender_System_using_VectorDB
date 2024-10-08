import streamlit as st
import pandas as pd
import pymongo
from sentence_transformers import SentenceTransformer
import torch
import os
from pymongo.mongo_client import MongoClient
from pymongo.operations import SearchIndexModel
from pymongo.errors import DuplicateKeyError
torch.cuda.empty_cache()

# Page layout
st.set_page_config(page_title="Movies Recommender System", page_icon="🎬", layout="wide")

# Initialize the SentenceTransformer model
model = SentenceTransformer('thenlper/gte-large')

# Function to create embeddings for a chosen column in the dataframe
def create_embedding(df, chosen_column):
    # Create the embeddings for the specified column
    embedding_column = chosen_column + '_embedding'
    df[embedding_column] = df[chosen_column].apply(lambda text: model.encode(text).tolist() if isinstance(text, str) and text.strip() else [])
    st.sidebar.success(f"Embeddings created for the column '{chosen_column}'!")
    return df  # Return the updated dataframe

# Function to connect to MongoDB
def connect_mongodb():
    # MongoDB credentials (replace with your actual username and password)
    username = "khim3"
    password = "RugewX0f7wVuaJ08"
    mongo_uri = f'mongodb+srv://{username}:{password}@cluster0.c6lq9.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'

    # Attempt to connect to MongoDB
    try:
        client = pymongo.MongoClient(mongo_uri)
        client.server_info()  # Test connection
        #st.sidebar.success("Successfully connected to MongoDB!")
        return client
    except pymongo.errors.ConnectionFailure as e:
        st.sidebar.error(f"Failed to connect to MongoDB: {e}")
        return None

# Function to create or access a MongoDB database and collection
def create_database(client, db_name, collection_name):
    # Check if the database exists
    if db_name in client.list_database_names():
        db = client[db_name]
        # Check if the collection exists within the database
        if collection_name in db.list_collection_names():
            collection = db[collection_name]  # Assign the existing collection
            st.sidebar.info(f"Using the existing database '{db_name}' and collection '{collection_name}'. No data upload will be performed.")
            upload_required = False
        else:
            collection = db[collection_name]  # Create a new collection if it does not exist
            st.sidebar.success(f"Successfully created a new collection: '{collection_name}' in the existing database '{db_name}'.")
            upload_required = True
    else:
        # Create a new database and collection
        db = client[db_name]
        collection = db[collection_name]
        st.sidebar.success(f"Successfully created the new database: '{db_name}' and collection: '{collection_name}'.")
        upload_required = True

    # Return the database, collection, and upload status
    return db, collection, upload_required

# Function to create a search index with a specific configuration
def create_search_index(client, db_name, collection_name, field_name, num_dimensions=1024):
    try:
        # Access the database and collection
        database = client[db_name]
        collection = database[collection_name]

        # Create the search index model with vector configuration
        search_index_model = SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",             # Use "vector" type for vector search index
                        "numDimensions": num_dimensions,  # Number of dimensions in the vector
                        "path": field_name,           # Path of the field in the document to index
                        "similarity": "cosine"        # Similarity measure for vector search (cosine, euclidean, or dotProduct)
                    }
                ]
            },
            name="vector_search_index",              # Name of the vector search index
            type="vectorSearch"                      # Specify the type as "vectorSearch"
        )

        # Create the search index in the collection
        result = collection.create_search_index(model=search_index_model)
        st.sidebar.success(f"Vector search index created successfully with result: {result}")

    except Exception as e:
        st.sidebar.error(f"Failed to create vector search index: {e}")

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
def vector_search(user_query, collection, field_name):
    # Get the embedding for the user query
    query_embedding = create_embedding_query(user_query)
    if not query_embedding:
        return 'Invalid query'
    
    # Vector search query with dynamic path based on field_name
    vector_search_stage = {
        "$vectorSearch": {
            "index": "vector_search_index",  # Ensure this matches the created search index name
            "queryVector": query_embedding,
            "path": field_name,  # Use the dynamic field name
            "numCandidates": 10,
            "limit": 5
        }
    }
    
    # Build the aggregation pipeline
    pipeline = [vector_search_stage]

    try:
        # Execute the pipeline against the collection
        results = collection.aggregate(pipeline)
        return list(results)
    except Exception as e:
        st.error(f"Failed to execute vector search: {e}")
        return []


# Function to display search results
def get_search_result(query, collection):
    get_info = vector_search(query, collection)
    search_result = ''
    
    for result in get_info:
        search_result += f"Title: {result.get('title', 'N/A')}\n"
        search_result += f"Countries: {result.get('countries', 'N/A')}\n"
        search_result += f"Genres: {result.get('genres', 'N/A')}\n"
        search_result += f"Plot: {result.get('fullplot', 'N/A')}\n"
        score = result.get('score', None)
        if score is not None:
            search_result += f"Score: {score:.3f}\n\n"
        else:
            search_result += "Score: N/A\n"
    
    return search_result

# Main function for Streamlit app
def main():
    chosen_column = None
    collection = None
    # Title and description
    st.title("Movies Recommender System")
    st.write("### Welcome to the Movies Recommender System!")

    # Sidebar settings
    st.sidebar.title("Settings")
    st.sidebar.subheader("Upload your own movie dataset")
    
    # File upload section
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    
    # Check if a file is uploaded
    if uploaded_file:
        # Read the uploaded file into a pandas DataFrame
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Dataset", df.head(10))

        # Get the uploaded file name
        file_name = os.path.splitext(uploaded_file.name)[0]
        # Display list of columns for selection
        st.sidebar.subheader("List of All Columns in the Dataset")
        chosen_column = st.sidebar.selectbox("Select a column for embedding", df.columns.tolist())
        st.sidebar.write(f"Chosen column: {chosen_column}")
        collection_name = file_name + '_' + chosen_column + "_collection"
        field_name = chosen_column + '_embedding'

        # Button to connect and upload to MongoDB
        if st.sidebar.button("Process and Upload to MongoDB"):
            df = create_embedding(df, chosen_column)  # Create embeddings for the chosen column

            # Connect to MongoDB
            client = connect_mongodb()
            if client:
                # Create the database and collection or use existing ones
                db, collection, upload_required = create_database(client, file_name, collection_name)

                if upload_required:
                    # Clear any existing data in the new collection (if it exists)
                    collection.delete_many({})
                    # Upload the dataframe with embeddings to MongoDB
                    collection.insert_many(df.to_dict('records'))
                    st.sidebar.success("Data successfully uploaded to MongoDB!")

                # Check if the index already exists before creating a new one
                create_search_index(client, db_name=file_name, collection_name=collection_name, field_name=field_name, num_dimensions=1024)


    # Text input for query
    user_query = st.text_input("Enter your query here", "")
    # Search button
    if user_query and st.button("Search"):
        # Ensure the collection is defined before performing the search
        if collection:
            information = get_search_result(user_query, collection)
            combined_info = f'### Query: {user_query}\n\n{information}'
            st.write(combined_info)
        else:
            st.write("Please upload a dataset and process it before searching.")

    # Optional: Add additional functionality like searching or credits
    credits = st.sidebar.button("Credits")
    if credits:
        st.sidebar.markdown("""
        **Nguyen Huy Bao** - ITDSIU21076  
        **Nguyen Nhat Khiem** - ITDSIU21091  
        **Trinh Binh Gold** - ITDSIU21103
        """)

if __name__ == "__main__":
    main()
