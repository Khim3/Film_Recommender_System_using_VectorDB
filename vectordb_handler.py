import streamlit as st
from pymongo.operations import SearchIndexModel
from pymongo.collection import Collection
import pymongo
from utils import *
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
    
# Function to create a search index with a specific configuration
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

def create_search_index(client, db_name, collection_name, field_name, num_dimensions=768):
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
       # st.sidebar.success("Search index created successfully!")
    except Exception:
        st.sidebar.info("Search index already exists!")
        
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
            'poster': 1,
            'score': {
                '$meta': 'vectorSearchScore'
            }
        }
    }

    pipeline = [vector_search_stage, unset_stage, project_stage]

    try:
        results = collection.aggregate(pipeline)
        return list(results)
    except Exception:
        st.error("Error performing vector search!")
        return []

