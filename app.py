import streamlit as st
import pandas as pd
import pymongo
from sentence_transformers import SentenceTransformer

# Page layout
st.set_page_config(page_title="Movies Recommender System", page_icon="🎬", layout="wide")

# Initialize the SentenceTransformer model
model = SentenceTransformer('thenlper/gte-large', device='cpu')

# Function to create embeddings for a chosen column in the dataframe
def create_embedding(df, chosen_column):
    # Create the embeddings for the specified column
    embedding_column = chosen_column + '_embedding'
    df[embedding_column] = df[chosen_column].apply(lambda text: model.encode(text).tolist() if isinstance(text, str) and text.strip() else [])
    
    # Write the updated dataframe to CSV (for local testing)
    #df.to_csv('test_output.csv', index=False)
    st.sidebar.success(f"Embeddings created and saved for the column '{chosen_column}'!")
    
    # Debugging: Display the first few rows of the updated DataFrame
    st.write("### DataFrame with Embeddings", df.head())
    
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
        st.sidebar.success("Successfully connected to MongoDB!")
        return client
    except pymongo.errors.ConnectionFailure as e:
        st.sidebar.error(f"Failed to connect to MongoDB: {e}")
        return None

# Function to create or access a MongoDB database and collection
def create_database(client, db_name, collection_name):
    db = client[db_name]
    collection = db[collection_name]
    st.sidebar.success(f"Successfully created or accessed the database: '{db_name}' and collection: '{collection_name}'")
    return db, collection

def main():
    # File upload section
    st.sidebar.title("Settings")
    st.sidebar.subheader("Upload your own movie dataset")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    # Check if a file is uploaded
    if uploaded_file:
        # Read the uploaded file into a pandas DataFrame
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Dataset", df)

        # Display list of columns for selection
        st.sidebar.subheader("List of All Columns in the Dataset")
        chosen_column = st.sidebar.selectbox("Select a column for embedding", df.columns.tolist())
        st.sidebar.write(f"Chosen column: {chosen_column}")

        # Button to connect and upload to MongoDB
        if st.sidebar.button("Process and Upload to MongoDB"):
            df = create_embedding(df, chosen_column) 
            
            st.write("### Updated DataFrame with Embeddings", df.head())
            client = connect_mongodb()
            if client:
                db, collection = create_database(client, 'test', 'test_collection')
                collection.delete_many({})  # Clear existing data
                st.write("### Data to be Uploaded to MongoDB", df.head())

                # Upload the dataframe with embeddings to MongoDB
                collection.insert_many(df.to_dict('records'))
                st.sidebar.success("Data successfully uploaded to MongoDB!")
                client.close()  # Close the connection
    else:
        st.sidebar.write("Please upload a CSV file.")

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
