import streamlit as st
import pandas as pd
import torch
import os

from utils import *
from vectordb_handler import *

# Empty the cache for embedding and small LLM
torch.cuda.empty_cache()

# Page layout
st.set_page_config(page_title="Movies Recommender System",
                   page_icon="🎬", layout="wide")


def main():
    # Initialize session state for essential variables
    if 'show_credits' not in st.session_state:
        st.session_state.show_credits = False
    if 'collection' not in st.session_state:
        st.session_state.collection = None
    if 'field_name' not in st.session_state:
        st.session_state.field_name = None
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []  # Store all fetched results
    if 'current_display_count' not in st.session_state:
        st.session_state.current_display_count = 0  # Track number of results displayed

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
        chosen_column = st.sidebar.selectbox(
            "Select a column for embedding", df.columns.tolist())
        collection_name = file_name + '_' + chosen_column + "_collection"
        field_name = chosen_column + '_embedding'

        if st.sidebar.button("Process and Upload to VectorDB"):
            client = connect_mongodb()
            if client:
                db, collection, upload_required = create_database(
                    client, file_name, collection_name)

                if upload_required:
                    df = create_embedding(df, chosen_column)
                    collection.delete_many({})
                    collection.insert_many(df.to_dict('records'))
                    st.sidebar.success("Data is processed successfully!")
                create_search_index(
                    client, db_name=file_name, collection_name=collection_name, field_name=field_name, num_dimensions=768)

                # Save collection and field_name to session state
                st.session_state.collection = collection
                st.session_state.field_name = field_name

    user_query = st.text_input("Enter your film plot you want to see here")

    collection = st.session_state.get('collection', None)
    field_name = st.session_state.get('field_name', None)
    if "stored_results" not in st.session_state:
        st.session_state.stored_results = []  # To store all search results
    if "current_display_count" not in st.session_state:
        st.session_state.current_display_count = 0  # To track the number of displayed results

    # Use a container to manage dynamic layout
    result_container = st.container()

    if user_query and collection is not None:
        if st.button("Search"):
            # Perform a new search
            st.session_state.stored_results = vector_search(user_query, collection, field_name)
            st.session_state.current_display_count = 5  # Reset to display the first 5 results

            # Clear previously displayed results
            with result_container:
                st.session_state.displayed_results = []  # Clear previously displayed results
                if st.session_state.stored_results:
                    # Display the first 5 results
                    new_results = st.session_state.stored_results[:st.session_state.current_display_count]
                    st.session_state.displayed_results.extend(new_results)
                    display_new_results(new_results)
                else:
                    st.write("No results found or query invalid.")

        # Show more results when "Get More Results" is clicked
        if st.session_state.stored_results:
            # Button for loading more results
            if st.button("Get More Results"):
                start_index = st.session_state.current_display_count
                st.session_state.current_display_count += 5  # Increment to show the next batch of results
                end_index = st.session_state.current_display_count

                # Get the next set of results
                new_results = st.session_state.stored_results[start_index:end_index]
                st.session_state.displayed_results.extend(new_results)  # Add to already displayed results

                # Append new results below the previously displayed ones
                with result_container:
                    display_new_results(new_results)
    # Toggle credits display
    if st.sidebar.button("Credits"):
        st.session_state.show_credits = not st.session_state.show_credits

    # Show credits if toggled
    if st.session_state.show_credits:
        st.sidebar.markdown("""
        **Nguyen Huy Bao** - ITDSIU21076  
        **Nguyen Nhat Khiem** - ITDSIU21091  
        **Trinh Binh Nguyen** - ITDSIU21103
        """)


if __name__ == "__main__":
    main()