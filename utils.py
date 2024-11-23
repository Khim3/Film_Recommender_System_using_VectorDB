import ollama
import pandas as pd
import streamlit as st


def create_embedding(df, chosen_column):
    embedding_column = chosen_column + '_embedding'
    df[embedding_column] = df[chosen_column].apply(lambda text: ollama.embeddings(
        model="nomic-embed-text", prompt=text)["embedding"] if isinstance(text, str) and text.strip() else [])
    return df

# Function to create an embedding for a text query


def create_embedding_query(text: str) -> list[float]:
    if not text or not text.strip():
        return []
    try:
        # Fetch embedding from Ollama
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        embedding = response.get("embedding", [])
        return embedding
    except Exception as e:
        st.error(f"Error creating embedding: {e}")
        return []


def generate_smooth_description(structured_data):
    prompt = f"""
    Title: {structured_data['title']}
    Countries: {', '.join(structured_data['countries'])}
    Genres: {', '.join(structured_data['genres'])}
    Plot: {structured_data['fullplot']}

    Please rewrite this information as a smooth, engaging description for a movie recommendation.
    """
    response = ollama.generate(
        model="llama3.2",
        prompt=prompt
    )
    return response['response']

def display_search_results(results):
    """Displays all the accumulated results."""
    for result in results:
        smooth_description = generate_smooth_description(result)
        st.write(smooth_description)

        # Display poster if available
        poster_url = result.get('poster', None)
        if poster_url:
            st.image(poster_url, caption="Movie Poster", width=250)
        else:
            st.write("No poster available")
        st.write("---")