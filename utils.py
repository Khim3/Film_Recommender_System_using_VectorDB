import ollama
import pandas as pd
import streamlit as st


def create_embedding(df, chosen_column):
    embedding_column = chosen_column + '_embedding'
    # Initialize an empty list to store embeddings
    embeddings = []

    for index, row in df.iterrows():
        text = row[chosen_column]
        if isinstance(text, str) and text.strip():  # Check for valid text
            try:
                # Generate the embedding for the text
                response = ollama.embeddings(model="nomic-embed-text", prompt=text)
                embedding = response.get("embedding", [])
            except Exception as e:
                embedding = []  # Fallback to empty if there's an error
                print(f"Error embedding row {index}: {e}")
        else:
            embedding = []  # Fallback to empty if text is invalid

        embeddings.append(embedding)  # Append the embedding to the list

    # Assign the embeddings list to the DataFrame
    df[embedding_column] = embeddings
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

    Please rewrite this information as a smooth, engaging description for a movie recommendation in natural way (remove system-like parts), make sure it's different from each one.
    """
    response = ollama.generate(
        model="llama3.2",
        prompt=prompt
    )
    return response['response']

def display_new_results(results):
    """Displays only the new results."""
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