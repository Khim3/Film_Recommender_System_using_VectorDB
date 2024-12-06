# Movies Recommender System

## Overview
**Movies Recommender System** is a project at school for course Machine Learning Platforms by instructor Ho Long Van.

The **Movies Recommender System** is designed to suggest movies by analyzing plot content and themes. By leveraging advanced machine learning techniques and database technologies, it offers personalized and meaningful recommendations tailored to user interests. The system focuses on semantic content rather than surface metadata, making it a unique and engaging tool for movie discovery.

## Objectives

- Provide personalized movie suggestions using advanced embedding techniques.
- Enhance the movie discovery experience with intuitive recommendations.
- Build a scalable and resource-efficient system for real-world applications.

## Features

- **Semantic Embedding Generation**: Convert movie plots into high-dimensional vectors using the Ollama Nomic text embedding model.
- **Efficient Vector Search**: Use MongoDB Atlas with HNSW indexing for fast similarity searches.
- **Natural Language Output**: Leverage Llama 3.2 local language model to transform structured data into conversational recommendations.
- **Interactive UI**: Built with Streamlit for an engaging user experience.

## Tools and Technologies

| Tool              | Purpose                                                                 |
|--------------------|-------------------------------------------------------------------------|
| **Python 3**      | Core language for system implementation.                               |
| **Streamlit**     | Create an interactive and user-friendly interface.                     |
| **MongoDB Atlas** | Store and manage vector embeddings for similarity searches.            |
| **Ollama**        | Generate text embeddings and refine recommendations using LLMs.        |

## Workflow

1. **Data Preparation**: Clean and preprocess the movie dataset for embedding generation.
2. **Embedding Creation**: Generate vector embeddings for movie plots using the Nomic-embedded-text model.
3. **Database Setup**: Store embeddings in MongoDB and enable similarity searches with HNSW indexing.
4. **Recommendation Engine**: Process user input, retrieve similar embeddings, and refine outputs with Llama 3.2.
5. **User Interface**: Display results interactively through Streamlit, allowing users to explore recommendations.

## Dataset
The dataset includes metadata for movies in genres like Western, Action, and Fantasy:
- **Attributes**: Titles, genres, runtime, cast, plot summaries, etc.
- **Custom Embeddings**: Replaced pre-generated embeddings with tailored ones for improved accuracy.

## Accomplishments

- Effective embedding generation and similarity search implementation.
- Seamless integration of AI for natural language recommendations.
- Resource-efficient design suitable for local deployments.

## Limitations

- Relies heavily on dataset quality.
- Limited to plot-based recommendations.
- Restricted by free-tier constraints of MongoDB Atlas.

## Future Work

- Expand dataset with more diverse and multilingual movies.
- Upgrade embedding models for deeper semantic understanding.
- Incorporate additional recommendation algorithms like collaborative filtering.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Khim3/Film_Recommender_System_using_VectorDB
   ```
2. Install dependencies:
   ```bash
    pip install -r requirements.txt
    ```
3. Run the application
   ```bash
    streamlit run app.py
   ```
## Contributing

This is a small project at school and is currently being developed and maintained solely by my team with 3 members. We really appreciate your interest, please give me constructive feedbacks for improving our project.
