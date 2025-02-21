# IMDB Movie Chatbot

This project builds a chatbot using LangChain and OpenAI to answer questions about the IMDB Top 1000 movies dataset. It leverages a combination of semantic search (using ChromaDB) and structured querying (using Pandas DataFrames) to provide comprehensive and insightful responses.

## Features

* **Semantic Search:** Ask questions about movie plots and themes using natural language.
* **Structured Querying:** Filter and analyze movies based on various criteria like genre, release year, rating, director, and more.
* **Sub-Query Handling:** Breaks down complex user queries into smaller, manageable sub-queries for efficient processing.
* **Intermediate Data Storage:** Stores and utilizes intermediate results for subsequent queries, enabling multi-step analysis.
* **Chain of Thought Reasoning:** The agent employs a chain of thought process for complex query resolution.

* Agent is built using langchain. Agent type is OPENAI_FUNCTIONS.
* Underlying LLM for agent is OPEN AI based, in this case, gpt-4.
* Chroma DB is used as a vector store, with meta data filtering.
* Underlying model for embedding is text-embedding-3-small from open AI.
* Pandas is used to handle structured queries.

## Installation

1. **Clone the repository:**
2. **Install dependencies:**
    - Run `pip install -r requirements.txt` in the project directory.
3. Unzip the vector databse `chroma_db.zip` in the project directory. 
4. **Set up OpenAI API Key:**
   - Create a `.env` file in the project directory.
   - Add your OpenAI API key: `OPENAI_API_KEY=<your_api_key>`

## Usage

2. **Prepare the data:**
   - Download the `imdb_top_1000.csv` dataset and place it in the project directory.
   - Run `python run.py` to load the data, create the vector store, and initialize the chatbot.
3. **Interact with the chatbot:**
   - Use the `bot.query()` function to ask questions:

## Project Structure

* `load_and_preprocess_data.py`: Loads and cleans the IMDB dataset.
* `create_vector_store.py`: Creates and loads the ChromaDB vector store.
* `query_schema.py`: Defines Pydantic schemas for query inputs.
* `agent.py`: Implements the chatbot agent using LangChain.
* `run.py`: Main script to run the application.
