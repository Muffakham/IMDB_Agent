
from load_and_preprocess_data import load_and_preprocess_data
from create_vector_store import create_vector_store, load_vector_store
from langchain_community.vectorstores import Chroma
from query_schema import *
from agent import IMDBBot


if __name__ == "__main__":
    dataset_path = "imdb_top_1000.csv"
    df = load_and_preprocess_data(dataset_path)
    create_vector_store(df)
    vector_store = load_vector_store()
    bot = IMDBBot(df, vector_store)
    print(bot.query("Summarize the movie plots of Steven Spielberg’s top-rated sci-fi movies."))




