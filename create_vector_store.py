
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.schema import Document


load_dotenv()

def create_vector_store(df):
        """Create and persist Chroma vector store"""
        documents = []
        for _, row in df.iterrows():
            doc = Document(
                page_content=row['Overview'],
                metadata={
                    'Series_Title': row['Series_Title'],
                    'Released_Year': row['Released_Year'],
                    'No_of_Votes': row['No_of_Votes'],
                    'Runtime': row['Runtime'],
                    'Genre': " ".join(row['Genre']),
                    'IMDB_Rating': row['IMDB_Rating'],
                    'Meta_score': row['Meta_score'],
                    'Director': row['Director'],
                    'Gross': row['Gross'],
                    'Cast': " ".join(row['Cast'])
                }
            )
            documents.append(doc)
        Chroma.from_documents(
            documents=documents,
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory="chroma_db"
        ).persist()


def load_vector_store():
        """Load Chroma vector store"""
        return Chroma(persist_directory="chroma_db", embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
