from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from stock_analyst.config.settings import settings


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits a list of documents into smaller chunks for better processing.

    Args:
        documents (List[Document]): The list of documents to be split.

    Returns:
        List[Document]: A list of smaller document chunks.
    """
    try:
        if not documents:
            raise ValueError("Document list is empty")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        print(f"Error in split_documents: {e}")
        return []


def create_vectorstore(documents: List[Document], collection_name: str, persist_directory: str) -> Chroma:
    """
    Creates a vector store using sentence embeddings for semantic retrieval.

    Args:
        documents (List[Document]): The list of documents to be stored.
        collection_name (str): The name of the vector store collection.
        persist_directory (str): The directory path to store the vector database.

    Returns:
        Chroma: A Chroma vector store instance for efficient document retrieval.
    """
    try:
        if not documents:
            raise ValueError("Document list is empty.")
        if not collection_name:
            raise ValueError("Collection name cannot be empty.")
        if not persist_directory:
            raise ValueError("Persist directory cannot be empty.")

        try:
            embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding model: {e}") from e

        try:
            vectorstore = Chroma.from_documents(
                collection_name=collection_name,
                documents=documents,
                embedding=embedding_function,
                persist_directory=persist_directory,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create vector store: {e}") from e

        return vectorstore
    except Exception as e:
        print(f"Error in create_vectorstore: {e}")
        return None
