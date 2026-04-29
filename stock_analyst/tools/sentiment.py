from pydantic import BaseModel
from langchain.tools import tool
from stock_analyst.utils.yfinance_utils import FinancialDataFetcher
from stock_analyst.utils.rag_utils import split_documents, create_vectorstore
from stock_analyst.config.settings import settings


class RagToolSchema(BaseModel):
    """Schema for RAG tool input — expects a question and a stock ticker symbol."""
    question: str
    ticker: str


@tool(args_schema=RagToolSchema)
def retriever_tool(question: str, ticker: str) -> str:
    """
    Tool to retrieve semantically similar documents based on a user question.

    This tool fetches the latest financial news articles for a given ticker,
    processes them into chunks, and stores them in a vector database. It then
    retrieves the most relevant articles based on the user's query.

    Args:
        question (str): The user's query related to financial news.
        ticker (str): The stock ticker symbol for retrieving relevant news.

    Returns:
        str: The most relevant extracted information or a message if no relevant
            articles are found.
    """
    try:
        print("\nUSING NEWS_RETRIEVER TOOL\n")
        fetcher = FinancialDataFetcher(ticker)
        docs = fetcher.get_latest_news()
        if not docs:
            return "No news articles found for the specified ticker."

        splits = split_documents(docs)

        collection_name = f"{ticker}_news_collection"
        persist_directory = f"{settings.chroma_db_path}/{ticker}"
        vectorstore = create_vectorstore(splits, collection_name, persist_directory)

        retriever = vectorstore.as_retriever(search_kwargs={"k": settings.retrieval_k})
        retriever_result = retriever.get_relevant_documents(question)

        if not retriever_result:
            return "No relevant information found for your question."

        return "\n\n".join(doc.page_content for doc in retriever_result)
    except Exception as e:
        print(f"Error in retriever_tool: {e}")
        return "An error occurred while processing your request."
