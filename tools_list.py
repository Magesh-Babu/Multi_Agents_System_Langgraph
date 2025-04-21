from pydantic import BaseModel
from langchain.tools import tool
from typing import Literal, List, Union, Annotated, Sequence
from rag_tool import split_documents, create_vectorstore
from finance_tool import FinancialDataFetcher
from pyscbwrapper import SCB

# Pydantic schema for the retriever tool
class RagToolSchema(BaseModel):
    """
    Schema for RAG (Retrieval-Augmented Generation) tool input.

    Defines the expected input structure for the RAG tool,
    including a question and a financial ticker symbol.
    """

    question: str
    ticker: str

# Pydantic schema for the financial ratio tools
class RatioToolSchema(BaseModel):
    """
    Schema for finance tool input.

    Defines the expected input structure for the methods of finance tool,
    which includes financial ticker symbol.
    """

    ticker: str

# Pydantic schema for the region_scb tool
class RegionToolSchema(BaseModel):
    region: Literal['Sweden', 'Greater Stockholm',
    'Greater Gothenburg',
    'Greater Malmö',
    'Stockholm production county',
    'Eastern Central Sweden',
    'Småland with the islands',
    'South Sweden',
    'West Sweden',
    'Northern Central Sweden',
    'Central Norrland',
    'Upper Norrland']


# Tool to retrieve semantically similar documents based on a user question
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
        print("\nUSING NEWS_RAG TOOL\n")
        # Fetch and process the latest news articles
        fetcher = FinancialDataFetcher(ticker)
        docs = fetcher.get_latest_news()
        if not docs:
            return "No news articles found for the specified ticker."

        # Split documents into chunks
        splits = split_documents(docs)
        print(f"Split the documents into {len(splits)} chunks.")

        # Create or load the vector store
        collection_name = f"{ticker}_news_collection"
        persist_directory = f"./chroma_db/{ticker}"
        vectorstore = create_vectorstore(splits, collection_name, persist_directory)

        # Retrieve relevant documents based on the user's question
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        retriever_result = retriever.get_relevant_documents(question)

        if not retriever_result:
            return "No relevant information found for your question."

        # Compile and return the relevant information
        return "\n\n".join(doc.page_content for doc in retriever_result)
    except Exception as e:
        print(f"Error in retriever_tool: {e}")
        return "An error occurred while processing your request."


@tool(args_schema=RatioToolSchema)
def income_statement_tool(ticker):
    """
    Tool to retrieve the income statement of past 5 years from the company's stock ticker.

    Args:
        ticker (str): The stock ticker symbol for retrieving relevant data.

    Returns:
        dataframe: The income statement data.    
    """
    try:
        print("\nUSING INCOME_STMT_TOOL\n")
        fetcher = FinancialDataFetcher(ticker)
        return fetcher.get_income_statement()
    except Exception as e:
        print(f"Error in income_statement_tool: {e}")
        return None


@tool(args_schema=RatioToolSchema)
def balance_sheet_tool(ticker):
    """
    Tool to retrieve the balance sheet of past 5 years from the company's stock ticker.

    Args:
        ticker (str): The stock ticker symbol for retrieving relevant data.

    Returns:
        dataframe: The balance sheet data.
    """
    try:
        print("\nUSING BALANCE_SHEET TOOL\n")
        fetcher = FinancialDataFetcher(ticker)
        return fetcher.get_balance_sheet()
    except Exception as e:
        print(f"Error in balance_sheet_tool: {e}")
        return None


@tool(args_schema=RatioToolSchema)
def cashflow_tool(ticker):
    """
    Tool to retrieve the cash flow details of past 5 years from the company's stock ticker

    Args:
        ticker (str): The stock ticker symbol for retrieving relevant data.

    Returns:
        dataframe: The cash flow data.    
    """
    try:
        print("\nUSING CASHFLOW TOOL\n")
        fetcher = FinancialDataFetcher(ticker)
        return fetcher.get_cashflow()
    except Exception as e:
        print(f"Error in cashflow_tool: {e}")
        return None


@tool(args_schema=RatioToolSchema)
def finance_ratio_tool(ticker):
    """
    Tool to retrieve the financial ratios/details from the company's stock ticker
    
    Args:
        ticker (str): The stock ticker symbol for retrieving relevant data.

    Returns:
        dataframe: The finance ratio data.    
    """
    try:
        print("\nUSING RATIOS TOOL\n")
        fetcher = FinancialDataFetcher(ticker)
        return fetcher.get_basic_financials()
    except Exception as e:
        print(f"Error in basic_finance_tool: {e}")
        return None


@tool(args_schema=RegionToolSchema)
def housing_price_index_tool(region):
    """
    Tool to retrieve housing price index data from Sweden Statistics.
    
    Args:
        region (str): Predefined regions of sweden in the SCB database.

    Returns:
        dataframe: The housing price index for the particular region.    
    """

    try:
        print("\nUSING HOUSING PRICE TOOL\n")
        # Initialize the SCB API wrapper in English.
        scb = SCB("en", "BO", "BO0501A", "FastpiPSRegAr")
        # Construct and send the query based on the selected dataset.
        scb.set_query(
        region=[region],
        year=["2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"],
    )
        data = scb.get_data()
        print("data from tool:", data)
        return data
    except Exception as e:
        return f"An error occurred while retrieving data: {str(e)}"

@tool(args_schema=RatioToolSchema)
def price_history_tool(ticker):
    """
    Tool to retrieve the price history of the stock for a specified period.
    
    Args:
        ticker (str): The stock ticker symbol for retrieving relevant data.

    Returns:
        dataframe: The price history data.    
    """
    try:
        print("\nUSING PRICE_HISTORY TOOL\n")
        fetcher = FinancialDataFetcher(ticker)
        data = fetcher.get_price_history()

        # 2.1 Simple Moving Average (SMA)
        data['SMA_50'] = data['Close'].rolling(window=50).mean()

        # 2.2 Relative Strength Index (RSI)
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI_14'] = 100 - (100 / (1 + rs))

        # 2.3 Moving Average Convergence Divergence (MACD)
        ema_fast = data['Close'].ewm(span=12, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD']      = ema_fast - ema_slow
        data['Signal_9']  = data['MACD'].ewm(span=9, adjust=False).mean()

        # 2.4 Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Std']    = data['Close'].rolling(window=20).std()
        data['BB_Upper']  = data['BB_Middle'] + 2 * data['BB_Std']
        data['BB_Lower']  = data['BB_Middle'] - 2 * data['BB_Std']

        return data[['Close','SMA_50','RSI_14','MACD','Signal_9','BB_Upper','BB_Middle','BB_Lower']]
    except Exception as e:
        print(f"Error in price_history_tool: {e}")
        return None    
