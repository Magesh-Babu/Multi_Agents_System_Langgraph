"""
Module for setting up a stateful AI agent using LangGraph and LangChain.

This module defines a reactive AI agent using LangGraph's state management 
and LangChain's AI models. It includes tools for document processing, 
vector storage, and financial data retrieval.
"""
import os
from typing import Literal
from langgraph.types import Command
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage
from langchain_openai.chat_models import AzureChatOpenAI
from pydantic import BaseModel
from langchain.tools import tool
from rag_tool import split_documents, create_vectorstore
from finance_tool import FinancialDataFetcher

AZURE_GPT_API = os.getenv("AZURE_GPT_API")
AZURE_GPT_ENDPOINT = os.getenv("AZURE_GPT_ENDPOINT")

# Pydantic schema for the retriever tool
class RagToolSchema(BaseModel):
    """
    Schema for RAG (Retrieval-Augmented Generation) tool input.

    Defines the expected input structure for the RAG tool, 
    including a question and a financial ticker symbol.
    """
    question: str
    ticker: str

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

class ToolSchema(BaseModel):
    """
    Schema for finance tool input.

    Defines the expected input structure for the methods of finance tool, 
    which includes financial ticker symbol.
    """
    query: str

@tool(args_schema=ToolSchema)
def income_statement_tool(query):
    '''Tool to retrieve the income statement of past 5 years from the company stock ticker'''
    try:
        print("\nUSING INCOME_STMT_TOOL\n")
        fetcher = FinancialDataFetcher(query)
        return fetcher.get_income_statement()
    except Exception as e:
        print(f"Error in income_statement_tool: {e}")
        return None

@tool(args_schema=ToolSchema)
def balance_sheet_tool(query):
    '''Tool to retrieve the balance sheet of past 5 years from the company stock ticker'''
    try:
        print("\nUSING BALANCE_SHEET TOOL\n")
        fetcher = FinancialDataFetcher(query)
        return fetcher.get_balance_sheet()
    except Exception as e:
        print(f"Error in balance_sheet_tool: {e}")
        return None

@tool(args_schema=ToolSchema)
def cashflow_tool(query):
    '''Tool to retrieve the cash flow details of past 5 years from the company stock ticker'''
    try:
        print("\nUSING CASHFLOW TOOL\n")
        fetcher = FinancialDataFetcher(query)
        return fetcher.get_cashflow()
    except Exception as e:
        print(f"Error in cashflow_tool: {e}")
        return None

@tool(args_schema=ToolSchema)
def basic_finance_tool(query):
    '''Tool to retrieve the financial ratios/details from the company stock ticker'''
    try:
        print("\nUSING RATIOS TOOL\n")
        fetcher = FinancialDataFetcher(query)
        return fetcher.get_basic_financials()
    except Exception as e:
        print(f"Error in basic_finance_tool: {e}")
        return None

try:
    model = AzureChatOpenAI(
        azure_endpoint=AZURE_GPT_ENDPOINT,
        azure_deployment="gpt-4o-mini",
        api_version="2024-05-01-preview",
        api_key=AZURE_GPT_API,
        temperature=0.2,
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize AzureChatOpenAI: {e}") from e

# Define available agents
members = ["Finance_Agent", "News_Agent"]

# Create system prompt for supervisor
system_prompt = (
    "You are a financial advisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

# Define router type for structured output
class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["Finance_Agent", "News_Agent", "FINISH"]
    

def advisor_node(state: MessagesState) -> Command[Literal["Finance_Agent", "News_Agent", "__end__"]]:
    """
    Determines the next agent or endpoint based on the user's query.

    This function processes the current conversation state, invokes the 
    routing model, and decides whether to route the query to the Finance 
    Agent, News Agent, or terminate the process.

    Args:
        state (MessagesState): The current conversation state containing messages.

    Returns:
        Command: A command directing the next step in the workflow.
    """
    try:
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = model.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        print(f"Next Worker: {goto}")
        if goto == "FINISH":
            goto = END
        return Command(goto=goto)
    except Exception as e:
        print(f"Error in advisor_node: {e}")
        return Command(goto=END)


tools = [income_statement_tool, balance_sheet_tool, cashflow_tool, basic_finance_tool]
FINANCE_AGENT_PROMPT = "You are responsible to provide financial analysis of stock ticker using provided tools"

finance_agent = create_react_agent(model, tools=tools, prompt=FINANCE_AGENT_PROMPT)

def finance_node(state: MessagesState) -> Command[Literal["Advisor_Agent"]]:
    """
    Processes financial queries and updates the conversation state.

    This function invokes the Finance Agent to handle financial queries, 
    updates the conversation state with the agent's response, and directs 
    the flow back to the Advisor Agent.

    Args:
        state (MessagesState): The current conversation state.

    Returns:
        Command: A command updating the conversation and routing to the Advisor Agent.
    """
    try:
        result = finance_agent.invoke(state)
        return Command(
            update={
                "messages": [
                    AIMessage(content=result["messages"][-1].content, name="Finance_Agent")
                ]
            },
            goto="Advisor_Agent",
        )
    except Exception as e:
        print(f"Error in finance_node: {e}")
        return Command(goto=END)


tools = [retriever_tool]
NEWS_AGENT_PROMPT = "You are responsible to provide lastest new analysis of stock ticker using provided tool"

news_agent = create_react_agent(model, tools=tools, prompt=NEWS_AGENT_PROMPT)

def news_node(state: MessagesState) -> Command[Literal["Advisor_Agent"]]:
    """
    Processes news-related queries and updates the conversation state.

    This function invokes the News Agent to handle news queries, 
    updates the conversation state with the agent's response, and routes 
    the flow back to the Advisor Agent.

    Args:
        state (MessagesState): The current conversation state.

    Returns:
        Command: A command updating the conversation and routing to the Advisor Agent.
    """
    try:
        result = news_agent.invoke(state)
        return Command(
            update={
                "messages": [
                    AIMessage(content=result["messages"][-1].content, name="News_Agent")
                ]
            },
            goto="Advisor_Agent",
        )
    except Exception as e:
        print(f"Error in news_node: {e}")
        return Command(goto=END)
    


memory = MemorySaver()

builder = StateGraph(MessagesState)
builder.add_edge(START, "Advisor_Agent")
builder.add_node("Finance_Agent", finance_node)
builder.add_node("News_Agent", news_node)
builder.add_node("Advisor_Agent", advisor_node)

graph = builder.compile(checkpointer=memory)
config_1 = {"configurable": {"thread_id": "1"}}

USER_QUESTION = "Give your financial advise on stock TSLA?"

events = graph.stream(
    {"messages": [{"role": "user", "content": USER_QUESTION}]},
    config_1,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

config_2 = {"configurable": {"thread_id": "2"}}

user_input = "Hi"

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config_2,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()