"""
Module for setting up a stateful AI agent using LangGraph and LangChain.

This module defines a reactive AI agent using LangGraph's state management 
and LangChain's AI models. It includes tools for document processing, 
vector storage, and financial data retrieval.
"""

import os
from typing import Literal, List, Union, Annotated, Sequence
from langgraph.types import Command
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
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
    """Tool to retrieve the income statement of past 5 years from the company stock ticker"""
    try:
        print("\nUSING INCOME_STMT_TOOL\n")
        fetcher = FinancialDataFetcher(query)
        return fetcher.get_income_statement()
    except Exception as e:
        print(f"Error in income_statement_tool: {e}")
        return None


@tool(args_schema=ToolSchema)
def balance_sheet_tool(query):
    """Tool to retrieve the balance sheet of past 5 years from the company stock ticker"""
    try:
        print("\nUSING BALANCE_SHEET TOOL\n")
        fetcher = FinancialDataFetcher(query)
        return fetcher.get_balance_sheet()
    except Exception as e:
        print(f"Error in balance_sheet_tool: {e}")
        return None


@tool(args_schema=ToolSchema)
def cashflow_tool(query):
    """Tool to retrieve the cash flow details of past 5 years from the company stock ticker"""
    try:
        print("\nUSING CASHFLOW TOOL\n")
        fetcher = FinancialDataFetcher(query)
        return fetcher.get_cashflow()
    except Exception as e:
        print(f"Error in cashflow_tool: {e}")
        return None


@tool(args_schema=ToolSchema)
def finance_ratio_tool(query):
    """Tool to retrieve the financial ratios/details from the company stock ticker"""
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
        streaming=True,
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize AzureChatOpenAI: {e}") from e

# Define available agents
members = ["Finance_Agent", "News_Agent", "__end__"]


# Define router type for structured output
class Router(TypedDict):
    """Worker to route to next."""
    next_worker: List[Literal["Finance_Agent", "News_Agent"]]


class ConversationalResponse(TypedDict):
    """Respond in a conversational manner. Be kind and helpful."""
    response: Annotated[str, ..., "A conversational response to the user's query"]

class FinalResponse(TypedDict):
    """
    Represents the final output of a system, which can either be:
    
    Router type, determining the next worker to handle the process.
    ConversationalResponse type, providing a user-friendly response.

    """
    final_output: Union[Router, ConversationalResponse]


# Router Agent
web_search_tool = TavilySearchResults(max_results=2)
tools = [web_search_tool]
ROUTER_AGENT_PROMPT = (
    "You are tasked with managing a conversation among the following workers: "
    f"{members}. Based on the user's request, determine which worker(s) should handle the query next. "
    "If the query involves income statement, balance sheet, cash flow statement and financial ratio, include 'Finance_Agent' in your answer. "
    "If the query pertains to news or current events, include 'News_Agent' in your answer. "
    "If the query is general, answer in conversational manner."
    "Your response should be either list of agent names or coversational response."
)
router_agent = create_react_agent(
    model, tools=tools, prompt=ROUTER_AGENT_PROMPT, response_format=FinalResponse
)


def router_node(state: MessagesState):
    """
    Determines the next agent or endpoint based on the user's query.

    This function processes the current conversation state, invokes the
    routing model, and decides whether to route the query to the Finance
    Agent, News Agent, or terminate the process.

    Args:
        state (MessagesState): The current conversation state containing messages.

    Returns:
        messages: Extracts last message of AIMessage Object.
    """
    try:
        response = router_agent.invoke(state)
        #print("\nThis is from router node: ", response)
        output = response["messages"][-1].content
        #print(f"Next Worker: {output}")
        if output == ["FINISH"]:
            output = END
        return {"messages": [
                     AIMessage(content=response["messages"][-1].content, name="Router_Agent")
                 ]}
    except Exception as e:
        print(f"Error in router_node: {e}")
        return END


# Finance Agent
tools = [income_statement_tool, balance_sheet_tool, cashflow_tool, finance_ratio_tool]
FINANCE_AGENT_PROMPT = "You are responsible to provide financial analysis of stock ticker using provided tools"

finance_agent = create_react_agent(model, tools=tools, prompt=FINANCE_AGENT_PROMPT)


def finance_node(state: MessagesState):
    """
    Processes financial queries and updates the conversation state.

    This function invokes the Finance Agent to handle financial queries,
    updates the conversation state with the agent's response, and directs
    the flow to the Final Agent.

    Args:
        state (MessagesState): The current conversation state.

    Returns:
        Command: A command updating the conversation and routing to the Final Agent.
    """
    try:
        result = finance_agent.invoke(state)
        #print("\nThis is from finance node -before command: ",result)
        command = Command(
            update={
                "messages": [
                    AIMessage(
                        content=result["messages"][-1].content, name="Finance_Agent"
                    )
                ]
            },
            goto="Final_Agent",
        )
        #print("\nThis is from finance node: ", command)
        return command
    except Exception as e:
        print(f"Error in finance_node: {e}")
        return Command(goto=END)


tools = [retriever_tool]
NEWS_AGENT_PROMPT = "You are responsible to provide lastest new analysis of stock ticker using provided tool"

news_agent = create_react_agent(model, tools=tools, prompt=NEWS_AGENT_PROMPT)


def news_node(state: MessagesState):
    """
    Processes news-related queries and updates the conversation state.

    This function invokes the News Agent to handle news queries,
    updates the conversation state with the agent's response, and routes
    the flow to the Final Agent.

    Args:
        state (MessagesState): The current conversation state.

    Returns:
        Command: A command updating the conversation and routing to the Final Agent.
    """
    try:
        result = news_agent.invoke(state)
        command = Command(
            update={
                "messages": [
                    AIMessage(content=result["messages"][-1].content, name="News_Agent")
                ]
            },
            goto="Final_Agent",
        )
        #print("\nThis is from news node: ", command)
        return command
    except Exception as e:
        print(f"Error in news_node: {e}")
        return Command(goto=END)


FINAL_AGENT_PROMPT = (
    "You are responsible for combining the outputs from the Finance, News and General Agents and providing "
    "a final summarized answer for the user."
)
# Here we create the Combine Agent. It doesn't need additional tools.
final_agent = create_react_agent(model, tools=[], prompt=FINAL_AGENT_PROMPT)


def final_node(state: MessagesState):
    """
    Aggregates responses from the Finance, News and General agents, passes them to the Final Agent,
    and returns a final summarized result.
    """
    try:
        result = final_agent.invoke(state)
        command = Command(
            update={
                "messages": [
                    AIMessage(
                        content=result["messages"][-1].content, name="Final_Agent"
                    )
                ]
            },
            goto=END,
        )
        #print("\nThis is from final node: ", command)
        return command
    except Exception as e:
        print(f"Error in final_node: {e}")
        return Command(goto=END)


def condition(state: MessagesState) -> Sequence[str]:
    """
    Defining the conditions to add as parameter in the conditional edge formation.

    Args:
        state (MessagesState): The current state containing message history.

    Returns:
        Sequence[str]: A list of agent names or ["__end__"] if no match is found.
    """
    try:
        last_message = state["messages"][-1].content
        #print("last_message:", last_message)

        if last_message == "['Finance_Agent']":
            #print("state from condition1:", state)
            return ["Finance_Agent"]

        if last_message == "['News_Agent']":
            #print("state from condition2:", state)
            return ["News_Agent"]

        if last_message == "['Finance_Agent', 'News_Agent']":
            #print("state from condition3:", state)
            return ["Finance_Agent", "News_Agent"]

        return ["__end__"]
    except (KeyError, IndexError, AttributeError) as e:
        print(f"Error in condition function: {e}")
        return ["__end__"]

memory = MemorySaver()

builder = StateGraph(MessagesState)
builder.add_edge(START, "Router_Agent")
builder.add_node("Router_Agent", router_node)
builder.add_node("Finance_Agent", finance_node)
builder.add_node("News_Agent", news_node)
builder.add_node("Final_Agent", final_node)

builder.add_conditional_edges("Router_Agent", condition, members)

builder.add_edge("Finance_Agent", "Final_Agent")
builder.add_edge("News_Agent", "Final_Agent")
builder.add_edge("Final_Agent", END)

graph = builder.compile(checkpointer=memory)

config_1 = {"configurable": {"thread_id": "1"}}

USER_QUESTION_1 = "Give your financial advise on stock TSLA?"

events = graph.stream(
    {"messages": [{"role": "user", "content": USER_QUESTION_1}]},
    config_1,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

config_2 = {"configurable": {"thread_id": "2"}}

USER_QUESTION_2 = "Hi"

events = graph.stream(
    {"messages": [{"role": "user", "content": USER_QUESTION_2}]},
    config_2,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
