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

AZURE_GPT_API = os.getenv("AZURE_GPT_API")
AZURE_GPT_ENDPOINT = os.getenv("AZURE_GPT_ENDPOINT")


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
members = ["Finance_Agent", "News_Agent", "Real_Estate_Agent", "__end__"]


# Define router type for structured output
class Router(TypedDict):
    """Worker to route to next."""
    next_worker: List[Literal["Finance_Agent", "News_Agent", "Real_Estate_Agent"]]


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
router_agent_tools = [web_search_tool]
ROUTER_AGENT_PROMPT = (
    "You are tasked with managing a conversation among the following workers: "
    f"{members}. Based on the user's request, determine which worker(s) should handle the query next. "
    "If the query involves income statement, balance sheet, cash flow statement and financial ratio, include 'Finance_Agent' in your answer. "
    "If the query pertains to news or current events, include 'News_Agent' in your answer. "
    "If the query is about housing price situation in Sweden, include 'Real_Estate_Agent' in your answer. "
    "If the query is general, answer in conversational manner."
    "Your response should be either list of agent names or coversational response."
)
router_agent = create_react_agent(
    model, tools=router_agent_tools, prompt=ROUTER_AGENT_PROMPT, response_format=FinalResponse
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
finance_agent_tools = [income_statement_tool, balance_sheet_tool, cashflow_tool, finance_ratio_tool]
FINANCE_AGENT_PROMPT = "You are responsible to provide financial analysis of stock ticker using provided tools"

finance_agent = create_react_agent(model, tools=finance_agent_tools, prompt=FINANCE_AGENT_PROMPT)


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


new_agent_tools = [retriever_tool]
NEWS_AGENT_PROMPT = "You are responsible to provide lastest new analysis of stock ticker using provided tool"

news_agent = create_react_agent(model, tools=new_agent_tools, prompt=NEWS_AGENT_PROMPT)


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

real_estate_agent_tools = [housing_price_index_tool, web_search_tool]
REAL_ESTATE_AGENT_PROMPT = "You are responsible to provide housing price analysis in sweden using provided tool"

real_estate_agent = create_react_agent(model, tools=real_estate_agent_tools, prompt=REAL_ESTATE_AGENT_PROMPT)

def real_estate_node(state: MessagesState):
    """
    Processes housing price-related queries and updates the conversation state.

    This function invokes the Real Estate Agent to handle housing queries,
    updates the conversation state with the agent's response, and routes
    the flow to the Final Agent.

    Args:
        state (MessagesState): The current conversation state.

    Returns:
        Command: A command updating the conversation and routing to the Final Agent.
    """
    try:
        result = real_estate_agent.invoke(state)
        command = Command(
            update={
                "messages": [
                    AIMessage(content=result["messages"][-1].content, name="Real_Estate_Agent")
                ]
            },
            goto=END,
        )
        #print("\nThis is from real_estate_node: ", command)
        return command
    except Exception as e:
        print(f"Error in real_estate_node: {e}")
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

        if last_message == "['Real_Estate_Agent']":
            #print("state from condition3:", state)
            return ["Real_Estate_Agent"]

        if last_message == "['Finance_Agent', 'News_Agent']":
            #print("state from condition4:", state)
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
builder.add_node("Real_Estate_Agent", real_estate_node)
builder.add_node("Final_Agent", final_node)

builder.add_conditional_edges("Router_Agent", condition, members)

builder.add_edge("Finance_Agent", "Final_Agent")
builder.add_edge("News_Agent", "Final_Agent")
builder.add_edge("Real_Estate_Agent", END)
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
