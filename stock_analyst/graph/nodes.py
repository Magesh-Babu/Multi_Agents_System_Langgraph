from typing import Sequence
from langgraph.graph import MessagesState, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langchain_core.messages import AIMessage
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from stock_analyst.config.settings import settings
from stock_analyst.graph.state import FinalResponse
from stock_analyst.agents.prompts import (
    ROUTER_AGENT_PROMPT,
    FUNDAMENTAL_ANALYSIS_AGENT_PROMPT,
    SENTIMENT_ANALYSIS_AGENT_PROMPT,
    TECHNICAL_ANALYSIS_AGENT_PROMPT,
    RISK_ASSESSMENT_AGENT_PROMPT,
    REAL_ESTATE_AGENT_PROMPT,
    FINAL_AGGREGATOR_AGENT_PROMPT,
)
from stock_analyst.tools.financial import (
    income_statement_tool,
    balance_sheet_tool,
    cashflow_tool,
    finance_ratio_tool,
    risk_assessment_tool,
)
from stock_analyst.tools.technical import technical_indicators_tool
from stock_analyst.tools.sentiment import retriever_tool
from stock_analyst.tools.real_estate import housing_price_index_tool

# Shared model instance
model = AzureChatOpenAI(
    azure_endpoint=settings.azure_gpt_endpoint,
    azure_deployment=settings.deployment_name,
    api_version=settings.api_version,
    api_key=settings.azure_gpt_api,
    temperature=settings.temperature,
    streaming=True,
)

# Shared web search tool
web_search_tool = TavilySearchResults(max_results=2)

# Agent instances (initialized once at module load)
router_agent = create_react_agent(
    model,
    tools=[web_search_tool],
    prompt=ROUTER_AGENT_PROMPT,
    response_format=FinalResponse,
)

fundamental_analysis_agent = create_react_agent(
    model,
    tools=[income_statement_tool, balance_sheet_tool, cashflow_tool, finance_ratio_tool],
    prompt=FUNDAMENTAL_ANALYSIS_AGENT_PROMPT,
)

sentiment_analysis_agent = create_react_agent(
    model,
    tools=[retriever_tool],
    prompt=SENTIMENT_ANALYSIS_AGENT_PROMPT,
)

technical_analysis_agent = create_react_agent(
    model,
    tools=[technical_indicators_tool],
    prompt=TECHNICAL_ANALYSIS_AGENT_PROMPT,
)

risk_assessment_agent = create_react_agent(
    model,
    tools=[risk_assessment_tool],
    prompt=RISK_ASSESSMENT_AGENT_PROMPT,
)

real_estate_agent = create_react_agent(
    model,
    tools=[housing_price_index_tool],
    prompt=REAL_ESTATE_AGENT_PROMPT,
)

final_agent = create_react_agent(
    model,
    tools=[web_search_tool],
    prompt=FINAL_AGGREGATOR_AGENT_PROMPT,
)

# Module-level variable shared between router_node and condition
_router_response = {}


def router_node(state: MessagesState):
    """
    Determines the next agent or endpoint based on the user's query.

    Args:
        state (MessagesState): The current conversation state containing messages.

    Returns:
        messages: Extracts last message of AIMessage Object.
    """
    try:
        global _router_response
        _router_response = router_agent.invoke(state)
        output = _router_response["messages"][-1].content
        if output == ["FINISH"]:
            output = END
        return {"messages": [
            AIMessage(content=_router_response["messages"][-1].content, name="Router_Agent")
        ]}
    except Exception as e:
        print(f"Error in router_node: {e}")
        return END


def fundamental_node(state: MessagesState):
    """
    Processes financial queries and updates the conversation state.

    Args:
        state (MessagesState): The current conversation state.

    Returns:
        Command: A command updating the conversation and routing to the Final Agent.
    """
    try:
        result = fundamental_analysis_agent.invoke(state)
        return Command(
            update={"messages": [
                AIMessage(content=result["messages"][-1].content, name="Fundamental_Analysis_Agent")
            ]},
            goto="Final_Aggregator_Agent",
        )
    except Exception as e:
        print(f"Error in fundamental_node: {e}")
        return Command(goto=END)


def sentiment_node(state: MessagesState):
    """
    Processes news-related queries and updates the conversation state.

    Args:
        state (MessagesState): The current conversation state.

    Returns:
        Command: A command updating the conversation and routing to the Final Agent.
    """
    try:
        result = sentiment_analysis_agent.invoke(state)
        return Command(
            update={"messages": [
                AIMessage(content=result["messages"][-1].content, name="Sentiment_Analysis_Agent")
            ]},
            goto="Final_Aggregator_Agent",
        )
    except Exception as e:
        print(f"Error in sentiment_node: {e}")
        return Command(goto=END)


def technical_node(state: MessagesState):
    """
    Processes technical analysis queries and updates the conversation state.

    Args:
        state (MessagesState): The current conversation state.

    Returns:
        Command: A command updating the conversation and routing to the Final Agent.
    """
    try:
        result = technical_analysis_agent.invoke(state)
        return Command(
            update={"messages": [
                AIMessage(content=result["messages"][-1].content, name="Technical_Analysis_Agent")
            ]},
            goto="Final_Aggregator_Agent",
        )
    except Exception as e:
        print(f"Error in technical_node: {e}")
        return Command(goto=END)


def risk_assessment_node(state: MessagesState):
    """
    Processes risk analysis queries and updates the conversation state.

    Args:
        state (MessagesState): The current conversation state.

    Returns:
        Command: A command updating the conversation and routing to the Final Agent.
    """
    try:
        result = risk_assessment_agent.invoke(state)
        return Command(
            update={"messages": [
                AIMessage(content=result["messages"][-1].content, name="Risk_Assessment_Agent")
            ]},
            goto="Final_Aggregator_Agent",
        )
    except Exception as e:
        print(f"Error in risk_assessment_node: {e}")
        return Command(goto=END)


def real_estate_node(state: MessagesState):
    """
    Processes housing price-related queries and updates the conversation state.

    Args:
        state (MessagesState): The current conversation state.

    Returns:
        Command: A command updating the conversation and routing to the Final Agent.
    """
    try:
        result = real_estate_agent.invoke(state)
        return Command(
            update={"messages": [
                AIMessage(content=result["messages"][-1].content, name="Real_Estate_Agent")
            ]},
            goto="Final_Aggregator_Agent",
        )
    except Exception as e:
        print(f"Error in real_estate_node: {e}")
        return Command(goto=END)


def final_node(state: MessagesState):
    """
    Aggregates responses from the specialized agents and returns a final summarized result.

    Args:
        state (MessagesState): The current conversation state.

    Returns:
        Command: A command updating the conversation and routing to END.
    """
    try:
        result = final_agent.invoke(state)
        return Command(
            update={"messages": [
                AIMessage(content=result["messages"][-1].content, name="Final_Aggregator_Agent")
            ]},
            goto=END,
        )
    except Exception as e:
        print(f"Error in final_node: {e}")
        return Command(goto=END)


def condition(state: MessagesState) -> Sequence[str]:
    """
    Determines the next agents to route to based on the Router_Agent's structured response.

    Args:
        state (MessagesState): The current state containing message history.

    Returns:
        Sequence[str]: A list of agent names to route to, or ["__end__"] if none are specified.
    """
    try:
        structured_response = _router_response.get("structured_response", {})
        final_output = structured_response.get("final_output", {})
        next_workers = final_output.get("next_worker", [])

        if isinstance(next_workers, list) and all(isinstance(w, str) for w in next_workers):
            return next_workers

        return ["__end__"]
    except Exception as e:
        print(f"Error in condition function: {e}")
        return ["__end__"]
