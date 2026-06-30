import logging
from typing import Sequence
from langgraph.graph import END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langchain_core.messages import AIMessage
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from stock_analyst.config.settings import settings

logger = logging.getLogger(__name__)
from stock_analyst.graph.state import FinalResponse, AgentState
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

def router_node(state: AgentState):
    """
    Determines the next agent or endpoint based on the user's query.

    Args:
        state (AgentState): The current conversation state containing messages.

    Returns:
        dict: Updated messages and next_worker list stored in state.
    """
    try:
        response = router_agent.invoke(state)
        structured = response.get("structured_response", {})
        final_output = structured.get("final_output", {})
        next_workers = final_output.get("next_worker", [])
        return {
            "messages": [AIMessage(content=response["messages"][-1].content, name="Router_Agent")],
            "next_worker": next_workers,
        }
    except Exception as e:
        logger.error("router_node failed", exc_info=True)
        return {"next_worker": []}


def fundamental_node(state: AgentState):
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
        logger.error("fundamental_node failed", exc_info=True)
        return Command(goto=END)


def sentiment_node(state: AgentState):
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
        logger.error("sentiment_node failed", exc_info=True)
        return Command(goto=END)


def technical_node(state: AgentState):
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
        logger.error("technical_node failed", exc_info=True)
        return Command(goto=END)


def risk_assessment_node(state: AgentState):
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
        logger.error("risk_assessment_node failed", exc_info=True)
        return Command(goto=END)


def real_estate_node(state: AgentState):
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
        logger.error("real_estate_node failed", exc_info=True)
        return Command(goto=END)


def final_node(state: AgentState):
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
        logger.error("final_node failed", exc_info=True)
        return Command(goto=END)


def condition(state: AgentState) -> Sequence[str]:
    """
    Determines the next agents to route to based on next_worker stored in state.

    Args:
        state (AgentState): The current state containing routing data set by router_node.

    Returns:
        Sequence[str]: A list of agent names to route to, or ["__end__"] if none are specified.
    """
    try:
        next_workers = state.get("next_worker", [])

        if isinstance(next_workers, list) and next_workers and all(isinstance(w, str) for w in next_workers):
            return next_workers

        return ["__end__"]
    except Exception as e:
        logger.error("condition function failed", exc_info=True)
        return ["__end__"]
