import pytest
from unittest.mock import patch, MagicMock
from stock_analyst.graph.state import MEMBERS, Router, FinalResponse, ConversationalResponse, AgentState


# ── State schema tests ────────────────────────────────────────────────────────

def test_members_list_complete():
    assert "Fundamental_Analysis_Agent" in MEMBERS
    assert "Technical_Analysis_Agent" in MEMBERS
    assert "Sentiment_Analysis_Agent" in MEMBERS
    assert "Risk_Assessment_Agent" in MEMBERS
    assert "Real_Estate_Agent" in MEMBERS
    assert len(MEMBERS) == 5


def test_router_schema_keys():
    assert "next_worker" in Router.__annotations__


def test_final_response_schema_keys():
    assert "final_output" in FinalResponse.__annotations__


# ── Graph compilation ─────────────────────────────────────────────────────────

@patch("stock_analyst.graph.nodes.AzureChatOpenAI")
@patch("stock_analyst.graph.nodes.TavilySearchResults")
def test_compile_graph_succeeds(MockTavily, MockAzure):
    MockAzure.return_value = MagicMock()
    MockTavily.return_value = MagicMock()
    from stock_analyst.graph.builder import compile_graph
    graph = compile_graph()
    assert graph is not None


# ── Condition function ────────────────────────────────────────────────────────

def test_condition_routes_to_single_agent():
    from stock_analyst.graph.nodes import condition
    result = condition({"messages": [], "next_worker": ["Technical_Analysis_Agent"]})
    assert result == ["Technical_Analysis_Agent"]


def test_condition_routes_to_multiple_agents():
    from stock_analyst.graph.nodes import condition
    result = condition({"messages": [], "next_worker": ["Fundamental_Analysis_Agent", "Risk_Assessment_Agent"]})
    assert "Fundamental_Analysis_Agent" in result
    assert "Risk_Assessment_Agent" in result


def test_condition_returns_end_when_no_workers():
    from stock_analyst.graph.nodes import condition
    result = condition({"messages": [], "next_worker": []})
    assert result == ["__end__"]


def test_condition_returns_end_on_empty_response():
    from stock_analyst.graph.nodes import condition
    result = condition({"messages": []})
    assert result == ["__end__"]
