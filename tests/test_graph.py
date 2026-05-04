import pytest
from unittest.mock import patch, MagicMock
from stock_analyst.graph.state import MEMBERS, Router, FinalResponse, ConversationalResponse


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
    import stock_analyst.graph.nodes as nodes
    nodes._router_response = {
        "structured_response": {
            "final_output": {"next_worker": ["Technical_Analysis_Agent"]}
        }
    }
    result = nodes.condition({})
    assert result == ["Technical_Analysis_Agent"]


def test_condition_routes_to_multiple_agents():
    import stock_analyst.graph.nodes as nodes
    nodes._router_response = {
        "structured_response": {
            "final_output": {"next_worker": ["Fundamental_Analysis_Agent", "Risk_Assessment_Agent"]}
        }
    }
    result = nodes.condition({})
    assert "Fundamental_Analysis_Agent" in result
    assert "Risk_Assessment_Agent" in result


def test_condition_returns_end_when_no_workers():
    import stock_analyst.graph.nodes as nodes
    nodes._router_response = {
        "structured_response": {"final_output": {}}
    }
    result = nodes.condition({})
    assert result == ["__end__"]


def test_condition_returns_end_on_empty_response():
    import stock_analyst.graph.nodes as nodes
    nodes._router_response = {}
    result = nodes.condition({})
    assert result == ["__end__"]
