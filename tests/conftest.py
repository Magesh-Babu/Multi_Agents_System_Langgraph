import os
import pytest
import pandas as pd
from unittest.mock import MagicMock
from langchain.docstore.document import Document

# Set fake credentials before any package imports trigger pydantic-settings
os.environ.setdefault("AZURE_GPT_API", "test-api-key")
os.environ.setdefault("AZURE_GPT_ENDPOINT", "https://test.openai.azure.com/")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")


@pytest.fixture
def sample_documents():
    return [
        Document(page_content="NVIDIA reported strong quarterly earnings driven by AI chip demand.", metadata={}),
        Document(page_content="Tesla faces increasing competition in the EV market from Chinese automakers.", metadata={}),
    ]


@pytest.fixture
def sample_price_df():
    """Minimal OHLCV DataFrame mimicking yfinance output — 60 rows for indicator windows."""
    close = [float(c) for c in range(100, 160)]
    return pd.DataFrame({
        "Open":   [c - 1 for c in close],
        "High":   [c + 2 for c in close],
        "Low":    [c - 2 for c in close],
        "Close":  close,
        "Volume": [1_000_000] * 60,
    })


@pytest.fixture
def mock_graph():
    """A mock LangGraph graph that returns a canned final answer."""
    graph = MagicMock()
    from langchain_core.messages import AIMessage

    def fake_stream(input, config, stream_mode):
        yield {"messages": [AIMessage(content="Router decision", name="Router_Agent")]}
        yield {"messages": [AIMessage(content="Technical analysis result", name="Technical_Analysis_Agent")]}
        yield {"messages": [AIMessage(content="Final synthesized answer.", name="Final_Aggregator_Agent")]}

    graph.stream.side_effect = fake_stream
    return graph
