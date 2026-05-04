import pytest
import pandas as pd
from unittest.mock import patch, MagicMock


# ── Financial tools ──────────────────────────────────────────────────────────

@patch("stock_analyst.tools.financial.FinancialDataFetcher")
def test_income_statement_tool(MockFetcher, sample_price_df):
    MockFetcher.return_value.get_income_statement.return_value = sample_price_df
    from stock_analyst.tools.financial import income_statement_tool
    result = income_statement_tool.invoke({"ticker": "NVDA"})
    assert result is not None
    MockFetcher.return_value.get_income_statement.assert_called_once()


@patch("stock_analyst.tools.financial.FinancialDataFetcher")
def test_balance_sheet_tool(MockFetcher, sample_price_df):
    MockFetcher.return_value.get_balance_sheet.return_value = sample_price_df
    from stock_analyst.tools.financial import balance_sheet_tool
    result = balance_sheet_tool.invoke({"ticker": "TSLA"})
    assert result is not None
    MockFetcher.return_value.get_balance_sheet.assert_called_once()


@patch("stock_analyst.tools.financial.FinancialDataFetcher")
def test_cashflow_tool(MockFetcher, sample_price_df):
    MockFetcher.return_value.get_cashflow.return_value = sample_price_df
    from stock_analyst.tools.financial import cashflow_tool
    result = cashflow_tool.invoke({"ticker": "AAPL"})
    assert result is not None
    MockFetcher.return_value.get_cashflow.assert_called_once()


@patch("stock_analyst.tools.financial.FinancialDataFetcher")
def test_finance_ratio_tool(MockFetcher):
    MockFetcher.return_value.get_basic_financials.return_value = {
        "trailingPE": 32.5, "marketCap": 1_000_000_000, "debtToEquity": 0.5
    }
    from stock_analyst.tools.financial import finance_ratio_tool
    result = finance_ratio_tool.invoke({"ticker": "MSFT"})
    assert isinstance(result, dict)
    assert "trailingPE" in result


@patch("stock_analyst.tools.financial.FinancialDataFetcher")
def test_risk_assessment_tool(MockFetcher):
    MockFetcher.return_value.get_risk_financials.return_value = {
        "beta": 1.2, "trailingPE": 28.0, "debtToEquity": 0.8
    }
    from stock_analyst.tools.financial import risk_assessment_tool
    result = risk_assessment_tool.invoke({"ticker": "AMD"})
    assert isinstance(result, dict)
    assert "beta" in result


@patch("stock_analyst.tools.financial.FinancialDataFetcher")
def test_financial_tool_returns_none_on_error(MockFetcher):
    MockFetcher.return_value.get_income_statement.side_effect = Exception("API error")
    from stock_analyst.tools.financial import income_statement_tool
    result = income_statement_tool.invoke({"ticker": "INVALID"})
    assert result is None


# ── Technical tool ───────────────────────────────────────────────────────────

@patch("stock_analyst.tools.technical.FinancialDataFetcher")
def test_technical_indicators_tool_columns(MockFetcher, sample_price_df):
    MockFetcher.return_value.get_price_history.return_value = sample_price_df
    from stock_analyst.tools.technical import technical_indicators_tool
    result = technical_indicators_tool.invoke({"ticker": "NVDA"})
    assert result is not None
    expected_cols = {"Close_Price", "SMA_50", "RSI_14", "MACD", "Signal_9", "BB_Upper", "BB_Middle", "BB_Lower"}
    assert expected_cols.issubset(set(result.columns))


@patch("stock_analyst.tools.technical.FinancialDataFetcher")
def test_technical_indicators_sma_window(MockFetcher, sample_price_df):
    MockFetcher.return_value.get_price_history.return_value = sample_price_df
    from stock_analyst.tools.technical import technical_indicators_tool
    result = technical_indicators_tool.invoke({"ticker": "NVDA"})
    # First 49 rows should be NaN for SMA_50
    assert result["SMA_50"].iloc[:49].isna().all()
    assert pd.notna(result["SMA_50"].iloc[49])


# ── Real estate tool ─────────────────────────────────────────────────────────

@patch("stock_analyst.tools.real_estate.SCB")
def test_housing_price_index_tool(MockSCB):
    MockSCB.return_value.get_data.return_value = {"data": [{"region": "Greater Stockholm", "value": 320}]}
    from stock_analyst.tools.real_estate import housing_price_index_tool
    result = housing_price_index_tool.invoke({"region": "Greater Stockholm"})
    assert result is not None
    MockSCB.return_value.get_data.assert_called_once()


@patch("stock_analyst.tools.real_estate.SCB")
def test_housing_price_index_tool_error(MockSCB):
    MockSCB.return_value.get_data.side_effect = Exception("SCB API down")
    from stock_analyst.tools.real_estate import housing_price_index_tool
    result = housing_price_index_tool.invoke({"region": "Sweden"})
    assert "error" in result.lower()


# ── Sentiment / RAG tool ─────────────────────────────────────────────────────

@patch("stock_analyst.tools.sentiment.create_vectorstore")
@patch("stock_analyst.tools.sentiment.split_documents")
@patch("stock_analyst.tools.sentiment.FinancialDataFetcher")
def test_retriever_tool_no_news(MockFetcher, mock_split, mock_vs):
    MockFetcher.return_value.get_latest_news.return_value = []
    from stock_analyst.tools.sentiment import retriever_tool
    result = retriever_tool.invoke({"question": "How is TSLA doing?", "ticker": "TSLA"})
    assert "no news" in result.lower()


@patch("stock_analyst.tools.sentiment.create_vectorstore")
@patch("stock_analyst.tools.sentiment.split_documents")
@patch("stock_analyst.tools.sentiment.FinancialDataFetcher")
def test_retriever_tool_returns_content(MockFetcher, mock_split, mock_vs, sample_documents):
    MockFetcher.return_value.get_latest_news.return_value = sample_documents
    mock_split.return_value = sample_documents

    mock_retriever = MagicMock()
    mock_retriever.get_relevant_documents.return_value = sample_documents
    mock_vs.return_value.as_retriever.return_value = mock_retriever

    from stock_analyst.tools.sentiment import retriever_tool
    result = retriever_tool.invoke({"question": "NVDA earnings?", "ticker": "NVDA"})
    assert "NVIDIA" in result or "Tesla" in result
