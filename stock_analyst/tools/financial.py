import logging
from pydantic import BaseModel
from langchain.tools import tool
from stock_analyst.utils.yfinance_utils import FinancialDataFetcher

logger = logging.getLogger(__name__)


class RatioToolSchema(BaseModel):
    """Schema for finance tool input — expects a stock ticker symbol."""
    ticker: str


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
        logger.debug("Tool invoked: income_statement | ticker=%s", ticker)
        fetcher = FinancialDataFetcher(ticker)
        return fetcher.get_income_statement()
    except Exception as e:
        logger.error("income_statement_tool failed | ticker=%s", ticker, exc_info=True)
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
        logger.debug("Tool invoked: balance_sheet | ticker=%s", ticker)
        fetcher = FinancialDataFetcher(ticker)
        return fetcher.get_balance_sheet()
    except Exception as e:
        logger.error("balance_sheet_tool failed | ticker=%s", ticker, exc_info=True)
        return None


@tool(args_schema=RatioToolSchema)
def cashflow_tool(ticker):
    """
    Tool to retrieve the cash flow details of past 5 years from the company's stock ticker.

    Args:
        ticker (str): The stock ticker symbol for retrieving relevant data.

    Returns:
        dataframe: The cash flow data.
    """
    try:
        logger.debug("Tool invoked: cashflow | ticker=%s", ticker)
        fetcher = FinancialDataFetcher(ticker)
        return fetcher.get_cashflow()
    except Exception as e:
        logger.error("cashflow_tool failed | ticker=%s", ticker, exc_info=True)
        return None


@tool(args_schema=RatioToolSchema)
def finance_ratio_tool(ticker):
    """
    Tool to retrieve the financial ratios/details from the company's stock ticker.

    Args:
        ticker (str): The stock ticker symbol for retrieving relevant data.

    Returns:
        dict: The finance ratio data.
    """
    try:
        logger.debug("Tool invoked: finance_ratio | ticker=%s", ticker)
        fetcher = FinancialDataFetcher(ticker)
        return fetcher.get_basic_financials()
    except Exception as e:
        logger.error("finance_ratio_tool failed | ticker=%s", ticker, exc_info=True)
        return None


@tool(args_schema=RatioToolSchema)
def risk_assessment_tool(ticker):
    """
    Tool to retrieve the risk metrics like market-based and leverage & liquidity metrics
    from the company's stock ticker.

    Args:
        ticker (str): The stock ticker symbol for retrieving relevant data.

    Returns:
        dict: The financial risk metrics data.
    """
    try:
        logger.debug("Tool invoked: risk_assessment | ticker=%s", ticker)
        fetcher = FinancialDataFetcher(ticker)
        return fetcher.get_risk_financials()
    except Exception as e:
        logger.error("risk_assessment_tool failed | ticker=%s", ticker, exc_info=True)
        return None
