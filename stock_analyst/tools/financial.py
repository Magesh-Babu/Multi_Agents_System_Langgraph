from pydantic import BaseModel
from langchain.tools import tool
from stock_analyst.utils.yfinance_utils import FinancialDataFetcher


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
        print("\nUSING INCOME_STATEMENT TOOL\n")
        fetcher = FinancialDataFetcher(ticker)
        return fetcher.get_income_statement()
    except Exception as e:
        print(f"Error in income_statement_tool: {e}")
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
        print("\nUSING BALANCE_SHEET TOOL\n")
        fetcher = FinancialDataFetcher(ticker)
        return fetcher.get_balance_sheet()
    except Exception as e:
        print(f"Error in balance_sheet_tool: {e}")
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
        print("\nUSING CASH_FLOW TOOL\n")
        fetcher = FinancialDataFetcher(ticker)
        return fetcher.get_cashflow()
    except Exception as e:
        print(f"Error in cashflow_tool: {e}")
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
        print("\nUSING FINANCE_RATIO TOOL\n")
        fetcher = FinancialDataFetcher(ticker)
        return fetcher.get_basic_financials()
    except Exception as e:
        print(f"Error in basic_finance_tool: {e}")
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
        print("\nUSING RISK_ASSESSMENT TOOL\n")
        fetcher = FinancialDataFetcher(ticker)
        return fetcher.get_risk_financials()
    except Exception as e:
        print(f"Error in risk_finance_tool: {e}")
        return None
