"""
Module for retrieving financial data and news using Yahoo Finance and LangChain.

This module provides functionalities for fetching stock-related financial data 
using `yfinance` and loading news articles using `WebBaseLoader` from LangChain.

Imports:
    os: Provides access to operating system functionalities.
    yfinance: Used for retrieving financial data from Yahoo Finance.
    WebBaseLoader: Loads web-based documents for processing.
"""
import os
import yfinance
from langchain_community.document_loaders.web_base import WebBaseLoader
COOKIE_YAHOO = os.getenv("COOKIE_YAHOO")

class FinancialDataFetcher:
    """
    A class for fetching financial data using Yahoo Finance."""
    def __init__(self, ticker_symbol):
        """
        Initializes the FinancialDataFetcher with a stock ticker symbol.

        Args:
            ticker_symbol (str): The stock ticker symbol of the company.
        """
        try:
            if not ticker_symbol:
                raise ValueError("Ticker symbol cannot be empty.")
            self.ticker_symbol = ticker_symbol
            self.ticker = yfinance.Ticker(ticker_symbol)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FinancialDataFetcher: {e}") from e


    def get_latest_news(self):
        """
        Fetches the latest news articles related to the specified stock ticker.

        Returns:
            list: A list of documents containing news content.
        """
        try:
            output = self.ticker.news
            if not output:
                raise ValueError("No news found for the given ticker symbol.")
            links = [
                n["content"]["canonicalUrl"]["url"]
                for n in output
                if n["content"]["contentType"] == "STORY"
            ][:2]
            if not links:
                raise ValueError("No valid news links found.")
            loader = WebBaseLoader(
                web_paths=links,
                header_template={"Cookie": COOKIE_YAHOO},
            )
            docs = loader.load()
            return docs
        except Exception as e:
            print(f"Error fetching latest news: {e}")
            return []

    def get_income_statement(self):
        """
        Retrieves key financial metrics from the company's income statement.

        Returns:
            DataFrame: A Pandas DataFrame containing EBITDA, EPS, Net Income, 
                       Gross Profit, Total Revenue, and Operating Income.
        """
        try:
            output = self.ticker.income_stmt.loc[
                ['EBITDA', 'Basic EPS', 'Net Income', 'Gross Profit', 'Total Revenue', 'Operating Income']
            ]
            return output
        except Exception as e:
            print(f"Error fetching income statement: {e}")
            return None

    def get_balance_sheet(self):
        """
        Retrieves the company's balance sheet, specifically the total debt.

        Returns:
            DataFrame: A Pandas DataFrame containing total debt data.
        """
        try:
            output = self.ticker.balance_sheet.loc[['Total Debt']]
            return output
        except Exception as e:
            print(f"Error fetching balance sheet: {e}")
            return None

    def get_cashflow(self):
        """
        Retrieves the company's cash flow data, including free cash flow 
        and operating cash flow.

        Returns:
            DataFrame: A Pandas DataFrame containing free cash flow and operating cash flow.
        """
        try:
            output = self.ticker.cashflow.loc[['Free Cash Flow', 'Operating Cash Flow']]
            return output
        except Exception as e:
            print(f"Error fetching cash flow: {e}")
            return None

    def get_basic_financials(self):
        """
        Retrieves key financial ratios and metrics of the company.

        Returns:
            dict: A dictionary containing various financial metrics such as P/E ratio, 
                  market cap, debt-to-equity ratio, and cash flow data.
        """
        try:
            output = self.ticker.info
            if not output:
                raise ValueError("No financial information available for this ticker.")
            keys = [
                'trailingPE', 'marketCap', 'currency', 'priceToBook', 'trailingEps',
                'symbol', 'currentPrice', 'totalDebt', 'debtToEquity', 'currentRatio',
                'freeCashflow', 'operatingCashflow'
            ]
            data = {key: output.get(key, 'N/A') for key in keys}
            return data
        except Exception as e:
            print(f"Error fetching basic financials: {e}")
            return {}
        
    def get_price_history(self, period='1y'):
        """
        Retrieves the price history of the stock for a specified period.

        Args:
            period (str): The time period for which to fetch the price history. 
                            Default is '1y' (1 year).

        Returns:
            DataFrame: A Pandas DataFrame containing the historical price data.
        """
        try:
            if not period:
                raise ValueError("Period cannot be empty.")
            output = self.ticker.history(period=period, interval="1d")
            return output
        except Exception as e:
            print(f"Error fetching price history: {e}")
            return None        
    