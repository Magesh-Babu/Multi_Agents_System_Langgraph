from pydantic import BaseModel
from langchain.tools import tool
from stock_analyst.utils.yfinance_utils import FinancialDataFetcher


class RatioToolSchema(BaseModel):
    """Schema for finance tool input — expects a stock ticker symbol."""
    ticker: str


@tool(args_schema=RatioToolSchema)
def technical_indicators_tool(ticker):
    """
    Tool to calculate various technical indicators such as SMA, RSI, MACD, and Bollinger Bands.

    Args:
        ticker (str): The stock ticker symbol for retrieving relevant data.

    Returns:
        dataframe: The price history data with calculated indicators.
    """
    try:
        print("\nUSING TECHNICAL_INDICATORS TOOL\n")
        fetcher = FinancialDataFetcher(ticker)
        data = fetcher.get_price_history()

        # Simple Moving Average (SMA)
        data['SMA_50'] = data['Close'].rolling(window=50).mean()

        # Relative Strength Index (RSI)
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI_14'] = 100 - (100 / (1 + rs))

        # Moving Average Convergence Divergence (MACD)
        ema_fast = data['Close'].ewm(span=12, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = ema_fast - ema_slow
        data['Signal_9'] = data['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Std'] = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + 2 * data['BB_Std']
        data['BB_Lower'] = data['BB_Middle'] - 2 * data['BB_Std']

        data['Close_Price'] = data['Close']

        return data[['Close_Price', 'SMA_50', 'RSI_14', 'MACD', 'Signal_9', 'BB_Upper', 'BB_Middle', 'BB_Lower']]
    except Exception as e:
        print(f"Error in price_history_tool: {e}")
        return None
