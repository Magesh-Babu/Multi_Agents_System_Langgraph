# Stock Analyst - Multi-Agent System (LangGraph)

## Project Overview
A graph-based multi-agent AI system for financial and stock market analysis. Uses LangGraph's stateful execution with 6 agents (1 router, 5 specialists, 1 aggregator) to answer investor queries comprehensively.

## Architecture
```
User Query
    ↓
Router Agent (intent detection + web search)
    ↓
[Conditional routing — one or more agents in parallel]
    ├── Fundamental_Analysis_Agent   → income statement, balance sheet, cash flow
    ├── Technical_Analysis_Agent     → SMA, RSI, MACD, Bollinger Bands
    ├── Sentiment_Analysis_Agent     → RAG on latest news via ChromaDB
    ├── Risk_Assessment_Agent        → beta, P/E, D/E, liquidity ratios
    └── Real_Estate_Agent            → Swedish housing price index (SCB)
    ↓
Final_Aggregator_Agent (synthesizes all outputs)
    ↓
Unified response
```

## File Map
| File | Purpose |
|---|---|
| `main.ipynb` | All agents, graph construction, example test runs (1,310 lines) |
| `tools_list.py` | 8 LangChain @tool definitions with Pydantic schemas |
| `yfinance_utils.py` | FinancialDataFetcher class — wraps yfinance API |
| `rag_utils.py` | split_documents() + create_vectorstore() for news RAG pipeline |
| `chroma_db/` | Persistent vector DB per ticker (AMD, MSFT, NVDA, SHOP, TSLA) |
| `MAS_graph.png` | Architecture diagram |

## Tools (tools_list.py)
1. `retriever_tool(question, ticker)` — RAG: fetch news → chunk → embed → retrieve top-2 docs
2. `income_statement_tool(ticker)` — 5-year: EBITDA, EPS, Net Income, Revenue, Gross Profit
3. `balance_sheet_tool(ticker)` — 5-year total debt
4. `cashflow_tool(ticker)` — Free Cash Flow + Operating Cash Flow
5. `finance_ratio_tool(ticker)` — P/E, P/B, market cap, D/E, current ratio, FCF, OCF
6. `risk_assessment_tool(ticker)` — Beta, P/E, P/B, EPS, quick ratio, D/E, current ratio
7. `housing_price_index_tool(region)` — SCB API, 12 Swedish regions, 2014–2024
8. `technical_indicators_tool(ticker)` — 1-year daily: SMA_50, RSI_14, MACD, Bollinger Bands

## Key Classes
- `FinancialDataFetcher` (yfinance_utils.py) — wrapper for yfinance; handles Yahoo cookie for news
- `Router` (TypedDict) — structured output for routing decisions
- `ConversationalResponse` / `FinalResponse` — union type for router output

## Environment Variables Required
- `AZURE_GPT_API` — Azure OpenAI API key
- `AZURE_GPT_ENDPOINT` — Azure OpenAI endpoint
- `COOKIE_YAHOO` — Yahoo Finance session cookie (for news scraping)

## Model Config
- Model: Azure GPT-4o (`gpt-4o`, deployment `gpt-4o`)
- API version: `2024-08-01-preview`
- Temperature: 0.2
- Streaming: enabled
- Memory: LangGraph MemorySaver (thread-based, thread_id configurable)

## RAG Pipeline
News → WebBaseLoader → RecursiveCharacterTextSplitter (1000/200) → BAAI/bge-small-en-v1.5 embeddings → ChromaDB (./chroma_db/{ticker}/) → retrieve k=2

## Dependencies (implicit — no requirements.txt)
- `langgraph`, `langchain`, `langchain-openai`, `langchain-huggingface`
- `yfinance`, `pyscbwrapper`, `chromadb`, `sentence-transformers`
- `tavily-python` (TavilySearchResults)
- `pydantic`, `python-dotenv`

## Known Gaps (pre-refactor state)
- No `requirements.txt` or `pyproject.toml`
- No `.gitignore`, no `.env` file
- No unit or integration tests
- All code in a single notebook (`main.ipynb`)
- Hard-coded chunking params, k=2, temperature
- No logging (uses print)
- No retry/rate-limit logic
- No FastAPI or any interface
- No CI/CD or Docker config
- No input validation on tickers

## Tested Query Patterns
- Single-agent: "TSLA balance sheet", "NVDA technical analysis"
- Multi-agent: "Stockholm housing prices AND AMD risks"
- Conversational: general market questions (router handles directly)
