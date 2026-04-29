# Stock Analyst — Multi-Agent System (LangGraph)

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
Unified response (CLI / FastAPI / Streamlit)
```

## Package Structure
```
stock_analyst/               ← core Python package
├── config/settings.py       ← all env vars + tunable params (pydantic-settings)
├── graph/
│   ├── state.py             ← Router, FinalResponse, ConversationalResponse, MEMBERS
│   ├── nodes.py             ← all 7 node functions + condition(); shared model instance
│   └── builder.py           ← compile_graph() — builds and returns compiled StateGraph
├── agents/prompts.py        ← all system prompts as named constants
├── tools/
│   ├── financial.py         ← income_statement, balance_sheet, cashflow, finance_ratio, risk_assessment
│   ├── technical.py         ← technical_indicators_tool (SMA, RSI, MACD, Bollinger Bands)
│   ├── sentiment.py         ← retriever_tool (RAG news pipeline)
│   └── real_estate.py       ← housing_price_index_tool (SCB API)
└── utils/
    ├── yfinance_utils.py    ← FinancialDataFetcher class
    └── rag_utils.py         ← split_documents(), create_vectorstore()

api/                         ← FastAPI backend
├── main.py                  ← app + 3 endpoints: /health, /analyze, /analyze/stream
├── schemas.py               ← AnalyzeRequest, AnalyzeResponse (Pydantic)
└── dependencies.py          ← get_graph() — compiled graph cached via lru_cache

main.py                      ← CLI entrypoint (python main.py)
```

## API Endpoints
| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/analyze` | Blocking — returns full answer + agents_used |
| POST | `/analyze/stream` | SSE stream — emits agent_start, agent_done, final_answer, done events |

Run the API: `uvicorn api.main:app --reload`
Swagger UI: `http://localhost:8000/docs`

## SSE Stream Event Types (`/analyze/stream`)
```json
{"type": "agent_start",  "agent": "Router_Agent"}
{"type": "agent_done",   "agent": "Technical_Analysis_Agent", "content": "..."}
{"type": "final_answer", "agent": "Final_Aggregator_Agent",   "content": "..."}
{"type": "done"}
```

## Key Configuration (config/settings.py)
All values configurable via environment variables or `.env` file:
- `AZURE_GPT_API`, `AZURE_GPT_ENDPOINT` — Azure OpenAI credentials
- `COOKIE_YAHOO` — Yahoo Finance session cookie (for news scraping)
- `TAVILY_API_KEY` — Tavily web search
- `DEPLOYMENT_NAME` (default: `gpt-4o`), `TEMPERATURE` (default: `0.2`)
- `CHUNK_SIZE` (default: `1000`), `CHUNK_OVERLAP` (default: `200`), `RETRIEVAL_K` (default: `2`)
- `CHROMA_DB_PATH` (default: `./chroma_db`)

## Tools (tools/)
1. `retriever_tool(question, ticker)` — RAG: fetch news → chunk → embed → retrieve top-k docs
2. `income_statement_tool(ticker)` — 5-year: EBITDA, EPS, Net Income, Revenue, Gross Profit
3. `balance_sheet_tool(ticker)` — 5-year total debt
4. `cashflow_tool(ticker)` — Free Cash Flow + Operating Cash Flow
5. `finance_ratio_tool(ticker)` — P/E, P/B, market cap, D/E, current ratio, FCF, OCF
6. `risk_assessment_tool(ticker)` — Beta, P/E, P/B, EPS, quick ratio, D/E, current ratio
7. `housing_price_index_tool(region)` — SCB API, 12 Swedish regions, 2014–2024
8. `technical_indicators_tool(ticker)` — 1-year daily: SMA_50, RSI_14, MACD, Bollinger Bands

## Dependencies
See `requirements.txt`. Key packages:
- `langgraph`, `langchain`, `langchain-openai`, `langchain-community`
- `yfinance`, `pyscbwrapper`, `chromadb`, `sentence-transformers`
- `fastapi`, `uvicorn`
- `pydantic-settings`, `python-dotenv`

## Commits So Far
1. Repo hygiene — `.gitignore`, `requirements.txt`, `.env.example`, `CLAUDE.md`
2. Notebook → Python package (`stock_analyst/`)
3. FastAPI backend (`api/`) + cleanup (`main.ipynb` removed, `__pycache__` un-tracked)

## Planned Next
4. Streamlit UI with live agent step visualization
5. pytest test suite
6. Docker + docker-compose
7. GitHub Actions CI
8. README polish + demo
