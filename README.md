# fin-agent: Multi-Agent Financial Research Pipeline

A production-grade, hierarchical multi-agent LLM system for automated equity research. Built without LangChain — pure Python, manual agent orchestration, and a self-healing feedback loop via LLM-as-judge.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RouterAgent                                 │
│  • Parses query via structured LLM call (JSON mode)            │
│  • Extracts ticker symbol, classifies query type               │
│  • Returns execution plan: [{agent, input, reason}]            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
  ┌──────────────┐   ┌────────────┐  ┌──────────────┐
  │ NewsFetcher  │   │    RAG     │  │   (skipped   │
  │    Agent     │   │ Retriever  │  │  if no tick) │
  │              │   │            │  └──────────────┘
  │ yfinance API │   │ FAISS flat │
  │ Market data  │   │ index over │
  │ Price/P/E/   │   │ 10-K text  │
  │ Revenue/news │   │ chunks     │
  └──────┬───────┘   └─────┬──────┘
         │                 │
         └────────┬────────┘
                  ▼
    ┌─────────────────────────┐
    │      AnalysisAgent      │
    │  • Market data context  │
    │  • Top-3 RAG chunks     │
    │  • LLM: gpt-4o-mini     │
    │  • Financial Health     │
    │  • Bull/Bear Thesis     │
    │  • Valuation Analysis   │
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │   ReportWriterAgent     │
    │  • Structured markdown  │
    │  • Metrics table        │
    │  • Source citations     │
    │  • Disclaimer footer    │
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │       JudgeAgent        │◄──── Ground truth: market data
    │  • Scores 1-10 across   │
    │    5 dimensions         │
    │  • Flags hallucinations │
    │  • Returns improvement  │
    │    suggestions          │
    └────────────┬────────────┘
                 │
         score < 7?
         ┌───────┴───────┐
         ▼               ▼
     RETRY loop      Return report
   inject feedback   with score
   → AnalysisAgent
   → ReportWriter
   → JudgeAgent
   (max 1 retry)
```

## Features

| Feature | Implementation |
|---------|---------------|
| Multi-agent orchestration | Manual async pipeline in `pipeline.py` |
| Market data | `yfinance` (live) + mock fallback |
| RAG / grounded Q&A | FAISS flat index, `text-embedding-3-small` |
| LLM-as-judge | Structured scoring with hallucination detection |
| Self-healing retry | Automatic re-run if judge score < 7/10 |
| Streaming API | FastAPI SSE (`/analyze/stream`) |
| Model-agnostic | Any OpenAI-compatible endpoint |

## Project Structure

```
fin-agent/
├── src/
│   ├── agents/
│   │   ├── router.py        # RouterAgent — query parsing + execution plan
│   │   ├── news_fetcher.py  # NewsFetcherAgent — yfinance + mock news
│   │   ├── analyst.py       # AnalysisAgent — LLM financial reasoning
│   │   ├── reporter.py      # ReportWriterAgent — structured output
│   │   └── judge.py         # JudgeAgent — quality scoring + hallucination detection
│   ├── rag/
│   │   ├── indexer.py       # Chunking, embedding, FAISS index build/save/load
│   │   └── retriever.py     # Similarity search (FAISS + numpy fallback)
│   ├── memory.py            # PipelineContext — shared state across all agents
│   ├── pipeline.py          # Full orchestration with retry loop
│   └── api.py               # FastAPI app with /analyze and /analyze/stream
├── data/
│   └── sample_10k.txt       # Apple 10-K excerpt for RAG corpus
├── examples/
│   └── demo.py              # CLI demo (blocking + streaming modes)
├── requirements.txt
└── .env.example
```

## Setup

### 1. Clone and install

```bash
git clone https://github.com/Shekhar-Jha22/fin-agent.git
cd fin-agent
python -m venv venv && source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 3. Run the CLI demo

```bash
# Default demo (Apple analysis)
python examples/demo.py

# Custom query
python examples/demo.py --query "Analyze Tesla's valuation" --ticker TSLA

# Streaming mode (see agent events as they happen)
python examples/demo.py --ticker AAPL --stream

# Pick from 3 preset demo queries
python examples/demo.py --demo 2   # Tesla
python examples/demo.py --demo 3   # Microsoft
```

### 4. Run the FastAPI server

```bash
uvicorn src.api:app --reload --port 8000
```

Then call the API:

```bash
# Blocking endpoint
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Is Apple a buy right now?", "ticker": "AAPL"}'

# Streaming SSE endpoint
curl -N -X POST http://localhost:8000/analyze/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze Microsoft cloud business", "ticker": "MSFT"}'
```

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## Using an OpenAI-Compatible Endpoint

Works with any OpenAI-compatible API (Together.ai, Groq, Ollama, etc.):

```bash
# Groq (fast inference)
OPENAI_API_KEY=gsk_... OPENAI_BASE_URL=https://api.groq.com/openai/v1 OPENAI_MODEL=llama-3.1-8b-instant python examples/demo.py

# Local Ollama
OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_MODEL=llama3.2 python examples/demo.py
```

## Example Output

```
╔══════════════════════════════════════════════════════════════╗
║          fin-agent: Multi-Agent Financial Research           ║
╚══════════════════════════════════════════════════════════════╝

  ▶ ROUTER starting...
  ✓ ROUTER done
  ▶ NEWS_FETCHER starting...
  ✓ NEWS_FETCHER done
  ▶ RAG_RETRIEVER starting...
  ✓ RAG_RETRIEVER done
  ▶ ANALYST starting...
  ✓ ANALYST done
  ▶ REPORTER starting...
  ✓ REPORTER done
  ▶ JUDGE starting...
  ✓ JUDGE done

================================================================
  JUDGE SCORE: 8/10  [PASS ✓]
================================================================

---
# EQUITY RESEARCH BRIEF
**AAPL: Apple Inc.**
Rating: BUY | Price Target: $200.00 | Current Price: $178.50

## EXECUTIVE SUMMARY
Apple maintains its position as the world's most valuable company...

[full report continues...]

Judge Score: 8/10
Sources    : 3
  [1] Reuters: AAPL beats Q3 earnings estimates...
  [2] Bloomberg: Analysts raise price target...
  [3] Financial Times: Regulatory headwinds...
```

## Design Decisions

### Why no LangChain?
LangChain adds significant abstraction overhead and obscures what's actually happening in the agent loop. By implementing the orchestration manually, every agent call, retry condition, and context injection is explicit and auditable — which matters for a financial application where you need to trust the system's behavior.

### LLM-as-Judge Feedback Loop
The judge agent receives both the generated report **and** the original market data (ground truth). It scores on 5 dimensions and specifically checks for factual inconsistencies between the report's claims and the source data. If score < 7, the judge's detailed reasoning is injected as a system-level prompt into the retry run — this is a form of self-critique that demonstrably improves output quality.

### FAISS RAG Layer
10-K filings contain critical financial context (risk factors, segment data, management commentary) that isn't available via yfinance. Chunking with 50-token overlap prevents information loss at chunk boundaries. The numpy fallback ensures the system degrades gracefully without FAISS installed.

### Async Throughout
All LLM calls use `AsyncOpenAI`, making the system ready for concurrent requests in production. The pipeline's `stream()` method yields SSE events as each agent completes, giving clients real-time visibility without polling.

## Requirements

- Python 3.10+
- OpenAI API key (or compatible endpoint)
- ~250MB disk for FAISS index (first run builds it)

## License

MIT
