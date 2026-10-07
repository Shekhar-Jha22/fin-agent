"""FastAPI microservice with SSE streaming for the financial research pipeline."""

from __future__ import annotations
import asyncio
import json
import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .pipeline import FinancialResearchPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="fin-agent: Financial Research Pipeline",
    description="Multi-agent LLM system for equity research with RAG and LLM-as-judge feedback",
    version="1.0.0",
)

# Lazy-initialized pipeline (built once on first request)
_pipeline: Optional[FinancialResearchPipeline] = None


def get_pipeline() -> FinancialResearchPipeline:
    global _pipeline
    if _pipeline is None:
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        base_url = os.environ.get("OPENAI_BASE_URL", None)
        _pipeline = FinancialResearchPipeline(model=model, base_url=base_url)
    return _pipeline


class AnalyzeRequest(BaseModel):
    query: str = Field(..., description="Research query, e.g. 'Analyze Apple stock performance'")
    ticker: Optional[str] = Field(None, description="Stock ticker symbol, e.g. 'AAPL'")


class AnalyzeResponse(BaseModel):
    report: str
    judge_score: int
    sources: list[str]
    ticker: Optional[str]
    retry_count: int


@app.get("/health")
async def health():
    return {"status": "ok", "service": "fin-agent"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Run the full research pipeline and return the completed report."""
    try:
        pipeline = get_pipeline()
        ctx = await pipeline.run(query=request.query, ticker=request.ticker)
        return AnalyzeResponse(
            report=ctx.report,
            judge_score=ctx.judge_score,
            sources=ctx.sources,
            ticker=ctx.ticker,
            retry_count=ctx.retry_count,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.post("/analyze/stream")
async def analyze_stream(request: AnalyzeRequest):
    """
    Stream pipeline events as Server-Sent Events (SSE).

    Events include:
    - agent_start: an agent has begun processing
    - agent_complete: an agent finished with partial results
    - judge_score: quality evaluation result
    - retry: pipeline retrying due to low quality score
    - result: final report with all metadata
    """

    async def event_generator():
        try:
            pipeline = get_pipeline()
            async for event in pipeline.stream(query=request.query, ticker=request.ticker):
                data = json.dumps(event, ensure_ascii=False)
                yield f"data: {data}\n\n"
                # Yield control so the event loop can flush to client
                await asyncio.sleep(0)
        except Exception as e:
            error_event = json.dumps({"event": "error", "message": str(e)})
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
async def root():
    return {
        "service": "fin-agent",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze": "Run full pipeline (blocking)",
            "POST /analyze/stream": "Run pipeline with SSE streaming",
            "GET /health": "Health check",
        },
    }
