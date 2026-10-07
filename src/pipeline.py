"""Orchestrates the full multi-agent financial research pipeline."""

from __future__ import annotations
import asyncio
import logging
import os
from pathlib import Path
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI

from .memory import PipelineContext
from .agents import RouterAgent, NewsFetcherAgent, AnalysisAgent, ReportWriterAgent, JudgeAgent
from .rag.indexer import RAGIndexer
from .rag.retriever import RAGRetriever

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
RAG_CACHE_DIR = Path(__file__).parent.parent / ".rag_cache"


class FinancialResearchPipeline:
    """
    Hierarchical multi-agent pipeline for equity research.

    Flow:
      1. RouterAgent   → parse query, build execution plan
      2. NewsFetcherAgent → fetch market data + news
      3. RAGRetriever  → retrieve relevant 10-K context
      4. AnalysisAgent → LLM reasoning over data + RAG
      5. ReportWriterAgent → format polished research brief
      6. JudgeAgent    → score quality (1-10), detect hallucinations
      7. If score < 7: retry steps 4+5 with judge feedback injected
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
    ):
        self.model = model
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = AsyncOpenAI(**client_kwargs)

        self.router = RouterAgent(self.client, model)
        self.news_fetcher = NewsFetcherAgent()
        self.analyst = AnalysisAgent(self.client, model)
        self.reporter = ReportWriterAgent(self.client, model)
        self.judge = JudgeAgent(self.client, model)

        self.indexer = RAGIndexer(
            client=self.client,
            cache_dir=str(RAG_CACHE_DIR),
        )
        self.retriever = RAGRetriever(self.indexer, top_k=3)
        self._rag_ready = False

    async def _ensure_rag_index(self) -> None:
        """Build or load the FAISS RAG index from sample 10-K data."""
        if self._rag_ready:
            return

        # Try loading cached index first
        if self.indexer.load():
            self._rag_ready = True
            return

        # Build from data directory
        docs = []
        for txt_file in DATA_DIR.glob("*.txt"):
            docs.append(txt_file.read_text(encoding="utf-8"))

        if not docs:
            logger.warning("No .txt files found in data/; RAG context will be empty")
            self._rag_ready = True
            return

        await self.indexer.build_index(docs)
        self.indexer.save()
        self._rag_ready = True

    async def run(self, query: str, ticker: Optional[str] = None) -> PipelineContext:
        """Run the full pipeline and return the completed context."""
        ctx = PipelineContext(query=query, ticker=ticker)

        # Ensure RAG index is ready
        await self._ensure_rag_index()

        # Step 1: Router
        logger.info("=== Step 1: RouterAgent ===")
        ctx = await self.router.run(ctx)

        # Step 2: News Fetcher (conditional on plan)
        agent_names = [step.get("agent") for step in ctx.plan]
        if "news_fetcher" in agent_names or ctx.ticker:
            logger.info("=== Step 2: NewsFetcherAgent ===")
            ctx = await self.news_fetcher.run(ctx)

        # Step 3: RAG retrieval
        logger.info("=== Step 3: RAG Retrieval ===")
        rag_query = f"{ctx.query} {ctx.ticker or ''} financial performance earnings revenue"
        ctx.rag_context = await self.retriever.retrieve(rag_query)
        logger.info(f"RAG retrieved {len(ctx.rag_context)} context chunks")

        # Step 4: Analysis
        logger.info("=== Step 4: AnalysisAgent ===")
        ctx = await self.analyst.run(ctx)

        # Step 5: Report
        logger.info("=== Step 5: ReportWriterAgent ===")
        ctx = await self.reporter.run(ctx)

        # Step 6: Judge
        logger.info("=== Step 6: JudgeAgent ===")
        ctx = await self.judge.run(ctx)

        # Step 7: Retry loop if quality below threshold
        if ctx.judge_score < self.judge.pass_threshold and ctx.retry_count < 1:
            logger.info(
                f"Judge score {ctx.judge_score} < {self.judge.pass_threshold}; "
                "retrying analysis + report with feedback"
            )
            ctx.retry_count += 1

            logger.info("=== Retry Step 4: AnalysisAgent (with judge feedback) ===")
            ctx = await self.analyst.run(ctx, retry_with_feedback=True)

            logger.info("=== Retry Step 5: ReportWriterAgent (with judge feedback) ===")
            ctx = await self.reporter.run(ctx, retry_with_feedback=True)

            logger.info("=== Retry Step 6: JudgeAgent (re-score) ===")
            ctx = await self.judge.run(ctx)
        else:
            logger.info(f"Report passed quality threshold: {ctx.judge_score}/10")

        return ctx

    async def stream(
        self, query: str, ticker: Optional[str] = None
    ) -> AsyncIterator[dict]:
        """Stream pipeline events as SSE-compatible dicts."""
        ctx = PipelineContext(query=query, ticker=ticker)
        await self._ensure_rag_index()

        yield {"event": "pipeline_start", "query": query, "ticker": ticker}

        # Step 1: Router
        yield {"event": "agent_start", "agent": "router"}
        ctx = await self.router.run(ctx)
        yield {"event": "agent_complete", "agent": "router", "content": ctx.plan}

        # Step 2: News
        agent_names = [step.get("agent") for step in ctx.plan]
        if "news_fetcher" in agent_names or ctx.ticker:
            yield {"event": "agent_start", "agent": "news_fetcher"}
            ctx = await self.news_fetcher.run(ctx)
            yield {
                "event": "agent_complete",
                "agent": "news_fetcher",
                "content": {
                    "ticker": ctx.ticker,
                    "price": ctx.market_data.get("current_price"),
                    "news_count": len(ctx.news_data),
                },
            }

        # Step 3: RAG
        yield {"event": "agent_start", "agent": "rag_retriever"}
        rag_query = f"{ctx.query} {ctx.ticker or ''} financial performance"
        ctx.rag_context = await self.retriever.retrieve(rag_query)
        yield {"event": "agent_complete", "agent": "rag_retriever", "chunks": len(ctx.rag_context)}

        # Step 4: Analysis
        yield {"event": "agent_start", "agent": "analyst"}
        ctx = await self.analyst.run(ctx)
        yield {"event": "agent_complete", "agent": "analyst", "length": len(ctx.analysis)}

        # Step 5: Report
        yield {"event": "agent_start", "agent": "reporter"}
        ctx = await self.reporter.run(ctx)
        yield {"event": "agent_complete", "agent": "reporter", "length": len(ctx.report)}

        # Step 6: Judge
        yield {"event": "agent_start", "agent": "judge"}
        ctx = await self.judge.run(ctx)
        yield {
            "event": "judge_score",
            "score": ctx.judge_score,
            "reasoning": ctx.judge_reasoning,
            "passed": ctx.judge_score >= self.judge.pass_threshold,
        }

        # Retry if needed
        if ctx.judge_score < self.judge.pass_threshold and ctx.retry_count < 1:
            ctx.retry_count += 1
            yield {
                "event": "retry",
                "reason": f"Score {ctx.judge_score} below threshold; retrying with judge feedback",
            }

            yield {"event": "agent_start", "agent": "analyst", "attempt": 2}
            ctx = await self.analyst.run(ctx, retry_with_feedback=True)

            yield {"event": "agent_start", "agent": "reporter", "attempt": 2}
            ctx = await self.reporter.run(ctx, retry_with_feedback=True)

            yield {"event": "agent_start", "agent": "judge", "attempt": 2}
            ctx = await self.judge.run(ctx)
            yield {
                "event": "judge_score",
                "score": ctx.judge_score,
                "attempt": 2,
                "passed": ctx.judge_score >= self.judge.pass_threshold,
            }

        yield {
            "event": "result",
            "content": ctx.report,
            "judge_score": ctx.judge_score,
            "sources": ctx.sources,
            "retry_count": ctx.retry_count,
        }
