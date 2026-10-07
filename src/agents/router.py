"""RouterAgent: parses user query and returns an execution plan."""

from __future__ import annotations
import json
import logging
from openai import AsyncOpenAI
from ..memory import PipelineContext

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are a financial research routing agent. Given a user query about a stock or financial topic, you must produce a structured execution plan.

Your job is to decide which agents to invoke and in what order. Available agents:
- news_fetcher: Fetches recent news and market data for a ticker symbol
- analyst: Performs LLM-based reasoning over fetched data and RAG context
- reporter: Formats a structured, cited research report from analysis

Return ONLY valid JSON with this exact structure:
{
  "ticker": "<SYMBOL or null>",
  "plan": [
    {"agent": "<agent_name>", "input": "<specific instruction for this agent>", "reason": "<why this agent is needed>"}
  ],
  "query_type": "<one of: earnings_analysis | price_movement | competitor_analysis | general_research | sentiment_analysis>",
  "complexity": "<low|medium|high>"
}

Rules:
1. Always include news_fetcher first if a ticker is mentioned
2. Always include analyst after news_fetcher
3. Always include reporter last
4. For general questions without a ticker, skip news_fetcher
5. Extract the ticker symbol from the query (e.g., AAPL for Apple, TSLA for Tesla)
"""


class RouterAgent:
    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model
        self.name = "router"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        logger.info(f"RouterAgent processing query: {ctx.query}")

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Query: {ctx.query}\nTicker hint: {ctx.ticker or 'none provided'}",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        raw = response.choices[0].message.content
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"RouterAgent JSON parse error: {e}\nRaw: {raw}")
            raise

        # Merge ticker from router if not already set
        if not ctx.ticker and result.get("ticker") and result["ticker"] != "null":
            ctx.ticker = result["ticker"]

        ctx.plan = result.get("plan", [])

        summary = (
            f"Routing plan: {len(ctx.plan)} steps | "
            f"Ticker: {ctx.ticker} | "
            f"Query type: {result.get('query_type', 'unknown')} | "
            f"Complexity: {result.get('complexity', 'medium')}"
        )
        ctx.add_message(self.name, summary, raw_plan=result)
        logger.info(summary)
        return ctx
