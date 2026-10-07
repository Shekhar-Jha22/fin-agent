"""AnalysisAgent: LLM-based reasoning over market data and RAG context."""

from __future__ import annotations
import json
import logging
from openai import AsyncOpenAI
from ..memory import PipelineContext

logger = logging.getLogger(__name__)

ANALYST_SYSTEM_PROMPT = """You are a senior equity research analyst with 20 years of experience at a top-tier investment bank. Your analysis is rigorous, data-driven, and actionable.

Given market data, recent news, and relevant 10-K context, produce a comprehensive financial analysis.

Your analysis MUST include:
1. **Financial Health Assessment**: Key metrics interpretation (P/E, margins, growth)
2. **Recent Catalysts**: What's driving price action recently
3. **Risk Factors**: Top 3-5 material risks
4. **Competitive Position**: Industry standing and moat analysis
5. **Valuation**: Is the stock fairly valued, overvalued, or undervalued?
6. **Investment Thesis**: Bull case vs Bear case in 2-3 sentences each

Be specific with numbers. Do not hedge excessively. If data is missing, state what additional information would be needed.

Return a thorough analysis in markdown format. Be precise and professional."""

ANALYST_RETRY_PROMPT = """The previous analysis was scored below the quality threshold.

Judge's feedback: {judge_reasoning}

Please revise the analysis addressing the specific weaknesses identified. Pay particular attention to:
- Adding more quantitative specificity
- Ensuring all claims are grounded in the provided data
- Strengthening the investment thesis with concrete arguments

Previous analysis for reference:
{previous_analysis}"""


def _format_market_data(data: dict) -> str:
    """Format market data dict into clean text for prompt injection."""
    if not data:
        return "No market data available."

    lines = [
        f"**{data.get('company_name', data.get('ticker', 'N/A'))} ({data.get('ticker', '')})**",
        f"Current Price: ${data.get('current_price', 'N/A')}",
        f"Market Cap: {_fmt_large_num(data.get('market_cap'))}",
        f"Sector: {data.get('sector', 'N/A')} | Industry: {data.get('industry', 'N/A')}",
        f"52-Week Range: ${data.get('52w_low', 'N/A')} - ${data.get('52w_high', 'N/A')}",
        f"P/E (Trailing): {data.get('pe_ratio', 'N/A')} | P/E (Forward): {data.get('forward_pe', 'N/A')}",
        f"EPS: ${data.get('eps', 'N/A')}",
        f"Revenue (TTM): {_fmt_large_num(data.get('revenue'))}",
        f"Profit Margin: {_fmt_pct(data.get('profit_margin'))}",
        f"Return on Equity: {_fmt_pct(data.get('return_on_equity'))}",
        f"Debt/Equity: {data.get('debt_to_equity', 'N/A')}",
        f"Beta: {data.get('beta', 'N/A')}",
        f"Dividend Yield: {_fmt_pct(data.get('dividend_yield'))}",
        f"Analyst Consensus: {(data.get('analyst_rating') or 'N/A').upper()} | Target Price: ${data.get('target_mean_price', 'N/A')}",
        f"5-Day Return: {data.get('five_day_return_pct', 'N/A')}%",
    ]
    if data.get("description"):
        lines.append(f"\nBusiness Description: {data['description']}")
    return "\n".join(lines)


def _fmt_large_num(val) -> str:
    if val is None:
        return "N/A"
    if val >= 1e12:
        return f"${val/1e12:.2f}T"
    if val >= 1e9:
        return f"${val/1e9:.1f}B"
    if val >= 1e6:
        return f"${val/1e6:.1f}M"
    return str(val)


def _fmt_pct(val) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.2f}%"


def _format_news(news_list: list[dict]) -> str:
    if not news_list:
        return "No recent news available."
    lines = []
    for i, item in enumerate(news_list[:5], 1):
        lines.append(
            f"{i}. [{item.get('source', 'Unknown')}] {item.get('title', '')}\n"
            f"   Sentiment: {item.get('sentiment', 'neutral')} | {item.get('published', '')[:10]}\n"
            f"   Summary: {item.get('summary', '')[:200]}"
        )
    return "\n".join(lines)


class AnalysisAgent:
    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model
        self.name = "analyst"

    async def run(self, ctx: PipelineContext, retry_with_feedback: bool = False) -> PipelineContext:
        logger.info(f"AnalysisAgent running (retry={retry_with_feedback})")

        market_section = _format_market_data(ctx.market_data)
        news_section = _format_news(ctx.news_data)
        rag_section = (
            "\n\n---\n".join(ctx.rag_context)
            if ctx.rag_context
            else "No 10-K/filing context retrieved."
        )

        user_message = f"""## User Query
{ctx.query}

## Market Data
{market_section}

## Recent News
{news_section}

## Relevant 10-K / Filing Context (RAG)
{rag_section}
"""

        if retry_with_feedback:
            user_message = (
                ANALYST_RETRY_PROMPT.format(
                    judge_reasoning=ctx.judge_reasoning,
                    previous_analysis=ctx.analysis[:1000] + "...",
                )
                + "\n\n"
                + user_message
            )

        messages = [
            {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
        )

        ctx.analysis = response.choices[0].message.content
        ctx.add_message(self.name, f"Analysis complete ({len(ctx.analysis)} chars)")
        logger.info(f"AnalysisAgent produced {len(ctx.analysis)} chars")
        return ctx
