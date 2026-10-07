"""ReportWriterAgent: formats structured output with source attribution."""

from __future__ import annotations
import logging
from datetime import datetime
from openai import AsyncOpenAI
from ..memory import PipelineContext

logger = logging.getLogger(__name__)

REPORTER_SYSTEM_PROMPT = """You are a financial report editor at a premier equity research firm. Your job is to transform raw analysis into a polished, institutional-quality research brief.

Format the report using this EXACT structure:

---
# EQUITY RESEARCH BRIEF
**[TICKER]: [COMPANY NAME]**
Rating: [BUY / HOLD / SELL] | Price Target: $[X] | Current Price: $[X]
*Generated: [DATE]*

---

## EXECUTIVE SUMMARY
[2-3 sentence TL;DR with the key thesis and rating rationale]

## KEY METRICS SNAPSHOT
| Metric | Value | vs. Sector Avg |
|--------|-------|----------------|
[Include: P/E, Revenue, Margin, 52W performance, Market Cap]

## INVESTMENT THESIS
### Bull Case 🟢
[3 bullet points - concrete catalysts]

### Bear Case 🔴
[3 bullet points - specific risks]

## ANALYSIS
[Full analysis content here - preserve all detail from the input analysis]

## RISK FACTORS
[Numbered list of top risks with severity: HIGH/MEDIUM/LOW]

## SOURCES & CITATIONS
[Numbered list of all sources used]

---
*Disclaimer: This report is AI-generated for research purposes only. Not financial advice.*
---

Preserve ALL quantitative data from the analysis. Do not summarize away numbers. The report should be print-ready."""

REPORTER_RETRY_PROMPT = """The previous report draft scored below threshold.

Judge feedback: {judge_reasoning}

Revise the report to address these issues. Strengthen the areas flagged by the judge while maintaining the required format structure."""


class ReportWriterAgent:
    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model
        self.name = "reporter"

    async def run(self, ctx: PipelineContext, retry_with_feedback: bool = False) -> PipelineContext:
        logger.info(f"ReportWriterAgent running (retry={retry_with_feedback})")

        sources_block = "\n".join(
            f"{i+1}. {src}" for i, src in enumerate(ctx.sources)
        ) if ctx.sources else "No external sources cited."

        ticker = ctx.ticker or "N/A"
        company_name = ctx.market_data.get("company_name", ticker)
        current_price = ctx.market_data.get("current_price", "N/A")
        analyst_rating = (ctx.market_data.get("analyst_rating") or "hold").upper()
        target_price = ctx.market_data.get("target_mean_price", "N/A")
        today = datetime.utcnow().strftime("%B %d, %Y")

        user_content = f"""Transform this analysis into a polished research report.

**Report Header Info:**
- Ticker: {ticker}
- Company: {company_name}
- Current Price: ${current_price}
- Consensus Rating: {analyst_rating}
- Analyst Target: ${target_price}
- Date: {today}

**Raw Analysis:**
{ctx.analysis}

**Sources:**
{sources_block}

**Original Query:**
{ctx.query}
"""

        if retry_with_feedback:
            user_content = (
                REPORTER_RETRY_PROMPT.format(judge_reasoning=ctx.judge_reasoning)
                + "\n\n"
                + user_content
            )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": REPORTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=2500,
        )

        ctx.report = response.choices[0].message.content
        ctx.add_message(self.name, f"Report formatted ({len(ctx.report)} chars)")
        logger.info(f"ReportWriterAgent produced {len(ctx.report)} chars")
        return ctx
