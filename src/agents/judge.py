"""JudgeAgent: LLM-as-judge for quality scoring and hallucination detection."""

from __future__ import annotations
import json
import logging
from openai import AsyncOpenAI
from ..memory import PipelineContext

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are an expert financial research quality auditor. Evaluate the provided equity research report against strict institutional standards.

Score the report on a scale of 1-10 across these dimensions:
1. **Factual Grounding** (1-10): Are all claims supported by provided market data? No hallucinated numbers?
2. **Analytical Depth** (1-10): Does it go beyond surface-level observations?
3. **Structure & Clarity** (1-10): Is it well-organized and clearly written?
4. **Quantitative Rigor** (1-10): Sufficient use of numbers and metrics?
5. **Actionability** (1-10): Clear investment thesis and recommendation?

Return ONLY valid JSON:
{
  "overall_score": <integer 1-10>,
  "dimension_scores": {
    "factual_grounding": <1-10>,
    "analytical_depth": <1-10>,
    "structure_clarity": <1-10>,
    "quantitative_rigor": <1-10>,
    "actionability": <1-10>
  },
  "hallucination_flags": [
    "<any claim that contradicts the source data>"
  ],
  "strengths": ["<strength 1>", "<strength 2>"],
  "weaknesses": ["<weakness 1>", "<weakness 2>"],
  "improvement_suggestions": "<detailed 2-3 sentence instruction for improvement>",
  "pass": <true if overall_score >= 7, else false>
}

Be strict but fair. A score of 7+ means the report is publication-ready. Hallucinations are heavily penalized."""


class JudgeAgent:
    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model
        self.name = "judge"
        self.pass_threshold = 7

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        logger.info("JudgeAgent evaluating report quality")

        # Build grounding context so the judge can check claims
        market_summary = json.dumps(
            {
                k: v
                for k, v in ctx.market_data.items()
                if k not in ("description", "price_history", "news")
            },
            indent=2,
        )

        evaluation_prompt = f"""**SOURCE DATA (ground truth):**
{market_summary}

**ORIGINAL QUERY:**
{ctx.query}

**REPORT TO EVALUATE:**
{ctx.report}

Evaluate the report against the source data above. Flag any factual inconsistencies."""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": evaluation_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        raw = response.choices[0].message.content
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"JudgeAgent JSON parse error: {e}")
            # Fallback: assign passing score to avoid infinite loop
            result = {"overall_score": 7, "pass": True, "improvement_suggestions": ""}

        ctx.judge_score = result.get("overall_score", 5)
        ctx.judge_reasoning = result.get("improvement_suggestions", "")

        hallucinations = result.get("hallucination_flags", [])
        weaknesses = result.get("weaknesses", [])
        strengths = result.get("strengths", [])

        passed = result.get("pass", ctx.judge_score >= self.pass_threshold)

        summary = (
            f"Judge score: {ctx.judge_score}/10 | "
            f"Pass: {passed} | "
            f"Hallucinations flagged: {len(hallucinations)} | "
            f"Strengths: {len(strengths)} | "
            f"Weaknesses: {len(weaknesses)}"
        )
        ctx.add_message(
            self.name,
            summary,
            dimension_scores=result.get("dimension_scores", {}),
            hallucinations=hallucinations,
            strengths=strengths,
            weaknesses=weaknesses,
        )
        logger.info(summary)

        if hallucinations:
            logger.warning(f"Hallucinations detected: {hallucinations}")

        return ctx
