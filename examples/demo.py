#!/usr/bin/env python3
"""CLI demo that runs the full fin-agent pipeline."""

from __future__ import annotations
import asyncio
import logging
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import FinancialResearchPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          fin-agent: Multi-Agent Financial Research           ║
║          Router → Fetcher → RAG → Analyst → Reporter        ║
║                    + LLM-as-Judge Feedback                   ║
╚══════════════════════════════════════════════════════════════╝
"""

DEMO_QUERIES = [
    ("Analyze Apple's current financial health and give me a buy/sell recommendation", "AAPL"),
    ("What are Tesla's key risk factors and how is its valuation?", "TSLA"),
    ("Explain Microsoft's competitive moat and AI strategy", "MSFT"),
]


async def run_demo(query: str, ticker: str | None, stream: bool = False) -> None:
    print(BANNER)
    print(f"Query : {query}")
    print(f"Ticker: {ticker or 'Auto-detect'}")
    print("=" * 64)

    pipeline = FinancialResearchPipeline()

    if stream:
        print("\n[STREAMING MODE]\n")
        async for event in pipeline.stream(query=query, ticker=ticker):
            ev = event.get("event", "unknown")
            if ev == "agent_start":
                agent = event.get("agent", "?")
                attempt = event.get("attempt", 1)
                suffix = f" (attempt {attempt})" if attempt > 1 else ""
                print(f"  ▶ {agent.upper()}{suffix} starting...")
            elif ev == "agent_complete":
                agent = event.get("agent", "?")
                print(f"  ✓ {agent.upper()} done")
            elif ev == "judge_score":
                score = event.get("score", 0)
                passed = event.get("passed", False)
                status = "PASS ✓" if passed else "FAIL ✗"
                print(f"\n{'='*64}")
                print(f"  JUDGE SCORE: {score}/10  [{status}]")
                if event.get("reasoning"):
                    print(f"  Feedback: {event['reasoning'][:120]}...")
                print(f"{'='*64}\n")
            elif ev == "retry":
                print(f"\n  ↻ RETRY: {event.get('reason', '')}\n")
            elif ev == "result":
                print("\n" + "=" * 64)
                print("FINAL REPORT")
                print("=" * 64)
                print(event["content"])
                print("\n" + "=" * 64)
                print(f"Judge Score  : {event['judge_score']}/10")
                print(f"Retry Count  : {event['retry_count']}")
                print(f"Sources      : {len(event['sources'])}")
                for i, src in enumerate(event["sources"], 1):
                    print(f"  [{i}] {src}")
            elif ev == "error":
                print(f"\n  ERROR: {event['message']}", file=sys.stderr)
    else:
        print("\n[BLOCKING MODE — running full pipeline...]\n")
        ctx = await pipeline.run(query=query, ticker=ticker)

        print("\n" + "=" * 64)
        print("FINAL REPORT")
        print("=" * 64)
        print(ctx.report)
        print("\n" + "=" * 64)
        print(f"Judge Score  : {ctx.judge_score}/10")
        print(f"Retry Count  : {ctx.retry_count}")
        print(f"Sources      : {len(ctx.sources)}")
        for i, src in enumerate(ctx.sources, 1):
            print(f"  [{i}] {src}")
        print(f"\nAgent messages ({len(ctx.messages)}):")
        for msg in ctx.messages:
            print(f"  [{msg.agent}] {msg.content}")


def main():
    parser = argparse.ArgumentParser(description="fin-agent CLI demo")
    parser.add_argument("--query", "-q", type=str, help="Research query")
    parser.add_argument("--ticker", "-t", type=str, help="Stock ticker (e.g. AAPL)")
    parser.add_argument("--stream", "-s", action="store_true", help="Use streaming mode")
    parser.add_argument(
        "--demo",
        "-d",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Run a preset demo query (1-3)",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        print("       Set it with: export OPENAI_API_KEY=sk-...", file=sys.stderr)
        sys.exit(1)

    if args.query:
        query = args.query
        ticker = args.ticker
    else:
        query, ticker = DEMO_QUERIES[args.demo - 1]

    asyncio.run(run_demo(query=query, ticker=ticker, stream=args.stream))


if __name__ == "__main__":
    main()
