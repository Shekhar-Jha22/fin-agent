"""NewsFetcherAgent: fetches market data via yfinance and mock/real news."""

from __future__ import annotations
import logging
import datetime
from typing import Any

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available; using mock market data")

from ..memory import PipelineContext

# Mock news data for fallback (realistic financial news)
MOCK_NEWS_TEMPLATE = [
    {
        "title": "{ticker} beats Q3 earnings estimates with strong Services revenue",
        "source": "Reuters",
        "published": "2024-10-15T09:30:00Z",
        "summary": "{ticker} reported quarterly earnings that exceeded Wall Street expectations, driven by robust growth in its high-margin segments. Revenue came in above consensus estimates.",
        "url": "https://reuters.com/mock/earnings",
        "sentiment": "positive",
    },
    {
        "title": "Analysts raise price target for {ticker} amid AI monetization prospects",
        "source": "Bloomberg",
        "published": "2024-10-14T14:00:00Z",
        "summary": "Several major investment banks raised their price targets for {ticker}, citing improved visibility on AI-related revenue streams and sustained margin expansion.",
        "url": "https://bloomberg.com/mock/analyst-upgrade",
        "sentiment": "positive",
    },
    {
        "title": "{ticker} faces regulatory headwinds in European markets",
        "source": "Financial Times",
        "published": "2024-10-13T11:00:00Z",
        "summary": "European regulators have opened a new investigation into {ticker}'s market practices, which could result in fines and operational changes in the region.",
        "url": "https://ft.com/mock/regulatory",
        "sentiment": "negative",
    },
]


def _fetch_yfinance_data(ticker: str) -> dict[str, Any]:
    """Fetch real market data from yfinance."""
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info

        # Pull key financials safely
        data: dict[str, Any] = {
            "ticker": ticker,
            "company_name": info.get("longName", ticker),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "eps": info.get("trailingEps"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "revenue": info.get("totalRevenue"),
            "profit_margin": info.get("profitMargins"),
            "debt_to_equity": info.get("debtToEquity"),
            "return_on_equity": info.get("returnOnEquity"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "analyst_rating": info.get("recommendationKey", "hold"),
            "target_mean_price": info.get("targetMeanPrice"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "description": (info.get("longBusinessSummary", "") or "")[:500],
        }

        # Recent price history (5 days)
        hist = yf_ticker.history(period="5d")
        if not hist.empty:
            data["price_history"] = [
                {
                    "date": str(row.name.date()),
                    "open": round(row["Open"], 2),
                    "close": round(row["Close"], 2),
                    "volume": int(row["Volume"]),
                }
                for _, row in hist.iterrows()
            ]
            # Calculate 5-day return
            if len(hist) >= 2:
                start_price = hist["Close"].iloc[0]
                end_price = hist["Close"].iloc[-1]
                data["five_day_return_pct"] = round(
                    ((end_price - start_price) / start_price) * 100, 2
                )

        # Recent news from yfinance
        news = yf_ticker.news
        if news:
            data["news"] = [
                {
                    "title": item.get("title", ""),
                    "source": item.get("publisher", "Unknown"),
                    "published": datetime.datetime.fromtimestamp(
                        item.get("providerPublishTime", 0)
                    ).isoformat()
                    + "Z",
                    "summary": item.get("summary", item.get("title", "")),
                    "url": item.get("link", ""),
                    "sentiment": "neutral",
                }
                for item in news[:5]
            ]

        return data

    except Exception as e:
        logger.warning(f"yfinance fetch failed for {ticker}: {e}")
        return {}


def _mock_market_data(ticker: str) -> dict[str, Any]:
    """Return realistic mock market data when yfinance is unavailable."""
    return {
        "ticker": ticker,
        "company_name": f"{ticker} Inc.",
        "current_price": 178.50,
        "market_cap": 2_800_000_000_000,
        "pe_ratio": 29.5,
        "forward_pe": 26.8,
        "eps": 6.05,
        "52w_high": 199.62,
        "52w_low": 124.17,
        "revenue": 383_285_000_000,
        "profit_margin": 0.2631,
        "debt_to_equity": 151.86,
        "return_on_equity": 1.6,
        "dividend_yield": 0.0055,
        "beta": 1.24,
        "analyst_rating": "buy",
        "target_mean_price": 200.00,
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "description": (
            "A multinational technology company that designs, develops, and sells consumer "
            "electronics, computer software, and online services. Known for its premium "
            "product ecosystem and services business."
        ),
        "price_history": [
            {"date": "2024-10-07", "open": 175.20, "close": 177.30, "volume": 58_000_000},
            {"date": "2024-10-08", "open": 177.50, "close": 178.50, "volume": 61_000_000},
        ],
        "five_day_return_pct": 1.83,
        "data_source": "mock",
    }


class NewsFetcherAgent:
    def __init__(self, model: str = "gpt-4o-mini"):
        # No LLM needed for data fetching
        self.name = "news_fetcher"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        ticker = ctx.ticker
        if not ticker:
            ctx.add_message(self.name, "No ticker provided; skipping market data fetch.")
            return ctx

        logger.info(f"NewsFetcherAgent fetching data for {ticker}")

        if YFINANCE_AVAILABLE:
            market_data = _fetch_yfinance_data(ticker)
        else:
            market_data = {}

        # Fall back to mock if yfinance returned nothing useful
        if not market_data or not market_data.get("current_price"):
            logger.info(f"Using mock market data for {ticker}")
            market_data = _mock_market_data(ticker)

        ctx.market_data = market_data

        # Build news list
        if "news" in market_data:
            ctx.news_data = market_data["news"]
        else:
            ctx.news_data = [
                {
                    **item,
                    "title": item["title"].replace("{ticker}", ticker),
                    "summary": item["summary"].replace("{ticker}", ticker),
                }
                for item in MOCK_NEWS_TEMPLATE
            ]

        # Add sources
        for item in ctx.news_data:
            ctx.sources.append(f"{item['source']}: {item['title'][:60]}...")

        price = market_data.get("current_price", "N/A")
        cap = market_data.get("market_cap", 0)
        cap_str = f"${cap/1e12:.2f}T" if cap and cap > 1e12 else (f"${cap/1e9:.1f}B" if cap else "N/A")

        summary = (
            f"Fetched data for {ticker} | "
            f"Price: ${price} | "
            f"Market Cap: {cap_str} | "
            f"News articles: {len(ctx.news_data)}"
        )
        ctx.add_message(self.name, summary)
        logger.info(summary)
        return ctx
