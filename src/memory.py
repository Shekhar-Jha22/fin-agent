"""Shared agent context and state management for the pipeline."""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentMessage:
    agent: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class PipelineContext:
    """Mutable shared state threaded through every agent in a pipeline run."""

    query: str
    ticker: str | None = None
    plan: list[dict[str, Any]] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)
    market_data: dict[str, Any] = field(default_factory=dict)
    news_data: list[dict[str, Any]] = field(default_factory=list)
    rag_context: list[str] = field(default_factory=list)
    analysis: str = ""
    report: str = ""
    judge_score: int = 0
    judge_reasoning: str = ""
    retry_count: int = 0
    sources: list[str] = field(default_factory=list)

    def add_message(self, agent: str, content: str, **metadata: Any) -> None:
        self.messages.append(AgentMessage(agent=agent, content=content, metadata=metadata))

    def get_conversation_history(self, max_messages: int = 10) -> list[dict[str, str]]:
        return [
            {"role": "assistant", "content": f"[{m.agent}]: {m.content}"}
            for m in self.messages[-max_messages:]
        ]
