from __future__ import annotations

import json
import math
import re
from typing import Any

from .config import get_settings

settings = get_settings()

try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover
    AsyncOpenAI = None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def keyword_similarity(query: str, text: str) -> float:
    query_terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2}
    text_terms = {term for term in re.findall(r"[a-z0-9]+", text.lower()) if len(term) > 2}
    if not query_terms or not text_terms:
        return 0.0
    overlap = len(query_terms & text_terms)
    return overlap / max(len(query_terms), 1)


class OpenAIService:
    def __init__(self):
        self._client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key and AsyncOpenAI else None

    @property
    def configured(self) -> bool:
        return self._client is not None

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self._client or not texts:
            return []
        kwargs: dict[str, Any] = {
            "model": settings.openai_embedding_model,
            "input": texts,
        }
        if settings.openai_embedding_dimensions:
            kwargs["dimensions"] = settings.openai_embedding_dimensions
        response = await self._client.embeddings.create(**kwargs)
        return [list(item.embedding) for item in response.data]

    async def answer_with_listings(self, *, question: str, slim_listings: list[dict]) -> str:
        if not self._client:
            raise RuntimeError("OpenAI is not configured.")
        prompt = (
            "User question:\n"
            f"{question}\n\n"
            "Listings JSON:\n"
            f"{json.dumps(slim_listings, indent=2)}"
        )
        response = await self._client.responses.create(
            model=settings.openai_model,
            instructions=(
                "You are a relocation assistant. Compare tradeoffs across the provided listings. "
                "Be practical, structured, and specific. Never invent missing facts. If data is missing, say so."
            ),
            input=prompt,
            reasoning={"effort": "low"},
        )
        return response.output_text.strip()

    async def synthesize_agent_run(self, *, goal: str, recommendations: list[dict], observations: list[dict], critique: dict, memory: dict) -> str:
        if not self._client:
            raise RuntimeError("OpenAI is not configured.")
        prompt = {
            "user_goal": goal,
            "ranked_recommendations": recommendations,
            "tool_observations": observations,
            "critic_findings": critique,
            "preference_memory": memory,
        }
        response = await self._client.responses.create(
            model=settings.openai_model,
            instructions=(
                "You are the synthesis agent in an agentic relocation system. Use only the supplied evidence. "
                "Do not invent listings, photos, crime data, school ratings, or commute times. "
                "Do not reveal private chain-of-thought. Provide a concise decision brief with: recommendation, "
                "why it fits, tradeoffs, missing data, and next action. Mention when a source is unavailable."
            ),
            input=json.dumps(prompt, indent=2),
            reasoning={"effort": "low"},
        )
        return response.output_text.strip()

    async def critique_agent_output(self, *, goal: str, draft: str, recommendations: list[dict]) -> dict | None:
        if not self._client:
            return None
        prompt = (
            "Critique this relocation recommendation. Return strict JSON only with keys: "
            "score (0-100), pass (boolean), issues (array of strings), revision_notes (array of strings). "
            "Check for invented facts, missing caveats, weak source grounding, and failure to answer the goal.\n\n"
            f"Goal:\n{goal}\n\nRecommendations:\n{json.dumps(recommendations, indent=2)}\n\nDraft:\n{draft}"
        )
        try:
            response = await self._client.responses.create(
                model=settings.openai_model,
                input=prompt,
                reasoning={"effort": "low"},
            )
            match = re.search(r"\{.*\}", response.output_text, re.DOTALL)
            if not match:
                return None
            return json.loads(match.group(0))
        except Exception:
            return None

    async def review_listing(self, summary: dict) -> dict | None:
        if not self._client:
            return None
        prompt = (
            "Review the following user-submitted housing listing for plausibility. "
            "Return strict JSON only with keys: score (0-100), verdict (pass|review|reject), "
            "flags (array of strings), rationale (string).\n\n"
            f"Listing:\n{json.dumps(summary, indent=2)}"
        )
        try:
            response = await self._client.responses.create(
                model=settings.openai_model,
                input=prompt,
                reasoning={"effort": "low"},
            )
            text = response.output_text
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            return json.loads(match.group(0))
        except Exception:
            return None


openai_service = OpenAIService()
