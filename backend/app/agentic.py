from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from .ai import cosine_similarity, keyword_similarity, openai_service
from .cache import hash_payload, search_cache_backend
from .config import get_settings
from .models import (
    AgentFeedback,
    AgentMemory,
    AgentQueryLog,
    AgentRun,
    AgentTraceEvent,
    SearchEvent,
    SemanticCacheEntry,
    User,
)
from .providers import provider_status
from .schemas import AgentRunRequest, SearchRequest
from .services import execute_search

settings = get_settings()

WEIGHT_KEYS = {
    "budget": ["budget", "cheap", "affordable", "price", "cost", "rent", "monthly"],
    "commute": ["commute", "drive", "transit", "train", "office", "work", "minutes"],
    "schools": ["school", "schools", "kids", "children", "district", "family"],
    "safety": ["safe", "safety", "crime", "quiet", "secure"],
    "walkability": ["walk", "walkable", "cafes", "coffee", "grocery", "parks", "transit"],
    "space": ["bedroom", "bath", "sqft", "space", "yard", "garage"],
}

BASE_WEIGHTS = {
    "budget": 0.24,
    "commute": 0.18,
    "schools": 0.14,
    "safety": 0.14,
    "walkability": 0.12,
    "space": 0.10,
    "provider_quality": 0.08,
}


def _now_ms() -> int:
    return int(time.perf_counter() * 1000)


def _safe_event(event: AgentTraceEvent | dict) -> dict:
    if isinstance(event, dict):
        return event
    return {
        "sequence": event.sequence,
        "event_type": event.event_type,
        "agent_name": event.agent_name,
        "title": event.title,
        "message": event.message,
        "status": event.status,
        "latency_ms": event.latency_ms,
        "payload": event.payload or {},
        "created_at": event.created_at.isoformat() if event.created_at else None,
    }


class AgentRunContext:
    def __init__(self, db: Session, run: AgentRun):
        self.db = db
        self.run = run
        self.events: list[dict] = []
        self.sequence = 0

    def emit(
        self,
        *,
        event_type: str,
        agent_name: str,
        title: str,
        message: str,
        status: str = "ok",
        payload: dict | None = None,
        latency_ms: int | None = None,
    ) -> dict:
        self.sequence += 1
        event = AgentTraceEvent(
            run_id=self.run.run_id,
            sequence=self.sequence,
            event_type=event_type,
            agent_name=agent_name,
            title=title,
            message=message,
            status=status,
            payload=payload or {},
            latency_ms=latency_ms,
        )
        self.db.add(event)
        self.db.commit()
        safe = _safe_event(event)
        self.events.append(safe)
        return safe


def _location_label(search: SearchRequest | None) -> str:
    if not search:
        return "current shortlist"
    loc = search.location
    return loc.address or ", ".join(part for part in [loc.city, loc.state, loc.zip_code] if part) or "unspecified location"


def _listing_text(listing: dict) -> str:
    pieces = [
        listing.get("title") or "",
        listing.get("address") or "",
        listing.get("city") or "",
        listing.get("property_type") or "",
        listing.get("description") or "",
        " ".join(listing.get("reasons") or []),
    ]
    scores = listing.get("scores", {}) or {}
    pieces.append(json.dumps(scores, sort_keys=True))
    return " ".join(str(p) for p in pieces if p)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_memory(db: Session, user: User | None) -> dict[str, float]:
    if not user:
        return {}
    rows = db.query(AgentMemory).filter(AgentMemory.user_id == user.id).all()
    return {row.memory_key: row.memory_value for row in rows}


def _infer_goal_weights(goal: str, memory: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    text = goal.lower()
    weights = dict(BASE_WEIGHTS)
    evidence: list[str] = []

    for key, terms in WEIGHT_KEYS.items():
        count = sum(1 for term in terms if term in text)
        if count:
            weights[key] = weights.get(key, 0.0) + min(0.18, 0.05 * count)
            evidence.append(f"Goal emphasizes {key} ({count} signal{'s' if count != 1 else ''}).")

    for key, value in memory.items():
        if key in weights and value:
            weights[key] = weights.get(key, 0.0) + max(-0.08, min(0.12, value))
            evidence.append(f"Preference memory nudged {key} by {round(value, 2)}.")

    total = sum(max(0.01, value) for value in weights.values())
    normalized = {key: round(max(0.01, value) / total, 3) for key, value in weights.items()}
    return normalized, evidence


def _score_with_agent_weights(listing: dict, weights: dict[str, float], goal: str) -> tuple[float, list[str], list[str]]:
    scores = listing.get("scores", {}) or {}
    neighborhood = listing.get("neighborhood", {}) or {}
    walk = neighborhood.get("walkability", {}) or {}
    schools = neighborhood.get("schools", {}) or {}
    crime = neighborhood.get("crime", {}) or {}
    commute = listing.get("commute", {}) or {}

    price = _num(listing.get("price"), 0)
    fit = _num(scores.get("fit_score"), 60)
    budget_score = _num(scores.get("budget_score"), 65)
    school_score = _num(scores.get("school_score"), _num(schools.get("avg_rating_10"), 0) * 10 if schools.get("avg_rating_10") else 55)
    safety_score = _num(scores.get("crime_score"), _num(crime.get("safety_score"), 55))
    walkability = _num(scores.get("walkability_score"), _num(walk.get("walkability_score"), 55))
    commute_minutes = commute.get("minutes")
    if commute_minutes is None:
        commute_score = 55
    else:
        commute_score = max(0, min(100, 110 - _num(commute_minutes) * 2.2))
    bedroom_score = _num(scores.get("bedroom_score"), 65)
    provider_quality = 70
    if listing.get("images"):
        provider_quality += 10
    if listing.get("source") == "rentcast":
        provider_quality += 8
    if listing.get("source_url"):
        provider_quality += 5
    provider_quality = min(100, provider_quality)

    agent_score = (
        budget_score * weights.get("budget", 0)
        + commute_score * weights.get("commute", 0)
        + school_score * weights.get("schools", 0)
        + safety_score * weights.get("safety", 0)
        + walkability * weights.get("walkability", 0)
        + bedroom_score * weights.get("space", 0)
        + provider_quality * weights.get("provider_quality", 0)
    )
    if goal:
        sem = keyword_similarity(goal, _listing_text(listing)) * 100
        agent_score = (agent_score * 0.86) + (sem * 0.14)

    evidence = []
    tradeoffs = []
    if price:
        evidence.append(f"Price: ${int(price):,}.")
    if listing.get("bedrooms") is not None or listing.get("bathrooms") is not None:
        evidence.append(f"Beds/baths: {listing.get('bedrooms') or '—'} bd / {listing.get('bathrooms') or '—'} ba.")
    if commute_minutes is not None:
        evidence.append(f"Commute estimate: {round(_num(commute_minutes), 1)} minutes.")
    else:
        tradeoffs.append("No routed commute estimate was available for this listing.")
    if school_score:
        evidence.append(f"School signal: {round(school_score, 1)}/100.")
    if safety_score:
        evidence.append(f"Safety signal: {round(safety_score, 1)}/100.")
    if walkability:
        evidence.append(f"Walkability signal: {round(walkability, 1)}/100.")
    if not listing.get("images"):
        tradeoffs.append("No provider/user photos were attached, so visual verification is limited.")
    if listing.get("provider") and listing.get("provider") != "User-submitted":
        evidence.append(f"Source: {listing.get('provider')}.")
    for reason in (listing.get("reasons") or [])[:2]:
        evidence.append(reason)

    return round(agent_score, 1), evidence[:7], tradeoffs[:5]


def _recommendations(goal: str, listings: list[dict], weights: dict[str, float]) -> list[dict]:
    recs = []
    for listing in listings:
        score, evidence, tradeoffs = _score_with_agent_weights(listing, weights, goal)
        recs.append(
            {
                "listing_id": listing.get("id"),
                "title": listing.get("title") or listing.get("address"),
                "address": listing.get("address"),
                "price": listing.get("price"),
                "fit_score": listing.get("scores", {}).get("fit_score"),
                "agent_score": score,
                "confidence": round(min(0.96, max(0.35, score / 105)), 2),
                "decision": "shortlist" if score >= 72 else "consider" if score >= 58 else "deprioritize",
                "tradeoffs": tradeoffs,
                "evidence": evidence,
                "source": listing.get("source"),
                "image_count": len(listing.get("images") or []),
            }
        )
    recs.sort(key=lambda item: item["agent_score"], reverse=True)
    return recs[:5]


def _source_validation(listings: list[dict]) -> dict:
    total = len(listings)
    with_images = sum(1 for item in listings if item.get("images"))
    real_sources = Counter(item.get("source") or "unknown" for item in listings)
    missing = []
    for item in listings[:10]:
        issues = []
        if item.get("price") is None:
            issues.append("price")
        if item.get("bedrooms") is None:
            issues.append("bedrooms")
        if item.get("bathrooms") is None:
            issues.append("bathrooms")
        if not item.get("latitude") or not item.get("longitude"):
            issues.append("coordinates")
        if issues:
            missing.append({"id": item.get("id"), "missing": issues})
    return {
        "total_listings_checked": total,
        "listings_with_images": with_images,
        "image_coverage_pct": round((with_images / total) * 100, 1) if total else 0,
        "source_mix": dict(real_sources),
        "missing_facts": missing[:8],
    }


def _deterministic_draft(goal: str, recs: list[dict], validation: dict, warnings: list[str]) -> str:
    if not recs:
        return (
            "I could not produce a confident recommendation because no listings matched the current constraints. "
            "Relax one constraint at a time: price cap, commute limit, or neighborhood conditions."
        )
    lines = [
        "Agentic recommendation summary",
        f"Goal: {goal}",
        "",
        "Top options:",
    ]
    for i, rec in enumerate(recs[:3], start=1):
        lines.append(
            f"{i}. {rec.get('title')} — agent score {rec.get('agent_score')}/100, "
            f"confidence {int(rec.get('confidence', 0) * 100)}%, decision: {rec.get('decision')}."
        )
        if rec.get("evidence"):
            lines.append("   Evidence: " + " ".join(rec["evidence"][:3]))
        if rec.get("tradeoffs"):
            lines.append("   Tradeoffs: " + " ".join(rec["tradeoffs"][:2]))
    lines.append("")
    lines.append(
        f"Source check: {validation.get('total_listings_checked', 0)} listings checked; "
        f"{validation.get('image_coverage_pct', 0)}% had attached photos."
    )
    if warnings:
        lines.append("Important caveats: " + " ".join(warnings[:3]))
    lines.append("Next action: save the top two options, verify current availability with the source/agent, and rerun with one relaxed constraint if the shortlist is too small.")
    return "\n".join(lines)


def _critic(goal: str, draft: str, recs: list[dict], validation: dict) -> dict:
    issues = []
    if not recs:
        issues.append("No recommendation could be made because the candidate set is empty.")
    if validation.get("missing_facts"):
        issues.append("Some listings are missing facts such as coordinates, beds, baths, or price.")
    if validation.get("image_coverage_pct", 0) < 50 and validation.get("total_listings_checked", 0):
        issues.append("Photo coverage is limited; avoid over-relying on visual assumptions.")
    if "Source check" not in draft:
        issues.append("Draft should state source/data limitations.")
    score = max(35, 95 - len(issues) * 12)
    return {"score": score, "pass": score >= 72, "issues": issues, "revision_notes": issues[:4]}


def _revise_with_critic(draft: str, critique: dict) -> str:
    notes = critique.get("revision_notes") or []
    if not notes:
        return draft
    return draft + "\n\nSelf-review adjustment: " + " ".join(str(note) for note in notes[:4])


def _agent_cache_scope(payload: AgentRunRequest, listings: list[dict]) -> str:
    search_part = payload.search.model_dump() if payload.search else None
    slim_listings = [
        {
            "id": item.get("id"),
            "price": item.get("price"),
            "fit": item.get("scores", {}).get("fit_score"),
            "updated": item.get("listed_date"),
        }
        for item in listings[:12]
    ]
    return hash_payload({"goal_scope": payload.goal, "search": search_part, "listings": slim_listings})


def _semantic_cache_hit(db: Session, scope_key: str, embedding: list[float]) -> tuple[SemanticCacheEntry | None, float]:
    now = datetime.utcnow()
    candidates = (
        db.query(SemanticCacheEntry)
        .filter(SemanticCacheEntry.scope_key == scope_key, SemanticCacheEntry.expires_at > now)
        .order_by(SemanticCacheEntry.updated_at.desc())
        .limit(25)
        .all()
    )
    best = None
    best_score = -1.0
    for candidate in candidates:
        score = cosine_similarity(embedding, candidate.question_embedding or [])
        if score > best_score:
            best = candidate
            best_score = score
    if best and best_score >= settings.semantic_similarity_threshold:
        best.hits += 1
        db.commit()
        return best, best_score
    return None, best_score


def _store_semantic(db: Session, scope_key: str, goal: str, embedding: list[float], response_payload: dict) -> None:
    entry = SemanticCacheEntry(
        scope_key=scope_key,
        question_text=goal,
        question_embedding=embedding,
        response_payload=response_payload,
        similarity_threshold=settings.semantic_similarity_threshold,
        model_name=settings.openai_model,
        expires_at=datetime.utcnow() + timedelta(seconds=settings.semantic_cache_ttl_seconds),
    )
    db.add(entry)
    db.commit()


def _log_agent_query(db: Session, user: User | None, question: str, scope_key: str, cache_status: str) -> None:
    db.add(
        AgentQueryLog(
            user_id=user.id if user else None,
            scope_key=scope_key,
            question_text=question,
            cache_status=cache_status,
        )
    )
    db.commit()


def _update_memory_from_goal(db: Session, user: User | None, goal: str) -> dict:
    if not user:
        return {"updated": False, "reason": "Anonymous run; no long-term preference memory was written."}
    updated = []
    lower = goal.lower()
    for key, terms in WEIGHT_KEYS.items():
        if any(term in lower for term in terms):
            row = db.query(AgentMemory).filter(AgentMemory.user_id == user.id, AgentMemory.memory_key == key).first()
            if not row:
                row = AgentMemory(user_id=user.id, memory_key=key, memory_value=0.03, evidence_count=1, last_evidence=goal[:500])
                db.add(row)
            else:
                row.memory_value = max(-0.15, min(0.25, row.memory_value + 0.015))
                row.evidence_count += 1
                row.last_evidence = goal[:500]
            updated.append(key)
    db.commit()
    return {"updated": bool(updated), "keys": updated[:8]}


def apply_agent_feedback(db: Session, run_id: str, user: User | None, rating: int, label: str | None, comment: str | None) -> dict:
    run = db.query(AgentRun).filter(AgentRun.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Agent run not found.")
    if user and run.user_id and run.user_id != user.id:
        raise HTTPException(status_code=403, detail="You cannot rate another user's agent run.")
    fb = AgentFeedback(run_id=run_id, user_id=user.id if user else None, rating=rating, label=label, comment=comment)
    db.add(fb)

    memory_updates = []
    if user and rating != 0:
        target_text = f"{run.goal} {label or ''} {comment or ''}".lower()
        direction = 0.025 if rating > 0 else -0.025
        for key, terms in WEIGHT_KEYS.items():
            if any(term in target_text for term in terms):
                row = db.query(AgentMemory).filter(AgentMemory.user_id == user.id, AgentMemory.memory_key == key).first()
                if not row:
                    row = AgentMemory(user_id=user.id, memory_key=key, memory_value=direction, evidence_count=1, last_evidence=target_text[:500])
                    db.add(row)
                else:
                    row.memory_value = max(-0.15, min(0.25, row.memory_value + direction))
                    row.evidence_count += 1
                    row.last_evidence = target_text[:500]
                memory_updates.append(key)
    db.commit()
    return {"ok": True, "memory_updates": memory_updates}


def list_agent_runs(db: Session, user: User | None, limit: int = 20) -> list[dict]:
    query = db.query(AgentRun)
    if user:
        query = query.filter(AgentRun.user_id == user.id)
    runs = query.order_by(AgentRun.started_at.desc()).limit(limit).all()
    return [
        {
            "run_id": run.run_id,
            "goal": run.goal,
            "status": run.status,
            "cache_status": run.cache_status,
            "autonomy_level": run.autonomy_level,
            "selected_listing_ids": run.selected_listing_ids,
            "metrics": run.metrics,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "ended_at": run.ended_at.isoformat() if run.ended_at else None,
        }
        for run in runs
    ]


def agent_observability_stats(db: Session) -> dict:
    total_runs = db.query(AgentRun).count()
    completed = db.query(AgentRun).filter(AgentRun.status == "completed").count()
    failed = db.query(AgentRun).filter(AgentRun.status == "failed").count()
    cache_hits = db.query(AgentRun).filter(AgentRun.cache_status.in_(["semantic_hit", "exact_hit"])).count()
    tool_events = db.query(AgentTraceEvent).filter(AgentTraceEvent.event_type == "tool_call").count()
    retry_events = db.query(AgentTraceEvent).filter(AgentTraceEvent.event_type == "retry").count()
    critic_events = db.query(AgentTraceEvent).filter(AgentTraceEvent.agent_name == "CriticAgent").count()

    recent = db.query(AgentRun).order_by(AgentRun.started_at.desc()).limit(10).all()
    latencies = []
    for run in recent:
        if run.metrics and run.metrics.get("total_latency_ms") is not None:
            latencies.append(run.metrics["total_latency_ms"])
    return {
        "agent_runs_total": total_runs,
        "agent_runs_completed": completed,
        "agent_runs_failed": failed,
        "agent_cache_hits": cache_hits,
        "agent_cache_hit_rate": round((cache_hits / total_runs) * 100, 1) if total_runs else 0,
        "tool_calls_total": tool_events,
        "retry_events_total": retry_events,
        "critic_events_total": critic_events,
        "average_recent_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "cache_backend": search_cache_backend(),
    }


async def _tool_with_retry(ctx: AgentRunContext, name: str, fn, *args, retries: int = 2, **kwargs):
    attempt = 0
    last_error = None
    while attempt <= retries:
        start = _now_ms()
        try:
            result = await fn(*args, **kwargs)
            ctx.emit(
                event_type="tool_call",
                agent_name="ToolAgent",
                title=f"Tool succeeded: {name}",
                message=f"{name} completed on attempt {attempt + 1}.",
                payload={"tool": name, "attempt": attempt + 1},
                latency_ms=_now_ms() - start,
            )
            return result
        except Exception as exc:  # pragma: no cover - exercised by live provider failures
            last_error = exc
            ctx.emit(
                event_type="retry",
                agent_name="ToolAgent",
                title=f"Retrying tool: {name}",
                message=f"Attempt {attempt + 1} failed: {exc}. The agent will retry with the same validated inputs.",
                status="warning",
                payload={"tool": name, "attempt": attempt + 1, "error": str(exc)},
                latency_ms=_now_ms() - start,
            )
            attempt += 1
            await asyncio.sleep(min(0.7, 0.15 * attempt))
    raise last_error


async def run_home_search_agent(payload: AgentRunRequest, db: Session, user: User | None = None) -> dict:
    start_run = _now_ms()
    run_id = f"run_{uuid.uuid4().hex[:16]}"
    run = AgentRun(
        run_id=run_id,
        user_id=user.id if user else None,
        goal=payload.goal.strip(),
        status="running",
        autonomy_level=payload.autonomy_level,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    ctx = AgentRunContext(db, run)

    plan = [
        {"step": 1, "agent": "PlannerAgent", "action": "parse goal and choose tools"},
        {"step": 2, "agent": "SearchAgent", "action": "retrieve or accept candidate listings"},
        {"step": 3, "agent": "EvaluatorAgent", "action": "score budget, commute, safety, schools, walkability, and evidence quality"},
        {"step": 4, "agent": "ValidatorAgent", "action": "check missing facts, photo/source coverage, and provider availability"},
        {"step": 5, "agent": "CriticAgent", "action": "review answer for weak evidence or invented facts"},
        {"step": 6, "agent": "OptimizerAgent", "action": "update preference memory and improve future weighting"},
    ]
    run.plan = plan
    db.commit()

    ctx.emit(
        event_type="plan",
        agent_name="PlannerAgent",
        title="Plan created",
        message="The agent decomposed the relocation goal into search, evaluation, validation, critique, and optimization steps.",
        payload={"plan": plan, "location": _location_label(payload.search), "autonomy_level": payload.autonomy_level},
    )

    provider_map = provider_status()
    ctx.emit(
        event_type="observe",
        agent_name="PlannerAgent",
        title="Provider status observed",
        message="The agent checked which external data sources are configured before choosing the execution path.",
        payload={"providers": provider_map},
    )

    listings: list[dict] = [item.model_dump() for item in (payload.listings or [])]
    scope_key = _agent_cache_scope(payload, listings)
    cache_embedding: list[float] | None = None
    if payload.use_cache and openai_service.configured:
        try:
            embed = await openai_service.embed_texts([payload.goal])
            if embed:
                cache_embedding = embed[0]
                hit, similarity = _semantic_cache_hit(db, scope_key, cache_embedding)
                ctx.emit(
                    event_type="cache_check",
                    agent_name="CacheAgent",
                    title="Semantic cache checked",
                    message=(
                        "A semantically similar completed agent run was found."
                        if hit
                        else "No sufficiently similar agent run was found; continuing through the tool path."
                    ),
                    status="ok" if hit else "miss",
                    payload={"similarity": round(similarity, 4), "threshold": settings.semantic_similarity_threshold},
                )
                if hit:
                    cached = dict(hit.response_payload)
                    cached["run_id"] = run_id
                    cached["cache_status"] = "semantic_hit"
                    cached["trace"] = ctx.events
                    run.status = "completed"
                    run.cache_status = "semantic_hit"
                    run.final_answer = cached.get("answer")
                    run.metrics = {"total_latency_ms": _now_ms() - start_run, "cache_hit": True}
                    run.ended_at = datetime.utcnow()
                    db.commit()
                    _log_agent_query(db, user, payload.goal, scope_key, "semantic_hit")
                    return cached
        except Exception as exc:
            ctx.emit(
                event_type="cache_check",
                agent_name="CacheAgent",
                title="Semantic cache unavailable",
                message=f"Semantic cache check failed and the agent continued without cached reuse: {exc}",
                status="warning",
            )

    warnings: list[str] = []
    search_payload_summary = payload.search.model_dump() if payload.search else None
    if not listings and payload.search:
        ctx.emit(
            event_type="act",
            agent_name="SearchAgent",
            title="Live listing retrieval selected",
            message="No shortlist was supplied, so the agent will run the backend listing search tool with the requested filters.",
            payload={"search": search_payload_summary},
        )

        async def _search_tool():
            return await execute_search(payload.search, db, user=user)

        search_result = await _tool_with_retry(ctx, "execute_search", _search_tool, retries=2)
        listings = search_result.get("results") or []
        warnings.extend(search_result.get("warnings") or [])
        ctx.emit(
            event_type="observe",
            agent_name="SearchAgent",
            title="Listings observed",
            message=f"The search tool returned {len(listings)} candidate listings before agent-level reranking.",
            payload={"result_count": len(listings), "cache": search_result.get("cache"), "summary": search_result.get("summary")},
        )
    elif listings:
        ctx.emit(
            event_type="observe",
            agent_name="SearchAgent",
            title="Shortlist accepted",
            message=f"The agent received {len(listings)} listings from the current page and will evaluate that shortlist.",
            payload={"result_count": len(listings)},
        )
    else:
        ctx.emit(
            event_type="observe",
            agent_name="SearchAgent",
            title="No listing source supplied",
            message="The agent needs either a search payload or an existing shortlist. It will return a corrective recommendation.",
            status="warning",
        )

    memory = _load_memory(db, user)
    weights, weight_evidence = _infer_goal_weights(payload.goal, memory)
    ctx.emit(
        event_type="reasoning_summary",
        agent_name="EvaluatorAgent",
        title="Preference weights selected",
        message="The evaluator converted the goal and prior feedback into explicit ranking weights. This is a safe summary, not hidden chain-of-thought.",
        payload={"weights": weights, "signals": weight_evidence[:8], "memory": memory},
    )

    recs = _recommendations(payload.goal, listings, weights)
    ctx.emit(
        event_type="tool_call",
        agent_name="EvaluatorAgent",
        title="Listings scored and reranked",
        message=f"The evaluator produced {len(recs)} ranked recommendations using weighted evidence rather than a plain list view.",
        payload={"top_listing_ids": [item.get("listing_id") for item in recs[:3]], "recommendations": recs[:3]},
    )

    validation = _source_validation(listings)
    validation_status = "ok"
    if validation.get("missing_facts") or validation.get("image_coverage_pct", 0) < 50:
        validation_status = "warning"
    ctx.emit(
        event_type="validate",
        agent_name="ValidatorAgent",
        title="Source and fact coverage checked",
        message="The validator checked provider mix, photo coverage, and missing facts before synthesis.",
        status=validation_status,
        payload=validation,
    )

    observations = [
        {"type": "provider_status", "payload": provider_map},
        {"type": "search", "payload": {"location": _location_label(payload.search), "warnings": warnings}},
        {"type": "validation", "payload": validation},
    ]
    draft = _deterministic_draft(payload.goal, recs, validation, warnings)
    used_openai = False
    if openai_service.configured and recs:
        try:
            start = _now_ms()
            draft = await openai_service.synthesize_agent_run(
                goal=payload.goal,
                recommendations=recs[:5],
                observations=observations,
                critique={"pre_synthesis": "Use evidence only and mention missing data."},
                memory={"weights": weights, "preference_memory": memory},
            )
            used_openai = True
            ctx.emit(
                event_type="act",
                agent_name="SynthesisAgent",
                title="OpenAI synthesis completed",
                message="The synthesis agent generated the decision brief from structured evidence and tool observations.",
                payload={"model": settings.openai_model},
                latency_ms=_now_ms() - start,
            )
        except Exception as exc:
            warnings.append(f"OpenAI synthesis failed; deterministic synthesis was used instead: {exc}")
            ctx.emit(
                event_type="retry",
                agent_name="SynthesisAgent",
                title="OpenAI synthesis fallback",
                message=f"The model call failed, so the agent continued with deterministic synthesis: {exc}",
                status="warning",
            )

    critique = _critic(payload.goal, draft, recs, validation)
    if openai_service.configured and recs:
        try:
            ai_critique = await openai_service.critique_agent_output(goal=payload.goal, draft=draft, recommendations=recs[:5])
            if ai_critique:
                critique = ai_critique
        except Exception:
            pass
    ctx.emit(
        event_type="critique",
        agent_name="CriticAgent",
        title="Self-review completed",
        message="The critic checked for invented facts, missing caveats, weak source grounding, and whether the answer satisfies the user goal.",
        status="ok" if critique.get("pass") else "warning",
        payload=critique,
    )

    final_answer = draft
    if not critique.get("pass") or critique.get("issues"):
        final_answer = _revise_with_critic(draft, critique)
        ctx.emit(
            event_type="revise",
            agent_name="CriticAgent",
            title="Answer revised after critique",
            message="The final answer was revised to include missing caveats and source limitations identified by the critic.",
            payload={"revision_notes": critique.get("revision_notes") or critique.get("issues") or []},
        )

    optimization = _update_memory_from_goal(db, user, payload.goal)
    ctx.emit(
        event_type="optimize",
        agent_name="OptimizerAgent",
        title="Preference memory updated",
        message="The optimizer used the current goal as weak preference evidence for future reranking. User feedback can strengthen or reverse this memory.",
        payload=optimization,
    )

    response_payload = {
        "run_id": run_id,
        "status": "completed",
        "answer": final_answer,
        "plan": plan,
        "recommendations": recs,
        "trace": ctx.events,
        "metrics": {
            "total_latency_ms": _now_ms() - start_run,
            "iterations": len(ctx.events),
            "tool_calls": sum(1 for event in ctx.events if event["event_type"] == "tool_call"),
            "retries": sum(1 for event in ctx.events if event["event_type"] == "retry"),
            "used_openai": used_openai,
            "used_live_search": bool(payload.search and not payload.listings),
            "candidate_count": len(listings),
            "provider_status": provider_map,
        },
        "warnings": warnings,
        "cache_status": "miss",
        "self_optimization": {"weights": weights, "memory_update": optimization, "weight_signals": weight_evidence[:8]},
    }

    if payload.use_cache and cache_embedding:
        try:
            cache_copy = dict(response_payload)
            cache_copy["trace"] = []
            _store_semantic(db, scope_key, payload.goal, cache_embedding, cache_copy)
        except Exception:
            pass

    run.status = "completed"
    run.cache_status = response_payload["cache_status"]
    run.final_answer = final_answer
    run.selected_listing_ids = [rec.get("listing_id") for rec in recs[:5]]
    run.metrics = response_payload["metrics"]
    run.ended_at = datetime.utcnow()
    db.commit()
    _log_agent_query(db, user, payload.goal, scope_key, response_payload["cache_status"])
    return response_payload
