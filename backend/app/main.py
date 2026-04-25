from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from .agentic import agent_observability_stats, apply_agent_feedback, list_agent_runs, run_home_search_agent
from .auth import authenticate_user, get_current_user, get_optional_user, require_admin
from .cache import search_cache_backend
from .config import get_settings
from .db import SessionLocal, get_db, init_db
from .models import AdvertisedPlace, Favorite, ModerationStatus, User
from .providers import provider_status
from .schemas import (
    AgentFeedbackRequest,
    AgentRequest,
    AgentRunRequest,
    AgentRunResponse,
    AdvertisedPlaceOut,
    FavoriteCreate,
    FavoriteOut,
    ModerationAction,
    SearchRequest,
    SearchResponse,
    TokenResponse,
    UserCreate,
    UserOut,
    UserLogin,
)
from .services import (
    admin_stats,
    answer_agent_question,
    create_advertised_place,
    create_login_response,
    create_user,
    ensure_admin_seed,
    execute_search,
    moderation_update,
    remove_favorite,
    to_user_dict,
    upsert_favorite,
)

settings = get_settings()
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version="2.0.0",
    description="Local full-stack backend for the agentic home search and relocation assistant.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings.uploads_path.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(settings.uploads_path)), name="uploads")


@app.on_event("startup")
def on_startup():
    init_db()
    db = SessionLocal()
    try:
        ensure_admin_seed(db)
    finally:
        db.close()

    for warning in settings.runtime_warnings():
        logger.warning(warning)


@app.get("/")
def root():
    return {
        "name": settings.app_name,
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
        "runtime": f"{settings.api_prefix}/system/runtime-info",
    }


@app.get(f"{settings.api_prefix}/health")
def health():
    return {
        "ok": True,
        "time": datetime.utcnow().isoformat(),
        "bootstrap_ready": not bool(settings.required_runtime_missing()),
        "providers": provider_status(),
    }


@app.get(f"{settings.api_prefix}/system/runtime-info")
def runtime_info():
    return {
        "app_name": settings.app_name,
        "environment": settings.app_env,
        "bootstrap_ready": not bool(settings.required_runtime_missing()),
        "bootstrap_missing": settings.required_runtime_missing(),
        "warnings": settings.runtime_warnings(),
        "providers": provider_status(),
        "cache": {
            "backend": search_cache_backend(),
            "search_ttl_seconds": settings.search_cache_ttl_seconds,
            "semantic_ttl_seconds": settings.semantic_cache_ttl_seconds,
            "semantic_similarity_threshold": settings.semantic_similarity_threshold,
        },
        "openai": {
            "model": settings.openai_model,
            "embedding_model": settings.openai_embedding_model,
            "configured": bool(settings.openai_api_key),
        },
        "agentic": {
            "patterns": ["ReAct loop", "tool use", "retry", "validator", "critic", "self-optimization", "semantic cache"],
            "safe_reasoning": "The API returns summarized trace events and evidence, not hidden chain-of-thought.",
        },
    }


@app.post(f"{settings.api_prefix}/auth/register", response_model=TokenResponse)
def register_user(payload: UserCreate, db: Session = Depends(get_db)):
    user = create_user(db, payload.name, payload.email, payload.password)
    return create_login_response(user)


@app.post(f"{settings.api_prefix}/auth/login", response_model=TokenResponse)
def login_user(payload: UserLogin, db: Session = Depends(get_db)):
    user = authenticate_user(db, payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    return create_login_response(user)


@app.get(f"{settings.api_prefix}/auth/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return to_user_dict(user)


@app.post(f"{settings.api_prefix}/search/listings", response_model=SearchResponse)
async def search_listings(
    payload: SearchRequest,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_optional_user),
):
    result = await execute_search(payload, db, user=user)
    result["generated_at"] = datetime.utcnow()
    return result


@app.post(f"{settings.api_prefix}/favorites", response_model=FavoriteOut)
def save_favorite(
    payload: FavoriteCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    favorite = upsert_favorite(db, user, payload.model_dump())
    return favorite


@app.get(f"{settings.api_prefix}/favorites", response_model=list[FavoriteOut])
def list_favorites(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(Favorite).filter(Favorite.user_id == user.id).order_by(Favorite.created_at.desc()).all()


@app.delete(f"{settings.api_prefix}/favorites/{{favorite_id}}")
def delete_favorite(favorite_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    remove_favorite(db, user, favorite_id)
    return {"ok": True}


@app.post(f"{settings.api_prefix}/ads", response_model=AdvertisedPlaceOut)
async def submit_advertised_place(
    mode: Annotated[str, Form()],
    title: Annotated[str, Form()],
    address_line1: Annotated[str, Form()],
    city: Annotated[str, Form()],
    state: Annotated[str, Form()],
    zip_code: Annotated[str, Form()],
    price: Annotated[float, Form()],
    bedrooms: Annotated[float | None, Form()] = None,
    bathrooms: Annotated[float | None, Form()] = None,
    square_feet: Annotated[int | None, Form()] = None,
    property_type: Annotated[str | None, Form()] = None,
    description: Annotated[str, Form()] = "",
    contact_name: Annotated[str | None, Form()] = None,
    contact_email: Annotated[str | None, Form()] = None,
    image_urls: Annotated[str | None, Form()] = None,
    files: list[UploadFile] | None = File(default=None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ad = await create_advertised_place(
        db=db,
        user=user,
        mode=mode,
        title=title,
        address_line1=address_line1,
        city=city,
        state=state,
        zip_code=zip_code,
        price=price,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        square_feet=square_feet,
        property_type=property_type,
        description=description,
        contact_name=contact_name,
        contact_email=contact_email,
        image_urls_raw=image_urls,
        files=files,
    )
    return ad


@app.get(f"{settings.api_prefix}/ads/mine", response_model=list[AdvertisedPlaceOut])
def my_ads(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return (
        db.query(AdvertisedPlace)
        .filter(AdvertisedPlace.user_id == user.id)
        .order_by(AdvertisedPlace.created_at.desc())
        .all()
    )


@app.get(f"{settings.api_prefix}/admin/stats")
def get_admin_stats(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    stats = admin_stats(db)
    stats["agent_observability"] = agent_observability_stats(db)
    return stats


@app.get(f"{settings.api_prefix}/admin/pending-ads", response_model=list[AdvertisedPlaceOut])
def pending_ads(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    return (
        db.query(AdvertisedPlace)
        .filter(AdvertisedPlace.status == ModerationStatus.pending)
        .order_by(AdvertisedPlace.created_at.asc())
        .all()
    )


@app.post(f"{settings.api_prefix}/admin/pending-ads/{{ad_id}}/approve", response_model=AdvertisedPlaceOut)
def approve_ad(
    ad_id: int,
    payload: ModerationAction,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return moderation_update(db, ad_id, ModerationStatus.approved, payload.notes)


@app.post(f"{settings.api_prefix}/admin/pending-ads/{{ad_id}}/reject", response_model=AdvertisedPlaceOut)
def reject_ad(
    ad_id: int,
    payload: ModerationAction,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return moderation_update(db, ad_id, ModerationStatus.rejected, payload.notes)


@app.post(f"{settings.api_prefix}/agent/query")
async def agent_query(
    payload: AgentRequest,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_optional_user),
):
    listings = [item.model_dump() for item in payload.listings] if payload.listings else []
    answer = await answer_agent_question(payload.question, listings, db, user=user)
    return answer


@app.post(f"{settings.api_prefix}/agent/stream")
async def agent_stream(
    payload: AgentRequest,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_optional_user),
):
    listings = [item.model_dump() for item in payload.listings] if payload.listings else []

    async def event_stream():
        result = await answer_agent_question(payload.question, listings, db, user=user)
        text = result["answer"]
        chunk_size = 140
        for start in range(0, len(text), chunk_size):
            chunk = text[start : start + chunk_size]
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post(f"{settings.api_prefix}/agent/run", response_model=AgentRunResponse)
async def agent_run(
    payload: AgentRunRequest,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_optional_user),
):
    return await run_home_search_agent(payload, db, user=user)


@app.post(f"{settings.api_prefix}/agent/run/stream")
async def agent_run_stream(
    payload: AgentRunRequest,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_optional_user),
):
    async def event_stream():
        result = await run_home_search_agent(payload, db, user=user)
        for event in result.get("trace", []):
            yield "event: trace\n"
            yield f"data: {json.dumps(event, default=str)}\n\n"
        yield "event: final\n"
        yield f"data: {json.dumps(result, default=str)}\n\n"
        yield "event: done\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get(f"{settings.api_prefix}/agent/runs")
def get_agent_runs(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return list_agent_runs(db, user=user)


@app.get(f"{settings.api_prefix}/admin/agent-runs")
def get_admin_agent_runs(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return list_agent_runs(db, user=None, limit=50)


@app.post(f"{settings.api_prefix}/agent/runs/{{run_id}}/feedback")
def submit_agent_feedback(
    run_id: str,
    payload: AgentFeedbackRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return apply_agent_feedback(db, run_id, user, payload.rating, payload.label, payload.comment)
