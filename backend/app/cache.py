from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

from sqlalchemy.orm import Session

from .config import get_settings
from .models import SearchCacheEntry

settings = get_settings()

try:
    import redis
except Exception:  # pragma: no cover
    redis = None


@lru_cache(maxsize=1)
def _redis_client():
    if not settings.redis_url or redis is None:
        return None
    return redis.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=1,
        socket_timeout=1,
    )


def redis_available() -> bool:
    client = _redis_client()
    if client is None:
        return False
    try:
        return bool(client.ping())
    except Exception:
        return False


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def search_cache_backend() -> str:
    return "redis" if redis_available() else "database"


def get_cached_search(db: Session, cache_key: str) -> dict | None:
    client = _redis_client()
    if client is not None and redis_available():
        try:
            raw = client.get(f"search-cache:{cache_key}")
            if raw:
                return json.loads(raw)
            return None
        except Exception:
            pass

    entry = db.query(SearchCacheEntry).filter(SearchCacheEntry.cache_key == cache_key).first()
    if not entry:
        return None
    if entry.expires_at <= datetime.utcnow():
        db.delete(entry)
        db.commit()
        return None
    entry.hits += 1
    db.commit()
    return entry.response_payload


def set_cached_search(db: Session, cache_key: str, payload: dict) -> None:
    client = _redis_client()
    if client is not None and redis_available():
        try:
            client.setex(f"search-cache:{cache_key}", settings.search_cache_ttl_seconds, canonical_json(payload))
            return
        except Exception:
            pass

    entry = db.query(SearchCacheEntry).filter(SearchCacheEntry.cache_key == cache_key).first()
    expires_at = datetime.utcnow() + timedelta(seconds=settings.search_cache_ttl_seconds)
    if entry:
        entry.response_payload = payload
        entry.expires_at = expires_at
    else:
        entry = SearchCacheEntry(cache_key=cache_key, response_payload=payload, expires_at=expires_at)
        db.add(entry)
    db.commit()
