from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from statistics import mean
from typing import Any

from fastapi import HTTPException, UploadFile, status
from PIL import Image
from sqlalchemy import func
from sqlalchemy.orm import Session

from .ai import cosine_similarity, keyword_similarity, openai_service
from .auth import create_access_token, get_password_hash
from .cache import get_cached_search, hash_payload, search_cache_backend, set_cached_search
from .config import get_settings
from .demo_inventory import demo_seed_listings
from .models import (
    AdvertisedPlace,
    AgentQueryLog,
    Favorite,
    ModerationStatus,
    SearchEvent,
    SemanticCacheEntry,
    SearchCacheEntry,
    User,
    UserRole,
)
from .providers import (
    commute_snapshot,
    crime_snapshot,
    fetch_market_snapshot,
    geocode_query,
    haversine_miles,
    normalize_state_code,
    provider_status,
    school_snapshot,
    search_rentcast,
    walkability_snapshot,
)
from .schemas import SearchRequest

settings = get_settings()


def to_user_dict(user: User) -> dict:
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role.value if hasattr(user.role, "value") else str(user.role),
        "created_at": user.created_at,
    }



def ensure_admin_seed(db: Session) -> None:
    if settings.required_runtime_missing():
        return

    admin_email = settings.admin_email.lower().strip()
    admin = db.query(User).filter(User.email == admin_email).first()
    if admin:
        admin.name = settings.admin_name.strip() or admin.name
        admin.hashed_password = get_password_hash(settings.admin_password)
        admin.role = UserRole.admin
    else:
        admin = User(
            name=settings.admin_name.strip(),
            email=admin_email,
            hashed_password=get_password_hash(settings.admin_password),
            role=UserRole.admin,
        )
        db.add(admin)
    db.commit()



def create_user(db: Session, name: str, email: str, password: str) -> User:
    existing = db.query(User).filter(User.email == email.lower()).first()
    if existing:
        raise HTTPException(status_code=400, detail="An account with that email already exists.")
    user = User(name=name.strip(), email=email.lower(), hashed_password=get_password_hash(password), role=UserRole.user)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user



def create_login_response(user: User) -> dict:
    token = create_access_token(user.email, user.role.value if hasattr(user.role, "value") else str(user.role))
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": to_user_dict(user),
    }



def listing_from_ad(ad: AdvertisedPlace) -> dict:
    return {
        "id": f"user-ad:{ad.id}",
        "source": "user_ad",
        "mode": ad.mode,
        "title": ad.title,
        "address": ad.address_line1,
        "city": ad.city,
        "state": ad.state,
        "zip_code": ad.zip_code,
        "latitude": ad.latitude,
        "longitude": ad.longitude,
        "price": ad.price,
        "bedrooms": ad.bedrooms,
        "bathrooms": ad.bathrooms,
        "square_feet": ad.square_feet,
        "property_type": ad.property_type,
        "description": ad.description,
        "images": ad.images or [],
        "source_url": None,
        "provider": "User-submitted",
        "contact_name": ad.contact_name,
        "contact_email": ad.contact_email,
        "contact_phone": None,
        "days_on_market": None,
        "listed_date": ad.created_at.isoformat() if ad.created_at else None,
        "listing_type": "User Submitted",
        "hoa_fee": None,
        "mls_name": None,
        "mls_number": None,
        "market": {},
        "neighborhood": {},
        "commute": {},
        "scores": {},
        "reasons": [],
        "raw": {"status": ad.status.value},
    }



def _match_internal_ads(search: SearchRequest, db: Session) -> list[dict]:
    query = db.query(AdvertisedPlace).filter(
        AdvertisedPlace.status == ModerationStatus.approved,
        AdvertisedPlace.mode == search.mode,
    )
    if search.location.zip_code:
        query = query.filter(AdvertisedPlace.zip_code == search.location.zip_code)
    else:
        if search.location.city:
            query = query.filter(func.lower(AdvertisedPlace.city) == search.location.city.lower())
        if search.location.state:
            normalized_state = normalize_state_code(search.location.state) or search.location.state
            query = query.filter(func.lower(AdvertisedPlace.state) == normalized_state.lower())

    ads = query.limit(100).all()
    return [listing_from_ad(ad) for ad in ads]



def _fits_base_filters(listing: dict, search: SearchRequest) -> bool:
    f = search.filters
    price = listing.get("price")
    bedrooms = listing.get("bedrooms")
    bathrooms = listing.get("bathrooms")
    sqft = listing.get("square_feet")
    property_type = (listing.get("property_type") or "").lower().replace("-", " ")

    if f.min_price is not None and price is not None and price < f.min_price:
        return False
    if f.max_price is not None and price is not None and price > f.max_price:
        return False
    if f.min_bedrooms is not None and bedrooms is not None and bedrooms < f.min_bedrooms:
        return False
    if f.max_bedrooms is not None and bedrooms is not None and bedrooms > f.max_bedrooms:
        return False
    if f.min_bathrooms is not None and bathrooms is not None and bathrooms < f.min_bathrooms:
        return False
    if f.min_square_feet is not None and sqft is not None and sqft < f.min_square_feet:
        return False
    if f.max_square_feet is not None and sqft is not None and sqft > f.max_square_feet:
        return False
    if f.property_types:
        normalized = {item.lower().replace("-", " ") for item in f.property_types}
        if property_type and property_type not in normalized:
            return False
    return True



def _budget_score(listing: dict, search: SearchRequest) -> tuple[float, str | None]:
    price = listing.get("price")
    if price is None:
        return 55.0, None
    min_price = search.filters.min_price
    max_price = search.filters.max_price
    if min_price is None and max_price is None:
        return 70.0, None
    if min_price is not None and max_price is not None:
        if min_price <= price <= max_price:
            return 100.0, f"Price fits your target range (${min_price:,}–${max_price:,})."
        midpoint = (min_price + max_price) / 2
        gap = abs(price - midpoint) / max(midpoint, 1)
        score = max(0.0, 100 - gap * 100)
        return score, "Price is outside target but still comparable to your budget."
    if max_price is not None:
        if price <= max_price:
            return 100.0, f"Price is under your ${max_price:,} cap."
        gap = (price - max_price) / max(max_price, 1)
        return max(0.0, 100 - gap * 100), "Price is above your cap."
    if min_price is not None:
        if price >= min_price:
            return 90.0, None
        gap = (min_price - price) / max(min_price, 1)
        return max(0.0, 80 - gap * 100), None
    return 70.0, None



def _constraint_score(listing: dict, search: SearchRequest) -> tuple[float, list[str]]:
    constraints = search.constraints
    neighborhood = listing.get("neighborhood", {})
    commute = listing.get("commute", {})

    reasons: list[str] = []
    points = 0.0
    weight = 0.0

    if constraints.low_crime:
        safety = neighborhood.get("crime", {}).get("safety_score")
        if safety is not None:
            points += safety * 0.25
            weight += 25
            if safety >= 70:
                reasons.append("Reported crime signal is relatively low for the selected radius.")
        else:
            reasons.append("Crime data was not available for this result.")

    if constraints.good_schools:
        school = neighborhood.get("schools", {})
        school_score = school.get("avg_rating_10")
        if school_score is not None:
            normalized = school_score * 10
            points += normalized * 0.25
            weight += 25
            if school_score >= 7:
                reasons.append("Nearby schools score well on GreatSchools.")
        else:
            access_score = school.get("school_access_score")
            if access_score is not None:
                points += access_score * 0.15
                weight += 15
                reasons.append("School access looks good, but provider ratings were unavailable.")
            else:
                reasons.append("School quality data was not available.")

    if constraints.high_walkability or constraints.food_scene or constraints.parks_nearby or constraints.transit_nearby or constraints.grocery_nearby:
        walk = neighborhood.get("walkability", {})
        walk_score = walk.get("walkability_score")
        if walk_score is not None:
            points += walk_score * 0.2
            weight += 20
            if walk_score >= 60:
                reasons.append("The amenity mix suggests stronger walkability.")
        if constraints.parks_nearby and (walk.get("parks") or 0) > 0:
            points += 90 * 0.05
            weight += 5
            reasons.append("Parks are available nearby.")
        if constraints.transit_nearby and (walk.get("transit") or 0) > 0:
            points += 90 * 0.05
            weight += 5
            reasons.append("Transit stops are present nearby.")
        if constraints.food_scene and (walk.get("food") or 0) >= 3:
            points += 90 * 0.05
            weight += 5
            reasons.append("The area has a denser food and cafe cluster.")
        if constraints.grocery_nearby and (walk.get("grocery") or 0) >= 1:
            points += 90 * 0.05
            weight += 5
            reasons.append("A grocery option appears within the local radius.")

    if constraints.max_commute_minutes:
        commute_minutes = commute.get("minutes")
        if commute_minutes is not None:
            weight += 20
            if commute_minutes <= constraints.max_commute_minutes:
                points += 100 * 0.2
                reasons.append(f"Estimated commute is within your {constraints.max_commute_minutes}-minute target.")
            else:
                overshoot = (commute_minutes - constraints.max_commute_minutes) / max(constraints.max_commute_minutes, 1)
                points += max(0.0, 100 - overshoot * 100) * 0.2
        else:
            reasons.append("Commute data was not available for this result.")

    if weight == 0:
        return 70.0, reasons
    return min(100.0, points / (weight / 100)), reasons



def score_listing(listing: dict, search: SearchRequest) -> dict:
    budget_score, budget_reason = _budget_score(listing, search)
    constraint_score, extra_reasons = _constraint_score(listing, search)

    bedroom_score = 70.0
    if search.filters.min_bedrooms is not None and listing.get("bedrooms") is not None:
        bedroom_score = 100.0 if listing["bedrooms"] >= search.filters.min_bedrooms else 20.0

    semantic_score = listing.get("scores", {}).get("semantic_score")
    if semantic_score is not None:
        overall = round((budget_score * 0.32) + (constraint_score * 0.33) + (bedroom_score * 0.15) + (semantic_score * 0.20), 1)
    else:
        overall = round((budget_score * 0.4) + (constraint_score * 0.4) + (bedroom_score * 0.2), 1)

    reasons = []
    if budget_reason:
        reasons.append(budget_reason)
    reasons.extend(extra_reasons)
    if listing.get("images"):
        reasons.append("The listing includes provider-supplied photos.")
    if search.semantic_query and semantic_score is not None and semantic_score >= 60:
        reasons.append("The listing semantically matches your natural-language intent.")
    if listing.get("market", {}).get("median_price"):
        reasons.append("Local market context is attached for this ZIP code.")

    listing.setdefault("scores", {})
    listing["scores"].update(
        {
            "fit_score": overall,
            "budget_score": round(budget_score, 1),
            "constraint_score": round(constraint_score, 1),
            "bedroom_score": round(bedroom_score, 1),
            "crime_score": listing.get("neighborhood", {}).get("crime", {}).get("safety_score"),
            "school_score": (
                (listing.get("neighborhood", {}).get("schools", {}).get("avg_rating_10") or 0) * 10
                if listing.get("neighborhood", {}).get("schools", {}).get("avg_rating_10") is not None
                else listing.get("neighborhood", {}).get("schools", {}).get("school_access_score")
            ),
            "walkability_score": listing.get("neighborhood", {}).get("walkability", {}).get("walkability_score"),
            "commute_minutes": listing.get("commute", {}).get("minutes"),
        }
    )
    listing["reasons"] = reasons[:6]
    return listing


async def enrich_listing(listing: dict, search: SearchRequest, warnings: list[str]) -> dict:
    lat = listing.get("latitude")
    lon = listing.get("longitude")

    if listing.get("zip_code"):
        market = await fetch_market_snapshot(listing.get("zip_code"), search.mode)
        if market:
            listing["market"] = market
        else:
            listing.setdefault("market", listing.get("market") or {})

    neighborhood = listing.get("neighborhood", {}) or {}
    if lat is not None and lon is not None:
        if not neighborhood.get("walkability"):
            neighborhood["walkability"] = await walkability_snapshot(lat, lon)

        if search.constraints.good_schools:
            if not neighborhood.get("schools"):
                neighborhood["schools"] = await school_snapshot(lat, lon)
        else:
            neighborhood.setdefault("schools", neighborhood.get("schools") or {})

        if search.constraints.low_crime:
            if not neighborhood.get("crime"):
                neighborhood["crime"] = await crime_snapshot(lat, lon)
        else:
            neighborhood.setdefault("crime", neighborhood.get("crime") or {})

        if search.constraints.commute_destination:
            listing["commute"] = await commute_snapshot(
                lat,
                lon,
                search.constraints.commute_destination,
                search.constraints.commute_mode,
            )
        else:
            listing.setdefault("commute", listing.get("commute") or {})
    else:
        warnings.append(
            f"Skipping neighborhood enrichment for {listing.get('title') or listing.get('address')} due to missing coordinates."
        )

    listing["neighborhood"] = neighborhood
    return score_listing(listing, search)



def _post_filter_listing(listing: dict, search: SearchRequest, providers: dict, warnings: list[str]) -> bool:
    constraints = search.constraints
    walk = listing.get("neighborhood", {}).get("walkability", {})

    if constraints.low_crime:
        if providers["crimeometer"]:
            safety = listing.get("neighborhood", {}).get("crime", {}).get("safety_score")
            if safety is not None and safety < 55:
                return False
        elif "Crime filter requested, but CRIMEOMETER_API_KEY is not configured. The filter was skipped." not in warnings:
            warnings.append("Crime filter requested, but CRIMEOMETER_API_KEY is not configured. The filter was skipped.")

    if constraints.good_schools:
        school_rating = listing.get("neighborhood", {}).get("schools", {}).get("avg_rating_10")
        school_access = listing.get("neighborhood", {}).get("schools", {}).get("school_access_score")
        if school_rating is not None and school_rating < 6.5:
            return False
        if school_rating is None and school_access is not None and school_access < 40:
            return False

    if constraints.high_walkability:
        score = walk.get("walkability_score")
        if score is not None and score < 55:
            return False

    if constraints.parks_nearby:
        parks = walk.get("parks")
        if parks is not None and parks < 1:
            return False

    if constraints.transit_nearby:
        transit = walk.get("transit")
        if transit is not None and transit < 1:
            return False

    if constraints.food_scene:
        food = walk.get("food")
        if food is not None and food < 3:
            return False

    if constraints.grocery_nearby:
        grocery = walk.get("grocery")
        if grocery is not None and grocery < 1:
            return False

    if constraints.max_commute_minutes and constraints.commute_destination:
        commute_minutes = listing.get("commute", {}).get("minutes")
        if commute_minutes is not None and commute_minutes > constraints.max_commute_minutes:
            return False

    return True



def _sort_key(search: SearchRequest, listing: dict):
    scores = listing.get("scores", {})
    if search.sort_by == "price_asc":
        return (listing.get("price") is None, listing.get("price") or 0)
    if search.sort_by == "price_desc":
        return (listing.get("price") is None, -(listing.get("price") or 0))
    if search.sort_by == "commute":
        return (listing.get("commute", {}).get("minutes") is None, listing.get("commute", {}).get("minutes") or 9999)
    if search.sort_by == "crime":
        return (scores.get("crime_score") is None, -(scores.get("crime_score") or 0))
    if search.sort_by == "schools":
        return (scores.get("school_score") is None, -(scores.get("school_score") or 0))
    if search.sort_by == "walkability":
        return (scores.get("walkability_score") is None, -(scores.get("walkability_score") or 0))
    if search.sort_by == "semantic":
        return (scores.get("semantic_score") is None, -(scores.get("semantic_score") or 0))
    return (scores.get("fit_score") is None, -(scores.get("fit_score") or 0))



def _location_label(search: SearchRequest) -> str:
    if search.location.address:
        return search.location.address
    if search.location.zip_code:
        return search.location.zip_code
    return ", ".join([item for item in [search.location.city, search.location.state] if item])



def _log_search_event(db: Session, user: User | None, search: SearchRequest, result_count: int, cache_status: str) -> None:
    event = SearchEvent(
        user_id=user.id if user else None,
        mode=search.mode,
        location_label=_location_label(search),
        search_payload=search.model_dump(mode="json"),
        result_count=result_count,
        cache_status=cache_status,
    )
    db.add(event)
    db.commit()



def _log_agent_query(db: Session, user: User | None, question: str, scope_key: str, cache_status: str) -> None:
    event = AgentQueryLog(
        user_id=user.id if user else None,
        scope_key=scope_key,
        question_text=question,
        cache_status=cache_status,
    )
    db.add(event)
    db.commit()



def _clean_expired_cache_rows(db: Session) -> None:
    now = datetime.utcnow()
    db.query(SearchCacheEntry).filter(SearchCacheEntry.expires_at <= now).delete(synchronize_session=False)
    db.query(SemanticCacheEntry).filter(SemanticCacheEntry.expires_at <= now).delete(synchronize_session=False)
    db.commit()



def _listing_semantic_document(listing: dict) -> str:
    walk = listing.get("neighborhood", {}).get("walkability", {})
    schools = listing.get("neighborhood", {}).get("schools", {})
    crime = listing.get("neighborhood", {}).get("crime", {})
    commute = listing.get("commute", {})
    market = listing.get("market", {})
    lines = [
        f"Title: {listing.get('title')}",
        f"Address: {listing.get('address')}, {listing.get('city')}, {listing.get('state')} {listing.get('zip_code')}",
        f"Price: {listing.get('price')}",
        f"Beds: {listing.get('bedrooms')}",
        f"Baths: {listing.get('bathrooms')}",
        f"Square feet: {listing.get('square_feet')}",
        f"Property type: {listing.get('property_type')}",
        f"Days on market: {listing.get('days_on_market')}",
        f"Description: {listing.get('description')}",
        f"Walkability score: {walk.get('walkability_score')}",
        f"Food count: {walk.get('food')} Grocery count: {walk.get('grocery')} Parks count: {walk.get('parks')} Transit count: {walk.get('transit')}",
        f"Nearest park miles: {walk.get('nearest_park_miles')} Nearest transit miles: {walk.get('nearest_transit_miles')} Nearest grocery miles: {walk.get('nearest_grocery_miles')}",
        f"School rating: {schools.get('avg_rating_10')} School access: {schools.get('school_access_score')} Nearest school miles: {schools.get('nearest_school_miles')}",
        f"Crime score: {crime.get('safety_score')}",
        f"Commute minutes: {commute.get('minutes')}",
        f"Local median price or rent: {market.get('median_price')}",
        "Reasons: " + "; ".join(listing.get("reasons", [])),
    ]
    return "\n".join(str(line) for line in lines if line is not None)


async def _semantic_rank_listings(query: str, listings: list[dict], warnings: list[str]) -> None:
    if not query or not listings:
        return
    docs = [_listing_semantic_document(item) for item in listings]
    try:
        if openai_service.configured:
            embeddings = await openai_service.embed_texts([query, *docs])
            if len(embeddings) == len(docs) + 1:
                query_embedding = embeddings[0]
                doc_embeddings = embeddings[1:]
                for listing, embedding in zip(listings, doc_embeddings):
                    score = max(0.0, cosine_similarity(query_embedding, embedding)) * 100
                    listing.setdefault("scores", {})["semantic_score"] = round(score, 1)
                return
    except Exception as exc:
        warnings.append(f"Semantic ranking fell back to keyword similarity because embeddings failed: {exc}")

    for listing, doc in zip(listings, docs):
        listing.setdefault("scores", {})["semantic_score"] = round(keyword_similarity(query, doc) * 100, 1)




async def _search_center(search: SearchRequest) -> dict | None:
    query = search.location.address
    if not query:
        query = " ".join(item for item in [search.location.city, search.location.state, search.location.zip_code] if item)
    if not query:
        return None
    return await geocode_query(query)


def _attach_distance_from_center(listings: list[dict], center: dict | None) -> None:
    if not center:
        return
    for listing in listings:
        lat, lon = listing.get("latitude"), listing.get("longitude")
        if lat is None or lon is None:
            continue
        try:
            listing["distance_from_search_miles"] = round(
                haversine_miles(float(lat), float(lon), center["latitude"], center["longitude"]), 1
            )
        except Exception:
            continue


def _coarse_location_match(listing: dict, search: SearchRequest, center: dict | None) -> bool:
    radius = search.radius_miles or 25
    try:
        radius = max(1, min(100, int(radius)))
    except Exception:
        radius = 25

    if center and listing.get("latitude") is not None and listing.get("longitude") is not None:
        try:
            distance = haversine_miles(float(listing["latitude"]), float(listing["longitude"]), center["latitude"], center["longitude"])
            if distance <= radius:
                return True
        except Exception:
            pass

    if search.location.zip_code and str(listing.get("zip_code") or "") == str(search.location.zip_code):
        return True
    if search.location.city and search.location.state:
        city_match = str(listing.get("city") or "").lower() == search.location.city.lower()
        state_match = str(normalize_state_code(listing.get("state")) or listing.get("state") or "").upper() == str(normalize_state_code(search.location.state) or search.location.state).upper()
        return city_match and state_match
    return True


def _demo_results_for_search(search: SearchRequest, center: dict | None) -> list[dict]:
    if not settings.demo_seed_listings_enabled:
        return []
    candidates = demo_seed_listings(search.mode)
    matched = [item for item in candidates if _coarse_location_match(item, search, center)]
    if not matched and search.location.state and str(search.location.state).upper() == "CA":
        matched = candidates
    _attach_distance_from_center(matched, center)
    return matched[: max(search.limit, min(settings.demo_seed_limit, 30))]

def _summarize_results(search: SearchRequest, results: list[dict], cache_status: str) -> dict:
    prices = [item["price"] for item in results if item.get("price") is not None]
    fit_scores = [item.get("scores", {}).get("fit_score") for item in results if item.get("scores", {}).get("fit_score") is not None]
    commute_minutes = [item.get("commute", {}).get("minutes") for item in results if item.get("commute", {}).get("minutes") is not None]
    walk_scores = [item.get("scores", {}).get("walkability_score") for item in results if item.get("scores", {}).get("walkability_score") is not None]

    return {
        "location": _location_label(search),
        "mode": search.mode,
        "result_count": len(results),
        "average_price": round(mean(prices), 0) if prices else None,
        "average_fit_score": round(mean(fit_scores), 1) if fit_scores else None,
        "average_commute_minutes": round(mean(commute_minutes), 1) if commute_minutes else None,
        "average_walkability": round(mean(walk_scores), 1) if walk_scores else None,
        "semantic_query_applied": bool(search.semantic_query),
        "cache_status": cache_status,
    }


async def execute_search(search: SearchRequest, db: Session, user: User | None = None) -> dict:
    if not (search.location.address or search.location.zip_code or (search.location.city and search.location.state)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide an address, a ZIP code, or a city + state to search.",
        )

    providers = provider_status()
    warnings: list[str] = []
    _clean_expired_cache_rows(db)

    payload_for_cache = search.model_dump(mode="json")
    cache_key = hash_payload(payload_for_cache)
    cached = get_cached_search(db, cache_key)
    if cached:
        cached["provider_status"] = providers
        cached["cache"] = {
            "status": "exact_hit",
            "backend": search_cache_backend(),
            "semantic_enabled": openai_service.configured,
            "demo_seed_enabled": settings.demo_seed_listings_enabled,
        }
        _log_search_event(db, user, search, len(cached.get("results", [])), "exact_hit")
        return cached

    center = await _search_center(search)

    external_results: list[dict] = []
    provider_messages: list[str] = []
    if providers["rentcast"]:
        external_results, provider_messages = await search_rentcast(payload_for_cache)
        warnings.extend(provider_messages)
    else:
        if settings.demo_seed_listings_enabled:
            warnings.append("RentCast is not configured. Showing seeded demo cache records so the interface and agent workflow remain demonstrable.")
        else:
            warnings.append("RENTCAST_API_KEY is not configured, so live third-party listings are disabled.")

    internal_results = _match_internal_ads(search, db)
    demo_results: list[dict] = []
    if settings.demo_seed_listings_enabled and (not external_results or settings.demo_seed_when_live_empty):
        demo_results = _demo_results_for_search(search, center)
        if demo_results:
            if external_results:
                warnings.append("Live listings were retrieved. Seeded demo cache records are also visible because demo cache display is enabled.")
            else:
                warnings.append("Live listing search returned no records or no provider was connected. Seeded demo cache records are shown and clearly labeled.")

    raw_results = external_results + internal_results + demo_results
    _attach_distance_from_center(raw_results, center)
    results = [item for item in raw_results if _fits_base_filters(item, search) and _coarse_location_match(item, search, center)]

    if search.location.address and center:
        narrowed = []
        radius = search.radius_miles or 25
        for listing in results:
            lat, lon = listing.get("latitude"), listing.get("longitude")
            if lat is None or lon is None:
                narrowed.append(listing)
                continue
            if haversine_miles(float(lat), float(lon), center["latitude"], center["longitude"]) <= radius:
                narrowed.append(listing)
        results = narrowed
    elif not center:
        warnings.append("The search location could not be geocoded, so radius filtering used city, state, and ZIP matching only.")

    results = results[: max(search.limit * 2, 24)]

    enriched: list[dict] = []
    for listing in results:
        enriched.append(await enrich_listing(listing, search, warnings))

    if search.semantic_query:
        await _semantic_rank_listings(search.semantic_query, enriched, warnings)
        for listing in enriched:
            score_listing(listing, search)

    filtered = [listing for listing in enriched if _post_filter_listing(listing, search, providers, warnings)]
    sorted_results = sorted(filtered, key=lambda item: _sort_key(search, item))
    final_results = sorted_results[: search.limit]

    data_sources = {
        "rentcast_live_count": len(external_results),
        "user_submitted_count": len(internal_results),
        "demo_seed_count": len(demo_results),
        "search_center": center,
        "radius_miles": search.radius_miles,
        "provider_messages": provider_messages[:8],
    }

    summary = _summarize_results(search, final_results, "miss")
    summary["data_sources"] = data_sources

    response = {
        "results": final_results,
        "provider_status": providers,
        "warnings": warnings,
        "summary": summary,
        "cache": {
            "status": "miss",
            "backend": search_cache_backend(),
            "semantic_enabled": openai_service.configured,
            "demo_seed_enabled": settings.demo_seed_listings_enabled,
        },
    }
    set_cached_search(db, cache_key, response)
    _log_search_event(db, user, search, len(final_results), "miss")
    return response


def upsert_favorite(db: Session, user: User, payload: dict) -> Favorite:
    favorite = (
        db.query(Favorite)
        .filter(Favorite.user_id == user.id, Favorite.listing_id == payload["listing_id"], Favorite.source == payload["source"])
        .first()
    )
    if favorite:
        favorite.title = payload.get("title")
        favorite.payload = payload.get("payload", {})
    else:
        favorite = Favorite(
            user_id=user.id,
            listing_id=payload["listing_id"],
            source=payload["source"],
            title=payload.get("title"),
            payload=payload.get("payload", {}),
        )
        db.add(favorite)
    db.commit()
    db.refresh(favorite)
    return favorite



def remove_favorite(db: Session, user: User, favorite_id: int) -> None:
    favorite = db.query(Favorite).filter(Favorite.id == favorite_id, Favorite.user_id == user.id).first()
    if not favorite:
        raise HTTPException(status_code=404, detail="Favorite not found.")
    db.delete(favorite)
    db.commit()



def _normalized_state_match(input_state: str, geo_state: str | None, geo_state_code: str | None) -> bool:
    if not input_state:
        return True
    normalized_input = input_state.strip().lower()
    normalized_geo_state = geo_state.strip().lower() if geo_state else None
    normalized_geo_code = geo_state_code.strip().lower() if geo_state_code else None
    return normalized_input in {value for value in [normalized_geo_state, normalized_geo_code] if value}


async def _save_uploaded_image(file: UploadFile) -> str:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail=f"{file.filename or 'An image'} was empty.")
    if len(content) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"{file.filename or 'An image'} exceeds the 8MB limit.")
    try:
        image = Image.open(BytesIO(content))
        image.verify()
        image = Image.open(BytesIO(content))
        if image.width < 640 or image.height < 480:
            raise HTTPException(status_code=400, detail=f"{file.filename or 'An image'} is too small. Minimum is 640x480.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{file.filename or 'An image'} is not a valid image file.") from exc

    extension = Path(file.filename or "upload.jpg").suffix.lower()
    if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
        extension = ".jpg"
    settings.uploads_path.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}{extension}"
    filepath = settings.uploads_path / filename
    with open(filepath, "wb") as output_file:
        output_file.write(content)
    return f"/uploads/{filename}"


async def _validate_remote_image_url(url: str) -> str:
    import httpx

    if not re.match(r"^https?://", url.strip(), flags=re.IGNORECASE):
        raise HTTPException(status_code=400, detail=f"Invalid image URL: {url}")

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers={"User-Agent": settings.user_agent}) as client:
        response = await client.get(url.strip())
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Could not fetch image URL: {url}")
        if not response.headers.get("content-type", "").startswith("image/"):
            raise HTTPException(status_code=400, detail=f"URL did not return an image: {url}")
        content = response.content
        try:
            image = Image.open(BytesIO(content))
            if image.width < 640 or image.height < 480:
                raise HTTPException(status_code=400, detail=f"Image URL is too small: {url}")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Image URL is not a valid image: {url}") from exc
    return url.strip()


async def create_advertised_place(
    *,
    db: Session,
    user: User,
    mode: str,
    title: str,
    address_line1: str,
    city: str,
    state: str,
    zip_code: str,
    price: float,
    bedrooms: float | None,
    bathrooms: float | None,
    square_feet: int | None,
    property_type: str | None,
    description: str,
    contact_name: str | None,
    contact_email: str | None,
    image_urls_raw: str | None,
    files: list[UploadFile] | None,
) -> AdvertisedPlace:
    if not any(char.isdigit() for char in address_line1):
        raise HTTPException(status_code=400, detail="Invalid address: a street number is required.")
    if price <= 0:
        raise HTTPException(status_code=400, detail="Price must be greater than zero.")
    if bedrooms is not None and not (0 <= bedrooms <= 20):
        raise HTTPException(status_code=400, detail="Bedrooms must be between 0 and 20.")
    if bathrooms is not None and not (0 <= bathrooms <= 20):
        raise HTTPException(status_code=400, detail="Bathrooms must be between 0 and 20.")
    if square_feet is not None and not (100 <= square_feet <= 20000):
        raise HTTPException(status_code=400, detail="Square footage must be between 100 and 20,000.")
    if len(description.strip()) < 30:
        raise HTTPException(status_code=400, detail="Description is too short. Please provide more real detail.")

    full_address = f"{address_line1}, {city}, {state} {zip_code}"
    geo = await geocode_query(full_address)
    if not geo:
        raise HTTPException(status_code=400, detail="Invalid address. We could not verify that location.")
    if geo.get("city"):
        normalized_input_city = city.strip().lower()
        normalized_geo_city = geo["city"].strip().lower()
        if normalized_input_city not in normalized_geo_city and normalized_geo_city not in normalized_input_city:
            raise HTTPException(status_code=400, detail="Invalid address. City does not match the geocoded result.")
    if not _normalized_state_match(state, geo.get("state"), geo.get("state_code")):
        raise HTTPException(status_code=400, detail="Invalid address. State does not match the geocoded result.")

    images: list[str] = []
    remote_urls = []
    if image_urls_raw:
        remote_urls = [item.strip() for item in re.split(r"[\n,]+", image_urls_raw) if item.strip()]
    if len(remote_urls) > 6:
        raise HTTPException(status_code=400, detail="At most 6 image URLs are allowed.")
    if files and len(files) > 6:
        raise HTTPException(status_code=400, detail="At most 6 uploaded images are allowed.")

    for url in remote_urls:
        images.append(await _validate_remote_image_url(url))
    for file in files or []:
        images.append(await _save_uploaded_image(file))

    if not images:
        raise HTTPException(status_code=400, detail="At least one valid property image is required.")

    validation_flags: list[str] = []
    validation_score = 85.0

    if "po box" in address_line1.lower():
        validation_flags.append("PO boxes are not accepted for property addresses.")
        validation_score -= 30
    if len(images) < 2:
        validation_flags.append("Only one image was provided.")
        validation_score -= 10
    if square_feet and bedrooms and square_feet < bedrooms * 180:
        validation_flags.append("Square footage looks low relative to bedroom count.")
        validation_score -= 12

    ai_review = await openai_service.review_listing(
        {
            "mode": mode,
            "title": title,
            "address": full_address,
            "price": price,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "square_feet": square_feet,
            "property_type": property_type,
            "description": description,
            "image_count": len(images),
            "contact_name": contact_name,
            "contact_email": contact_email,
            "geocoded_city": geo.get("city"),
            "geocoded_state": geo.get("state"),
            "zip_code": geo.get("zip_code"),
        }
    )
    if ai_review:
        if isinstance(ai_review.get("flags"), list):
            validation_flags.extend([str(item) for item in ai_review["flags"][:5]])
        if ai_review.get("score") is not None:
            try:
                validation_score = min(validation_score, float(ai_review["score"]))
            except (TypeError, ValueError):
                pass
        if ai_review.get("verdict") == "reject":
            raise HTTPException(status_code=400, detail="Listing validation failed. Please check the address, details, and images.")

    validation_score = max(0.0, min(100.0, validation_score))

    ad = AdvertisedPlace(
        user_id=user.id,
        mode=mode,
        title=title.strip(),
        address_line1=address_line1.strip(),
        city=city.strip(),
        state=state.strip(),
        zip_code=zip_code.strip(),
        latitude=geo["latitude"],
        longitude=geo["longitude"],
        price=price,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        square_feet=square_feet,
        property_type=property_type.strip() if property_type else None,
        description=description.strip(),
        images=images,
        contact_name=contact_name.strip() if contact_name else None,
        contact_email=contact_email.strip() if contact_email else None,
        status=ModerationStatus.pending,
        validation_score=validation_score,
        validation_flags=validation_flags,
    )
    db.add(ad)
    db.commit()
    db.refresh(ad)
    return ad



def moderation_update(db: Session, ad_id: int, status_value: ModerationStatus, notes: str | None) -> AdvertisedPlace:
    ad = db.query(AdvertisedPlace).filter(AdvertisedPlace.id == ad_id).first()
    if not ad:
        raise HTTPException(status_code=404, detail="Listing not found.")
    ad.status = status_value
    ad.moderation_notes = notes
    db.commit()
    db.refresh(ad)
    return ad



def admin_stats(db: Session) -> dict:
    total_searches = db.query(SearchEvent).count()
    exact_hits = db.query(SearchEvent).filter(SearchEvent.cache_status == "exact_hit").count()
    total_agent_queries = db.query(AgentQueryLog).count()
    semantic_hits = db.query(AgentQueryLog).filter(AgentQueryLog.cache_status == "semantic_hit").count()

    top_locations = (
        db.query(SearchEvent.location_label, func.count(SearchEvent.id).label("count"))
        .group_by(SearchEvent.location_label)
        .order_by(func.count(SearchEvent.id).desc())
        .limit(5)
        .all()
    )

    recent_modes = Counter(
        row[0]
        for row in db.query(SearchEvent.mode)
        .order_by(SearchEvent.created_at.desc())
        .limit(50)
        .all()
    )

    return {
        "users": db.query(User).count(),
        "favorites": db.query(Favorite).count(),
        "submitted_ads": db.query(AdvertisedPlace).count(),
        "pending_ads": db.query(AdvertisedPlace).filter(AdvertisedPlace.status == ModerationStatus.pending).count(),
        "approved_ads": db.query(AdvertisedPlace).filter(AdvertisedPlace.status == ModerationStatus.approved).count(),
        "rejected_ads": db.query(AdvertisedPlace).filter(AdvertisedPlace.status == ModerationStatus.rejected).count(),
        "searches_total": total_searches,
        "search_cache_hits": exact_hits,
        "search_cache_hit_rate": round((exact_hits / total_searches) * 100, 1) if total_searches else 0,
        "agent_queries_total": total_agent_queries,
        "semantic_cache_hits": semantic_hits,
        "semantic_hit_rate": round((semantic_hits / total_agent_queries) * 100, 1) if total_agent_queries else 0,
        "cache_backend": search_cache_backend(),
        "provider_status": provider_status(),
        "top_locations": [{"location": location, "count": count} for location, count in top_locations],
        "recent_modes": dict(recent_modes),
    }



def _fallback_agent_answer(question: str, listings: list[dict]) -> dict:
    if not listings:
        return {
            "answer": "I do not have any listings to compare yet. Run a search first, then ask me to rank or compare the results.",
            "highlights": [],
            "warnings": [],
            "cache_status": "fallback",
        }

    top = listings[:3]
    parts = []
    highlights = []
    for idx, listing in enumerate(top, start=1):
        fit = listing.get("scores", {}).get("fit_score")
        reasons = "; ".join(listing.get("reasons", [])[:2]) or "Matches several of your constraints."
        parts.append(
            f"{idx}. {listing.get('title') or listing.get('address')} — ${int(listing.get('price') or 0):,}. "
            f"Fit score: {fit}. {reasons}"
        )
        highlights.append(listing.get("title") or listing.get("address"))
    return {
        "answer": "Top matches based on the current search:\n" + "\n".join(parts),
        "highlights": highlights,
        "warnings": [],
        "cache_status": "fallback",
    }



def _agent_scope_key(listings: list[dict]) -> str:
    slim = [
        {
            "id": item.get("id"),
            "price": item.get("price"),
            "fit": item.get("scores", {}).get("fit_score"),
            "address": item.get("address"),
        }
        for item in listings[:10]
    ]
    return hash_payload(slim)



def _get_semantic_cache_hit(db: Session, scope_key: str, question_embedding: list[float]) -> tuple[SemanticCacheEntry | None, float]:
    threshold = settings.semantic_similarity_threshold
    now = datetime.utcnow()
    candidates = (
        db.query(SemanticCacheEntry)
        .filter(SemanticCacheEntry.scope_key == scope_key, SemanticCacheEntry.expires_at > now)
        .order_by(SemanticCacheEntry.updated_at.desc())
        .limit(20)
        .all()
    )
    best_entry: SemanticCacheEntry | None = None
    best_score = -1.0
    for entry in candidates:
        similarity = cosine_similarity(question_embedding, entry.question_embedding or [])
        if similarity > best_score:
            best_score = similarity
            best_entry = entry
    if best_entry and best_score >= max(best_entry.similarity_threshold, threshold):
        best_entry.hits += 1
        db.commit()
        return best_entry, best_score
    return None, best_score



def _store_semantic_cache(db: Session, scope_key: str, question_text: str, question_embedding: list[float], response_payload: dict) -> None:
    entry = SemanticCacheEntry(
        scope_key=scope_key,
        question_text=question_text,
        question_embedding=question_embedding,
        response_payload=response_payload,
        similarity_threshold=settings.semantic_similarity_threshold,
        model_name=settings.openai_model,
        expires_at=datetime.utcnow() + timedelta(seconds=settings.semantic_cache_ttl_seconds),
    )
    db.add(entry)
    db.commit()


async def answer_agent_question(question: str, listings: list[dict], db: Session, user: User | None = None) -> dict:
    scope_key = _agent_scope_key(listings)

    if not openai_service.configured:
        fallback = _fallback_agent_answer(question, listings)
        _log_agent_query(db, user, question, scope_key, "fallback")
        return fallback

    slim_listings = [
        {
            "title": listing.get("title"),
            "address": listing.get("address"),
            "price": listing.get("price"),
            "bedrooms": listing.get("bedrooms"),
            "bathrooms": listing.get("bathrooms"),
            "square_feet": listing.get("square_feet"),
            "fit_score": listing.get("scores", {}).get("fit_score"),
            "semantic_score": listing.get("scores", {}).get("semantic_score"),
            "commute_minutes": listing.get("commute", {}).get("minutes"),
            "crime_score": listing.get("scores", {}).get("crime_score"),
            "school_score": listing.get("scores", {}).get("school_score"),
            "walkability_score": listing.get("scores", {}).get("walkability_score"),
            "reasons": listing.get("reasons", []),
        }
        for listing in listings[:8]
    ]

    question_embedding: list[float] | None = None
    try:
        embeds = await openai_service.embed_texts([question])
        if embeds:
            question_embedding = embeds[0]
    except Exception:
        question_embedding = None

    if question_embedding:
        cache_entry, _ = _get_semantic_cache_hit(db, scope_key, question_embedding)
        if cache_entry:
            payload = dict(cache_entry.response_payload)
            payload["cache_status"] = "semantic_hit"
            _log_agent_query(db, user, question, scope_key, "semantic_hit")
            return payload

    try:
        text = await openai_service.answer_with_listings(question=question, slim_listings=slim_listings)
        highlights = [item.get("title") for item in slim_listings[:3] if item.get("title")]
        payload = {"answer": text, "highlights": highlights, "warnings": [], "cache_status": "miss"}
        if question_embedding:
            _store_semantic_cache(db, scope_key, question, question_embedding, payload)
        _log_agent_query(db, user, question, scope_key, "miss")
        return payload
    except Exception as exc:
        fallback = _fallback_agent_answer(question, listings)
        fallback["warnings"] = [f"OpenAI call failed, so a rule-based answer was returned instead: {exc}"]
        _log_agent_query(db, user, question, scope_key, "fallback")
        return fallback
