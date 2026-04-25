from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from .cache import redis_available
from .config import get_settings

settings = get_settings()


STATE_NAME_TO_CODE = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
    "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH",
    "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "cal": "CA", "cali": "CA", "wash": "WA", "newyork": "NY"
}


def normalize_state_code(value: str | None) -> str | None:
    if not value:
        return None
    clean = value.strip()
    if not clean:
        return None
    upper = clean.upper()
    if len(upper) == 2 and upper.isalpha():
        return upper
    return STATE_NAME_TO_CODE.get(clean.lower())


class TTLCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str):
        value = self._store.get(key)
        if not value:
            return None
        expires_at, payload = value
        if expires_at < time.time():
            self._store.pop(key, None)
            return None
        return payload

    def set(self, key: str, payload: Any):
        self._store[key] = (time.time() + self.ttl_seconds, payload)


geo_cache = TTLCache(ttl_seconds=60 * 60 * 24)
walk_cache = TTLCache(ttl_seconds=60 * 60 * 8)
school_cache = TTLCache(ttl_seconds=60 * 60 * 8)
crime_cache = TTLCache(ttl_seconds=60 * 60 * 6)
commute_cache = TTLCache(ttl_seconds=60 * 60 * 6)
market_cache = TTLCache(ttl_seconds=60 * 60 * 12)


def bool_configured(value: str | None) -> bool:
    return bool(value and value.strip())


def provider_status() -> dict:
    return {
        "rentcast": bool_configured(settings.rentcast_api_key),
        "openrouteservice": bool_configured(settings.openrouteservice_api_key),
        "greatschools": bool_configured(settings.greatschools_api_key),
        "crimeometer": bool_configured(settings.crimeometer_api_key),
        "openai": bool_configured(settings.openai_api_key),
        "redis": redis_available(),
        "demo_seed": bool(settings.demo_seed_listings_enabled),
        "nominatim": True,
        "overpass": True,
    }


async def _async_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=25.0,
        follow_redirects=True,
        headers={"User-Agent": settings.user_agent},
    )


def _get_value(record: dict, key: str, default=None):
    value: Any = record
    for part in key.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default
    return value


def _coalesce(record: dict, keys: list[str], default=None):
    for key in keys:
        value = _get_value(record, key)
        if value not in (None, "", [], {}):
            return value
    return default



def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))



def _extract_images(record: dict) -> list[str]:
    photos = _coalesce(record, ["photos", "images", "photoLinks"], []) or []
    urls: list[str] = []
    if isinstance(photos, list):
        for item in photos:
            if isinstance(item, str) and item.startswith("http"):
                urls.append(item)
            elif isinstance(item, dict):
                maybe_url = _coalesce(item, ["href", "url", "link", "src"])
                if isinstance(maybe_url, str) and maybe_url.startswith("http"):
                    urls.append(maybe_url)
    if not urls:
        for key in ["photo", "image", "primaryPhotoUrl"]:
            value = record.get(key)
            if isinstance(value, str) and value.startswith("http"):
                urls.append(value)
    deduped = []
    seen = set()
    for url in urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped[:10]



def normalize_rentcast_listing(record: dict, mode: str) -> dict:
    address = _coalesce(
        record,
        ["formattedAddress", "address", "addressLine1", "streetAddress", "fullAddress"],
        "Address unavailable",
    )
    city = _coalesce(record, ["city", "addressCity"])
    state = _coalesce(record, ["state", "stateCode", "addressState"])
    zip_code = _coalesce(record, ["zipCode", "postalCode"])
    listing_id = str(_coalesce(record, ["id", "listingId", "propertyId", "mlsNumber"], "unknown"))

    title = _coalesce(record, ["title", "formattedAddress", "addressLine1"])
    if not title:
        title = f"{record.get('bedrooms', '?')} bd {record.get('propertyType', 'Home')}"

    return {
        "id": f"rentcast:{listing_id}",
        "source": "rentcast",
        "mode": mode,
        "title": title,
        "address": address,
        "city": city,
        "state": state,
        "zip_code": zip_code,
        "latitude": _coalesce(record, ["latitude", "lat"]),
        "longitude": _coalesce(record, ["longitude", "lon", "lng"]),
        "price": _coalesce(record, ["price", "rent", "listPrice"]),
        "bedrooms": _coalesce(record, ["bedrooms"]),
        "bathrooms": _coalesce(record, ["bathrooms"]),
        "square_feet": _coalesce(record, ["squareFootage", "livingArea", "sqft"]),
        "property_type": _coalesce(record, ["propertyType"]),
        "description": _coalesce(record, ["description", "summary"]),
        "images": _extract_images(record),
        "source_url": _coalesce(record, ["listingUrl", "propertyUrl", "url", "mlsUrl"]),
        "provider": "RentCast",
        "contact_name": _coalesce(record, ["listingAgent.name", "listingAgentName"]),
        "contact_email": _coalesce(record, ["listingAgent.email"]),
        "contact_phone": _coalesce(record, ["listingAgent.phone"]),
        "days_on_market": _coalesce(record, ["daysOnMarket"]),
        "listed_date": _coalesce(record, ["listedDate", "createdDate"]),
        "listing_type": _coalesce(record, ["listingType"]),
        "hoa_fee": _coalesce(record, ["hoa.fee"]),
        "mls_name": _coalesce(record, ["mlsName"]),
        "mls_number": _coalesce(record, ["mlsNumber"]),
        "market": {},
        "neighborhood": {},
        "commute": {},
        "scores": {},
        "reasons": [],
        "raw": record,
    }


async def geocode_query(query: str) -> dict | None:
    query = query.strip()
    if not query:
        return None
    cache_key = f"geo::{query.lower()}"
    cached = geo_cache.get(cache_key)
    if cached:
        return cached

    params = {
        "q": query,
        "format": "jsonv2",
        "addressdetails": 1,
        "limit": 1,
    }

    async with await _async_client() as client:
        response = await client.get(f"{settings.nominatim_base_url}/search", params=params)
        if response.status_code != 200:
            return None
        payload = response.json()
        if not payload:
            return None
        item = payload[0]
        address = item.get("address", {})
        city = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("municipality")
            or address.get("county")
        )
        data = {
            "display_name": item.get("display_name"),
            "latitude": float(item["lat"]),
            "longitude": float(item["lon"]),
            "city": city,
            "state": address.get("state"),
            "state_code": address.get("ISO3166-2-lvl4", "").split("-")[-1] if address.get("ISO3166-2-lvl4") else None,
            "zip_code": address.get("postcode"),
            "country": address.get("country"),
            "address": address,
        }
        geo_cache.set(cache_key, data)
        return data



def _range_param(min_value, max_value):
    if min_value is None and max_value is None:
        return None
    if min_value is not None and max_value is not None:
        return f"{min_value}:{max_value}"
    if min_value is not None:
        return f"{min_value}:"
    return str(max_value)


def _base_rentcast_params(search: dict) -> dict:
    filters = search.get("filters") or {}
    limit = max(int(search.get("limit") or 12) * 3, 24)
    params: dict[str, Any] = {"limit": min(limit, 500), "status": "Active", "includeTotalCount": True}

    prop_types = filters.get("property_types") or []
    if prop_types:
        normalized_types = [item.replace("Multi Family", "Multi-Family") for item in prop_types]
        params["propertyType"] = "|".join(normalized_types)

    price = _range_param(filters.get("min_price"), filters.get("max_price"))
    bedrooms = _range_param(filters.get("min_bedrooms"), filters.get("max_bedrooms"))
    bathrooms = _range_param(filters.get("min_bathrooms"), None)
    sqft = _range_param(filters.get("min_square_feet"), filters.get("max_square_feet"))
    if price:
        params["price"] = price
    if bedrooms:
        params["bedrooms"] = bedrooms
    if bathrooms:
        params["bathrooms"] = bathrooms
    if sqft:
        params["squareFootage"] = sqft
    if filters.get("listing_age_days"):
        params["daysOld"] = str(filters["listing_age_days"])
    return params


def _city_state_params(search: dict) -> dict | None:
    location = search.get("location") or {}
    if not (location.get("city") and location.get("state")):
        return None
    params = _base_rentcast_params(search)
    params["city"] = location["city"]
    params["state"] = normalize_state_code(location.get("state")) or location["state"]
    return params


def _exact_rentcast_params(search: dict) -> dict:
    location = search.get("location") or {}
    params = _base_rentcast_params(search)
    if location.get("zip_code"):
        params["zipCode"] = location["zip_code"]
    elif location.get("city") and location.get("state"):
        params["city"] = location["city"]
        params["state"] = normalize_state_code(location.get("state")) or location["state"]
    elif location.get("address"):
        params["address"] = location["address"]
    return params


async def _radius_rentcast_params(search: dict) -> dict | None:
    radius = search.get("radius_miles") or 25
    try:
        radius = max(1, min(100, int(radius)))
    except Exception:
        radius = 25
    location = search.get("location") or {}
    query = location.get("address") or " ".join(
        item for item in [location.get("city"), location.get("state"), location.get("zip_code")] if item
    )
    if not query:
        return None
    center = await geocode_query(query)
    if not center:
        return None
    params = _base_rentcast_params(search)
    params["latitude"] = round(center["latitude"], 6)
    params["longitude"] = round(center["longitude"], 6)
    params["radius"] = radius
    return params


async def search_rentcast(search: dict) -> tuple[list[dict], list[str]]:
    messages: list[str] = []
    if not settings.rentcast_api_key:
        return [], messages

    endpoint = "/listings/rental/long-term" if search["mode"] == "rental" else "/listings/sale"
    strategies: list[tuple[str, dict]] = [("exact", _exact_rentcast_params(search))]
    city_params = _city_state_params(search)
    if city_params and city_params != strategies[0][1]:
        strategies.append(("city", city_params))
    radius_params = await _radius_rentcast_params(search)
    if radius_params:
        strategies.append(("radius", radius_params))

    seen_param_signatures = set()
    output: list[dict] = []
    seen_ids: set[str] = set()

    async with await _async_client() as client:
        for label, params in strategies:
            signature = tuple(sorted((key, str(value)) for key, value in params.items()))
            if signature in seen_param_signatures:
                continue
            seen_param_signatures.add(signature)
            try:
                response = await client.get(
                    f"{settings.rentcast_base_url}{endpoint}",
                    params=params,
                    headers={"X-Api-Key": settings.rentcast_api_key, "User-Agent": settings.user_agent},
                )
            except Exception as exc:
                messages.append(f"RentCast {label} request failed before response: {exc}")
                continue

            if response.status_code != 200:
                text = response.text[:220].replace("\n", " ")
                messages.append(f"RentCast {label} request returned HTTP {response.status_code}: {text}")
                continue

            data = response.json()
            records = data if isinstance(data, list) else data.get("results", data.get("listings", []))
            if not records:
                messages.append(f"RentCast {label} request returned zero records for the current filters.")
                continue

            for record in records:
                listing = normalize_rentcast_listing(record, search["mode"])
                listing_id = listing.get("id") or str(record)
                if listing_id in seen_ids:
                    continue
                seen_ids.add(listing_id)
                output.append(listing)

            if len(output) >= int(search.get("limit") or 12):
                break

    if output:
        messages.append(f"RentCast returned {len(output)} unique live listing records before enrichment.")
    return output, messages

async def fetch_market_snapshot(zip_code: str | None, mode: str) -> dict:
    if not zip_code or not settings.rentcast_api_key:
        return {}
    cache_key = f"market::{mode}::{zip_code}"
    cached = market_cache.get(cache_key)
    if cached:
        return cached

    params = {"zipCode": zip_code, "dataType": "Rental" if mode == "rental" else "Sale"}
    async with await _async_client() as client:
        response = await client.get(
            f"{settings.rentcast_base_url}/markets",
            params=params,
            headers={"X-Api-Key": settings.rentcast_api_key, "User-Agent": settings.user_agent},
        )
        if response.status_code != 200:
            return {}
        data = response.json()
        if not isinstance(data, dict):
            return {}
        market_data = data.get("rentalData") if mode == "rental" else data.get("saleData")
        market_data = market_data or {}
        summary = {
            "last_updated": data.get("lastUpdatedDate"),
            "median_price": _coalesce(market_data, ["medianPrice", "medianRent"]),
            "average_price": _coalesce(market_data, ["averagePrice", "averageRent"]),
            "median_price_per_sqft": _coalesce(market_data, ["medianPricePerSquareFoot", "medianRentPerSquareFoot"]),
            "days_on_market": _coalesce(market_data, ["medianDaysOnMarket", "averageDaysOnMarket"]),
            "new_listings": _coalesce(market_data, ["newListings"]),
            "source": "RentCast market stats",
        }
        market_cache.set(cache_key, summary)
        return summary


async def _greatschools_nearby(lat: float, lon: float) -> dict | None:
    if not settings.greatschools_api_key:
        return None
    cache_key = f"schools::gs::{round(lat, 3)}::{round(lon, 3)}"
    cached = school_cache.get(cache_key)
    if cached:
        return cached

    params = {"lat": lat, "lon": lon, "limit": 5, "distance": 5}
    async with await _async_client() as client:
        response = await client.get(
            f"{settings.greatschools_base_url}/nearby-schools",
            params=params,
            headers={
                "x-api-key": settings.greatschools_api_key,
                "Accept": "application/json, application/xml;q=0.9, */*;q=0.8",
                "User-Agent": settings.user_agent,
            },
        )
        if response.status_code != 200:
            return None
        schools = []
        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            payload = response.json()
            schools = payload.get("schools", []) if isinstance(payload, dict) else []
        else:
            root = ET.fromstring(response.text)
            schools_nodes = root.findall(".//school")
            for node in schools_nodes:
                school = {child.tag: child.text for child in node}
                schools.append(school)
        ratings = []
        distances = []
        for school in schools:
            try:
                if school.get("rating") is not None:
                    ratings.append(float(school.get("rating")))
            except (TypeError, ValueError):
                pass
            try:
                if school.get("distance") is not None:
                    distances.append(float(school.get("distance")))
            except (TypeError, ValueError):
                pass
        result = {
            "source": "GreatSchools NearbySchools API",
            "nearby_school_count": len(schools),
            "avg_rating_10": round(sum(ratings) / len(ratings), 2) if ratings else None,
            "nearest_school_miles": round(min(distances), 2) if distances else None,
            "schools": [
                {
                    "name": school.get("name"),
                    "type": school.get("type"),
                    "distance": school.get("distance"),
                    "rating": school.get("rating"),
                    "overview_url": school.get("overview-url") or school.get("overview_url"),
                }
                for school in schools[:5]
            ],
        }
        school_cache.set(cache_key, result)
        return result



def _distance_to_element_miles(origin_lat: float, origin_lon: float, element: dict) -> float | None:
    lat = element.get("lat") or _get_value(element, "center.lat")
    lon = element.get("lon") or _get_value(element, "center.lon")
    if lat is None or lon is None:
        return None
    try:
        return round(haversine_miles(origin_lat, origin_lon, float(lat), float(lon)), 2)
    except Exception:
        return None



def _update_nearest(store: dict, category: str, miles: float | None):
    if miles is None:
        return
    key = f"nearest_{category}_miles"
    current = store.get(key)
    if current is None or miles < current:
        store[key] = miles



def _parse_overpass_items(payload: dict, origin_lat: float, origin_lon: float) -> dict:
    counts = {
        "food": 0,
        "shops": 0,
        "grocery": 0,
        "parks": 0,
        "transit": 0,
        "schools": 0,
        "total": 0,
        "nearest_food_miles": None,
        "nearest_shop_miles": None,
        "nearest_grocery_miles": None,
        "nearest_park_miles": None,
        "nearest_transit_miles": None,
        "nearest_school_miles": None,
    }
    for element in payload.get("elements", []):
        tags = element.get("tags", {})
        if not tags:
            continue
        counts["total"] += 1
        amenity = tags.get("amenity")
        shop = tags.get("shop")
        leisure = tags.get("leisure")
        railway = tags.get("railway")
        highway = tags.get("highway")
        public_transport = tags.get("public_transport")
        miles = _distance_to_element_miles(origin_lat, origin_lon, element)

        if amenity in {"restaurant", "cafe", "bar", "fast_food", "pub"}:
            counts["food"] += 1
            _update_nearest(counts, "food", miles)
        if shop in {"supermarket", "convenience", "mall", "bakery", "department_store", "greengrocer"}:
            counts["shops"] += 1
            _update_nearest(counts, "shop", miles)
        if shop in {"supermarket", "convenience", "greengrocer"}:
            counts["grocery"] += 1
            _update_nearest(counts, "grocery", miles)
        if leisure in {"park", "nature_reserve", "playground", "fitness_centre", "garden"}:
            counts["parks"] += 1
            _update_nearest(counts, "park", miles)
        if public_transport or highway == "bus_stop" or railway in {"station", "halt", "tram_stop", "subway_entrance"}:
            counts["transit"] += 1
            _update_nearest(counts, "transit", miles)
        if amenity in {"school", "kindergarten", "college", "university"}:
            counts["schools"] += 1
            _update_nearest(counts, "school", miles)
    score = min(100, counts["food"] * 6 + counts["shops"] * 8 + counts["parks"] * 10 + counts["transit"] * 12)
    counts["walkability_score"] = score
    counts["source"] = "OpenStreetMap Overpass"
    return counts


async def walkability_snapshot(lat: float, lon: float) -> dict:
    cache_key = f"walk::{round(lat, 3)}::{round(lon, 3)}"
    cached = walk_cache.get(cache_key)
    if cached:
        return cached

    query = f"""
    [out:json][timeout:25];
    (
      node(around:1000,{lat},{lon})[amenity];
      way(around:1000,{lat},{lon})[amenity];
      node(around:1000,{lat},{lon})[shop];
      way(around:1000,{lat},{lon})[shop];
      node(around:1000,{lat},{lon})[leisure];
      way(around:1000,{lat},{lon})[leisure];
      node(around:1000,{lat},{lon})[public_transport];
      node(around:1000,{lat},{lon})[highway=bus_stop];
      node(around:1000,{lat},{lon})[railway];
    );
    out tags center;
    """

    async with await _async_client() as client:
        response = await client.post(
            settings.overpass_base_url,
            content=query,
            headers={"Content-Type": "text/plain", "User-Agent": settings.user_agent},
        )
        if response.status_code != 200:
            return {"source": "OpenStreetMap Overpass", "walkability_score": None}
        payload = response.json()
        result = _parse_overpass_items(payload, lat, lon)
        walk_cache.set(cache_key, result)
        return result


async def school_snapshot(lat: float, lon: float) -> dict:
    gs = await _greatschools_nearby(lat, lon)
    if gs:
        return gs

    walk = await walkability_snapshot(lat, lon)
    school_access = min(100, (walk.get("schools") or 0) * 18)
    return {
        "source": "OSM school proximity fallback",
        "nearby_school_count": walk.get("schools"),
        "avg_rating_10": None,
        "school_access_score": school_access,
        "nearest_school_miles": walk.get("nearest_school_miles"),
        "schools": [],
    }


async def crime_snapshot(lat: float, lon: float) -> dict:
    if not settings.crimeometer_api_key:
        return {"source": "Crimeometer not configured", "safety_score": None, "incidents_count": None}
    cache_key = f"crime::{round(lat, 3)}::{round(lon, 3)}"
    cached = crime_cache.get(cache_key)
    if cached:
        return cached

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=180)
    params = {
        "lat": lat,
        "lon": lon,
        "datetime_ini": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "datetime_end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "distance": 2,
    }
    async with await _async_client() as client:
        response = await client.get(
            f"{settings.crimeometer_base_url}/crime-incidents",
            params=params,
            headers={"x-api-key": settings.crimeometer_api_key, "User-Agent": settings.user_agent},
        )
        if response.status_code != 200:
            return {"source": "Crimeometer", "safety_score": None, "incidents_count": None}
        payload = response.json()
        count = payload.get("incidents_count")
        try:
            count = int(count)
        except (TypeError, ValueError):
            count = None
        safety_score = None if count is None else max(0, min(100, 100 - count * 3))
        if safety_score is None:
            label = "unknown"
        elif safety_score >= 75:
            label = "low reported incidents"
        elif safety_score >= 50:
            label = "moderate reported incidents"
        else:
            label = "higher reported incidents"
        result = {
            "source": "Crimeometer Crime Data API",
            "incidents_count": count,
            "safety_score": safety_score,
            "safety_label": label,
            "window_days": 180,
            "radius_miles": 2,
        }
        crime_cache.set(cache_key, result)
        return result


async def commute_snapshot(
    origin_lat: float,
    origin_lon: float,
    destination_text: str,
    mode: str,
) -> dict:
    cache_key = f"commute::{round(origin_lat, 3)}::{round(origin_lon, 3)}::{destination_text.lower()}::{mode}"
    cached = commute_cache.get(cache_key)
    if cached:
        return cached

    destination = await geocode_query(destination_text)
    if not destination:
        return {"source": "Geocoding failed", "minutes": None, "distance_miles": None}

    if settings.openrouteservice_api_key:
        body = {
            "coordinates": [
                [origin_lon, origin_lat],
                [destination["longitude"], destination["latitude"]],
            ]
        }
        async with await _async_client() as client:
            response = await client.post(
                f"{settings.ors_base_url}/v2/directions/{mode}/json",
                json=body,
                headers={
                    "Authorization": settings.openrouteservice_api_key,
                    "Content-Type": "application/json",
                    "User-Agent": settings.user_agent,
                },
            )
            if response.status_code == 200:
                payload = response.json()
                route = (payload.get("routes") or [{}])[0]
                summary = route.get("summary", {})
                minutes = round((summary.get("duration") or 0) / 60, 1) if summary.get("duration") is not None else None
                miles = round((summary.get("distance") or 0) / 1609.344, 1) if summary.get("distance") is not None else None
                result = {
                    "source": "openrouteservice Directions API",
                    "minutes": minutes,
                    "distance_miles": miles,
                    "mode": mode,
                    "destination_display_name": destination.get("display_name"),
                }
                commute_cache.set(cache_key, result)
                return result

    miles = haversine_miles(origin_lat, origin_lon, destination["latitude"], destination["longitude"])
    avg_speed = {"driving-car": 30, "cycling-regular": 12, "foot-walking": 3}.get(mode, 30)
    minutes = round((miles / avg_speed) * 60, 1) if avg_speed else None
    result = {
        "source": "Estimated from geodesic distance",
        "minutes": minutes,
        "distance_miles": round(miles, 1),
        "mode": mode,
        "destination_display_name": destination.get("display_name"),
    }
    commute_cache.set(cache_key, result)
    return result
