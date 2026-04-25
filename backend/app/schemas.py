from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=120)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: EmailStr
    role: str
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class SearchLocation(BaseModel):
    address: str | None = None
    city: str | None = None
    state: str | None = None
    zip_code: str | None = None


class SearchFilters(BaseModel):
    min_price: int | None = None
    max_price: int | None = None
    min_bedrooms: float | None = None
    max_bedrooms: float | None = None
    min_bathrooms: float | None = None
    property_types: list[str] = Field(default_factory=list)
    min_square_feet: int | None = None
    max_square_feet: int | None = None
    listing_age_days: int | None = None


class SearchConstraints(BaseModel):
    low_crime: bool = False
    good_schools: bool = False
    high_walkability: bool = False
    parks_nearby: bool = False
    transit_nearby: bool = False
    food_scene: bool = False
    grocery_nearby: bool = False
    max_commute_minutes: int | None = None
    commute_destination: str | None = None
    commute_mode: Literal["driving-car", "cycling-regular", "foot-walking"] = "driving-car"


class SearchRequest(BaseModel):
    mode: Literal["rental", "sale"] = "rental"
    location: SearchLocation
    filters: SearchFilters = Field(default_factory=SearchFilters)
    constraints: SearchConstraints = Field(default_factory=SearchConstraints)
    semantic_query: str | None = Field(default=None, max_length=500)
    radius_miles: int | None = Field(default=25, ge=1, le=100)
    sort_by: Literal["fit", "price_asc", "price_desc", "commute", "crime", "schools", "walkability", "semantic"] = "fit"
    limit: int = Field(default=12, ge=1, le=30)


class ListingResult(BaseModel):
    id: str
    source: str
    mode: str
    title: str | None = None
    address: str
    city: str | None = None
    state: str | None = None
    zip_code: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    distance_from_search_miles: float | None = None
    price: float | None = None
    bedrooms: float | None = None
    bathrooms: float | None = None
    square_feet: int | None = None
    property_type: str | None = None
    description: str | None = None
    images: list[str] = Field(default_factory=list)
    source_url: str | None = None
    provider: str | None = None
    contact_name: str | None = None
    contact_email: str | None = None
    contact_phone: str | None = None
    days_on_market: int | None = None
    listed_date: str | None = None
    listing_type: str | None = None
    hoa_fee: float | None = None
    mls_name: str | None = None
    mls_number: str | None = None
    market: dict = Field(default_factory=dict)
    neighborhood: dict = Field(default_factory=dict)
    commute: dict = Field(default_factory=dict)
    scores: dict = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    raw: dict | None = None


class SearchResponse(BaseModel):
    results: list[ListingResult]
    provider_status: dict
    warnings: list[str]
    summary: dict = Field(default_factory=dict)
    cache: dict = Field(default_factory=dict)
    generated_at: datetime


class FavoriteCreate(BaseModel):
    listing_id: str
    source: str
    title: str | None = None
    payload: dict = Field(default_factory=dict)


class FavoriteOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    listing_id: str
    source: str
    title: str | None = None
    payload: dict
    created_at: datetime


class AdvertisedPlaceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    mode: str
    title: str
    address_line1: str
    city: str
    state: str
    zip_code: str
    latitude: float | None = None
    longitude: float | None = None
    distance_from_search_miles: float | None = None
    price: float
    bedrooms: float | None = None
    bathrooms: float | None = None
    square_feet: int | None = None
    property_type: str | None = None
    description: str
    images: list[str]
    contact_name: str | None = None
    contact_email: str | None = None
    status: str
    validation_score: float | None = None
    validation_flags: list[str] = Field(default_factory=list)
    moderation_notes: str | None = None
    created_at: datetime
    updated_at: datetime | None = None


class ModerationAction(BaseModel):
    notes: str | None = None


class AgentRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    search: SearchRequest | None = None
    listings: list[ListingResult] | None = None


class AgentResponse(BaseModel):
    answer: str
    highlights: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    cache_status: str = "miss"


class AgentRunRequest(BaseModel):
    goal: str = Field(min_length=5, max_length=4000)
    search: SearchRequest | None = None
    listings: list[ListingResult] | None = None
    autonomy_level: Literal["guided", "auto"] = "guided"
    max_iterations: int = Field(default=6, ge=3, le=10)
    use_cache: bool = True


class AgentTraceEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    sequence: int
    event_type: str
    agent_name: str
    title: str
    message: str
    status: str = "ok"
    latency_ms: int | None = None
    payload: dict = Field(default_factory=dict)
    created_at: datetime | None = None


class AgentRecommendation(BaseModel):
    listing_id: str
    title: str | None = None
    address: str | None = None
    price: float | None = None
    fit_score: float | None = None
    agent_score: float | None = None
    confidence: float = 0.0
    decision: str = "consider"
    tradeoffs: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    source: str | None = None
    image_count: int | None = None


class AgentRunResponse(BaseModel):
    run_id: str
    status: str
    answer: str
    plan: list[dict] = Field(default_factory=list)
    recommendations: list[AgentRecommendation] = Field(default_factory=list)
    trace: list[AgentTraceEventOut] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    cache_status: str = "miss"
    self_optimization: dict = Field(default_factory=dict)


class AgentFeedbackRequest(BaseModel):
    rating: int = Field(ge=-1, le=1)
    label: str | None = Field(default=None, max_length=80)
    comment: str | None = Field(default=None, max_length=1000)
