from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RelocateIQ Agentic Home Search"
    app_env: str = "development"
    api_prefix: str = "/api"

    secret_key: str = ""
    access_token_expires_minutes: int = 60 * 24 * 7
    database_url: str = "sqlite:///./app.db"

    cors_origins: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )
    frontend_origin: str = "http://localhost:5173"
    user_agent: str = "RelocateIQ/2.0 (local-dev)"

    admin_name: str = ""
    admin_email: str = ""
    admin_password: str = ""

    rentcast_api_key: str | None = None
    openrouteservice_api_key: str | None = None
    greatschools_api_key: str | None = None
    crimeometer_api_key: str | None = None

    openai_api_key: str | None = None
    openai_model: str = "gpt-5.4-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int | None = None

    redis_url: str | None = None
    demo_seed_listings_enabled: bool = True
    demo_seed_when_live_empty: bool = True
    demo_seed_limit: int = 24
    search_cache_ttl_seconds: int = 60 * 15
    semantic_cache_ttl_seconds: int = 60 * 60 * 6
    semantic_similarity_threshold: float = 0.86

    nominatim_base_url: str = "https://nominatim.openstreetmap.org"
    overpass_base_url: str = "https://overpass-api.de/api/interpreter"
    crimeometer_base_url: str = "https://api.crimeometer.com/v2"
    greatschools_base_url: str = "https://gs-api.greatschools.org"
    rentcast_base_url: str = "https://api.rentcast.io/v1"
    ors_base_url: str = "https://api.openrouteservice.org"

    uploads_dir: str = "uploads"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


    @field_validator("openai_embedding_dimensions", mode="before")
    @classmethod
    def empty_string_to_none(cls, value):
        if value in ("", None):
            return None
        return value

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def uploads_path(self) -> Path:
        return Path(self.uploads_dir).resolve()

    def required_runtime_missing(self) -> list[str]:
        missing: list[str] = []
        if not self.secret_key.strip():
            missing.append("SECRET_KEY")
        if not self.admin_name.strip():
            missing.append("ADMIN_NAME")
        if not self.admin_email.strip():
            missing.append("ADMIN_EMAIL")
        if not self.admin_password.strip():
            missing.append("ADMIN_PASSWORD")
        return missing

    def runtime_warnings(self) -> list[str]:
        warnings: list[str] = []
        missing = self.required_runtime_missing()
        if missing:
            warnings.append(
                "Missing required bootstrap settings: " + ", ".join(missing) + ". Run `python backend/scripts/bootstrap.py`."
            )
        if not self.rentcast_api_key:
            if self.demo_seed_listings_enabled:
                warnings.append("RENTCAST_API_KEY is not configured, so live third-party listings are disabled. Seeded demo cache listings are enabled for local presentation.")
            else:
                warnings.append("RENTCAST_API_KEY is not configured, so live third-party listings are disabled.")
        if not self.openrouteservice_api_key:
            warnings.append("OPENROUTESERVICE_API_KEY is not configured, so commute routing will fall back to straight-line estimates.")
        if not self.greatschools_api_key:
            warnings.append("GREATSCHOOLS_API_KEY is not configured, so school quality will fall back to proximity-only scoring.")
        if not self.crimeometer_api_key:
            warnings.append("CRIMEOMETER_API_KEY is not configured, so incident-based safety filtering will be limited.")
        if not self.openai_api_key:
            warnings.append("OPENAI_API_KEY is not configured, so AI reasoning, semantic cache, and semantic ranking will use deterministic fallbacks.")
        return warnings


@lru_cache
def get_settings() -> Settings:
    return Settings()
