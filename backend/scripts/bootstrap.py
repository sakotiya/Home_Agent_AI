from __future__ import annotations

import argparse
import secrets
from pathlib import Path


def random_admin_email() -> str:
    return f"admin+{secrets.token_hex(3)}@example.com"


def strong_secret(length: int = 48) -> str:
    return secrets.token_urlsafe(length)


def parse_env(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def write_env(path: Path, values: dict[str, str]) -> None:
    ordered_keys = [
        "APP_NAME",
        "APP_ENV",
        "API_PREFIX",
        "SECRET_KEY",
        "ACCESS_TOKEN_EXPIRES_MINUTES",
        "DATABASE_URL",
        "CORS_ORIGINS",
        "FRONTEND_ORIGIN",
        "USER_AGENT",
        "ADMIN_NAME",
        "ADMIN_EMAIL",
        "ADMIN_PASSWORD",
        "RENTCAST_API_KEY",
        "OPENROUTESERVICE_API_KEY",
        "GREATSCHOOLS_API_KEY",
        "CRIMEOMETER_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_EMBEDDING_MODEL",
        "OPENAI_EMBEDDING_DIMENSIONS",
        "REDIS_URL",
        "DEMO_SEED_LISTINGS_ENABLED",
        "DEMO_SEED_WHEN_LIVE_EMPTY",
        "DEMO_SEED_LIMIT",
        "SEARCH_CACHE_TTL_SECONDS",
        "SEMANTIC_CACHE_TTL_SECONDS",
        "SEMANTIC_SIMILARITY_THRESHOLD",
        "NOMINATIM_BASE_URL",
        "OVERPASS_BASE_URL",
        "CRIMEOMETER_BASE_URL",
        "GREATSCHOOLS_BASE_URL",
        "RENTCAST_BASE_URL",
        "ORS_BASE_URL",
        "UPLOADS_DIR",
    ]
    lines = []
    for key in ordered_keys:
        lines.append(f"{key}={values.get(key, '')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap local env files for RelocateIQ.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing .env files.")
    parser.add_argument("--admin-email", default="", help="Optional admin email to seed.")
    parser.add_argument("--admin-name", default="Local Admin", help="Optional admin display name.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    backend_dir = root / "backend"
    frontend_dir = root / "frontend"
    backend_env = backend_dir / ".env"
    frontend_env = frontend_dir / ".env"

    if (backend_env.exists() or frontend_env.exists()) and not args.force:
        print("Existing .env file(s) found. Re-run with --force to overwrite.")
        raise SystemExit(0)

    backend_values = parse_env(backend_dir / ".env.example")
    backend_values.update(
        {
            "APP_NAME": backend_values.get("APP_NAME") or "RelocateIQ Agentic Home Search",
            "APP_ENV": backend_values.get("APP_ENV") or "development",
            "API_PREFIX": backend_values.get("API_PREFIX") or "/api",
            "SECRET_KEY": strong_secret(48),
            "ACCESS_TOKEN_EXPIRES_MINUTES": backend_values.get("ACCESS_TOKEN_EXPIRES_MINUTES") or "10080",
            "DATABASE_URL": backend_values.get("DATABASE_URL") or "sqlite:///./app.db",
            "CORS_ORIGINS": backend_values.get("CORS_ORIGINS") or "http://localhost:5173,http://127.0.0.1:5173",
            "FRONTEND_ORIGIN": backend_values.get("FRONTEND_ORIGIN") or "http://localhost:5173",
            "USER_AGENT": backend_values.get("USER_AGENT") or "RelocateIQ/2.0 (local-dev)",
            "ADMIN_NAME": args.admin_name,
            "ADMIN_EMAIL": args.admin_email or random_admin_email(),
            "ADMIN_PASSWORD": strong_secret(18),
            "OPENAI_MODEL": backend_values.get("OPENAI_MODEL") or "gpt-5.4-mini",
            "OPENAI_EMBEDDING_MODEL": backend_values.get("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small",
            "DEMO_SEED_LISTINGS_ENABLED": backend_values.get("DEMO_SEED_LISTINGS_ENABLED") or "true",
            "DEMO_SEED_WHEN_LIVE_EMPTY": backend_values.get("DEMO_SEED_WHEN_LIVE_EMPTY") or "true",
            "DEMO_SEED_LIMIT": backend_values.get("DEMO_SEED_LIMIT") or "24",
            "SEARCH_CACHE_TTL_SECONDS": backend_values.get("SEARCH_CACHE_TTL_SECONDS") or "900",
            "SEMANTIC_CACHE_TTL_SECONDS": backend_values.get("SEMANTIC_CACHE_TTL_SECONDS") or "21600",
            "SEMANTIC_SIMILARITY_THRESHOLD": backend_values.get("SEMANTIC_SIMILARITY_THRESHOLD") or "0.86",
            "UPLOADS_DIR": backend_values.get("UPLOADS_DIR") or "uploads",
        }
    )

    frontend_values = {
        "VITE_API_BASE_URL": "http://localhost:8000",
        "VITE_APP_NAME": "RelocateIQ",
    }

    write_env(backend_env, backend_values)
    frontend_dir.mkdir(parents=True, exist_ok=True)
    frontend_env.write_text(
        "\n".join(f"{key}={value}" for key, value in frontend_values.items()) + "\n",
        encoding="utf-8",
    )
    (backend_dir / backend_values["UPLOADS_DIR"]).mkdir(parents=True, exist_ok=True)

    print("Created:")
    print(f"  {backend_env}")
    print(f"  {frontend_env}")
    print()
    print("Seeded admin login:")
    print(f"  email:    {backend_values['ADMIN_EMAIL']}")
    print(f"  password: {backend_values['ADMIN_PASSWORD']}")
    print()
    print("Next:")
    print("  1) Put provider keys into backend/.env")
    print("  2) Start the backend and frontend")
