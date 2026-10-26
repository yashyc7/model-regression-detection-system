import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# src/regression_guard/config.py -> parents[2] is the repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Reads the .env file and puts every line into os.environ.
# If a variable is already set in the real environment (for example in
# GitHub Actions or Docker), that real value wins over the .env file.
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """All config for the app, read once from environment variables."""

    openai_api_key: str
    openai_base_url: str | None
    llm_model: str
    judge_model: str
    slack_webhook_url: str | None


@lru_cache
def get_settings() -> Settings:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Copy .env.example to .env and add your key."
        )

    # `or None` / `or "default"` also handles the case where the line in
    # .env exists but is left blank (e.g. OPENAI_BASE_URL=)
    return Settings(
        openai_api_key=api_key,
        openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        llm_model=os.getenv("LLM_MODEL") or "gpt-4o-mini",
        judge_model=os.getenv("JUDGE_MODEL") or "gpt-4o-mini",
        slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL") or None,
    )
