"""
Application settings.

All configuration flows through this module. Never hardcode paths,
thresholds, or model parameters elsewhere. Pydantic validates types
at startup, so misconfiguration fails fast instead of silently.
"""
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Paths
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    models_dir: Path = PROJECT_ROOT / "artifacts" / "models"
    encoders_dir: Path = PROJECT_ROOT / "artifacts" / "encoders"

    # Normalization behavior
    resolve_redirects: bool = Field(
        default=False,
        description=(
            "If True, the normalizer performs an HTTP HEAD request to follow "
            "redirect chains. Disabled by default because it adds latency and "
            "network dependency. Enable explicitly when needed."
        ),
    )
    redirect_timeout_seconds: float = 3.0
    max_redirect_hops: int = 5
    max_url_length: int = 2048

    # Model thresholds
    phishing_threshold: float = 0.5
    high_risk_threshold: float = 0.85
    medium_risk_threshold: float = 0.55

    # Allowlist behavior
    allowlist_enabled: bool = True
    allowlist_top_n: int = 100_000

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_title: str = "Phishing URL Detection API"
    api_version: str = "0.1.0"

    # CORS for Flutter client
    cors_origins: list[str] = ["*"]


settings = Settings()