"""
FastAPI application entry point.

The application loads the LightGBM model and SHAP explainer ONCE at
startup (via the lifespan context manager) and reuses them for every
request. This is critical for performance: cold-loading the model on
each request would add ~500ms per call.

CORS is permissive by default because this API is consumed by a
mobile (Flutter) client; restrict via settings.cors_origins in
production deployments.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.logging_config import configure_logging
from config.settings import settings
from src.api.routes import router
from src.explainability.shap_explainer import ShapExplainer
from src.models.lgbm_model import load_model
from src.utils.logger import get_logger


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup; release at shutdown."""
    configure_logging()
    logger.info("api_starting", version=settings.api_version)

    model_path: Path = settings.models_dir / "lgbm_v1.joblib"
    if not model_path.exists():
        logger.error("model_not_found", path=str(model_path))
        raise RuntimeError(
            f"Model artifact not found at {model_path}. "
            "Train the model first: python scripts/train_lgbm.py"
        )

    app.state.model = load_model(model_path)
    app.state.shap = ShapExplainer(app.state.model)
    logger.info("api_ready")

    try:
        yield
    finally:
        logger.info("api_shutting_down")
        app.state.model = None
        app.state.shap = None


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=(
            "Hybrid phishing URL detection API. Combines a LightGBM "
            "classifier on 65 hand-engineered features with a "
            "deterministic rule engine targeting known phishing patterns "
            "(typosquats, brand stuffing, IP hosts, abusive TLDs, "
            "homograph attacks, embedded credentials). SHAP attributions "
            "provide per-feature explanations for ML-driven decisions."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(router)

    return app


app = create_app()