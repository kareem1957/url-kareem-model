"""
Pydantic schemas for the API.

These are the contract between the API and its clients (Flutter app,
curl users, third-party integrations). Pydantic validates every
incoming request at the network boundary and produces structured
error responses for malformed input.

Schema changes are breaking changes for clients. Bump the API version
in config/settings.py whenever you alter a schema's shape.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Single-URL prediction request."""

    url: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="URL to classify. Scheme optional; http:// prepended if missing.",
        examples=["https://paypal-secure.evil.com/login"],
    )


class BatchPredictRequest(BaseModel):
    """Batch prediction request. Cap at 100 to prevent abuse."""

    urls: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of URLs to classify (max 100 per request).",
    )


class PredictionResponse(BaseModel):
    """Single-URL prediction response."""

    url: str = Field(..., description="Echo of the input URL.")
    prediction: Literal["phishing", "legitimate"] = Field(
        ..., description="Final classification."
    )
    phishing_probability: float = Field(
        ..., ge=0.0, le=100.0,
        description="Probability the URL is phishing, in percent (0-100).",
    )
    legitimate_probability: float = Field(
        ..., ge=0.0, le=100.0,
        description="Probability the URL is legitimate, in percent (0-100).",
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=100.0,
        description=(
            "Confidence in the prediction, in percent. 100 means the model "
            "is very sure of its answer (regardless of direction); 0 means "
            "the model is on the decision boundary."
        ),
    )
    risk_level: Literal["low", "medium", "high", "critical"] = Field(
        ..., description="Coarse risk band derived from phishing probability."
    )
    explanations: list[str] = Field(
        default_factory=list,
        description=(
            "Human-readable reasons the URL was classified this way. "
            "Empty for clearly legitimate URLs."
        ),
    )
    triggered_rules: list[str] = Field(
        default_factory=list,
        description="IDs of deterministic rules that fired for this URL.",
    )
    ml_probability: float = Field(
        ..., ge=0.0, le=100.0,
        description="Raw ML model probability (before rule combination), in percent.",
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction response. Failed URLs are reported in errors."""

    results: list[PredictionResponse] = Field(
        default_factory=list,
        description="Successful predictions, in the same order as the input.",
    )
    errors: list[dict] = Field(
        default_factory=list,
        description="Errors keyed by input index, e.g. {'index': 3, 'error': 'malformed url'}.",
    )


class HealthResponse(BaseModel):
    """Liveness/readiness probe response."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    shap_loaded: bool


class ModelInfoResponse(BaseModel):
    """Public model metadata."""

    api_version: str
    model_version: str
    n_features: int
    test_metrics: dict
    feature_names: list[str]