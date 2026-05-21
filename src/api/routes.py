"""
API route definitions.

Routes are kept thin: they validate input via Pydantic, call into
src/models/ensemble.predict(), and serialize the result. All business
logic lives in src/. This separation makes the routes easy to test
and the business logic reusable (e.g. for batch jobs or CLI tools).
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from config.settings import settings
from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictionResponse,
)
from src.features.feature_pipeline import FEATURE_NAMES, N_FEATURES
from src.models.ensemble import predict
from src.utils.exceptions import (
    FeatureExtractionError,
    URLNormalizationError,
    URLValidationError,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


router = APIRouter()


def _result_to_response(result) -> PredictionResponse:
    """Convert a dataclass PredictionResult into a Pydantic response."""
    return PredictionResponse(
        url=result.url,
        prediction=result.prediction,
        phishing_probability=result.phishing_probability,
        legitimate_probability=result.legitimate_probability,
        confidence_score=result.confidence_score,
        risk_level=result.risk_level,
        explanations=list(result.explanations),
        triggered_rules=list(result.triggered_rules),
        ml_probability=result.ml_probability,
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify a single URL",
    tags=["predict"],
)
async def predict_single(req: PredictRequest, request: Request) -> PredictionResponse:
    """Run the hybrid ML + rules + SHAP pipeline on one URL."""
    model = request.app.state.model
    shap_explainer = request.app.state.shap

    try:
        result = predict(req.url, model, shap_explainer)
    except URLValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"URL validation failed: {exc}",
        )
    except URLNormalizationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"URL could not be normalized: {exc}",
        )
    except FeatureExtractionError as exc:
        logger.error("feature_extraction_failed", url=req.url, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal feature extraction error.",
        )
    except Exception as exc:
        logger.exception("predict_unexpected_error", url=req.url)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal prediction error: {exc}",
        )

    return _result_to_response(result)


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Classify multiple URLs in one request",
    tags=["predict"],
)
async def predict_batch(
    req: BatchPredictRequest, request: Request
) -> BatchPredictionResponse:
    """Bulk version of /predict. Per-URL failures are collected in
    `errors` rather than failing the whole batch."""
    model = request.app.state.model
    shap_explainer = request.app.state.shap

    results: list[PredictionResponse] = []
    errors: list[dict] = []

    for idx, url in enumerate(req.urls):
        try:
            result = predict(url, model, shap_explainer)
            results.append(_result_to_response(result))
        except (URLValidationError, URLNormalizationError) as exc:
            errors.append({"index": idx, "url": url, "error": str(exc)})
        except Exception as exc:
            logger.exception("batch_predict_unexpected_error", url=url)
            errors.append({"index": idx, "url": url, "error": "internal_error"})

    return BatchPredictionResponse(results=results, errors=errors)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness/readiness probe",
    tags=["meta"],
)
async def health(request: Request) -> HealthResponse:
    """Used by load balancers and orchestrators to verify the service
    is alive and the model is loaded."""
    model_loaded = request.app.state.model is not None
    shap_loaded = request.app.state.shap is not None
    return HealthResponse(
        status="ok" if (model_loaded and shap_loaded) else "degraded",
        model_loaded=model_loaded,
        shap_loaded=shap_loaded,
    )


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Model and API metadata",
    tags=["meta"],
)
async def info(request: Request) -> ModelInfoResponse:
    """Returns the API version, the trained model's test metrics, and
    the feature schema. Useful for debugging and audit trails."""
    model = request.app.state.model
    return ModelInfoResponse(
        api_version=settings.api_version,
        model_version="lgbm_v1",
        n_features=N_FEATURES,
        test_metrics=model.metrics if model else {},
        feature_names=list(FEATURE_NAMES),
    )