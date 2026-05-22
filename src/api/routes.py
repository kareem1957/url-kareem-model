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
from src.api import urlscan_integration
from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictionResponse,
    ScreenshotResponse,
)
from src.api.urlscan_integration import UrlscanError
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


def _result_to_response(result, scan_id: str | None = None, cached: bool = False) -> PredictionResponse:
    """Convert a dataclass PredictionResult into a Pydantic response.

    The optional scan_id/cached parameters are populated when a
    urlscan.io screenshot scan was triggered for this prediction.
    """
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
        screenshot_scan_id=scan_id,
        screenshot_cached=cached,
    )


def _maybe_request_screenshot(url: str, prediction: str) -> tuple[str | None, bool]:
    """Submit URL to urlscan.io for screenshot rendering when it's a phishing
    verdict and the integration is configured. Returns (scan_id, cached).

    Returns (None, False) on legitimate verdicts, when urlscan is disabled,
    or when submission fails. We NEVER want a urlscan hiccup to block the
    user from getting their phishing verdict — the screenshot is enrichment,
    not critical-path.
    """
    if prediction != "phishing":
        return None, False
    if not urlscan_integration.is_enabled():
        return None, False

    try:
        submission = urlscan_integration.submit_scan(url)
        logger.info(
            "screenshot_requested",
            url=url,
            scan_id=submission.scan_id,
            cached=submission.cached,
        )
        return submission.scan_id, submission.cached
    except UrlscanError as exc:
        logger.warning(
            "screenshot_submit_failed_continuing_without",
            url=url,
            error=str(exc),
        )
        return None, False


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

    # Trigger screenshot rendering on phishing verdicts (best-effort)
    scan_id, cached = _maybe_request_screenshot(req.url, result.prediction)

    return _result_to_response(result, scan_id=scan_id, cached=cached)


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
    `errors` rather than failing the whole batch.

    Note: batch predictions deliberately do NOT trigger screenshot scans —
    batch consumers are typically background jobs, and submitting 100 scans
    to urlscan in one go would burn through the quota quickly. Use the
    single /predict endpoint when you want screenshots.
    """
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
    "/screenshot/{scan_id}",
    response_model=ScreenshotResponse,
    summary="Poll for screenshot completion",
    tags=["predict"],
)
async def screenshot(scan_id: str) -> ScreenshotResponse:
    """Returns the urlscan.io scan status for a previously-submitted phishing
    URL. The Flutter client polls this every 5 seconds (after an initial
    15-second wait) until status='ready', then renders the screenshot_url
    directly via Image.network().

    The screenshot URL itself is publicly accessible on urlscan's CDN —
    the client does not need to proxy through our API or send any auth.
    """
    if not urlscan_integration.is_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Screenshot feature is not configured on this server.",
        )

    result = urlscan_integration.get_scan_result(scan_id)
    return ScreenshotResponse(
        scan_id=scan_id,
        status=result.status,
        screenshot_url=result.screenshot_url,
        urlscan_verdict_malicious=result.urlscan_verdict_malicious,
        urlscan_score=result.urlscan_score,
        detected_brands=list(result.detected_brands),
        report_url=result.report_url,
        error=result.error,
    )


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