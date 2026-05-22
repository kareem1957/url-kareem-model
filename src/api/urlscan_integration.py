"""
urlscan.io integration for screenshot generation on phishing verdicts.

Architecture:
- submit_scan(url): POST to urlscan.io, returns scan_id (UUID)
- get_scan_result(scan_id): polls urlscan, returns status + screenshot URL
- Module-level cache (24h TTL) so repeat scans of the same URL are instant

Why this design:
- urlscan takes 10-30s to render a page, too long to block /predict
- We return scan_id immediately; Flutter polls /screenshot/{scan_id}
- urlscan.io provides FREE 5000 scans/month for non-commercial use
- Their screenshot URLs are publicly viewable (no auth needed in Flutter)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

from config.settings import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

URLSCAN_BASE = "https://urlscan.io/api/v1"
SUBMIT_ENDPOINT = f"{URLSCAN_BASE}/scan/"
RESULT_ENDPOINT = f"{URLSCAN_BASE}/result"
SCREENSHOT_URL_TEMPLATE = "https://urlscan.io/screenshots/{uuid}.png"

# Simple in-memory cache: url -> (scan_id, expiry_timestamp)
_url_cache: dict[str, tuple[str, float]] = {}
_CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours


@dataclass(frozen=True)
class ScanSubmission:
    """Result of submitting a URL to urlscan.io for scanning."""
    scan_id: str
    submitted_at: float
    cached: bool = False


@dataclass(frozen=True)
class ScanResult:
    """Polled result from urlscan.io. Status indicates current state."""
    status: str  # 'pending' | 'ready' | 'failed' | 'not_found'
    screenshot_url: Optional[str] = None
    urlscan_verdict_malicious: Optional[bool] = None
    urlscan_score: Optional[int] = None
    detected_brands: list[str] = field(default_factory=list)
    report_url: Optional[str] = None
    error: Optional[str] = None


class UrlscanError(Exception):
    """Base exception for urlscan integration."""


def _clean_expired_cache() -> None:
    """Remove expired entries from the URL cache."""
    now = time.time()
    expired = [k for k, (_, exp) in _url_cache.items() if exp < now]
    for k in expired:
        del _url_cache[k]


def submit_scan(url: str) -> ScanSubmission:
    """Submit a URL to urlscan.io for scanning.

    Returns immediately with a scan_id. The actual scan takes 10-30 seconds
    to complete; clients should poll get_scan_result(scan_id) until ready.

    Uses an in-memory 24-hour cache to avoid re-submitting URLs we've
    recently scanned.

    Raises UrlscanError if the API key is not configured or the submission
    fails. The caller should catch this and fall back to no-screenshot mode.
    """
    if not settings.urlscan_api_key:
        raise UrlscanError("URLSCAN_API_KEY not configured")

    _clean_expired_cache()

    # Cache hit: return existing scan_id immediately
    if url in _url_cache:
        scan_id, _ = _url_cache[url]
        log.info("urlscan_cache_hit", url=url, scan_id=scan_id)
        return ScanSubmission(scan_id=scan_id, submitted_at=time.time(), cached=True)

    # Cache miss: submit to urlscan
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                SUBMIT_ENDPOINT,
                headers={
                    "API-Key": settings.urlscan_api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "url": url,
                    "visibility": "unlisted",
                    "tags": ["phishing-detector", "automated"],
                },
            )
    except httpx.RequestError as e:
        log.error("urlscan_submit_network_error", url=url, error=str(e))
        raise UrlscanError(f"Network error submitting to urlscan: {e}") from e

    if response.status_code == 429:
        log.warning("urlscan_rate_limited", url=url)
        raise UrlscanError("urlscan.io rate limit exceeded; try again later")

    if response.status_code == 400:
        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        message = body.get("message", "Unknown error")
        log.warning("urlscan_rejected", url=url, reason=message)
        raise UrlscanError(f"urlscan.io rejected this URL: {message}")

    if response.status_code != 200:
        log.error("urlscan_submit_failed", url=url, status=response.status_code)
        raise UrlscanError(f"urlscan.io returned HTTP {response.status_code}")

    data = response.json()
    scan_id = data.get("uuid")
    if not scan_id:
        raise UrlscanError("urlscan response missing uuid field")

    _url_cache[url] = (scan_id, time.time() + _CACHE_TTL_SECONDS)
    log.info("urlscan_submitted", url=url, scan_id=scan_id)

    return ScanSubmission(scan_id=scan_id, submitted_at=time.time(), cached=False)


def get_scan_result(scan_id: str) -> ScanResult:
    """Poll urlscan.io for scan completion.

    Returns status='pending' if the scan is still running (urlscan returns 404).
    Returns status='ready' with the screenshot_url and verdict data once done.
    Returns status='failed' for permanent failures.
    """
    if not scan_id or len(scan_id) < 32:
        return ScanResult(status="failed", error="Invalid scan_id format")

    try:
        with httpx.Client(timeout=10.0) as client:
            headers = {}
            if settings.urlscan_api_key:
                headers["API-Key"] = settings.urlscan_api_key
            response = client.get(
                f"{RESULT_ENDPOINT}/{scan_id}/",
                headers=headers,
            )
    except httpx.RequestError as e:
        log.error("urlscan_result_network_error", scan_id=scan_id, error=str(e))
        return ScanResult(status="failed", error=f"Network error: {e}")

    # 404 while pending is documented behavior, NOT an error
    if response.status_code == 404:
        return ScanResult(status="pending")

    if response.status_code == 429:
        return ScanResult(status="pending", error="rate_limited; please retry")

    if response.status_code != 200:
        log.error("urlscan_result_failed", scan_id=scan_id, status=response.status_code)
        return ScanResult(
            status="failed",
            error=f"urlscan returned HTTP {response.status_code}",
        )

    data = response.json()

    task = data.get("task", {})
    screenshot_url = task.get("screenshotURL") or SCREENSHOT_URL_TEMPLATE.format(uuid=scan_id)

    verdicts = data.get("verdicts", {})
    urlscan_v = verdicts.get("urlscan", {})

    brands_raw = urlscan_v.get("brands", []) or []
    brand_names = [b.get("name") for b in brands_raw if isinstance(b, dict) and b.get("name")]

    return ScanResult(
        status="ready",
        screenshot_url=screenshot_url,
        urlscan_verdict_malicious=bool(urlscan_v.get("malicious", False)),
        urlscan_score=int(urlscan_v.get("score", 0)),
        detected_brands=brand_names,
        report_url=f"https://urlscan.io/result/{scan_id}/",
    )


def is_enabled() -> bool:
    """Returns True if the urlscan integration is configured and ready."""
    return bool(settings.urlscan_api_key)