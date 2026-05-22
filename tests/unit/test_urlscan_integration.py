"""
Unit tests for urlscan_integration.

Mocks httpx so tests don't make real network calls. This is critical:
- Unit tests must be fast (no network)
- Unit tests must be deterministic (no external service)
- Unit tests must work offline (in CI environments)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.api import urlscan_integration
from src.api.urlscan_integration import (
    ScanResult,
    ScanSubmission,
    UrlscanError,
    get_scan_result,
    submit_scan,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Reset the module-level cache before each test for isolation."""
    urlscan_integration._url_cache.clear()
    yield
    urlscan_integration._url_cache.clear()


@pytest.fixture
def mock_api_key(monkeypatch):
    """Set a test API key on the settings object."""
    monkeypatch.setattr(
        urlscan_integration.settings, "urlscan_api_key", "test_key_12345"
    )


class TestSubmitScan:
    def test_raises_when_api_key_missing(self, monkeypatch):
        monkeypatch.setattr(urlscan_integration.settings, "urlscan_api_key", "")
        with pytest.raises(UrlscanError, match="not configured"):
            submit_scan("https://example.com")

    def test_successful_submission(self, mock_api_key):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": "Submission successful",
            "uuid": "abc-123-def-456-aaaa-bbbb-cccc-dddd",
        }
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = submit_scan("https://paypa1.com/login")

        assert isinstance(result, ScanSubmission)
        assert result.scan_id == "abc-123-def-456-aaaa-bbbb-cccc-dddd"
        assert result.cached is False
        assert result.submitted_at > 0

    def test_cache_hit_returns_existing_scan_id(self, mock_api_key):
        url = "https://paypa1.com/login"
        cached_scan_id = "cached-uuid-12345-abcdef-678901234567"
        urlscan_integration._url_cache[url] = (
            cached_scan_id,
            time.time() + 3600,
        )

        with patch("httpx.Client") as mock_client_class:
            result = submit_scan(url)
            mock_client_class.assert_not_called()

        assert result.scan_id == cached_scan_id
        assert result.cached is True

    def test_expired_cache_entry_triggers_resubmit(self, mock_api_key):
        url = "https://paypa1.com/login"
        urlscan_integration._url_cache[url] = (
            "old-uuid",
            time.time() - 100,
        )

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"uuid": "new-uuid-after-expiry-12345678"}
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = submit_scan(url)

        assert result.scan_id == "new-uuid-after-expiry-12345678"
        assert result.cached is False

    def test_rate_limit_raises_clear_error(self, mock_api_key):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            with pytest.raises(UrlscanError, match="rate limit"):
                submit_scan("https://example.com")

    def test_rejected_url_raises_with_reason(self, mock_api_key):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "message": "URL contains an invalid host or scheme",
        }
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            with pytest.raises(UrlscanError, match="invalid host"):
                submit_scan("http://192.168.1.1/")

    def test_network_error_raises_urlscan_error(self, mock_api_key):
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.post.side_effect = httpx.ConnectError("DNS failed")
            mock_client_class.return_value = mock_client

            with pytest.raises(UrlscanError, match="Network error"):
                submit_scan("https://example.com")

    def test_response_without_uuid_raises(self, mock_api_key):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": "ok but no uuid"}
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            with pytest.raises(UrlscanError, match="missing uuid"):
                submit_scan("https://example.com")


class TestGetScanResult:
    def test_404_means_pending_not_failed(self, mock_api_key):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = get_scan_result("abc-123-def-456-still-pending-uuid")

        assert result.status == "pending"
        assert result.screenshot_url is None
        assert result.error is None

    def test_ready_with_full_verdict(self, mock_api_key):
        scan_id = "abc-123-def-456-aaaa-bbbb-cccc-dddd"
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "task": {
                "uuid": scan_id,
                "screenshotURL": f"https://urlscan.io/screenshots/{scan_id}.png",
                "url": "https://paypa1.com/login",
            },
            "verdicts": {
                "urlscan": {
                    "malicious": True,
                    "score": 90,
                    "brands": [
                        {"name": "PayPal", "vertical": ["financial"]},
                    ],
                },
            },
        }
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = get_scan_result(scan_id)

        assert result.status == "ready"
        assert result.screenshot_url == f"https://urlscan.io/screenshots/{scan_id}.png"
        assert result.urlscan_verdict_malicious is True
        assert result.urlscan_score == 90
        assert "PayPal" in result.detected_brands
        assert scan_id in result.report_url

    def test_invalid_scan_id_returns_failed(self, mock_api_key):
        result = get_scan_result("too-short")
        assert result.status == "failed"
        assert "Invalid" in result.error

    def test_rate_limit_returns_pending_with_message(self, mock_api_key):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = get_scan_result("abc-123-def-456-rate-limited-uuid-1234")

        assert result.status == "pending"
        assert "rate_limited" in (result.error or "")