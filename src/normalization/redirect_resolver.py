"""
Redirect chain resolver.

Phishing URLs often hide behind redirect chains:
  - URL shorteners (bit.ly, t.co, tinyurl)
  - Open redirects on trusted domains
  - Multi-hop redirects through legitimate-looking intermediaries

Resolving these requires network I/O, which we keep OPT-IN
(settings.resolve_redirects). When enabled, we follow up to N hops with
a short timeout, never execute JavaScript, and use HEAD requests to
avoid downloading payloads. Output is the final URL plus the hop chain
for forensic analysis.

For offline training and the default API path, the normalizer skips
this stage. Operators can enable it for higher-fidelity detection
in production where latency budget allows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from config.settings import settings
from src.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class RedirectAnalysis:
    """Result of redirect chain resolution."""

    final_url: str
    chain: tuple[str, ...] = field(default_factory=tuple)
    hop_count: int = 0
    resolved: bool = False        # True if we actually tried to resolve
    error: str | None = None


def resolve_redirects(url: str) -> RedirectAnalysis:
    """
    Follow redirects starting at url. Returns the final URL and chain.

    Respects settings.resolve_redirects. If disabled (default), returns
    the input URL unchanged. Network errors are caught and reported;
    the function never raises.
    """
    if not settings.resolve_redirects:
        return RedirectAnalysis(final_url=url, chain=(url,), hop_count=0,
                                resolved=False, error=None)

    chain: list[str] = [url]
    current = url
    try:
        with httpx.Client(
            follow_redirects=False,
            timeout=settings.redirect_timeout_seconds,
            headers={"User-Agent": "PhishingDetector/1.0 (security-scanner)"},
        ) as client:
            for _ in range(settings.max_redirect_hops):
                response = client.head(current)
                if response.status_code not in (301, 302, 303, 307, 308):
                    break
                location = response.headers.get("location")
                if not location:
                    break
                # Handle relative redirects
                current = str(httpx.URL(current).join(location))
                chain.append(current)

        return RedirectAnalysis(
            final_url=current,
            chain=tuple(chain),
            hop_count=len(chain) - 1,
            resolved=True,
            error=None,
        )
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        logger.warning("redirect_resolution_failed", url=url, error=str(exc))
        return RedirectAnalysis(
            final_url=current,
            chain=tuple(chain),
            hop_count=len(chain) - 1,
            resolved=True,
            error=str(exc),
        )
