"""
Main URL normalization pipeline.

The single entry point for the rest of the system. Given a raw URL
(possibly hostile, possibly malformed), it returns a NormalizedURL
object containing:

  - The canonical form for feature extraction
  - The original input for audit logging
  - Structured parts (scheme, host, path, query, fragment)
  - Adversarial signals detected during normalization (these become
    features in their own right; we don't throw them away)

Order of operations is deliberate:
  1. Input validation (length, type, scheme allowlist)
  2. Iterative percent-decoding (defeats %xx obfuscation)
  3. Parsing into components
  4. Embedded credential stripping (the @ attack)
  5. Hostname Unicode analysis and punycode decoding
  6. Path normalization
  7. Optional redirect resolution
  8. TLD extraction

Each stage's output is preserved on the result, so feature engineering
can compare original vs normalized to detect obfuscation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit, SplitResult

from config.settings import settings
from src.normalization.encoding_handler import (
    EncodingAnalysis,
    iterative_unquote,
)
from src.normalization.redirect_resolver import (
    RedirectAnalysis,
    resolve_redirects,
)
from src.normalization.tld_extractor import DomainParts, extract_domain_parts
from src.normalization.unicode_handler import (
    UnicodeAnalysis,
    analyze_hostname,
)
from src.utils.exceptions import URLNormalizationError, URLValidationError
from src.utils.logger import get_logger


logger = get_logger(__name__)


ALLOWED_SCHEMES = frozenset({"http", "https"})
SUSPICIOUS_SCHEMES = frozenset({"javascript", "data", "file", "ftp"})


@dataclass(frozen=True)
class NormalizedURL:
    """Canonical URL plus everything we learned along the way."""

    # Inputs and outputs
    original: str
    canonical: str

    # Parsed components (from canonical form)
    scheme: str
    hostname: str
    port: int | None
    path: str
    query: str
    fragment: str

    # Sub-analyses (each is a feature source)
    encoding: EncodingAnalysis
    unicode: UnicodeAnalysis
    domain_parts: DomainParts
    redirect: RedirectAnalysis

    # Adversarial signals captured during normalization
    had_embedded_credentials: bool
    had_suspicious_scheme: bool
    had_at_symbol_in_authority: bool
    stripped_credentials: str | None

    # Additional context
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _validate_input(url: str) -> str:
    """Reject obviously malformed inputs early."""
    if not isinstance(url, str):
        raise URLValidationError("URL must be a string")
    url = url.strip()
    if not url:
        raise URLValidationError("URL must be non-empty")
    if len(url) > settings.max_url_length:
        raise URLValidationError(
            f"URL exceeds maximum length of {settings.max_url_length}"
        )
    return url


def _ensure_scheme(url: str) -> tuple[str, tuple[str, ...]]:
    """
    Ensure the URL has a scheme. If none is present, prepend http://.

    A URL has a scheme if it starts with letters followed by a colon
    (per RFC 3986). We deliberately do NOT check only for "://" because
    that would miss schemes like javascript:, data:, file:, mailto:
    which must be flagged downstream as suspicious, not silently
    replaced with http://.

    Returns (url_with_scheme, warnings).
    """
    warnings: list[str] = []
    scheme_end = url.find(":")
    has_scheme = (
        scheme_end > 0
        and url[:scheme_end].isalpha()
    )
    if not has_scheme:
        warnings.append("no_scheme_provided_defaulted_to_http")
        url = "http://" + url
    return url, tuple(warnings)


def _strip_embedded_credentials(
    parsed: SplitResult,
) -> tuple[SplitResult, str | None, bool]:
    """
    Remove user:password@ from the authority component.

    Returns (cleaned_parsed, stripped_credentials, had_credentials).
    The @ attack is a classic phishing technique:
      https://bank.com@evil.com/login
    The browser navigates to evil.com; the user reads "bank.com".
    """
    netloc = parsed.netloc
    if "@" not in netloc:
        return parsed, None, False

    credentials, _, host_part = netloc.rpartition("@")
    new_parsed = parsed._replace(netloc=host_part)
    return new_parsed, credentials, True


def normalize(url: str) -> NormalizedURL:
    """
    Normalize a URL through the full pipeline.

    Raises:
        URLValidationError: input failed basic validation.
        URLNormalizationError: a downstream stage failed unrecoverably.
    """
    original = url
    url = _validate_input(url)
    url, scheme_warnings = _ensure_scheme(url)
    warnings: list[str] = list(scheme_warnings)

    # Stage 1: iterative percent-decoding
    encoding = iterative_unquote(url)
    decoded_url = encoding.decoded

    # Stage 2: parse into components
    try:
        parsed = urlsplit(decoded_url)
    except ValueError as exc:
        raise URLNormalizationError(f"urlsplit failed: {exc}") from exc

    scheme = (parsed.scheme or "").lower()
    if scheme in SUSPICIOUS_SCHEMES:
        warnings.append(f"suspicious_scheme:{scheme}")
    had_suspicious_scheme = scheme in SUSPICIOUS_SCHEMES

    # Stage 3: strip embedded credentials
    had_at_in_authority = "@" in (parsed.netloc or "")
    parsed, stripped_creds, had_creds = _strip_embedded_credentials(parsed)
    if had_creds:
        warnings.append("embedded_credentials_stripped")

    # Stage 4: hostname Unicode analysis
    hostname_raw = parsed.hostname or ""
    try:
        port = parsed.port
    except ValueError:
        # Non-numeric port. Often a sign of a malformed or hostile URL
        # (e.g. javascript:alert(1) where urllib treats "alert(1)" as netloc).
        port = None
        warnings.append("malformed_port_in_authority")

    unicode_analysis = analyze_hostname(hostname_raw)
    canonical_host = unicode_analysis.decoded.lower()
    if unicode_analysis.is_mixed_script:
        warnings.append("mixed_script_hostname")
    if unicode_analysis.has_invisible_chars:
        warnings.append("invisible_chars_in_hostname")

    # Stage 5: path normalization (lowercase scheme/host only; path is case-sensitive)
    path = parsed.path or "/"
    query = parsed.query or ""
    fragment = parsed.fragment or ""

    # Rebuild canonical authority
    if port is not None:
        canonical_authority = f"{canonical_host}:{port}"
    else:
        canonical_authority = canonical_host

    canonical = urlunsplit((
        scheme if scheme in ALLOWED_SCHEMES else (scheme or "http"),
        canonical_authority,
        path,
        query,
        fragment,
    ))

    # Stage 6: optional redirect resolution
    redirect = resolve_redirects(canonical)

    # Stage 7: TLD extraction on the canonical (post-redirect) host
    if redirect.resolved:
        try:
            final_host = urlsplit(redirect.final_url).hostname or canonical_host
        except ValueError:
            final_host = canonical_host
    else:
        final_host = canonical_host
    domain_parts = extract_domain_parts(final_host)

    logger.debug(
        "url_normalized",
        original=original,
        canonical=canonical,
        warnings=warnings,
    )

    return NormalizedURL(
        original=original,
        canonical=canonical,
        scheme=scheme,
        hostname=canonical_host,
        port=port,
        path=path,
        query=query,
        fragment=fragment,
        encoding=encoding,
        unicode=unicode_analysis,
        domain_parts=domain_parts,
        redirect=redirect,
        had_embedded_credentials=had_creds,
        had_suspicious_scheme=had_suspicious_scheme,
        had_at_symbol_in_authority=had_at_in_authority,
        stripped_credentials=stripped_creds,
        warnings=tuple(warnings),
    )