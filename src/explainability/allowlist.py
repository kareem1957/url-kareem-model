"""
Domain allowlist for known-legitimate sites.

Real-world phishing detection products combine ML with reputation
databases. The ML model handles the long tail of unknown domains;
the allowlist short-circuits decisions for the well-known good
domains that ML often mis-classifies due to training data biases.

Source: Tranco top-100k registered domains (already downloaded to
data/raw/tranco/top-1m.csv during dataset preparation).

Lookup is exact-match on the REGISTERED DOMAIN only. This is critical:
  - 'github.com' is in the allowlist -> github.com/anthropics is allowlisted
  - 'github.com' is in the allowlist -> paypal.github.com is allowlisted
    (a real subdomain of github.com)
  - 'github.com' is in the allowlist -> NOT a match for github-secure.com
    (that's a different registered domain)
  - 'github.com' is in the allowlist -> NOT a match for github.com.evil.com
    (registered domain is evil.com, not github.com)

The TLD extractor in src/normalization/tld_extractor.py handles all
the parsing correctly via the Public Suffix List, so we rely on
nurl.domain_parts.registered_domain rather than doing string ops.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from config.settings import settings
from src.utils.logger import get_logger


logger = get_logger(__name__)


TRANCO_CSV = settings.raw_data_dir / "tranco" / "top-1m.csv"


@lru_cache(maxsize=1)
def load_allowlist(top_n: int = 100_000) -> frozenset[str]:
    """
    Load the top-N Tranco domains into a frozenset for O(1) lookup.

    Cached: only read from disk once per process lifetime. The set
    of 100k strings occupies ~5 MB of memory, negligible.
    """
    if not TRANCO_CSV.exists():
        logger.warning(
            "allowlist_source_missing",
            path=str(TRANCO_CSV),
            note="Allowlist disabled; ML will handle all URLs.",
        )
        return frozenset()

    domains: set[str] = set()
    with TRANCO_CSV.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= top_n:
                break
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                domains.add(parts[1].strip().lower())

    logger.info("allowlist_loaded", count=len(domains), source=str(TRANCO_CSV))
    return frozenset(domains)


def is_allowlisted(registered_domain: str) -> bool:
    """
    Check if a registered domain is in the Tranco top-N allowlist.

    Use the registered_domain field from DomainParts. Subdomains and
    paths are intentionally ignored - the allowlist applies to the
    registered domain as a whole.
    """
    if not registered_domain:
        return False
    return registered_domain.lower() in load_allowlist()