"""
TLD extraction using the Public Suffix List.

Naive splitting on '.' is wrong: 'example.co.uk' has registrable domain
'example.co.uk', not 'co.uk'. The Public Suffix List (PSL) is the
authoritative source for what counts as a TLD; tldextract bundles it.

We wrap tldextract in a single function so the rest of the code never
imports the library directly. This makes the dependency swappable.

Performance: we instantiate TLDExtract once at module load (loads the
suffix list once) and add a small LRU cache on top because URLs in a
single dataset have high host repetition. Bulk pipelines that called
extract on 100k+ rows previously hit a 50-100x slowdown from
re-instantiating per call.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import tldextract


# suffix_list_urls=() disables network fetches; uses bundled snapshot.
# This is critical for reproducibility and for sandbox environments
# where outbound network calls are blocked.
# cache_dir=None tells tldextract not to write to disk.
_extractor = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


@dataclass(frozen=True)
class DomainParts:
    """Parsed domain components."""

    subdomain: str
    domain: str
    suffix: str
    registered_domain: str
    is_ip_address: bool


@lru_cache(maxsize=100_000)
def _extract_cached(hostname: str) -> DomainParts:
    """LRU-cached TLD extraction. The cache amortizes repeat lookups
    common in batch jobs (multiple URLs per host)."""
    if not hostname:
        return DomainParts("", "", "", "", False)

    result = _extractor(hostname)
    is_ip = bool(result.ipv4) or bool(getattr(result, "ipv6", ""))

    registered = (
        f"{result.domain}.{result.suffix}"
        if result.domain and result.suffix
        else result.domain or hostname
    )

    return DomainParts(
        subdomain=result.subdomain or "",
        domain=result.domain or "",
        suffix=result.suffix or "",
        registered_domain=registered,
        is_ip_address=is_ip,
    )


def extract_domain_parts(hostname: str) -> DomainParts:
    """Split a hostname into subdomain, domain, and public-suffix parts."""
    return _extract_cached(hostname)