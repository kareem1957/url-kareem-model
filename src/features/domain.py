"""
Domain features: signals from the registered domain itself.

These features focus on the part the user MOST trusts when reading
a URL: the registered domain (everything-before-the-TLD). Phishing
attackers attack this part because users have learned to look at it.

Key patterns this catches:
  - Typosquatting    -> paypa1.com, micr0soft.com, arnazon.com
  - Cheap TLD abuse  -> .xyz, .top, .tk, .ml domains
  - IP-as-host       -> http://203.0.113.5/login
  - Long random TLDs -> .accountant, .download, etc.
  - Hyphenated brand impersonation -> apple-support-login.com
"""
from __future__ import annotations

from src.normalization.url_normalizer import NormalizedURL


# TLDs disproportionately associated with phishing.
# Source: APWG / Spamhaus / academic phishing studies.
# Score 1.0 = highest risk, 0.0 = clean.
TLD_RISK_SCORES: dict[str, float] = {
    "tk": 1.0, "ml": 1.0, "ga": 1.0, "cf": 1.0, "gq": 1.0,
    "xyz": 0.85, "top": 0.85, "club": 0.75, "work": 0.75,
    "loan": 0.95, "download": 0.95, "accountant": 0.95,
    "win": 0.85, "review": 0.80, "men": 0.80, "stream": 0.80,
    "country": 0.80, "racing": 0.85, "science": 0.80,
    "party": 0.80, "trade": 0.80, "date": 0.80,
    "click": 0.85, "link": 0.65, "online": 0.60,
    "site": 0.55, "info": 0.50, "icu": 0.85, "rest": 0.70,
    "monster": 0.85, "bar": 0.55, "buzz": 0.70, "cyou": 0.85,
    "fit": 0.55, "fun": 0.55, "live": 0.45, "shop": 0.45,
    "store": 0.45, "space": 0.55, "tech": 0.40, "uno": 0.60,
    "world": 0.45, "host": 0.55,
    # Common safe TLDs get explicit 0 so missing-key lookups stay neutral.
    "com": 0.0, "org": 0.0, "net": 0.0, "edu": 0.0, "gov": 0.0,
    "io": 0.05, "co": 0.10, "ai": 0.05,
}


# Brands phishing attackers most commonly impersonate.
# We compare the registered domain (without TLD) against these.
TARGET_BRANDS: frozenset[str] = frozenset({
    "paypal", "apple", "microsoft", "amazon", "google",
    "facebook", "instagram", "netflix", "ebay", "linkedin",
    "twitter", "whatsapp", "telegram", "discord",
    "chase", "wellsfargo", "bankofamerica", "citibank", "hsbc",
    "barclays", "santander", "lloyds",
    "outlook", "yahoo", "gmail", "icloud", "office365",
    "dropbox", "onedrive", "github", "adobe",
    "binance", "coinbase", "kraken", "metamask",
    "spotify", "twitch", "steam", "playstation", "xbox",
})


def _levenshtein(a: str, b: str) -> int:
    """
    Compute Levenshtein distance: min edits (insert/delete/substitute)
    to turn a into b. We use a small DP table; both strings are short
    so O(n*m) is fine.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr_row[j] = min(
                curr_row[j - 1] + 1,        # insert
                prev_row[j] + 1,            # delete
                prev_row[j - 1] + cost,     # substitute
            )
        prev_row = curr_row
    return prev_row[-1]


def _min_brand_distance(domain: str) -> tuple[int, str]:
    """
    Find the closest brand to domain and return (distance, brand).
    If domain exactly matches a brand, returns (0, brand) - meaning
    it's likely the real brand (or a perfect copy on a fake TLD).
    """
    if not domain:
        return (999, "")
    best_dist = 999
    best_brand = ""
    for brand in TARGET_BRANDS:
        d = _levenshtein(domain.lower(), brand)
        if d < best_dist:
            best_dist = d
            best_brand = brand
    return best_dist, best_brand


def extract_domain_features(nurl: NormalizedURL) -> dict[str, float]:
    """Compute domain-level features for a normalized URL."""
    domain = nurl.domain_parts.domain
    suffix = nurl.domain_parts.suffix
    subdomain = nurl.domain_parts.subdomain
    registered = nurl.domain_parts.registered_domain

    # TLD risk: cheap/abuse-prone TLDs score higher
    # For compound TLDs (co.uk), take the last segment for lookup
    tld_lookup = suffix.split(".")[-1] if suffix else ""
    f_tld_risk = TLD_RISK_SCORES.get(tld_lookup.lower(), 0.30)
    # 0.30 default for unknown TLDs (mildly suspicious by ignorance)

    # Domain length signals
    f_domain_len = float(len(domain))
    f_registered_len = float(len(registered))

    # Hyphens in domain (e.g. paypal-secure-login.com)
    f_domain_hyphens = float(domain.count("-"))
    f_domain_has_hyphen = float("-" in domain)

    # Digits in registered domain (apple1234.com)
    f_domain_digit_count = float(sum(1 for c in domain if c.isdigit()))
    f_domain_has_digits = float(any(c.isdigit() for c in domain))

    # IP-as-host (no domain at all)
    f_is_ip_host = float(nurl.domain_parts.is_ip_address)

    # Brand similarity: edit distance to closest known brand (on the
    # registered domain). "Looks like a brand but isn't" is the
    # classic typosquat signal.
    brand_dist, _ = _min_brand_distance(domain)
    f_brand_distance = float(brand_dist)
    # distance 1-3 = high suspicion; 0 = exact match (could be real);
    # >5 = unrelated.
    f_is_typosquat_candidate = float(1 <= brand_dist <= 3)

    # contains_brand_name: brand string appears in subdomain OR domain
    # but the *registered* domain is not actually that brand.
    # This catches:
    #   paypal-secure.evil.com   (brand in domain part)
    #   login.paypal.evil.com    (brand in subdomain)
    # without false-positiving on real paypal.com.
    domain_lower = domain.lower()
    subdomain_lower = subdomain.lower()
    contains_brand = False
    for brand in TARGET_BRANDS:
        if brand == domain_lower:
            # Exact match -> this IS the real brand domain (or a clone
            # on a fake TLD, which the tld_risk feature handles).
            # Don't flag it as "contains brand name".
            continue
        if brand in domain_lower or brand in subdomain_lower:
            contains_brand = True
            break
    f_contains_brand_name = float(contains_brand)

    # TLD length (very long TLDs like .accountant are often abused)
    f_tld_len = float(len(suffix))

    # Empty/missing domain (catastrophic parse failure or IP host)
    f_has_domain = float(bool(domain))

    return {
        "tld_risk": f_tld_risk,
        "tld_len": f_tld_len,
        "domain_len": f_domain_len,
        "registered_len": f_registered_len,
        "domain_hyphens": f_domain_hyphens,
        "domain_has_hyphen": f_domain_has_hyphen,
        "domain_digit_count": f_domain_digit_count,
        "domain_has_digits": f_domain_has_digits,
        "is_ip_host": f_is_ip_host,
        "brand_distance": f_brand_distance,
        "is_typosquat_candidate": f_is_typosquat_candidate,
        "contains_brand_name": f_contains_brand_name,
        "has_domain": f_has_domain,
    }