"""
Deterministic rule engine for phishing detection.

The ML model in src/models/lgbm_model.py learns patterns from data and
covers a broad range of phishing variants. But ML models have blind
spots: they can be fooled by URLs that look "normal" on average across
65 features but contain one extremely suspicious signal (e.g., a
single-edit typosquat of a major brand).

The rule engine fills these gaps with high-precision, hand-written
checks. Each rule:

  1. Targets a SPECIFIC, well-known phishing pattern.
  2. Carries a human-readable explanation, used in the API response.
  3. Fires only when we are NEARLY CERTAIN - false positives here
     are more damaging than false negatives.

The engine is intentionally conservative. It is NOT a replacement
for the ML model; it is a complement.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.normalization.url_normalizer import NormalizedURL


@dataclass(frozen=True)
class RuleVerdict:
    """A single rule's output."""

    rule_id: str
    triggered: bool
    severity: float          # 0.0 (informational) to 1.0 (definitive phishing)
    explanation: str


@dataclass(frozen=True)
class RuleEngineResult:
    """Aggregated output of all rules for a URL."""

    triggered_rules: tuple[RuleVerdict, ...] = field(default_factory=tuple)

    @property
    def is_phishing(self) -> bool:
        """True if any high-severity rule fired."""
        return any(r.severity >= 0.85 for r in self.triggered_rules)

    @property
    def max_severity(self) -> float:
        """Severity of the most confident triggered rule."""
        if not self.triggered_rules:
            return 0.0
        return max(r.severity for r in self.triggered_rules)

    @property
    def explanations(self) -> list[str]:
        """Human-readable explanation strings."""
        return [r.explanation for r in self.triggered_rules]


# --------------------------------------------------------------------------- #
# Individual rules                                                            #
# --------------------------------------------------------------------------- #

# Top brands phishing attackers most commonly impersonate. Mirrors
# the list in src/features/domain.py.
_TARGET_BRANDS = frozenset({
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

# TLDs almost exclusively used for phishing/abuse.
_DEFINITIVE_ABUSE_TLDS = frozenset({"tk", "ml", "ga", "cf", "gq"})

# TLDs heavily associated with abuse (but with some legitimate use).
_HIGH_RISK_TLDS = frozenset({
    "xyz", "top", "loan", "download", "accountant",
    "click", "racing", "icu", "monster", "cyou",
})


def _levenshtein(a: str, b: str) -> int:
    """Inline copy of the edit-distance function from features/domain.py
    so the rule engine has no dependency on the feature layer."""
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
                curr_row[j - 1] + 1,
                prev_row[j] + 1,
                prev_row[j - 1] + cost,
            )
        prev_row = curr_row
    return prev_row[-1]


def rule_typosquat(nurl: NormalizedURL) -> RuleVerdict:
    """
    Catch brand typosquats: a domain 1-2 edits away from a known brand,
    but not actually any of those brands.

    Examples that fire:
        paypa1.com, paypall.com, micr0soft.com, arnazon.com
    Examples that don't:
        paypal.com (exact match - it's the real brand)
        anthropic.com (5+ edits from any brand)
    """
    domain = nurl.domain_parts.domain.lower()
    if not domain or domain in _TARGET_BRANDS:
        return RuleVerdict(
            rule_id="typosquat",
            triggered=False,
            severity=0.0,
            explanation="",
        )

    best_dist = min(
        (_levenshtein(domain, brand) for brand in _TARGET_BRANDS),
        default=999,
    )
    if 1 <= best_dist <= 2:
        return RuleVerdict(
            rule_id="typosquat",
            triggered=True,
            severity=0.95,
            explanation=(
                f"Domain '{domain}' is {best_dist} character edit(s) away "
                "from a major brand - classic typosquatting pattern."
            ),
        )
    return RuleVerdict("typosquat", False, 0.0, "")


def rule_ip_as_host(nurl: NormalizedURL) -> RuleVerdict:
    """
    IP-literal hostnames are almost exclusively malicious.
    Legitimate services use domain names.
    """
    if nurl.domain_parts.is_ip_address:
        return RuleVerdict(
            rule_id="ip_as_host",
            triggered=True,
            severity=0.90,
            explanation=(
                "URL uses a raw IP address instead of a domain name. "
                "Legitimate services use domain names; raw IPs are a "
                "strong phishing indicator."
            ),
        )
    return RuleVerdict("ip_as_host", False, 0.0, "")


def rule_brand_in_subdomain(nurl: NormalizedURL) -> RuleVerdict:
    """
    Brand name in the subdomain of an unrelated registered domain.

    Examples that fire:
        paypal.evil.com           (brand in subdomain)
        login.apple-id.evil.xyz   (brand in subdomain)
    Examples that don't:
        accounts.google.com       (google IS the registered domain)
    """
    subdomain = nurl.domain_parts.subdomain.lower()
    domain = nurl.domain_parts.domain.lower()
    if not subdomain:
        return RuleVerdict("brand_in_subdomain", False, 0.0, "")

    for brand in _TARGET_BRANDS:
        if brand in subdomain and brand != domain:
            return RuleVerdict(
                rule_id="brand_in_subdomain",
                triggered=True,
                severity=0.92,
                explanation=(
                    f"Brand name '{brand}' appears in a subdomain of "
                    f"'{nurl.domain_parts.registered_domain}', which is "
                    "not the real brand. This is a deceptive subdomain "
                    "stuffing pattern."
                ),
            )
    return RuleVerdict("brand_in_subdomain", False, 0.0, "")


def rule_brand_with_hyphens(nurl: NormalizedURL) -> RuleVerdict:
    """
    Hyphenated domain containing a brand name (apple-support-login.com).
    Legitimate brands almost never use hyphenated impersonations of
    themselves.
    """
    domain = nurl.domain_parts.domain.lower()
    if "-" not in domain:
        return RuleVerdict("brand_with_hyphens", False, 0.0, "")

    parts = domain.split("-")
    matching_brand = next(
        (brand for brand in _TARGET_BRANDS if brand in parts),
        None,
    )
    if matching_brand:
        return RuleVerdict(
            rule_id="brand_with_hyphens",
            triggered=True,
            severity=0.88,
            explanation=(
                f"Domain contains the brand name '{matching_brand}' "
                "combined with other words via hyphens (e.g. "
                "'brand-secure-login'). This is a classic impersonation "
                "pattern; legitimate brands do not register such domains."
            ),
        )
    return RuleVerdict("brand_with_hyphens", False, 0.0, "")


def rule_definitive_abuse_tld(nurl: NormalizedURL) -> RuleVerdict:
    """
    A handful of TLDs (.tk, .ml, .ga, .cf, .gq) are free-registration
    domains used almost exclusively for abuse. Even a benign-looking
    URL on one of these warrants a phishing verdict.
    """
    tld_lookup = (
        nurl.domain_parts.suffix.split(".")[-1].lower()
        if nurl.domain_parts.suffix else ""
    )
    if tld_lookup in _DEFINITIVE_ABUSE_TLDS:
        return RuleVerdict(
            rule_id="definitive_abuse_tld",
            triggered=True,
            severity=0.87,
            explanation=(
                f"Domain uses TLD '.{tld_lookup}', which is a free-"
                "registration TLD used overwhelmingly for phishing and "
                "malware distribution."
            ),
        )
    return RuleVerdict("definitive_abuse_tld", False, 0.0, "")


def rule_embedded_credentials(nurl: NormalizedURL) -> RuleVerdict:
    """The user:pass@host trick: https://paypal.com@evil.com.
    Always malicious in modern web usage."""
    if nurl.had_embedded_credentials:
        return RuleVerdict(
            rule_id="embedded_credentials",
            triggered=True,
            severity=0.93,
            explanation=(
                "URL embeds credentials before an '@' symbol, a classic "
                "deception trick where a victim reads the part before "
                "'@' as the destination but the browser navigates to "
                "the part after."
            ),
        )
    return RuleVerdict("embedded_credentials", False, 0.0, "")


def rule_suspicious_scheme(nurl: NormalizedURL) -> RuleVerdict:
    """javascript:, data:, file: URLs in a "phishing detection" context
    are essentially always malicious."""
    if nurl.had_suspicious_scheme:
        return RuleVerdict(
            rule_id="suspicious_scheme",
            triggered=True,
            severity=0.90,
            explanation=(
                f"URL uses a non-web scheme ('{nurl.scheme}:'), which "
                "is used to execute code or open local files rather "
                "than navigate to a webpage."
            ),
        )
    return RuleVerdict("suspicious_scheme", False, 0.0, "")


def rule_mixed_script_hostname(nurl: NormalizedURL) -> RuleVerdict:
    """Mixed-script hostnames (Latin + Cyrillic, etc) are the foundation
    of homograph attacks."""
    if nurl.unicode.is_mixed_script:
        return RuleVerdict(
            rule_id="mixed_script_hostname",
            triggered=True,
            severity=0.93,
            explanation=(
                "Hostname mixes characters from multiple writing systems "
                f"({', '.join(sorted(nurl.unicode.scripts_detected))}). "
                "This is a Unicode homograph attack used to spoof "
                "trusted domains with look-alike characters."
            ),
        )
    return RuleVerdict("mixed_script_hostname", False, 0.0, "")


# Registry: order matters only for the explanation list; severity
# determines the actual verdict.
ALL_RULES = (
    rule_embedded_credentials,
    rule_mixed_script_hostname,
    rule_typosquat,
    rule_brand_in_subdomain,
    rule_ip_as_host,
    rule_suspicious_scheme,
    rule_brand_with_hyphens,
    rule_definitive_abuse_tld,
)


def evaluate_rules(nurl: NormalizedURL) -> RuleEngineResult:
    """Run every rule against the URL and aggregate the verdicts."""
    triggered = tuple(
        verdict for verdict in (rule(nurl) for rule in ALL_RULES)
        if verdict.triggered
    )
    return RuleEngineResult(triggered_rules=triggered)