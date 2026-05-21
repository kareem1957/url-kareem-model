"""
Lexical features: character-level patterns on the URL string.

These features operate on the raw text of the URL (canonical form,
post-normalization). They are cheap to compute (microseconds) and
catch a wide range of phishing patterns:

  - High entropy   -> randomly generated domains (DGA, fast-flux)
  - Long URLs      -> obfuscation via padding
  - Many digits    -> IP-literal hosts, random subdomains
  - Special chars  -> @ - _ ~ = & abuse
  - Keywords       -> login, verify, account, secure, etc.

Returns a flat dict[str, float] so downstream code can use the same
shape regardless of which extractor produced what.
"""
from __future__ import annotations

import math
from collections import Counter

from src.normalization.url_normalizer import NormalizedURL


# Keywords that disproportionately appear in phishing URLs.
# Sourced from PhishTank/OpenPhish empirical studies.
SUSPICIOUS_KEYWORDS: frozenset[str] = frozenset({
    "login", "signin", "sign-in", "log-in",
    "verify", "verification", "verifying",
    "account", "accounts", "update",
    "secure", "security", "safety",
    "confirm", "confirmation",
    "password", "passwd", "pwd",
    "bank", "banking",
    "paypal", "amazon", "apple", "microsoft", "google",
    "facebook", "instagram", "netflix", "ebay",
    "support", "service", "billing",
    "wallet", "crypto", "bitcoin",
    "free", "bonus", "prize", "winner",
    "alert", "warning", "suspended", "locked",
    "webscr", "cmd",
})

SPECIAL_CHARS: frozenset[str] = frozenset("-_~!*'();:@&=+$,/?#[]%")


def _shannon_entropy(text: str) -> float:
    """Shannon entropy in bits. Random strings score high (~4+)."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum(
        (c / length) * math.log2(c / length) for c in counts.values()
    )


def _char_ratio(text: str, predicate) -> float:
    """Fraction of characters in text matching predicate. 0 if empty."""
    if not text:
        return 0.0
    return sum(1 for c in text if predicate(c)) / len(text)


def extract_lexical_features(nurl: NormalizedURL) -> dict[str, float]:
    """Compute all lexical features for a normalized URL."""
    url = nurl.canonical
    host = nurl.hostname
    path = nurl.path
    query = nurl.query

    # Length-based features (longer URLs correlate with phishing)
    f_url_len = float(len(url))
    f_host_len = float(len(host))
    f_path_len = float(len(path))
    f_query_len = float(len(query))

    # Character composition of the full URL
    f_digit_ratio = _char_ratio(url, str.isdigit)
    f_alpha_ratio = _char_ratio(url, str.isalpha)
    f_special_ratio = _char_ratio(url, lambda c: c in SPECIAL_CHARS)
    f_upper_ratio = _char_ratio(url, str.isupper)

    # Entropy: random-looking strings (DGA-style) score high
    f_url_entropy = _shannon_entropy(url)
    f_host_entropy = _shannon_entropy(host)

    # Punctuation counts (specific characters with phishing signal)
    f_count_dots = float(url.count("."))
    f_count_hyphens = float(url.count("-"))
    f_count_underscores = float(url.count("_"))
    f_count_slashes = float(url.count("/"))
    f_count_questionmarks = float(url.count("?"))
    f_count_equals = float(url.count("="))
    f_count_at = float(url.count("@"))
    f_count_ampersand = float(url.count("&"))
    f_count_percent = float(url.count("%"))
    f_count_tilde = float(url.count("~"))
    f_count_plus = float(url.count("+"))
    f_count_asterisk = float(url.count("*"))
    f_count_hash = float(url.count("#"))
    f_count_dollar = float(url.count("$"))
    f_count_comma = float(url.count(","))

    # Host-specific punctuation (hyphens and digits in host are red flags)
    f_host_hyphens = float(host.count("-"))
    f_host_digits = float(sum(1 for c in host if c.isdigit()))
    f_host_digit_ratio = (
        f_host_digits / f_host_len if f_host_len > 0 else 0.0
    )

    # Suspicious keyword count (case-insensitive substring match)
    url_lower = url.lower()
    f_keyword_count = float(
        sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)
    )
    f_keyword_in_host = float(
        sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in host.lower())
    )
    f_keyword_in_path = float(
        sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in path.lower())
    )

    # Adversarial signals captured during normalization
    f_had_punycode = float(nurl.unicode.had_punycode)
    f_mixed_script = float(nurl.unicode.is_mixed_script)
    f_invisible_chars = float(nurl.unicode.has_invisible_chars)
    f_had_credentials = float(nurl.had_embedded_credentials)
    f_suspicious_scheme = float(nurl.had_suspicious_scheme)
    f_double_encoded = float(nurl.encoding.had_double_encoding)
    f_decode_iterations = float(nurl.encoding.decode_iterations)

    return {
        "url_len": f_url_len,
        "host_len": f_host_len,
        "path_len": f_path_len,
        "query_len": f_query_len,
        "digit_ratio": f_digit_ratio,
        "alpha_ratio": f_alpha_ratio,
        "special_ratio": f_special_ratio,
        "upper_ratio": f_upper_ratio,
        "url_entropy": f_url_entropy,
        "host_entropy": f_host_entropy,
        "count_dots": f_count_dots,
        "count_hyphens": f_count_hyphens,
        "count_underscores": f_count_underscores,
        "count_slashes": f_count_slashes,
        "count_questionmarks": f_count_questionmarks,
        "count_equals": f_count_equals,
        "count_at": f_count_at,
        "count_ampersand": f_count_ampersand,
        "count_percent": f_count_percent,
        "count_tilde": f_count_tilde,
        "count_plus": f_count_plus,
        "count_asterisk": f_count_asterisk,
        "count_hash": f_count_hash,
        "count_dollar": f_count_dollar,
        "count_comma": f_count_comma,
        "host_hyphens": f_host_hyphens,
        "host_digits": f_host_digits,
        "host_digit_ratio": f_host_digit_ratio,
        "keyword_count": f_keyword_count,
        "keyword_in_host": f_keyword_in_host,
        "keyword_in_path": f_keyword_in_path,
        "had_punycode": f_had_punycode,
        "mixed_script": f_mixed_script,
        "invisible_chars": f_invisible_chars,
        "had_credentials": f_had_credentials,
        "suspicious_scheme": f_suspicious_scheme,
        "double_encoded": f_double_encoded,
        "decode_iterations": f_decode_iterations,
    }