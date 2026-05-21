"""
Structural features: URL component counts and depths.

These features describe the SHAPE of the URL rather than its
character content. They are powerful against subdomain-stuffing
attacks like:

    https://login.paypal.com.security.update.evil.example.com/

A naive human reads "paypal.com" and trusts it. Our structural
features see 5 subdomain levels (huge red flag) and 0 of them
are the registered domain.
"""
from __future__ import annotations

from urllib.parse import parse_qsl

from src.normalization.url_normalizer import NormalizedURL


def extract_structural_features(nurl: NormalizedURL) -> dict[str, float]:
    """Compute structural features for a normalized URL."""
    subdomain = nurl.domain_parts.subdomain
    path = nurl.path
    query = nurl.query
    fragment = nurl.fragment

    # Subdomain depth: how many labels before the registered domain.
    # 0 means apex (example.com), 1 means www.example.com,
    # 4+ is a strong phishing signal (login.account.secure.example.com).
    f_subdomain_count = (
        float(subdomain.count(".") + 1) if subdomain else 0.0
    )
    f_subdomain_len = float(len(subdomain))

    # Path depth: number of /-separated segments
    path_segments = [s for s in path.split("/") if s]
    f_path_depth = float(len(path_segments))
    f_avg_path_segment_len = (
        sum(len(s) for s in path_segments) / len(path_segments)
        if path_segments
        else 0.0
    )
    f_max_path_segment_len = (
        float(max((len(s) for s in path_segments), default=0))
    )

    # Query parameter analysis (key=value&key2=value2 ...)
    # Many params or very long params correlate with tracking/phishing.
    try:
        query_pairs = parse_qsl(query, keep_blank_values=True)
    except ValueError:
        query_pairs = []
    f_query_param_count = float(len(query_pairs))
    f_query_avg_value_len = (
        sum(len(v) for _, v in query_pairs) / len(query_pairs)
        if query_pairs
        else 0.0
    )
    f_query_max_value_len = (
        float(max((len(v) for _, v in query_pairs), default=0))
    )

    # Fragment (the #... portion). Rare in legitimate URLs.
    f_has_fragment = float(bool(fragment))
    f_fragment_len = float(len(fragment))

    # Port presence (legitimate sites usually don't expose ports)
    f_has_port = float(nurl.port is not None)
    f_nonstandard_port = float(
        nurl.port is not None and nurl.port not in (80, 443)
    )

    # HTTPS as a feature: phishing sites can use HTTPS too (Let's Encrypt),
    # but absence of HTTPS is still mildly informative.
    f_is_https = float(nurl.scheme == "https")

    # Redirect chain length (only meaningful when redirect resolution
    # is enabled in settings; otherwise it's 0).
    f_redirect_hops = float(nurl.redirect.hop_count)

    return {
        "subdomain_count": f_subdomain_count,
        "subdomain_len": f_subdomain_len,
        "path_depth": f_path_depth,
        "avg_path_segment_len": f_avg_path_segment_len,
        "max_path_segment_len": f_max_path_segment_len,
        "query_param_count": f_query_param_count,
        "query_avg_value_len": f_query_avg_value_len,
        "query_max_value_len": f_query_max_value_len,
        "has_fragment": f_has_fragment,
        "fragment_len": f_fragment_len,
        "has_port": f_has_port,
        "nonstandard_port": f_nonstandard_port,
        "is_https": f_is_https,
        "redirect_hops": f_redirect_hops,
    }