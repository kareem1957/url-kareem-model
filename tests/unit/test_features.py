"""
Tests for the feature extraction pipeline.

We test:
  1. Shape and dtype contracts (the model depends on these)
  2. Deterministic feature ordering (training/inference alignment)
  3. Specific feature values on known-adversarial URLs
  4. Robustness on edge-case URLs (empty path, IP host, etc.)
"""
from __future__ import annotations

import numpy as np
import pytest

from src.features.feature_pipeline import (
    FEATURE_NAMES,
    N_FEATURES,
    extract_features,
    extract_features_batch,
)


class TestFeatureContract:
    """The model trusts these guarantees. They must hold."""

    def test_returns_correct_shape(self):
        vec, _ = extract_features("https://example.com/")
        assert vec.shape == (N_FEATURES,)

    def test_returns_float32(self):
        vec, _ = extract_features("https://example.com/")
        assert vec.dtype == np.float32

    def test_feature_names_are_sorted(self):
        assert list(FEATURE_NAMES) == sorted(FEATURE_NAMES)

    def test_feature_names_are_unique(self):
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))

    def test_no_nan_or_inf(self):
        vec, _ = extract_features("https://example.com/")
        assert not np.isnan(vec).any()
        assert not np.isinf(vec).any()


class TestAdversarialDetection:
    """Each test verifies the pipeline catches a specific attack."""

    def _feat(self, url: str, name: str) -> float:
        vec, _ = extract_features(url)
        idx = FEATURE_NAMES.index(name)
        return float(vec[idx])

    def test_typosquat_is_flagged(self):
        # paypa1 (digit 1 swapped for letter l) is one edit from "paypal"
        assert self._feat("https://paypa1.com/", "brand_distance") == 1.0
        assert self._feat("https://paypa1.com/", "is_typosquat_candidate") == 1.0

    def test_legitimate_brand_not_flagged_as_typosquat(self):
        # paypal.com exact match -> distance 0 -> NOT typosquat
        assert self._feat("https://paypal.com/", "brand_distance") == 0.0
        assert self._feat("https://paypal.com/", "is_typosquat_candidate") == 0.0

    def test_risky_tld_scored_high(self):
        # .tk is one of the most-abused TLDs
        assert self._feat("https://account-verify.tk/", "tld_risk") >= 0.95

    def test_safe_tld_scored_zero(self):
        assert self._feat("https://example.com/", "tld_risk") == 0.0

    def test_excessive_subdomains_detected(self):
        url = "https://a.b.c.d.e.f.example.com/"
        # subdomain "a.b.c.d.e.f" has 6 dot-separated labels
        assert self._feat(url, "subdomain_count") == 6.0

    def test_suspicious_keywords_counted(self):
        # login, verify, account all appear
        url = "https://example.com/login/verify/account"
        assert self._feat(url, "keyword_count") >= 3.0

    def test_hyphenated_domain_flagged(self):
        url = "https://apple-id-support.com/"
        assert self._feat(url, "domain_hyphens") == 2.0
        assert self._feat(url, "domain_has_hyphen") == 1.0

    def test_contains_brand_name_substring(self):
        # "paypal" inside a non-brand domain is the substring trick
        assert self._feat("https://paypal-secure.evil.com/", "contains_brand_name") == 1.0

    def test_embedded_credentials_detected(self):
        url = "https://paypal.com@evil.example.com/login"
        assert self._feat(url, "had_credentials") == 1.0

    def test_punycode_detected(self):
        # IDN test case (genuine punycode)
        url = "https://xn--80ak6aa92e.com/"
        assert self._feat(url, "had_punycode") == 1.0

    def test_double_encoding_detected(self):
        # %25%36%63 decodes to %6c which decodes to l
        url = "https://example.com/%2525%2536%2563ogin"
        assert self._feat(url, "double_encoded") == 1.0


class TestEdgeCases:

    def test_minimal_url(self):
        vec, _ = extract_features("http://a.io")
        assert vec.shape == (N_FEATURES,)
        assert not np.isnan(vec).any()

    def test_url_with_only_path(self):
        vec, _ = extract_features("https://example.com/")
        assert vec.shape == (N_FEATURES,)

    def test_url_with_query_string(self):
        vec, _ = extract_features("https://example.com/?a=1&b=2&c=3")
        assert vec.shape == (N_FEATURES,)

    def test_url_with_fragment(self):
        vec, _ = extract_features("https://example.com/page#section")
        assert vec.shape == (N_FEATURES,)


class TestBatchExtraction:

    def test_batch_returns_matrix(self):
        urls = [
            "https://example.com/",
            "https://google.com/",
            "https://github.com/",
        ]
        matrix, nurls = extract_features_batch(urls)
        assert matrix.shape == (3, N_FEATURES)
        assert len(nurls) == 3

    def test_batch_skips_invalid_urls(self):
        urls = ["https://example.com/", "", "https://github.com/"]
        matrix, nurls = extract_features_batch(urls)
        # The empty URL is skipped; 2 valid URLs remain
        assert matrix.shape == (2, N_FEATURES)
        assert len(nurls) == 2

    def test_empty_batch(self):
        matrix, nurls = extract_features_batch([])
        assert matrix.shape == (0, N_FEATURES)
        assert nurls == []