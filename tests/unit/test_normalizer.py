"""
Tests for the URL normalization pipeline.

Each test corresponds to a specific phishing evasion technique from
the threat model. New evasion techniques should be added here as
regression tests before any code changes.
"""
from __future__ import annotations

import pytest

from src.normalization.url_normalizer import normalize
from src.utils.exceptions import URLValidationError


class TestInputValidation:

    def test_empty_url_rejected(self):
        with pytest.raises(URLValidationError):
            normalize("")

    def test_whitespace_only_rejected(self):
        with pytest.raises(URLValidationError):
            normalize("   ")

    def test_non_string_rejected(self):
        with pytest.raises(URLValidationError):
            normalize(None)  # type: ignore[arg-type]

    def test_excessive_length_rejected(self):
        long_url = "http://example.com/" + ("a" * 5000)
        with pytest.raises(URLValidationError):
            normalize(long_url)


class TestSchemeHandling:

    def test_missing_scheme_defaulted(self):
        result = normalize("example.com")
        assert result.scheme == "http"
        assert "no_scheme_provided_defaulted_to_http" in result.warnings

    def test_scheme_lowercased(self):
        result = normalize("HTTPS://Example.COM")
        assert result.scheme == "https"
        assert result.hostname == "example.com"

    def test_suspicious_scheme_flagged(self):
        result = normalize("javascript:alert(1)")
        assert result.had_suspicious_scheme
        assert any("suspicious_scheme" in w for w in result.warnings)


class TestEmbeddedCredentialsAttack:
    """The @-symbol attack: https://trusted.com@evil.com"""

    def test_credentials_stripped(self):
        result = normalize("https://paypal.com@evil.example.com/login")
        # After stripping, host is evil.example.com (the real destination)
        assert result.hostname == "evil.example.com"
        assert result.had_embedded_credentials
        assert result.stripped_credentials == "paypal.com"

    def test_no_credentials_no_flag(self):
        result = normalize("https://example.com/login")
        assert not result.had_embedded_credentials
        assert result.stripped_credentials is None


class TestPercentEncodingObfuscation:

    def test_single_encoding_decoded(self):
        # /%6c%6f%67%69%6e == /login
        result = normalize("https://example.com/%6c%6f%67%69%6e")
        assert result.path == "/login"
        assert result.encoding.decode_iterations >= 1

    def test_double_encoding_detected(self):
        # %25%36%63 -> %6c -> l
        result = normalize("https://example.com/%2525%2536%2563ogin")
        assert result.encoding.had_double_encoding
        assert result.encoding.decode_iterations >= 2


class TestPunycodeAndHomograph:

    def test_punycode_decoded(self):
        # xn--80ak6aa92e.com decodes to аррӏе.com (Cyrillic look-alikes for apple)
        result = normalize("https://xn--80ak6aa92e.com")
        assert result.unicode.had_punycode

    def test_pure_ascii_not_flagged_as_punycode(self):
        result = normalize("https://example.com")
        assert not result.unicode.had_punycode
        assert not result.unicode.is_mixed_script

    def test_mixed_script_detected(self):
        # 'gооgle' with Cyrillic 'о' (U+043E) mixed with Latin
        mixed = "https://g\u043e\u043egle.com"
        result = normalize(mixed)
        assert result.unicode.is_mixed_script
        assert "mixed_script_hostname" in result.warnings


class TestCaseNormalization:

    def test_host_lowercased_path_preserved(self):
        result = normalize("HTTPS://EXAMPLE.COM/Login/Page")
        assert result.hostname == "example.com"
        # Path case is significant on many servers; preserve it.
        assert result.path == "/Login/Page"


class TestDomainExtraction:

    def test_simple_tld(self):
        result = normalize("https://www.example.com/path")
        assert result.domain_parts.domain == "example"
        assert result.domain_parts.suffix == "com"
        assert result.domain_parts.subdomain == "www"
        assert result.domain_parts.registered_domain == "example.com"

    def test_compound_tld(self):
        result = normalize("https://shop.example.co.uk")
        assert result.domain_parts.domain == "example"
        assert result.domain_parts.suffix == "co.uk"
        assert result.domain_parts.registered_domain == "example.co.uk"

    def test_deep_subdomain(self):
        result = normalize("https://login.secure.account.example.com")
        assert result.domain_parts.subdomain == "login.secure.account"


class TestCanonicalForm:

    def test_canonical_includes_all_parts(self):
        result = normalize("https://Example.com:8443/path?q=1#frag")
        assert result.canonical.startswith("https://example.com:8443/path")
        assert "?q=1" in result.canonical
        assert "#frag" in result.canonical

    def test_default_path_is_root(self):
        result = normalize("https://example.com")
        assert result.path == "/"
