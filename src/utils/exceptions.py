"""
Custom exception hierarchy.

Use these instead of raising generic Exception. The API layer maps each
to the appropriate HTTP status code, and security audits become much
easier when failures are categorized.
"""


class PhishingDetectorError(Exception):
    """Base exception for all application errors."""


class URLValidationError(PhishingDetectorError):
    """URL failed input validation (malformed, too long, wrong scheme)."""


class URLNormalizationError(PhishingDetectorError):
    """URL could not be normalized (invalid encoding, broken punycode)."""


class FeatureExtractionError(PhishingDetectorError):
    """Feature extraction failed on an otherwise valid URL."""


class ModelInferenceError(PhishingDetectorError):
    """Model prediction failed (corrupt artifact, shape mismatch)."""


class ModelNotLoadedError(PhishingDetectorError):
    """Inference requested but model artifacts have not been loaded."""
