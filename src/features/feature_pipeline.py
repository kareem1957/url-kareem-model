"""
Unified feature extraction pipeline.

This module is the single entry point for converting a URL into a
numerical feature vector for the model. It:

  1. Calls all sub-extractors (lexical, structural, domain)
  2. Merges them into one dict (rejecting duplicate keys)
  3. Returns features in a STABLE, SORTED order

The order matters because ML models learn weights tied to specific
input positions. If feature #5 is "url_len" at training time but
"tld_risk" at inference time, the model produces silent garbage.

The FEATURE_NAMES tuple is the source of truth. Both training and
inference import from here, guaranteeing alignment.
"""
from __future__ import annotations

import numpy as np

from src.features.domain import extract_domain_features
from src.features.lexical import extract_lexical_features
from src.features.structural import extract_structural_features
from src.normalization.url_normalizer import NormalizedURL, normalize
from src.utils.exceptions import FeatureExtractionError


def _merge_no_duplicates(
    *dicts: dict[str, float],
) -> dict[str, float]:
    """Merge dicts; raise if any key appears in more than one."""
    merged: dict[str, float] = {}
    for d in dicts:
        overlap = merged.keys() & d.keys()
        if overlap:
            raise FeatureExtractionError(
                f"Duplicate feature names across extractors: {overlap}. "
                "Each feature must live in exactly one extractor."
            )
        merged.update(d)
    return merged


def _compute_all_features(nurl: NormalizedURL) -> dict[str, float]:
    """Run every sub-extractor and merge results."""
    return _merge_no_duplicates(
        extract_lexical_features(nurl),
        extract_structural_features(nurl),
        extract_domain_features(nurl),
    )


# Compute the canonical feature ordering ONCE at import time, using a
# benign reference URL. This freezes the feature schema for the entire
# application lifetime. If a new feature is added to any extractor,
# the order updates automatically here (sorted alphabetically).
_REFERENCE_URL = "https://example.com/"
try:
    _reference_features = _compute_all_features(normalize(_REFERENCE_URL))
    FEATURE_NAMES: tuple[str, ...] = tuple(sorted(_reference_features.keys()))
except Exception as exc:
    raise RuntimeError(
        f"Failed to compute reference feature schema: {exc}"
    ) from exc


N_FEATURES: int = len(FEATURE_NAMES)


def extract_features(url: str) -> tuple[np.ndarray, NormalizedURL]:
    """
    Extract features from a raw URL string.

    Returns:
        (feature_vector, normalized_url)
        - feature_vector is a 1-D numpy array of shape (N_FEATURES,)
          in the order defined by FEATURE_NAMES.
        - normalized_url is the NormalizedURL for downstream use
          (explainability, logging, etc.).

    Raises:
        URLValidationError: input failed validation.
        URLNormalizationError: normalization stage failed.
        FeatureExtractionError: a sub-extractor raised unexpectedly.
    """
    nurl = normalize(url)
    try:
        feature_dict = _compute_all_features(nurl)
    except FeatureExtractionError:
        raise
    except Exception as exc:
        raise FeatureExtractionError(
            f"Sub-extractor failed: {exc}"
        ) from exc

    if set(feature_dict.keys()) != set(FEATURE_NAMES):
        missing = set(FEATURE_NAMES) - set(feature_dict.keys())
        extra = set(feature_dict.keys()) - set(FEATURE_NAMES)
        raise FeatureExtractionError(
            f"Feature schema drift detected. Missing: {missing}. "
            f"Extra: {extra}"
        )

    vector = np.array(
        [feature_dict[name] for name in FEATURE_NAMES],
        dtype=np.float32,
    )
    return vector, nurl


def extract_features_batch(urls: list[str]) -> tuple[np.ndarray, list[NormalizedURL]]:
    """
    Vectorized extraction for a batch of URLs.

    Returns:
        (feature_matrix, normalized_urls)
        - feature_matrix shape: (len(urls), N_FEATURES)
        - normalized_urls is the list of NormalizedURL objects.

    URLs that fail extraction are skipped; check len(normalized_urls)
    against len(urls) to detect dropouts.
    """
    vectors: list[np.ndarray] = []
    nurls: list[NormalizedURL] = []
    for url in urls:
        try:
            vec, nurl = extract_features(url)
            vectors.append(vec)
            nurls.append(nurl)
        except Exception:
            # Silent skip: bad inputs are logged at the API layer.
            continue
    if not vectors:
        return np.empty((0, N_FEATURES), dtype=np.float32), []
    return np.vstack(vectors), nurls