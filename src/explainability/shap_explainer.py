"""
SHAP-based per-prediction feature attribution.

SHAP (SHapley Additive exPlanations) decomposes a model's prediction
for a single input into per-feature contributions. For tree models we
use TreeExplainer, which is exact (not approximate) and fast (~ms per
prediction).

User-facing explanations describe WHICH feature drove the decision in
plain language. We deliberately do NOT show raw SHAP magnitudes to
end users - raw SHAP values are log-odds units that can exceed 100
and read as nonsense outside of an ML context. Internal callers can
still access the raw contributions via the ShapExplanation dataclass.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import shap

from src.features.feature_pipeline import FEATURE_NAMES
from src.models.lgbm_model import TrainedLGBM
from src.utils.logger import get_logger


logger = get_logger(__name__)


# Suppress the noisy SHAP/LightGBM compatibility warning. Our code
# already handles both old and new return shapes from TreeExplainer.
warnings.filterwarnings(
    "ignore",
    message=".*LightGBM binary classifier with TreeExplainer.*",
)


# Per-feature templates: each function turns a feature's raw value
# into a human sentence. No raw SHAP magnitudes appear in user text.

def _t_url_len(v: float) -> str:
    return f"URL is {int(v)} characters long, which is unusually long."

def _t_host_len(v: float) -> str:
    return f"Hostname is {int(v)} characters long, suggesting obfuscation."

def _t_subdomain_count(v: float) -> str:
    return f"URL has {int(v)} subdomain levels, hiding the real domain."

def _t_count_hyphens(v: float) -> str:
    return f"URL contains {int(v)} hyphens, often used to chain words deceptively."

def _t_count_dots(v: float) -> str:
    return f"URL contains {int(v)} dots, indicating excessive subdomains."

def _t_count_at(v: float) -> str:
    return "URL contains an '@' symbol, a credential-injection deception pattern."

def _t_tld_risk(v: float) -> str:
    return f"Top-level domain has an elevated abuse risk."

def _t_brand_distance(v: float) -> str:
    if v == 0:
        return "Registered domain exactly matches a known brand (possible imitation)."
    if v <= 3:
        return f"Registered domain is only {int(v)} character(s) different from a known brand."
    return "Registered domain has some similarity to a known brand."

def _t_keyword_count(v: float) -> str:
    return f"URL contains {int(v)} phishing-associated keywords (such as login, verify, account)."

def _t_keyword_in_host(v: float) -> str:
    return f"Hostname contains {int(v)} phishing-associated keyword(s)."

def _t_keyword_in_path(v: float) -> str:
    return f"URL path contains {int(v)} phishing-associated keyword(s)."

def _t_url_entropy(v: float) -> str:
    return f"URL has high character entropy, suggesting random-generated content."

def _t_host_entropy(v: float) -> str:
    return f"Hostname has high character entropy, suggesting random subdomain generation."

def _t_is_ip_host(v: float) -> str:
    return "URL uses a raw IP address instead of a domain name."

def _t_had_punycode(v: float) -> str:
    return "URL uses internationalized (punycode) characters, possible homograph attack."

def _t_mixed_script(v: float) -> str:
    return "Hostname mixes characters from different writing systems (homograph indicator)."

def _t_invisible_chars(v: float) -> str:
    return "Hostname contains invisible Unicode characters used to spoof similar-looking domains."

def _t_double_encoded(v: float) -> str:
    return "URL is double-percent-encoded, a common obfuscation trick."

def _t_decode_iterations(v: float) -> str:
    if v <= 1:
        return "URL has only light percent-encoding (normal)."
    return f"URL required {int(v)} rounds of percent-decoding (heavy obfuscation)."

def _t_path_depth(v: float) -> str:
    return f"URL path has {int(v)} segments, deeper than typical legitimate URLs."

def _t_domain_hyphens(v: float) -> str:
    return f"Registered domain contains {int(v)} hyphen(s), unusual for legitimate brands."

def _t_contains_brand_name(v: float) -> str:
    return "URL contains a brand name in a misleading position (subdomain or hyphenated)."

def _t_is_typosquat_candidate(v: float) -> str:
    return "Domain has typosquat characteristics (close to a known brand)."

def _t_had_credentials(v: float) -> str:
    return "URL contained embedded user credentials (@-symbol deception pattern)."

def _t_suspicious_scheme(v: float) -> str:
    return "URL uses a non-web scheme (javascript:, data:, etc.)."

def _t_count_slashes(v: float) -> str:
    return f"URL contains {int(v)} slashes, suggesting deep path nesting."

def _t_special_ratio(v: float) -> str:
    return f"URL has a high proportion of special characters."

def _t_digit_ratio(v: float) -> str:
    return f"URL has many digits relative to letters, common in randomized URLs."

def _t_count_questionmarks(v: float) -> str:
    return f"URL contains {int(v)} question marks, indicating complex query strings."

def _t_query_max_value_len(v: float) -> str:
    return f"URL has a long query parameter value, potentially carrying obfuscated payload."

def _t_nonstandard_port(v: float) -> str:
    return "URL uses a non-standard port number, unusual for legitimate websites."


_FEATURE_TEMPLATES = {
    "url_len": _t_url_len,
    "host_len": _t_host_len,
    "subdomain_count": _t_subdomain_count,
    "count_hyphens": _t_count_hyphens,
    "count_dots": _t_count_dots,
    "count_at": _t_count_at,
    "count_slashes": _t_count_slashes,
    "count_questionmarks": _t_count_questionmarks,
    "tld_risk": _t_tld_risk,
    "brand_distance": _t_brand_distance,
    "keyword_count": _t_keyword_count,
    "keyword_in_host": _t_keyword_in_host,
    "keyword_in_path": _t_keyword_in_path,
    "url_entropy": _t_url_entropy,
    "host_entropy": _t_host_entropy,
    "is_ip_host": _t_is_ip_host,
    "had_punycode": _t_had_punycode,
    "mixed_script": _t_mixed_script,
    "invisible_chars": _t_invisible_chars,
    "double_encoded": _t_double_encoded,
    "decode_iterations": _t_decode_iterations,
    "path_depth": _t_path_depth,
    "domain_hyphens": _t_domain_hyphens,
    "contains_brand_name": _t_contains_brand_name,
    "is_typosquat_candidate": _t_is_typosquat_candidate,
    "had_credentials": _t_had_credentials,
    "suspicious_scheme": _t_suspicious_scheme,
    "special_ratio": _t_special_ratio,
    "digit_ratio": _t_digit_ratio,
    "query_max_value_len": _t_query_max_value_len,
    "nonstandard_port": _t_nonstandard_port,
}


@dataclass(frozen=True)
class ShapExplanation:
    """Per-feature contribution from SHAP for one prediction."""

    feature: str
    value: float
    contribution: float


class ShapExplainer:
    """Wraps a LightGBM TreeExplainer with a tidy interface."""

    def __init__(self, lgbm: TrainedLGBM) -> None:
        self._explainer = shap.TreeExplainer(lgbm.model)
        self._n_features = lgbm.n_features
        logger.info("shap_explainer_initialized", n_features=lgbm.n_features)

    def explain(self, X: np.ndarray) -> list[ShapExplanation]:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, got {X.shape[1]}."
            )

        raw = self._explainer.shap_values(X)
        if isinstance(raw, list):
            shap_values = raw[1] if len(raw) == 2 else raw[0]
        else:
            shap_values = raw

        contributions = np.asarray(shap_values[0]).ravel()
        values = np.asarray(X[0]).ravel()

        explanations = [
            ShapExplanation(
                feature=name,
                value=float(values[i]),
                contribution=float(contributions[i]),
            )
            for i, name in enumerate(FEATURE_NAMES)
        ]
        explanations.sort(key=lambda e: -abs(e.contribution))
        return explanations


def shap_explanations_to_strings(
    shap_results: list[ShapExplanation],
    top_n: int = 5,
    only_positive: bool = True,
) -> list[str]:
    """Convert SHAP results to clean human-readable strings.

    Features without a custom template are SKIPPED rather than rendered
    with technical jargon. This keeps user-facing explanations clean.
    Internal debugging can iterate the raw ShapExplanation list."""
    candidates = (
        [r for r in shap_results if r.contribution > 0]
        if only_positive else shap_results
    )
    out: list[str] = []
    for r in candidates:
        if len(out) >= top_n:
            break
        template = _FEATURE_TEMPLATES.get(r.feature)
        if template is None:
            continue
        sentence = template(r.value)
        if sentence not in out:
            out.append(sentence)
    return out