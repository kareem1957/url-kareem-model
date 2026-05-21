"""
Hybrid ML + Rules + Allowlist + SHAP combiner.

Decision order:
  1. Run normalization and feature extraction.
  2. Run the deterministic rule engine.
  3. If any high-severity rule fires (typosquat, brand-stuffing,
     embedded credentials, etc.) -> verdict is PHISHING regardless
     of anything else. Rules carry highest priority because they
     target unambiguous phishing patterns.
  4. Otherwise, if the registered domain is in the allowlist
     (Tranco top-N), short-circuit to LEGITIMATE with high
     confidence. This avoids ML false positives on well-known
     legitimate sites (github.com, stackoverflow.com, etc.).
  5. Otherwise, fall back to the LightGBM model probability.

This ordering matters: rules-then-allowlist-then-ML means a
github.com@evil.com URL (embedded credentials rule fires) still
gets flagged correctly. The allowlist NEVER overrides a fired
rule; rules NEVER bypass the allowlist for non-malicious URLs.
"""
from __future__ import annotations

from dataclasses import dataclass

from config.settings import settings
from src.explainability.allowlist import is_allowlisted
from src.explainability.rule_engine import (
    RuleEngineResult,
    evaluate_rules,
)
from src.explainability.shap_explainer import (
    ShapExplainer,
    shap_explanations_to_strings,
)
from src.features.feature_pipeline import extract_features
from src.models.lgbm_model import TrainedLGBM
from src.normalization.url_normalizer import NormalizedURL


@dataclass(frozen=True)
class PredictionResult:
    """Final prediction returned to the API."""

    url: str
    prediction: str
    phishing_probability: float
    legitimate_probability: float
    confidence_score: float
    risk_level: str
    explanations: tuple[str, ...]
    triggered_rules: tuple[str, ...]
    ml_probability: float


def _risk_level(prob: float) -> str:
    if prob >= 0.90:
        return "critical"
    if prob >= 0.70:
        return "high"
    if prob >= 0.40:
        return "medium"
    return "low"


def predict(
    url: str,
    lgbm: TrainedLGBM,
    shap_explainer: ShapExplainer | None = None,
) -> PredictionResult:
    """
    Full hybrid prediction pipeline.

    Args:
        url: input URL to classify.
        lgbm: trained LightGBM model.
        shap_explainer: optional SHAP explainer. If provided, SHAP-based
                        feature explanations are included for ML-driven
                        decisions.
    """
    vec, nurl = extract_features(url)

    ml_prob = float(lgbm.predict_proba(vec)[0])
    rules = evaluate_rules(nurl)

    # Decision priority 1: rules override everything (highest precision)
    if rules.is_phishing:
        final_prob = max(ml_prob, rules.max_severity)
        explanations: list[str] = list(rules.explanations)
        triggered_rules = tuple(r.rule_id for r in rules.triggered_rules)

    # Decision priority 2: allowlist short-circuits for known-good domains
    elif (
        settings.allowlist_enabled
        and is_allowlisted(nurl.domain_parts.registered_domain)
    ):
        # Force legitimate with high confidence. The ML probability is
        # preserved in the response for transparency, but the verdict
        # is driven by the allowlist.
        final_prob = min(ml_prob, 0.05)  # cap at 5% phishing
        explanations = []
        triggered_rules = ()

    # Decision priority 3: fall back to ML model
    else:
        final_prob = ml_prob
        explanations = list(rules.explanations)
        triggered_rules = tuple(r.rule_id for r in rules.triggered_rules)

        # Add SHAP-based explanations for ML-driven phishing decisions
        if (
            shap_explainer is not None
            and final_prob >= 0.5
            and not rules.is_phishing
        ):
            shap_results = shap_explainer.explain(vec)
            ml_explanations = shap_explanations_to_strings(
                shap_results, top_n=4, only_positive=True
            )
            if ml_explanations:
                explanations.extend(ml_explanations)
            else:
                explanations.append(
                    "Statistical model flagged this URL but no single "
                    "feature dominated; review the URL carefully."
                )

    prediction = "phishing" if final_prob >= 0.5 else "legitimate"
    confidence = abs(final_prob - 0.5) * 200.0

    return PredictionResult(
        url=url,
        prediction=prediction,
        phishing_probability=round(final_prob * 100.0, 2),
        legitimate_probability=round((1.0 - final_prob) * 100.0, 2),
        confidence_score=round(confidence, 2),
        risk_level=_risk_level(final_prob),
        explanations=tuple(explanations),
        triggered_rules=triggered_rules,
        ml_probability=round(ml_prob * 100.0, 2),
    )