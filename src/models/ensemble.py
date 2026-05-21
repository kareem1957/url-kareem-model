"""
Hybrid ML + Rules + SHAP combiner.

Given a URL, this module:
  1. Normalizes it
  2. Extracts features and gets the LightGBM probability
  3. Runs the deterministic rule engine
  4. Computes SHAP attributions for the ML decision
  5. Combines everything into a final verdict with rich explanations

Explanation policy:
  - Rule-based explanations come first (highest precision, human-written)
  - SHAP-based explanations fill the gap when no rule fires but ML
    still flagged the URL, telling the user WHICH features drove
    the model's decision
  - If neither rule nor strong ML signal exists (legitimate URL),
    explanations are empty
"""
from __future__ import annotations

from dataclasses import dataclass

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
                        decisions. If None, only rule explanations are
                        used (faster but less informative).
    """
    vec, nurl = extract_features(url)

    ml_prob = float(lgbm.predict_proba(vec)[0])
    rules = evaluate_rules(nurl)

    if rules.is_phishing:
        final_prob = max(ml_prob, rules.max_severity)
    else:
        final_prob = ml_prob

    # Compose explanations:
    # 1) Rule explanations (always included if rules fired)
    # 2) SHAP explanations (included when ML flagged AND no rule fired,
    #    OR when both fired and we want richer info)
    explanations: list[str] = list(rules.explanations)

    needs_shap = (
        shap_explainer is not None
        and final_prob >= 0.5
        and not rules.is_phishing
    )
    if needs_shap:
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
        triggered_rules=tuple(r.rule_id for r in rules.triggered_rules),
        ml_probability=round(ml_prob * 100.0, 2),
    )