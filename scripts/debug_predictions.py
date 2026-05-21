"""Diagnostic: hybrid ML + rules + SHAP prediction on a curated URL set."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings  # noqa: E402
from src.explainability.shap_explainer import ShapExplainer  # noqa: E402
from src.models.ensemble import predict  # noqa: E402
from src.models.lgbm_model import load_model  # noqa: E402


DIAGNOSTIC_URLS = [
    ("LEGIT_homepage", "https://www.google.com/"),
    ("LEGIT_apex", "https://github.com/anthropics"),
    ("LEGIT_deep_path", "https://stackoverflow.com/questions/12345"),
    ("LEGIT_simple", "https://example.com/"),
    ("LEGIT_youtube", "https://youtube.com/watch?v=abc123"),
    ("LEGIT_wiki", "https://en.wikipedia.org/wiki/Python"),
    ("PHISH_typosquat", "https://paypa1.com/login"),
    ("PHISH_brand_stuff", "http://accounts-secure.paypal-login.evil.xyz/verify"),
    ("PHISH_tld_abuse", "https://account-verify-update.tk/"),
    ("PHISH_ip_host", "http://203.0.113.5/secure-bank-login"),
    ("PHISH_at_attack", "https://paypal.com@evil.example.com/login"),
    ("PHISH_brand_hyphen", "https://apple-id-support-login.com/verify"),
]


def main() -> int:
    lgbm = load_model(settings.models_dir / "lgbm_v1.joblib")
    shap_explainer = ShapExplainer(lgbm)

    print(f"\n{'Label':<22} {'Final %':>8} {'Verdict':>10}  Rules / SHAP explanations")
    print("=" * 110)

    for label, url in DIAGNOSTIC_URLS:
        result = predict(url, lgbm, shap_explainer)
        verdict = "PHISH" if result.prediction == "phishing" else "safe"
        print(
            f"\n{label:<22} {result.phishing_probability:>7.2f}% "
            f"{verdict:>10}  rules: {','.join(result.triggered_rules) or '-'}"
        )
        for exp in result.explanations:
            print(f"    - {exp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())