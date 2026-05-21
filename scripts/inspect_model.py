"""Inspect the trained model: which features carry the most weight,
and what values they take across diagnostic URLs."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings  # noqa: E402
from src.features.feature_pipeline import FEATURE_NAMES, extract_features  # noqa: E402
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
]


def main() -> int:
    m = load_model(settings.models_dir / "lgbm_v1.joblib")

    # Top 20 features by importance
    importances = sorted(
        zip(FEATURE_NAMES, m.model.feature_importances_),
        key=lambda x: -x[1],
    )
    top_names = [n for n, _ in importances[:20]]

    # Build a table: rows = URLs, cols = features
    print("=" * 120)
    print("TOP 20 FEATURES BY IMPORTANCE")
    print("=" * 120)
    for n, imp in importances[:20]:
        print(f"  {n:30s}  {imp}")
    print()

    print("=" * 120)
    print("PREDICTIONS AND FEATURE VALUES")
    print("=" * 120)
    header = f"{'feature':<28}"
    for label, _ in DIAGNOSTIC_URLS:
        header += f" {label[:16]:>17s}"
    print(header)
    print("-" * len(header))

    values_by_url: list[tuple[str, float, dict]] = []
    for label, url in DIAGNOSTIC_URLS:
        vec, _ = extract_features(url)
        prob = float(m.predict_proba(vec)[0])
        values_by_url.append((label, prob, {n: vec[FEATURE_NAMES.index(n)] for n in top_names}))

    # Print prediction row
    pred_row = f"{'>>> prediction %':<28}"
    for label, prob, _ in values_by_url:
        pred_row += f" {prob*100:>16.2f}%"
    print(pred_row)
    print("-" * len(header))

    # Print each feature row
    for fname in top_names:
        row = f"{fname:<28}"
        for label, _, vals in values_by_url:
            v = vals[fname]
            row += f" {v:>17.4f}"
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())