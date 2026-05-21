---
title: Phishing URL Detection API
emoji: shield
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Phishing URL Detection API

A hybrid machine learning + rule-based system for real-time phishing URL detection.

## Architecture

- URL Normalization Engine: defends against Unicode homograph attacks, punycode abuse, percent-encoding obfuscation, embedded credentials (the @-symbol attack), and suspicious schemes - all before features are extracted.
- Feature Engineering: 65 hand-engineered features across lexical, structural, and domain-level dimensions.
- LightGBM Classifier: gradient-boosting model trained on a custom dataset.
- Deterministic Rule Engine: high-precision rules catching typosquats, brand stuffing, IP-as-host, abusive TLDs, and homograph attacks.
- SHAP Explainability: per-feature contributions translated into plain-language reasons.

## API Endpoints

- POST /predict - classify a single URL
- POST /predict/batch - classify up to 100 URLs in one request
- GET /health - liveness/readiness probe
- GET /info - model metadata
- GET /docs - interactive Swagger UI

## Built With

Python 3.11, FastAPI, LightGBM, SHAP, tldextract, structlog.
