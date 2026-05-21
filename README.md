\---

title: Phishing URL Detection API

emoji: 🛡️

colorFrom: red

colorTo: blue

sdk: docker

app\_port: 7860

pinned: false

license: mit

\---



\# Phishing URL Detection API



A hybrid machine learning + rule-based system for real-time phishing URL detection.



\## Architecture



\- URL Normalization Engine: defends against Unicode homograph attacks, punycode abuse, percent-encoding obfuscation, embedded credentials (the @-symbol attack), and suspicious schemes - all before features are extracted.

\- Feature Engineering: 65 hand-engineered features across lexical, structural, and domain-level dimensions.

\- LightGBM Classifier: gradient-boosting model trained on a custom dataset (PhiUSIIL phishing + Tranco top-500k legit URLs with realistic path augmentation). Test accuracy \~98%, precision \~99%, ROC-AUC \~0.99.

\- Deterministic Rule Engine: 8 high-precision rules catching typosquats, brand stuffing in subdomains, IP-as-host, abusive TLDs, embedded credentials, suspicious schemes, brand-with-hyphens patterns, and mixed-script hostnames.

\- SHAP Explainability: per-feature contributions translated into plain-language reasons for every flagged URL.



\## API Endpoints



\- POST /predict - classify a single URL

\- POST /predict/batch - classify up to 100 URLs in one request

\- GET /health - liveness/readiness probe

\- GET /info - model metadata

\- GET /docs - interactive Swagger UI



\## Example



Request:

