"""
Dataset loaders.

After diagnosing a distribution artifact in PhiUSIIL's legitimate URLs
(predominantly bare apex/www homepages, causing the model to learn
"URL has a path -> phishing"), we restructure the dataset:

  - PHISHING URLs:    PhiUSIIL (verified, diverse, high quality)
  - LEGITIMATE URLs:  Tranco top 500k + synthetic-path augmentation

Tranco gives us bare domains across all sizes/categories of legit
sites. We then augment a portion with realistic paths (search queries,
documentation pages, repository paths, etc.) so the model learns that
legitimate URLs can have arbitrary paths too.

This approach trades raw dataset size for distribution fidelity. The
model trained on this data generalizes to real-world URLs like
github.com/anthropics that the original PhiUSIIL-derived model
catastrophically misclassified.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd

from config.settings import settings
from src.utils.logger import get_logger


logger = get_logger(__name__)


PHIUSIIL_CSV = (
    settings.raw_data_dir / "phiusiil" / "PhiUSIIL_Phishing_URL_Dataset.csv"
)
TRANCO_CSV = settings.raw_data_dir / "tranco" / "top-1m.csv"


# Realistic path patterns. Each is a plausible URL path on a real site.
# We intentionally include suspicious-looking-but-legitimate keywords
# (login, account, secure) so the model doesn't learn "keyword = phishing"
# as a shortcut.
REALISTIC_PATHS: tuple[str, ...] = (
    "/",
    "/about",
    "/about/team",
    "/contact",
    "/blog",
    "/blog/2024/announcing-our-new-product",
    "/news",
    "/news/category/tech",
    "/docs",
    "/docs/getting-started",
    "/docs/api/v2/reference",
    "/pricing",
    "/products",
    "/products/enterprise",
    "/services",
    "/help",
    "/help/faq",
    "/support",
    "/support/contact",
    "/login",
    "/signin",
    "/account",
    "/account/settings",
    "/profile",
    "/profile/edit",
    "/dashboard",
    "/settings",
    "/search",
    "/search?q=python+tutorial",
    "/search?q=best+laptop+2024",
    "/category/electronics",
    "/category/books/fiction",
    "/product/12345",
    "/product/wireless-headphones-bluetooth",
    "/article/how-to-learn-machine-learning",
    "/post/2024/03/15/new-features",
    "/users/alice",
    "/users/bob/repositories",
    "/repo/owner/project-name",
    "/repo/owner/project-name/issues/42",
    "/questions/12345/how-do-i-do-x",
    "/wiki/Article_Name",
    "/topics/machine-learning",
    "/tag/python",
    "/api/v1/users",
    "/api/v2/products?limit=20",
    "/download",
    "/download/version-2.0",
    "/checkout",
    "/cart",
    "/orders/recent",
    "/forum/general",
    "/community/welcome",
    "/events/2024",
    "/jobs",
    "/careers/engineering",
    "/privacy",
    "/terms",
    "/legal/privacy-policy",
    "/sitemap.xml",
    "/feed.rss",
    "/feed/all",
    "/page/2",
    "/archive/2023",
    "/posts/recent",
    "/courses/intro-to-cs",
    "/lesson/3/exercises",
)


_WWW_PREFIX_RE = re.compile(r"^(https?://)www\.", re.IGNORECASE)


def load_phiusiil_phishing_only(path: Path | None = None) -> pd.DataFrame:
    """
    Load ONLY the phishing URLs from PhiUSIIL.

    PhiUSIIL's legitimate URLs have a distribution artifact (bare-homepage
    bias) that corrupts the trained model. We drop them and use Tranco
    for legitimate examples instead.

    PhiUSIIL's phishing URLs are high quality - manually verified - so
    we keep them.

    PhiUSIIL labels in the raw CSV: 1 = legit, 0 = phishing.
    """
    path = path or PHIUSIIL_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"PhiUSIIL CSV not found at {path}. "
            "Run scripts/download_datasets.py first."
        )

    logger.info("phiusiil_phishing_loading", path=str(path))
    df = pd.read_csv(path, usecols=["URL", "label"])
    df = df.rename(columns={"URL": "url"})

    # Keep only phishing (label == 0 in raw -> phishing)
    df = df[df["label"] == 0].copy()
    df["label"] = 1  # convert to our convention: 1 = phishing

    df = df.dropna(subset=["url"])
    df = df[df["url"].str.strip().str.len() > 0]

    logger.info("phiusiil_phishing_loaded", rows=len(df))
    return df[["url", "label"]].reset_index(drop=True)


def load_tranco_with_paths(
    path: Path | None = None,
    top_n: int = 500_000,
    path_fraction: float = 0.6,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load Tranco top-N domains as legitimate URLs, augmenting a fraction
    with realistic paths.

    Args:
        top_n: how many top-ranked Tranco domains to use (default 500k)
        path_fraction: fraction of domains to augment with a path
                       (default 60%). The remaining 40% are kept as
                       bare apex URLs so both forms are represented.

    The path-augmented URLs use realistic paths from REALISTIC_PATHS,
    including some that contain suspicious-looking keywords (login,
    account). This is deliberate: phishing keyword features should be
    ONE signal among many, not a death-sentence shortcut for the model.
    """
    path = path or TRANCO_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Tranco CSV not found at {path}. "
            "Run scripts/download_datasets.py first."
        )

    logger.info(
        "tranco_loading",
        path=str(path),
        top_n=top_n,
        path_fraction=path_fraction,
    )

    df = pd.read_csv(
        path,
        header=None,
        names=["rank", "domain"],
        nrows=top_n,
    )
    df = df.dropna(subset=["domain"])
    df["domain"] = df["domain"].astype(str)

    # Mix schemes: about 70% https, 30% http. Most modern sites use https,
    # but plenty of legitimate sites still serve http. The model should not
    # learn "http = phishing".
    rng = random.Random(random_state)
    schemes = [
        "https" if rng.random() < 0.70 else "http"
        for _ in range(len(df))
    ]

    # Mix www-prefix: about 35% get www. prepended. Real-world distribution
    # of www-vs-apex is roughly 30-40% www; this matches that.
    www_flags = [rng.random() < 0.35 for _ in range(len(df))]

    # Path augmentation: about path_fraction of URLs get a non-trivial path.
    path_flags = [rng.random() < path_fraction for _ in range(len(df))]
    chosen_paths = [
        rng.choice(REALISTIC_PATHS) if has_path else "/"
        for has_path in path_flags
    ]

    urls: list[str] = []
    for domain, scheme, use_www, p in zip(
        df["domain"], schemes, www_flags, chosen_paths
    ):
        host = f"www.{domain}" if use_www else domain
        urls.append(f"{scheme}://{host}{p}")

    out = pd.DataFrame({"url": urls, "label": 0})
    logger.info(
        "tranco_loaded_with_paths",
        rows=len(out),
        with_path=sum(path_flags),
        with_www=sum(www_flags),
    )
    return out


def load_combined_dataset(
    tranco_top_n: int = 500_000,
    path_fraction: float = 0.6,
) -> pd.DataFrame:
    """
    Build the training dataset:
      - Phishing from PhiUSIIL (verified)
      - Legitimate from Tranco with path augmentation (diverse)

    Returns a shuffled DataFrame with columns:
        url:   str
        label: int  (1 = phishing, 0 = legitimate)
    """
    phishing = load_phiusiil_phishing_only()
    legitimate = load_tranco_with_paths(
        top_n=tranco_top_n,
        path_fraction=path_fraction,
    )

    logger.info(
        "combining_datasets",
        phishing_rows=len(phishing),
        legitimate_rows=len(legitimate),
    )

    combined = pd.concat([phishing, legitimate], ignore_index=True)
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["url"], keep="first")
    deduped = before_dedup - len(combined)

    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

    logger.info(
        "dataset_combined",
        total_rows=len(combined),
        duplicates_removed=deduped,
        phishing=int(combined["label"].sum()),
        legitimate=int((combined["label"] == 0).sum()),
        imbalance_ratio=float(
            (combined["label"] == 0).sum() / max(combined["label"].sum(), 1)
        ),
    )
    return combined