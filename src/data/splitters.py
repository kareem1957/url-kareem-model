"""
Train/validation/test splitters.

We use a DOMAIN-AWARE split rather than a random split. Naive random
splitting leaks signal: if a phishing campaign uses 50 distinct paths
on the same domain, splitting per-URL puts some paths in train and
some in test, letting the model memorize the domain instead of
learning generalizable URL patterns. Real-world performance then
collapses on novel domains.

GroupShuffleSplit keeps all URLs from a single registered domain in
the same split. The model is forced to generalize across domains,
which is what we actually need in production.

We also verify that each split keeps a reasonable phishing/legit
ratio. Pure group splitting without stratification can produce wild
imbalances if a few mega-domains dominate. We log the split balance
and let the caller decide whether to re-shuffle.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.normalization.url_normalizer import normalize
from src.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class DatasetSplits:
    """Container for the three splits."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> dict:
        """Return a dict summary of all three splits for logging."""
        out = {}
        for name, df in (("train", self.train), ("val", self.val),
                         ("test", self.test)):
            out[name] = {
                "n": len(df),
                "phishing": int(df["label"].sum()),
                "legitimate": int((df["label"] == 0).sum()),
                "phishing_ratio": float(df["label"].mean()) if len(df) else 0.0,
            }
        return out


def _registered_domain(url: str) -> str:
    """
    Extract the registered domain for a URL. Used as the grouping key.

    Falls back to the raw URL string on normalization failure so the
    row still gets a group (its own, trivially).
    """
    try:
        nurl = normalize(url)
        return nurl.domain_parts.registered_domain or url
    except Exception:
        return url


def split_by_domain(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> DatasetSplits:
    """
    Domain-grouped train/val/test split.

    Procedure (two-stage):
      1. Split off the test set: (train+val) vs test
      2. Split train+val into train vs val
    Both stages use GroupShuffleSplit on registered_domain so a domain
    never appears in more than one split.

    Args:
        df: must have 'url' and 'label' columns.
        val_size, test_size: fractions of the full dataset.
        random_state: fixed for reproducibility.

    Returns DatasetSplits.
    """
    if not 0 < val_size < 1 or not 0 < test_size < 1:
        raise ValueError("val_size and test_size must be in (0, 1)")
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0")

    logger.info(
        "split_starting",
        total=len(df),
        val_size=val_size,
        test_size=test_size,
    )

    df = df.reset_index(drop=True).copy()
    logger.info("extracting_registered_domains")
    df["_group"] = df["url"].map(_registered_domain)

    # Stage 1: separate the test set
    gss_test = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    trainval_idx, test_idx = next(
        gss_test.split(df, groups=df["_group"])
    )
    df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # Stage 2: split train_val into train vs val.
    # val_size_relative is the fraction of trainval that becomes val:
    #   val_size / (1 - test_size)
    val_size_relative = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size_relative,
        random_state=random_state,
    )
    train_idx, val_idx = next(
        gss_val.split(df_trainval, groups=df_trainval["_group"])
    )
    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
    df_val = df_trainval.iloc[val_idx].reset_index(drop=True)

    # Drop the helper column before returning
    for split in (df_train, df_val, df_test):
        split.drop(columns=["_group"], inplace=True)

    splits = DatasetSplits(train=df_train, val=df_val, test=df_test)

    summary = splits.summary()
    logger.info("split_complete", **{f"{k}_{kk}": vv
                                     for k, v in summary.items()
                                     for kk, vv in v.items()})

    # Sanity warning if any split has wildly imbalanced class ratio
    overall_ratio = float(df["label"].mean())
    for name in ("train", "val", "test"):
        ratio = summary[name]["phishing_ratio"]
        if abs(ratio - overall_ratio) > 0.10:
            logger.warning(
                "class_ratio_drift_in_split",
                split=name,
                expected_ratio=overall_ratio,
                actual_ratio=ratio,
            )

    return splits