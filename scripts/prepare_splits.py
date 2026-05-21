"""
One-shot script: load datasets, split, persist to data/processed/.

Run this once after downloading datasets. Subsequent training/evaluation
scripts read from data/processed/ directly without re-running the
expensive load+split pipeline.

Run from project root:
    python scripts/prepare_splits.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.logging_config import configure_logging  # noqa: E402
from config.settings import settings  # noqa: E402
from src.data.loaders import load_combined_dataset  # noqa: E402
from src.data.splitters import split_by_domain  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


logger = get_logger(__name__)


def main() -> int:
    configure_logging()
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Preparing train/val/test splits")
    print("=" * 60)
    print()

    df = load_combined_dataset()
    splits = split_by_domain(df)

    # Persist each split as Parquet for fast, type-safe reloading.
    # We use Parquet over CSV because:
    #   - Smaller files (~3x compression)
    #   - Preserves int vs object dtypes (no "label became float")
    #   - Faster to read (binary format, columnar layout)
    out_dir = settings.processed_data_dir
    paths = {
        "train": out_dir / "train.parquet",
        "val": out_dir / "val.parquet",
        "test": out_dir / "test.parquet",
    }
    for name, df_split in (("train", splits.train), ("val", splits.val),
                           ("test", splits.test)):
        df_split.to_parquet(paths[name], index=False)
        logger.info("split_persisted", split=name, path=str(paths[name]),
                    rows=len(df_split))

    print()
    print("Splits persisted to:")
    for name, path in paths.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {name}: {path} ({size_mb:.1f} MB)")
    print()
    print("Summary:")
    import json
    print(json.dumps(splits.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())