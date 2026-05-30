#!/usr/bin/env python3
"""CLI wrapper around :func:`src.dataset.ensure_authoridentification_dataset`.

Run from the project root::

    python data/fetch_dataset.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataset import (  # noqa: E402
    authoridentification_clone_path,
    ensure_authoridentification_dataset,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Ensure the AuthorIdentification GitHub repo is cloned under data/."
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"Clone directory (default: {authoridentification_clone_path()})",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Remove a broken directory and clone again.",
    )
    args = ap.parse_args()
    try:
        root = ensure_authoridentification_dataset(
            clone_root=args.output,
            force=args.force,
        )
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"OK: {root / 'Dataset'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
