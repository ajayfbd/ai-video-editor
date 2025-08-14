#!/usr/bin/env python3
"""
Cleanup utility to remove generated artifacts and temporary files.
- Deletes out/ (optional), temp/, .pytest_cache/, and common cache dirs.
- Keeps curated docs/reports.

Usage:
  python scripts/cleanup_workspace.py [--all]

By default, does not remove out/. Pass --all to also remove out/.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES = [
    ROOT / "temp",
    ROOT / ".pytest_cache",
    ROOT / ".cache",
    ROOT / ".whisper",
]

OUT_DIR = ROOT / "out"


def rm(path: Path):
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="Also remove out/ directory (generated artifacts)")
    args = ap.parse_args()

    for p in CANDIDATES:
        print(f"Removing {p} ...")
        rm(p)

    if args.all:
        print(f"Removing {OUT_DIR} ...")
        rm(OUT_DIR)
    else:
        print(f"Skipping {OUT_DIR} (use --all to remove)")

    print("Cleanup completed.")


if __name__ == "__main__":
    main()
