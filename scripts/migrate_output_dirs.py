#!/usr/bin/env python3
"""One-time migration: move ./output/* into ./out/* to unify outputs.
Safe: creates out/ if missing; does not overwrite existing files.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "output"
DST = ROOT / "out"


def main():
    if not SRC.exists():
        print("No ./output directory found; nothing to migrate.")
        return
    DST.mkdir(parents=True, exist_ok=True)
    moved = 0
    for p in SRC.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(SRC)
        target = DST / rel
        if target.exists():
            print(f"Skip (exists): {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Move: {p} -> {target}")
        shutil.move(str(p), str(target))
        moved += 1
    print(f"Migration complete. Files moved: {moved}")


if __name__ == "__main__":
    main()
