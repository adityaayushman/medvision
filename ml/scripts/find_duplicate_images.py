"""Detect images that appear in BOTH of two image sets -- exact copies and
re-encoded/resized near-copies.

Why this exists: public Kaggle "compilations" of the same medical imaging
domain frequently re-package each other. The obvious second brain-tumour
dataset (masoudnickparvar/brain-tumor-mri-dataset) is documented as a mix of
figshare + SARTAJ + Br35H, and SARTAJ is the very set this project trains
on. Evaluating on it naively would score partly on memorised training images
and report an inflated "generalization" number -- worse than not running the
experiment at all. So for a cross-dataset claim, deduplication isn't a
nicety, it's the experiment.

Two detectors, because one isn't enough:
  * MD5 of raw bytes  -> catches byte-identical copies only.
  * 64-bit dHash      -> catches the same image re-encoded, resized, or
                         recompressed, which is exactly how a picture ends up
                         in two different compilations. Resize to 9x8 grey,
                         compare horizontally adjacent pixels, pack to 64
                         bits; near-duplicate if Hamming distance <= --max-distance.

Usage:
    # sanity check: a set against itself must report 100% overlap
    python ml/scripts/find_duplicate_images.py --set-a ml/data/brain_mri/merged \
        --set-b ml/data/brain_mri/merged --out ml/artifacts/dedup_selftest.csv

    python ml/scripts/find_duplicate_images.py \
        --set-a ml/data/brain_mri/merged \
        --set-b ml/data/brain_mri_external/raw \
        --out ml/artifacts/brain_mri_overlap.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path) -> int | None:
    """64-bit difference hash. None if the image can't be read."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]          # (8, 8) bools
    bits = diff.flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hash_set(root: Path, label: str) -> Tuple[List[Path], List[str], np.ndarray]:
    paths = list_images(root)
    print(f"[{label}] {len(paths)} images under {root}")
    md5s: List[str] = []
    hashes: List[int] = []
    kept: List[Path] = []
    for i, p in enumerate(paths):
        d = dhash(p)
        if d is None:
            continue
        kept.append(p)
        md5s.append(md5_of(p))
        hashes.append(d)
        if (i + 1) % 2000 == 0:
            print(f"  [{label}] hashed {i + 1}/{len(paths)}", flush=True)
    return kept, md5s, np.array(hashes, dtype=np.uint64)


def _popcount64(x: np.ndarray) -> np.ndarray:
    """Vectorised popcount over uint64 via byte-wise table lookup -- keeps the
    full pairwise comparison in numpy instead of a Python loop."""
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    view = x.view(np.uint8).reshape(*x.shape, 8)
    return table[view].sum(axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-a", required=True, help="reference set (e.g. our training images)")
    ap.add_argument("--set-b", required=True, help="candidate external set to be cleaned")
    ap.add_argument("--out", required=True, help="CSV of colliding pairs")
    ap.add_argument("--max-distance", type=int, default=5,
                    help="Hamming distance <= this counts as a near-duplicate (default 5)")
    args = ap.parse_args()

    a_paths, a_md5, a_hash = hash_set(Path(args.set_a), "A")
    b_paths, b_md5, b_hash = hash_set(Path(args.set_b), "B")

    a_md5_index: Dict[str, int] = {}
    for i, m in enumerate(a_md5):
        a_md5_index.setdefault(m, i)

    rows = []
    exact_hits = 0
    near_hits = 0
    flagged_b = set()

    # Pairwise Hamming in chunks: (chunk_b x all_a) uint64 XOR -> popcount.
    CHUNK = 512
    for start in range(0, len(b_hash), CHUNK):
        chunk = b_hash[start:start + CHUNK]
        xor = chunk[:, None] ^ a_hash[None, :]
        dist = _popcount64(xor)
        best_idx = dist.argmin(axis=1)
        best_dist = dist[np.arange(len(chunk)), best_idx]

        for j, (bi, bd) in enumerate(zip(best_idx, best_dist)):
            b_i = start + j
            is_exact = b_md5[b_i] in a_md5_index
            if is_exact:
                exact_hits += 1
                flagged_b.add(str(b_paths[b_i]))
                rows.append({
                    "b_path": str(b_paths[b_i]),
                    "a_path": str(a_paths[a_md5_index[b_md5[b_i]]]),
                    "kind": "exact_md5", "hamming": 0,
                })
            elif bd <= args.max_distance:
                near_hits += 1
                flagged_b.add(str(b_paths[b_i]))
                rows.append({
                    "b_path": str(b_paths[b_i]),
                    "a_path": str(a_paths[int(bi)]),
                    "kind": "near_dhash", "hamming": int(bd),
                })
        print(f"  compared {min(start + CHUNK, len(b_hash))}/{len(b_hash)}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["b_path", "a_path", "kind", "hamming"])
        w.writeheader()
        w.writerows(rows)

    pct = 100.0 * len(flagged_b) / max(len(b_paths), 1)
    print(f"\nSet A: {len(a_paths)} images | Set B: {len(b_paths)} images")
    print(f"Exact (md5) collisions:     {exact_hits}")
    print(f"Near-duplicates (dHash<={args.max_distance}): {near_hits}")
    print(f"UNIQUE set-B images flagged: {len(flagged_b)}  ({pct:.2f}% of set B)")
    print(f"Set-B images that are genuinely unseen: {len(b_paths) - len(flagged_b)}")
    print(f"Pairs written -> {out}")


if __name__ == "__main__":
    main()
