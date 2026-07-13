"""Download a shortlisted Kaggle dataset into ml/data/.

Requires Kaggle API credentials — see docs/KAGGLE_SETUP.md. Only the
Kaggle-hosted datasets in the registry can be fetched here; credentialed sets
(VinDr, CheXpert) must be requested through their own portals.

    python ml/scripts/download_data.py --dataset rsna_pneumonia
    python ml/scripts/download_data.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from medchron.data import REGISTRY, get_spec

# registry key -> Kaggle dataset ref (owner/slug)
KAGGLE_REFS = {
    "rsna_pneumonia": "parin30/rsna-pneumonia-detection",
    "nih_cxr14": "nih-chest-xrays/data",
    "montgomery_shenzhen": "iamtapendu/chest-x-ray-lungs-segmentation",
}


def _authenticate():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("kaggle is not installed. Run: pip install kaggle")
    try:
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as exc:  # missing / invalid kaggle.json
        sys.exit(
            f"Kaggle authentication failed: {exc}\n"
            "Set up credentials first — see docs/KAGGLE_SETUP.md"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Download a shortlisted Kaggle dataset")
    ap.add_argument("--dataset", default="rsna_pneumonia", choices=sorted(KAGGLE_REFS))
    ap.add_argument("--out", default=None, help="target dir (default ml/data/<dataset>)")
    ap.add_argument("--list", action="store_true", help="list downloadable datasets and exit")
    args = ap.parse_args()

    if args.list:
        print("Kaggle-downloadable datasets:\n")
        for key, ref in KAGGLE_REFS.items():
            spec = get_spec(key)
            print(f"  {key:22s} -> kaggle.com/datasets/{ref}   ({spec.approx_images}, {spec.task})")
        return

    ref = KAGGLE_REFS[args.dataset]
    out = Path(args.out or f"ml/data/{args.dataset}")
    out.mkdir(parents=True, exist_ok=True)

    api = _authenticate()
    print(f"Downloading {ref} -> {out} (this can take a while)...")
    api.dataset_download_files(ref, path=str(out), unzip=True, quiet=False)
    print(f"Done. Files in {out}:")
    for p in sorted(out.iterdir())[:20]:
        print("   ", p.name)


if __name__ == "__main__":
    main()
