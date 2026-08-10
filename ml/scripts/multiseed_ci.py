"""Mean + 95% confidence interval across independent seed runs of the same
config -- e.g. "3-way ensemble, evaluated at seeds 42/0/1/2/3".

Method: sample mean +/- t(0.975, df=n-1) * (sample_std / sqrt(n)). This is
the exact method already used (ad hoc, not previously scripted) to produce
the brain MRI ResNet50 5-seed result documented in evaluation-data.ts and
EXPERIMENTS.md -- verified below to reproduce it exactly, so every new
multi-seed claim in this project uses one shared, checked-in method instead
of a repeated by-hand calculation.

Usage:
    # explicit values
    python ml/scripts/multiseed_ci.py --values 0.8551,0.8245,0.8408,0.8571,0.8571

    # read a metric out of several metrics.json files
    python ml/scripts/multiseed_ci.py --metric accuracy \
        --metrics-json ml/artifacts/brain_mri/metrics.json \
                       ml/artifacts/brain_mri_resnet50_seed0/metrics.json \
                       ml/artifacts/brain_mri_resnet50_seed1/metrics.json \
                       ml/artifacts/brain_mri_resnet50_seed2/metrics.json \
                       ml/artifacts/brain_mri_resnet50_seed3/metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Sequence

# t-critical (two-tailed, 95%) for small df, used only if scipy isn't
# installed -- scipy.stats.t.ppf is preferred and exact for any df.
_T_TABLE_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042,
}


def _t_critical(df: int) -> float:
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except ImportError:
        if df in _T_TABLE_95:
            return _T_TABLE_95[df]
        closest = min(_T_TABLE_95, key=lambda k: abs(k - df))
        return _T_TABLE_95[closest]


def mean_ci(values: Sequence[float]) -> dict:
    n = len(values)
    if n < 2:
        raise ValueError("Need at least 2 seed runs to compute a CI.")
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(variance)
    sem = std / math.sqrt(n)
    t_crit = _t_critical(n - 1)
    margin = t_crit * sem
    return {
        "n": n, "mean": mean, "std": std, "sem": sem,
        "t_critical": t_crit, "margin": margin,
        "ci_low": mean - margin, "ci_high": mean + margin,
    }


def _load_values(args) -> List[float]:
    if args.values:
        return [float(v) for v in args.values.split(",") if v.strip()]
    if args.metrics_json:
        vals = []
        for path in args.metrics_json:
            data = json.loads(Path(path).read_text())
            if args.metric not in data:
                raise SystemExit(f"{path} has no key {args.metric!r} (keys: {sorted(data)})")
            vals.append(float(data[args.metric]))
        return vals
    raise SystemExit("Provide either --values or --metrics-json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--values", help="comma-separated metric values, one per seed")
    ap.add_argument("--metrics-json", nargs="+", help="metrics.json paths, one per seed")
    ap.add_argument("--metric", default="accuracy", help="key to read from each metrics.json (default: accuracy)")
    ap.add_argument("--pct", action="store_true", help="print mean/CI as a percentage instead of a fraction")
    args = ap.parse_args()

    values = _load_values(args)
    result = mean_ci(values)

    scale = 100 if args.pct else 1
    suffix = "%" if args.pct else ""
    print(f"n = {result['n']}, values = {values}")
    print(f"mean = {result['mean']*scale:.4f}{suffix}  (std={result['std']*scale:.4f}, sem={result['sem']*scale:.4f})")
    print(f"t_critical(df={result['n']-1}) = {result['t_critical']:.3f}")
    print(f"95% CI: [{result['ci_low']*scale:.4f}{suffix}, {result['ci_high']*scale:.4f}{suffix}]  (margin ±{result['margin']*scale:.4f}{suffix})")


if __name__ == "__main__":
    main()
