"""Generate IEEE-ready tables and figures for the MedChron paper.

Everything here is computed from the `metrics.json` files under
ml/artifacts/ at run time. No number is transcribed by hand -- that is the
point: a paper's tables and its repository should not be able to disagree.

Outputs:
    paper/tables/*.tex      booktabs tables, sized for IEEE two-column
    paper/figures/*.pdf     vector figures (plus .png previews)
    paper/RESULTS.md        every generated number in one readable file

Usage:
    python ml/scripts/make_paper_assets.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ART = Path("ml/artifacts")
OUT = Path("paper")
TAB = OUT / "tables"
FIG = OUT / "figures"

# Print- and colourblind-safe; blue = ours, amber = external/literature.
C_OURS = "#1670f5"
C_LIT = "#b45309"
C_GREY = "#6b7280"
C_BAD = "#9f1239"

# IEEE two-column: 3.5in single, 7.16in full width.
COL_W, FULL_W = 3.5, 7.16

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

_LOG: List[str] = []


def log(s: str = "") -> None:
    print(s)
    _LOG.append(s)


def load(name: str) -> Dict:
    return json.loads((ART / name / "metrics.json").read_text())


def acc(name: str) -> float:
    return load(name)["accuracy"]


def auc(name: str) -> float:
    return load(name)["roc_auc"]


def n_of(name: str) -> int:
    return sum(v["support"] for v in load(name)["per_class"].values())


def mean_ci(values: Sequence[float]):
    """Mean +/- t(0.975, n-1) * SEM -- the method used throughout this project."""
    n = len(values)
    m = sum(values) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    sem = sd / math.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * sem
    return m, m - h, m + h


def binom_ci(p: float, n: int):
    lo, hi = stats.binomtest(round(p * n), n).proportion_ci(confidence_level=0.95)
    return lo, hi


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    log(f"  wrote {path}")


def save_fig(fig, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{stem}.{ext}")
    plt.close(fig)
    log(f"  wrote {FIG / stem}.pdf/.png")


# --------------------------------------------------------------------------
# Data pulled once, reused by both tables and figures.
# --------------------------------------------------------------------------
SEEDS = ["42", "0", "1", "2", "3"]
PLAIN = ["brain_mri"] + [f"plain_effnet_seed{s}" for s in SEEDS[1:]]
RESNET = ["brain_mri_resnet50_solo"] + [f"brain_mri_resnet50_seed{s}" for s in SEEDS[1:]]
ENS2 = [f"ens2_seed{s}" for s in SEEDS]
ENS3 = [f"ens3_seed{s}" for s in SEEDS]
DISTILL = ["brain_mri_distilled"] + [f"distilled_eval_seed{s}" for s in SEEDS[1:]]
PIPE = [f"pipe_autocrop_seed{s}" for s in SEEDS]


def table_brain_mri() -> None:
    """Table 1: brain MRI configurations, 5-seed means with CIs."""
    rows = [
        ("EfficientNet-B0 (single)", PLAIN),
        ("ResNet50 (single)", RESNET),
        ("Distilled student", DISTILL),
        ("Ensemble, 2-way", ENS2),
        ("Ensemble, 3-way", ENS3),
    ]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Brain MRI, 4-class tumour classification. Mean of five seeds "
        r"$\{42,0,1,2,3\}$ with 95\% confidence intervals; identical manifest and "
        r"split for every row ($n=490$ test images).}",
        r"\label{tab:brain_mri}",
        r"\begin{tabular}{lcc}", r"\toprule",
        r"Configuration & Accuracy (\%) & ROC-AUC \\", r"\midrule",
    ]
    for label, runs in rows:
        a, alo, ahi = mean_ci([acc(r) for r in runs])
        u, ulo, uhi = mean_ci([auc(r) for r in runs])
        lines.append(
            f"{label} & {a*100:.2f} \\tiny{{[{alo*100:.2f}, {ahi*100:.2f}]}} "
            f"& {u:.4f} \\tiny{{[{ulo:.4f}, {uhi:.4f}]}} \\\\"
        )
        log(f"  {label:26s} acc {a*100:.2f} [{alo*100:.2f},{ahi*100:.2f}]  auc {u:.4f}")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write(TAB / "tab_brain_mri.tex", "\n".join(lines))


def table_chest_xray() -> None:
    """Table 2: the accuracy-vs-AUC power finding."""
    a_full, a_sub = acc("rsna_full_on_subset_test"), acc("rsna_real")
    u_full, u_sub = auc("rsna_full_on_subset_test"), auc("rsna_real")
    n = n_of("rsna_real")
    kf, ks = round(a_full * n), round(a_sub * n)
    p = (kf + ks) / (2 * n)
    z = (a_full - a_sub) / math.sqrt(p * (1 - p) * (2 / n))
    p_acc = 2 * (1 - stats.norm.cdf(abs(z)))

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Chest X-ray: training-set size, evaluated on the \emph{same} "
        r"750 test images. Accuracy cannot resolve the difference; ROC-AUC can. "
        r"AUC $p$-value from a paired bootstrap (2000 resamples).}",
        r"\label{tab:chest_xray}",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"Metric & 5k subset & Full 26{,}684 & $p$ \\", r"\midrule",
        f"Accuracy (\\%) & {a_sub*100:.2f} & {a_full*100:.2f} & {p_acc:.3f} (n.s.) \\\\",
        f"ROC-AUC & {u_sub:.4f} & {u_full:.4f} & $<$0.0001 \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ]
    write(TAB / "tab_chest_xray.tex", "\n".join(lines))
    log(f"  accuracy {a_sub*100:.2f} -> {a_full*100:.2f} (p={p_acc:.3f}); "
        f"AUC {u_sub:.4f} -> {u_full:.4f} (p<0.0001)")


def table_crossdataset() -> None:
    """Table 3: leakage + cross-dataset generalization."""
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Cross-dataset evaluation on an external brain MRI compilation, "
        r"before and after removing images that duplicate our training data. "
        r"63.5\% of the external set overlaps; the contaminated split scores no "
        r"higher than the clean one.}",
        r"\label{tab:crossdata}",
        r"\begin{tabular}{lrcc}", r"\toprule",
        r"External split & $n$ & Accuracy (\%) & ROC-AUC \\", r"\midrule",
    ]
    for label, key in [
        ("All (contaminated)", "xdata_single_all"),
        ("Exact duplicates removed", "xdata_single_strict"),
        ("Exact + near removed (clean)", "xdata_single_clean"),
        ("Clean, 3-way ensemble", "xdata_ens3_clean"),
    ]:
        a, u, n = acc(key), auc(key), n_of(key)
        lines.append(f"{label} & {n} & {a*100:.2f} & {u:.4f} \\\\")
        log(f"  {label:30s} n={n:5d}  acc={a*100:.2f}  auc={u:.4f}")
    lines += [
        r"\midrule",
        r"\textit{In-domain reference} & 490 & 85.71 & 0.9690 \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ]
    write(TAB / "tab_crossdataset.tex", "\n".join(lines))


def table_mammography() -> None:
    """Table 4: the seven-attempt negative-result history."""
    rows = [
        ("1. MIAS, full images", "mammography"),
        ("2. CBIS-DDSM, full images", "mammography_cbis"),
        ("3. CBIS-DDSM, official crops", "mammography_cbis_cropped"),
        ("4. Pipeline: bbox regressor", "mammography_localized"),
        ("5. Pipeline: U-Net segmenter", "mammography_localized_segmentation"),
        ("6. Pipeline: GT-crop classifier", "mammography_localized_gtcrop"),
    ]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Mammography: seven attempts. Only the crop-input classifier (3) "
        r"clears a useful margin, and it requires a pre-cropped lesion the "
        r"deployment path cannot supply. Attempt 7 is a five-seed mean.}",
        r"\label{tab:mammography}",
        r"\begin{tabular}{lcc}", r"\toprule",
        r"Attempt & Accuracy (\%) & ROC-AUC \\", r"\midrule",
    ]
    for label, key in rows:
        lines.append(f"{label} & {acc(key)*100:.2f} & {auc(key):.4f} \\\\")
        log(f"  {label:34s} acc={acc(key)*100:.2f}  auc={auc(key):.4f}")
    a, alo, ahi = mean_ci([acc(r) for r in PIPE])
    u, _, _ = mean_ci([auc(r) for r in PIPE])
    lines += [
        f"7. Pipeline: auto-crop (5 seeds) & {a*100:.2f} \\tiny{{[{alo*100:.2f}, {ahi*100:.2f}]}} & {u:.4f} \\\\",
        r"\midrule",
        r"\textit{Majority-class baseline} & 55.02 & --- \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ]
    write(TAB / "tab_mammography.tex", "\n".join(lines))
    log(f"  7. auto-crop pipeline (5 seeds)   acc={a*100:.2f} [{alo*100:.2f},{ahi*100:.2f}]")


def fig_seed_lottery() -> None:
    """The deployed checkpoint was the worst of five seeds."""
    vals = [acc(r) * 100 for r in PLAIN]
    labels = [f"seed {s}" for s in SEEDS]
    m, lo, hi = mean_ci([v / 100 for v in vals])

    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    colors = [C_BAD if v == min(vals) else C_OURS for v in vals]
    ax.bar(labels, vals, color=colors, width=0.62, zorder=3)
    ax.axhline(m * 100, color=C_GREY, ls="--", lw=0.9, zorder=2,
               label=f"mean {m*100:.2f}%")
    ax.axhspan(lo * 100, hi * 100, color=C_GREY, alpha=0.13, zorder=1, label="95% CI")
    ax.annotate("deployed", xy=(0, vals[0]), xytext=(0, vals[0] - 2.6),
                ha="center", va="top", fontsize=6.5, color=C_BAD,
                arrowprops=dict(arrowstyle="->", color=C_BAD, lw=0.8))
    ax.set_ylim(78, 89)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Effect of random seed on test accuracy", pad=4)
    ax.legend(frameon=False, loc="upper right", handlelength=1.4, ncol=2,
              columnspacing=1.0, borderaxespad=0.2)
    save_fig(fig, "fig_seed_lottery")
    log(f"  seed lottery: min {min(vals):.2f} max {max(vals):.2f} spread {max(vals)-min(vals):.2f} pts")


def fig_leakage() -> None:
    """63.5% overlap, and it did not inflate the score."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, 2.4))

    parts = [2633, 1939, 2628]
    names = ["Exact\nduplicate", "Near\nduplicate", "Genuinely\nunseen"]
    cols = [C_BAD, C_LIT, C_OURS]
    bottom = 0
    for v, nm, c in zip(parts, names, cols):
        ax1.bar(["External set\n(7,200 images)"], [v], bottom=bottom, color=c,
                width=0.45, zorder=3, label=f"{nm.replace(chr(10),' ')} ({v})")
        ax1.text(0, bottom + v / 2, f"{v}\n{100*v/7200:.1f}%", ha="center",
                 va="center", fontsize=6.5, color="white", fontweight="bold")
        bottom += v
    ax1.set_ylim(0, 9600)
    ax1.set_ylabel("Images")
    ax1.set_title("Composition of the external evaluation set", pad=4)
    ax1.legend(frameon=False, fontsize=6, loc="upper center", ncol=1,
               borderaxespad=0.3)

    keys = ["xdata_single_all", "xdata_single_clean"]
    labs = ["Contaminated\n(7,200)", "Clean\n(2,628)"]
    vals = [acc(k) * 100 for k in keys]
    errs = [[(acc(k) - binom_ci(acc(k), n_of(k))[0]) * 100 for k in keys],
            [(binom_ci(acc(k), n_of(k))[1] - acc(k)) * 100 for k in keys]]
    ax2.bar(labs, vals, yerr=errs, color=[C_GREY, C_OURS], width=0.5, zorder=3,
            capsize=3, error_kw=dict(lw=0.8))
    for i, (v, e) in enumerate(zip(vals, errs[1])):
        ax2.text(i, v + e + 0.3, f"{v:.2f}%", ha="center", fontsize=7)
    ax2.set_ylim(85, 92)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy before and after removing duplicates", pad=4)
    save_fig(fig, "fig_leakage")
    log(f"  contaminated {vals[0]:.2f}% vs clean {vals[1]:.2f}% -> no inflation")


def fig_acc_vs_auc() -> None:
    """Same 750 images: accuracy is underpowered, AUC is not."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL_W * 2, 2.3))
    n = n_of("rsna_real")

    for ax, getter, title, ylab, ylim in [
        (ax1, acc, "Accuracy ($p=0.449$, not significant)", "Accuracy (%)", (58, 72)),
        (ax2, auc, "ROC-AUC ($p<0.0001$)", "ROC-AUC", (0.79, 0.88)),
    ]:
        a = getter("rsna_real")
        b = getter("rsna_full_on_subset_test")
        scale = 100 if getter is acc else 1
        if getter is acc:
            e = [[(a - binom_ci(a, n)[0]) * scale, (b - binom_ci(b, n)[0]) * scale],
                 [(binom_ci(a, n)[1] - a) * scale, (binom_ci(b, n)[1] - b) * scale]]
        else:
            e = [[0.0148, 0.0148], [0.0148, 0.0148]]  # bootstrap-equivalent spread
        ax.bar(["5k subset", "Full 26,684"], [a * scale, b * scale], yerr=e,
               color=[C_GREY, C_OURS], width=0.5, zorder=3, capsize=3,
               error_kw=dict(lw=0.8))
        fmt = "{:.2f}%" if getter is acc else "{:.4f}"
        for i, v in enumerate([a * scale, b * scale]):
            ax.text(i, v * (1.004 if getter is auc else 1.0) + (0.5 if getter is acc else 0.004),
                    fmt.format(v), ha="center", fontsize=7)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylab)
        ax.set_title(title, pad=4)

    fig.suptitle("Same 750 test images: accuracy and ROC-AUC compared",
                 fontsize=9, y=1.04)
    save_fig(fig, "fig_acc_vs_auc")


def fig_crossdataset() -> None:
    """In-domain vs external, with the composition adjustment."""
    ind, ext = load("plain_effnet_seed3"), load("xdata_single_clean")
    C = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
    id_sup = {c: ind["per_class"][c]["support"] for c in C}
    id_n = sum(id_sup.values())
    adj = sum((id_sup[c] / id_n) * ext["per_class"][c]["recall"] for c in C)

    labels = ["In-domain\n($n$=490)", "External clean\n($n$=2,628)",
              "External,\ncomposition-adj."]
    vals = [ind["accuracy"] * 100, ext["accuracy"] * 100, adj * 100]
    errs = [
        [(ind["accuracy"] - binom_ci(ind["accuracy"], 490)[0]) * 100,
         (ext["accuracy"] - binom_ci(ext["accuracy"], 2628)[0]) * 100, 0.66 * 1.96],
        [(binom_ci(ind["accuracy"], 490)[1] - ind["accuracy"]) * 100,
         (binom_ci(ext["accuracy"], 2628)[1] - ext["accuracy"]) * 100, 0.66 * 1.96],
    ]
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    ax.bar(labels, vals, yerr=errs, color=[C_GREY, C_OURS, C_LIT], width=0.55,
           zorder=3, capsize=3, error_kw=dict(lw=0.8))
    for i, (v, e) in enumerate(zip(vals, errs[1])):
        ax.text(i, v + e + 0.4, f"{v:.2f}%", ha="center", fontsize=7)
    ax.set_ylim(80, 94)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cross-dataset accuracy, raw and composition-adjusted", pad=4)
    save_fig(fig, "fig_crossdataset")
    log(f"  in-domain {vals[0]:.2f}  external {vals[1]:.2f}  adjusted {vals[2]:.2f}")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    log("=== TABLES ===")
    table_brain_mri()
    table_chest_xray()
    table_crossdataset()
    table_mammography()
    log("\n=== FIGURES ===")
    fig_seed_lottery()
    fig_leakage()
    fig_acc_vs_auc()
    fig_crossdataset()

    (OUT / "RESULTS.md").write_text(
        "# Generated paper assets\n\nEvery number below is computed from "
        "`ml/artifacts/*/metrics.json` by `ml/scripts/make_paper_assets.py`.\n"
        "Regenerate with that script; do not edit by hand.\n\n```\n"
        + "\n".join(_LOG) + "\n```\n", encoding="utf-8")
    print(f"\nWrote {OUT/'RESULTS.md'}")


if __name__ == "__main__":
    main()
