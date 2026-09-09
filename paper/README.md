# Paper assets

`paper.pdf` is the compiled IEEE two-column PDF (7 pages), built from
`paper.tex` and this folder's tables and figures. It was produced with
[Tectonic](https://tectonic-typesetting.github.io/) rather than a full
TeX Live/MiKTeX install, since Tectonic is a single ~50MB binary that
fetches only the packages a document actually uses (about 43MB for this
paper) instead of several gigabytes.

To recompile after editing `paper.tex` or regenerating a table/figure:

```bash
# one-time: download the binary for your platform from
# https://github.com/tectonic-typesetting/tectonic/releases
# (use the -msvc- build on Windows; the -gnu- build fails to load
# api-ms-win-core-winrt-error-l1-1-0.dll on some systems)

tectonic -o build paper.tex
cp build/paper.pdf paper.pdf
```

If a full LaTeX distribution is already installed, `pdflatex paper.tex`
run twice (for cross-references) works identically. Overleaf, which has
IEEEtran preinstalled, is the simplest option if no local toolchain is
available.

Tables (`tables/*.tex`) and figures (`figures/*.pdf`, `.png`) generated from
`ml/artifacts/*/metrics.json` by `ml/scripts/make_paper_assets.py`. Nothing
in this folder is typed by hand; regenerate with that script after any new
run rather than editing a `.tex` file directly, so the paper can never
drift from the repository's own recorded results.

## Using the tables

Each `.tex` file is a standalone `table` environment using `booktabs`
(`\toprule`/`\midrule`/`\bottomrule`). Add `\usepackage{booktabs}` to the
document preamble and `\input{tables/tab_brain_mri.tex}` (etc.) at the point
of use. Figures are vector PDFs, sized for a single (3.5in) or full-width
(7.16in) IEEE column — drop the `.pdf` into `\includegraphics`.

## What each asset supports

| File | Claim it backs |
|---|---|
| `tab_brain_mri.tex`, `fig_seed_lottery` | Backbone/ensemble comparison; the deployed checkpoint was the worst of five seeds |
| `tab_chest_xray.tex`, `fig_acc_vs_auc` | More training data improved ranking ability (ROC-AUC, $p<0.0001$) but not accuracy at this sample size ($p=0.449$) |
| `tab_crossdataset.tex`, `fig_leakage`, `fig_crossdataset` | 63.5% of a public external compilation duplicates the training set; the model generalizes at least as well out-of-domain as in-domain, once class composition is accounted for |
| `tab_mammography.tex` | Seven-attempt mammography history; the best pipeline result does not reach a clinically useful margin |

## Caveats to state in prose, not just the tables

- **Chest X-ray (full 26,684-image retrain) is a single seed.** The ROC-AUC
  gain is significant on a paired bootstrap; a seed-variance claim (as given
  for every brain-MRI row) is not yet available for this configuration.
- **The cross-dataset generalization claim is "at least as well as
  in-domain," not "better than."** The raw gap is nominally significant;
  after adjusting for class composition it is not (95% CI includes zero).
- Mammography's best full-pipeline result clears its majority baseline with
  a statistically significant but practically thin margin — report the
  effect size alongside the $p$-value, not instead of it.
