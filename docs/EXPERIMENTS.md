# Experiment Inventory

Every model and experiment run in this project, across all three
modalities, with verified numbers pulled directly from each run's
`metrics.json` (or `segmentation_history.json` where no `metrics.json`
exists). This is the "what's actually been tried" reference — see
[`TRAINING.md`](TRAINING.md) for the shared architecture/fine-tuning recipe
every classifier below uses, and [`ENSEMBLE_MODELS.md`](ENSEMBLE_MODELS.md)
for how the brain MRI ensembles work mechanically.

## Chest X-ray

| Run | Backbone | Result | Status |
|---|---|---|---|
| `ml/artifacts/rsna_real` | EfficientNet-B0 | 64.0% acc / 0.825 ROC-AUC | ✅ live |

3-class RSNA pneumonia task (Normal / Lung Opacity / No Lung Opacity-Not
Normal), trained on a 5,000-image class-balanced subset (not the full
~26k-image set) on CPU — a deliberate reproducibility baseline, not the
strongest possible number; see `frontend/lib/evaluation-data.ts`'s
`chest_xray.literature` section for the honest gap-to-literature writeup.

**Not a result**: the loose files at `ml/artifacts/metrics.json` /
`history.json` / `confusion_matrix.png` / `roc_curve.png` (repo root of
`ml/artifacts/`) are a 5-image smoke-test fixture (acc 0.8, AUC 1.0, classes
`normal`/`pneumonia`) used to generate `gradcam_demo.png`/`pipeline_demo.png`
for documentation — don't mistake it for a trained model's metrics.

## Brain MRI

4-class tumor classification (glioma / meningioma / pituitary / no tumor),
3,264 images, same manifest/split across every run below — the cleanest
apples-to-apples comparison in the project.

| Run | Config | Accuracy | ROC-AUC | Status |
|---|---|---|---|---|
| `brain_mri` | EfficientNet-B0 | 82.04% | 0.9564 | ✅ live |
| `brain_mri_resnet50_solo` | ResNet50, seed 42 | 85.51% | 0.9701 | not deployed |
| `brain_mri_resnet50_seed0` | ResNet50, seed 0 | 82.45% | 0.9631 | not deployed |
| `brain_mri_resnet50_seed1` | ResNet50, seed 1 | 84.08% | 0.9664 | not deployed |
| `brain_mri_resnet50_seed2` | ResNet50, seed 2 | 85.71% | 0.9684 | not deployed |
| `brain_mri_resnet50_seed3` | ResNet50, seed 3 | 85.71% | 0.9682 | not deployed |
| ResNet50, mean of 5 seeds (42,0,1,2,3) | — | **84.69%** | **0.9673** | 95% CI: 82.9–86.5% acc, 96.4–97.1% AUC |
| `brain_mri_densenet121_solo` | DenseNet121 | 83.06% | 0.9576 | not deployed |
| `brain_mri_vgg16_solo` | VGG16 | 85.31% | 0.9684 | not deployed |
| `ens2_seed{42,0,1,2,3}` | Ensemble: EfficientNet-B0 + DenseNet121, **mean of 5 seeds** | **87.67%** | **0.9741** | 95% CI: 85.6–89.7% acc, 96.8–98.0% AUC — not deployed |
| `ens3_seed{42,0,1,2,3}` | Ensemble: EfficientNet-B0 + ResNet50 + DenseNet121 (3-way), **mean of 5 seeds** | **88.12%** | **0.9771** | 95% CI: 87.4–88.9% acc, 97.4–98.0% AUC — not deployed (memory ceiling — see `ENSEMBLE_MODELS.md`) |
| `brain_mri_2way_effnet_dense` | 2-way ensemble, seed 42 only (superseded) | 84.90% | 0.9660 | single run — kept for provenance |
| `brain_mri_ensemble` | 3-way ensemble, seed 42 only (superseded) | 87.14% | 0.9727 | single run — kept for provenance |

Two things worth being precise about, since the project has a history of
over/under-stating exactly this kind of thing:

- **The single-run ResNet50 number (85.5%) is not the honest comparison
  point.** The 5-seed mean (84.69%) is what's statistically defensible — the
  published ResNet101 baseline it's compared against (84.5% acc) falls
  *inside* that 5-seed 95% CI, so the accuracy win over that baseline is
  **not statistically significant** (one-sample t-test p=0.80). The ROC-AUC
  win over that same baseline (90.1%) *is* significant (our CI floor is
  96.4%, p<0.000001). See `frontend/lib/evaluation-data.ts`'s `brain_mri`
  `literature` block for the full writeup — this is already correctly
  reflected on the live `/evaluation` page.
- **DenseNet121 solo (83.06%) has never been mentioned in any narrative
  copy** (README/ROADMAP/evaluation-data.ts all cite ResNet50 and VGG16 as
  "both beat EfficientNet-B0" without mentioning DenseNet121's own solo
  result, which also beats it, just less than the others). Listed here so
  the full backbone comparison is actually complete.
- **Both ensembles are now 5-seed results, and seed 42 was the worst seed
  for both.** The previously-published single-run numbers (3-way 87.14%,
  2-way 84.90%) were therefore pessimistic, not optimistic — re-running
  seeds 0–3 raised the 3-way mean to 88.12% and the 2-way mean to 87.67%.
  The seed-42 runs reproduce the old artifacts exactly, which is what
  validates the new evaluation setup.

### The deployed brain MRI model is the worst of five seeds

Evaluating the four extra EfficientNet-B0 seed checkpoints standalone (they
were trained as ensemble members, but never scored on their own) produced
the most actionable finding in this whole line of work:

| Plain EfficientNet-B0 | Accuracy | ROC-AUC |
|---|---|---|
| seed 42 — **currently deployed** | 82.04% | 0.9564 |
| seed 0 | 84.49% | 0.9670 |
| seed 1 | 84.90% | 0.9681 |
| seed 2 | 84.90% | 0.9693 |
| seed 3 | **85.71%** | 0.9690 |
| **5-seed mean** | **84.41%** (95% CI 82.67–86.14) | **0.9660** (95% CI 0.9593–0.9726) |

The live model sits at the very bottom of its own seed distribution.
Retraining the identical architecture with seed 3 is **+3.67 accuracy
points** at byte-identical checkpoint size (16.35MB), zero memory change,
and zero deployment risk — a bigger, cheaper, and far more certain win than
either the ensemble (blocked on memory) or distillation (below). The
originally-published 82.04% was never wrong, it was just unlucky; nothing
about it was known to be a low draw until the other seeds existed.

### Knowledge distillation — tested, and it does not work here

Full implementation in `ml/src/medchron/models/distill.py` (Hinton KD,
T=4.0, α=0.7) with teacher soft targets cached by
`ml/scripts/cache_teacher_logits.py` from the 3-way ensemble. The premise:
the student is the same architecture and size as the deployed model, so any
accuracy recovered from the 88.12% teacher would ship at zero memory cost.

It made things **significantly worse**, on the same 5 seeds, paired t-test:

| | Plain (5 seeds) | Distilled (5 seeds) | Δ | p |
|---|---|---|---|---|
| Accuracy | 84.41% | **81.88%** (95% CI 80.15–83.61) | **−2.53 pts** | 0.027 |
| ROC-AUC | 0.9660 | **0.9580** (95% CI 0.9563–0.9598) | **−0.79 pts** | 0.014 |

**Diagnosed mechanism, not a mystery.** The teacher scores only **87.35% on
the training split itself** (val 85.10%, test 87.14%) — so roughly 1 in 8
soft targets points at the wrong class, and α=0.7 weights those errors at
more than double the true hard labels. Standard KD assumes a teacher that
has effectively memorised its training data (typically >95% train
accuracy); this ensemble never did, because its members were early-stopped
around 8 epochs with augmentation. With only ~3.7 points of teacher
headroom over a plain student and 12.65% of the guidance actively wrong,
the soft targets are net noise rather than "dark knowledge".

Worth testing if revisited (not attempted here): a much lower α so hard
labels dominate, or masking the soft term on samples where the teacher
disagrees with ground truth. Neither is likely to beat simply reseeding,
which costs nothing and delivers more.

The distillation code stays in the repo — it is correct, tested, and the
negative result is only meaningful because the implementation is sound.

### 3-way ensemble vs. the published NAS paper (5 seeds, one-sample t-test)

| Comparison | Ours (n=5) | Reference | Result |
|---|---|---|---|
| Accuracy vs ResNet101 baseline | 88.12% | 84.52% | **significantly above** (p=0.00018) |
| Accuracy vs LeaSE+DARTS SOTA | 88.12% | 90.61% | **significantly below** (p=0.00076) |
| ROC-AUC vs ResNet101 baseline | 97.71% | 90.06% | **significantly above** (p<0.000001) |
| ROC-AUC vs LeaSE+DARTS SOTA | 97.71% | 95.60% | **significantly above** (p=0.000057) |

The honest headline: on this dataset the 3-way ensemble **beats the
published architecture-search SOTA on ROC-AUC** while remaining
**significantly below it on accuracy**. Both directions are real and
statistically supported; neither should be quoted without the other. The
2-way ensemble also significantly beats the ResNet101 baseline on accuracy
(87.67%, p=0.0126) but has a much wider CI than the 3-way (±2.03 vs ±0.75
points), i.e. it is less stable across seeds.

## Mammography

The platform's one open, honestly-documented gap — full seven-attempt
history. Two baselines apply, and they are **not interchangeable**: MIAS
(3-class, attempt 1 only) has a 63.3% majority baseline; every CBIS-DDSM
full-mammogram run (attempts 2, 4, 5, 6, 7) has its own, separate 55.0%
majority baseline (test split: 241 Benign / 197 Malignant = 55.02%). Mixing
these up produced a real error in an earlier draft of this project's own
docs — corrected as of this writing.

| # | Run(s) | Dataset | Result | Verdict |
|---|---|---|---|---|
| 1 | `mammography` | MIAS, full mammograms, 3-class | 59.18% acc / 0.618 AUC | below 63.3% baseline — fail |
| 2 | `mammography_cbis` | CBIS-DDSM, full mammograms | 59.13% acc / 0.642 AUC | above 55.0% baseline, thinly — thin win |
| 3 | `mammography_cbis_cropped` | CBIS-DDSM, official lesion crops | **71.12% acc / 0.785 AUC** | real win — best standalone result, but needs a pre-cropped image the upload flow doesn't provide |
| 4 | `mammography_localized` | bbox regressor → attempt 3's classifier | 49.77% acc / 0.486 AUC | below baseline — fail |
| 5 | `mammography_localized_segmentation` | U-Net segmenter → attempt 3's classifier | 48.86% acc / 0.541 AUC | below baseline — fail, despite a much better localizer |
| 6a | `mammography_gtcrop` (standalone) | classifier retrained on ground-truth-box crops | 62.53% acc / 0.660 AUC | — |
| 6b | `mammography_localized_gtcrop` (pipeline) | segmenter → 6a's classifier | 52.97% acc / 0.534 AUC | below baseline — partial recovery, still fails |
| 7a | `mammography_autocrop` (standalone) | classifier retrained on segmenter's own predicted crops | 56.39% acc / 0.566 AUC | — |
| 7b | `pipe_autocrop_seed{42,0,1,2,3}` (pipeline, **mean of 5 seeds**) | segmenter → 7a's classifier | **56.58% acc / 0.584 AUC** | 95% CI 55.17–57.98% acc — thin win, best full-pipeline result yet |

**Attempt 7 statistics (5 seeds, added after the fact — the original
write-up was a single run):** accuracy mean 56.58%, 95% CI [55.17%, 57.98%];
ROC-AUC mean 58.35%, 95% CI [56.05%, 60.66%]. Against the 55.02% majority
baseline the accuracy win is *statistically* significant (one-sample t-test,
p=0.0377) but the CI's lower bound clears the baseline by only 0.15
points — significant, yet practically negligible. The more robust finding is
that ROC-AUC sits well above chance (58.35% vs 0.500, p=0.00055), i.e. the
pipeline has genuine discriminative signal rather than just tracking the
majority class. Per-seed accuracies ranged 55.02–58.22%, so any single-run
number from this configuration is worth ±1.4 points of seed noise. **Still
not deployed** — a 1.6-point margin over majority guessing is not clinically
useful regardless of its p-value.

Attempt 4's localizer (the bounding-box regressor, `SpatialBBoxNet` in
`ml/src/medchron/models/detect.py`) has **no saved metrics file** — its
IoU 0.043 → 0.068 progression across three architecture iterations (naive
pooled head → spatial soft-argmax → attention-weighted size pooling) is
documented only in the module's docstring, not as a `metrics.json`. Attempt
5's segmenter (`ml/artifacts/mammography_segmentation`, no `metrics.json`
either) has a `segmentation_history.json` whose best validation Dice is
0.2785 (epoch 10) — the "test Dice 0.254" figure cited in `ROADMAP.md` is a
separately-run held-out test evaluation, not the same number as the training
curve's best val Dice; both are real, they're just measuring different
splits.

Full method write-ups for each attempt live in `ROADMAP.md`'s Version 2
section and the live `/case-study` page — this table exists to give the
verified numbers a single place to be checked against source files.

## Full `ml/artifacts/` reference table

Every subfolder as of this writing, so nothing in the repo is a mystery:

| Folder | Has `metrics.json`? | Headline |
|---|---|---|
| `brain_mri` | yes | 82.04% acc / 0.9564 AUC |
| `brain_mri_2way_effnet_dense` | yes | 84.90% acc / 0.9660 AUC (seed 42, superseded by `ens2_seed*`) |
| `brain_mri_densenet121_seed0..3` | no (checkpoints only) | ensemble members, evaluated via `ens2_*`/`ens3_*` |
| `brain_mri_densenet121_solo` | yes | 83.06% acc / 0.9576 AUC |
| `brain_mri_distilled` | yes | 81.84% acc / 0.9560 AUC (distilled, seed 42) |
| `brain_mri_distilled_seed0..3` | no (checkpoints only) | distilled students, evaluated via `distilled_eval_*` |
| `brain_mri_efficientnet_b0_seed0..3` | no (checkpoints only) | ensemble members; standalone scores via `plain_effnet_seed*` |
| `brain_mri_teacher_soft.npz` | (loose file) | cached 3-way ensemble soft targets, 3,264 rows |
| `distilled_eval_seed0..3` | yes (each) | distilled student per seed; mean 81.88% / 0.9580 |
| `plain_effnet_seed0..3` | yes (each) | plain EfficientNet-B0 per seed; mean (with seed 42) 84.41% / 0.9660 |
| `brain_mri_ensemble` | yes | 87.14% acc / 0.9727 AUC (seed 42, superseded by `ens3_seed*`) |
| `brain_mri_resnet50_seed0..3` | yes (each) | see table above |
| `ens2_seed{42,0,1,2,3}` | yes (each) | 2-way ensemble per seed; mean 87.67% / 0.9741 |
| `ens3_seed{42,0,1,2,3}` | yes (each) | 3-way ensemble per seed; mean 88.12% / 0.9771 |
| `pipe_autocrop_seed{42,0,1,2,3}` | yes (each) | mammography pipeline per seed; mean 56.58% / 0.5835 |
| `brain_mri_resnet50_solo` | yes | 85.51% acc / 0.9701 AUC |
| `brain_mri_resnet50_multiseed.json` | (loose file, not a folder) | raw per-seed array backing the 5-seed mean above |
| `brain_mri_vgg16_solo` | yes | 85.31% acc / 0.9684 AUC |
| `mammography` | yes | 59.18% acc / 0.618 AUC (MIAS) |
| `mammography_autocrop` | yes | 56.39% acc / 0.566 AUC (standalone classifier, seed 42) |
| `mammography_autocrop_seed0..3` | no (checkpoints only) | pipeline members, evaluated via `pipe_autocrop_*` |
| `mammography_cbis` | yes | 59.13% acc / 0.642 AUC |
| `mammography_cbis_cropped` | yes | 71.12% acc / 0.785 AUC |
| `mammography_gtcrop` | yes | 62.53% acc / 0.660 AUC |
| `mammography_localized` | yes | 49.77% acc / 0.486 AUC |
| `mammography_localized_autocrop` | yes | 56.62% acc / 0.574 AUC |
| `mammography_localized_gtcrop` | yes | 52.97% acc / 0.534 AUC |
| `mammography_localized_segmentation` | yes | 48.86% acc / 0.541 AUC |
| `mammography_segmentation` | no (`segmentation_history.json` only) | best val Dice 0.2785 |
| `rsna_real` | yes | 64.0% acc / 0.825 AUC |
| root loose files | yes (toy fixture) | 80% acc / 1.0 AUC — 5-image smoke test, not a real model |

## How results reach the live site

Two separate, non-overlapping paths — don't confuse them:

- **`ml/scripts/log_experiment.py`** publishes a run's `metrics.json` to the
  backend's `ExperimentRun` table (`backend/app/models_db.py`, `kind` one of
  `classification`/`bbox_regression`/`segmentation`/`ensemble`), which
  powers the authenticated, role-gated **Research Workspace**
  (`/dashboard/(app)/research`) — a live, DB-driven table, not static
  content. This is where every run above is expected to be logged as it's
  produced.
- **`frontend/lib/evaluation-data.ts`** is a separate, hand-curated dataset
  backing the public **`/evaluation`** page — headline metrics, training
  curves, and literature comparisons for the three *deployed-or-headline*
  models (chest X-ray, brain MRI, mammography's best standalone result). It
  does not auto-sync from `ml/artifacts/` or the Research Workspace; updates
  to it are a manual editing step, same as this document.
