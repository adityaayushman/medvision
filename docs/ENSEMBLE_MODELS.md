# Ensemble Models

MedChron supports **soft-voting ensembles** of 2+ single-backbone image
classifiers per modality. This doc covers what's implemented, where it lives
in the codebase, its current deployment status, and how to train/evaluate/use
one.

## Status at a glance

| Modality | Ensemble | Status |
|---|---|---|
| Brain MRI | 3-way: EfficientNet-B0 + ResNet50 + DenseNet121 | ✅ built & evaluated over **5 seeds**: 88.1% acc (95% CI 87.4–88.9), ROC-AUC 0.977 (95% CI 0.974–0.980) — beats the 82.0% single-model baseline — ⛔ **not deployed** (memory ceiling, see below) |
| Brain MRI | 2-way: EfficientNet-B0 + DenseNet121 | ✅ built & evaluated over **5 seeds**: 87.7% acc (95% CI 85.6–89.7), ROC-AUC 0.974 (95% CI 0.968–0.980) — a lighter alternative, but noticeably less stable across seeds (±2.03 vs the 3-way's ±0.75 accuracy points) — still ⛔ **not deployed** |
| Chest X-ray | — | single model only (EfficientNet-B0) |
| Mammography | — | single model only (EfficientNet-B0) |

Both ensemble rows are means over seeds {42, 0, 1, 2, 3}, computed with
`ml/scripts/multiseed_ci.py`. Against the published NAS paper benchmarked on
this same dataset, the 3-way ensemble is **significantly above** the
ResNet101 baseline on both metrics and **significantly above the LeaSE+DARTS
SOTA on ROC-AUC** (97.71% vs 95.60%, p=0.000057), while remaining
**significantly below that SOTA on accuracy** (88.12% vs 90.61%, p=0.00076).

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the full brain MRI backbone
comparison (every solo backbone plus both ensemble configurations), the
significance-test table, and every other modality's experiment history.

The ensemble **code path is fully functional and tested**; it's just not
switched on in the current production deployment (Render free tier).

## Concept

`EnsemblePredictor` (`ml/src/medchron/models/inference.py:92`) wraps N
`Predictor` instances that were trained independently on the **same
manifest/split** and therefore share the same `class_to_idx`. At inference
time it:

1. Runs every member's forward pass on the input image.
2. Averages their softmax probability vectors (unweighted soft-vote).
3. Reports the argmax of the averaged distribution as the ensemble label.
4. Returns a `per_model` breakdown — each member's own label + confidence —
   alongside the ensemble result, so agreement/disagreement between backbones
   is visible.

Constructor raises `ValueError` if fewer than 2 checkpoints are given, or if
members disagree on `class_to_idx` (they must be trained on the same label
space to be ensembled).

### Grad-CAM handling

Averaging saliency heatmaps across architecturally different backbones isn't
a well-defined operation, so `EnsemblePredictor.explain()` doesn't attempt
it. Instead:
- The ensemble vote (steps 1–3 above) decides the predicted class.
- Grad-CAM is computed from the **primary member only** (`self.members[0]`),
  explicitly explaining the *ensemble's* predicted class — not whatever
  class the primary model alone would have argmax'd to. This matters when
  the primary model disagrees with the ensemble vote; the overlay always
  matches the label actually reported.

## Code map

| File | Role |
|---|---|
| [`ml/src/medchron/models/inference.py`](../ml/src/medchron/models/inference.py) | `EnsemblePredictor` — inference-time soft-voting wrapper |
| [`ml/src/medchron/models/evaluate.py`](../ml/src/medchron/models/evaluate.py) | `evaluate_ensemble()` — offline metrics for a set of checkpoints (multiclass + multilabel) |
| [`ml/src/medchron/models/__init__.py`](../ml/src/medchron/models/__init__.py) | Exports `EnsemblePredictor`, `evaluate_ensemble` |
| [`ml/scripts/evaluate.py`](../ml/scripts/evaluate.py) | CLI: comma-separated `--checkpoint` paths trigger ensemble evaluation |
| [`ml/scripts/log_experiment.py`](../ml/scripts/log_experiment.py) | Publishes ensemble eval results to the Research Workspace (`--kind ensemble`) |
| [`backend/app/ml.py`](../backend/app/ml.py) | `AnalyzerService` — loads ensembles **lazily, per-request**, not at startup |
| [`backend/app/config.py`](../backend/app/config.py) | `MODEL_CHECKPOINT_BRAIN_MRI` — comma-separated paths, auto-degrades to whichever checkpoints exist on disk |
| [`backend/app/schemas.py`](../backend/app/schemas.py) | `ExperimentKind` includes `"ensemble"`; `Prediction.per_model` |
| [`backend/app/models_db.py`](../backend/app/models_db.py) | `ExperimentRun.kind` persists `"ensemble"` runs |
| [`backend/app/reports.py`](../backend/app/reports.py) | PDF report renders a "Model agreement (ensemble members)" table from `per_model` |
| [`frontend/lib/types.ts`](../frontend/lib/types.ts) | `EnsembleMember` type, `Prediction.per_model` |
| [`frontend/app/records/[id]/page.tsx`](../frontend/app/records/[id]/page.tsx) | Renders per-member agreement in the study report UI |
| [`frontend/lib/evaluation-data.ts`](../frontend/lib/evaluation-data.ts) | Literature-benchmarking entry for the 3-way brain MRI ensemble |
| [`ml/tests/test_ensemble.py`](../ml/tests/test_ensemble.py) | Unit tests: validation errors, soft-vote averaging, Grad-CAM class consistency, end-to-end train+evaluate |
| [`backend/tests/test_ensemble_persistence.py`](../backend/tests/test_ensemble_persistence.py) | API test: `per_model` round-trips through analyze → DB → study read → PDF report |

## Training & evaluating an ensemble (CLI)

Train each member independently (different backbone and/or seed), then
evaluate them together — comma-separated checkpoint paths trigger ensemble
mode automatically:

```bash
python ml/scripts/evaluate.py \
  --manifest ml/data/brain_mri/manifest.csv \
  --checkpoint ml/artifacts/brain_mri/model_efficientnet_b0.pt,ml/artifacts/brain_mri/model_resnet50.pt,ml/artifacts/brain_mri/model_densenet121.pt \
  --out-dir ml/artifacts/brain_mri_ensemble
```

This writes `metrics.json` (+ plots for multiclass) computed on the averaged
probabilities, directly comparable to a single-model `evaluate_checkpoint`
run. `evaluate_ensemble()` supports both `multiclass` and `multilabel` tasks,
inferred from the first checkpoint's saved `task` field.

Supported backbones (`ml/src/medchron/models/backbone.py`): `vgg16`,
`resnet50`, `densenet121`, `efficientnet_b0`.

## Backend integration

`AnalyzerService` (`backend/app/ml.py`) reads `MODEL_CHECKPOINT_BRAIN_MRI` as
a comma-separated list. At startup, for each modality:
- **1 checkpoint found** → loaded once, held resident (normal path).
- **2+ checkpoints found** → **not** loaded at startup. The paths are stashed
  in `_lazy_ckpts`, and `EnsemblePredictor` is constructed fresh inside
  `analyze()` for that single request, then discarded.

This lazy-load design exists because `AnalyzerService` eagerly loads every
*other* modality at startup too — keeping a 2–3 model ensemble permanently
resident would stack its memory on top of everything else for the process's
entire lifetime. Loading per-request instead bounds the extra memory to the
duration of one request.

The default env value (`backend/app/config.py`) lists all three brain MRI
checkpoint paths; `AnalyzerService` filters to whichever actually exist on
disk. This means the deployment silently upgrades from single-model to
3-way ensemble the moment all three checkpoints are present — no config
change needed.

## Why it isn't deployed

Even with lazy per-request loading, the ensemble's own marginal memory
footprint — measured with the exact CPU-only torch build Render runs — is:
- ~490MB for a 2-way ensemble
- ~583MB for a 3-way ensemble

Render's free tier caps the whole instance at 512MB. Both were tried live:
the 3-way ensemble OOM'd outright; the 2-way ensemble completed one 30–40s
request and then crash-looped on the next. This is a genuine capacity
ceiling, not a scheduling bug — see `render.yaml`'s `MODEL_CHECKPOINT_BRAIN_MRI`
comment and [`ROADMAP.md`](../ROADMAP.md) (Version 3) for the full account.
Currently deployed: single EfficientNet-B0 checkpoint per modality
(`model_efficientnet_b0_brain_mri.pt`).

Unblocking it needs either a paid Render tier with more RAM, or a
smaller/quantized backbone — not further loading-strategy changes.

### Distillation was tried as a way around this, and failed

The obvious escape is knowledge distillation: train a single
EfficientNet-B0 student on the ensemble's soft outputs, so the accuracy
ships at single-model memory cost. It is fully implemented
(`ml/src/medchron/models/distill.py`, `ml/scripts/distill.py`) and it
**significantly underperforms plain training** — 81.88% vs 84.41% over the
same 5 seeds, p=0.027. Cause: the teacher is only 87.35% accurate on the
training split, so ~1 in 8 soft targets is wrong while α=0.7 weights them
above the true labels. See `EXPERIMENTS.md` for the full write-up.

**What actually helps instead:** the deployed brain MRI model is seed 42,
the *worst* of five seeds (82.04% against a 84.41% 5-seed mean, best seed
85.71%). Reseeding is +3.67 points at byte-identical size and zero risk —
more gain than distillation promised, with none of its complexity.

## API/UI surface

When an ensemble prediction is served, `Prediction.per_model` (a list of
`{backbone, label, confidence}`) is populated alongside the usual
`label`/`confidence`/`probabilities`. This flows through:
- `POST /api/analyze` and `GET /api/studies/{id}` responses
- The PDF/JSON study report (`GET /api/studies/{id}/report[.pdf]`) — rendered
  as a "Model agreement (ensemble members)" table
- The frontend study page — same per-member breakdown, so a clinician can see
  where backbones agree or disagree, not just the final vote
