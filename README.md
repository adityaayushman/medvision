# MedChron AI

**AI-powered medical imaging intelligence platform** — image processing, ROI
extraction, explainable deep-learning classification, and a longitudinal patient
record ("Digital Twin"), unified into one workflow.

**Live at [medchron-ai.vercel.app](https://medchron-ai.vercel.app)** (frontend,
Vercel) backed by a FastAPI service on Render.

> ⚕️ **Research & educational software. Not a medical device. Not for clinical
> use, diagnosis, or treatment decisions.**

---

## Why this is different

Most imaging projects stop at *"disease / no disease."* MedChron AI is built as a
**platform**, not a single classifier:

```
Upload → Quality Check → Preprocess (DIP) → Segment → ROI →
        Backbone (EfficientNet-B0 / swappable) → Classify → Grad-CAM →
        Report → Patient Digital Record → Timeline
```

Two deliberate engineering decisions make it professional and scalable:

1. **Modality-agnostic by design.** Chest X-ray, brain MRI, and mammography are
   all *config presets*, not separate rewrites — see `PRESETS` in
   [`config.py`](ml/src/medchron/config.py). Adding a modality is training a
   checkpoint and adding a preset, not new pipeline code.
2. **Backbone-swappable model layer.** `backbone.py` supports VGG16, ResNet50,
   DenseNet121, and EfficientNet-B0 interchangeably. In practice every shipped
   model uses EfficientNet-B0 (fastest, smallest, best-performing of the four
   here) — VGG16 is registered as an option but has never actually been
   trained in this project.

## Monorepo layout

```
MedChronAI/
├── ml/                     # Python: DIP pipeline, training, evaluation, XAI
│   ├── src/medchron/       # installable `medchron` package
│   │   ├── config.py       # PreprocessConfig + per-modality presets
│   │   └── imaging/        # quality → enhancement → segmentation → roi → pipeline
│   ├── scripts/            # runnable entry points (demo, train, evaluate)
│   ├── tests/              # pytest (no dataset required)
│   ├── data/               # datasets (git-ignored)
│   └── artifacts/          # model weights & outputs (git-ignored)
├── backend/                # FastAPI inference API + auth + patient records (live)
├── frontend/               # Next.js analyze/records UI + clinical dashboard (live)
├── assets/samples/         # sample images
├── docs/                   # dataset shortlist, design docs
└── _legacy/                # original prototype scripts, kept for reference
```

## Quick start (imaging core)

```bash
cd ml
python -m pip install -e ".[dev]"     # deps: numpy, opencv (+ pytest, ruff)
python -m pytest                       # 6 passing tests, no dataset needed
python scripts/demo_pipeline.py        # writes ml/artifacts/pipeline_demo.png
```

```python
from medchron import MedicalImagePipeline, get_config

pipe = MedicalImagePipeline(get_config("chest_xray"))
result = pipe.run_path("assets/samples/chest_xray_sample.jpg")

print(result.summary())                # quality verdict + detected ROIs
tensor = pipe.model_input(result)      # 224x224x3 float32, ready for a backbone
```

## Status

| Layer | State |
|-------|-------|
| DIP pipeline (quality → enhance → segment → ROI) | ✅ done, tested, modality-agnostic |
| Dataset layer (registry, manifest, leak-safe splits) | ✅ done, tested |
| Model layer (VGG16/ResNet50/DenseNet121/EfficientNet-B0, swappable) | ✅ done, tested |
| Evaluation (metrics, ROC/AUC, confusion matrix) | ✅ done |
| Explainability (Grad-CAM) | ✅ done, verified |
| Backend (FastAPI + auth + patient records + timeline) | ✅ **live** |
| Frontend (Next.js analyze/records + clinical dashboard) | ✅ **live** |
| Chest X-ray | ✅ **live** — 64% acc, ROC-AUC 0.825 |
| Brain MRI (4-class tumor classification) | ✅ **live** — 82.0% acc, macro F1 0.819, ROC-AUC 0.956 |
| Brain MRI ensemble (2-3 EfficientNet-B0/ResNet50/DenseNet121) | ✅ built, evaluated, beats the single-model baseline — **not deployed**: its own memory footprint (~490-580MB, measured with the exact CPU-only torch build Render runs) exceeds the free tier's 512MB ceiling regardless of loading strategy |
| Mammography (Benign/Malignant) | ⏸️ **not deployed** — see below |
| AI-assisted report drafting | ✅ **live** |
| Clinical dashboard (auth, org-scoped records, audit trail) | ✅ **live** at `/dashboard` |
| Mobile app | ⏳ planned (PWA — see `ROADMAP.md`) |

**Mammography, in detail:** the honest, still-open problem on this platform.
A classifier trained on CBIS-DDSM's lesion-*cropped* patches scores 71.1%
acc / 0.785 AUC — a real, clear win — but needs an already-cropped lesion
image, which the app's upload flow doesn't provide (like every other
modality, it takes a full mammogram). Two independent attempts to build an
automatic crop step (bounding-box regression, then U-Net pixel segmentation)
each measurably improved at their own task (IoU 0.043 → 0.068, then Dice
0.254 with the segmentation model) but neither closed the gap in the full
detect → crop → classify pipeline, which stayed below the plain full-image
baseline both times. Diagnosis: the classifier itself, trained only on
official hand-verified crops, doesn't generalize to *any* automatically
derived crop — the bottleneck was never localization accuracy. See
`ml/src/medchron/models/detect.py` and `segment.py` for the (correct, tested,
currently unused) code, and recent commit messages for the full numbers.

## Run the full stack

```bash
# 1) Backend (from repo root, D: venv active)
.venv/Scripts/python -m pip install -r backend/requirements.txt
cd backend && uvicorn app.main:app --reload          # :8000, /docs

# 2) Frontend (new terminal)
cd frontend && npm install && npm run dev             # :3000

# 3) Train a model so /analyze returns predictions (else preprocess-only)
python ml/scripts/train.py --root ml/data/<dataset> --backbone efficientnet_b0 --device cuda
```

Automated tests: `ml/tests/` (pytest, no dataset required) and
`backend/tests/` (hermetic, no real checkpoint required) both stay green on
every change — run `python -m pytest tests/ -q` from either directory.

## Training hardware

Render (the production deploy target) is **CPU-only** —
`backend/Dockerfile` explicitly installs the CPU-only torch/torchvision
wheels (`--index-url https://download.pytorch.org/whl/cpu`), since the
service has no GPU and the CUDA wheels are ~10x larger for no benefit there.

Local training happens on whatever hardware is available; every
training/eval/explain script takes `--device {auto,cpu,cuda}`. If you're
measuring memory footprint for a deploy decision, measure it with the
CPU-only build specifically — a GPU-capable local torch install can report
marginal memory usage ~2x higher than what Render's container actually uses,
which matters a lot when you're up against a hard 512MB ceiling.

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for the version progression beyond this
release — the mobile PWA, a path to making the ensemble deployable (more
RAM or a lighter architecture, not a scheduling fix), and an honestly-scoped
section on PACS/federated learning (both need a real institutional partner
to be more than a simulation).

## License

Apache-2.0 (see `pyproject.toml`).
