# MedChron AI

**AI-powered medical imaging intelligence platform** — image processing, ROI
extraction, explainable deep-learning classification, and a longitudinal patient
record ("Digital Twin"), unified into one workflow.

> ⚕️ **Research & educational software. Not a medical device. Not for clinical
> use, diagnosis, or treatment decisions.**

---

## Why this is different

Most imaging projects stop at *"disease / no disease."* MedChron AI is built as a
**platform**, not a single classifier:

```
Upload → Quality Check → Preprocess (DIP) → Segment → ROI →
        Backbone (VGG16 / swappable) → Classify → Grad-CAM →
        Report → Patient Digital Record → Timeline
```

Two deliberate engineering decisions make it professional and scalable:

1. **Modality-agnostic by design.** V1 ships **chest X-ray** done well. A new
   modality (brain MRI, mammography, …) is a *config preset*, not a rewrite — see
   `PRESETS` in [`config.py`](ml/src/medchron/config.py).
2. **Backbone-swappable model layer.** VGG16 is the teaching baseline; the model
   API is designed so a modern backbone (EfficientNet/ConvNeXt) drops in without
   touching data, evaluation, or serving code.

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
├── backend/                # FastAPI inference API + patient records (planned)
├── frontend/               # Next.js dashboard + timeline (planned)
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
| DIP pipeline (quality → enhance → segment → ROI) | ✅ done, tested |
| Dataset layer (registry, manifest, splits) | ⏳ next |
| Model (VGG16 transfer learning + fine-tuning) | ⏳ planned |
| Evaluation (metrics, ROC/AUC, confusion matrix) | ⏳ planned |
| Explainability (Grad-CAM) | ⏳ planned |
| Backend (FastAPI + patient records) | ⏳ planned |
| Frontend (Next.js dashboard + timeline) | ⏳ planned |
| Mobile app | ⏳ planned |

## GPU on Windows — read before building the model layer

This machine has an **RTX 4050 (6 GB)**. Modern **TensorFlow (≥2.11) has no
native Windows GPU support** — it only sees the GPU through **WSL2**. Native
Windows CUDA is a **PyTorch** capability. So when we build training, pick one:

- **PyTorch** — trains on the GPU natively on Windows (simplest here), or
- **TensorFlow/Keras in WSL2** — matches the original plan's Keras VGG16, or
- **Google Colab** — free GPU, zero local setup.

The imaging core above has **no deep-learning dependency**, so this choice is
deferred with zero cost.

## License

Apache-2.0 (see `pyproject.toml`).
