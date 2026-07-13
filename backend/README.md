# MedChron AI — Backend (FastAPI)

REST API that runs the MedChron imaging pipeline, optionally classifies with a
trained model, generates Grad-CAM overlays, and stores a longitudinal patient
record (patients → studies → predictions).

> Research/educational software. **Not for clinical use.**

## Run

```bash
# from repo root, with the D: venv active
d:/MedChronAI/.venv/Scripts/python -m pip install -r backend/requirements.txt
cd backend
uvicorn app.main:app --reload
```

- Interactive docs: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health>

If no trained checkpoint exists at `ml/artifacts/model_vgg16.pt`, the API runs in
**preprocess-only mode**: `/api/analyze` still returns the quality verdict and
detected ROIs, just without a classification/heatmap. Train a model and it
automatically starts returning predictions.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/health` | status + whether a model is loaded |
| POST | `/api/analyze` | upload an image → quality, ROIs, prediction, Grad-CAM; persists a Study |
| POST | `/api/patients` | create a patient |
| GET  | `/api/patients` | list patients |
| GET  | `/api/patients/{id}` | one patient |
| GET  | `/api/patients/{id}/timeline` | chronological studies (disease monitoring) |

## Configuration (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `DATABASE_URL` | `sqlite:///backend/medchron.db` | database connection |
| `MODEL_CHECKPOINT` | `ml/artifacts/model_vgg16.pt` | trained model to serve |
| `MEDCHRON_MODALITY` | `chest_xray` | preprocessing preset |
| `STORAGE_DIR` | `backend/storage` | uploads + overlays (served at `/static`) |
| `CORS_ORIGINS` | `http://localhost:3000` | allowed frontend origins |
