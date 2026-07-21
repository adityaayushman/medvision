# MedChron AI — Roadmap

This document tracks the platform's version progression. Each phase is
scoped by what it actually takes to build and verify — not just what it
takes to describe — so priorities can be set honestly. v1-v4 are live;
what's below their headers is what actually happened, not a plan.

> Live at [medchron-ai.vercel.app](https://medchron-ai.vercel.app)
> (frontend) + a FastAPI backend on Render. See `README.md` for the
> present-tense status table.

---

## Guiding principle: additive, never disruptive

Every phase below is designed to be a **config preset, a new route, or a new
table** — not a rewrite of what's shipped. The chest X-ray pipeline, the
existing patient records, and the live deployment must keep working
unmodified while each phase lands. Concretely:

- New modalities are new `PreprocessConfig` presets + new checkpoints, never
  a change to the chest X-ray path.
- New model backbones register alongside the current one; nothing is
  swapped out from under existing predictions.
- Schema changes are additive columns with safe migrations (the pattern
  already used for `quality_score`/`analysis_stopped`), never a rename or
  drop of an existing field the frontend depends on.
- Each phase ships behind its own route/page so a half-finished feature
  never blocks the ones already live.

---

## Version 2 — Multi-Modality Expansion
**Status: chest X-ray and brain MRI live. Mammography remains the platform's
one open, honestly-documented gap** — not for lack of trying: three
independent modeling attempts, each a real, evaluated result.

| Modality | Status | Real result |
|---|---|---|
| Chest X-ray | ✅ Live (v1) | accuracy 64%, ROC-AUC 0.825 |
| Brain MRI (4-class tumor classification) | ✅ Live | accuracy 82.0%, macro F1 0.819, ROC-AUC 0.956 (held-out test, sartajbhuvaji/brain-tumor-classification-mri, 3,264 images) |
| Mammography (Benign/Malignant) | ⏸️ **Not serving predictions** | see below |

A cross-modality bug was caught and fixed along the way: quality-gate
thresholds (blur/brightness/contrast) were implicitly tuned for chest X-ray
and don't transfer — verified this rejected 60% of genuinely fine brain MRI
scans before `min_focus` became a per-modality config value instead of a
global constant.

**Mammography — the full honest history**, because it's the platform's
best example of the honest-gate principle actually being followed rather
than just stated:

1. **MIAS (322 images), full mammograms**: 59.2% acc, macro F1 0.37 — *below*
   the 63.3% majority-class baseline. Not deployed.
2. **CBIS-DDSM (2,857 images), full mammograms**: 59.1% acc, ROC-AUC 0.642 —
   technically beats the 55.0% majority baseline, but thinly, and malignant
   recall was only 43.7%. Not deployed (see `ml/artifacts/mammography_cbis`).
3. **CBIS-DDSM, lesion-*cropped* patches**: 71.1% acc, 0.785 AUC, 66.4%
   malignant recall — a real, clear win. Root cause of (1) and (2)'s
   weakness identified: a full mammogram shrinks the lesion to a handful of
   pixels at the model's 224×224 input size, destroying the texture that
   distinguishes benign from malignant. **Still not deployed** — this model
   needs an already-cropped lesion image, and the app's upload flow (like
   every other modality) provides a full mammogram.
4. **Automatic crop, attempt 1 — bounding-box regression**
   (`ml/src/medchron/models/detect.py`): three architecture iterations
   (naive pooled head → spatial soft-argmax → attention-weighted size
   pooling), test IoU 0.043 → 0.068. Full detect→crop→classify pipeline:
   49.8% acc / 0.486 AUC — *below* both baselines.
5. **Automatic crop, attempt 2 — U-Net pixel segmentation**
   (`ml/src/medchron/models/segment.py`): EfficientNet-B0 encoder + a
   skip-connected decoder, trained on the same CBIS-DDSM ROI masks. Test
   Dice 0.254 — roughly 4x better than attempt 4's IoU by any reasonable
   comparison, and visually confirmed on real predictions. Full pipeline:
   48.9% acc / 0.541 AUC — **still below baseline**, despite the much better
   localizer. Diagnosis: localization was never the actual bottleneck — the
   classifier (step 3) was trained only on official hand-verified crops and
   doesn't generalize to *any* automatically-derived crop, regardless of how
   accurately it's centered. Two different localizer architectures hit the
   same wall because neither touched that mismatch.

**Next step, if revisited:** retrain the classifier on crops the way the
pipeline actually produces them (segmenter output + padding), not on
official ground-truth crops — a different experiment from "improve the
localizer," which is what both attempts 4 and 5 tried.

## Version 3 — AI-Assisted Reporting & Model Improvements
**Status: report drafting live. Ensemble built and evaluated, not
deployable on the current hosting tier.**

- **AI-assisted report drafting** — ✅ **live.** The prediction payload
  renders as a structured draft report (PDF/text), labeled **"AI Draft —
  Requires Clinician Review"**, no auto-finalization path.
- **Multi-model ensemble** — ✅ built (`EnsemblePredictor`, soft-voting
  across 2-3 backbones on the same split) and ✅ evaluated: a 3-way brain
  MRI ensemble clearly beat the single-model baseline. ⛔ **Not deployed.**
  Two live deploy attempts (3-way, then 2-way) OOM'd / crash-looped Render's
  512MB free-tier instance. Root cause isolated with real measurements,
  not guesses: `AnalyzerService` eagerly loads every modality at startup, so
  a permanently-resident ensemble stacks on top of everything else for the
  process's whole lifetime — fixed with load-per-request instead (see
  `backend/app/ml.py`). That *reduced* memory waste but didn't solve the
  underlying problem: measured with the exact CPU-only torch build Render
  runs, the ensemble's own marginal footprint alone is ~490MB (2-way) to
  ~583MB (3-way) — at or over the entire 512MB ceiling before counting
  anything else in the process. This is a real capacity ceiling, not a
  scheduling bug; shipping the ensemble needs either a paid tier with more
  RAM or a smaller/quantized backbone, not more loading-strategy cleverness.

## Version 4 — Clinical Dashboard for Hospitals
**Status: live at `/dashboard`.** Auth (PBKDF2 password hashing, JWT
sessions, roles: radiologist/admin), org-scoped patient lists, a
case-review queue, and an audit trail all shipped as a separate surface —
the single-user Records/Patients pages kept working unmodified throughout.

Two production gaps found and fixed after initial ship: `JWT_SECRET` was
falling back to a hardcoded dev default readable in this public repo
(anyone could forge an admin token), and `DATABASE_URL` was unset so the
backend silently ran on SQLite on the container's ephemeral disk — every
redeploy wiped all accounts/patients/audit logs. `JWT_SECRET` now uses
Render's `generateValue` so the real secret never touches the repo;
`DATABASE_URL` still needs a real Postgres/Supabase connection string set
manually in Render's dashboard (`sync: false` keeps it out of the repo on
principle — the backend is already Postgres-ready, `psycopg2-binary` is
installed and `db.py` normalizes `postgres://` URLs correctly).

## Version 5 — Mobile Companion
**Status: not started.** The pragmatic path: make the existing Next.js
frontend installable (manifest + service worker + camera-capture upload
flow), which reuses 100% of the current API with no new backend work. A
native iOS/Android app is a separate codebase and a materially larger
commitment — worth revisiting only if the PWA proves the demand.

---

## Institutional-scale phase (needs external partners — flagged honestly)

These two are listed because they were asked for, but neither can be
**meaningfully built or verified** by a solo/research project without an
institutional partner. Building a hollow version that no real system talks to
would misrepresent the platform's actual capability, so each has an explicit
prerequisite:

- **PACS integration** (DICOM C-STORE/C-FIND, HL7/FHIR) — implementable, but
  untestable without a real PACS server or a vendor sandbox. Prerequisite:
  a hospital/imaging-center partnership or access to a PACS test environment
  (e.g. Orthanc as a stand-in is feasible for a *demo*, but that's not the
  same claim as "integrates with hospital PACS").
- **Federated learning** — the algorithm (federated averaging across sites)
  is buildable and can be *simulated* on synthetic data splits on one
  machine, but that simulation doesn't demonstrate the actual value
  proposition (training across real institutions' private data without
  centralizing it). Prerequisite: 2+ real institutional partners each with
  their own local data and infrastructure.

Both stay on the roadmap as directions, not committed near-term work, until
those prerequisites exist.

---

## What's actually left

1. **v5 (PWA)** — the only untouched version. Small, high-visibility, no
   backend changes.
2. **Mammography** — open, not blocking anything else. Next real step
   (if pursued) is retraining the classifier on auto-derived crops, not
   another localizer architecture (see Version 2 above).
3. **Ensemble deployability** — needs a paid Render tier or a
   smaller/quantized backbone; the code is done and waiting either way.
4. **`DATABASE_URL`** — a five-minute task that only the project owner can
   do (Render dashboard access), not an engineering gap.
5. **Institutional phase** — opportunistic, contingent on a real partner.
