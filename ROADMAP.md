# MedChron AI — Roadmap

This document tracks the platform's version progression beyond the current
release. Each phase is scoped by what it actually takes to build and verify —
not just what it takes to describe — so priorities can be set honestly.

> Current release (v1): chest X-ray only, single EfficientNet-B0 model,
> quality-gated inference, Grad-CAM, patient records, live at
> [medchron-ai.vercel.app](https://medchron-ai.vercel.app). See `README.md`
> for the present-tense status table.

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
**Feasibility: High.** The architecture was built for this — `PRESETS` in
[`config.py`](ml/src/medchron/config.py) already has `brain_mri` and
`mammography` entries; they've never been paired with a trained model.

| Modality | What's missing | Effort |
|---|---|---|
| Brain MRI (tumor classification) | Dataset (e.g. Br35H, Figshare brain tumor set) + training run | 1 training pipeline run |
| Mammography (lesion classification) | Dataset (e.g. CBIS-DDSM, MIAS) + training run | 1 training pipeline run |

**Build shape:** a modality switcher on the Analyze page → routes to the
matching config preset + checkpoint. The DIP pipeline, quality gate, Grad-CAM,
and patient records all already work modality-agnostically — this phase is
almost entirely data + training, not new engineering.

## Version 3 — AI-Assisted Reporting & Model Improvements
**Feasibility: High.** Natural extension of the existing prediction payload.

- **AI-assisted report drafting** — turn the existing `processing_metadata` +
  `quality` + `prediction` payload into a structured draft report (PDF/text),
  explicitly labeled **"AI Draft — Requires Clinician Review"** with no
  auto-finalization path. This is a reporting/export feature, not a new
  diagnostic claim.
- **Multi-model ensemble** — the backbone layer already supports
  VGG16/ResNet50/DenseNet121/EfficientNet-B0 interchangeably
  ([`backbone.py`](ml/src/medchron/models/backbone.py)); ensembling means
  training 2-3 backbones on the same split and averaging/voting at inference,
  surfaced as an "ensemble confidence" alongside the existing single-model one.

## Version 4 — Clinical Dashboard for Hospitals
**Feasibility: Medium.** Buildable solo, but the first phase that requires
a genuinely new subsystem: **the platform currently has zero authentication.**
Before any multi-user/organizational feature, that has to exist — this
is the actual blocker, not the dashboard UI itself.

- Auth (roles: radiologist / admin), org-scoped patient lists, a case-review
  queue, and an audit trail on who viewed/attached what.
- Ships as a separate `/dashboard` surface; the existing single-user
  Records/Patients pages keep working for the non-org use case.

## Version 5 — Mobile Companion
**Feasibility: High as a PWA, low as native apps without a dedicated mobile
team.** The pragmatic path: make the existing Next.js frontend installable
(manifest + service worker + camera-capture upload flow), which reuses 100%
of the current API with no new backend work. A native iOS/Android app is a
separate codebase and a materially larger commitment — worth revisiting only
if the PWA proves the demand.

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

## Suggested sequencing

1. **v2 (multi-modality)** — highest leverage-to-effort ratio; the codebase
   is already waiting for it.
2. **v3 (reporting + ensemble)** — deepens v1 without new infrastructure.
3. **v5 (PWA)** — small, high-visibility, no backend changes.
4. **v4 (clinical dashboard)** — once auth exists, everything else in v4
   builds on it.
5. **Institutional phase** — opportunistic, contingent on a real partner.
