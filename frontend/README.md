# MedChron AI — Frontend (Next.js)

Dashboard for the MedChron platform: upload a scan, see quality/ROI/prediction/
Grad-CAM, and browse a patient's study timeline.

> Research/educational software. **Not for clinical use.**

## Run

```bash
npm install
npm run dev          # http://localhost:3000
```

The dev server proxies `/api/*` and `/static/*` to the FastAPI backend
(`http://localhost:8000` by default — set `BACKEND_URL` to change). So start the
backend first, then the frontend, and everything is same-origin in the browser.

## Pages
- `/` — overview of the pipeline and platform
- `/analyze` — upload → quality, ROIs, prediction, Grad-CAM overlay
- `/patients` — create / list patients
- `/patients/[id]` — chronological study timeline (disease monitoring)

## Stack
Next.js 15 (App Router) · React 19 · TypeScript · Tailwind CSS · lucide-react.
Verified: `npm run build` compiles all routes with no type errors.
