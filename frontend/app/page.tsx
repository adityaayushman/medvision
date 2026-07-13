import Link from "next/link";
import {
  ScanLine,
  Sparkles,
  Layers,
  Eye,
  History,
  ShieldCheck,
} from "lucide-react";

const PIPELINE = [
  "Quality check",
  "Denoise + CLAHE",
  "Segmentation",
  "ROI extraction",
  "VGG16 classify",
  "Grad-CAM",
  "Patient record",
];

const FEATURES = [
  {
    icon: Layers,
    title: "Full DIP pipeline",
    body: "Every scan is quality-gated, denoised, CLAHE-enhanced, segmented and reduced to regions of interest before the model ever sees it.",
  },
  {
    icon: Eye,
    title: "Explainable, not opaque",
    body: "Grad-CAM overlays show where the model looked, alongside calibrated class probabilities — the why, not just the what.",
  },
  {
    icon: History,
    title: "Longitudinal record",
    body: "Each patient accumulates a timeline of studies and predictions, turning one-shot diagnosis into disease monitoring.",
  },
  {
    icon: Sparkles,
    title: "Modality-agnostic",
    body: "Chest X-ray ships first; brain MRI and mammography are configuration presets, not rewrites.",
  },
];

export default function Home() {
  return (
    <div className="space-y-12">
      <section className="card overflow-hidden">
        <div className="bg-gradient-to-br from-brand-600 to-brand-800 px-8 py-12 text-white">
          <div className="mb-3 inline-flex items-center gap-2 rounded-full bg-white/15 px-3 py-1 text-xs font-medium">
            <ShieldCheck className="h-3.5 w-3.5" /> Research &amp; educational · not for clinical use
          </div>
          <h1 className="max-w-2xl text-4xl font-bold leading-tight">
            Medical imaging intelligence, from raw scan to explainable insight.
          </h1>
          <p className="mt-4 max-w-2xl text-brand-50/90">
            MedChron AI processes a medical image, extracts the clinically relevant
            region, classifies it with an explainable model, and files the result
            into a longitudinal patient record.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Link href="/analyze" className="btn bg-white text-brand-700 hover:bg-brand-50">
              <ScanLine className="h-4 w-4" /> Analyze a scan
            </Link>
            <Link href="/patients" className="btn border border-white/30 text-white hover:bg-white/10">
              View patients
            </Link>
          </div>
        </div>
      </section>

      <section>
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-500">
          The pipeline
        </h2>
        <div className="flex flex-wrap items-center gap-2">
          {PIPELINE.map((step, i) => (
            <div key={step} className="flex items-center gap-2">
              <span className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm font-medium dark:border-slate-800 dark:bg-slate-900">
                {step}
              </span>
              {i < PIPELINE.length - 1 && <span className="text-slate-400">→</span>}
            </div>
          ))}
        </div>
      </section>

      <section className="grid gap-4 sm:grid-cols-2">
        {FEATURES.map(({ icon: Icon, title, body }) => (
          <div key={title} className="card p-6">
            <div className="mb-3 grid h-10 w-10 place-items-center rounded-xl bg-brand-50 text-brand-600 dark:bg-brand-900/40 dark:text-brand-300">
              <Icon className="h-5 w-5" />
            </div>
            <h3 className="text-lg font-semibold">{title}</h3>
            <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">{body}</p>
          </div>
        ))}
      </section>
    </div>
  );
}
