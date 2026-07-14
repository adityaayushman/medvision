"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Loader2, Upload, ScanLine } from "lucide-react";
import { analyze, getHealth, listPatients } from "@/lib/api";
import type { AnalyzeResponse, Health, Patient } from "@/lib/types";
import { MODALITY_LABELS } from "@/lib/types";
import { pct } from "@/lib/utils";
import { QualityScorePanel } from "@/components/QualityScorePanel";
import { ProcessingTimeline } from "@/components/ProcessingTimeline";
import { RoiViewer } from "@/components/RoiViewer";
import { AnalysisStoppedBanner } from "@/components/AnalysisStoppedBanner";
import { ProcessingMetadataPanel } from "@/components/ProcessingMetadataPanel";

export default function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [patients, setPatients] = useState<Patient[]>([]);
  const [patientId, setPatientId] = useState<string>("");
  const [health, setHealth] = useState<Health | null>(null);
  const [modality, setModality] = useState("chest_xray");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    listPatients().then(setPatients).catch(() => setPatients([]));
    getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  const modalities = health ? Object.keys(health.modalities) : ["chest_xray"];

  function pick(f: File | null) {
    setError(null);
    setResult(null);
    setFile(f);
    setPreview(f ? URL.createObjectURL(f) : null);
  }

  async function run() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await analyze(file, patientId ? Number(patientId) : undefined, modality));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Analyze a scan</h1>
        <p className="text-sm text-ink-3">
          Upload a medical image. The pipeline runs quality assessment, enhancement,
          ROI extraction and — if the scan clears the quality gate — classification with Grad-CAM.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="card p-6">
          <div
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              pick(e.dataTransfer.files?.[0] ?? null);
            }}
            className="flex min-h-[260px] cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-line-2 p-4 text-center transition hover:border-brand-400/60"
          >
            {preview ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={preview} alt="preview" className="max-h-[240px] rounded-lg object-contain" />
            ) : (
              <>
                <Upload className="mb-2 h-8 w-8 text-ink-4" />
                <p className="text-sm font-medium">Click or drop an image here</p>
                <p className="text-xs text-ink-4">PNG / JPG — pick the modality below</p>
              </>
            )}
          </div>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => pick(e.target.files?.[0] ?? null)}
          />

          <label className="mt-4 flex flex-col gap-1 text-xs font-medium text-ink-4">
            Modality
            <select value={modality} onChange={(e) => setModality(e.target.value)} className="input">
              {modalities.map((m) => (
                <option key={m} value={m}>
                  {MODALITY_LABELS[m] ?? m}
                  {health && !health.modalities[m] ? " (preprocessing only — no model loaded)" : ""}
                </option>
              ))}
            </select>
          </label>

          <label className="mt-4 flex flex-col gap-1 text-xs font-medium text-ink-4">
            Attach to patient (optional)
            <select value={patientId} onChange={(e) => setPatientId(e.target.value)} className="input">
              <option value="">— none (saved to records only) —</option>
              {patients.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </label>

          <button onClick={run} disabled={!file || loading} className="btn-primary mt-4 w-full">
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <ScanLine className="h-4 w-4" />}
            {loading ? "Analyzing…" : "Analyze"}
          </button>
          {error && <p className="mt-3 text-sm text-bad">{error}</p>}
        </div>

        <div className="card p-6">
          {!result ? (
            <div className="grid h-full min-h-[260px] place-items-center text-sm text-ink-4">
              Results appear here.
            </div>
          ) : (
            <QuickSummary result={result} />
          )}
        </div>
      </div>

      {result && (
        <div className="space-y-4">
          <ProcessingTimeline steps={result.pipeline_steps} />

          <div className="grid gap-4 lg:grid-cols-2">
            <QualityScorePanel quality={result.quality} />
            <ProcessingMetadataPanel meta={result.processing_metadata} />
          </div>

          {result.analysis_stopped ? (
            <AnalysisStoppedBanner quality={result.quality} />
          ) : result.prediction ? (
            <PredictionCard result={result} />
          ) : (
            <p className="rounded-lg bg-surface p-3 text-xs text-ink-3">
              No trained model loaded — showing preprocessing only.
            </p>
          )}

          {result.stages && result.stages.length > 0 && (
            <div>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-4">
                Processing pipeline — interactive viewer
              </p>
              <RoiViewer stages={result.stages} />
            </div>
          )}

          <Link href="/records" className="block text-center text-xs text-brand-600 hover:underline dark:text-brand-400">
            Saved to records →
          </Link>
        </div>
      )}
    </div>
  );
}

/** Compact panel next to the upload box — full detail lives below in the report. */
function QuickSummary({ result }: { result: AnalyzeResponse }) {
  const q = result.quality;
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold uppercase tracking-wide text-ink-4">Quality score</span>
        <span
          className={`text-2xl font-bold tabular-nums ${q.passed ? "text-ok" : "text-bad"}`}
        >
          {q.overall_score}
        </span>
      </div>
      {result.analysis_stopped ? (
        <p className="text-sm font-semibold text-bad">Analysis stopped — quality gate failed.</p>
      ) : result.prediction ? (
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-ink-4">Prediction</p>
          <p className="text-xl font-bold capitalize">{result.prediction.label}</p>
          <p className="text-xs text-ink-4">{pct(result.prediction.confidence)} confidence</p>
        </div>
      ) : (
        <p className="text-sm text-ink-3">Preprocessing only — no model loaded.</p>
      )}
      <p className="text-xs text-ink-4">Full report below.</p>
    </div>
  );
}

function PredictionCard({ result }: { result: AnalyzeResponse }) {
  const pred = result.prediction!;
  return (
    <div className="card p-5">
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-sm font-semibold uppercase tracking-wide text-ink-4">Prediction</span>
        {pred.backbone && <span className="text-xs text-ink-4">{pred.backbone}</span>}
      </div>
      <p className="text-2xl font-bold capitalize">{pred.label}</p>
      <div className="mt-3 space-y-2">
        {Object.entries(pred.probabilities)
          .sort((a, b) => b[1] - a[1])
          .map(([label, p]) => (
            <div key={label}>
              <div className="mb-0.5 flex justify-between text-xs">
                <span className="capitalize">{label}</span>
                <span>{pct(p)}</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-surface-2">
                <div className="h-full rounded-full bg-brand-500" style={{ width: pct(p) }} />
              </div>
            </div>
          ))}
      </div>
    </div>
  );
}
