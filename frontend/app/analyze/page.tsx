"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Loader2, Upload, ShieldAlert, ShieldCheck, ScanLine } from "lucide-react";
import { analyze, listPatients } from "@/lib/api";
import type { AnalyzeResponse, Patient } from "@/lib/types";
import { cn, pct } from "@/lib/utils";

export default function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [patients, setPatients] = useState<Patient[]>([]);
  const [patientId, setPatientId] = useState<string>("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    listPatients().then(setPatients).catch(() => setPatients([]));
  }, []);

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
      setResult(await analyze(file, patientId ? Number(patientId) : undefined));
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
          ROI extraction and — if a model is trained — classification with Grad-CAM.
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
                <p className="text-xs text-ink-4">PNG / JPG chest X-ray</p>
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
            <Results result={result} />
          )}
        </div>
      </div>
    </div>
  );
}

function Results({ result }: { result: AnalyzeResponse }) {
  const q = result.quality;
  return (
    <div className="space-y-5">
      <div
        className={cn(
          "flex items-start gap-3 rounded-xl p-3 text-sm",
          q.passed ? "chip-ok" : "chip-warn",
        )}
      >
        {q.passed ? <ShieldCheck className="h-5 w-5" /> : <ShieldAlert className="h-5 w-5" />}
        <div>
          <p className="font-semibold">Image quality: {q.passed ? "passed" : "flagged"}</p>
          {q.reasons.length > 0 && <p className="text-xs">{q.reasons.join("; ")}</p>}
          <p className="mt-1 text-xs opacity-80">
            focus {q.focus.toFixed(0)} · brightness {q.brightness.toFixed(0)} · contrast{" "}
            {q.contrast.toFixed(0)}
          </p>
        </div>
      </div>

      {result.prediction ? (
        <div>
          <div className="mb-1 flex items-baseline justify-between">
            <span className="text-sm font-semibold uppercase tracking-wide text-ink-4">
              Prediction
            </span>
            {result.prediction.backbone && (
              <span className="text-xs text-ink-4">{result.prediction.backbone}</span>
            )}
          </div>
          <p className="text-2xl font-bold capitalize">{result.prediction.label}</p>
          <div className="mt-3 space-y-2">
            {Object.entries(result.prediction.probabilities)
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
      ) : (
        <p className="rounded-lg bg-surface p-3 text-xs text-ink-3">
          No trained model loaded — showing preprocessing only.
        </p>
      )}

      {/* full DIP pipeline gallery */}
      <div>
        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-4">
          Processing pipeline
        </p>
        {result.stages && result.stages.length > 0 ? (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            {result.stages.map((s, i) => (
              <Figure key={s.name} title={`${i + 1}. ${s.name}`} src={s.url} />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-3">
            <Figure title={`ROIs (${result.num_rois})`} src={result.annotated_url} />
            {result.heatmap_url && <Figure title="Grad-CAM" src={result.heatmap_url} />}
          </div>
        )}
      </div>

      <Link href="/records" className="block text-center text-xs text-brand-600 dark:text-brand-400 hover:underline">
        Saved to records →
      </Link>
    </div>
  );
}

function Figure({ title, src }: { title: string; src: string }) {
  return (
    <div>
      <p className="mb-1 text-[11px] font-medium text-ink-4">{title}</p>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt={title} className="aspect-square w-full rounded-lg border border-line object-cover" />
    </div>
  );
}
