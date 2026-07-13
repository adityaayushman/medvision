"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { FileStack, Loader2, ScanLine, ShieldAlert, ShieldCheck } from "lucide-react";
import { listStudies } from "@/lib/api";
import type { StudyRead } from "@/lib/types";
import { cn, pct } from "@/lib/utils";

export default function RecordsPage() {
  const [studies, setStudies] = useState<StudyRead[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listStudies()
      .then(setStudies)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load records"))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <FileStack className="h-5 w-5 text-brand-400" />
          <h1 className="text-2xl font-bold">Analyzed records</h1>
        </div>
        <Link href="/analyze" className="btn-primary">
          <ScanLine className="h-4 w-4" /> New analysis
        </Link>
      </div>
      <p className="max-w-2xl text-sm text-slate-400">
        Every scan run through the analyzer, newest first — across all patients
        and unassigned ones.
      </p>

      {loading ? (
        <div className="grid place-items-center p-10 text-slate-500">
          <Loader2 className="h-5 w-5 animate-spin" />
        </div>
      ) : error ? (
        <p className="text-sm text-red-400">{error}</p>
      ) : studies.length === 0 ? (
        <div className="card p-8 text-center">
          <p className="text-sm font-medium text-slate-200">No analyzed records yet.</p>
          <p className="mx-auto mt-1 max-w-md text-sm text-slate-500">
            Run a scan on the{" "}
            <Link href="/analyze" className="text-brand-400 underline">
              Analyze
            </Link>{" "}
            page and it will appear here.
          </p>
          <p className="mx-auto mt-4 max-w-md rounded-lg border border-amber-400/20 bg-amber-400/[0.06] p-3 text-xs text-amber-200/90">
            Note: the demo backend runs on a free tier with temporary storage —
            records reset when the server restarts or after it sleeps (~15 min
            idle). Persistent storage is a configuration upgrade.
          </p>
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {studies.map((s) => (
            <div key={s.id} className="card card-hover overflow-hidden">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={s.image_url}
                alt={`study ${s.id}`}
                className="h-40 w-full border-b border-white/10 object-cover"
              />
              <div className="p-4">
                <div className="flex items-center justify-between gap-2">
                  {s.prediction ? (
                    <span className="font-semibold capitalize">{s.prediction.label}</span>
                  ) : (
                    <span className="text-sm text-slate-400">Preprocessing only</span>
                  )}
                  {s.prediction && (
                    <span className="text-xs tabular-nums text-slate-400">
                      {pct(s.prediction.confidence)}
                    </span>
                  )}
                </div>
                <div className="mt-2 flex items-center gap-2 text-xs text-slate-500">
                  <span
                    className={cn(
                      "inline-flex items-center gap-1",
                      s.quality_passed ? "text-emerald-400" : "text-amber-400",
                    )}
                  >
                    {s.quality_passed ? (
                      <ShieldCheck className="h-3.5 w-3.5" />
                    ) : (
                      <ShieldAlert className="h-3.5 w-3.5" />
                    )}
                    {s.quality_passed ? "quality ok" : "flagged"}
                  </span>
                  <span>·</span>
                  <span>{s.num_rois} ROI{s.num_rois === 1 ? "" : "s"}</span>
                  {s.patient_id && (
                    <>
                      <span>·</span>
                      <Link href={`/patients/${s.patient_id}`} className="text-brand-400 hover:underline">
                        patient #{s.patient_id}
                      </Link>
                    </>
                  )}
                </div>
                <p className="mt-2 text-[11px] text-slate-600">
                  {new Date(s.uploaded_at).toLocaleString()}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
