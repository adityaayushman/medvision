"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, Loader2, Activity } from "lucide-react";
import { getTimeline } from "@/lib/api";
import type { StudyRead } from "@/lib/types";
import { pct } from "@/lib/utils";

export default function PatientTimeline({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [studies, setStudies] = useState<StudyRead[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getTimeline(Number(id))
      .then(setStudies)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load timeline"))
      .finally(() => setLoading(false));
  }, [id]);

  return (
    <div className="space-y-6">
      <Link href="/patients" className="inline-flex items-center gap-1 text-sm text-slate-500 hover:text-brand-400">
        <ArrowLeft className="h-4 w-4" /> All patients
      </Link>

      <div className="flex items-center gap-2">
        <Activity className="h-5 w-5 text-brand-400" />
        <h1 className="text-2xl font-bold">Study timeline</h1>
      </div>

      {loading ? (
        <div className="grid place-items-center p-10 text-slate-500">
          <Loader2 className="h-5 w-5 animate-spin" />
        </div>
      ) : error ? (
        <p className="text-sm text-red-400">{error}</p>
      ) : studies.length === 0 ? (
        <p className="card p-6 text-sm text-slate-500">
          No studies yet for this patient. Upload one on the{" "}
          <Link href="/analyze" className="text-brand-400 underline">Analyze</Link> page.
        </p>
      ) : (
        <ol className="relative space-y-4 border-l border-white/10 pl-6">
          {studies.map((s) => (
            <li key={s.id} className="relative">
              <span className="absolute -left-[29px] top-2 h-3 w-3 rounded-full bg-brand-500 ring-4 ring-[#05070e]" />
              <div className="card flex flex-wrap items-center gap-4 p-4">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={s.image_url} alt={`study ${s.id}`}
                  className="h-20 w-20 rounded-lg border border-white/10 object-cover" />
                <div className="min-w-0 flex-1">
                  <p className="text-xs text-slate-500">
                    {new Date(s.uploaded_at).toLocaleString()} · {s.modality}
                  </p>
                  {s.prediction ? (
                    <p className="font-semibold capitalize">
                      {s.prediction.label}{" "}
                      <span className="text-sm font-normal text-slate-500">
                        ({pct(s.prediction.confidence)})
                      </span>
                    </p>
                  ) : (
                    <p className="text-sm text-slate-500">Preprocessing only</p>
                  )}
                  <p className="text-xs text-slate-500">
                    {s.num_rois} ROI{s.num_rois === 1 ? "" : "s"} ·{" "}
                    {s.quality_passed ? "quality passed" : "quality flagged"}
                  </p>
                </div>
              </div>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
