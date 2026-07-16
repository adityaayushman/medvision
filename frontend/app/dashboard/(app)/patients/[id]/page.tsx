"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import { Activity, ArrowLeft, Loader2 } from "lucide-react";
import { useAuth } from "@/components/dashboard/AuthContext";
import { getDashboardPatient, getDashboardTimeline } from "@/lib/dashboard-api";
import type { Patient, StudyRead } from "@/lib/types";
import { REVIEW_STATUS_META } from "@/lib/dashboard-ui";
import { cn, pct } from "@/lib/utils";

export default function DashboardPatientTimeline({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { token } = useAuth();
  const [patient, setPatient] = useState<Patient | null>(null);
  const [studies, setStudies] = useState<StudyRead[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!token) return;
    getDashboardPatient(token, Number(id)).then(setPatient).catch(() => setPatient(null));
    getDashboardTimeline(token, Number(id))
      .then(setStudies)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load timeline"))
      .finally(() => setLoading(false));
  }, [token, id]);

  return (
    <div className="space-y-6">
      <Link href="/dashboard/patients" className="inline-flex items-center gap-1 text-sm text-ink-4 hover:text-brand-600 dark:hover:text-brand-400">
        <ArrowLeft className="h-4 w-4" /> All patients
      </Link>

      <div className="flex items-center gap-2">
        <Activity className="h-5 w-5 text-brand-600 dark:text-brand-400" />
        <h1 className="text-2xl font-bold">{patient ? patient.name : "Study timeline"}</h1>
      </div>
      {patient && (
        <p className="text-sm text-ink-4">
          {[patient.sex, patient.birth_year].filter(Boolean).join(" · ") || "—"} · {patient.study_count} scan
          {patient.study_count === 1 ? "" : "s"} on record
        </p>
      )}

      {loading ? (
        <div className="grid place-items-center p-10 text-ink-4">
          <Loader2 className="h-5 w-5 animate-spin" />
        </div>
      ) : error ? (
        <p className="text-sm text-bad">{error}</p>
      ) : studies.length === 0 ? (
        <p className="card p-6 text-sm text-ink-4">
          No studies yet for this patient. Upload one on the{" "}
          <Link href="/dashboard/queue" className="text-brand-600 dark:text-brand-400 underline">
            Queue
          </Link>{" "}
          page.
        </p>
      ) : (
        <ol className="relative space-y-4 border-l border-line pl-6">
          {studies.map((s) => (
            <li key={s.id} className="relative">
              <span className="absolute -left-[29px] top-2 h-3 w-3 rounded-full bg-brand-500 ring-4 ring-page" />
              <Link href={`/dashboard/studies/${s.id}`} className="card card-hover flex flex-wrap items-center gap-4 p-4">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={s.image_url} alt={`study ${s.id}`} className="h-20 w-20 rounded-lg border border-line object-cover" />
                <div className="min-w-0 flex-1">
                  <p className="text-xs text-ink-4">
                    {new Date(s.uploaded_at).toLocaleString()} · {s.modality}
                  </p>
                  {s.prediction ? (
                    <p className="font-semibold capitalize">
                      {s.prediction.label} <span className="text-sm font-normal text-ink-4">({pct(s.prediction.confidence)})</span>
                    </p>
                  ) : (
                    <p className="text-sm text-ink-4">Preprocessing only</p>
                  )}
                </div>
                {s.review_status && (
                  <span className={cn("rounded-full px-2 py-0.5 text-[11px] font-medium", REVIEW_STATUS_META[s.review_status].chip)}>
                    {REVIEW_STATUS_META[s.review_status].label}
                  </span>
                )}
              </Link>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
