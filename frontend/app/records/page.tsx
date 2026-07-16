"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { FileStack, FileText, Loader2, ScanLine, ShieldAlert, ShieldCheck, UserPlus } from "lucide-react";
import { assignPatient, listPatients, listStudies } from "@/lib/api";
import type { Patient, StudyRead } from "@/lib/types";
import { MODALITY_LABELS } from "@/lib/types";
import { cn, pct } from "@/lib/utils";

export default function RecordsPage() {
  const [studies, setStudies] = useState<StudyRead[]>([]);
  const [patients, setPatients] = useState<Patient[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listStudies()
      .then(setStudies)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load records"))
      .finally(() => setLoading(false));
    listPatients().then(setPatients).catch(() => setPatients([]));
  }, []);

  async function attach(studyId: number, patientId: number) {
    const updated = await assignPatient(studyId, patientId);
    setStudies((prev) => prev.map((s) => (s.id === studyId ? updated : s)));
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <FileStack className="h-5 w-5 text-brand-600 dark:text-brand-400" />
          <h1 className="text-2xl font-bold">Analyzed records</h1>
        </div>
        <Link href="/analyze" className="btn-primary">
          <ScanLine className="h-4 w-4" /> New analysis
        </Link>
      </div>
      <p className="max-w-2xl text-sm text-ink-3">
        Every scan run through the analyzer, newest first — across all patients
        and unassigned ones. {studies.length > 0 && `${studies.length} total.`}
      </p>

      {loading ? (
        <div className="grid place-items-center p-10 text-ink-4">
          <Loader2 className="h-5 w-5 animate-spin" />
        </div>
      ) : error ? (
        <p className="text-sm text-bad">{error}</p>
      ) : studies.length === 0 ? (
        <div className="card p-8 text-center">
          <p className="text-sm font-medium text-ink">No analyzed records yet.</p>
          <p className="mx-auto mt-1 max-w-md text-sm text-ink-4">
            Run a scan on the{" "}
            <Link href="/analyze" className="text-brand-600 dark:text-brand-400 underline">
              Analyze
            </Link>{" "}
            page and it will appear here.
          </p>
          <p className="mx-auto mt-4 max-w-md rounded-lg bg-surface p-3 text-xs text-ink-4">
            Records and every pipeline image are stored persistently in the
            database — they survive server restarts and redeploys.
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
                className="h-40 w-full border-b border-line object-cover"
              />
              <div className="p-4">
                <span className="mb-2 inline-block rounded-full bg-surface-2 px-2 py-0.5 text-[10px] font-medium text-ink-3">
                  {MODALITY_LABELS[s.modality] ?? s.modality}
                </span>
                <div className="flex items-center justify-between gap-2">
                  {s.analysis_stopped ? (
                    <span className="font-semibold text-bad">Analysis stopped</span>
                  ) : s.prediction ? (
                    <span className="font-semibold capitalize">{s.prediction.label}</span>
                  ) : (
                    <span className="text-sm text-ink-3">Preprocessing only</span>
                  )}
                  {s.prediction && !s.analysis_stopped && (
                    <span className="text-xs tabular-nums text-ink-3">
                      {pct(s.prediction.confidence)}
                    </span>
                  )}
                </div>
                <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-ink-4">
                  <span
                    className={cn(
                      "inline-flex items-center gap-1",
                      s.quality_passed ? "text-ok" : "text-warn",
                    )}
                  >
                    {s.quality_passed ? (
                      <ShieldCheck className="h-3.5 w-3.5" />
                    ) : (
                      <ShieldAlert className="h-3.5 w-3.5" />
                    )}
                    {s.quality_score != null ? `quality ${s.quality_score}/100` : s.quality_passed ? "quality ok" : "flagged"}
                  </span>
                  <span>·</span>
                  <span>{s.num_rois} ROI{s.num_rois === 1 ? "" : "s"}</span>
                </div>

                <div className="mt-2">
                  {s.patient_id ? (
                    <Link
                      href={`/patients/${s.patient_id}`}
                      className="text-xs font-medium text-brand-600 hover:underline dark:text-brand-400"
                    >
                      {s.patient_name ?? `Patient #${s.patient_id}`}
                    </Link>
                  ) : patients.length > 0 ? (
                    <AttachPicker
                      patients={patients}
                      onAttach={(pid) => attach(s.id, pid)}
                    />
                  ) : (
                    <span className="text-xs text-ink-5">No patient attached</span>
                  )}
                </div>

                <div className="mt-2 flex items-center justify-between">
                  <p className="text-[11px] text-ink-5">
                    {new Date(s.uploaded_at).toLocaleString()}
                  </p>
                  <Link
                    href={`/records/${s.id}`}
                    className="inline-flex items-center gap-1 text-xs font-medium text-brand-600 hover:underline dark:text-brand-400"
                  >
                    <FileText className="h-3.5 w-3.5" /> Report
                  </Link>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function AttachPicker({
  patients,
  onAttach,
}: {
  patients: Patient[];
  onAttach: (patientId: number) => Promise<void>;
}) {
  const [busy, setBusy] = useState(false);

  return (
    <label className="inline-flex items-center gap-1.5 text-xs text-ink-4">
      <UserPlus className="h-3.5 w-3.5" />
      <select
        disabled={busy}
        defaultValue=""
        onChange={async (e) => {
          const pid = Number(e.target.value);
          if (!pid) return;
          setBusy(true);
          try {
            await onAttach(pid);
          } finally {
            setBusy(false);
          }
        }}
        className="rounded-md border border-line bg-surface px-1.5 py-0.5 text-xs text-ink-3"
      >
        <option value="" disabled>
          {busy ? "Attaching…" : "Attach to patient…"}
        </option>
        {patients.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name}
          </option>
        ))}
      </select>
    </label>
  );
}
