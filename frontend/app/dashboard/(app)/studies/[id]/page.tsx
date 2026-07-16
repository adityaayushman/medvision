"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, Loader2, ShieldAlert, ShieldCheck, UserPlus } from "lucide-react";
import { useAuth } from "@/components/dashboard/AuthContext";
import { assignDashboardPatient, getDashboardStudy, listDashboardPatients, updateReviewStatus } from "@/lib/dashboard-api";
import type { Patient, ReviewStatus, StudyRead } from "@/lib/types";
import { MODALITY_LABELS } from "@/lib/types";
import { REVIEW_STATUS_META, REVIEW_STATUSES } from "@/lib/dashboard-ui";
import { cn, pct } from "@/lib/utils";

export default function DashboardStudyDetail({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { token } = useAuth();
  const [study, setStudy] = useState<StudyRead | null>(null);
  const [patients, setPatients] = useState<Patient[]>([]);
  const [note, setNote] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  async function refresh() {
    if (!token) return;
    try {
      setStudy(await getDashboardStudy(token, Number(id)));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load study");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
    if (token) listDashboardPatients(token).then(setPatients).catch(() => setPatients([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, id]);

  async function setStatus(status: ReviewStatus) {
    if (!token) return;
    setSaving(true);
    try {
      setStudy(await updateReviewStatus(token, Number(id), status, note || undefined));
      setNote("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to update review status");
    } finally {
      setSaving(false);
    }
  }

  async function attach(patientId: number) {
    if (!token) return;
    setStudy(await assignDashboardPatient(token, Number(id), patientId));
  }

  if (loading) {
    return (
      <div className="grid place-items-center p-10 text-ink-4">
        <Loader2 className="h-5 w-5 animate-spin" />
      </div>
    );
  }
  if (error || !study) {
    return <p className="text-sm text-bad">{error ?? "Study not found"}</p>;
  }

  return (
    <div className="space-y-6">
      <Link href="/dashboard/queue" className="inline-flex items-center gap-1 text-sm text-ink-4 hover:text-brand-600 dark:hover:text-brand-400">
        <ArrowLeft className="h-4 w-4" /> Queue
      </Link>

      <div className="grid gap-6 md:grid-cols-2">
        <div className="card overflow-hidden">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={study.image_url} alt={`study ${study.id}`} className="max-h-[420px] w-full object-contain" />
        </div>

        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <span className="rounded-full bg-surface-2 px-2 py-0.5 text-[11px] font-medium text-ink-3">
              {MODALITY_LABELS[study.modality] ?? study.modality}
            </span>
            {study.review_status && (
              <span className={cn("rounded-full px-2 py-0.5 text-[11px] font-medium", REVIEW_STATUS_META[study.review_status].chip)}>
                {REVIEW_STATUS_META[study.review_status].label}
              </span>
            )}
          </div>

          {study.analysis_stopped ? (
            <p className="text-lg font-semibold text-bad">Analysis stopped</p>
          ) : study.prediction ? (
            <p className="text-lg font-semibold capitalize">
              {study.prediction.label}{" "}
              <span className="text-sm font-normal text-ink-4">({pct(study.prediction.confidence)})</span>
            </p>
          ) : (
            <p className="text-sm text-ink-3">Preprocessing only — no trained model for this modality.</p>
          )}

          <div className="flex items-center gap-2 text-sm text-ink-4">
            {study.quality_passed ? <ShieldCheck className="h-4 w-4 text-ok" /> : <ShieldAlert className="h-4 w-4 text-warn" />}
            {study.quality_score != null ? `Quality ${study.quality_score}/100` : study.quality_passed ? "Quality ok" : "Quality flagged"}
            <span>·</span>
            {study.num_rois} ROI{study.num_rois === 1 ? "" : "s"}
          </div>

          <p className="text-xs text-ink-4">{new Date(study.uploaded_at).toLocaleString()}</p>

          <div>
            {study.patient_id ? (
              <Link
                href={`/dashboard/patients/${study.patient_id}`}
                className="text-sm font-medium text-brand-600 hover:underline dark:text-brand-400"
              >
                {study.patient_name ?? `Patient #${study.patient_id}`}
              </Link>
            ) : (
              <PatientPicker patients={patients} onAttach={attach} />
            )}
          </div>

          <div className="card space-y-3 p-4">
            <p className="text-sm font-semibold">Review</p>
            <textarea
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="Optional note for this transition…"
              className="input min-h-[70px] w-full resize-y"
            />
            <div className="flex flex-wrap gap-2">
              {REVIEW_STATUSES.map((s) => (
                <button
                  key={s}
                  disabled={saving || study.review_status === s}
                  onClick={() => setStatus(s)}
                  className="btn-ghost text-xs disabled:opacity-40"
                >
                  Mark {REVIEW_STATUS_META[s].label}
                </button>
              ))}
            </div>
            {study.review_note && (
              <p className="text-xs text-ink-4">
                Last note ({study.reviewed_by}, {study.reviewed_at && new Date(study.reviewed_at).toLocaleString()}):{" "}
                {study.review_note}
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function PatientPicker({ patients, onAttach }: { patients: Patient[]; onAttach: (id: number) => Promise<void> }) {
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
