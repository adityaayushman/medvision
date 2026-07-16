"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ClipboardList, Loader2, ShieldAlert, ShieldCheck, Upload } from "lucide-react";
import { useAuth } from "@/components/dashboard/AuthContext";
import { analyzeDashboardStudy, listDashboardPatients, listDashboardStudies } from "@/lib/dashboard-api";
import type { Patient, ReviewStatus, StudyRead } from "@/lib/types";
import { MODALITY_LABELS } from "@/lib/types";
import { REVIEW_STATUS_META, REVIEW_STATUSES } from "@/lib/dashboard-ui";
import { cn, pct } from "@/lib/utils";

const TABS: { key: ReviewStatus | "all"; label: string }[] = [
  { key: "all", label: "All" },
  ...REVIEW_STATUSES.map((s) => ({ key: s, label: REVIEW_STATUS_META[s].label })),
];

export default function QueuePage() {
  const { token } = useAuth();
  const [tab, setTab] = useState<ReviewStatus | "all">("all");
  const [studies, setStudies] = useState<StudyRead[]>([]);
  const [patients, setPatients] = useState<Patient[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    if (!token) return;
    setLoading(true);
    try {
      setStudies(await listDashboardStudies(token, tab === "all" ? undefined : tab));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load queue");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, tab]);

  useEffect(() => {
    if (token) listDashboardPatients(token).then(setPatients).catch(() => setPatients([]));
  }, [token]);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <ClipboardList className="h-5 w-5 text-brand-600 dark:text-brand-400" />
        <h1 className="text-2xl font-bold">Case queue</h1>
      </div>

      {token && <UploadCard token={token} patients={patients} onUploaded={refresh} />}

      <div className="flex flex-wrap items-center gap-1 rounded-full border border-line bg-surface p-1 w-fit">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={cn(
              "rounded-full px-3 py-1.5 text-sm font-medium transition",
              tab === t.key
                ? "bg-brand-500/20 text-brand-700 shadow-[inset_0_0_0_1px_rgba(120,170,255,0.35)] dark:text-brand-200"
                : "text-ink-3 hover:bg-surface-2 hover:text-ink",
            )}
          >
            {t.label}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="grid place-items-center p-10 text-ink-4">
          <Loader2 className="h-5 w-5 animate-spin" />
        </div>
      ) : error ? (
        <p className="text-sm text-bad">{error}</p>
      ) : studies.length === 0 ? (
        <p className="card p-6 text-sm text-ink-4">No studies in this view yet.</p>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {studies.map((s) => (
            <Link key={s.id} href={`/dashboard/studies/${s.id}`} className="card card-hover overflow-hidden">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={s.image_url} alt={`study ${s.id}`} className="h-36 w-full border-b border-line object-cover" />
              <div className="space-y-2 p-4">
                <div className="flex items-center justify-between gap-2">
                  <span className="inline-block rounded-full bg-surface-2 px-2 py-0.5 text-[10px] font-medium text-ink-3">
                    {MODALITY_LABELS[s.modality] ?? s.modality}
                  </span>
                  {s.review_status && (
                    <span className={cn("rounded-full px-2 py-0.5 text-[10px] font-medium", REVIEW_STATUS_META[s.review_status].chip)}>
                      {REVIEW_STATUS_META[s.review_status].label}
                    </span>
                  )}
                </div>
                {s.analysis_stopped ? (
                  <span className="font-semibold text-bad">Analysis stopped</span>
                ) : s.prediction ? (
                  <span className="font-semibold capitalize">
                    {s.prediction.label} <span className="text-xs font-normal text-ink-4">({pct(s.prediction.confidence)})</span>
                  </span>
                ) : (
                  <span className="text-sm text-ink-3">Preprocessing only</span>
                )}
                <div className="flex items-center gap-1 text-xs text-ink-4">
                  {s.quality_passed ? <ShieldCheck className="h-3.5 w-3.5 text-ok" /> : <ShieldAlert className="h-3.5 w-3.5 text-warn" />}
                  {s.patient_name ?? "Unassigned"} · {new Date(s.uploaded_at).toLocaleDateString()}
                </div>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

function UploadCard({
  token,
  patients,
  onUploaded,
}: {
  token: string;
  patients: Patient[];
  onUploaded: () => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [patientId, setPatientId] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setSubmitting(true);
    setError(null);
    try {
      await analyzeDashboardStudy(token, file, patientId ? Number(patientId) : undefined);
      setFile(null);
      setPatientId("");
      onUploaded();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={onSubmit} className="card flex flex-wrap items-end gap-3 p-4">
      <label className="flex flex-col gap-1 text-xs font-medium text-ink-4">
        Scan
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="text-sm text-ink-3"
        />
      </label>
      <label className="flex flex-col gap-1 text-xs font-medium text-ink-4">
        Patient (optional)
        <select value={patientId} onChange={(e) => setPatientId(e.target.value)} className="input">
          <option value="">Unassigned</option>
          {patients.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
      </label>
      <button className="btn-primary" disabled={submitting || !file}>
        {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
        Upload &amp; analyze
      </button>
      {error && <p className="w-full text-sm text-bad">{error}</p>}
    </form>
  );
}
