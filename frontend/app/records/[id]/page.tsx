"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  Download,
  FileText,
  Loader2,
  ShieldAlert,
  ShieldCheck,
} from "lucide-react";
import { getReport, getStudy, reportPdfUrl } from "@/lib/api";
import type { ReportRead, StudyRead } from "@/lib/types";
import { MODALITY_LABELS } from "@/lib/types";
import { cn, pct } from "@/lib/utils";

export default function StudyReportPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const studyId = Number(id);
  const [study, setStudy] = useState<StudyRead | null>(null);
  const [report, setReport] = useState<ReportRead | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getStudy(studyId), getReport(studyId)])
      .then(([s, r]) => {
        setStudy(s);
        setReport(r);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load report"))
      .finally(() => setLoading(false));
  }, [studyId]);

  if (loading) {
    return (
      <div className="grid place-items-center p-10 text-ink-4">
        <Loader2 className="h-5 w-5 animate-spin" />
      </div>
    );
  }
  if (error || !study || !report) {
    return <p className="text-sm text-bad">{error ?? "Report not available."}</p>;
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div className="flex items-center justify-between gap-3">
        <Link
          href="/records"
          className="inline-flex items-center gap-1 text-sm text-ink-4 hover:text-brand-600 dark:hover:text-brand-400"
        >
          <ArrowLeft className="h-4 w-4" /> All records
        </Link>
        <a href={reportPdfUrl(studyId)} target="_blank" rel="noopener noreferrer" className="btn-primary">
          <Download className="h-4 w-4" /> Download PDF
        </a>
      </div>

      <div className="flex items-center gap-2">
        <FileText className="h-5 w-5 text-brand-600 dark:text-brand-400" />
        <h1 className="text-2xl font-bold">Study #{report.study_id} report</h1>
      </div>

      <div className="note-warn rounded-lg p-3 text-sm">
        <strong>AI Draft — Requires Clinician Review.</strong> {report.disclaimer.replace(
          "AI Draft — Requires Clinician Review. ",
          "",
        )}
      </div>

      <Section title="Patient">
        {report.patient ? (
          <KeyValue rows={[
            ["Name", report.patient.name],
            ["Sex", report.patient.sex ?? "—"],
            ["Birth year", report.patient.birth_year ? String(report.patient.birth_year) : "—"],
          ]} />
        ) : (
          <p className="text-sm text-ink-4">Not attached to a patient record.</p>
        )}
      </Section>

      <Section title="Study">
        <KeyValue rows={[
          ["Modality", report.modality_label ?? MODALITY_LABELS[report.modality] ?? report.modality],
          ["Uploaded", new Date(report.uploaded_at).toLocaleString()],
          ["Model version", report.model_version ?? "—"],
        ]} />
      </Section>

      <Section title="Quality assessment">
        <div className="flex items-center gap-2 text-sm">
          {report.quality.passed ? (
            <ShieldCheck className="h-4 w-4 text-ok" />
          ) : (
            <ShieldAlert className="h-4 w-4 text-warn" />
          )}
          <span className={cn(report.quality.passed ? "text-ok" : "text-warn")}>
            {report.quality.passed ? "Passed" : "Flagged"}
          </span>
          {report.quality.score != null && (
            <span className="text-ink-4">· {report.quality.score}/100</span>
          )}
        </div>
        {report.quality.reasons.length > 0 && (
          <p className="mt-2 text-sm text-ink-4">{report.quality.reasons.join("; ")}</p>
        )}
      </Section>

      <Section title="AI findings">
        {report.analysis_stopped ? (
          <p className="text-sm text-bad">
            Analysis was stopped before inference (image failed the quality gate). No
            prediction was produced.
          </p>
        ) : report.prediction ? (
          <>
            <div className="flex items-center justify-between">
              <span className="text-lg font-semibold capitalize">{report.prediction.label}</span>
              <span className="text-sm tabular-nums text-ink-3">
                {pct(report.prediction.confidence)}
              </span>
            </div>
            <div className="mt-3 space-y-1.5">
              {Object.entries(report.prediction.probabilities)
                .sort((a, b) => b[1] - a[1])
                .map(([label, p]) => (
                  <div key={label} className="flex items-center gap-2 text-sm">
                    <span className="w-32 shrink-0 truncate capitalize text-ink-3">{label}</span>
                    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-2">
                      <div
                        className="h-full rounded-full bg-brand-500"
                        style={{ width: `${Math.max(p * 100, 2)}%` }}
                      />
                    </div>
                    <span className="w-12 shrink-0 text-right tabular-nums text-ink-4">{pct(p)}</span>
                  </div>
                ))}
            </div>
            {report.prediction.per_model && report.prediction.per_model.length > 0 && (
              <div className="mt-4 border-t border-line pt-3">
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-4">
                  Model agreement (ensemble members)
                </p>
                <div className="space-y-1.5">
                  {report.prediction.per_model.map((m) => (
                    <div key={m.backbone} className="flex items-center justify-between text-sm">
                      <span className="text-ink-3">{m.backbone}</span>
                      <span className="capitalize">
                        {m.label} <span className="text-ink-4">({pct(m.confidence)})</span>
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : (
          <p className="text-sm text-ink-4">No prediction available for this study.</p>
        )}
      </Section>

      <Section title={`Imaging · ${report.num_rois} ROI${report.num_rois === 1 ? "" : "s"} detected`}>
        <div className="grid grid-cols-3 gap-3">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <ImageTile src={study.image_url} label="Original" />
          {study.annotated_url && <ImageTile src={study.annotated_url} label="ROI overlay" />}
          {study.prediction?.heatmap_url && (
            <ImageTile src={study.prediction.heatmap_url} label="Grad-CAM" />
          )}
        </div>
      </Section>

      <div className="note-warn rounded-lg p-3 text-xs">{report.disclaimer}</div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="card p-4">
      <h2 className="mb-3 text-sm font-semibold text-ink-2">{title}</h2>
      {children}
    </div>
  );
}

function KeyValue({ rows }: { rows: [string, string][] }) {
  return (
    <dl className="space-y-1.5 text-sm">
      {rows.map(([k, v]) => (
        <div key={k} className="flex justify-between gap-4">
          <dt className="text-ink-4">{k}</dt>
          <dd className="text-right font-medium text-ink-2">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

function ImageTile({ src, label }: { src: string; label: string }) {
  return (
    <div className="space-y-1">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt={label} className="aspect-square w-full rounded-lg border border-line object-cover" />
      <p className="text-center text-xs text-ink-4">{label}</p>
    </div>
  );
}
