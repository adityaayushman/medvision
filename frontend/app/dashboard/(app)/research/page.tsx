"use client";

import { useEffect, useState } from "react";
import { ChevronDown, FlaskConical, Loader2, ShieldAlert } from "lucide-react";
import { useAuth } from "@/components/dashboard/AuthContext";
import { listExperimentRuns } from "@/lib/dashboard-api";
import type { ExperimentKind, ExperimentRunRead } from "@/lib/dashboard-types";
import { ConfusionMatrix, MetricBar, StatTile } from "@/components/eval-charts";
import { cn, pct } from "@/lib/utils";

const KIND_LABELS: Record<ExperimentKind, string> = {
  classification: "Classification",
  bbox_regression: "BBox regression",
  segmentation: "Segmentation",
  ensemble: "Ensemble",
};

const TABS: { key: ExperimentKind | "all"; label: string }[] = [
  { key: "all", label: "All" },
  { key: "classification", label: "Classification" },
  { key: "bbox_regression", label: "BBox regression" },
  { key: "segmentation", label: "Segmentation" },
  { key: "ensemble", label: "Ensemble" },
];

function headlineMetric(run: ExperimentRunRead): string {
  const m = run.metrics;
  if (typeof m.accuracy === "number") return pct(m.accuracy);
  if (typeof m.mean_iou === "number") return `IoU ${m.mean_iou.toFixed(3)}`;
  if (typeof m.mean_dice === "number") return `Dice ${m.mean_dice.toFixed(3)}`;
  return "—";
}

export default function ResearchPage() {
  const { token, user } = useAuth();
  const [tab, setTab] = useState<ExperimentKind | "all">("all");
  const [runs, setRuns] = useState<ExperimentRunRead[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<number | null>(null);

  useEffect(() => {
    if (!token) return;
    setLoading(true);
    listExperimentRuns(token, tab === "all" ? undefined : { kind: tab })
      .then(setRuns)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load experiment runs"))
      .finally(() => setLoading(false));
  }, [token, tab]);

  if (user && user.role !== "admin" && user.role !== "researcher") {
    return (
      <p className="card flex items-center gap-2 p-6 text-sm text-bad">
        <ShieldAlert className="h-4 w-4" /> Admin or researcher access required.
      </p>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <FlaskConical className="h-5 w-5 text-brand-600 dark:text-brand-400" />
        <h1 className="text-2xl font-bold">Research workspace</h1>
      </div>
      <p className="text-sm text-ink-3">
        Every published evaluation run, across every modality and architecture attempt — the honest
        record of what actually beat baseline and what didn't. Publish a new run with{" "}
        <code className="rounded bg-surface-2 px-1 py-0.5 text-xs">ml/scripts/log_experiment.py</code>.
      </p>

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
      ) : runs.length === 0 ? (
        <p className="card p-6 text-sm text-ink-4">No experiment runs published yet.</p>
      ) : (
        <div className="card divide-y divide-line">
          {runs.map((run) => {
            const isOpen = expanded === run.id;
            return (
              <div key={run.id}>
                <button
                  onClick={() => setExpanded(isOpen ? null : run.id)}
                  className="flex w-full items-center justify-between gap-3 p-4 text-left hover:bg-surface-2"
                >
                  <div className="min-w-0">
                    <p className="font-medium">{run.label}</p>
                    <p className="text-xs text-ink-4">
                      {KIND_LABELS[run.kind]} · {run.modality} · {run.backbone} ·{" "}
                      {new Date(run.created_at).toLocaleDateString()}
                      {run.created_by_email ? ` · ${run.created_by_email}` : ""}
                    </p>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="rounded-full bg-surface-2 px-2.5 py-1 text-xs font-semibold tabular-nums text-ink-2">
                      {headlineMetric(run)}
                    </span>
                    <ChevronDown className={cn("h-4 w-4 text-ink-4 transition-transform", isOpen && "rotate-180")} />
                  </div>
                </button>
                {isOpen && (
                  <div className="space-y-4 border-t border-line bg-surface/60 p-4">
                    {run.notes && <p className="text-sm text-ink-3">{run.notes}</p>}
                    <RunMetrics run={run} />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function RunMetrics({ run }: { run: ExperimentRunRead }) {
  const m = run.metrics as Record<string, unknown>;

  if (run.kind === "bbox_regression" || run.kind === "segmentation") {
    const value = (m.mean_iou ?? m.mean_dice) as number | undefined;
    return (
      <div className="grid max-w-xs gap-3 sm:grid-cols-1">
        <StatTile label={run.kind === "bbox_regression" ? "Mean IoU" : "Mean Dice"} value={value != null ? value.toFixed(3) : "—"} />
      </div>
    );
  }

  const perClass = m.per_class as Record<string, { precision?: number; recall?: number; f1?: number }> | undefined;
  const confusion = m.confusion_matrix as number[][] | undefined;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {typeof m.accuracy === "number" && <StatTile label="Accuracy" value={pct(m.accuracy)} emphasis />}
        {typeof m.macro_f1 === "number" && <StatTile label="Macro F1" value={(m.macro_f1 as number).toFixed(3)} />}
        {typeof m.roc_auc === "number" && <StatTile label="ROC-AUC" value={(m.roc_auc as number).toFixed(3)} />}
        {typeof m.macro_recall === "number" && <StatTile label="Macro recall" value={(m.macro_recall as number).toFixed(3)} />}
      </div>

      {perClass && (
        <div className="space-y-2">
          <p className="text-xs font-medium uppercase tracking-wide text-ink-4">Per-class</p>
          {Object.entries(perClass).map(([label, stats]) => (
            <div key={label} className="flex items-center gap-3 text-sm">
              <span className="w-28 shrink-0 truncate">{label}</span>
              {typeof stats.precision === "number" && <MetricBar value={stats.precision} />}
              {typeof stats.recall === "number" && <MetricBar value={stats.recall} />}
            </div>
          ))}
        </div>
      )}

      {confusion && (
        <div>
          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-ink-4">Confusion matrix</p>
          <ConfusionMatrix matrix={confusion} labels={perClass ? Object.keys(perClass) : confusion.map((_, i) => `${i}`)} />
        </div>
      )}
    </div>
  );
}
