"use client";

import { useState } from "react";
import { BarChart3, CheckCircle2, ShieldCheck, FlaskConical, Cpu, ShieldOff, BookOpen } from "lucide-react";
import { EVALUATIONS, PROCEDURES, SERIES } from "@/lib/evaluation-data";
import type { LiteratureBenchmark } from "@/lib/evaluation-data";
import { ConfusionMatrix, LineChart, MetricBar, StatTile } from "@/components/eval-charts";
import { cn } from "@/lib/utils";

const MODALITY_ORDER = ["chest_xray", "brain_mri", "mammography"];

export function EvaluationDashboard() {
  const [modality, setModality] = useState("chest_xray");
  const evalData = EVALUATIONS[modality];
  const xLabels = evalData.history.map((h) => String(h.step));
  const short = evalData.shortLabels ?? {};

  return (
    <div className="space-y-10">
      {/* header */}
      <header>
        <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-ink-4">
          <BarChart3 className="h-3.5 w-3.5" /> Evaluation
        </div>
        <h1 className="text-2xl font-bold sm:text-3xl">Model evaluation &amp; test procedure</h1>
        <p className="mt-2 max-w-2xl text-sm text-ink-3">
          Every deployed (and attempted) model, measured on its own held-out test set with a
          documented, reproducible procedure. Every number below is the actual output of{" "}
          <code className="rounded bg-surface-2 px-1 py-0.5 text-[12px] text-ink-2">
            ml/scripts/evaluate.py
          </code>{" "}
          (metrics via scikit-learn) — not illustrative placeholders.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <Badge icon={ShieldCheck}>Leak-safe splits</Badge>
          <Badge icon={FlaskConical}>Held-out test sets</Badge>
          <Badge icon={CheckCircle2}>63 automated tests passing</Badge>
        </div>
      </header>

      {/* modality switcher */}
      <div className="flex flex-wrap gap-1 rounded-full border border-line bg-surface p-1 sm:inline-flex">
        {MODALITY_ORDER.map((key) => {
          const e = EVALUATIONS[key];
          return (
            <button
              key={key}
              type="button"
              onClick={() => setModality(key)}
              className={cn(
                "flex items-center gap-1.5 rounded-full px-4 py-1.5 text-sm font-medium transition",
                modality === key
                  ? "bg-brand-500/20 text-brand-700 dark:text-brand-200"
                  : "text-ink-3 hover:bg-surface-2 hover:text-ink",
              )}
            >
              {e.label}
              {!e.deployed && <ShieldOff className="h-3.5 w-3.5 text-bad" />}
            </button>
          );
        })}
      </div>

      {!evalData.deployed && (
        <section className="rounded-xl border note-bad p-4 text-sm">
          <div className="flex items-start gap-3">
            <ShieldOff className="mt-0.5 h-5 w-5 shrink-0" />
            <div>
              <strong>Not deployed — shown for transparency.</strong>
              <p className="mt-1">{evalData.notDeployedReason}</p>
            </div>
          </div>
        </section>
      )}

      {evalData.supplementaryNote && (
        <section className="rounded-xl border note-warn p-4 text-sm">
          <div className="flex items-start gap-3">
            <FlaskConical className="mt-0.5 h-5 w-5 shrink-0" />
            <div>
              <strong>Follow-up experiments.</strong>
              <p className="mt-1">{evalData.supplementaryNote}</p>
            </div>
          </div>
        </section>
      )}

      {/* model card */}
      <section className="card p-6">
        <div className="mb-4 flex items-center gap-2">
          <Cpu className="h-4 w-4 text-brand-600 dark:text-brand-400" />
          <h2 className="font-semibold">Model under test — {evalData.label}</h2>
        </div>
        <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-3">
          <Field k="Backbone" v={evalData.modelInfo.name} />
          <Field k="Task" v={evalData.modelInfo.task} />
          <Field k="Dataset" v={evalData.modelInfo.dataset} />
          <Field k="Training data" v={evalData.modelInfo.trainedOn} />
          <Field
            k="Split (train / val / test)"
            v={`${evalData.modelInfo.split.train} / ${evalData.modelInfo.split.val} / ${evalData.modelInfo.split.test}`}
          />
          <Field k="Schedule" v={evalData.modelInfo.schedule} />
        </dl>
      </section>

      {/* headline metrics */}
      <section className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatTile
          label="ROC-AUC"
          value={evalData.headline.roc_auc.toFixed(3)}
          sub="macro, one-vs-rest"
          emphasis={evalData.deployed}
        />
        <StatTile
          label="Accuracy"
          value={`${(evalData.headline.accuracy * 100).toFixed(1)}%`}
          sub={`vs. ${(evalData.randomBaseline * 100).toFixed(0)}% random (${evalData.cmLabels.length} classes)`}
        />
        <StatTile label="Macro F1" value={evalData.headline.macro_f1.toFixed(3)} sub="unweighted class mean" />
        <StatTile label="Test images" value={String(evalData.headline.test_images)} sub="held-out, unseen" />
      </section>

      {/* confusion matrix + per-class table */}
      <section className="grid gap-4 lg:grid-cols-2">
        <div className="card p-6">
          <h3 className="mb-4 font-semibold">Confusion matrix</h3>
          <ConfusionMatrix
            matrix={evalData.confusionMatrix}
            labels={evalData.cmLabels.map((l) => short[l] ?? l)}
          />
        </div>

        <div className="card p-6">
          <h3 className="mb-4 font-semibold">Per-class performance</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs uppercase tracking-wide text-ink-4">
                  <th className="pb-2 pr-3 font-medium">Class</th>
                  <th className="pb-2 pr-3 font-medium">Prec.</th>
                  <th className="pb-2 pr-3 font-medium">Recall</th>
                  <th className="pb-2 pr-3 font-medium">F1</th>
                  <th className="pb-2 font-medium">n</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {evalData.perClass.map((c) => (
                  <tr key={c.label}>
                    <td className="py-2.5 pr-3 font-medium text-ink">{short[c.label] ?? c.label}</td>
                    <td className="py-2.5 pr-3 tabular-nums text-ink-3">{c.precision.toFixed(2)}</td>
                    <td className="py-2.5 pr-3 tabular-nums text-ink-3">{c.recall.toFixed(2)}</td>
                    <td className="py-2.5 pr-3">
                      <MetricBar value={c.f1} />
                    </td>
                    <td className="py-2.5 tabular-nums text-ink-3">{c.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* training curves */}
      {evalData.history.length > 0 && (
        <section>
          <div className="mb-4 text-xs font-semibold uppercase tracking-[0.18em] text-ink-4">
            Training history ({evalData.history.length} epochs)
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <LineChart
              title="Loss per epoch"
              xLabels={xLabels}
              phaseBoundaryAfter={evalData.history.filter((h) => h.phase === "head").length}
              series={[
                { name: "train", color: SERIES.train, values: evalData.history.map((h) => h.train_loss) },
                { name: "validation", color: SERIES.val, values: evalData.history.map((h) => h.val_loss) },
              ]}
              yFormat={(n) => n.toFixed(2)}
            />
            <LineChart
              title="Accuracy per epoch"
              xLabels={xLabels}
              phaseBoundaryAfter={evalData.history.filter((h) => h.phase === "head").length}
              series={[
                { name: "train", color: SERIES.train, values: evalData.history.map((h) => h.train_acc) },
                { name: "validation", color: SERIES.val, values: evalData.history.map((h) => h.val_acc) },
              ]}
              yFormat={(n) => `${(n * 100).toFixed(0)}%`}
            />
          </div>
        </section>
      )}

      {/* literature comparison */}
      {evalData.literature && <LiteratureSection lit={evalData.literature} />}

      {/* procedures */}
      <section>
        <div className="mb-4 text-xs font-semibold uppercase tracking-[0.18em] text-ink-4">
          Procedures followed
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          {PROCEDURES.map((p) => (
            <div key={p.title} className="card p-5">
              <div className="flex items-start gap-3">
                <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-ok" />
                <div>
                  <h4 className="font-semibold text-ink">{p.title}</h4>
                  <p className="mt-1 text-sm text-ink-3">{p.body}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* honesty note */}
      <section className="rounded-xl border note-warn p-4 text-sm">
        <strong>Honest scope.</strong> Each model above is a deliberately modest baseline —
        small-to-modest datasets, a handful of epochs, trained on CPU. These are real,
        reproducible results, not state of the art. This is research/educational
        software, <strong>not a medical device and not for clinical use.</strong>
      </section>
    </div>
  );
}

function LiteratureSection({ lit }: { lit: LiteratureBenchmark }) {
  // Scale bars against the largest AUC (preferred) or accuracy present, so
  // "ours" vs "published" is visually comparable within the panel.
  const vals = lit.refs.flatMap((r) => [r.roc_auc, r.accuracy].filter((v): v is number => v != null));
  const max = Math.max(...vals, 1);

  return (
    <section>
      <div className="mb-4 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-ink-4">
        <BookOpen className="h-3.5 w-3.5" /> Versus published results
      </div>
      <div className="card p-6">
        <p className="mb-5 max-w-2xl text-sm text-ink-3">{lit.intro}</p>
        <div className="space-y-4">
          {lit.refs.map((r) => (
            <div key={r.system} className={cn("rounded-lg border p-3", r.ours ? "border-brand-400/40 bg-brand-500/5" : "border-line")}>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className={cn("text-sm font-semibold", r.ours ? "text-brand-700 dark:text-brand-300" : "text-ink")}>
                  {r.system}
                  {r.ours && <span className="ml-2 rounded-full bg-brand-500/20 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-brand-700 dark:text-brand-200">ours</span>}
                </span>
                <span className="font-mono text-xs tabular-nums text-ink-3">
                  {r.roc_auc != null && <>AUC {r.roc_auc.toFixed(3)}</>}
                  {r.roc_auc != null && r.accuracy != null && <span className="text-ink-4"> · </span>}
                  {r.accuracy != null && <>{(r.accuracy * 100).toFixed(1)}% acc</>}
                </span>
              </div>
              {/* bar: prefer AUC, else accuracy */}
              <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-surface-2">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${((r.roc_auc ?? r.accuracy ?? 0) / max) * 100}%`,
                    background: r.ours ? "var(--series-train)" : "var(--chart-grid-strong)",
                  }}
                />
              </div>
              <div className="mt-2 flex flex-wrap items-baseline gap-x-2 text-[11px] text-ink-4">
                {r.url ? (
                  <a href={r.url} target="_blank" rel="noopener noreferrer" className="underline decoration-dotted hover:text-ink-2">
                    {r.citation}
                  </a>
                ) : (
                  <span>{r.citation}</span>
                )}
                {r.caveat && <span className="italic">— {r.caveat}</span>}
              </div>
            </div>
          ))}
        </div>
        <div className="mt-5 rounded-lg border-l-2 border-brand-400 bg-surface-2/50 py-2 pl-3 pr-2">
          <p className="text-sm text-ink-2"><strong className="text-ink">Takeaway.</strong> {lit.takeaway}</p>
        </div>
      </div>
    </section>
  );
}

function Badge({ icon: Icon, children }: { icon: any; children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-3 py-1 text-xs font-medium text-ink-2">
      <Icon className="h-3.5 w-3.5 text-brand-600 dark:text-brand-400" />
      {children}
    </span>
  );
}

function Field({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <dt className="text-xs text-ink-4">{k}</dt>
      <dd className="mt-0.5 font-medium text-ink">{v}</dd>
    </div>
  );
}
