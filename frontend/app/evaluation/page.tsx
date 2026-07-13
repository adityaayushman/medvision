import { BarChart3, CheckCircle2, ShieldCheck, FlaskConical, Cpu } from "lucide-react";
import {
  CONFUSION_MATRIX,
  CM_LABELS,
  HEADLINE,
  HISTORY,
  MODEL_INFO,
  PER_CLASS,
  PROCEDURES,
  SERIES,
} from "@/lib/evaluation-data";
import { ConfusionMatrix, LineChart, MetricBar, StatTile } from "@/components/eval-charts";

// short labels for the confusion-matrix axes
const SHORT: Record<string, string> = {
  "Lung Opacity": "Opacity",
  "No Lung Opacity / Not Normal": "No opacity",
  Normal: "Normal",
};

export const metadata = {
  title: "Model Evaluation — MedChron AI",
};

export default function EvaluationPage() {
  const xLabels = HISTORY.map((h) => String(h.step));

  return (
    <div className="space-y-10">
      {/* header */}
      <header>
        <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-ink-4">
          <BarChart3 className="h-3.5 w-3.5" /> Evaluation
        </div>
        <h1 className="text-2xl font-bold sm:text-3xl">Model evaluation &amp; test procedure</h1>
        <p className="mt-2 max-w-2xl text-sm text-ink-3">
          The deployed model, measured on a held-out test set with a documented,
          reproducible procedure. Every number below is the actual output of{" "}
          <code className="rounded bg-surface-2 px-1 py-0.5 text-[12px] text-ink-2">
            ml/scripts/evaluate.py
          </code>{" "}
          (metrics via scikit-learn) — not illustrative placeholders.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <Badge icon={ShieldCheck}>Leak-safe patient split</Badge>
          <Badge icon={FlaskConical}>Held-out test set</Badge>
          <Badge icon={CheckCircle2}>22 automated tests passing</Badge>
        </div>
      </header>

      {/* model card */}
      <section className="card p-6">
        <div className="mb-4 flex items-center gap-2">
          <Cpu className="h-4 w-4 text-brand-600 dark:text-brand-400" />
          <h2 className="font-semibold">Model under test</h2>
        </div>
        <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-3">
          <Field k="Backbone" v={MODEL_INFO.name} />
          <Field k="Task" v={MODEL_INFO.task} />
          <Field k="Dataset" v={MODEL_INFO.dataset} />
          <Field k="Training data" v={MODEL_INFO.trainedOn} />
          <Field
            k="Split (train / val / test)"
            v={`${MODEL_INFO.split.train} / ${MODEL_INFO.split.val} / ${MODEL_INFO.split.test}`}
          />
          <Field k="Schedule" v={MODEL_INFO.schedule} />
        </dl>
      </section>

      {/* headline metrics */}
      <section className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatTile label="ROC-AUC" value={HEADLINE.roc_auc.toFixed(3)} sub="macro, one-vs-rest" emphasis />
        <StatTile label="Accuracy" value={`${(HEADLINE.accuracy * 100).toFixed(1)}%`} sub="vs. 33% random (3 classes)" />
        <StatTile label="Macro F1" value={HEADLINE.macro_f1.toFixed(3)} sub="unweighted class mean" />
        <StatTile label="Test images" value={String(HEADLINE.test_images)} sub="held-out, unseen" />
      </section>

      {/* confusion matrix + per-class table */}
      <section className="grid gap-4 lg:grid-cols-2">
        <div className="card p-6">
          <h3 className="mb-4 font-semibold">Confusion matrix</h3>
          <ConfusionMatrix matrix={CONFUSION_MATRIX} labels={CM_LABELS.map((l) => SHORT[l] ?? l)} />
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
                {PER_CLASS.map((c) => (
                  <tr key={c.label}>
                    <td className="py-2.5 pr-3 font-medium text-ink">{c.label}</td>
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
      <section>
        <div className="mb-4 text-xs font-semibold uppercase tracking-[0.18em] text-ink-4">
          Training history (8 epochs)
        </div>
        <div className="grid gap-4 lg:grid-cols-2">
          <LineChart
            title="Loss per epoch"
            xLabels={xLabels}
            phaseBoundaryAfter={3}
            series={[
              { name: "train", color: SERIES.train, values: HISTORY.map((h) => h.train_loss) },
              { name: "validation", color: SERIES.val, values: HISTORY.map((h) => h.val_loss) },
            ]}
            yFormat={(n) => n.toFixed(2)}
          />
          <LineChart
            title="Accuracy per epoch"
            xLabels={xLabels}
            phaseBoundaryAfter={3}
            series={[
              { name: "train", color: SERIES.train, values: HISTORY.map((h) => h.train_acc) },
              { name: "validation", color: SERIES.val, values: HISTORY.map((h) => h.val_acc) },
            ]}
            yFormat={(n) => `${(n * 100).toFixed(0)}%`}
          />
        </div>
      </section>

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
        <strong>Honest scope.</strong> This is a deliberately modest baseline — a
        5,000-image subset, 8 epochs, trained on CPU. An AUC of 0.83 is a real,
        reproducible result, not state of the art. It is research/educational
        software, <strong>not a medical device and not for clinical use.</strong>
      </section>
    </div>
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
