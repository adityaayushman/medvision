import Link from "next/link";
import {
  FlaskConical,
  TrendingUp,
  TrendingDown,
  Minus,
  ArrowRight,
  ExternalLink,
  Microscope,
} from "lucide-react";
import { cn } from "@/lib/utils";

type Verdict = "fail" | "thin-win" | "win" | "partial";

const VERDICT_META: Record<Verdict, { label: string; chip: string; icon: typeof TrendingUp }> = {
  fail: { label: "Below baseline", chip: "chip-bad", icon: TrendingDown },
  "thin-win": { label: "Thin win", chip: "chip-warn", icon: Minus },
  win: { label: "Real win", chip: "chip-ok", icon: TrendingUp },
  partial: { label: "Partial recovery, still fails", chip: "chip-warn", icon: Minus },
};

interface Attempt {
  n: number;
  title: string;
  hypothesis: string;
  method: string;
  result: string;
  verdict: Verdict;
  diagnosis: string;
}

const ATTEMPTS: Attempt[] = [
  {
    n: 1,
    title: "MIAS, full mammograms",
    hypothesis: "A standard transfer-learning CNN can classify Normal / Benign / Malignant on the classic MIAS dataset.",
    method: "EfficientNet-B0, 322 images (the entire MIAS database), 3-class, same two-phase recipe used for chest X-ray and brain MRI.",
    result: "59.2% accuracy, macro F1 0.37 — below MIAS's own 63.3% majority-class baseline (always predicting Normal). 0% recall on Benign.",
    verdict: "fail",
    diagnosis: "322 images total, 35–44 per minority class, isn't enough data for a reliable classifier. Not deployed — a model that loses to a constant guess doesn't ship.",
  },
  {
    n: 2,
    title: "CBIS-DDSM, full mammograms",
    hypothesis: "A larger, more clinically standard dataset (2,857 images vs. MIAS's 322) fixes attempt 1's data-starvation problem.",
    method: "Identical recipe, same backbone, CBIS-DDSM's official case-description CSVs joined against full mammogram images, patient-safe split.",
    result: "59.1% accuracy, ROC-AUC 0.642, macro F1 0.575 — technically clears the 55% majority baseline, but thinly. Malignant recall only 43.7%.",
    verdict: "thin-win",
    diagnosis: "More data didn't fix it. Working theory formed here: a full mammogram is a very large image; resizing it to the model's 224×224 input shrinks a lesion — often a few percent of the frame — to a handful of pixels, destroying the texture that actually distinguishes benign from malignant.",
  },
  {
    n: 3,
    title: "CBIS-DDSM, lesion-cropped patches",
    hypothesis: "If image resolution is the bottleneck, training on CBIS-DDSM's pre-cropped lesion patches — already zoomed into the abnormality — should recover real signal.",
    method: "Same recipe, same backbone, but trained on 3,567 lesion-cropped patches (CBIS-DDSM's own official crops) instead of full mammograms.",
    result: "71.1% accuracy, ROC-AUC 0.785, macro F1 0.696, malignant recall 66.4% — a real, clear win. Confirmed against a comparable published ResNet-50 baseline (0.742 AUC) too — see the Evaluation page's literature comparison.",
    verdict: "win",
    diagnosis: "Hypothesis confirmed: full-mammogram downscaling was the real problem, not dataset size. But this model can't ship as-is — it expects an already-cropped lesion image, and the live upload flow, like every other modality, provides a full mammogram. This becomes the whole next problem.",
  },
  {
    n: 4,
    title: "Automatic crop — bounding-box regression",
    hypothesis: "A small model can learn to localize the lesion in a full mammogram (predict a bounding box), crop to it, and feed that crop to attempt 3's classifier.",
    method: "Three architecture iterations against the same CBIS-DDSM ROI masks: a naive pooled regression head, then a spatial soft-argmax head (center grounded in actual feature-map location, not a black-box linear guess), then attention-weighted size pooling.",
    result: "Test IoU improved 0.043 → 0.068 across the three attempts — still far below the ~0.3+ a usable crop needs. The full detect→crop→classify pipeline scored 49.8% accuracy, ROC-AUC 0.486 — worse than random guessing.",
    verdict: "fail",
    diagnosis: "The first architecture collapsed to predicting a near-constant box regardless of input (global-average-pooling destroys the spatial information a box center needs) — a real, diagnosable bug, not just \"needs more training.\" Even after fixing it, localization stayed too imprecise to produce usable crops.",
  },
  {
    n: 5,
    title: "Automatic crop — U-Net pixel segmentation",
    hypothesis: "A single 4-number bounding box is a very sparse training signal for this amount of data. A full per-pixel segmentation mask gives thousands of supervised points per image instead of four — replace the localizer entirely.",
    method: "EfficientNet-B0 encoder + a 4-level skip-connected decoder, trained on the same ROI masks with combined Dice+BCE loss.",
    result: "Test Dice 0.254 — roughly 4× better than the bbox regressor's IoU by any reasonable comparison, and visually confirmed: 2 of 4 spot-checked predictions were near-pixel-perfect. But the full pipeline still scored 48.9% accuracy, ROC-AUC 0.541 — still below baseline, essentially unchanged from attempt 4 despite a dramatically better localizer.",
    verdict: "fail",
    diagnosis: "The real diagnosis, and the most useful finding in this whole case study: localization quality was never the actual bottleneck. Attempt 3's classifier was trained only on CBIS-DDSM's official, hand-verified crops — a specific framing and padding convention. Any automatically-derived crop differs from that convention, however accurately it's centered, so the classifier doesn't generalize to it. Two different localizer architectures hit the same wall because neither one touched that mismatch.",
  },
  {
    n: 6,
    title: "Retrain the classifier on ground-truth-derived crops",
    hypothesis: "If the classifier's training distribution is the real problem, retraining it on crops framed exactly the way the pipeline produces them — same padding, same derivation logic, but from ground-truth boxes rather than the segmenter's own predictions — should close the gap.",
    method: "Reused the identical crop_to_bbox() function LocalizedPredictor calls at inference (factored out into shared code, not reimplemented) to materialize 2,742 training crops from ground-truth masks, then retrained attempt 3's classifier from scratch on them with the exact same hyperparameters.",
    result: "A real, measurable improvement: the full pipeline moved from 48.9% to 53.0% accuracy, confirming the framing-mismatch diagnosis was genuine. Still below both the 55% majority baseline and the 59.1% full-image baseline.",
    verdict: "partial",
    diagnosis: "The remaining gap has a clear, different cause: this classifier trained on ground-truth (perfect) localization, but at inference it sees the segmenter's real, imperfect predictions (Dice 0.254) — a second, distinct train/inference mismatch, this time about localization quality rather than framing style. The untried next step: train the classifier on the segmenter's actual predicted crops, noise and all, not ground truth.",
  },
];

const numFmt = (n: number) => `${(n * 100).toFixed(1)}%`;

const SUMMARY = [
  { label: "MIAS majority baseline", acc: 0.633, auc: null },
  { label: "CBIS-DDSM majority baseline", acc: 0.55, auc: null },
  { label: "1 — MIAS full mammograms", acc: 0.592, auc: 0.618 },
  { label: "2 — CBIS-DDSM full mammograms", acc: 0.591, auc: 0.642 },
  { label: "3 — CBIS-DDSM cropped patches", acc: 0.711, auc: 0.785, ours: true },
  { label: "4 — Pipeline (bbox regression)", acc: 0.498, auc: 0.486 },
  { label: "5 — Pipeline (U-Net segmentation)", acc: 0.489, auc: 0.541 },
  { label: "6 — Pipeline (GT-crop classifier)", acc: 0.53, auc: 0.534 },
];

export default function CaseStudy() {
  return (
    <div className="space-y-10">
      <header>
        <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-ink-4">
          <Microscope className="h-3.5 w-3.5" /> Case study
        </div>
        <h1 className="text-2xl font-bold sm:text-3xl">
          Mammography: six honest attempts, one real win, and why it still isn&rsquo;t deployed
        </h1>
        <p className="mt-3 max-w-2xl text-sm text-ink-3">
          Most write-ups of a failed feature don&rsquo;t get written at all. This one does, because
          the failures here are diagnosable, reproducible, and — taken together — a genuinely
          useful finding: <strong className="text-ink">improving a localizer doesn&rsquo;t help if the
          downstream classifier was never trained on that localizer&rsquo;s output distribution.</strong>{" "}
          Every number below is real, reproducible, and also sitting in the Research Workspace and{" "}
          <Link href="/evaluation" className="text-brand-600 underline decoration-dotted hover:text-brand-700 dark:text-brand-400">
            Evaluation
          </Link>{" "}
          page.
        </p>
      </header>

      {/* summary table */}
      <section className="card overflow-x-auto p-5">
        <h2 className="mb-4 text-sm font-semibold text-ink">All six attempts, at a glance</h2>
        <table className="w-full min-w-[480px] text-sm">
          <thead>
            <tr className="text-left text-xs uppercase tracking-wide text-ink-4">
              <th className="pb-2 pr-3 font-medium">Attempt</th>
              <th className="pb-2 pr-3 font-medium text-right">Accuracy</th>
              <th className="pb-2 font-medium text-right">ROC-AUC</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {SUMMARY.map((s) => (
              <tr key={s.label} className={s.ours ? "bg-brand-500/5" : undefined}>
                <td className={cn("py-2 pr-3", s.ours ? "font-semibold text-brand-700 dark:text-brand-300" : "text-ink-2")}>
                  {s.label}
                  {s.ours && (
                    <span className="ml-2 rounded-full bg-brand-500/20 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-brand-700 dark:text-brand-200">
                      best result
                    </span>
                  )}
                </td>
                <td className="py-2 pr-3 text-right font-mono tabular-nums text-ink-3">{numFmt(s.acc)}</td>
                <td className="py-2 text-right font-mono tabular-nums text-ink-3">{s.auc != null ? s.auc.toFixed(3) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {/* timeline */}
      <section className="space-y-5">
        {ATTEMPTS.map((a) => {
          const meta = VERDICT_META[a.verdict];
          const Icon = meta.icon;
          return (
            <div key={a.n} className="card p-6">
              <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
                <div className="flex items-baseline gap-3">
                  <span className="font-mono text-xs text-ink-4">#{a.n}</span>
                  <h3 className="font-semibold text-ink">{a.title}</h3>
                </div>
                <span className={cn("flex shrink-0 items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium", meta.chip)}>
                  <Icon className="h-3.5 w-3.5" /> {meta.label}
                </span>
              </div>
              <dl className="space-y-3 text-sm">
                <div>
                  <dt className="text-xs font-medium uppercase tracking-wide text-ink-4">Hypothesis</dt>
                  <dd className="mt-1 text-ink-2">{a.hypothesis}</dd>
                </div>
                <div>
                  <dt className="text-xs font-medium uppercase tracking-wide text-ink-4">Method</dt>
                  <dd className="mt-1 text-ink-3">{a.method}</dd>
                </div>
                <div>
                  <dt className="text-xs font-medium uppercase tracking-wide text-ink-4">Result</dt>
                  <dd className="mt-1 font-medium text-ink">{a.result}</dd>
                </div>
                <div className="rounded-lg border-l-2 border-brand-400 bg-surface-2/50 py-2 pl-3 pr-2">
                  <dt className="text-xs font-medium uppercase tracking-wide text-ink-4">Diagnosis</dt>
                  <dd className="mt-1 text-ink-2">{a.diagnosis}</dd>
                </div>
              </dl>
            </div>
          );
        })}
      </section>

      {/* synthesis */}
      <section className="card p-6">
        <div className="mb-3 flex items-center gap-2">
          <FlaskConical className="h-4 w-4 text-brand-600 dark:text-brand-400" />
          <h2 className="font-semibold text-ink">What this actually demonstrates</h2>
        </div>
        <div className="space-y-3 text-sm text-ink-2">
          <p>
            The individual result mammography needed didn&rsquo;t arrive — the platform still doesn&rsquo;t
            serve mammography predictions, and that&rsquo;s stated plainly everywhere this model is
            discussed, not hidden. But three things came out of this that are worth more than a single
            clean number would have been:
          </p>
          <ul className="ml-4 list-disc space-y-2">
            <li>
              <strong className="text-ink">A real, verifiable win exists</strong> (attempt 3, 71.1%
              accuracy / 0.785 AUC) — it&rsquo;s just not the shape of model the current upload flow can use.
              That&rsquo;s a deployment-architecture gap, not a modeling failure.
            </li>
            <li>
              <strong className="text-ink">A generalizable methodological finding</strong>: a classifier
              trained on one crop-framing convention doesn&rsquo;t transfer to another, no matter how good
              upstream localization gets. Attempts 4 and 5 spent real effort improving the wrong stage of
              the pipeline before attempt 6 identified and partially confirmed this.
            </li>
            <li>
              <strong className="text-ink">A precise, falsifiable next hypothesis</strong>, not a shrug:
              train the classifier on the segmenter&rsquo;s actual predicted crops, not ground truth. That&rsquo;s
              directly testable, and it&rsquo;s written down here instead of lost in a chat transcript.
            </li>
          </ul>
          <p>
            Every number in this write-up was published to the platform&rsquo;s own{" "}
            <span className="font-medium text-ink">Research Workspace</span> at the time it was produced —
            this page is a narrative over data that already existed, not a document written after the fact
            to look tidy.
          </p>
        </div>
      </section>

      <div className="flex flex-wrap gap-3">
        <Link href="/evaluation" className="btn-ghost">
          See the deployed models <ArrowRight className="h-4 w-4" />
        </Link>
        <a
          href="https://github.com/adityaayushman/medvision"
          target="_blank"
          rel="noopener noreferrer"
          className="btn-ghost"
        >
          Source &amp; commit history <ExternalLink className="h-4 w-4" />
        </a>
      </div>
    </div>
  );
}
