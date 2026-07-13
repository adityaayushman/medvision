"use client";

import { useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, Database, ExternalLink, Loader2, Star, XCircle } from "lucide-react";
import { getHealth, listDatasets } from "@/lib/api";
import type { DatasetSpec, Health } from "@/lib/types";
import { cn } from "@/lib/utils";

const ACCESS_LABEL: Record<DatasetSpec["access"], string> = {
  open: "Open access",
  kaggle: "Kaggle",
  credentialed: "Credentialed",
};

const ACCESS_STYLE: Record<DatasetSpec["access"], string> = {
  open: "bg-emerald-50 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300",
  kaggle: "bg-sky-50 text-sky-700 dark:bg-sky-900/30 dark:text-sky-300",
  credentialed: "bg-amber-50 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300",
};

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<DatasetSpec[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listDatasets()
      .then(setDatasets)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load datasets"))
      .finally(() => setLoading(false));
    getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <Database className="h-5 w-5 text-brand-600" />
        <h1 className="text-2xl font-bold">Training data</h1>
      </div>
      <p className="max-w-2xl text-sm text-slate-600 dark:text-slate-400">
        This is a <strong>catalog of candidate datasets</strong> the platform's
        training pipeline can target — not a record of what has already been
        trained. Version 1 targets chest X-ray, pairing a classification
        dataset with a lung-segmentation dataset for anatomical ROI extraction.
      </p>

      {health && (
        <div
          className={cn(
            "flex items-start gap-3 rounded-xl p-3 text-sm",
            health.model_loaded
              ? "bg-emerald-50 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300"
              : "bg-amber-50 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300",
          )}
        >
          {health.model_loaded ? (
            <CheckCircle2 className="h-5 w-5 shrink-0" />
          ) : (
            <AlertTriangle className="h-5 w-5 shrink-0" />
          )}
          <p>
            {health.model_loaded
              ? "A trained model is currently live — but that doesn't mean every dataset below has been used. Check the individual training run for details."
              : "No trained model is currently live. None of the datasets below have completed training yet — predictions on /analyze are disabled until one does."}
          </p>
        </div>
      )}

      {loading ? (
        <div className="grid place-items-center p-10 text-slate-500">
          <Loader2 className="h-5 w-5 animate-spin" />
        </div>
      ) : error ? (
        <p className="text-sm text-red-600">{error}</p>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {datasets.map((d) => (
            <div key={d.key} className="card p-5">
              <div className="mb-2 flex items-start justify-between gap-2">
                <h3 className="font-semibold">{d.name}</h3>
                <span className={cn("shrink-0 rounded-full px-2 py-0.5 text-xs font-medium", ACCESS_STYLE[d.access])}>
                  {ACCESS_LABEL[d.access]}
                </span>
              </div>
              <div className="mb-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500">
                <span>{d.modality}</span>
                <span className="capitalize">{d.task}</span>
                <span>{d.approx_images} images</span>
                <span className="flex items-center gap-1">
                  {d.roi_support ? (
                    <>
                      <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" /> ROI support
                    </>
                  ) : (
                    <>
                      <XCircle className="h-3.5 w-3.5 text-slate-400" /> No ROI data
                    </>
                  )}
                </span>
              </div>
              {d.notes && <p className="mb-3 text-sm text-slate-600 dark:text-slate-400">{d.notes}</p>}
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex flex-wrap gap-1">
                  {d.recommended_for.map((tag) => (
                    <span key={tag} className="inline-flex items-center gap-1 rounded-md bg-slate-100 px-2 py-0.5 text-[11px] text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                      <Star className="h-2.5 w-2.5" /> {tag}
                    </span>
                  ))}
                </div>
                <a href={d.url} target="_blank" rel="noreferrer"
                  className="inline-flex items-center gap-1 text-xs font-medium text-brand-600 hover:underline">
                  Source <ExternalLink className="h-3 w-3" />
                </a>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
