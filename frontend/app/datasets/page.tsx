"use client";

import { useEffect, useState } from "react";
import { Database, Loader2, ExternalLink, CheckCircle2, XCircle, Star } from "lucide-react";
import { listDatasets } from "@/lib/api";
import type { DatasetSpec } from "@/lib/types";
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
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listDatasets()
      .then(setDatasets)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load datasets"))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <Database className="h-5 w-5 text-brand-600" />
        <h1 className="text-2xl font-bold">Training data</h1>
      </div>
      <p className="max-w-2xl text-sm text-slate-600 dark:text-slate-400">
        MedChron AI is built and evaluated against a shortlisted set of public
        medical-imaging datasets. Version 1 targets chest X-ray, pairing a
        classification dataset with a lung-segmentation dataset for anatomical
        ROI extraction — the same registry the training pipeline reads from.
      </p>

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
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                  ) : (
                    <XCircle className="h-3.5 w-3.5 text-slate-400" />
                  )}
                  ROI support
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
