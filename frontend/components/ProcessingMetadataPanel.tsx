"use client";

import { Cpu, ListChecks, ScanSearch, Timer } from "lucide-react";
import type { ProcessingMetadata } from "@/lib/types";
import { cn } from "@/lib/utils";

export function ProcessingMetadataPanel({ meta }: { meta: ProcessingMetadata }) {
  return (
    <div className="card p-5">
      <h3 className="mb-3 text-sm font-semibold text-ink">Processing Metadata</h3>

      <div className="mb-3">
        <p className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-ink-4">
          <ListChecks className="h-3.5 w-3.5" /> Preprocessing operations applied
        </p>
        <ul className="space-y-0.5 pl-5 text-xs text-ink-3">
          {meta.preprocessing_ops.map((op) => (
            <li key={op} className="list-disc">{op}</li>
          ))}
        </ul>
      </div>

      <div className="grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
        <Stat
          icon={ScanSearch}
          label="Segmentation"
          value={meta.segmentation_success ? "Success" : "Uncertain"}
          tone={meta.segmentation_success ? "ok" : "warn"}
        />
        <Stat
          icon={ScanSearch}
          label="ROI confidence"
          value={meta.roi_confidence?.overall_label ?? "N/A"}
          tone={
            meta.roi_confidence?.overall_label === "High"
              ? "ok"
              : meta.roi_confidence?.overall_label === "Medium"
                ? "warn"
                : "muted"
          }
        />
        <Stat icon={Cpu} label="Model version" value={meta.model_version ?? "N/A"} tone="muted" small />
        <Stat
          icon={Timer}
          label="Inference time"
          value={meta.inference_time_ms != null ? `${meta.inference_time_ms.toFixed(0)} ms` : "N/A"}
          tone="muted"
        />
      </div>

      {meta.roi_confidence && (
        <p className="mt-3 text-[11px] text-ink-5">{meta.roi_confidence.note}</p>
      )}
      <p className="mt-1 text-[11px] text-ink-5">
        Total processing time: {meta.processing_time_ms.toFixed(0)} ms
      </p>
    </div>
  );
}

function Stat({
  icon: Icon,
  label,
  value,
  tone,
  small,
}: {
  icon: any;
  label: string;
  value: string;
  tone: "ok" | "warn" | "muted";
  small?: boolean;
}) {
  return (
    <div className="rounded-lg bg-surface p-2.5">
      <p className="mb-1 flex items-center gap-1 text-[10px] uppercase tracking-wide text-ink-4">
        <Icon className="h-3 w-3" /> {label}
      </p>
      <p
        className={cn(
          "font-semibold",
          small ? "truncate text-[11px]" : "text-sm",
          tone === "ok" ? "text-ok" : tone === "warn" ? "text-warn" : "text-ink-2",
        )}
        title={value}
      >
        {value}
      </p>
    </div>
  );
}
