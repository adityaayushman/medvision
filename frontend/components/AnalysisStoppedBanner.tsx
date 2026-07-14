"use client";

import { ShieldOff } from "lucide-react";
import type { QualityReport } from "@/lib/types";

export function AnalysisStoppedBanner({ quality }: { quality: QualityReport }) {
  return (
    <div className="rounded-xl border note-bad p-4">
      <div className="flex items-start gap-3">
        <ShieldOff className="mt-0.5 h-5 w-5 shrink-0" />
        <div>
          <p className="font-semibold">Analysis Stopped</p>
          <p className="mt-1 text-sm opacity-90">
            Reason: {quality.reasons.join("; ")}
          </p>
          <p className="mt-2 text-sm font-medium">Recommendation: Retake the scan.</p>
        </div>
      </div>
    </div>
  );
}
