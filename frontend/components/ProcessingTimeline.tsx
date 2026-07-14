"use client";

import { Check, X, MinusCircle } from "lucide-react";
import type { PipelineStep } from "@/lib/types";
import { cn } from "@/lib/utils";

const NODE_STYLE = {
  done: "border-ok bg-ok/15 text-ok",
  skipped: "border-line-2 bg-surface-2 text-ink-4",
  stopped: "border-bad bg-bad/15 text-bad",
} as const;

const LINE_STYLE = {
  done: "bg-ok/50",
  skipped: "bg-line-2",
  stopped: "bg-bad/50",
} as const;

function NodeIcon({ status }: { status: PipelineStep["status"] }) {
  if (status === "done") return <Check className="h-3.5 w-3.5" />;
  if (status === "stopped") return <X className="h-3.5 w-3.5" />;
  return <MinusCircle className="h-3.5 w-3.5" />;
}

export function ProcessingTimeline({ steps }: { steps: PipelineStep[] }) {
  return (
    <div className="card p-5">
      <h3 className="mb-4 text-sm font-semibold text-ink">AI Processing Timeline</h3>
      <div className="flex items-start overflow-x-auto pb-1">
        {steps.map((s, i) => (
          <div key={s.name} className="flex items-center">
            <div className="flex w-[92px] flex-col items-center text-center">
              <div
                title={s.detail || s.name}
                className={cn(
                  "grid h-8 w-8 shrink-0 place-items-center rounded-full border-2",
                  NODE_STYLE[s.status],
                )}
              >
                <NodeIcon status={s.status} />
              </div>
              <span className="mt-1.5 text-[10.5px] font-medium leading-tight text-ink-3">
                {s.name}
              </span>
            </div>
            {i < steps.length - 1 && (
              <div className={cn("mb-6 h-0.5 w-6 shrink-0", LINE_STYLE[s.status])} />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
