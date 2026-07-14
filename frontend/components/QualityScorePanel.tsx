"use client";

import { CheckCircle2, AlertTriangle, XCircle } from "lucide-react";
import type { QualityReport } from "@/lib/types";
import { cn } from "@/lib/utils";

const STATUS_ICON = { ok: CheckCircle2, warn: AlertTriangle, fail: XCircle } as const;
const STATUS_CLASS = { ok: "text-ok", warn: "text-warn", fail: "text-bad" } as const;

function scoreRingColor(score: number): string {
  if (score >= 85) return "rgb(var(--c-ok))";
  if (score >= 60) return "rgb(var(--c-warn))";
  return "rgb(var(--c-bad))";
}

export function QualityScorePanel({ quality }: { quality: QualityReport }) {
  const score = quality.overall_score;
  const circumference = 2 * Math.PI * 34;
  const offset = circumference * (1 - score / 100);

  return (
    <div className="card p-5">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-ink">Image Quality Report</h3>
        <span
          className={cn(
            "rounded-full px-2 py-0.5 text-[11px] font-semibold",
            quality.passed ? "chip-ok" : "chip-bad",
          )}
        >
          {quality.passed ? "Passed" : "Failed"}
        </span>
      </div>

      <div className="flex items-center gap-5">
        <div className="relative grid h-20 w-20 shrink-0 place-items-center">
          <svg viewBox="0 0 80 80" className="h-20 w-20 -rotate-90">
            <circle cx="40" cy="40" r="34" fill="none" stroke="var(--line)" strokeWidth="7" />
            <circle
              cx="40"
              cy="40"
              r="34"
              fill="none"
              stroke={scoreRingColor(score)}
              strokeWidth="7"
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={offset}
              style={{ transition: "stroke-dashoffset 0.6s ease" }}
            />
          </svg>
          <div className="absolute flex flex-col items-center">
            <span className="text-xl font-bold tabular-nums text-ink">{score}</span>
            <span className="text-[9px] text-ink-4">/ 100</span>
          </div>
        </div>

        <ul className="flex-1 space-y-1.5">
          {quality.checks.map((c) => {
            const Icon = STATUS_ICON[c.status];
            return (
              <li key={c.name} className="flex items-start gap-2 text-xs">
                <Icon className={cn("mt-0.5 h-3.5 w-3.5 shrink-0", STATUS_CLASS[c.status])} />
                <div>
                  <span className="font-medium text-ink-2">{c.name}</span>
                  <span className="ml-1 text-ink-4">— {c.detail}</span>
                </div>
              </li>
            );
          })}
        </ul>
      </div>

      <p
        className={cn(
          "mt-3 rounded-lg p-2.5 text-xs",
          quality.passed ? "bg-surface text-ink-3" : "note-bad",
        )}
      >
        {quality.recommendation}
      </p>
    </div>
  );
}
