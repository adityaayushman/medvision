"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";

type Stage = { name: string; url: string };

export function RoiViewer({ stages }: { stages: Stage[] }) {
  const [active, setActive] = useState(0);
  if (stages.length === 0) return null;
  const current = stages[Math.min(active, stages.length - 1)];

  return (
    <div className="card overflow-hidden">
      <div className="flex flex-wrap gap-1 border-b border-line p-2">
        {stages.map((s, i) => (
          <button
            key={s.name}
            type="button"
            onClick={() => setActive(i)}
            className={cn(
              "rounded-lg px-3 py-1.5 text-xs font-medium transition",
              i === active
                ? "bg-brand-500/20 text-brand-700 dark:text-brand-200"
                : "text-ink-3 hover:bg-surface-2 hover:text-ink",
            )}
          >
            {s.name}
          </button>
        ))}
      </div>
      <div className="grid place-items-center bg-surface p-3">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          key={current.url}
          src={current.url}
          alt={current.name}
          className="max-h-[420px] w-full rounded-lg object-contain"
        />
      </div>
    </div>
  );
}
