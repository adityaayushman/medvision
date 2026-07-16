import { cn } from "@/lib/utils";

const INK_MUTED = "var(--chart-ink-muted)";
const GRID = "var(--chart-grid)";

export function StatTile({
  label,
  value,
  sub,
  emphasis = false,
}: {
  label: string;
  value: string;
  sub?: string;
  emphasis?: boolean;
}) {
  return (
    <div className={cn("card p-5", emphasis && "ring-1 ring-inset ring-brand-400/30")}>
      <div className="text-xs font-medium uppercase tracking-wide text-ink-4">{label}</div>
      <div
        className={cn(
          "mt-1 text-3xl font-bold tabular-nums",
          emphasis ? "text-gradient" : "text-ink",
        )}
      >
        {value}
      </div>
      {sub && <div className="mt-1 text-xs text-ink-4">{sub}</div>}
    </div>
  );
}

export function MetricBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  return (
    <div className="flex items-center gap-2">
      <div className="h-2 w-24 overflow-hidden rounded-full bg-surface-2">
        <div
          className="h-full rounded-full"
          style={{ width: `${pct}%`, background: "var(--series-train)" }}
          title={`${(value * 100).toFixed(1)}%`}
        />
      </div>
      <span className="w-11 text-right text-xs tabular-nums text-ink-2">{value.toFixed(2)}</span>
    </div>
  );
}

export function ConfusionMatrix({
  matrix,
  labels,
}: {
  matrix: number[][];
  labels: string[];
}) {
  const max = Math.max(...matrix.flat());
  return (
    <div className="overflow-x-auto">
      <div className="inline-grid" style={{ gridTemplateColumns: `auto repeat(${labels.length}, 1fr)` }}>
        {/* header row */}
        <div />
        {labels.map((l) => (
          <div key={l} className="px-1 pb-2 text-center text-[11px] font-medium text-ink-3">
            {l}
          </div>
        ))}
        {matrix.map((row, i) => (
          <RowCells key={i} row={row} i={i} labels={labels} max={max} />
        ))}
      </div>
      <p className="mt-3 text-[11px] text-ink-4">
        Rows = actual · Columns = predicted · diagonal = correct. Darker = more cases.
      </p>
    </div>
  );
}

function RowCells({ row, i, labels, max }: { row: number[]; i: number; labels: string[]; max: number }) {
  return (
    <>
      <div className="flex items-center justify-end whitespace-nowrap pr-3 text-[11px] font-medium text-ink-3">
        {labels[i]}
      </div>
      {row.map((v, j) => {
        const t = max > 0 ? v / max : 0;
        const isDiag = i === j;
        return (
          <div
            key={j}
            title={`Actual ${labels[i]}, predicted ${labels[j]}: ${v}`}
            className={cn(
              "m-[2px] grid h-16 min-w-[64px] place-items-center rounded-lg text-sm font-semibold tabular-nums transition",
              isDiag ? "ring-1 ring-inset ring-brand-300/40" : "",
            )}
            style={{
              background: `rgba(59,130,246,${(0.06 + t * 0.82).toFixed(3)})`,
              color: t > 0.5 ? "#ffffff" : "var(--chart-cell-ink)",
            }}
          >
            {v}
          </div>
        );
      })}
    </>
  );
}

type Series = { name: string; color: string; values: number[] };

export function LineChart({
  series,
  xLabels,
  yFormat = (n) => n.toFixed(2),
  phaseBoundaryAfter,
  title,
}: {
  series: Series[];
  xLabels: string[];
  yFormat?: (n: number) => string;
  phaseBoundaryAfter?: number; 
  title: string;
}) {
  const W = 520;
  const H = 240;
  const pad = { top: 16, right: 44, bottom: 30, left: 40 };
  const n = xLabels.length;

  const all = series.flatMap((s) => s.values);
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  const span = hi - lo || 1;
  lo -= span * 0.12;
  hi += span * 0.12;

  const xOf = (i: number) => pad.left + (i / (n - 1)) * (W - pad.left - pad.right);
  const yOf = (v: number) => pad.top + (1 - (v - lo) / (hi - lo)) * (H - pad.top - pad.bottom);

  const ticks = 4;
  const gridVals = Array.from({ length: ticks + 1 }, (_, k) => lo + (k / ticks) * (hi - lo));

  return (
    <figure className="card p-5">
      <figcaption className="mb-1 text-sm font-semibold text-ink">{title}</figcaption>
      <div className="mb-3 flex flex-wrap gap-4">
        {series.map((s) => (
          <span key={s.name} className="flex items-center gap-1.5 text-xs text-ink-3">
            <span className="inline-block h-0.5 w-4 rounded-full" style={{ background: s.color }} />
            {s.name}
          </span>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={title}>
        {/* gridlines + y labels */}
        {gridVals.map((gv, k) => (
          <g key={k}>
            <line x1={pad.left} x2={W - pad.right} y1={yOf(gv)} y2={yOf(gv)} stroke={GRID} strokeWidth={1} />
            <text x={pad.left - 6} y={yOf(gv) + 3} textAnchor="end" fontSize={10} fill={INK_MUTED}>
              {yFormat(gv)}
            </text>
          </g>
        ))}
        {/* phase boundary */}
        {phaseBoundaryAfter && phaseBoundaryAfter < n && (
          <g>
            <line
              x1={xOf(phaseBoundaryAfter - 0.5)}
              x2={xOf(phaseBoundaryAfter - 0.5)}
              y1={pad.top}
              y2={H - pad.bottom}
              stroke="var(--chart-grid-strong)"
              strokeWidth={1}
              strokeDasharray="3 3"
            />
            <text x={xOf(phaseBoundaryAfter - 0.5) + 4} y={pad.top + 10} fontSize={9} fill={INK_MUTED}>
              fine-tune →
            </text>
          </g>
        )}
        {/* x labels */}
        {xLabels.map((xl, i) => (
          <text key={i} x={xOf(i)} y={H - pad.bottom + 16} textAnchor="middle" fontSize={10} fill={INK_MUTED}>
            {xl}
          </text>
        ))}
        {/* series */}
        {series.map((s) => {
          const d = s.values.map((v, i) => `${i === 0 ? "M" : "L"} ${xOf(i)} ${yOf(v)}`).join(" ");
          const last = s.values.length - 1;
          return (
            <g key={s.name}>
              <path d={d} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
              {s.values.map((v, i) => (
                <circle key={i} cx={xOf(i)} cy={yOf(v)} r={3} fill={s.color}>
                  <title>{`${s.name} · ${xLabels[i]}: ${yFormat(v)}`}</title>
                </circle>
              ))}
              {/* direct end-value label */}
              <text x={xOf(last) + 6} y={yOf(s.values[last]) + 3} fontSize={10} fill={s.color} fontWeight={600}>
                {yFormat(s.values[last])}
              </text>
            </g>
          );
        })}
      </svg>
    </figure>
  );
}
