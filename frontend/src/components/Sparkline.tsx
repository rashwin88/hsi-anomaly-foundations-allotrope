// Tufte sparkline (top bar, Monitoring cards): a word-sized line, 1 px in --ink-2, with a
// range frame - a hairline spanning only the values the window actually took - and the last
// value written at the line's end in tabular figures. No separate value column.

interface SparklineProps {
  label: string;
  values: number[];   // history, oldest → newest
  current: number;    // latest value (0–100)
  unit?: string;      // default "%"
}

const W = 60;       // line width, px
const H = 16;       // line height, px
const TEXT_W = 30;  // room for the end label

export function Sparkline({
  label,
  values,
  current,
  unit = "%",
}: SparklineProps) {
  if (values.length < 2) {
    return null;
  }

  const y = (v: number) => H - (Math.max(0, Math.min(100, v)) / 100) * (H - 2) - 1;
  const path = values
    .map((v, i) => `${i === 0 ? "M" : "L"} ${((i / (values.length - 1)) * W).toFixed(2)},${y(v).toFixed(2)}`)
    .join(" ");
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const last = values[values.length - 1];

  return (
    <div className="sparkline" title={`${label} ${Math.round(current)}${unit} (range ${Math.round(lo)}–${Math.round(hi)}${unit})`}>
      <span className="sparkline__label">{label}</span>
      <svg width={W + 2 + TEXT_W} height={H} aria-hidden="true" style={{ overflow: "visible" }}>
        {/* range frame: from the window's minimum to its maximum, nothing more */}
        <line x1={0.5} x2={0.5} y1={y(hi)} y2={y(lo)} stroke="var(--ink-3)" strokeWidth={1} />
        <path d={path} transform="translate(2,0)" fill="none" stroke="var(--ink-2)" strokeWidth={1} />
        <text
          x={W + 5}
          y={y(last)}
          dominantBaseline="middle"
          fill="var(--ink)"
          style={{ fontVariantNumeric: "tabular-nums", fontSize: "0.8rem", fontFamily: "var(--font-ui)" }}
        >
          {Math.round(current)}
          {unit}
        </text>
      </svg>
    </div>
  );
}
