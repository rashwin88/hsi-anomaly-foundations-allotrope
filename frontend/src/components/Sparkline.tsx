// Tiny inline SVG sparkline. Stroke-only path through normalized values
// (0–100), with a label and current-value readout to its left/right.

interface SparklineProps {
  label: string;
  values: number[];   // history, oldest → newest
  current: number;    // latest value (0–100)
  unit?: string;      // default "%"
}

const W = 60;
const H = 16;

export function Sparkline({
  label,
  values,
  current,
  unit = "%",
}: SparklineProps) {
  if (values.length < 2) {
    return null;
  }

  const path = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * W;
      const y = H - (v / 100) * H;
      return `${i === 0 ? "M" : "L"} ${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");

  return (
    <div className="sparkline" title={`${label} ${Math.round(current)}${unit}`}>
      <span className="sparkline__label">{label}</span>
      <svg
        className="sparkline__svg"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        <path d={path} />
      </svg>
      <span className="sparkline__value">
        {Math.round(current)}
        {unit}
      </span>
    </div>
  );
}
