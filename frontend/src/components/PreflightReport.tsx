// The generic_hsi pre-flight report: what the reader decided about an uploaded file.
//
// Shown by IngestPanel (from the finished onboarding job's payload.preflight) and by
// SceneDetailPage (from scene.metadata.preflight). The report is written by
// app/utils/files/generic_hsi_preflight.py; this component prints it and nothing more,
// so a field the backend adds later simply does not show until it is listed here.
// Plain markup on existing classes only - see docs/lld/generic-hsi.md.

type Json = Record<string, unknown>;

function obj(v: unknown): Json {
  return v && typeof v === "object" ? (v as Json) : {};
}

function num(v: unknown, digits = 1): string {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";
}

/** [label, value] rows of the report, in the order an operator checks them. */
export function preflightRows(report: Json): [string, string][] {
  const shape = obj(report.shape);
  const wl = obj(report.wavelengths);
  const fwhm = obj(report.fwhm);
  const scale = obj(report.scale);
  const nodata = obj(report.nodata);
  const georef = obj(report.georef);
  const declared = nodata.declared;
  return [
    ["File", `${report.file ?? "—"}${report.subdataset ? ` · ${report.subdataset}` : ""}`],
    ["Format", [report.format, report.interleave, report.dtype].filter(Boolean).join(" · ")],
    ["Bands", `${shape.bands ?? "—"} kept of ${shape.bands_in_file ?? "—"} · ${shape.height}×${shape.width} px`],
    ["Wavelengths", `${num(wl.min_nm)}–${num(wl.max_nm)} nm · from ${wl.source ?? "—"}${wl.units_read ? ` (${wl.units_read})` : ""}`],
    ["FWHM", `${num(fwhm.min_nm, 2)}–${num(fwhm.max_nm, 2)} nm · ${fwhm.source ?? "—"}`],
    ["Scale", `${scale.decision ?? "—"} ×${scale.factor ?? "—"} · ${scale.source ?? "—"}: ${scale.reason ?? ""}`],
    ["No-data", `${declared === null || declared === undefined ? "none declared" : String(declared)} (${nodata.source ?? "—"}); also NaN, -9999, all-zero spectra`],
    ["Georef", georef.exportable ? `${georef.source} · ${georef.crs ?? ""}` : `none — export will refuse`],
  ];
}

export function PreflightReport({ report }: { report: Json }) {
  const warnings = Array.isArray(report.warnings) ? (report.warnings as string[]) : [];
  const assumptions = Array.isArray(report.assumptions) ? (report.assumptions as string[]) : [];
  return (
    <div className="preflight">
      {warnings.map((w) => (
        <p className="form__error" role="alert" key={w}>
          Warning: {w}
        </p>
      ))}
      <dl className="scene-detail__meta-list">
        {preflightRows(report).map(([label, value]) => (
          <div className="scene-detail__meta-row" key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      {assumptions.length > 0 && (
        <>
          <p className="small muted">Assumptions</p>
          <ul className="small muted">
            {assumptions.map((a) => (
              <li key={a}>{a}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
