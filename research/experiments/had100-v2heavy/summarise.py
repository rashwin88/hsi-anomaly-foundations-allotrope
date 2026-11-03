"""mAUC table: V2 Heavy (this run) vs V1 v0.2.0 and GRX / PCA30-RX (V1's step6 per-crop JSON), with and
without the 6 Kakinada crops; paired Wilcoxon V2 vs V1 on the per-crop AUCs."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
v2 = json.loads((HERE / "results" / "had100_v2heavy_per_crop.json").read_text())
v1 = json.loads(Path(r"C:\Users\FITT\Documents\public_had_datasets\dual_window_public\had100_protocol.json").read_text())
assert set(v1) == set(v2) and len(v2) == 100
(HERE / "results" / "v1_step6_per_crop.json").write_text(json.dumps(v1, indent=1))
COLS = ("50", "100", "200", "full")
V1N = {"50": "50", "100": "100", "200": "200", "full": "(all 276 bands)"}
rows = []
for v in ("A", "B"):
    lab = {"A": "pooled affine", "B": "blackbody TOA"}[v]
    rows += [(f"V2 Heavy + RM global [{lab}]", lambda r, c, v=v: r[f"{v} RM global {c}"], v2),
             (f"V2 Heavy + RM local 5/15 [{lab}]", lambda r, c, v=v: r[f"{v} RM local 5/15 {c}"], v2),
             (f"V2 Heavy + RM product 21/61 [{lab}]", lambda r, c, v=v: r[f"{v} RM product 21/61 {c}"], v2),
             (f"V2 Heavy L1 [{lab}]", lambda r, c, v=v: r[f"{v} L1 {c}"], v2)]
rows += [("V1 v0.2.0 + RM global (165 grid, all cols)", lambda r, c: r["Indradhanu + Mahalanobis (full spectrum)"], v1),
         ("V1 v0.2.0 + RM local 5/15 (165 grid)", lambda r, c: r["Indradhanu + local-mean 5/15 (full spectrum)"], v1),
         ("V1 v0.2.0 L1 (165 grid)", lambda r, c: r["Indradhanu L1 (full spectrum)"], v1),
         ("GRX", lambda r, c: r[f"GRX {V1N[c]}"], v1),
         ("PCA30-RX", lambda r, c: r[f"PCA30-RX {V1N[c]}"], v1)]
names = sorted(v2)
nok = [n for n in names if not n.startswith("ang20180227t082814")]
out = {}
for subset, ns in (("all 100", names), (f"without Kakinada ({len(nok)})", nok)):
    print(f"\n### {subset}\n\n| method | 50 | 100 | 200 | full (276) |\n|---|---|---|---|---|")
    for lab, f, src in rows:
        vals = [float(np.mean([f(src[n], c) for n in ns])) for c in COLS]
        out.setdefault(subset, {})[lab] = dict(zip(COLS, vals))
        print(f"| {lab} | " + " | ".join(f"{x:.4f}" for x in vals) + " |")
    a = [v2[n]["A RM global full"] for n in ns]
    b = [v1[n]["Indradhanu + Mahalanobis (full spectrum)"] for n in ns]
    d = np.array(a) - np.array(b)
    print(f"\nV2 A RM global full vs V1 RM: wins/losses {int((d > 0).sum())}/{int((d < 0).sum())}, "
          f"Wilcoxon p {wilcoxon(a, b).pvalue:.2g}")
bands = {c: sorted({v2[n][f'A bands_read {c}'] for n in names}) for c in COLS}
print("\nbands V2 actually read per setting:", bands)
(HERE / "results" / "mauc_summary.json").write_text(json.dumps(out, indent=1))
