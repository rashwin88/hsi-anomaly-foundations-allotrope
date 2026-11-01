# tcase.sh <case>: provenance summary, report comparison numbers, warnings
cd /content/t11/$1
python3 - <<'PY'
import json, glob
p = json.load(open(glob.glob("*_provenance.json")[0])); a = p.get("atm_post") or {}
keys = ("scene_aot550", "scene_h2o", "scene_elev_km", "snow_fraction", "outlier_pixels_nodata", "bright_pixels_flagged",
        "adjacency_applied", "terrain_applied", "steep_fraction")
print({k: (round(a[k], 4) if isinstance(a.get(k), float) else a.get(k)) for k in keys})
print("georef:", p["georeference"], "| bands kept", p["bands_kept"], "| harm gated:", (p.get("harmonisation") or {}).get("gated"),
      "| grade", p["scene"]["grade"], "| times", p["write_times_s"])
PY
grep -o 'MAE (all bands)</td><td>[^<]*\|Pixels compared</td><td>[^<]*\|Bias, ours - reference</td><td>[^<]*\|Pairing</td><td>[^<]*\|Not available: [^<]*' *_report.html
grep -h 'WARNING\|ERROR\|Traceback' ../$1.log | grep -v 'Token auth\|surface model' | cut -c1-160 | head -5
cat _work/stage_times.txt | awk 'NR>1{printf "%s %.1f min | ", p, ($2-t)/60} {p=$1; t=$2} END{print ""}'
