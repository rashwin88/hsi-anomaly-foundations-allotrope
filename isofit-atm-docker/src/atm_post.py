"""
Per-sensor post-processing of an ISOFIT apply_oe run, applied in place.

Goal: reproduce each vendor's L2 product, not ground truth. Our anomaly models
were trained on vendor reflectance, so a step belongs here only if the vendor does
the equivalent (README "What the vendors actually do"; experiment IDs refer to
research/experiments/atm-correction/EXPERIMENT_LOG.md):

  enmap     DLR PACO Land_Mode: adjacency at R = 1 km, terrain correction
  prisma    ASI L2D: no terrain, no adjacency (ours over-corrects against it, AC26-A)
  avirisng  JPL v2 corr (ATREM): fixed 50 km visibility, no adjacency, no terrain
  tanager   Planet SR: EnMAP's rules (validated on 3 scenes, AC26-ACC), Tanager noise file
  ghost     OSK delivers QUAC SR (no adjacency, no terrain): the same here; also the rule for unknown sensors

Order: inversion-window mask, outlier clip, then adjacency and terrain where the
sensor's rules switch them on, then the NDSI snow mask (all sensors, read-only). Clipping first keeps ISOFIT's occasional wild
values (-8971, +514) out of the 1 km adjacency background.

The input-side rules (noise file, LUT config, AOT prior) and the harmonisation model live in the same table, so
one place states what each sensor gets. readers.prep records the rule name in scene.json (rule_name()).

usage: python -m atm_post --sensor enmap --isofit-dir <work>/isofit --obs <obs> [--px-m 30]
                          [--report atm_post.json] [--aot-prior <work>/aot_prior.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from spectral import envi

import adjacency
import rfl_qc
import sixs_parts
import terrain

log = logging.getLogger("atm_post")


@dataclass(frozen=True)
class SensorRules:
    # apply_oe inputs
    instrument_noise: str | None    # file under ISOFIT's data dir; None = ISOFIT's default for the sensor
    lut_config: str | None          # config under CONFIG_ROOT; None = the season preset
    # post-processing; None disables the step
    adjacency_below_elev_km: float | None  # on only if the scene-median elevation is below this
    adjacency_range_km: float
    terrain_min_steep_frac: float | None   # on only if more than this fraction of pixels is steep
    terrain_steep_deg: float = 6.0
    # apply_oe input: pin the AOT LUT axis to the POWER climatology window (aot_prior.py)
    aot_prior: bool = False
    # harmonise/ model file (final_models/native); None = identity, our ISOFIT output as is
    harmonise_model: str | None = None
    note: str = ""


SENSOR_RULES = {
    # Adjacency over-corrected the 3.2 km Himalaya scene (AC26-02); PACO itself
    # switches terrain on above 1% of pixels steeper than 6 deg (ATBD 5.8.2).
    "enmap": SensorRules(None, None, adjacency_below_elev_km=2.0, adjacency_range_km=1.0,
                         terrain_min_steep_frac=0.01, harmonise_model="enmap_blue600.npz"),
    # Free AOT collapses to 0.05-0.15 on PRISMA (AC26-A2); the climatology window
    # fixes a third of the visible bias and is what the PRISMA harmonisation was
    # trained on (AC26-A2b). ASI's SWIR artefacts stay with harmonisation.
    "prisma": SensorRules(None, None, adjacency_below_elev_km=None, adjacency_range_km=1.0,
                          terrain_min_steep_frac=None, aot_prior=True, harmonise_model="prisma_native.npz"),
    # AOT pinned to ATREM's 50 km visibility (AC26-P); airborne adjacency range 0.5 km if ever enabled.
    "avirisng": SensorRules("avirisng_noise.txt", "avirisng_lut.json", adjacency_below_elev_km=None,
                            adjacency_range_km=0.5, terrain_min_steep_frac=None),
    # 30 m spaceborne like EnMAP, and EnMAP's rules gave land MAE 0.0061 / 0.0144 / 0.0079 against Planet's basic SR
    # on three scenes (AC26-ACC). The noise file is named here so ENVI / GeoTIFF deliveries (apply_oe NA- mode) get
    # it too, not only HDF5 (apply_oe tanager mode picks it itself). No Planet-paired harmonisation fitted: identity.
    "tanager": SensorRules("tanager1_noise_20241016.txt", None, adjacency_below_elev_km=2.0, adjacency_range_km=1.0,
                           terrain_min_steep_frac=0.01, note="EnMAP-like rules, validated on 3 scenes vs Planet SR"),
    # No vendor product to match except OSK's QUAC SR (scene-wide, no adjacency, no terrain), no published noise
    # model, and deliveries seen so far without a CRS (no DEM): the conservative choice. Untested against truth.
    "ghost": SensorRules(None, None, adjacency_below_elev_km=None, adjacency_range_km=1.0,
                         terrain_min_steep_frac=None, note="generic noise, no adjacency / terrain; untested"),
    "generic": SensorRules(None, None, adjacency_below_elev_km=None, adjacency_range_km=1.0,
                           terrain_min_steep_frac=None, note="unknown sensor: GHOSt's conservative rules"),
}
# readers.sensors ids -> rule names
RULE_FOR = {"enmap": "enmap", "prisma": "prisma", "aviris_ng": "avirisng", "tanager": "tanager", "ghost": "ghost"}


def rule_name(sensor_id: str) -> str:
    return RULE_FOR.get(sensor_id, "generic")


def find_outputs(isofit_dir: Path) -> tuple[str, Path, Path]:
    """(rfl path without .hdr, *_atm_interp, lut_full dir) of a finished apply_oe run."""
    out = isofit_dir / "output"
    rfl = [p for p in out.glob("*_rfl") if "_subs_" not in p.name]
    atm = [p for p in out.glob("*_atm_interp") if "_subs_" not in p.name]
    if not rfl or not atm:
        raise FileNotFoundError(f"no *_rfl / *_atm_interp under {out}")
    return str(rfl[0]), atm[0], isofit_dir / "lut_full"


def steep_fraction(obs: str, steep_deg: float) -> float:
    slope = np.asarray(envi.open(obs + ".hdr", obs).open_memmap(interleave="bip")[..., terrain.OBS_SLOPE])
    v = slope[slope > -9990]
    return float((v > steep_deg).mean()) if v.size else 0.0


def run(sensor: str, isofit_dir: Path, obs: str, px_m: float = 30.0, aot_prior: Path | None = None) -> dict:
    """Apply the sensor's post-processing to the run's reflectance in place; return the report."""
    rules = SENSOR_RULES[sensor]
    rfl, atm, lut = find_outputs(isofit_dir)
    aot, h2o, elev = sixs_parts.scene_state(atm)
    rep = {"sensor": sensor, "rules": asdict(rules), "reflectance": rfl,
           "scene_aot550": aot, "scene_h2o": h2o, "scene_elev_km": elev,
           # the record aot_prior.py wrote before apply_oe; harmonisation reads aod550_clim from it
           "aot_prior": json.loads(aot_prior.read_text()) if aot_prior and aot_prior.exists() else None,
           "bands_outside_inversion_windows": rfl_qc.mask_windows(rfl)}
    rep["outlier_pixels_nodata"], rep["bright_pixels_flagged"], rep["outlier_mask"] = rfl_qc.clip_outliers(rfl)
    log.info("masked %d window bands; %d negative pixels set to nodata, %d above 2.5 kept and flagged (%s)",
             rep["bands_outside_inversion_windows"], rep["outlier_pixels_nodata"], rep["bright_pixels_flagged"],
             rep["outlier_mask"])

    thr = rules.adjacency_below_elev_km
    rep["adjacency_applied"] = thr is not None and elev < thr
    if rep["adjacency_applied"]:
        rep["adjacency"] = adjacency.correct(rfl, lut, atm, rules.adjacency_range_km, px_m)
    log.info("adjacency %s (scene elevation %.2f km)", "applied" if rep["adjacency_applied"] else "skipped", elev)

    rep["steep_fraction"] = steep_fraction(obs, rules.terrain_steep_deg)
    thr = rules.terrain_min_steep_frac
    rep["terrain_applied"] = thr is not None and rep["steep_fraction"] > thr
    if rep["terrain_applied"]:
        rep["terrain"] = terrain.correct(rfl, obs, lut, atm, px_m)
    log.info("terrain %s (%.1f%% steeper than %g deg)", "applied" if rep["terrain_applied"] else "skipped",
             100 * rep["steep_fraction"], rules.terrain_steep_deg)

    # last: the flag must describe the final reflectance
    rep["snow_mask"], rep["snow_fraction"] = rfl_qc.snow_mask(rfl)
    log.info("NDSI snow %.1f%% -> %s", 100 * rep["snow_fraction"], rep["snow_mask"])
    return rep


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sensor", required=True, choices=sorted(SENSOR_RULES))
    p.add_argument("--isofit-dir", required=True, type=Path, help="apply_oe working directory")
    p.add_argument("--obs", required=True, help="ISOFIT obs cube (ENVI, no .hdr) of the same scene")
    p.add_argument("--px-m", type=float, default=30.0, help="pixel size in metres")
    p.add_argument("--report", type=Path, help="JSON report path (default <isofit-dir>/atm_post.json)")
    p.add_argument("--aot-prior", type=Path, help="record written by `aot_prior --record` before apply_oe")
    a = p.parse_args(argv)
    rep = run(a.sensor, a.isofit_dir, a.obs, a.px_m, a.aot_prior)
    out = a.report or a.isofit_dir / "atm_post.json"
    out.write_text(json.dumps(rep, indent=2))
    log.info("report: %s", out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
