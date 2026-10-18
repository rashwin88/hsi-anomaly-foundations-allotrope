"""
Build the implanted-target anomaly benchmark (docs/lld/implant-bench.md): real held-out EnMAP / PRISMA /
AVIRIS-NG reflectance scenes with synthetic objects whose size, background and contrast follow the 909 labelled
objects of the PRISMA bench, in three difficulty tiers (app/evaluation/implant.py).

    python scripts/build_implant_bench.py real  <stage1_datasets root> <work>/real.json
    python scripts/build_implant_bench.py build <test shards dir> <work>/real.json <out dir> [--seed 0]

`real` measures the bench objects: size, contrast z, background NDVI and the difference signature o - r of each.
`build` reassembles the chosen test scenes from their 128 px tiles (meta.json x0 / y0), places objects on valid,
cloud- and water-free land at the real density and NDVI mix, and implants half as transplanted real signatures and
half as USGS splib07 materials, each at a contrast drawn from its tier's real values. Writes per host: cube.npy
(int16, x 0.0001 = reflectance, -32768 = nodata), wavelengths.npy, gt.npy (int32 object ids, 0 = none),
objects.json; and manifest.json with every file's sha256.

Every random draw comes from one seeded generator, so a build is reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app.evaluation.implant import (  # noqa: E402
    TIERS, background_subspace, contrast_z, foreign_fraction, grow_blob, implant_material, implant_signature,
    interp_spectrum, match_foreignness, psf_weights, ring_mask, solve_realised, solve_strength, tier_of)

NODATA, SCALE = -32768, 1e-4
HOST_ROWS = 1152                                   # AVIRIS-NG lines are cut to this many rows
SPACING_PX = 10                                    # minimum gap between implanted objects
MAX_SIZE_PX = 20
SIGNATURE_GAP_NM = 15.0                            # host bands farther than this from a bench band get no signature
HOSTS = {   # held-out test scenes, one datatake / date / line each (chosen 2026-10-17 from the test-split scan)
    "enmap": ["ENMAP01-____L2A-DT0000172009_20251228T054749Z_012", "ENMAP01-____L2A-DT0000180775_20260224T055028Z_002",
              "ENMAP01-____L2A-DT0000183687_20260319T054949Z_006", "ENMAP01-____L2A-DT0000172638_20260106T061858Z_019",
              "ENMAP01-____L2A-DT0000013391_20230408T055252Z_013", "ENMAP01-____L2A-DT0000114938_20250208T060938Z_001"],
    "prisma": ["PRS_L2D_STD_20250807052906", "PRS_L2D_STD_20240508060446", "PRS_L2D_STD_20260224045640",
               "PRS_L2D_STD_20241208055621", "PRS_L2D_STD_20230311055340", "PRS_L2D_STD_20251215060925"],
    "aviris_ng": ["ang20160304t071326", "ang20180304t091703", "ang20151226t095013", "ang20151226t084531",
                  "ang20151228t052147", "ang20180415t054748"],
}
LIB_CHAPTERS = ("artificial", "minerals", "organics", "soils", "vegetation")


def band_means(refl: np.ndarray, wl: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sel = (wl > lo) & (wl < hi)
    return np.nanmean(refl[sel], axis=0)


# --- real: measure the PRISMA bench objects ---------------------------------------------------------------------

def measure_real(root: str, out: str) -> None:
    from app.evaluation.datasets import PRISMA_SCENES, load_prisma_scene
    objects, valid_px, scene_wl = [], 0, {}
    for name in PRISMA_SCENES:
        cube, _, wl, spatial, gt = load_prisma_scene(root, name, "native")
        cube, keep = cube.astype(np.float64), spatial.astype(bool)
        bv = np.ones(cube.shape, bool)
        ndvi = ndvi_map(cube, wl)
        P = background_subspace(cube, keep)
        labels, n = ndimage.label(gt)
        valid_px += int(keep.sum())
        scene_wl[name] = [float(w) for w in wl]          # set1 has 185 bands, the others 186
        for i in range(1, n + 1):
            obj = labels == i
            s_gt = gt.astype(bool)
            ring = ring_mask(obj, keep, exclude=s_gt)
            if ring.sum() < 8:
                continue
            o, r = cube[:, obj].mean(1), cube[:, ring].mean(1)
            z = contrast_z(cube, bv, obj, ring)
            halo = ndimage.binary_dilation(obj) & ~obj & keep & ~s_gt
            zh = contrast_z(cube, bv, halo, ring) if halo.sum() >= 2 else float("nan")
            objects.append(dict(scene=name, px=int(obj.sum()), z=z, ndvi=float(np.nanmean(ndvi[ring])),
                                foreign=foreign_fraction(o - r, P), halo=zh / max(z, 1e-6), d=(o - r).round(6).tolist()))
        print(name, n, "objects", flush=True)
    json.dump({"wavelengths_nm": scene_wl, "valid_px": valid_px, "objects": objects}, open(out, "w"))
    print(f"{len(objects)} objects; density {len(objects) / valid_px * 1e6:.0f} per Mpx")


def ndvi_map(refl: np.ndarray, wl: np.ndarray) -> np.ndarray:
    vis, nir = band_means(refl, wl, 450, 650), band_means(refl, wl, 800, 900)
    return (nir - vis) / (nir + vis + 1e-6)


# --- build ------------------------------------------------------------------------------------------------------

def reassemble(shard_dir: Path, sensor: str, prefix: str):
    """(cube int16 (N, H, W), wavelengths (N,), labels dict of (H, W) uint8 or {}) for the scene whose id starts
    with prefix, from every test tile of it. AVIRIS-NG lines are cut to the HOST_ROWS window with most tiles."""
    from app.utils.shard_codec import decode_i16dz
    tiles = []
    for f in sorted(shard_dir.glob(f"{sensor}/*.tar")):
        with tarfile.open(f) as t:
            members = {m.name: m for m in t.getmembers()}
            for name, m in members.items():
                if not name.endswith(".meta.json"):
                    continue
                meta = json.load(t.extractfile(m))
                if not meta["scene_id"].startswith(prefix):
                    continue
                key = name[: -len(".meta.json")]
                raw = t.extractfile(members.get(key + ".pixels.i16dz") or members[key + ".pixels.npy"]).read()
                pix = decode_i16dz(raw) if (key + ".pixels.i16dz") in members else np.load(io.BytesIO(raw))
                wl = np.load(io.BytesIO(t.extractfile(members[key + ".wavelengths.npy"]).read()))
                labels = {k: np.load(io.BytesIO(t.extractfile(members[f"{key}.label_{k}.npy"]).read()))[0]
                          for k in ("classes", "cloud") if f"{key}.label_{k}.npy" in members}
                tiles.append((meta["y0"], meta["x0"], pix, wl, labels))
    if not tiles:
        raise SystemExit(f"{sensor}: no tiles for {prefix}")
    if sensor == "aviris_ng":                       # the HOST_ROWS window holding the most tiles
        starts = sorted({y for y, *_ in tiles})
        best = max(starts, key=lambda s: sum(s <= y < s + HOST_ROWS for y, *_ in tiles))
        tiles = [(y - best, x, p, w, l) for y, x, p, w, l in tiles if best <= y < best + HOST_ROWS]
    wl = tiles[0][3].astype(np.float64)
    for *_, w, _ in tiles:
        if w.shape != wl.shape or np.abs(w - wl).max() > 0.5:
            raise SystemExit(f"{prefix}: tiles carry different band lists")
    H, W = max(y for y, *_ in tiles) + 128, max(x for _, x, *_ in tiles) + 128
    cube = np.full((len(wl), H, W), NODATA, np.int16)
    labels = {k: np.zeros((H, W), np.uint8) for k in tiles[0][4]}
    for y, x, pix, _, lab in tiles:
        cube[:, y:y + 128, x:x + 128] = pix
        for k, v in lab.items():
            labels[k][y:y + 128, x:x + 128] = v
    return cube, wl, labels


def library_for(wl: np.ndarray, cache_dir: Path) -> list:
    from app.spectral_match.library import build_sensor_cache, load_slim_bundle
    fwhm = np.clip(np.abs(np.gradient(wl)), 4.0, 15.0)   # band spacing as width (PRISMA lists bands descending)
    entries, _ = build_sensor_cache(bundle=load_slim_bundle(REPO / "data" / "splib07_slim"), sensor_id="implant",
                                    target_wl_nm=wl, target_fwhm_nm=fwhm, cache_dir=cache_dir,
                                    chapters=list(LIB_CHAPTERS), min_coverage=0.9, progress=False)
    return entries


def place_and_implant(rng, refl, wl, labels, real, lib, version: int = 1, psf_sigma: float = 0.0,
                      foreign_tol: float = 1.5):
    """Implant objects into refl (B, H, W float, NaN = nodata) in place. Returns (gt (H, W) int32, records).

    version 2 (docs/lld/implant-bench.md, "bench-v2"): z and foreignness drawn as a pair from one real object; the
    difference spectrum's out-of-scene part rescaled to that foreignness; a third source, a material from elsewhere
    in the same host; additive implant through a PSF-blurred footprint; z solved on the realised ring contrast."""
    finite = np.isfinite(refl)
    valid = finite.mean(0) >= 0.9
    vis, nir = band_means(refl, wl, 450, 650), band_means(refl, wl, 800, 900)
    ndvi = (nir - vis) / (nir + vis + 1e-6)
    if "classes" in labels:                         # EnMAP: the provider's labels (class 0 = cloud, 2 = water)
        water, cloud = labels["classes"] == 2, (labels["classes"] == 0) | (labels.get("cloud", 0) == 1)
    else:
        water, cloud = (nir < 0.08) & (ndvi < 0.0), vis > 0.35
    eligible = ndimage.binary_erosion(valid & ~water & ~cloud & np.isfinite(ndvi), iterations=4)
    # NDVI mix of the real backgrounds: weight each eligible pixel by real share / host share of its NDVI bin
    edges = np.linspace(-1, 1, 21)
    real_h = np.histogram([o["ndvi"] for o in real["objects"]], edges)[0] / len(real["objects"])
    bins = np.clip(np.digitize(ndvi, edges) - 1, 0, 19)
    host_h = np.bincount(bins[eligible], minlength=20) / max(eligible.sum(), 1)
    weight = np.where(eligible, (real_h / np.maximum(host_h, 1e-9))[bins], 0.0)
    n_obj = int(round(eligible.sum() * len(real["objects"]) / real["valid_px"]))
    by_tier = {t: [o for o in real["objects"] if tier_of(o["z"]) == t] for t in TIERS}
    real_wl = {k: np.asarray(v) for k, v in real["wavelengths_nm"].items()}
    usable_lib = [e for e in lib if np.isfinite(e.refl).mean() >= 0.95]
    chapters = sorted({e.chapter for e in usable_lib})
    gt = np.zeros(refl.shape[1:], np.int32)
    blocked = np.zeros_like(eligible)
    records, H, W = [], *refl.shape[1:]
    flat_w = weight.ravel()
    sources = ("signature", "material") if version == 1 else ("signature", "material", "host")
    margin = 4 if version == 1 else 7
    # v2 measures and matches on the bands the evaluation reads (app/evaluation/datasets.load_implant_host): the
    # product exclusions out, bands mostly empty over valid pixels out. The real objects were measured on the
    # product reading; the noisy absorption / edge bands would otherwise swamp foreignness and contrast.
    from app.evaluation.datasets import PRODUCT_EXCLUSIONS
    ev = np.array([not any(lo <= w <= hi for lo, hi in PRODUCT_EXCLUSIONS) for w in wl]) & (finite[:, valid].mean(1) >= 0.6)
    P = background_subspace(refl[ev], valid) if version == 2 else None
    land = np.flatnonzero(eligible.ravel())
    for k in range(n_obj):
        tier, source = TIERS[k % 3], sources[(k // 3) % len(sources)]
        size = int(min(rng.choice([o["px"] for o in real["objects"]]), MAX_SIZE_PX))
        if version == 1:
            z_target, f_target = float(rng.choice([o["z"] for o in by_tier[tier]])), None
        else:                                       # contrast and foreignness as a pair, from one real object
            pick = by_tier[tier][int(rng.integers(len(by_tier[tier])))]
            z_target, f_target = float(pick["z"]), float(pick["foreign"])
        if source == "signature":
            j = int(rng.integers(len(real["objects"])))
            rw = real_wl[real["objects"][j]["scene"]]
            near = np.abs(wl[:, None] - rw[None]).min(1) <= SIGNATURE_GAP_NM
            spectrum = np.where(near, interp_spectrum(wl, rw, real["objects"][j]["d"]), np.nan)
            what = f"real:{real['objects'][j]['scene']}#{j}"
        elif source == "material":
            chapter = chapters[int(rng.integers(len(chapters)))]
            pool = [x for x in usable_lib if x.chapter == chapter]
            e = pool[int(rng.integers(len(pool)))]
            spectrum, what = e.refl, f"{e.chapter}:{e.name}"
        else:                                       # host: a material from elsewhere in this scene (v2)
            spectrum, what = None, "host"
        for _ in range(200):                        # a free, eligible spot where the target contrast is reachable
            c = rng.choice(flat_w.size, p=flat_w / flat_w.sum())
            cy, cx = divmod(int(c), W)
            off = grow_blob(rng, size) + (cy, cx)
            if not ((off.min() >= margin) and off[:, 0].max() < H - margin and off[:, 1].max() < W - margin
                    and eligible[off[:, 0], off[:, 1]].all() and not blocked[off[:, 0], off[:, 1]].any()):
                continue
            y0, y1 = off[:, 0].min() - margin, off[:, 0].max() + margin + 1
            x0, x1 = off[:, 1].min() - margin, off[:, 1].max() + margin + 1
            win = (slice(None), slice(y0, y1), slice(x0, x1))
            obj = np.zeros((y1 - y0, x1 - x0), bool)
            obj[off[:, 0] - y0, off[:, 1] - x0] = True
            sub, fin = refl[win], np.isfinite(refl[win])
            ring = ring_mask(obj, valid[y0:y1, x0:x1], exclude=gt[y0:y1, x0:x1] > 0)
            rpx = np.where(fin[:, ring], sub[:, ring], np.nan)
            r, sigma, o0 = np.nanmean(rpx, 1), np.nanstd(rpx, 1), np.nanmean(sub[:, obj], 1)
            if version == 1:
                step = spectrum if source == "signature" else spectrum - o0
                strength, _ = solve_strength(o0, r, sigma, step, z_target, s_max=None if source == "signature" else 1.0)
            else:
                rpx_e = np.where(fin[ev][:, ring], sub[ev][:, ring], np.nan)
                r_e, sigma_e, o0_e = np.nanmean(rpx_e, 1), np.nanstd(rpx_e, 1), np.nanmean(sub[ev][:, obj], 1)
                if source == "host":                # a land pixel >= 100 px away
                    py, px = divmod(int(land[int(rng.integers(len(land)))]), W)
                    if abs(py - cy) + abs(px - cx) < 100:
                        continue
                    d0, what = refl[:, py, px] - r, f"host:{py},{px}"
                else:
                    d0 = spectrum if source == "signature" else spectrum - r
                wgt = psf_weights(obj, psf_sigma)       # solve on the realised object (implant.solve_realised)
                got = solve_realised(o0_e, r_e, sigma_e, d0[ev], P, z_target, f_target, 1.0 - wgt[ring].mean(),
                                     tol=foreign_tol)
                if got is None:
                    continue
                strength, d_ev = got
                d = np.array(d0, dtype=np.float64)
                d[ev] = d_ev
            if strength is not None:
                break
        else:
            continue
        if version == 2:
            sub += strength * wgt[None] * np.nan_to_num(d)[:, None, None]
            np.maximum(sub, 0.0, out=sub, where=np.isfinite(sub))
        elif source == "signature":
            implant_signature(sub, obj, spectrum, strength)
        else:
            implant_material(sub, obj, spectrum, strength)
        se = sub[ev] if version == 2 else sub
        z = contrast_z(se, np.isfinite(se), obj, ring)
        extra = {}
        if version == 2:
            halo = ndimage.binary_dilation(obj) & ~obj & valid[y0:y1, x0:x1] & ~(gt[y0:y1, x0:x1] > 0)
            o_new = np.nanmean(se[:, obj], 1)
            r_new = np.nanmean(np.where(np.isfinite(se[:, ring]), se[:, ring], np.nan), 1)
            extra = dict(foreign_target=round(f_target, 5), foreign=round(foreign_fraction(o_new - r_new, P), 5),
                         halo=round(contrast_z(se, np.isfinite(se), halo, ring) / max(z, 1e-6), 4)
                         if halo.sum() >= 2 else None)
        oid = len(records) + 1
        gt[y0:y1, x0:x1][obj] = oid
        blocked[max(0, y0 - SPACING_PX):y1 + SPACING_PX, max(0, x0 - SPACING_PX):x1 + SPACING_PX] = True
        records.append(dict(id=oid, y=cy, x=cx, px=size, source=source, what=what, tier_target=tier,
                            z_target=round(z_target, 4), strength=round(strength, 5), z=round(z, 4), tier=tier_of(z),
                            ndvi=round(float(np.nanmean(ndvi[y0:y1, x0:x1][ring])), 3), **extra))
    return gt, records


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def build(shards: str, real_path: str, out: str, seed: int, version: int = 1, psf_sigma: float = 0.0,
          hosts: dict | None = None, foreign_tol: float = 1.5) -> None:
    real = json.load(open(real_path))
    rng = np.random.default_rng(seed)
    out_dir = Path(out)
    manifest = {"seed": seed, "version": version, "psf_sigma": psf_sigma, "foreign_tol": foreign_tol,
                "real": Path(real_path).name,
                "real_sha256": sha256(Path(real_path)), "hosts": []}
    for sensor, prefixes in (hosts or HOSTS).items():
        for prefix in prefixes:
            cube, wl, labels = reassemble(Path(shards), sensor, prefix)
            refl = np.where(cube == NODATA, np.nan, cube.astype(np.float32) * SCALE)
            del cube
            lib = library_for(wl, out_dir / "libcache")
            gt, records = place_and_implant(rng, refl, wl, labels, real, lib, version, psf_sigma, foreign_tol)
            host = out_dir / sensor / prefix
            host.mkdir(parents=True, exist_ok=True)
            np.save(host / "cube.npy", np.where(np.isfinite(refl), np.round(refl / SCALE), NODATA).astype(np.int16))
            np.save(host / "wavelengths.npy", wl)
            np.save(host / "gt.npy", gt)
            json.dump(records, open(host / "objects.json", "w"), indent=0)
            tiers = {t: sum(r["tier"] == t for r in records) for t in TIERS}
            print(f"{sensor} {prefix}: {refl.shape} {len(records)} objects {tiers}", flush=True)
            manifest["hosts"].append({"sensor": sensor, "scene": prefix, "shape": list(refl.shape),
                                      "objects": len(records), "tiers": tiers,
                                      "files": {f.name: sha256(f) for f in sorted(host.iterdir())}})
            del refl
    json.dump(manifest, open(out_dir / "manifest.json", "w"), indent=1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("real"); r.add_argument("root"); r.add_argument("out")
    b = sub.add_parser("build"); b.add_argument("shards"); b.add_argument("real"); b.add_argument("out")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--version", type=int, choices=(1, 2), default=1)
    b.add_argument("--psf-sigma", type=float, default=0.0, help="v2: Gaussian PSF std in px")
    b.add_argument("--foreign-tol", type=float, default=1.5, help="v2: accept a spot only if its realised foreignness is within this factor")
    b.add_argument("--hosts", help="sensor=prefix,prefix;sensor=... subset (calibration runs)")
    a = p.parse_args()
    if a.cmd == "real":
        measure_real(a.root, a.out)
    else:
        hosts = None if not a.hosts else {k: v.split(",") for k, v in (x.split("=") for x in a.hosts.split(";"))}
        build(a.shards, a.real, a.out, a.seed, a.version, a.psf_sigma, hosts, a.foreign_tol)


if __name__ == "__main__":
    main()
