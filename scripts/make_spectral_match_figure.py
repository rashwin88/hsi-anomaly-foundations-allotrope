"""Publication-grade figure for the spectral-library-match pipeline.

Flow:
  ① Anomaly pixel extraction (from upstream anomaly map)
  ② Gaussian SRF resampling of USGS splib07 onto the sensor grid
  ③ Savitzky-Golay smoothing of unknown spectra
  ④ Pattern-bucketed Spectral Angle Mapper (SAM) → top-K matches
  ⑤ Confidence map + material assignment

Output: final design/diagrams/spectral_match_pipeline.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "final design/diagrams/spectral_match_pipeline"


# ------------------------------------------------------------------
# Synthesis (matches the shape of real EnMAP / PRISMA data)
# ------------------------------------------------------------------

def fake_library_spectrum(wl_nm, kind):
    """Build a plausible high-resolution lab reflectance spectrum."""
    x = wl_nm
    if kind == "kaolinite":
        # broad VNIR rise + diagnostic 1400 OH + 2200 Al-OH doublet
        r = 0.35 + 0.30 * np.exp(-((x - 600) ** 2) / (2 * 350 ** 2))
        r -= 0.12 * np.exp(-((x - 1400) ** 2) / (2 * 25 ** 2))
        r -= 0.18 * np.exp(-((x - 2165) ** 2) / (2 * 25 ** 2))
        r -= 0.14 * np.exp(-((x - 2208) ** 2) / (2 * 18 ** 2))
    elif kind == "calcite":
        r = 0.55 + 0.10 * np.exp(-((x - 800) ** 2) / (2 * 600 ** 2))
        r -= 0.18 * np.exp(-((x - 2340) ** 2) / (2 * 35 ** 2))
    elif kind == "gypsum":
        r = 0.65 + 0.05 * np.exp(-((x - 700) ** 2) / (2 * 500 ** 2))
        r -= 0.20 * np.exp(-((x - 1450) ** 2) / (2 * 35 ** 2))
        r -= 0.15 * np.exp(-((x - 1750) ** 2) / (2 * 30 ** 2))
        r -= 0.18 * np.exp(-((x - 1940) ** 2) / (2 * 35 ** 2))
    elif kind == "vegetation":
        r = 0.05 + 0.50 / (1 + np.exp(-(x - 720) / 15))
        r *= np.exp(-((x - 1000) ** 2) / (2 * 1500 ** 2))
        r -= 0.10 * np.exp(-((x - 1450) ** 2) / (2 * 40 ** 2))
        r -= 0.13 * np.exp(-((x - 1940) ** 2) / (2 * 40 ** 2))
    else:
        r = 0.4 + 0 * x
    return np.clip(r, 0.0, 1.0)


def gaussian_srf_resample(lib_wl, lib_refl, target_wl, target_fwhm,
                          n_sigma=3.0):
    """Mimic gaussian_resample_to_target — Gaussian SRF integration."""
    sigma = target_fwhm / (2 * np.sqrt(2 * np.log(2)))
    out = np.full(target_wl.shape, np.nan, dtype=np.float64)
    for i, (mu, s) in enumerate(zip(target_wl, sigma)):
        lo, hi = mu - n_sigma * s, mu + n_sigma * s
        m = (lib_wl >= lo) & (lib_wl <= hi)
        if m.sum() < 3:
            continue
        w = np.exp(-((lib_wl[m] - mu) ** 2) / (2 * s ** 2))
        out[i] = (w * lib_refl[m]).sum() / w.sum()
    return out


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 10.5,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })

    # ---- 1. high-res library spectra ----
    lib_wl = np.arange(350.0, 2500.5, 1.0)
    libs = {
        "kaolinite":  fake_library_spectrum(lib_wl, "kaolinite"),
        "calcite":    fake_library_spectrum(lib_wl, "calcite"),
        "gypsum":     fake_library_spectrum(lib_wl, "gypsum"),
        "vegetation": fake_library_spectrum(lib_wl, "vegetation"),
    }

    # ---- 2. sensor grid (EnMAP-ish, 224 bands 420-2440 nm) ----
    target_wl = np.linspace(420.0, 2440.0, 224)
    target_fwhm = np.full_like(target_wl, 6.5)
    target_fwhm[target_wl > 1000] = 10.0  # SWIR has wider bands

    # H2O windows where SWIR bands are atmospherically gone
    bad = (
        ((target_wl > 1340) & (target_wl < 1450))
        | ((target_wl > 1790) & (target_wl < 1960))
        | (target_wl > 2400)
    )

    # ---- 3. resample library to sensor grid (with NaNs in bad bands) ----
    lib_on_sensor = {}
    for name, refl in libs.items():
        s = gaussian_srf_resample(lib_wl, refl, target_wl, target_fwhm)
        s[bad] = np.nan
        lib_on_sensor[name] = s

    # ---- 4. fake "unknown" pixel: kaolinite + scene illumination + noise ----
    rng = np.random.default_rng(1)
    noise = 0.02 * rng.standard_normal(target_wl.shape)
    illum = 0.95 + 0.05 * np.sin(target_wl / 300.0)
    unknown_raw = lib_on_sensor["kaolinite"].copy() * illum + noise
    unknown_raw[bad] = np.nan

    # ---- 5. Savitzky-Golay smoothing (ignore NaN gaps) ----
    def smooth(arr, w=11, p=3):
        out = arr.copy()
        good = ~np.isnan(arr)
        idx = np.arange(len(arr))
        if good.sum() >= 5:
            filled = PchipInterpolator(idx[good], arr[good],
                                       extrapolate=False)(idx)
            # back-fill any remaining NaNs (outside PCHIP support)
            if np.any(np.isnan(filled)):
                first = np.nanmin(arr[good])
                filled = np.where(np.isnan(filled), first, filled)
            sm = savgol_filter(filled, w, p, mode="interp")
            out = np.where(good, sm, np.nan)
        return out

    unknown_smooth = smooth(unknown_raw, w=11, p=3)

    # ---- 6. SAM (cosine angle in degrees) ----
    def sam_deg(u, l):
        m = ~np.isnan(u) & ~np.isnan(l)
        if m.sum() < 5:
            return np.nan
        a = u[m]; b = l[m]
        cos = np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12),
                      -1, 1)
        return float(np.degrees(np.arccos(cos)))

    angles = {name: sam_deg(unknown_smooth, lib)
              for name, lib in lib_on_sensor.items()}

    # ---- 7. fake confidence map + match assignment ----
    H, W = 100, 130
    yy, xx = np.mgrid[0:H, 0:W]
    # anomaly mask is the same blob set used elsewhere
    rng2 = np.random.default_rng(7)
    blob1 = np.exp(-((yy - 35) ** 2 + (xx - 50) ** 2) / (2 * 5 ** 2))
    blob2 = np.exp(-((yy - 60) ** 2 + (xx - 90) ** 2) / (2 * 4 ** 2))
    blob3 = np.exp(-((yy - 70) ** 2 + (xx - 30) ** 2) / (2 * 4 ** 2))
    anom = (blob1 + 0.8 * blob2 + 0.7 * blob3) > 0.4

    # per-anomaly fake assignment (just for visual): cluster by location
    assign = np.full((H, W), -1, dtype=int)
    assign[blob1 > 0.4] = 0      # kaolinite
    assign[blob2 > 0.4] = 1      # calcite
    assign[blob3 > 0.4] = 2      # gypsum
    angles_map = np.full((H, W), np.nan)
    angles_map[blob1 > 0.4] = 4.2 + 1.5 * rng2.standard_normal(int((blob1 > 0.4).sum()))
    angles_map[blob2 > 0.4] = 7.5 + 2.0 * rng2.standard_normal(int((blob2 > 0.4).sum()))
    angles_map[blob3 > 0.4] = 11.0 + 2.5 * rng2.standard_normal(int((blob3 > 0.4).sum()))
    angles_map = np.clip(angles_map, 1.0, 18.0)

    # ==================================================================
    # ----------------------------- FIGURE -----------------------------
    # ==================================================================
    fig = plt.figure(figsize=(16.5, 11.5))
    gs = GridSpec(
        4, 4, figure=fig,
        height_ratios=[0.32, 1.0, 1.0, 1.1],
        width_ratios=[1, 1, 1, 1],
        hspace=0.55, wspace=0.32,
        left=0.06, right=0.985, top=0.93, bottom=0.05,
    )

    fig.suptitle(
        "Allotrope · Spectral Library Match (USGS splib07)",
        fontsize=18, fontweight="bold", y=0.985,
    )
    fig.text(0.5, 0.945,
             "Lab reflectance → Gaussian-SRF resample to sensor grid "
             "→ Savitzky-Golay smooth → pattern-bucketed SAM → "
             "confidence-thresholded material assignment",
             ha="center", fontsize=11, color="#475569")

    # =================== Row 0: flow strip ===================
    ax_flow = fig.add_subplot(gs[0, :])
    ax_flow.set_xlim(0, 10); ax_flow.set_ylim(0, 1); ax_flow.axis("off")
    stages = [
        ("Anomaly pixels\n(from upstream)",       "#fee2e2", "#b91c1c"),
        ("USGS splib07\nlab reflectance",         "#fef3c7", "#92400e"),
        ("Gaussian-SRF\nresample → sensor grid",  "#dcfce7", "#166534"),
        ("Savitzky-Golay\nsmoothing",             "#dbeafe", "#1e40af"),
        ("SAM\npattern-bucketed",                 "#ede9fe", "#5b21b6"),
        ("Top-K + confidence\nmaterial map",      "#fee2e2", "#b91c1c"),
    ]
    n = len(stages); pad = 0.10
    box_w = (10 - 2 * pad - (n - 1) * 0.30) / n
    centers = []
    for i, (txt, fc, ec) in enumerate(stages):
        x0 = pad + i * (box_w + 0.30)
        ax_flow.add_patch(FancyBboxPatch(
            (x0, 0.18), box_w, 0.66,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.5, edgecolor=ec, facecolor=fc, zorder=2,
        ))
        ax_flow.text(x0 + box_w / 2, 0.51, txt,
                     ha="center", va="center", fontsize=10, color=ec,
                     fontweight="bold")
        centers.append((x0, x0 + box_w))
    for i in range(n - 1):
        ax_flow.add_patch(FancyArrowPatch(
            (centers[i][1], 0.51), (centers[i + 1][0], 0.51),
            arrowstyle="-|>", mutation_scale=14,
            linewidth=1.6, color="#374151", zorder=1,
        ))

    # =================== Row 1: library + resample + smoothing ===================
    # (1) lab reflectance library
    axA = fig.add_subplot(gs[1, 0])
    cmap = {"kaolinite": "#b45309", "calcite": "#0ea5e9",
            "gypsum": "#84cc16", "vegetation": "#16a34a"}
    for name in ["kaolinite", "calcite", "gypsum", "vegetation"]:
        axA.plot(lib_wl, libs[name], color=cmap[name], lw=1.2, label=name)
    axA.set_title("① USGS splib07  ·  lab reflectance")
    axA.set_xlabel("λ (nm)"); axA.set_ylabel("Reflectance")
    axA.set_xlim(400, 2500); axA.set_ylim(0, 1)
    axA.legend(loc="upper right", frameon=False, fontsize=8.5)
    axA.grid(alpha=0.25)

    # (2) SRF resample illustration
    axB = fig.add_subplot(gs[1, 1])
    axB.plot(lib_wl, libs["kaolinite"], color="#b45309", lw=1.0, alpha=0.5,
             label="lab (1 nm)")
    axB.plot(target_wl, lib_on_sensor["kaolinite"],
             color="#166534", marker="o", ms=2.5, lw=1.0,
             label="sensor bands (~10 nm)")
    # show a single SRF window
    mu = 2200.0; s = 10.0 / (2 * np.sqrt(2 * np.log(2)))
    x_srf = np.linspace(mu - 4 * s, mu + 4 * s, 200)
    srf = np.exp(-((x_srf - mu) ** 2) / (2 * s ** 2))
    axB.fill_between(x_srf, 0, srf * 0.10, color="#166534", alpha=0.3,
                     label="Gaussian SRF\n(FWHM=10 nm)")
    axB.set_title("② Gaussian-SRF resampling")
    axB.set_xlabel("λ (nm)"); axB.set_ylabel("Reflectance")
    axB.set_xlim(2050, 2350); axB.set_ylim(0, 0.45)
    axB.legend(loc="lower left", frameon=False, fontsize=8.5)
    axB.grid(alpha=0.25)

    # (3) noisy + smoothed unknown pixel
    axC = fig.add_subplot(gs[1, 2])
    axC.plot(target_wl, unknown_raw, color="#94a3b8", lw=0.9,
             label="raw unknown")
    axC.plot(target_wl, unknown_smooth, color="#1e40af", lw=1.6,
             label="Savitzky-Golay\n(w=11, p=3)")
    for lo, hi in [(1340, 1450), (1790, 1960), (2400, 2500)]:
        axC.axvspan(lo, hi, color="#b91c1c", alpha=0.10)
    axC.set_title("③ Smoothing the unknown pixel")
    axC.set_xlabel("λ (nm)"); axC.set_ylabel("Reflectance")
    axC.set_xlim(target_wl.min(), target_wl.max())
    axC.legend(loc="upper right", frameon=False, fontsize=8.5)
    axC.grid(alpha=0.25)

    # (4) library candidates overlaid on the smoothed unknown
    axD = fig.add_subplot(gs[1, 3])
    axD.plot(target_wl, unknown_smooth, color="#1e40af", lw=2.0,
             label=f"unknown", zorder=5)
    for name in ["kaolinite", "calcite", "gypsum", "vegetation"]:
        axD.plot(target_wl, lib_on_sensor[name], color=cmap[name],
                 lw=1.0, alpha=0.85,
                 label=f"{name}  ({angles[name]:.2f}°)")
    for lo, hi in [(1340, 1450), (1790, 1960), (2400, 2500)]:
        axD.axvspan(lo, hi, color="#b91c1c", alpha=0.08)
    axD.set_title("④ Candidate overlay  ·  SAM angle")
    axD.set_xlabel("λ (nm)"); axD.set_ylabel("Reflectance")
    axD.set_xlim(target_wl.min(), target_wl.max())
    axD.legend(loc="upper right", frameon=False, fontsize=8)
    axD.grid(alpha=0.25)

    # =================== Row 2: SAM mechanics ===================
    # (5) SAM geometric intuition (2D projection)
    axE = fig.add_subplot(gs[2, 0])
    # show 3 vectors emanating from origin
    axE.set_xlim(-0.05, 1.0); axE.set_ylim(-0.05, 1.0)
    axE.set_aspect("equal")
    axE.spines["left"].set_visible(True)
    axE.spines["bottom"].set_visible(True)
    axE.spines["left"].set_color("#94a3b8")
    axE.spines["bottom"].set_color("#94a3b8")
    axE.set_title("⑤ SAM geometry")
    axE.set_xlabel("band A reflectance")
    axE.set_ylabel("band B reflectance")
    vecs = [("unknown", (0.7, 0.55), "#1e40af"),
            ("kaolinite", (0.78, 0.50), "#b45309"),
            ("calcite",  (0.55, 0.78), "#0ea5e9")]
    for name, (vx, vy), c in vecs:
        axE.annotate("", xy=(vx, vy), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color=c, lw=2.0))
        axE.text(vx + 0.02, vy + 0.01, name, fontsize=8.5, color=c)
    # angle arc
    theta1 = np.degrees(np.arctan2(0.55, 0.7))
    theta2 = np.degrees(np.arctan2(0.50, 0.78))
    from matplotlib.patches import Arc
    axE.add_patch(Arc((0, 0), 0.45, 0.45, angle=0,
                      theta1=min(theta1, theta2), theta2=max(theta1, theta2),
                      color="#b45309", lw=1.6))
    axE.text(0.30, 0.16, "θ (small)", fontsize=8.5, color="#b45309")
    axE.grid(alpha=0.25)
    axE.text(0.02, 0.95,
             r"$\theta = \arccos\frac{\langle u, l\rangle}{\|u\|\|l\|}$",
             fontsize=11, transform=axE.transAxes, va="top",
             bbox=dict(boxstyle="round,pad=0.3", fc="#f8fafc", ec="#cbd5e1",
                       lw=0.6))

    # (6) Pattern bucketing visual
    axF = fig.add_subplot(gs[2, 1])
    # show a strip of B bands × P patterns
    B = 60
    P = 4
    grid = np.ones((P, B))
    rng3 = np.random.default_rng(11)
    masks = []
    for p in range(P):
        m = np.ones(B, dtype=bool)
        # mark water-window-like gaps
        gaps = [(15, 18), (32, 36)]
        for lo, hi in gaps:
            m[lo:hi] = False
        # per-pattern saturation
        if p > 0:
            sat_idx = rng3.integers(0, B, size=p + 1)
            m[sat_idx] = False
        masks.append(m)
        grid[p] = m.astype(float)
    axF.imshow(grid, aspect="auto", cmap="Greens", vmin=0, vmax=1.2,
               interpolation="nearest")
    axF.set_yticks(range(P))
    axF.set_yticklabels([f"pattern P{i + 1}\n({m.sum()} bands)"
                          for i, m in enumerate(masks)], fontsize=8.5)
    axF.set_xticks([])
    axF.set_xlabel("band index →")
    axF.set_title("⑥ Pattern-bucketed pixels")
    axF.text(
        0, -0.40,
        "Group anomaly pixels by valid-band pattern; one BLAS matmul per bucket.",
        transform=axF.transAxes, fontsize=8.5, color="#64748b",
    )

    # (7) Top-K SAM-angle bar chart
    axG = fig.add_subplot(gs[2, 2])
    names = list(angles.keys())
    vals = [angles[n] for n in names]
    order = np.argsort(vals)
    names_s = [names[i] for i in order]
    vals_s = [vals[i] for i in order]
    colors = [cmap[n] for n in names_s]
    bars = axG.barh(range(len(names_s)), vals_s, color=colors, edgecolor="white")
    axG.set_yticks(range(len(names_s)))
    axG.set_yticklabels(names_s)
    axG.invert_yaxis()
    axG.set_xlabel("SAM angle (°) · lower = closer match")
    axG.set_title("⑦ Top-K library matches")
    axG.axvline(8.0, color="#b91c1c", lw=1.0, linestyle="--", alpha=0.7)
    axG.text(8.2, len(names_s) - 0.3, "confidence\nthreshold",
             fontsize=8.5, color="#b91c1c")
    for i, v in enumerate(vals_s):
        axG.text(v + 0.4, i, f"{v:.2f}°", va="center", fontsize=9,
                 color="#1e293b")
    axG.set_xlim(0, max(vals_s) * 1.25)
    axG.grid(axis="x", alpha=0.25)

    # (8) Confidence histogram across anomaly pixels
    axH = fig.add_subplot(gs[2, 3])
    flat = angles_map[~np.isnan(angles_map)]
    axH.hist(flat, bins=24, color="#1e40af", edgecolor="white", alpha=0.85)
    axH.axvline(8.0, color="#b91c1c", lw=1.4, linestyle="--",
                label="threshold 8°")
    axH.set_xlabel("SAM angle (°)")
    axH.set_ylabel("anomaly pixel count")
    axH.set_title("⑧ Confidence distribution")
    axH.legend(loc="upper right", frameon=False, fontsize=8.5)
    axH.grid(alpha=0.25)

    # =================== Row 3: scene-level outputs ===================
    # (9) anomaly scene with kaolinite/calcite/gypsum overlay
    axI = fig.add_subplot(gs[3, 0])
    canvas = np.full((H, W, 4), 0.95)
    palette = [cmap["kaolinite"], cmap["calcite"], cmap["gypsum"]]
    palette_rgb = [tuple(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)) for c in palette]
    for k in range(3):
        m = (assign == k)
        canvas[m] = (*palette_rgb[k], 1.0)
    axI.imshow(canvas)
    axI.set_title("⑨ Material map  ·  per anomaly pixel")
    axI.set_xticks([]); axI.set_yticks([])
    # custom legend
    for k, (lab, col) in enumerate(zip(
            ["kaolinite", "calcite", "gypsum"], palette_rgb)):
        axI.add_patch(FancyBboxPatch(
            (4, 4 + k * 10), 7, 7,
            boxstyle="round,pad=0.01,rounding_size=1",
            facecolor=col, edgecolor="white",
            transform=axI.transData, linewidth=1.0,
        ))
        axI.text(14, 8 + k * 10, lab, fontsize=8.5, color="#0f172a")

    # (10) SAM-angle heatmap (confidence)
    axJ = fig.add_subplot(gs[3, 1])
    im = axJ.imshow(angles_map, cmap="viridis_r", vmin=1, vmax=15)
    axJ.set_title("⑩ Per-pixel SAM angle (°)")
    axJ.set_xticks([]); axJ.set_yticks([])
    cb = fig.colorbar(im, ax=axJ, shrink=0.85, pad=0.02)
    cb.set_label("SAM angle (°)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # (11) Outputs card
    axK = fig.add_subplot(gs[3, 2])
    axK.axis("off"); axK.set_xlim(0, 1); axK.set_ylim(0, 1)
    axK.add_patch(FancyBboxPatch(
        (0.02, 0.05), 0.96, 0.92,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.2, edgecolor="#5b21b6", facecolor="#faf5ff",
    ))
    axK.text(0.5, 0.90, "Outputs", ha="center", fontsize=12,
             fontweight="bold", color="#5b21b6")
    items = [
        "match_map.tif        per-pixel best library ID",
        "sam_angles.tif       per-pixel SAM angle (°)",
        "top_k.npz            (P, K) angles + library indices",
        "matches.shp          polygonized vector output",
        "stats.json           per-material pixel counts",
    ]
    y = 0.78
    for it in items:
        axK.text(0.07, y, "•", fontsize=12, fontweight="bold", color="#7c3aed")
        axK.text(0.12, y, it, fontsize=9.5, color="#1e293b",
                 family="DejaVu Sans Mono", va="center")
        y -= 0.12

    # (12) Performance card
    axL = fig.add_subplot(gs[3, 3])
    axL.axis("off"); axL.set_xlim(0, 1); axL.set_ylim(0, 1)
    axL.add_patch(FancyBboxPatch(
        (0.02, 0.05), 0.96, 0.92,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.2, edgecolor="#15803D", facecolor="#dcfce7",
    ))
    axL.text(0.5, 0.90, "Performance", ha="center", fontsize=12,
             fontweight="bold", color="#15803D")
    perf = [
        ("Library size",       "~1,200 entries"),
        ("Anomaly pixels",     "a few thousand"),
        ("Distinct patterns",  "1 – 3 typically"),
        ("Match time (CPU)",   "< 2 s per scene"),
        ("Smoothing",          "Savitzky-Golay  w=11  p=3"),
        ("Min coverage",       "10 valid bands per pair"),
    ]
    y = 0.78
    for k, v in perf:
        axL.text(0.07, y, k, fontsize=9.5, fontweight="bold", color="#166534")
        axL.text(0.55, y, v, fontsize=9.5, color="#1e293b")
        y -= 0.11

    fig.savefig(OUT.with_suffix(".png"), dpi=240, bbox_inches="tight",
                facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight",
                facecolor="white")
    print(f"wrote {OUT.with_suffix('.png')}")
    print(f"wrote {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
