"""Publication-quality figure: per-band standardization to a fixed grid via PCHIP,
with invalid-band masking. Uses a real EnMAP L2A patch.

Output: final design/diagrams/band_standardization_pipeline.png (+ .pdf)
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent.parent
SCENE = (
    ROOT / "tests/test_payloads/"
    "ENMAP01-____L2A-DT0000118813_20250309T052400Z_002_V010506_20260305T174411Z"
)
TIF = SCENE / (SCENE.name + "-SPECTRAL_IMAGE.TIF")
XML = SCENE / (SCENE.name + "-METADATA.XML")
OUT = ROOT / "final design/diagrams/band_standardization_pipeline"


def load_wavelengths(xml_path: Path) -> np.ndarray:
    root = ET.parse(xml_path).getroot()
    vals = []
    for el in root.iter():
        if el.tag.split("}")[-1] == "wavelengthCenterOfBand":
            try:
                vals.append(float(el.text))
            except (TypeError, ValueError):
                pass
    return np.asarray(vals, dtype=float)


def load_patch(tif: Path, size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Return (cube[B,H,W] reflectance 0..1, validity[B,H,W] bool)."""
    with rasterio.open(tif) as src:
        nodata = src.nodata
        h, w = src.height, src.width
        # pick a window with content (avoid black borders)
        row, col = h // 3, w // 3
        win = rasterio.windows.Window(col, row, size, size)
        arr = src.read(window=win).astype(np.float32)
    valid = arr != (nodata if nodata is not None else -32768)
    # EnMAP L2A reflectance scale = 10000
    refl = np.where(valid, arr / 10000.0, np.nan)
    return refl, valid


def water_absorption_mask(wl: np.ndarray) -> np.ndarray:
    """Flag known atmospheric water absorption windows as invalid."""
    bad = np.zeros_like(wl, dtype=bool)
    for lo, hi in [(1340, 1450), (1790, 1960), (2400, 2500)]:
        bad |= (wl >= lo) & (wl <= hi)
    return bad


def main() -> None:
    wl = load_wavelengths(XML)
    cube, valid = load_patch(TIF, size=128)
    B = cube.shape[0]
    wl = wl[:B]

    # Center pixel spectrum for the spectral panels.
    cy, cx = cube.shape[1] // 2, cube.shape[2] // 2
    spec = cube[:, cy, cx].copy()

    # Native validity (nodata) OR water-vapor windows -> "bad bands".
    bad_bands = water_absorption_mask(wl) | np.isnan(spec)
    good = ~bad_bands

    # Fixed common grid (5 nm from 420..2440 nm).
    grid = np.arange(420.0, 2441.0, 5.0)

    # PCHIP on good samples only.
    pchip = PchipInterpolator(wl[good], spec[good], extrapolate=False)
    spec_pchip = pchip(grid)

    # Linear comparison.
    spec_linear = np.interp(grid, wl[good], spec[good], left=np.nan, right=np.nan)

    # ---------- figure ----------
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
    })

    fig = plt.figure(figsize=(15, 9.5))
    gs = GridSpec(
        3, 6,
        figure=fig,
        height_ratios=[0.55, 1.0, 1.0],
        width_ratios=[1, 1, 1, 1, 1, 1],
        hspace=0.55, wspace=0.55,
        left=0.05, right=0.98, top=0.93, bottom=0.06,
    )

    fig.suptitle(
        "Per-Band Standardization to a Common Spectral Grid",
        fontsize=14, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.955,
        "EnMAP L2A patch  ·  native bands → bad-band mask → PCHIP resample → fixed 5 nm grid",
        ha="center", fontsize=10, color="#555",
    )

    # ===== Row 0: flow diagram =====
    ax_flow = fig.add_subplot(gs[0, :])
    ax_flow.set_xlim(0, 10)
    ax_flow.set_ylim(0, 1)
    ax_flow.axis("off")

    stages = [
        ("Native cube\n(B native bands,\nirregular Δλ)", "#dbeafe", "#1e40af"),
        ("Bad-band mask\n(nodata ∪ H₂O\nabsorption windows)", "#fee2e2", "#b91c1c"),
        ("PCHIP\ninterpolation\nover good samples", "#fef3c7", "#92400e"),
        ("Resample onto\nfixed common grid\n(420–2440 nm, 5 nm)", "#dcfce7", "#166534"),
        ("Standardized cube\n(uniform B, shared λ\nacross sensors)", "#ede9fe", "#5b21b6"),
    ]
    n = len(stages)
    pad = 0.15
    box_w = (10 - 2 * pad - (n - 1) * 0.45) / n
    centers = []
    for i, (txt, fc, ec) in enumerate(stages):
        x0 = pad + i * (box_w + 0.45)
        box = FancyBboxPatch(
            (x0, 0.25), box_w, 0.55,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.5, edgecolor=ec, facecolor=fc, zorder=2,
        )
        ax_flow.add_patch(box)
        ax_flow.text(
            x0 + box_w / 2, 0.525, txt,
            ha="center", va="center", fontsize=9.5, color=ec, fontweight="bold",
        )
        centers.append((x0, x0 + box_w))
    for i in range(n - 1):
        a = FancyArrowPatch(
            (centers[i][1], 0.525), (centers[i + 1][0], 0.525),
            arrowstyle="-|>", mutation_scale=18,
            linewidth=1.8, color="#374151", zorder=1,
        )
        ax_flow.add_patch(a)

    # ===== Row 1: spectral panels =====
    # (1) raw spectrum with bad bands shaded
    ax1 = fig.add_subplot(gs[1, 0:2])
    ax1.plot(wl, spec, color="#1e40af", lw=1.2, label="Native EnMAP spectrum")
    for lo, hi in [(1340, 1450), (1790, 1960), (2400, 2500)]:
        ax1.axvspan(lo, hi, color="#b91c1c", alpha=0.15)
    ax1.scatter(
        wl[bad_bands], np.where(bad_bands, spec, np.nan)[bad_bands],
        s=8, color="#b91c1c", zorder=3, label="Masked bands",
    )
    ax1.set_title("① Native spectrum + bad-band mask")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Reflectance")
    ax1.legend(loc="upper right", frameon=False)
    ax1.set_xlim(wl.min(), wl.max())

    # (2) PCHIP vs linear over a zoomed region (covers a water gap)
    ax2 = fig.add_subplot(gs[1, 2:4])
    zoom_lo, zoom_hi = 1250, 1550
    g_z = (grid >= zoom_lo) & (grid <= zoom_hi)
    w_z = (wl >= zoom_lo) & (wl <= zoom_hi) & good
    ax2.scatter(wl[w_z], spec[w_z], s=22, color="#1e40af",
                label="Good samples", zorder=4)
    ax2.plot(grid[g_z], spec_pchip[g_z], color="#92400e", lw=2.0,
             label="PCHIP", zorder=3)
    ax2.plot(grid[g_z], spec_linear[g_z], color="#9ca3af", lw=1.5,
             linestyle="--", label="Linear (reference)", zorder=2)
    ax2.axvspan(1340, 1450, color="#b91c1c", alpha=0.12, label="Masked window")
    ax2.set_title("② PCHIP across a masked window (zoom)")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Reflectance")
    ax2.legend(loc="upper right", frameon=False)
    ax2.set_xlim(zoom_lo, zoom_hi)

    # (3) resampled spectrum on fixed grid
    ax3 = fig.add_subplot(gs[1, 4:6])
    ax3.plot(grid, spec_pchip, color="#166534", lw=1.3,
             label="Resampled (5 nm grid)")
    ax3.plot(wl, spec, color="#1e40af", lw=0.8, alpha=0.35,
             label="Native (overlay)")
    for lo, hi in [(1340, 1450), (1790, 1960), (2400, 2500)]:
        ax3.axvspan(lo, hi, color="#b91c1c", alpha=0.10)
    ax3.set_title("③ Standardized spectrum on common grid")
    ax3.set_xlabel("Wavelength (nm)")
    ax3.set_ylabel("Reflectance")
    ax3.legend(loc="upper right", frameon=False)
    ax3.set_xlim(grid.min(), grid.max())

    # ===== Row 2: patch progression =====
    # Build a 'standardized cube' for visualization: a thin selection on the grid.
    cube_good = np.where(bad_bands[:, None, None], np.nan, cube)
    H, W = cube.shape[1:]

    # Pick three reference wavelengths near R/G/B-ish + a SWIR + a water gap.
    def nearest(arr, x):
        return int(np.argmin(np.abs(arr - x)))

    showcase_wls = [550, 660, 1400, 1650, 2200]

    def band_image(c, w_arr, target):
        i = nearest(w_arr, target)
        return c[i]

    # (4) raw native patch as RGB
    ax4 = fig.add_subplot(gs[2, 0])
    r = band_image(cube, wl, 660)
    g = band_image(cube, wl, 550)
    b = band_image(cube, wl, 470)
    rgb = np.dstack([r, g, b])
    rgb = np.clip((rgb - np.nanpercentile(rgb, 2)) /
                  (np.nanpercentile(rgb, 98) - np.nanpercentile(rgb, 2) + 1e-9), 0, 1)
    rgb = np.nan_to_num(rgb)
    ax4.imshow(rgb)
    ax4.set_title("Native patch (RGB)")
    ax4.set_xticks([]); ax4.set_yticks([])
    ax4.text(
        0.02, 0.02, f"{H}×{W}×{B}",
        transform=ax4.transAxes, color="white", fontsize=8,
        bbox=dict(facecolor="black", alpha=0.55, pad=2, edgecolor="none"),
    )

    # (5) bad-band overlay: show a single band that lives inside the water window
    ax5 = fig.add_subplot(gs[2, 1])
    bad_band_img = band_image(cube, wl, 1400)
    ax5.imshow(bad_band_img, cmap="gray")
    ax5.imshow(
        np.ones_like(bad_band_img), cmap="Reds", alpha=0.35, vmin=0, vmax=1,
    )
    ax5.set_title("Masked band (λ≈1400 nm)")
    ax5.set_xticks([]); ax5.set_yticks([])

    # (6–8) three resampled grid bands (after standardization)
    cube_resamp_preview = {}
    # Resample three target wavelengths across all pixels (small subset for speed).
    H2, W2 = cube.shape[1], cube.shape[2]
    flat = cube.reshape(B, -1)  # (B, N)
    good_mask = ~bad_bands
    # vectorize PCHIP across pixels: build once per pixel is slow; instead loop on
    # a strided subset for the preview bands only — fine for figure rendering.
    targets = [660, 1650, 2200]
    titles = ["Resampled λ=660 nm", "Resampled λ=1650 nm", "Resampled λ=2200 nm"]
    resamp_imgs = []
    for tgt in targets:
        out = np.full((H2, W2), np.nan, dtype=np.float32)
        # PCHIP per-pixel using only good bands (fast enough for 128x128).
        wls_g = wl[good_mask]
        for i in range(H2):
            for j in range(W2):
                y = flat[good_mask, i * W2 + j]
                if np.any(np.isnan(y)):
                    continue
                out[i, j] = PchipInterpolator(wls_g, y, extrapolate=False)(tgt)
        resamp_imgs.append(out)

    for k, (img, ttl) in enumerate(zip(resamp_imgs, titles)):
        ax = fig.add_subplot(gs[2, 2 + k])
        lo, hi = np.nanpercentile(img, 2), np.nanpercentile(img, 98)
        ax.imshow(img, cmap="viridis", vmin=lo, vmax=hi)
        ax.set_title(ttl)
        ax.set_xticks([]); ax.set_yticks([])

    # (9) standardized cube schematic
    ax9 = fig.add_subplot(gs[2, 5])
    ax9.set_xlim(0, 1); ax9.set_ylim(0, 1); ax9.axis("off")
    # draw a stack of slabs
    n_slabs = 12
    for i in range(n_slabs):
        y = 0.15 + i * 0.045
        rect = FancyBboxPatch(
            (0.18 - i * 0.012, y), 0.55, 0.06,
            boxstyle="round,pad=0.005,rounding_size=0.01",
            linewidth=0.8, edgecolor="#5b21b6",
            facecolor=plt.cm.viridis(i / n_slabs), alpha=0.85,
        )
        ax9.add_patch(rect)
    ax9.text(
        0.5, 0.06,
        "Standardized cube\nuniform Δλ = 5 nm\nshared across sensors",
        ha="center", va="center", fontsize=8.5, color="#5b21b6",
        fontweight="bold",
    )
    ax9.text(0.5, 0.92, "④ Output", ha="center", fontsize=10, fontweight="bold")

    # row labels
    fig.text(0.012, 0.83, "Pipeline", rotation=90, va="center",
             fontsize=10, fontweight="bold", color="#374151")
    fig.text(0.012, 0.50, "Spectral view", rotation=90, va="center",
             fontsize=10, fontweight="bold", color="#374151")
    fig.text(0.012, 0.18, "Patch progression", rotation=90, va="center",
             fontsize=10, fontweight="bold", color="#374151")

    fig.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {OUT.with_suffix('.png')}")
    print(f"wrote {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
