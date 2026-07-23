"""Publication-grade graphic of the masking + ensemble anomaly-scoring pipeline.

Top half:    Scene segmentation (NDVI / NDWI / VNIR brightness → keep_mask)
             followed by binary erosion.
Bottom half: Generic N-model reconstruction fan-out → CDF-normalize → fuse
             (product / max / mean) → p99.5 threshold → binary anomaly map.

Output: final design/diagrams/ensemble_pipeline.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from scipy.ndimage import binary_erosion, gaussian_filter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "final design/diagrams/ensemble_pipeline"


def synth_scene(H: int = 110, W: int = 140, seed: int = 7):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    veg = np.exp(-((yy - 70) ** 2 / (2 * 25 ** 2)
                    + (xx - 35) ** 2 / (2 * 30 ** 2)))
    bright = np.exp(-((yy - 25) ** 2 / (2 * 7 ** 2)
                       + (xx - 100) ** 2 / (2 * 8 ** 2)))
    water = np.exp(-((yy - 90) ** 2 / (2 * 6 ** 2)
                      + (xx - 115) ** 2 / (2 * 16 ** 2)))
    red = 0.10 + 0.08 * (1 - veg) + 0.55 * bright + 0.02 * rng.standard_normal((H, W))
    nir = 0.12 + 0.55 * veg + 0.50 * bright + 0.02 * rng.standard_normal((H, W))
    green = 0.10 + 0.20 * (1 - veg) + 0.50 * bright + 0.30 * water + \
        0.02 * rng.standard_normal((H, W))
    red = np.clip(red, 0, 1); nir = np.clip(nir, 0, 1); green = np.clip(green, 0, 1)
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndwi = (green - nir) / (green + nir + 1e-6)
    brightness = (red + green + nir) / 3.0
    spatial_valid = np.ones((H, W), dtype=bool)
    spatial_valid[:8, :12] = False
    mask_veg = ndvi > 0.30
    mask_water = ndwi > 0.10
    mask_cloud = brightness > 0.70
    mask_shadow = brightness < 0.05
    keep_mask = spatial_valid & ~(mask_veg | mask_water | mask_cloud | mask_shadow)
    eroded = binary_erosion(keep_mask, iterations=2)
    rgb = np.dstack([np.clip(red * 1.1, 0, 1),
                     np.clip(green * 1.1, 0, 1),
                     np.clip(nir * 0.6 + green * 0.4, 0, 1)])
    return {
        "rgb": rgb, "ndvi": ndvi, "ndwi": ndwi, "brightness": brightness,
        "keep_mask": keep_mask, "eroded": eroded,
        "spatial_valid": spatial_valid,
        "mask_veg": mask_veg, "mask_water": mask_water,
        "mask_cloud": mask_cloud, "mask_shadow": mask_shadow,
    }


def synth_scores(scene, n_models: int = 4, seed: int = 11):
    rng = np.random.default_rng(seed)
    H, W = scene["rgb"].shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    anom = (1.4 * np.exp(-((yy - 55) ** 2 + (xx - 80) ** 2) / (2 * 3 ** 2))
            + 1.1 * np.exp(-((yy - 35) ** 2 + (xx - 60) ** 2) / (2 * 3 ** 2)))
    scores = []
    for i in range(n_models):
        bg = gaussian_filter(rng.standard_normal((H, W)), 4) * 0.4 + 0.15
        local = 0.10 * gaussian_filter(rng.standard_normal((H, W)), 1.5)
        scores.append((0.85 + 0.18 * i / max(n_models - 1, 1)) * anom + bg + local)
    return scores


def cdf_norm(score, mask):
    out = np.zeros_like(score, dtype=np.float32)
    vals = score[mask]
    if vals.size == 0:
        return out
    order = np.argsort(vals)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(vals))
    out[mask] = (ranks / max(len(vals) - 1, 1)).astype(np.float32)
    return out


def main() -> None:
    scene = synth_scene()
    eroded = scene["eroded"]
    km = scene["keep_mask"]

    n_models = 4
    raw_scores = synth_scores(scene, n_models=n_models)
    cdf_maps = [cdf_norm(s, eroded) for s in raw_scores]
    stack = np.stack(cdf_maps)

    fused_mean = stack.mean(axis=0) * eroded
    fused_max = stack.max(axis=0) * eroded
    fused_prod = np.prod(np.clip(stack, 1e-3, 1), axis=0) ** (1 / stack.shape[0])
    fused_prod *= eroded

    thr = np.percentile(fused_prod[eroded], 99.5)
    bin_prod = ((fused_prod >= thr) & eroded).astype(np.uint8)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
        "figure.dpi": 150,
    })

    fig = plt.figure(figsize=(20, 16))

    # ---------- title ----------
    title_gs = GridSpec(1, 1, figure=fig,
                        left=0.015, right=0.985, top=0.985, bottom=0.94)
    ax_title = fig.add_subplot(title_gs[0])
    ax_title.axis("off")
    ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1)
    ax_title.text(0.5, 0.72,
                  "Allotrope · Masking & Ensemble Anomaly Pipeline",
                  ha="center", fontsize=26, fontweight="bold",
                  color="#0f172a")
    ax_title.text(0.5, 0.20,
                  "Scene segmentation (NDVI / NDWI / brightness) → keep_mask "
                  "→ binary erosion       ·       "
                  "model fan-out → CDF-normalize → fuse → p99.5 threshold",
                  ha="center", fontsize=12.5, color="#475569")

    # ==================================================================
    # TOP HALF — masking
    # ==================================================================
    top_gs = GridSpec(
        2, 6, figure=fig,
        height_ratios=[0.20, 1.0],
        left=0.018, right=0.985, top=0.92, bottom=0.50,
        hspace=0.32, wspace=0.18,
    )
    ax_top_title = fig.add_subplot(top_gs[0, :])
    ax_top_title.axis("off")
    ax_top_title.set_xlim(0, 1); ax_top_title.set_ylim(0, 1)
    ax_top_title.text(0.012, 0.5, "①  Keep-mask construction",
                      fontsize=18, fontweight="bold", color="#0f172a",
                      va="center")
    ax_top_title.text(0.012, 0.0,
                      "spectral indices → component masks → keep_mask → erosion",
                      fontsize=11, color="#64748b", va="center", style="italic")

    def show(ax, img, title, cmap=None, vmin=None, vmax=None, sub=None,
             title_color="#0f172a"):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=4,
                     color=title_color)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if sub:
            ax.text(0.02, -0.06, sub, transform=ax.transAxes,
                    fontsize=9.5, color="#64748b")

    show(fig.add_subplot(top_gs[1, 0]), scene["rgb"], "Scene  (RGB approx)")
    show(fig.add_subplot(top_gs[1, 1]), scene["ndvi"], "NDVI",
         cmap="RdYlGn", vmin=-0.3, vmax=0.7,
         sub="(NIR − Red) / (NIR + Red)")
    show(fig.add_subplot(top_gs[1, 2]), scene["ndwi"], "NDWI",
         cmap="RdBu", vmin=-0.4, vmax=0.4,
         sub="(Green − NIR) / (Green + NIR)")
    show(fig.add_subplot(top_gs[1, 3]), scene["brightness"], "VNIR brightness",
         cmap="cividis", vmin=0, vmax=1, sub="mean VNIR reflectance")

    # component masks composite
    overlay = np.zeros((*scene["mask_veg"].shape, 3))
    overlay[scene["mask_veg"]] = [0.13, 0.55, 0.13]
    overlay[scene["mask_water"]] = [0.15, 0.40, 0.78]
    overlay[scene["mask_cloud"]] = [0.95, 0.95, 0.95]
    overlay[scene["mask_shadow"]] = [0.20, 0.20, 0.25]
    show(fig.add_subplot(top_gs[1, 4]), overlay, "Component masks",
         sub="vegetation · water · cloud · shadow")

    # keep_mask + erosion overlay
    ax_keep = fig.add_subplot(top_gs[1, 5])
    canvas = np.zeros((*km.shape, 4), dtype=np.float32)
    canvas[km] = [0.15, 0.55, 0.20, 0.65]
    canvas[eroded] = [0.10, 0.45, 0.18, 1.0]
    canvas[(km & ~eroded)] = [0.83, 0.18, 0.18, 0.85]
    canvas[~scene["spatial_valid"]] = [0.85, 0.85, 0.85, 1.0]
    ax_keep.imshow(canvas)
    ax_keep.set_title("keep_mask  ∧  erode(K=5)",
                      fontsize=12, fontweight="bold", pad=4)
    ax_keep.set_xticks([]); ax_keep.set_yticks([])
    for spine in ax_keep.spines.values():
        spine.set_visible(False)
    ax_keep.text(0.02, -0.06,
                 "dark green = scored core   ·   red = eroded boundary "
                 "·   grey = invalid",
                 transform=ax_keep.transAxes, fontsize=9.0, color="#475569")

    # ---- divider band ----
    divider_gs = GridSpec(1, 1, figure=fig,
                          left=0.018, right=0.985, top=0.495, bottom=0.475)
    ax_div = fig.add_subplot(divider_gs[0])
    ax_div.axis("off")
    ax_div.set_xlim(0, 1); ax_div.set_ylim(0, 1)
    ax_div.add_patch(FancyBboxPatch(
        (0.0, 0.30), 1.0, 0.40,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        linewidth=0, facecolor="#f1f5f9",
    ))
    ax_div.add_patch(FancyArrowPatch(
        (0.495, 0.85), (0.505, 0.15),
        arrowstyle="-|>", mutation_scale=24,
        linewidth=2.2, color="#475569",
    ))
    ax_div.text(0.5, 0.5,
                "eroded keep_mask is passed into every model's score map",
                ha="center", va="center", fontsize=11, color="#334155",
                fontweight="bold", style="italic")

    # ==================================================================
    # BOTTOM HALF — ensembling
    # ==================================================================
    bot_gs = GridSpec(
        2, 1, figure=fig,
        height_ratios=[0.10, 1.0],
        left=0.018, right=0.985, top=0.465, bottom=0.02,
        hspace=0.06,
    )
    ax_bot_title = fig.add_subplot(bot_gs[0])
    ax_bot_title.axis("off")
    ax_bot_title.set_xlim(0, 1); ax_bot_title.set_ylim(0, 1)
    ax_bot_title.text(0.012, 0.65, "②  Ensemble anomaly scoring",
                      fontsize=18, fontweight="bold", color="#0f172a",
                      va="center")
    ax_bot_title.text(0.012, 0.15,
                      "N reconstruction models in parallel → CDF normalize → "
                      "fuse via product / max / mean → threshold at p99.5",
                      fontsize=11, color="#64748b", va="center", style="italic")

    # 3 columns: per-model fan-out · fusion math + fused maps · binary output
    bot_body = GridSpec(
        2, 3, figure=fig,
        width_ratios=[0.85, 1.0, 0.85],
        height_ratios=[1.0, 0.18],
        left=0.018, right=0.985, top=0.42, bottom=0.02,
        wspace=0.10, hspace=0.06,
    )

    # ----- Column 1: per-model score maps (generic, no names) -----
    col1_gs = GridSpec(
        n_models, 2, figure=fig,
        width_ratios=[0.18, 1.0],
        left=0.018, right=0.290,
        top=0.42, bottom=0.10,
        hspace=0.20, wspace=0.05,
    )
    for i in range(n_models):
        ax_av = fig.add_subplot(col1_gs[i, 0])
        ax_av.axis("off")
        ax_av.set_xlim(0, 1); ax_av.set_ylim(0, 1)
        col = "#475569"
        ax_av.add_patch(Circle((0.5, 0.5), 0.34,
                               facecolor=col, edgecolor="white",
                               linewidth=2.0, zorder=3))
        ax_av.text(0.5, 0.51, f"M{i + 1}",
                   ha="center", va="center", fontsize=14,
                   fontweight="bold", color="white", zorder=4)

        ax_sc = fig.add_subplot(col1_gs[i, 1])
        cdf_img = cdf_maps[i].copy()
        cdf_img[~eroded] = np.nan
        ax_sc.imshow(cdf_img, cmap="magma", vmin=0, vmax=1)
        ax_sc.set_xticks([]); ax_sc.set_yticks([])
        for spine in ax_sc.spines.values():
            spine.set_visible(False)
        ax_sc.set_title(f"model {i + 1}  ·  |x − x̂|  ·  CDF-rank",
                        fontsize=10.5, color="#334155", pad=3, loc="left")

    # caption strip under col 1
    ax_cap = fig.add_subplot(bot_body[1, 0])
    ax_cap.axis("off"); ax_cap.set_xlim(0, 1); ax_cap.set_ylim(0, 1)
    ax_cap.text(0.5, 0.5,
                "each model: forward pass → reconstruction error → mask "
                "with eroded keep_mask → CDF-rank to [0, 1]",
                ha="center", fontsize=10, color="#475569", style="italic")

    # ----- Column 2: fusion math + fused maps -----
    col2_gs = GridSpec(
        4, 1, figure=fig,
        height_ratios=[0.32, 1, 1, 1],
        left=0.310, right=0.665,
        top=0.42, bottom=0.10,
        hspace=0.28,
    )
    ax_math = fig.add_subplot(col2_gs[0])
    ax_math.axis("off"); ax_math.set_xlim(0, 1); ax_math.set_ylim(0, 1)
    ax_math.add_patch(FancyBboxPatch(
        (0.0, 0.05), 1.0, 0.9,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0, edgecolor="#cbd5e1", facecolor="#f8fafc",
    ))
    ax_math.text(0.5, 0.78, "Fusion strategies",
                 ha="center", fontsize=12.5, fontweight="bold",
                 color="#0f172a")
    ax_math.text(
        0.5, 0.36,
        r"Product:  $S_{\mathrm{prod}} = \left(\prod_m s_m\right)^{1/M}$"
        r"        Mean:  $S_{\mathrm{mean}} = \frac{1}{M}\sum_m s_m$"
        r"        Max:  $S_{\mathrm{max}} = \max_m s_m$",
        ha="center", fontsize=11, color="#0f172a",
    )

    def show_fused(ax, img, title, anno):
        plot = img.copy()
        plot[~eroded] = np.nan
        ax.imshow(plot, cmap="magma", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.02, -0.07, anno, transform=ax.transAxes,
                fontsize=9.5, color="#64748b")

    show_fused(fig.add_subplot(col2_gs[1]), fused_prod,
               "Product fusion",
               "emphasizes pixels flagged by every model")
    show_fused(fig.add_subplot(col2_gs[2]), fused_max,
               "Max fusion",
               "emphasizes any single strong signal")
    show_fused(fig.add_subplot(col2_gs[3]), fused_mean,
               "Mean fusion",
               "averaged consensus")

    # ----- Column 3: thresholding + outputs -----
    col3_gs = GridSpec(
        3, 1, figure=fig,
        height_ratios=[0.32, 1.0, 0.62],
        left=0.685, right=0.985,
        top=0.42, bottom=0.10,
        hspace=0.30,
    )
    ax_thr = fig.add_subplot(col3_gs[0])
    ax_thr.axis("off"); ax_thr.set_xlim(0, 1); ax_thr.set_ylim(0, 1)
    ax_thr.add_patch(FancyBboxPatch(
        (0.0, 0.05), 1.0, 0.9,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0, edgecolor="#cbd5e1", facecolor="#f8fafc",
    ))
    ax_thr.text(0.5, 0.74, "Thresholding",
                ha="center", fontsize=12.5, fontweight="bold", color="#0f172a")
    ax_thr.text(0.5, 0.32,
                r"anomaly = $\{p : S(p) \geq \mathrm{percentile}_{99.5}(S \mid M_{\mathrm{eroded}})\}$",
                ha="center", fontsize=11, color="#0f172a")

    ax_bin = fig.add_subplot(col3_gs[1])
    canvas = np.zeros((*bin_prod.shape, 4), dtype=np.float32)
    canvas[eroded] = [0.94, 0.94, 0.94, 1.0]
    canvas[~eroded] = [0.78, 0.78, 0.78, 1.0]
    canvas[bin_prod == 1] = [0.83, 0.18, 0.18, 1.0]
    ax_bin.imshow(canvas)
    ax_bin.set_title("Binary anomaly map",
                     fontsize=13, fontweight="bold", pad=4)
    ax_bin.set_xticks([]); ax_bin.set_yticks([])
    for spine in ax_bin.spines.values():
        spine.set_visible(False)
    n_det = int(bin_prod.sum())
    n_kept = int(eroded.sum())
    ax_bin.text(0.02, -0.07,
                f"{n_det} flagged of {n_kept:,} scored pixels  "
                f"·  strategy: product fusion",
                transform=ax_bin.transAxes, fontsize=9.5, color="#64748b")

    ax_out = fig.add_subplot(col3_gs[2])
    ax_out.axis("off"); ax_out.set_xlim(0, 1); ax_out.set_ylim(0, 1)
    ax_out.add_patch(FancyBboxPatch(
        (0.0, 0.05), 1.0, 0.9,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0, edgecolor="#7c3aed", facecolor="#faf5ff",
    ))
    ax_out.text(0.5, 0.83, "Outputs",
                ha="center", fontsize=12, fontweight="bold", color="#5b21b6")
    items = [
        "anomaly_product.tif   binary, 1 = anomaly",
        "anomaly_maximum.tif   binary  (max-fusion)",
        "anomaly_mean.tif      binary  (mean-fusion)",
        "stats.json   thresholds, pixel counts",
    ]
    y = 0.66
    for it in items:
        ax_out.text(0.05, y, "•", fontsize=12, fontweight="bold",
                    color="#7c3aed")
        ax_out.text(0.09, y, it, fontsize=10, color="#1e293b",
                    va="center", family="DejaVu Sans Mono")
        y -= 0.15

    fig.savefig(OUT.with_suffix(".png"), dpi=240, bbox_inches="tight",
                facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight",
                facecolor="white")
    print(f"wrote {OUT.with_suffix('.png')}")
    print(f"wrote {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
