"""
Investigate extreme RX score outliers in the 20231229 scene.
Checks where they are spatially and what their spectral signature looks like.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
logging.basicConfig(level=logging.WARNING)

from app.models.file_processing.sources import FileSourceConfig
from app.utils.dataset_builder.prisma_dataset_builder import PrismaDatasetBuilder
from app.utils.data_transformations.composite_destriper import CombinedDestriper
from app.detectors.global_rx_detector import GlobalRXDetector

HE5 = (
    PROJECT_ROOT
    / "tests/test_payloads/phase_2/Set-4/Hyper"
    / "PRS_L2D_STD_20231229050902_20231229050907_0001.he5"
)
OUTPUT_DIR = Path(__file__).resolve().parent

vendable = PrismaDatasetBuilder(
    file_source_configuration=FileSourceConfig(source_path=str(HE5))
).vend_dataset()

cube        = vendable.normalized_hyperspectral_cube
validity    = vendable.validity_cube
wavelengths = np.array(vendable.band_cw_order)

# Destripe
destriper = CombinedDestriper()
destriped = destriper.transform(
    cube, validity_mask=validity, diagnostics=False,
    band_wavelengths=vendable.band_cw_order,
)

# RX
detector = GlobalRXDetector(vendable)
detector.fit(band_failure_threshold=0.05)
score_map = detector.detect(destriped, validity)
result    = detector.result

valid_scores = score_map[~np.isnan(score_map)]
p999 = float(np.percentile(valid_scores, 99.9))
p99  = float(np.percentile(valid_scores, 99))
p98  = float(np.percentile(valid_scores, 98))

print(f"p98={p98:.1f}  p99={p99:.1f}  p99.9={p999:.1f}  max={valid_scores.max():.1f}")

# Find extreme outlier pixels (above p99.9)
outlier_mask = score_map > p999
rows, cols = np.where(outlier_mask)
print(f"\nOutlier pixels (>{p999:.0f}): {len(rows)}")

# Print location and score for top-20
order = np.argsort(score_map[rows, cols])[::-1][:20]
print(f"{'Rank':>4}  {'Row':>5}  {'Col':>5}  {'Score':>10}  {'In-swath':>8}")
spatial_mask = result.spatial_mask
for rank, k in enumerate(order):
    r, c = rows[k], cols[k]
    in_swath = "YES" if spatial_mask[r, c] else "NO"
    print(f"{rank+1:>4}  {r:>5}  {c:>5}  {score_map[r,c]:>10.1f}  {in_swath:>8}")

# Spectral signature of the most extreme pixel vs scene mean
r0, c0 = rows[order[0]], cols[order[0]]
good_bands = result.good_band_indices
good_wl    = np.array(result.good_band_wavelengths)
px_spectrum = destriped[good_bands, r0, c0]
scene_mean  = np.array([
    destriped[b][validity[b].astype(bool)].mean() for b in good_bands
])
print(f"\nMost extreme pixel: ({r0}, {c0})  score={score_map[r0,c0]:.1f}")
print(f"Pixel spectrum range: {px_spectrum.min():.4f} – {px_spectrum.max():.4f}")
print(f"Scene mean   range : {scene_mean.min():.4f} – {scene_mean.max():.4f}")

# Check if spectrum has any non-finite or negative values
print(f"Any non-finite in pixel spectrum: {not np.all(np.isfinite(px_spectrum))}")
print(f"Any negative in pixel spectrum  : {np.any(px_spectrum < 0)}")
print(f"Any >1.0 in pixel spectrum      : {np.any(px_spectrum > 1.0)}")

# Plot: score map clipped at p99 with outliers highlighted, + spectrum
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Outlier Analysis — 20231229", fontsize=10)

cmap = plt.cm.inferno.copy()
cmap.set_bad("lightgray")
axes[0].imshow(
    np.where(spatial_mask, score_map, np.nan),
    cmap=cmap, vmin=p98, vmax=p99, interpolation="nearest",
)
axes[0].set_title(f"RX score (clipped p98–p99)\n{len(rows)} outliers above p99.9")
axes[0].axis("off")

# Highlight outliers on blank canvas
ol_img = np.zeros(score_map.shape)
ol_img[outlier_mask] = 1.0
axes[1].imshow(ol_img, cmap="hot", interpolation="nearest")
axes[1].set_title(f"Outlier locations (>{p999:.0f})")
axes[1].axis("off")

# Spectrum of extreme pixel vs scene mean
axes[2].plot(good_wl, scene_mean, label="scene mean", color="steelblue", lw=1)
axes[2].plot(good_wl, px_spectrum, label=f"outlier ({r0},{c0})", color="red", lw=1)
axes[2].set_xlabel("Wavelength (nm)")
axes[2].set_ylabel("Reflectance")
axes[2].set_title("Spectrum: outlier vs scene mean")
axes[2].legend(fontsize=8)

fig.tight_layout()
out = OUTPUT_DIR / "outlier_analysis.png"
fig.savefig(str(out), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out.name}")
