"""Numbers for chapter 09-recon. Run from the repo root with the project venv:
    PYTHONPATH=. python research/textbook/tools/compute_recon.py > research/textbook/tools/compute_recon_output.txt
"""
from app.foundation_models.components.hyperspectral_seg_former_mae import HyperspectralSegFormerMAE
from app.foundation_models.components.spatial_encoder import SpatialEncoder

m = HyperspectralSegFormerMAE(compressed_channels=32, pixel_mean=[0.0] * 165, pixel_std=[1.0] * 165)
n_param = sum(p.numel() for p in m.parameters())
n_state = sum(v.numel() for v in m.state_dict().values())
print("Indradhanu v0.2 (D=32) parameters():", n_param)
print("Indradhanu v0.2 (D=32) state_dict() values:", n_state)
print("difference (buffers):", n_state - n_param)
for k, v in m.state_dict().items():
    if "running" in k or "num_batches" in k or k.endswith(".mean") or k.endswith(".std"):
        print("  buffer", k, tuple(v.shape), v.numel())
print("SpatialEncoder(1, 32, 3) parameters:", sum(p.numel() for p in SpatialEncoder(1, 32, 3).parameters()))

# The six-pixel example of section 'A one-dimensional example'.
x = [1.0, 1.1, 1.2, 5.0, 1.4, 1.5]
loo = [x[1]] + [(x[k - 1] + x[k + 1]) / 2 for k in range(1, 5)] + [x[4]]
print("leave-one-out predictions:", [round(v, 4) for v in loo])
print("residuals:", [round(a - b, 4) for a, b in zip(x, loo)])
print("self-including pred at pixel 4:", round(sum(x[2:5]) / 3, 4), "residual", round(x[3] - sum(x[2:5]) / 3, 4))
