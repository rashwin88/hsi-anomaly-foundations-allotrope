"""Numbers for the convolutional-autoencoder chapter (09a-convae.tex): Pratibimba, Antardhana, Tirohita, Asanskrita,
Drashta. CPU, no training.

Run from the repo root:
    PYTHONPATH=. C:/av/Scripts/python.exe research/textbook/tools/compute_convae.py [MODELS_DIR]

MODELS_DIR holds a copy of the stack's /models volume (copied 2026-10-31 with `docker cp allotrope_worker:/models/<arch>`
inside WSL Ubuntu, to a short path because of Windows' 260-character limit). Default:
C:/Users/FITT/AppData/Local/Temp/claude/m5.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
sys.path.insert(0, "backend")
from allotrope.foundation_models.resolver import _from_manifest
from app.foundation_models.components.normalized_masked_spatial_auto_encoder import NormalizedMaskedSpatialAutoencoder
from app.foundation_models.components.spatial_auto_encoder import SpatialAutoencoder
from app.foundation_models.components.unnormalized_spatial_auto_encoder import UnNormalizedSpatialAutoencoder
from app.foundation_models.inferencers.inferencer_factory import get_inferencer
from app.models.training.inference_config import InferenceConfig

MODELS = Path(sys.argv[1] if len(sys.argv) > 1 else "C:/Users/FITT/AppData/Local/Temp/claude/m5")
ARCHS = ["spatial_autoencoder", "spatial_masked_autoencoder", "spatial_masked_autoencoder_l1",
         "spatial_masked_autoencoder_l1_unnormalized", "normalized_masked_autoencoder"]
torch.manual_seed(0)


def n(m):
    return sum(p.numel() for p in m.parameters())


def conv(cin, cout, k=4):
    return cin * cout * k * k + cout


yy, xx = torch.meshgrid(torch.arange(128.0), torch.arange(128.0), indexing="ij")
field = 25.0 + 3.0 * torch.sin(xx / 40.0) + 2.0 * torch.cos(yy / 30.0)
tests = {"plain": field.clone()}
t = field.clone(); t[62:65, 62:65] = 45.0; tests["hot 3x3"] = t
t = field.clone(); t[61, 61] = 45.0; tests["hot 1 pixel"] = t

for arch in ARCHS:
    d = MODELS / arch
    man = json.loads((d / "current.json").read_text(encoding="utf-8"))
    rm = _from_manifest(d, d / "current.json")
    ck = torch.load(d / man["current"]["file"], map_location="cpu", weights_only=False)
    mc, sd = ck["config"]["model_config"], ck["model_state_dict"]
    print(f"===== {man['codename']['name']} ({arch}) file {man['current']['file']}")
    print("  manifest params", man["current"]["params"], "| encoder_dims", man["current"]["encoder_dims"],
          "| val_loss", man["current"]["val_loss"], "| inferencer named in manifest:", man.get("inferencer"))
    print("  checkpoint model_config", mc, "| epoch", ck["epoch"], "| val_losses",
          {k: round(v, 4) for k, v in ck["val_losses"].items()}, "| avg", round(ck["avg_val_loss"], 5))
    data = ck["config"]["data"]
    print("  train samples/epoch", data["train_samples_per_epoch"], "| epochs", data["num_epochs"],
          "| batch", data["batch_size"], "| lr", ck["config"]["learning_rate"])
    print("  resolver model_config", rm.model_config, "| fm name", rm.foundation_model_name,
          "| default", rm.default_scoring_method, rm.scoring_methods, "| patch", rm.default_patch_size,
          "stride", rm.default_stride, "| stats", "yes" if rm.pixel_stats_abs_path else "none")
    k = mc.get("kernel_size", 4)
    if mc["model_type"] == "spatial_autoencoder":
        stats = dict(pixel_mean=sd["normalize.mean"].flatten().tolist(), pixel_std=sd["normalize.std"].flatten().tolist())
        model = SpatialAutoencoder(mc["in_channels"], mc["base_channels"], mc["num_stages"], kernel_size=k, **stats)
    elif mc["model_type"] == "spatial_masked_autoencoder":
        model = UnNormalizedSpatialAutoencoder(mc["in_channels"], mc["base_channels"], mc["num_stages"], kernel_size=k)
    else:
        stats = dict(pixel_mean=sd["normalize.mean"].flatten().tolist(), pixel_std=sd["normalize.std"].flatten().tolist())
        model = NormalizedMaskedSpatialAutoencoder(mc["in_channels"], mc["base_channels"], mc["num_stages"],
                                                   kernel_size=k, **stats)
    print("  strict load:", model.load_state_dict(sd, strict=True))
    bufs = sum(b.numel() for b in model.buffers())
    print(f"  trainable {n(model):,}  buffers {bufs:,}  state_dict numel {sum(v.numel() for v in sd.values()):,}"
          f"  encoder {n(model.encoder):,}  decoder {n(model.decoder):,}")
    enc_in = mc["in_channels"] + (2 if mc["model_type"] != "spatial_autoencoder" else 0)
    ch = [enc_in] + [mc["base_channels"] * 2 ** i for i in range(mc["num_stages"])]
    hand_e = sum(conv(ch[i], ch[i + 1], k) + 2 * ch[i + 1] for i in range(mc["num_stages"]))
    dch = [mc["base_channels"] * 2 ** i for i in range(mc["num_stages"])][::-1] + [mc["in_channels"]]
    hand_d = sum(conv(dch[i], dch[i + 1], k) + (2 * dch[i + 1] if i < mc["num_stages"] - 1 else 0)
                 for i in range(mc["num_stages"]))
    print("  by hand: encoder", f"{hand_e:,}", "channels", ch, "| decoder", f"{hand_d:,}", "channels", dch,
          "| total", f"{hand_e + hand_d:,}")
    for i, st in enumerate(model.encoder.stages):
        print(f"    enc stage {i}: conv {n(st.block[0]):,} bn {n(st.block[1]):,}")
    for i, st in enumerate(model.decoder.stages):
        print(f"    dec stage {i}: convT {n(st.block[0]):,}" + (f" bn {n(st.block[1]):,}" if len(st.block) > 1 else " (final)"))
    shapes = []
    for i, st in enumerate(model.encoder.stages):
        st.register_forward_hook(lambda m, a, o, i=i: shapes.append((f"enc{i}", tuple(a[0].shape), tuple(o.shape))))
    for i, st in enumerate(model.decoder.stages):
        st.register_forward_hook(lambda m, a, o, i=i: shapes.append((f"dec{i}", tuple(a[0].shape), tuple(o.shape))))
    model.eval()
    x = field[None, None]
    m1 = torch.ones_like(x)
    with torch.no_grad():
        if mc["model_type"] == "spatial_autoencoder":
            model(x, mask=m1)
        else:
            model(x, validity_mask=m1, input_mask=m1)
    for s in shapes:
        print("    shape", s)
    icfg = InferenceConfig(foundation_model_name=rm.foundation_model_name, model_config=rm.model_config,
                           checkpoint_path=str(d / man["current"]["file"]), patch_size=128, stride=64,
                           pixel_stats_path=rm.pixel_stats_abs_path, device="cpu")
    inf = get_inferencer(icfg)
    print("  inferencer actually used:", type(inf).__name__)
    with torch.no_grad():
        for name, f in tests.items():
            rec = inf.predict(f[None, None], torch.ones(1, 1, 128, 128))[0, 0]
            r = (f - rec).abs()[8:-8, 8:-8]
            full = (f - rec).abs()
            print(f"    {name}: |res| median {r.median():.3f} max {r.max():.3f} | centre(63,63) {full[63,63]:.3f}"
                  f" | (61,61) {full[61,61]:.3f}")

print("===== normalised value of a hidden pixel in the masked normalised models")
print("  (0 - 24.5756) / 13.5744 =", (0 - 24.575641632080078) / 13.57435131072998)

print("===== exercise checks")
m = SpatialAutoencoder(1, 16, 3)
print("  SpatialAutoencoder(1, base 16, 3 stages): encoder", n(m.encoder), "decoder", n(m.decoder), "total", n(m))
tc = torch.nn.ConvTranspose1d(1, 1, 4, stride=2, padding=1, bias=False)
with torch.no_grad():
    tc.weight[:] = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
    print("  transposed conv (1,2) * (1,1,1,1):", tc(torch.tensor([[[1.0, 2.0]]])).flatten().tolist())
    tc.weight[:] = torch.tensor([[[1.0, 2.0, 2.0, 1.0]]])
    print("  transposed conv (3,-1) * (1,2,2,1):", tc(torch.tensor([[[3.0, -1.0]]])).flatten().tolist())
for k in (4, 3):
    t = torch.nn.ConvTranspose2d(1, 1, k, stride=2, padding=1)
    print(f"  ConvTranspose2d k={k} s=2 p=1 on 16:", tuple(t(torch.zeros(1, 1, 16, 16)).shape))
