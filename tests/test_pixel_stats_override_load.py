"""A pixel_stats_override survives checkpoint loading (app/abstract_classes/foundation_inferencer.py _load_weights).

Before 2026-10-31 the checkpoint's PixelNormalize buffers were loaded back over the override, so HotSat-1's per-scene
DN z-score never took effect: runs with and without it were identical.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from app.abstract_classes.foundation_inferencer import FoundationInferencer
from app.foundation_models.components.pixel_normalization import PixelDenormalize, PixelNormalize
from app.models.training.inference_config import PixelStatsOverride


class _Tiny(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.normalize, self.conv, self.denormalize = PixelNormalize(mean, std), nn.Conv2d(1, 1, 1), \
            PixelDenormalize(mean, std)


class _Inf(FoundationInferencer):
    def build_model(self):
        o = self.config.pixel_stats_override
        return _Tiny(o.mean, o.std) if o is not None else _Tiny([24.58], [13.57])

    def infer(self, tensor, mask):
        return tensor


def _config(tmp_path, override):
    trained = _Tiny([24.58], [13.57])                          # Chakshu's stored Celsius stats
    torch.save({"model_state_dict": trained.state_dict(), "epoch": 1}, tmp_path / "ck.pt")
    return SimpleNamespace(device="cpu", checkpoint_path=str(tmp_path / "ck.pt"), pixel_stats_override=override,
                           foundation_model_name=SimpleNamespace(value="x"), patch_size=8), trained


def test_override_survives_loading(tmp_path):
    cfg, trained = _config(tmp_path, PixelStatsOverride(mean=[5000.0], std=[400.0], source="per_scene_dn_zscore"))
    m = _Inf(cfg).model
    assert m.normalize.mean.item() == 5000.0 and m.denormalize.std.item() == 400.0
    assert torch.equal(m.conv.weight, trained.conv.weight)    # every other weight still comes from the checkpoint


def test_without_override_the_checkpoint_stats_load(tmp_path):
    cfg, _ = _config(tmp_path, None)
    assert abs(_Inf(cfg).model.normalize.mean.item() - 24.58) < 1e-4


def test_thermal_vendable_keeps_hotsat_units():
    """Known issue 7 (fixed 2026-10-31): Pydantic dropped units=, so the DN guard never triggered for HotSat-1."""
    import numpy as np
    from app.models.dataset.vendables import VendableThermalDataset
    fields = {name: None for name, f in VendableThermalDataset.model_fields.items() if f.is_required()}
    fields.update(normalized_thermal_cube=np.zeros((2, 2)), validity_cube=np.ones((2, 2)))
    v = VendableThermalDataset(**fields, units="DN_14bit_relative")
    assert v.units.startswith("DN_")
    assert VendableThermalDataset(**fields).units is None
