"""Build-time patches to the installed ISOFIT 4.1.5. Each one replaces an exact anchor that must occur once, so a
future ISOFIT bump fails the image build loudly instead of shipping silently unpatched.

usage: python isofit_patches.py
  shards  luts/stores/zarr.py calc_shards: keep every water-vapour column inside one shard when such a layout
          exists. A shard runs its points in sequence on one Ray worker, and the H2O nodes of a group then follow
          their first node there, so the 6S scattering cache (sixs_wrapper.sh) is filled once and read by the
          rest. Left alone, calc_shards picks the layout whose file count is closest to n_cores: 12 cores give
          one shard per AOT node (H2O whole), 64 cores give 66 shards with H2O split three ways, and the three
          siblings then run at the same time and none of them reads the cache. Grids with H2O as the only
          dimension (the H2O presolve) are left as they are, so their points still run in parallel.
  fast_al utils/apply_oe.py: take analytical_line from src/fast_al.py (batched per-pixel solve, AC26-BSC), which
          runs ISOFIT's own function with a faster worker and falls back to ISOFIT's code by itself (ATM_FAST_AL=0
          turns it off). If fast_al is not importable (no /opt/pipeline/src on PYTHONPATH), apply_oe imports
          ISOFIT's own as before.
"""
import os

import isofit

ROOT = os.path.dirname(isofit.__file__)
PATCHES = {
    "shards": ("luts/stores/zarr.py",
               "    shards = shards[viable]\n    cpf = cpf[viable]\n",
               "    shards = shards[viable]\n    cpf = cpf[viable]\n"
               "    # isofit-atm-docker (isofit_patches.py): keep each H2O column in one shard\n"
               "    names = list(grid)\n"
               "    if \"H2OSTR\" in names and len(names) > 1:\n"
               "        whole = shards[:, 1 + names.index(\"H2OSTR\")] == shape[1 + names.index(\"H2OSTR\")]\n"
               "        if whole.any():\n"
               "            shards, cpf = shards[whole], cpf[whole]\n"),
    "fast_al": ("utils/apply_oe.py",
                "from isofit.utils import analytical_line as ALAlg\n",
                "try:  # isofit-atm-docker (isofit_patches.py): batched per-pixel solve, ATM_FAST_AL=0 for ISOFIT's\n"
                "    from fast_al import analytical_line as ALAlg\n"
                "except ImportError:\n"
                "    from isofit.utils import analytical_line as ALAlg\n"),
}

for name, (rel, old, new) in PATCHES.items():
    f = os.path.join(ROOT, rel)
    t = open(f).read()
    assert t.count(old) == 1, f"{name}: anchor not found once in {f}"
    open(f, "w").write(t.replace(old, new))
    print(f"patched {name}: {f}")
