"""H2 add-on: a scale-invariant physical baseline - NNLS in reflectance (no sum-to-one), abundances then normalised
to sum 1 - with the same GT / VCA endmembers as h2_unmixing.py. Merged into results/h2.json as space "refl_nnls_norm"."""
import json, sys
import numpy as np
from scipy.optimize import linear_sum_assignment, nnls
import common as c
from h2_unmixing import scores

sys.path.insert(0, c.UPSTREAM)
from engine.VCA import vca

res = json.load(open("results/h2.json"))
for name in c.H_DATASETS:
    s = c.load(name)
    K = len(s.E)
    np.random.seed(0)
    Ev = np.clip(vca(s.Y.T, K, verbose=False)[0].T, 0, None)
    sad = np.degrees(np.arccos(np.clip(c.unit(Ev) @ c.unit(s.E).T, -1, 1)))
    r, cc = linear_sum_assignment(sad)
    Ev_al = np.empty_like(Ev); Ev_al[cc] = Ev[r]
    for label, E in (("gt", s.E), ("vca", Ev_al)):
        A = np.stack([nnls(E.T, y)[0] for y in s.Y], 1)
        A = A / np.maximum(A.sum(0, keepdims=True), 1e-12)
        res[name][label]["refl_nnls_norm"] = scores(A, s.A)
    print(name, res[name]["gt"]["refl_nnls_norm"], res[name]["vca"]["refl_nnls_norm"], flush=True)
json.dump(res, open("results/h2.json", "w"), indent=1)
