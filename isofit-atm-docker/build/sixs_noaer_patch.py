"""AC26-6SX: remove 6S's aerosol-only OSPOL call from ATMREF.f.

usage: python sixs_noaer_patch.py <6S source dir>
In the polarised branch ATMREF calls OSPOL three times: Rayleigh only, aerosol only, Rayleigh + aerosol. The
aerosol-only results (roaero, rqaero, ruaero) feed only 6S's printed summary, never the rows ISOFIT parses or the
"Multipart transmittances" table (AC26-MPC: 0 rows differ). Dropping the call saves about a third of the
scattering time. The three outputs are set to 0 so the summary prints zeros instead of garbage.
Fails loudly if the source does not look as expected; a second run is a no-op.
"""
import os
import re
import sys

MARK = "c AC26-6SX: aerosol-only OSPOL call removed"
f = os.path.join(sys.argv[1], "ATMREF.f")
t = open(f).read()
if MARK in t:
    sys.exit(print("already patched", f))
i = t.index("-----aerosol reflectance")
j = t.index("-----rayleigh+aerosol", i)
seg = t[i:j]
m = re.search(r"\n[ \t]+call ospol\(iaer_prof,taer,tamol,.*?rolutd,rolutd,rolutd\)", seg, re.S)
assert m, "aerosol-only ospol call not found"
seg = seg[:m.start()] + f"\n{MARK} (outputs feed only the printed summary)" + seg[m.end():]
for v, x in (("rqaero", "xqm1"), ("ruaero", "xum1"), ("roaero", "xlm1")):
    old = f"{v}={x}(-mu,1)/xmus"
    assert seg.count(old) == 1, f"{old} not found once"
    seg = seg.replace(old, f"{v}=0.")
open(f, "w").write(t[:i] + seg + t[j:])
print("patched", f)
