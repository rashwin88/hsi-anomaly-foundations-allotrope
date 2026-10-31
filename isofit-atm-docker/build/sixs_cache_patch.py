"""AC26-6SC: patch 6S's DISCOM.f so its scattering results can be cached across water-vapour nodes.

usage: python sixs_cache_patch.py <6S source dir>
DISCOM computes everything that depends on scattering (OSPOL / KERNELPOL / ISO at the 20 reference wavelengths) and
nothing that depends on water vapour (AC26-6SP: those columns are byte-identical across H2O). With the environment
variable SIXS_DISCOM_CACHE set to a file path: if the file exists, DISCOM reads all it would have written and
returns; if not, it computes as usual and writes them (to <path>.tmp, then renamed, so a reader never sees a half
file). Without the variable 6S is unchanged. The caller (the 6S wrapper) must give points that differ only in H2O
the same path, and must run one of them first.
What is saved: /sixs_disc/ (except wldis, an input), /sixs_polar/, /aeroprof/ taer_z, and the arguments xlm1, xlm2,
roatm_fi, roluts, rolutsq, rolutsu.
Fails loudly if the source does not look as expected; a second run is a no-op.
"""
import os
import sys

MARK = "c AC26-6SC: optional cache of the scattering results"
f = os.path.join(sys.argv[1], "DISCOM.f")
t = open(f).read()
if MARK in t:
    sys.exit(print("already patched", f))
IO = ("roatm,dtdir,dtdif,utdir,utdif,sphal,trayl,traypl,rqatm,ruatm",
      "pha,qha,uha,alphal,betal,gammal,zetal",
      "taer_z",
      "xlm1,xlm2,roatm_fi,roluts,rolutsq,rolutsu")
Q = chr(39)   # a single quote, for the Fortran string literals
head = ("      character*512 cfn\n      logical cex\n"
        f"{MARK} (see sixs_cache_patch.py)\n"
        f"      call getenv({Q}SIXS_DISCOM_CACHE{Q}, cfn)\n      cex = .false.\n"
        f"      if (cfn .ne. {Q} {Q}) then\n        inquire(file=trim(cfn), exist=cex)\n        if (cex) then\n"
        f"          open(77, file=trim(cfn), form={Q}unformatted{Q}, status={Q}old{Q})\n"
        + "".join(f"          read(77) {v}\n" for v in IO) +
        "          close(77)\n          return\n        endif\n      endif\n\n")
anchor = "c     computation of all scattering parameters at wavelength"
assert t.count(anchor) == 1, "anchor not found"
t = t.replace(anchor, head + anchor)
tail_old = "   50 continue\n\n      return\n      end"
assert t.count(tail_old) == 1, "tail not found"
tail_new = ("   50 continue\n"
            f"      if (cfn .ne. {Q} {Q}) then\n"
            f"        open(77, file=trim(cfn)//{Q}.tmp{Q}, form={Q}unformatted{Q}, status={Q}replace{Q})\n"
            + "".join(f"        write(77) {v}\n" for v in IO) +
            f"        close(77)\n        call rename(trim(cfn)//{Q}.tmp{Q}, trim(cfn))\n      endif\n\n      return\n      end")
t = t.replace(tail_old, tail_new)
open(f, "w").write(t)
print("patched", f)
