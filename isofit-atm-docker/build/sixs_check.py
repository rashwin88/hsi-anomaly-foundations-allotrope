"""Build-time check: the fast 6S builds must reproduce stock 6S on the rows anything downstream reads.

usage: python sixs_check.py <stock sixsV2.1> <out dir of build_sixs.sh> <fixtures dir>
The fixtures are one group of ISOFIT 6S inputs that differ only in water vapour (EnMAP LUT, AOT 0.39, 0.5 km).
Compared per fixture against the stock binary: the rows ISOFIT parses ("* 0.4000 ...") and the "Multipart
transmittances" table src/sixs_parts.py reads. Each build runs twice, without the cache and with it (first
fixture fills the cache, the others read it); the two runs must agree exactly, which is the cache's own test.
  check   (-O, ISOFIT's flags)  must equal stock exactly: the patches change nothing.
  v3, v4  (-O3)                 may differ in the last printed digit only (AC26-MPC: max 1e-4).
A build the CPU cannot run (v4 without AVX-512) is skipped with a note; the wrapper never picks it there.
Exits non-zero on any failure, which fails the image build.
"""
import os
import re
import subprocess
import sys
import tempfile

ROW = re.compile(r"^\* ?[0-9]\.[0-9]{3,4} ")
NUM = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?|NaN")
TOL = 2e-4
V4_FLAGS = {"avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"}


def rows(text):
    out, mp = [l for l in text.splitlines() if ROW.match(l)], []
    on = False
    for line in text.splitlines():
        if "Multipart transmittances" in line:
            on = True
        elif on and re.search(r"\d", line):
            mp.append(line.rstrip())
        elif on and mp:
            break
    assert out and mp, "6S output has no ISOFIT rows or no Multipart table"
    return out + mp


def worst(a, b):
    """Largest numeric difference between two row lists (inf if their shape differs)."""
    if len(a) != len(b):
        return float("inf")
    mx = 0.0
    for x, y in zip(a, b):
        u, v = NUM.findall(x), NUM.findall(y)
        if len(u) != len(v) or [s == "NaN" for s in u] != [s == "NaN" for s in v]:
            return float("inf")
        mx = max([mx] + [abs(float(p) - float(q)) for p, q in zip(u, v) if "NaN" not in (p, q)])
    return mx


def run(exe, inp, cache=None):
    env = dict(os.environ)
    env.pop("SIXS_DISCOM_CACHE", None)
    if cache:
        env["SIXS_DISCOM_CACHE"] = cache
    with open(inp) as i:
        r = subprocess.run([exe], stdin=i, capture_output=True, text=True, env=env, timeout=600)
    return rows(r.stdout)


def main():
    stock, out, fx = sys.argv[1:4]
    inps = sorted(os.path.join(fx, f) for f in os.listdir(fx) if f.endswith(".inp"))
    flags = set(open("/proc/cpuinfo").read().split())
    ref = [run(stock, p) for p in inps]
    bad = 0
    for b in ("check", "v3", "v4"):
        exe = f"{out}/{b}/sixsV2.1"
        if not os.path.exists(exe):
            print(f"sixs_check {b}: not installed (the -O check build is deleted after the image build)")
            continue
        if b == "v4" and not V4_FLAGS <= flags:
            print(f"sixs_check {b}: SKIPPED, this CPU has no AVX-512 (checked again by the wrapper's selftest)")
            continue
        with tempfile.TemporaryDirectory() as td:
            plain = [run(exe, p) for p in inps]
            cached = [run(exe, p, f"{td}/discom") for p in inps]
            filled = os.path.exists(f"{td}/discom")
        d_ref = max(worst(r, p) for r, p in zip(ref, plain))
        same_cache = all(c == p for c, p in zip(cached, plain))
        ok = filled and same_cache and (d_ref == 0.0 if b == "check" else d_ref <= TOL)
        bad += not ok
        print(f"sixs_check {b}: {'OK' if ok else 'FAIL'}  max |diff| vs stock {d_ref:.1e}  "
              f"cache written {filled}  cached == uncached {same_cache}  ({len(inps)} fixtures, {len(ref[0])} rows each)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
