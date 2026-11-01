# Build-PC agent brief: build, test and ship isofit-atm-docker

You are a Claude Code agent on the connected build PC, a Windows 11 box with WSL2: Core Ultra 9 285 (no AVX-512),
RTX 4000 Ada. Your job is to build this Docker image, test it, scan it, run it once on a scene with the network
off (the "mini fake VM"), and report back. You do not change pipeline code. If something fails, stop and report it
with the exact error; do not work around it.

## What this is

`isofit-atm-docker/` is a self-contained atmospheric-correction stack. It turns L1 radiance from EnMAP, PRISMA,
AVIRIS-NG, Tanager or GHOSt into L2 surface reflectance, using ISOFIT 4.1.5 + sRTMnet. The image runs on an
air-gapped H200 VM (NVIDIA driver 580.173, CUDA 13.0), so everything it needs is baked in at build time.
`README.md` describes the layout.

Do not touch:
- `docker/`: the product stack.
- `enmap-isofit-docker/`: the predecessor.
- Any other folder in the repo.

Do not commit or push unless the user asks you to.

## Prerequisites (check first, report what is missing)

```bash
docker version && docker buildx version          # Docker Desktop with the WSL2 backend, BuildKit on
df -h .                                          # >= 60 GB free where Docker stores images
nvidia-smi                                       # host driver visible in WSL2
docker run --rm --gpus all ubuntu:24.04 nvidia-smi    # GPU visible to containers (nvidia-container-toolkit)
command -v trivy pigz || true                    # optional: trivy for the scan, pigz for fast compression
```

If the GPU is not visible to containers, carry on. The image also runs CPU-only. Report the GPU problem.

## 1. Get the code

```bash
git fetch origin && git checkout feature/isofit-atm-docker && git pull
cd isofit-atm-docker
```

## 2. Build, test, scan, export: one script

```bash
time ./scripts/build_image.sh 2>&1 | tee build.log
```

What the script does, in order:
1. `docker build --target test`: builds the image (downloads about 10 GB: Ubuntu packages, pip wheels including
   CUDA torch, ISOFIT data, the 5.5 GB sRTMnet emulator). It compiles 6S and checks the fast builds against the
   stock build, then runs the unit tests on top of the exact runtime image. A failing test or coverage under
   30 % fails the build. Expect `81 passed` and about 54 % coverage.
2. `docker build --target runtime -t isofit-atm-docker:1.3.0`, reusing the layers from step 1.
3. Smoke test as a non-root uid: the 6S selftest and the imports.
4. trivy, if installed: CRITICAL table in `dist/trivy-critical-1.3.0.txt`, CRITICAL+HIGH JSON with a count per
   package type. Target: fewer than 100 CRITICAL.
5. `docker save` + pigz into `dist/isofit-atm-docker-1.3.0.tar.gz`, then `.sha256` and `dist/aux-manifest.txt`.

Expected time: 45 to 75 min for a clean build on a ~100 Mbps link, mostly downloads. Image size: about 13 to
15 GB uncompressed, 9 to 10 GB compressed.

Known, harmless messages:
- `sixs_check v4: SKIPPED, this CPU has no AVX-512`. The build PC lacks AVX-512; the VM checks v4 itself.
- Pip resolver notices about torch's nvidia-* wheels.

Things that must not appear:
- `sixs_check ... FAIL`.
- `missing library for /opt/sixs/...`.
- An `AssertionError` from the `apply_oe uses fast_al` check.
- Any test failure.

If any of these appear, stop and report.

## 3. GPU check (build PC)

```bash
docker run --rm --user 1000:1000 --gpus all --entrypoint python isofit-atm-docker:1.3.0 \
  -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `2.14.1+cu130 True NVIDIA RTX 4000 Ada ...`. cu130 wheels need host driver >= 580. If the build PC's driver
is older, `False` here is not a blocker: the VM has 580.173. Report the driver version.

## 4. Mini fake VM: the tarball, offline

This checks what the air-gapped VM will do. The image is loaded from the tarball, not from the build cache, and
the scene runs with no network.

```bash
docker rmi isofit-atm-docker:1.3.0                       # make sure the tarball is what gets tested
./scripts/deploy_airgap.sh dist/isofit-atm-docker-1.3.0.tar.gz dist/isofit-atm-docker-1.3.0.sha256

# One scene with the network disabled. Use the PRISMA trial scene the user provides, e.g.
#   <scenes>/prisma/input/PRS_L1_STD_OFFL_20260514055637_20260514055641_0001.he5
#   <scenes>/prisma/reference/PRS_L2C_STD_20260514055637_20260514055641_0001.he5
#   <scenes>/dem/   (cop30_mosaic.vrt + tiles)
S=<scenes>                                                # ask the user for this path
docker run --rm --network none --user "$(id -u):$(id -g)" --shm-size 16g --gpus all \
  -v $S/prisma/input:/data/in:ro -v $S/prisma/reference:/data/ref:ro -v $S/dem:/aux/dem:ro \
  -v $PWD/fakevm_out:/data/out isofit-atm-docker:1.3.0 \
  /data/in/PRS_L1_STD_OFFL_20260514055637_20260514055641_0001.he5 /data/out \
  --ref /data/ref/PRS_L2C_STD_20260514055637_20260514055641_0001.he5 2>&1 | tee fakevm.log
```

(`scripts/run_scene.sh` does the same with `DEM_DIR=` and `REF=`. The explicit form above adds `--network none`.)

Expected:
- **Run:** about 30 to 40 min, ending with `DONE. L2A products in: /data/out`.
- **Products** in `fakevm_out/`:
  - `PRS_20260514055637_L2_SR.tif`, `.bsq`, `.hdr`;
  - `_latlon.tif`, `_outlier.tif`, `_snow.tif`;
  - `_provenance.json`, `_report.html`.
- **Report section 5:** MAE around 0.0145 against L2C over all bands. The Colab trial on 2026-10-31 gave 0.0145.
  The land-only research scorer gave 0.0097 raw / 0.0057 harmonised.

A network access attempt in the log means something was not baked in. Report it.

## 5. Report back to the user

Send one message with:
1. Prerequisites: what was present, what was missing.
2. Build: wall time, `docker images isofit-atm-docker:1.3.0` size, tarball size, test summary line
   (`81 passed`, coverage %).
3. trivy: the per-type count lines and `CRITICAL total`. If over 100, the top 10 CRITICAL packages with their
   installed and fixed versions.
4. GPU check output.
5. Fake VM: run time, the tail of `fakevm.log`, the list of files in `fakevm_out/`, and report section 5's MAE,
   bias and pixels compared.
6. Every WARNING or ERROR line you did not expect from this brief.

Then hand the user the three files for the VM:
- `dist/isofit-atm-docker-1.3.0.tar.gz`;
- `.sha256`;
- `aux-manifest.txt`.

## Reference

- **Design and evidence:** `research/experiments/atm-correction/EXPERIMENT_LOG.md`, entries AC26-*.
  Item 10 is AC26-D10 / D10b; the trials are AC26-T11.
- **Switches** (environment, at run time):
  - `SIXS_FAST=0`: stock 6S;
  - `SIXS_CACHE=0`: no scattering cache;
  - `SIXS_ARCH=v4`: the AVX-512 6S build;
  - `ATM_FAST_AL=0`: ISOFIT's own per-pixel solve.
- **Run without the GPU:** `GPUS=none` with `run_scene.sh`, or drop `--gpus all`.
