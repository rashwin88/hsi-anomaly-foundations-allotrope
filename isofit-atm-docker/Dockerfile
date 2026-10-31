# =============================================================================
# isofit-atm-docker  --  L1 -> L2 surface reflectance (ISOFIT 4.1.5 + sRTMnet)
# for EnMAP, PRISMA, AVIRIS-NG, Tanager, GHOSt and unknown sensors.  v1.3.0
#
# A stack of its own: nothing here is shared with the product's docker/ stack.
# Python is Ubuntu 24.04's own 3.12 with a venv from requirements.txt (pip only,
# no conda).
#
# Two stages:
#   builder  compilers + the venv, ISOFIT's 6S and the fast 6S builds with their
#            checks, sRTMnet / ISOFIT data / examples, the surface prior, the
#            ISOFIT patches.
#   runtime  a fresh Ubuntu 24.04 with python3.12 and the 6S runtime libraries;
#            the finished /opt/venv, /root/.isofit, /opt/sixs, /opt/aux and the
#            pipeline are copied in. No gcc / gfortran / make / git / curl /
#            kernel headers ship. The checks that matter at run time (6S
#            selftest, fast_al wired into apply_oe, every 6S build's libraries
#            resolvable, the pipeline imports) run again here, so anything the
#            copy misses fails the build, not a scene on the VM.
#
# Build:   scripts/build_image.sh   (docker build -t isofit-atm-docker:1.3.0 .)
# Run:     scripts/run_scene.sh     (GPU, shm size, mounts) or docker compose
# =============================================================================

ARG UBUNTU=ubuntu:24.04

# ----------------------------------------------------------------------------- builder
FROM ${UBUNTU} AS builder

ARG ISOFIT_VERSION=4.1.5
# cu130 needs NVIDIA driver >= 580 (VM: 580.173, CUDA 13.0). The wheels carry
# their own CUDA runtime: no CUDA base image, only the host driver
# (nvidia-container-toolkit, docker run --gpus).
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG DEBIAN_FRONTEND=noninteractive

# gfortran + make compile 6S; curl / unzip / git serve ISOFIT's downloads.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        build-essential gfortran git curl ca-certificates unzip libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# -- Python venv from requirements.txt ----------------------------------------
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url "${TORCH_INDEX}" -r /tmp/requirements.txt && \
    python -c "import isofit; assert isofit.__version__ == '${ISOFIT_VERSION}', isofit.__version__"

# -- 6SV2.1 (sRTMnet's preSim runs real 6S) -----------------------------------
# `isofit download sixs` patches the Makefile (-std=legacy) and runs make; it
# swallows make failures, hence the ls guard.
RUN isofit download sixs --path /root/.isofit/sixs --overwrite --debug-make && \
    ls -l /root/.isofit/sixs/sixsV* && \
    python -c "from isofit.data.cli.sixs import get_exe; print('6S exe:', get_exe())"

# -- Fast 6S builds (AC26-6SX, AC26-6SC) --------------------------------------
# Both patches (aerosol-only call removed; scattering cache across H2O) at -O3,
# for x86-64-v3 and v4 into /opt/sixs/{v3,v4}. sixs_check.py fails the build
# unless the -O patched build equals stock exactly and v3/v4 agree to the last
# printed digit (v4 only on an AVX-512 build host; `sixsV2.1 --selftest` covers
# it on the VM). Stock moves to /opt/sixs/stock; sixs_wrapper.sh takes its name,
# so ISOFIT calls the wrapper (fast binary + H2O cache; SIXS_FAST=0 = stock).
# isofit_patches.py: each H2O column in one LUT shard; apply_oe uses fast_al.
COPY build/ /opt/build/
RUN bash /opt/build/build_sixs.sh /root/.isofit/sixs /opt/sixs && \
    python /opt/build/sixs_check.py /root/.isofit/sixs/sixsV2.1 /opt/sixs /opt/build/sixs_fixtures && \
    rm -rf /opt/sixs/check && mkdir -p /opt/sixs/stock && \
    mv /root/.isofit/sixs/sixsV2.1 /opt/sixs/stock/sixsV2.1 && \
    install -m 755 /opt/build/sixs_wrapper.sh /root/.isofit/sixs/sixsV2.1 && \
    python /opt/build/isofit_patches.py && \
    /root/.isofit/sixs/sixsV2.1 --selftest && \
    find /root/.isofit/sixs -name "*.o" -delete

# -- sRTMnet emulator ----------------------------------------------------------
RUN isofit download srtmnet --path /root/.isofit/srtmnet && ls -lh /root/.isofit/srtmnet/

# -- ISOFIT data + examples (prism_optimized_irr.dat) + default surface prior ---
COPY configs/surface_multicomponent.json /tmp/surface_config.json
COPY configs/enmap_wavelengths.txt /opt/aux/enmap_wavelengths.txt
RUN isofit download data     --path /root/.isofit/data && \
    isofit download examples --path /root/.isofit/examples && \
    ln -sf /root/.isofit/data /opt/aux/isofit-data && \
    mkdir -p /opt/aux/surface && \
    isofit surface_model /tmp/surface_config.json \
        --wavelength_path /opt/aux/enmap_wavelengths.txt \
        --output_path /opt/aux/surface/multicomponent_surface.mat \
        --seed 42 && \
    ls -lh /root/.isofit/data /root/.isofit/examples /opt/aux/surface/

# -- empirical_line output band count (upstream bug, 3.6.0 through 4.1.5) -----
# initialize_output() is handed the LOCATION cube's band count (nlb, always 3)
# while output_metadata carries the radiance bands: "could not broadcast (L,3,S)
# into (L,224,S)". It breaks --empirical_line for every sensor. The greps make a
# future ISOFIT bump fail the build loudly instead of shipping an unpatched module.
ARG EL=/opt/venv/lib/python3.12/site-packages/isofit/utils/empirical_line.py
RUN grep -qc "(nll, nlb, nls)" "$EL" && \
    sed -i "s/\(output_reflectance_file, (nll, \)nlb\(, nls)\)/\1n_input_bands\2/; \
            s/\(output_uncertainty_file, (nll, \)nlb\(, nls)\)/\1n_input_bands\2/" "$EL" && \
    [ "$(grep -c 'n_input_bands, nls' "$EL")" = "2" ] && \
    python -c "import isofit.utils.empirical_line" && \
    echo "patched empirical_line: nlb -> n_input_bands (2 sites)"

# Not needed at run time: bytecode and caches.
RUN find /opt/venv -name "__pycache__" -type d -prune -exec rm -rf {} + && rm -rf /root/.cache

# ----------------------------------------------------------------------------- runtime
FROM ${UBUNTU} AS runtime

LABEL org.opencontainers.image.title="isofit-atm-docker: L1 to L2 surface reflectance via ISOFIT/sRTMnet"
LABEL org.opencontainers.image.version="1.3.0"
LABEL org.opencontainers.image.source="internal"
ENV PIPELINE_VERSION=1.3.0
ARG DEBIAN_FRONTEND=noninteractive

# python3.12 for the venv; libgfortran5 (pulls libquadmath0) and libgomp1 for
# the 6S binaries. upgrade: the base's own security fixes since its tag.
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends python3.12 libgfortran5 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv     /opt/venv
COPY --from=builder /root/.isofit /root/.isofit
COPY --from=builder /opt/sixs     /opt/sixs
COPY --from=builder /opt/aux      /opt/aux
COPY --from=builder /opt/build    /opt/build

# -- Pipeline code ------------------------------------------------------------
COPY src/       /opt/pipeline/src/
COPY configs/   /opt/pipeline/configs/
# harmonisation package (python -m harmonise); its frozen models are configs/harmonise/*.npz
COPY harmonise/ /opt/pipeline/harmonise/
COPY entrypoint.sh /opt/pipeline/entrypoint.sh
RUN chmod +x /opt/pipeline/entrypoint.sh

# -- Runtime env --------------------------------------------------------------
# HOME=/root for every user: ISOFIT finds its data through ~/.isofit/isofit.ini.
# scripts/run_scene.sh runs the container as the caller's uid, so /root is made
# readable (never writable); caches that would go under HOME go to /tmp.
ENV PATH=/opt/venv/bin:$PATH \
    HOME=/root \
    PYTHONPATH=/opt/pipeline/src:/opt/pipeline \
    AUX_ROOT=/opt/aux \
    CONFIG_ROOT=/opt/pipeline/configs \
    ISOFIT_DEBUG=0 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache \
    DEM_VRT=/aux/dem/cop30_mosaic.vrt
RUN chmod 755 /root && chmod -R a+rX /root/.isofit /opt/sixs /opt/aux /opt/build /opt/pipeline && \
    mkdir -p /work && chmod 1777 /work

# -- Runtime checks (fail the build, not a run) -------------------------------
RUN for b in /opt/sixs/*/sixsV2.1; do \
        ldd "$b" | grep -q "not found" && { echo "missing library for $b"; ldd "$b"; exit 1; }; done; \
    /root/.isofit/sixs/sixsV2.1 --selftest && \
    python -c "import importlib, fast_al; A = importlib.import_module('isofit.utils.apply_oe'); \
        assert A.ALAlg is fast_al.analytical_line, A.ALAlg; print('apply_oe uses fast_al')" && \
    python -c "import isofit, torch, rasterio, spectral, h5py, matplotlib, scipy, pyproj; \
        import readers.prep, atm_post, harmonise, writers, report; \
        print('isofit', isofit.__version__, 'torch', torch.__version__)"

WORKDIR /work

HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import isofit, glob; \
        assert glob.glob('/root/.isofit/srtmnet/**/*.h5', recursive=True), 'no sRTMnet .h5'; \
        assert glob.glob('/opt/sixs/v3/sixsV2.1'), 'no 6S build'; \
        assert isofit.__version__.startswith('4.1'), isofit.__version__" || exit 1

ENTRYPOINT ["/opt/pipeline/entrypoint.sh"]
CMD ["--help"]

# ----------------------------------------------------------------------------- test
# The unit tests on top of the exact runtime image (tests/, synthetic data only, no network): a failing test or
# coverage under 30 % (pytest.ini) fails `docker build --target test`, which scripts/build_image.sh runs before it
# tags the runtime image. Nothing from this stage ships.
FROM runtime AS test
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
WORKDIR /opt/pipeline
COPY requirements.txt requirements-dev.txt pytest.ini /opt/pipeline/
COPY tests/ /opt/pipeline/tests/
RUN pip install --no-cache-dir --extra-index-url "${TORCH_INDEX}" -r requirements-dev.txt && \
    python -m pytest -p no:cacheprovider

# The default target stays the runtime image.
FROM runtime
