# =============================================================================
# EnMAP L1C -> L2A pipeline (ISOFIT + sRTMnet)  --  v1.1.4
#
# Bumped to ISOFIT 3.7.7. Adds gfortran and builds 6SV2.1 (required by
# sRTMnet's preSim step — without it, LUT sims produce all-NaN outputs and
# svd_inv_sqrt crashes downstream). Drops the 3.4 noise_path dispatch patch,
# no longer needed on 3.7.
#
# Build:   docker build -t enmap-isofit:1.1 .
# Export:  docker save enmap-isofit:1.1 | pigz -9 > enmap-isofit-1.1.tar.gz
# =============================================================================

FROM condaforge/mambaforge:24.9.2-0

LABEL org.opencontainers.image.title="EnMAP L1C to L2A via ISOFIT/sRTMnet"
LABEL org.opencontainers.image.version="1.1.4"
ENV PIPELINE_VERSION=1.1.4
LABEL org.opencontainers.image.source="internal"

ARG ISOFIT_VERSION=3.7.7
ARG DEBIAN_FRONTEND=noninteractive

# -- OS-level build deps (gfortran needed to compile 6S) ----------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        git \
        curl \
        ca-certificates \
        libgomp1 \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# -- Conda env ----------------------------------------------------------------
COPY environment.yml /tmp/environment.yml
RUN mamba env update -n base -f /tmp/environment.yml \
    && mamba clean -afy \
    && rm /tmp/environment.yml

# -- ISOFIT 3.7.7 + Python-only extras ----------------------------------------
RUN pip install --no-cache-dir \
        "isofit==${ISOFIT_VERSION}" \
        "spectral==0.23.*" \
        "pystac==1.10.*" \
        "lxml==5.*"

# -- 6SV2.1 radiative transfer binary (required by sRTMnet's preSim) ---------
# sRTMnet is a 6S emulator but preSim still runs actual 6S to build the
# transmittance base. On 3.7 `isofit download sixs` pulls the isofit/6S
# release, patches the Makefile with -std=legacy and runs make itself, so no
# hand-rolled gfortran invocation is needed. It swallows make failures, hence
# the explicit ls guard. --path matches env.sixs so runtime resolves it.
RUN isofit download sixs --path /root/.isofit/sixs --overwrite --debug-make && \
    ls -l /root/.isofit/sixs/sixsV* && \
    python -c "from isofit.data.cli.sixs import get_exe; print('6S exe:', get_exe())"

# -- sRTMnet emulator (Keras HDF5, from JPL AVIRIS-NG mirror) -----------------
RUN isofit download srtmnet --path /root/.isofit/srtmnet && \
    ls -lh /root/.isofit/srtmnet/

# -- ISOFIT external data + examples + build multicomponent surface prior -----
# Examples folder contains prism_optimized_irr.dat (6S solar irradiance)
# required by sRTMnet's postSim step.
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

# -- setuptools: REQUIRED, not cosmetic --------------------------------------
# Ray 2.9's dashboard agent imports pkg_resources. Without setuptools the agent
# dies, the raylet dies with it, and every ray.init() ends in
# "Failed to register worker ... to Raylet" - a native abort with no Python
# traceback, which reads like an OOM. Pinned <81 as pkg_resources is slated
# for removal. Deliberately placed after the sRTMnet layer so adding it does
# not invalidate a 5.8 GB download.
RUN pip install --no-cache-dir "setuptools<81"

# -- Fix empirical_line's output band count (upstream bug, 3.6.0 through 4.1.5)
# initialize_output() is handed the LOCATION cube's band count (nlb, always 3)
# while output_metadata carries the radiance bands, so the zero-fill broadcast
# dies: "could not broadcast (L,3,S) into (L,224,S)". n_input_bands is defined
# 10 lines above. Not EnMAP-specific - it breaks --empirical_line for every
# sensor, which is why upstream has not noticed: analytical_line is the
# maintained path. The greps make a future ISOFIT bump fail the build loudly
# instead of silently shipping an unpatched module.
ARG EL=/opt/conda/lib/python3.10/site-packages/isofit/utils/empirical_line.py
RUN grep -qc "(nll, nlb, nls)" "$EL" && \
    sed -i "s/\(output_reflectance_file, (nll, \)nlb\(, nls)\)/\1n_input_bands\2/; \
            s/\(output_uncertainty_file, (nll, \)nlb\(, nls)\)/\1n_input_bands\2/" "$EL" && \
    [ "$(grep -c 'n_input_bands, nls' "$EL")" = "2" ] && \
    python -c "import isofit.utils.empirical_line" && \
    echo "patched empirical_line: nlb -> n_input_bands (2 sites)"

# -- Pipeline code ------------------------------------------------------------
COPY src/       /opt/pipeline/src/
COPY configs/   /opt/pipeline/configs/
COPY entrypoint.sh /opt/pipeline/entrypoint.sh
RUN chmod +x /opt/pipeline/entrypoint.sh

# -- Runtime env --------------------------------------------------------------
ENV PYTHONPATH=/opt/pipeline/src \
    AUX_ROOT=/opt/aux \
    CONFIG_ROOT=/opt/pipeline/configs \
    ISOFIT_DEBUG=0 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

ENV DEM_VRT=/aux/dem/cop30_mosaic.vrt
WORKDIR /work

HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import isofit, os, glob; \
        import pkg_resources; \
        h5s = glob.glob('/root/.isofit/srtmnet/**/*.h5', recursive=True); \
        assert h5s, 'no sRTMnet .h5 found'; \
        sixs = [p for p in glob.glob('/root/.isofit/sixs/sixsV*') if 'lutaero' not in p]; \
        assert sixs, 'no 6S binary'; \
        assert isofit.__version__.startswith('3.7'), f'wrong ISOFIT: {isofit.__version__}'" || exit 1

ENTRYPOINT ["/opt/pipeline/entrypoint.sh"]
CMD ["--help"]
