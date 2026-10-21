# ACOLITE as a second L1 -> L2 engine, next to the ISOFIT image.
# Pinned to one upstream commit so runs are reproducible.
FROM condaforge/miniforge3:latest

ARG ACOLITE_COMMIT=8e04b97
# Own env, not base: changing python in base breaks mamba's package cache.
RUN mamba create -y -n acolite -c conda-forge python=3.12 pyresample cartopy scipy numpy \
        pyhdf gdal libgdal-jp2openjpeg libgdal-netcdf netcdf4 h5py pygrib requests \
        matplotlib scikit-image pyproj zarr fsspec aiohttp rasterio \
    && mamba clean -afy
ENV PATH=/opt/conda/envs/acolite/bin:$PATH
# The env is never "activated", so point PROJ/GDAL at its data by hand;
# without this every CRS lookup warns and reprojection can fail.
ENV PROJ_DATA=/opt/conda/envs/acolite/share/proj \
    GDAL_DATA=/opt/conda/envs/acolite/share/gdal

RUN git clone https://github.com/acolite/acolite.git /opt/acolite \
    && cd /opt/acolite && git checkout ${ACOLITE_COMMIT} && rm -rf .git

WORKDIR /opt/acolite
ENTRYPOINT ["python", "launch_acolite.py", "--cli"]
