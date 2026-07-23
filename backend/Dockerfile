# Allotrope api image
#
# Build context is the REPO ROOT (see docker-compose.yml api.build) so the
# image can COPY both `backend/` (the api itself) and the legacy `app/`
# package (needed to unpickle VendableDataset for the spectrum endpoint).
#
# Heavy ML deps (torch, CUDA) belong in the worker image, not here.
# numpy is the only big addition for the api: pickled vendables embed
# numpy arrays.

FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /srv

# Same build-tools fork as the worker — rasterio (added for the
# on-demand annotation overlay re-render at custom radii) needs to
# compile native bindings against GDAL when no cp314 wheel exists.
# Dropable once cp314 wheels for rasterio land.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgdal-dev \
        gdal-bin \
        libhdf5-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# Legacy app/ — only the model-class definitions are touched at request
# time (vendable unpickling), but COPY the whole package since it's small
# Python source.
COPY app ./app

# allotrope api package + migrations.
COPY backend/allotrope ./allotrope
COPY backend/alembic ./alembic
COPY backend/alembic.ini ./alembic.ini

# Bootstrap one-shot — used by the `bootstrap` compose service to run
# migrations + seeds before api / worker boot. Same image as the api so
# we don't ship a second one just for a 30-line shell script.
COPY backend/bootstrap.sh /usr/local/bin/allotrope-bootstrap
RUN chmod +x /usr/local/bin/allotrope-bootstrap

EXPOSE 8000

CMD ["uvicorn", "allotrope.main:app", "--host", "0.0.0.0", "--port", "8000"]
