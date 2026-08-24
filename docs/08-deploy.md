# 8. Deploy

## The stack

Five compose services. Everything binds to `127.0.0.1` — nothing is exposed publicly.

| Service | Image | Port |
|---|---|---|
| `postgres` | `postgres:16-alpine` | 5432 |
| `bootstrap` | api image, one-shot | — |
| `api` | `backend/Dockerfile` | 8010 → 8000 |
| `worker` | `backend/Dockerfile.worker` | none (queue-driven) |
| `frontend` | `frontend/Dockerfile` (nginx) | **3010** → 80 |

Startup order: `postgres` (healthcheck) → `bootstrap` (migrate + seed, then exits) →
`api` + `worker` → `frontend`.

**Dockerfiles are not in `docker/`** — they sit next to their source. The api and worker
build with the **repo root** as context so `app/` can be copied in alongside `backend/`.

## Five named volumes

Each has an explicit `name:` to defeat compose's project prefixing — the bundle loader
depends on that.

| Volume | Mount | Holds |
|---|---|---|
| `allotrope_db` | — | Postgres data |
| `allotrope_data` | `/data` | staging, raw scenes, vendables, annotations |
| `allotrope_models` | `/models` | checkpoints (api mounts **read-only**) |
| `allotrope_artifacts` | `/artifacts` | thumbnails, action outputs, exports |
| `allotrope_splib07` | `/splib07_cache` | per-sensor USGS spectral library caches |

Write access is one-directional: the api writes only to `/data/staging`; the worker is the
sole writer of vendables, artifacts and models.

## Compose file variants

| File | Difference |
|---|---|
| `docker/docker-compose.yml` | the source of truth, heavily commented |
| `docker-compose.remote.yml` | identical graph with every `build:` stripped, using bare `image:` tags — for a host that got its images via `docker load` |
| `docker/docker-compose.gpu.yml` | override adding the nvidia device reservation to `worker` |
| `docker-compose.gpu.remote.yml` | same override, remote variant |

GPU is an **override** rather than the default because an unconditional GPU request breaks
`up` on a machine without nvidia-container-toolkit.

## Shipping to a rented GPU box

The whole stack — images *and* data — travels as one tarball.

```bash
./scripts/snapshot_bundle.sh                    # on your machine
# → dist/allotrope-bundle-<sha>-<ts>.tar.zst   (10–50 GB)

rsync -av --progress -e "ssh -p <PORT>" dist/allotrope-bundle-*.tar.zst root@<HOST>:/root/

ssh -p <PORT> root@<HOST>
cd /root && tar --use-compress-program=zstd -xf allotrope-bundle-*.tar.zst && bash bootstrap.sh

./scripts/remote_tunnel.sh root@<HOST> -p <PORT>   # back on your machine, leave running
```

Then browse `http://localhost:3010`.

`snapshot_bundle.sh` cross-builds linux/amd64 images, briefly stops the stack for a
consistent volume snapshot, and zstd-compresses everything. First build is **~15 min**
(see below); later builds reuse the cache and take 5–10.

`bootstrap.sh` on the host = `scripts/remote_load.sh`: checks free disk, installs Docker and
the NVIDIA toolkit if absent, tests GPU pass-through, loads four images, restores five
volumes, starts the stack.

**Three things that catch people out:**
- Rent a **VM-type** instance, not a container-type — container types cannot run this
  Docker setup.
- The bundle ships your `.env` **with secrets in it**. Rotate afterwards.
- Re-running `bootstrap.sh` **destroys the remote volumes** and restores from the bundle.
  Anything created on the remote is lost. Treat the remote as a one-way destination.

## Why the images are big, and what actually compiles

The worker image is **~14 GB**. Most of that is `torch`, which on PyPI bundles the CUDA
runtime — `nvidia_cudnn` (707 MB) and `nvidia_cublas` (594 MB) alone. The GPU host needs
this; a CPU-only host does not, but both use the same image today. A CPU-only build would
need `--index-url https://download.pytorch.org/whl/cpu`, which would then break GPU
inference on the remote box, so it is a deliberate trade rather than an oversight.

As of 2026-08, numpy, h5py, pandas and torch all ship cp314 wheels. Only **rasterio** and
**fiona** still compile from source — they bind to GDAL. That is why both Dockerfiles
install `build-essential`, `libgdal-dev` and `libhdf5-dev`; those deps are load-bearing and
removing them breaks the build.

Note the training VM runs **Python 3.12**, so the containers and the research environment
are not the same interpreter.

## Monitoring

Two unrelated things, easily confused:

- **In-app** — the api serves `/metrics/host` and `/metrics/workload`, rendered on the
  `/monitoring` page (CPU/RAM/disk/GPU + queue depth and throughput).
- **`monitoring/monitor.py`** — a standalone single-file Flask dashboard on port 8080 that
  polls `psutil` and shells out to `nvidia-smi`. It watches the **training VM** and is
  completely decoupled from the Docker stack. Reach it with
  `ssh -L 8080:localhost:8080 …`.

There is **no Prometheus, no Grafana, and no CI** — there is no `.github/` directory.

See also [9. Known issues](09-known-issues.md) for the port mismatch in the deploy scripts.
