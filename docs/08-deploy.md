# 8. Deploy

## The stack

Five compose services, plus one opt-in. Everything binds to `127.0.0.1` — nothing is exposed publicly.

| Service | Image | Port |
|---|---|---|
| `postgres` | `postgres:16-alpine` | 5432 |
| `bootstrap` | api image, one-shot | — |
| `api` | `backend/Dockerfile` | 8010 → 8000 |
| `worker` | `backend/Dockerfile.worker` | none (queue-driven) |
| `frontend` | `frontend/Dockerfile` (nginx) | **3010** → 80 |
| `inventory` *(opt-in)* | `backend/Dockerfile.inventory` | 8020 |

Startup order: `postgres` (healthcheck) → `bootstrap` (migrate + seed, then exits) →
`api` + `worker` → `frontend`.

`inventory` sits behind the compose profile `inventory` and starts only with
`--profile inventory up -d inventory`. It is independent of everything above — Supabase for
its database, S3 for its bytes, migrations run by its own CMD — so a missing
`backend/.env.inventory` never blocks the main stack. It needs `AWS_ACCESS_KEY_ID` /
`AWS_SECRET_ACCESS_KEY` in that env file to reach S3. See
[6. Backend](06-backend.md#the-raw-dataset-inventory--a-third-separate-service).

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
| `docker/docker-compose.prod.yml` | EC2 override: Caddy on 80/443 with automatic TLS, `COOKIE_SECURE=true` — see [Production on EC2](#production-on-ec2-prod-account) |

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

## Production on EC2 (Prod account)

Live since 2026-10-13. One box runs the whole compose stack, CPU only, with Caddy as the
only public listener.

| | |
|---|---|
| Instance | `allotrope-prod` · `i-07f79c11c54120f63` · m7i.xlarge (4 vCPU, 16 GiB) · ap-south-1a |
| Address | Elastic IP `13.206.165.200` → **https://13-206-165-200.sslip.io** until the Cloudflare subdomain exists |
| Disk | 300 GB gp3, encrypted, root volume (images + all named volumes) |
| Network | default VPC, public subnet; SG `allotrope-prod-web` allows **80 and 443 only**, from anywhere |
| Shell | **SSM Session Manager** — no SSH port, no key pair |
| Identity | instance role `allotrope-prod-ec2`: inventory bucket read/write, `deploy/*` read on `allotrope-training-data-prod-india`, `/allotrope/prod/*` parameters, SSM core. No long-lived keys on the box |
| Secrets | SSM Parameter Store `/allotrope/prod/*` (SecureString), written to `docker/.env` and `backend/.env.inventory` (mode 600) by `scripts/ec2/write_env.sh` |
| Code | `/opt/allotrope`, cloned over SSH with a **read-only GitHub deploy key** (`/root/.ssh/allotrope_deploy`) |
| Cost | ≈ $186/month on-demand (instance $155, disk $27, EIP $3.65) |

**Request path:** browser → Caddy :443 (Let's Encrypt, `docker/Caddyfile`) → frontend
nginx :80 → `/api/*` → api, `/inventory-api/*` → inventory (read-only). Everything except
Caddy binds to `127.0.0.1`, exactly as on a laptop.

**Compose:** `docker-compose.yml` + `docker-compose.prod.yml` (Caddy, `COOKIE_SECURE=true`),
with `--profile inventory`. Never add `docker-compose.gpu.yml` here.

### Operating it

```bash
# Shell on the box (needs the Session Manager plugin for the AWS CLI)
aws ssm start-session --profile Allotrope-Prod --region ap-south-1 --target i-07f79c11c54120f63

# On the box
cd /opt/allotrope
C="docker compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml --profile inventory"
$C ps -a                          # worker must be 'running'
git pull --ff-only && $C build && $C up -d      # deploy the branch head

# Admin login for the app (username: admin)
aws ssm get-parameter --profile Allotrope-Prod --region ap-south-1 \
    --name /allotrope/prod/admin_password --with-decryption --query Parameter.Value --output text
```

First-time setup, in order: `scripts/ec2/bootstrap.sh` (Docker, AWS CLI, deploy key — add
its public half on GitHub), clone to `/opt/allotrope`, `scripts/ec2/write_env.sh`,
`scripts/ec2/restore_volumes.sh` (models + splib07 from
`s3://allotrope-training-data-prod-india/deploy/volumes/`), then build and `up -d`.

**Switching to the Cloudflare subdomain:** add an `A` record → `13.206.165.200` with proxy
status **DNS only** (Cloudflare's proxy caps request bodies at 100 MB, which breaks scene
uploads), update `/allotrope/prod/site_address`, re-run `write_env.sh`, and
`$C up -d caddy`. Caddy fetches the new certificate itself.

**Rotating a secret:** change it in Parameter Store, re-run `write_env.sh`, recreate the
affected services. Rotating `inventory_db_password` also needs the new value in Supabase.

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
