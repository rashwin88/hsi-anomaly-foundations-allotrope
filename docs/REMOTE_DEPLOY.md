# Remote GPU deployment — one tarball, scp, run

End-to-end recipe for moving the entire local stack (code + state) onto a
remote linux/amd64 GPU host (vast.ai, lambda, any cloud) and reaching the
UI from your laptop's browser.

The flow:

  1. **On your laptop:** `./scripts/snapshot_bundle.sh` builds one
     self-contained tarball with pre-built images, snapshotted volumes,
     compose files, and a host-side loader script.
  2. **SCP** the tarball to the remote host.
  3. **On the host:** extract → `bash bootstrap.sh` → stack is live.
  4. **On your laptop:** `./scripts/remote_tunnel.sh` opens the SSH
     port forward; the UI shows up at <http://localhost:3000>.

No `docker build` ever runs on the host. No re-onboarding scenes. No
re-running actions. The remote stack starts in **exactly the same state**
as your laptop at snapshot time.


## Why this design

### How docker + CUDA actually work across architectures

Common confusion: "my Mac has no CUDA, can the image work on a GPU host?"
Yes. Here's why.

CUDA splits cleanly into two halves:

| Half | Where it lives | When packed |
|---|---|---|
| **Kernel driver + GPU hardware** | the host OS | provided by the cloud (vast.ai instances ship with NVIDIA drivers pre-installed) |
| **Userland libraries** (cuDNN, libcublas, torch wheels) | the image's filesystem | baked at `docker buildx --platform linux/amd64` time |

The bridge is **`nvidia-container-toolkit`**: at `docker run` time when
`--gpus all` is passed, the toolkit mounts the host's driver libraries
into the container. The container then has both halves and torch can
talk to the GPU.

So **your Mac never has CUDA and never runs CUDA code**. It just packs a
tarball of Linux binaries. Those binaries happen to know how to use a
GPU *if one shows up at runtime* — but they don't need one at build
time. `docker buildx` cross-builds for `linux/amd64` via qemu emulation
(slow but works), and the resulting image runs natively on the x86_64
host (no emulation at runtime).

You cannot *test* the GPU path on Mac — `tensor.cuda()` will raise
"CUDA not available." That's expected. The Mac is the build box, not a
GPU test target.


### Why ship a tarball instead of rsync + build

Building on the host means ~20 minutes of compiling cp314 scientific
wheels (h5py, rasterio, pyarrow) under a Python version with no
prebuilt wheels yet. We pay that cost **once on the Mac** during the
snapshot. The host just `docker load`s the resulting images — seconds.

Also: building on the host means the host needs the full toolchain,
build deps, source tree, the lot. Tarball-only means the host can be
the minimal NVIDIA-CUDA-driver image vast.ai ships, plus
nvidia-container-toolkit (which the loader installs if missing). No
git, no python, no compilers.


### Why snapshot volumes (not re-onboard)

Onboarding PRISMA takes ~30s, EnMAP ~30s, AVIRIS-NG ~3 min. Multiply by
however many scenes you've onboarded. Plus every action you've run.
Plus the splib07 cache.

If you snapshot the named volumes (`allotrope_db`, `allotrope_data`,
`allotrope_artifacts`, `allotrope_models`, `allotrope_splib07`) you
get all that state for free on the remote. Database rows reference
volume-relative paths (`scenes/<id>/raw`, not `/Users/.../scenes/...`),
so transplanting just works — no rewriting required.

The trade-off: postgres bytes must be **stopped** during snapshot for
WAL consistency. The snapshot script does that automatically — `docker
compose down`, tar the volumes, restart. ~10 seconds of local downtime.


## Step 1 — Build the bundle (on your laptop)

From the repo root:

```bash
./scripts/snapshot_bundle.sh
```

What this does, in order:

1. Verifies `docker buildx` and `zstd` are installed.
2. Cross-builds `docker-api`, `docker-worker`, `docker-frontend` for
   `linux/amd64`. **First run: 30–45 min** (compiling scientific wheels
   under qemu emulation, no shortcut). Subsequent runs reuse the buildx
   cache — typically 1–3 min for code-only changes.
3. Saves all four images (api, worker, frontend, postgres) as tarballs.
4. Brings the local stack down to stop postgres cleanly.
5. Tars each named volume by spinning up a throwaway alpine container.
6. Restarts the local stack.
7. Bundles compose files + `docker/.env` (AS-IS — see security note below).
8. Embeds `scripts/remote_load.sh` as `bootstrap.sh` inside the bundle.
9. Compresses everything with `zstd -19` into
   `dist/allotrope-bundle-<sha>-<ts>.tar.zst`.

Final bundle is typically **20–50 GB** depending on how many scenes
you've onboarded. The worker image is ~6–8 GB on its own.

### Security note about `.env`

`docker/.env` ships **as-is** in the bundle. It contains
`POSTGRES_PASSWORD`, `JWT_SECRET`, and `ADMIN_PASSWORD`. Shipping the
same secrets means:

- The transplanted postgres volume works (its users were hashed with
  this `POSTGRES_PASSWORD`).
- Your existing browser session cookies remain valid (JWT signed with
  the same secret).
- The bundle file on the remote host now contains your laptop's
  secrets.

**After teardown, treat the bundle file on the remote as compromised**
— delete it, rotate `JWT_SECRET` locally, force re-login.

For a cleaner separation, rotate secrets manually before snapshotting
and only keep a "shipping" set on the laptop during deployment.


## Step 2 — SCP the bundle to the host

```bash
scp -P <vast-port> -i ~/.ssh/<key> \
    dist/allotrope-bundle-*.tar.zst \
    root@<host>:/root/
```

Time: a 30 GB bundle over 200 Mbit/s is ~20 minutes. vast.ai's typical
uplink is 500 Mbit–1 Gbit, often faster.


## Step 3 — Extract + bootstrap (on the host)

```bash
ssh -p <vast-port> -i ~/.ssh/<key> root@<host>
cd /root
tar --use-compress-program=zstd -xf allotrope-bundle-*.tar.zst
bash bootstrap.sh
```

What `bootstrap.sh` does:

1. Checks disk space (≥ 60 GB free on `/`).
2. Installs docker + docker-compose-plugin if missing.
3. Installs nvidia-container-toolkit if missing.
4. Smoke-tests GPU pass-through with `docker run --gpus all nvidia/cuda…`.
5. `docker load`s all four image tarballs.
6. Re-creates the named volumes and restores their contents.
7. Drops the compose files into `/root/allotrope/`.
8. `docker compose up -d` (no `--build` — images are pre-loaded).
9. Prints health-check results.

Total time: 1–3 minutes (mostly volume restore + container startup).

Idempotent — re-running on an already-loaded host destroys the existing
volumes and re-restores from the bundle. **Run this only when you want
to overwrite the remote with fresh local state.**


## Step 4 — SSH tunnel (back on your laptop)

```bash
./scripts/remote_tunnel.sh root@<host> -p <vast-port> -i ~/.ssh/<key>
```

Forwards `localhost:3000` → remote frontend, `localhost:8000` → remote
api. Open <http://localhost:3000> in the browser.

Leave the terminal open. Ctrl-C closes the tunnel; the remote services
keep running. If the tunnel drops (laptop sleeps, network blip), re-run.

For auto-reconnect on flaky networks: `USE_AUTOSSH=1 ./scripts/remote_tunnel.sh ...`


## Iterating on code

The whole point of the bundle is "shippable state." For inner-loop
iteration (a typo fix in the api), shipping a 30 GB bundle every time
is wasteful. Two patterns:

### a) Quick path: re-snapshot, smaller in practice

The buildx cache makes code-only image rebuilds fast (~1–3 min for an
api change). The volume tars only change for volumes that actually
changed bytes — if you didn't onboard a new scene, `allotrope_data.tar`
is identical to before but still gets re-packed. **Plan to ship a fresh
bundle for "real" iterations** and use the host's running state for
quick sanity checks via the tunnel.

### b) Future improvement: code-only bundle (not built yet)

A future `--code-only` flag to `snapshot_bundle.sh` would build only
the three images, skip the volume tarballs, and ship a much smaller
bundle. The host's `remote_load.sh` would `docker load` the new images
and `docker compose up -d` without touching volumes. Roadmap item;
ping when you need it.


## Teardown

```bash
ssh -p <vast-port> -i ~/.ssh/<key> root@<host> \
  "cd /root/allotrope && docker compose down"
```

Then destroy the vast.ai instance from the dashboard. Volumes only live
on the rented host — they evaporate when the instance terminates.

**Save anything you care about before destroying the instance**:

```bash
# Run on your laptop, while the tunnel + remote are still up:
./scripts/snapshot_bundle.sh   # except this is currently a laptop-side script;
                               # the inverse — snapshot remote, restore local —
                               # would be the symmetric op for keeping work.
```

That inverse direction isn't scripted yet — if you find yourself
needing it (i.e., you onboarded scenes on the remote and want them on
the laptop), tell me and I'll add a `pull_bundle.sh`.


## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `snapshot_bundle.sh` fails at "Cross-building images" | Docker Desktop low on memory or disk | Bump Docker Desktop's VM RAM to ≥ 8 GB; free disk |
| First buildx run looks frozen for 10 min | qemu emulation compiling numpy/scipy/h5py from source under cp314 | Normal. Total first build ~30–45 min. Subsequent runs are fast. |
| `tar: zstd: write error` on the host during extract | Disk full | Need ≥ 60 GB free on `/`. Delete the `.tar.zst` after extracting if disk-tight. |
| `bootstrap.sh` says "container can see the GPU" but worker boots and torch falls back to CPU | Compose didn't load the GPU override | Verify `docker-compose.gpu.yml` is in the bundle and `bootstrap.sh` is picking it up. |
| Tunnel connects but browser says "connection refused" | Stack still booting | Wait. `docker ps` on host should show all four containers `Up`. |
| Login page rejects credentials | Bundle's `.env` didn't ship (or got overwritten on host) | Confirm `/root/allotrope/.env` exists and has the same `JWT_SECRET` + `ADMIN_PASSWORD` as your laptop. |
| `bootstrap.sh` says "running postgres is not postgres:16-alpine" | Local docker has a different postgres version cached | This breaks volume portability. Pull the right image: `docker pull postgres:16-alpine`, then re-snapshot. |


## Reference

| Script | Where it runs | When |
|---|---|---|
| [`scripts/snapshot_bundle.sh`](../scripts/snapshot_bundle.sh) | laptop | every time you want to ship state to remote |
| [`scripts/remote_load.sh`](../scripts/remote_load.sh) | host (embedded as `bootstrap.sh`) | once per extract |
| [`scripts/remote_tunnel.sh`](../scripts/remote_tunnel.sh) | laptop | every working session |
