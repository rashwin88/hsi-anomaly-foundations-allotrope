# Deploy to a remote host — quick guide

Everything you need to take the stack from your Mac to a rented GPU box,
in plain language.

---

## What this does

You'll build a single tarball on your Mac that contains the running app
**plus** all its data (scenes, vendables, action outputs, models,
splib07 caches, postgres database). Copy that one file to a remote
Linux host. Run one script. The whole stack starts there with the
exact same state as your Mac.

Then you open a browser on your Mac and see the remote app.

---

## What you need first

1. **A remote Linux host with a GPU.** vast.ai, lambda, any cloud
   provider. Pick a **"Virtual Machine"** type instance — *not*
   "container" type (container types can't run our Docker setup).
2. **SSH access to the host.** Your public SSH key (`~/.ssh/id_ed25519.pub`)
   must be added to the host's accepted keys. Most cloud providers
   have a "SSH Keys" setting in your account; paste it there.
3. **The SSH command for the host.** Looks something like
   `ssh -p 17055 root@103.116.53.7`. Note the port and IP.
4. **Docker Desktop running on your Mac** with at least 8 GB RAM
   allocated (Settings → Resources). The bundle build needs room.

---

## Step 1 — Build the bundle (on your Mac)

Open a terminal in the repo:

```bash
cd "/Users/ashwinravi/Desktop/Code Repos/hsi-anomaly-foundations-allotrope"
./scripts/snapshot_bundle.sh
```

What happens:

- It cross-builds Docker images for the remote (Linux x86_64).
  **The first run takes 30–45 minutes**, mostly compiling scientific
  Python wheels. Subsequent builds reuse the cache and finish in 5–10 min.
- It briefly stops your local app (~10 seconds) to take a clean
  snapshot of postgres + the named volumes. Your local app comes
  back up automatically.
- It compresses everything with zstd. Slow but smaller.
- Final file lands in `dist/allotrope-bundle-<short-sha>-<timestamp>.tar.zst`.
  Expect anywhere from 10 GB to 50 GB depending on how many scenes
  you've onboarded.

You'll see a green banner at the end with the exact filename and the
`scp` command to copy it.

**While it runs, you can keep using your Mac**, but don't try to
onboard new scenes during the brief postgres-down window.

---

## Step 2 — Copy the bundle to the host

In a **second terminal** on your Mac (so the build can keep running):

```bash
scp -P <PORT> dist/allotrope-bundle-*.tar.zst root@<HOST>:/root/
```

Replace `<PORT>` and `<HOST>` with your instance's values.

**Note**: `scp` doesn't show progress. For a multi-GB upload you'll
want `rsync` instead — it shows percentage and resumes if interrupted:

```bash
rsync -av --progress -e "ssh -p <PORT>" \
    dist/allotrope-bundle-*.tar.zst \
    root@<HOST>:/root/
```

How long this takes depends on your upload speed:
- 50 Mbit/s home internet → ~1 hour per 20 GB
- Office fibre → minutes

---

## Step 3 — Extract and start the app (on the host)

SSH into the host:

```bash
ssh -p <PORT> root@<HOST>
```

Then on the host:

```bash
cd /root
tar --use-compress-program=zstd -xf allotrope-bundle-*.tar.zst
bash bootstrap.sh
```

What `bootstrap.sh` does:

1. Checks the host has enough disk space (≥ 60 GB free).
2. Installs Docker + NVIDIA toolkit if they aren't already there
   (vast.ai instances usually have both pre-installed).
3. Tests that Docker can see the GPU.
4. Loads all four Docker images from the bundle.
5. Restores the five data volumes.
6. Starts the app.

Takes 1–3 minutes. When it's done, you'll see a status box with
health checks (`api: 200, frontend: 200`). The app is now running
on the host, but **only listening on `127.0.0.1`** for security —
not reachable from the public internet yet.

---

## Step 4 — Open a tunnel so your browser can reach it

Back on your **Mac**, in a third terminal:

```bash
cd "/Users/ashwinravi/Desktop/Code Repos/hsi-anomaly-foundations-allotrope"
./scripts/remote_tunnel.sh root@<HOST> -p <PORT>
```

This terminal will sit there silently — that's correct. It's holding
the port forward open.

Then in your browser: **<http://localhost:3010>**

Log in with the same admin credentials you use on your Mac (the
postgres database came across with the bundle, so the same users
exist).

**Leave the tunnel terminal open** while you're working. Closing it
ends the tunnel; the app on the host keeps running.

---

## Step 5 — When you're done for the day

You have two choices:

### Pause (keep the host, save money on a different cloud)

Most cloud providers let you "stop" an instance — it pauses, you
pay storage-only. When you restart it, your bundle is still there,
just run `cd /root && bash bootstrap.sh` again to start the app.

### Destroy (one-time test, throw it away)

On the cloud dashboard, destroy the instance. All data on the host
is gone. **Anything you onboarded or generated on the remote is
lost** — only what was on your Mac at bundle-build time survives.

If you want to keep work you did on the remote, do this BEFORE
destroying:

```bash
# On the host:
docker compose -f /root/allotrope/docker-compose.yml down
# (Future: a "snapshot_remote_bundle.sh" script — not built yet.
#  Ping me if you need it before then.)
```

---

## Common problems

### "Permission denied (publickey)"
Your SSH key isn't on the host yet. Add it via your cloud provider's
SSH Keys page, then restart or edit the instance (some providers
require this to push the key into the running container).

### "Connection refused" when SSHing
The instance is still booting (wait 30s) or the port number is
wrong (re-check the cloud dashboard's connect panel).

### "scp" hangs at 0 bytes
Auth issue — test with `ssh -p <PORT> root@<HOST> echo ok` first.
If that hangs or asks for a password, scp will too.

### Bootstrap fails with "less than 60 GB free"
The host's disk is too small. Either pick a bigger instance or
destroy and re-rent with more disk.

### Bootstrap says "host nvidia-smi not available"
The host doesn't have a GPU driver. You rented a container-type
instance, not a VM-type. Destroy and re-rent as VM type.

### Browser shows "connection refused" at localhost:3010
The tunnel terminal isn't running, or the app on the host isn't up
yet. Check the tunnel terminal is open with no error. On the host,
run `docker ps` — all four `allotrope_*` containers should be `Up`.

### Login fails on the remote
The bundle ships your local `.env` (containing admin password +
JWT secret), so the same login should work. If it doesn't, run on
the host:
`cat /root/allotrope/.env | grep -E 'ADMIN|JWT'`
and confirm it matches your Mac's `docker/.env`.

---

## Iterating on code

If you change code on your Mac and want it on the remote:

1. Rerun `./scripts/snapshot_bundle.sh` on your Mac. Subsequent
   builds are fast (~5 min) because Docker reuses cached layers.
2. SCP the new bundle to the host.
3. On the host: extract + `bash bootstrap.sh`. **Warning**: this
   destroys the remote volumes and restores from the new bundle.
   Anything done on the remote in the meantime is lost. So either
   work only on the Mac, or accept that the remote is a one-way
   destination.

---

## Reference

The three scripts you'll use:

| Script | Where it runs | When |
|---|---|---|
| `scripts/snapshot_bundle.sh` | your Mac | every time you want to push to a remote |
| `bootstrap.sh` (inside the bundle) | the host | once per extracted bundle |
| `scripts/remote_tunnel.sh` | your Mac | every working session |

Longer design doc (why this approach, how it works under the hood):
[`docs/REMOTE_DEPLOY.md`](REMOTE_DEPLOY.md).
