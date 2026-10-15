# Scripted downloads from the DLR Geoservice portal

How to pull large files from `https://download.geoservice.dlr.de/` without a browser,
resumably, and verify them. Worked out on SpectralEarth (2026-09-21/22); the same auth model
applies to anything else under that host (e.g. `.../ENMAP/files/...`) — only the base URL and
the checksum list change.

```
browser login ──▶ sso.eoc.dlr.de (EO-Lab CAS) ──302 + ticket──▶ download.geoservice.dlr.de
                                                                  └─ sets `session` cookie
session.txt (that cookie's value) ──▶ curl probe ──▶ wget -c --max-redirect=0 ──▶ md5sum
```

## The helpers

Three bash scripts, currently in `C:\Users\FITT\Downloads\` (not yet in the repo):

| Script | Use it for |
|---|---|
| `se_get.sh [file] [outdir]` | One file. Probes the session, checks free space, resumes, verifies MD5. |
| `se_watchdog.sh` | Babysits a `se_get` systemd unit; restarts it **only** on a genuine stall. |
| `se_all.sh <workdir> <s3-prefix> [profile] [parts...]` | Cloud VM: download → MD5 → `aws s3 cp` → delete, one part at a time. |

All three hard-code `BASE=https://download.geoservice.dlr.de/ENMAP/files/SPECTRAL_EARTH` and
the SpectralEarth MD5 table. For another collection, change `BASE` and the checksum heredoc.

## Getting a session cookie

You need the `session` cookie **on the download host**, not anything on the SSO host.

1. In Chrome, open the folder URL (e.g. `https://download.geoservice.dlr.de/ENMAP/files/SPECTRAL_EARTH/`)
   and finish the EO-Lab login **until you see the file list**. That final bounce back to
   `download.geoservice.dlr.de` is what sets the cookie.
2. F12 → Application → Cookies → `https://download.geoservice.dlr.de` → row `session` → copy
   the **Value**.
3. Paste it alone on one line into `session.txt` next to the script. (A Netscape
   `cookies.txt` jar also works as a fallback for `se_get.sh`.)

The session lasts **about 2–2.5 hours**. Expect to repeat these steps.

## Download one file

```bash
# Windows box: Docker/WSL lives in the Ubuntu distro
wsl -d Ubuntu --cd "C:\Users\FITT\Downloads" -- bash se_get.sh spectral_earth_part_2.tar.gz /mnt/e/spectral_earth
```

What it does, in order:

1. **Probe** — `curl -r 0-99 --max-redirs 0` with the cookie. `206`/`200` = session good;
   `302` = log in again (it prints the steps and exits).
2. **Space check** — remote size from `Content-Range` minus bytes already on disk vs `df`.
3. **Resume** — `wget -c --max-redirect=0 --tries=1 -P <outdir>`.
4. **Verify** — once the file is complete, `md5sum` against the published sum.

When wget exits non-zero (almost always: session expired), refresh `session.txt` and re-run the
same command. It picks up from the byte it stopped at.

## Gotchas — each one cost us a run

**1. Send a browser User-Agent.** The portal rejects `Wget/…` and curl's default UA with a
`302 → sso.eoc.dlr.de/eoc/auth/login` *even with a valid cookie*. It looks exactly like an
expired session. The scripts send a Chrome UA (`UA=` env var overrides).

**2. Never download with the browser.** When the session expires mid-transfer the next request
returns the login page; Chrome decides the download is unresumable and **deletes the partial
file**. Lost ~300 GB once.

**3. Always `--max-redirect=0`.** Without it, an expired session makes wget follow the 302 and
write the HTML login page into the middle of your file.

**4. `--tries=1` — no in-process retries.** On the WSL → NTFS mount, a wget internal retry
re-stats the file, and the stat can lag the true write position by a few KB. The resumed bytes
land misaligned, the total size still comes out exact, and only the MD5 catches it. Every resume
done by a *fresh* wget process verified clean. So: let wget die, re-run a new process.

**5. Use `-P <dir>`, not `-O <file>`.** `wget -c -O` is not a well-defined resume in GNU wget.

**6. Never restart a transfer that is still moving.** An already-open connection keeps
streaming after the session expires — only a *new* request is bounced to SSO. Killing a slow
but live 2.5 MB/s transfer once lost the session and stopped the download dead. Act on zero
bytes, never on "slow". `se_watchdog.sh` enforces this.

**7. Don't check gzip magic on split archives.** SpectralEarth's nine
`spectral_earth_part_N.tar.gz` files are pieces of **one** archive; only part 1 starts with
`1f 8b`. Verify with the published MD5s instead.

**8. The session is tied to the browser's User-Agent.** A fresh, unexpired cookie still gets
the SSO `302` if it is sent with a different UA, from any IP (seen 2026-10-13: Chrome 154 cookie
+ a Chrome 129 UA string failed from both the laptop and Colab; the browser's exact UA worked from
Colab). Copy the UA together with the cookie. The easiest way to get both is DevTools → Network
→ right-click the file's request → **Copy as cURL (bash)**: the `-b 'session=…'` and
`User-Agent:` lines are the two values you need. The Network tab only lists requests made
while it is open, so tick *Preserve log* and reload the page first.

## EnMAP products (L2A, L1C) via STAC

For EnMAP scenes, no folder listing is needed: the EOC STAC catalogue
(`https://geoservice.dlr.de/eoc/ogc/stac/v1/`, collections `ENMAP_HSI_L2A`, `ENMAP_HSI_L1C`, …)
is public, and each item's asset `href`s point straight at
`download.geoservice.dlr.de/ENMAP/files/<level>/<yyyy>/<mm>/<dd>/<datatake>/<tile>/<product>/…`.
Only the downloads need the cookie. Unlike SpectralEarth, there is no published MD5 list, so
files are checked by size (from `Content-Range` on a 1-byte probe) and by file signature;
COGs come as both classic TIFF (`II*`) and BigTIFF (`II+`).

The Colab scripts used for the 2026-10-13 stage_3 batch live on Drive in
`MyDrive/PS_11/EnMAP_datasets/_downloads/`:

| File | Use it for |
|---|---|
| `select_enmap.py` | STAC search per region, newest processor version per product, skip scenes already held |
| `enmap_dl.py` | Download a candidate list: local disk first, size + signature check, then copy the finished scene folder to Drive. Re-runnable; resumes partials |
| `stage3_batch_20261013.json` | What that batch added (region, date, cloud, land fraction, file sizes) |

`enmap_dl.py` reads the cookie from `/content/session.txt` and the UA from `/content/ua.txt`
on every request, so you can refresh the cookie mid-run. On Colab, 4 parallel scenes ran at
~15 MB/s combined including the Drive copy: 55 scenes (24.7 GB) in about 30 minutes, inside
one session. See `colab-ssh.md` for driving a Colab runtime from your PC.

## Unattended runs (watchdog)

```bash
# in WSL Ubuntu
sudo systemd-run --unit=se_get --working-directory=/mnt/c/Users/FITT/Downloads \
     /bin/bash /mnt/c/Users/FITT/Downloads/se_get.sh
sudo systemd-run --unit=se_watchdog /bin/bash /mnt/c/Users/FITT/Downloads/se_watchdog.sh
journalctl -fu se_get -u se_watchdog
```

Do **not** pass `--collect` — the unit then vanishes on failure and cannot be restarted. The
watchdog samples size every 60 s, restarts after 5 consecutive zero-byte samples (300 s
cooldown), probes the session first, and **exits** with instructions if the session is dead —
only a browser login can fix that. Configure it with `F=`, `URL=`, `TOTAL=` env vars for a
file other than part 2.

## Bulk ingest to S3 (cloud VM)

```bash
JOBS=3 ./se_all.sh /data s3://<bucket>/spectral_earth Allotrope-Prod        # all 9 parts
./se_all.sh /data s3://<bucket>/spectral_earth Allotrope-Prod 3 4 5          # a subset
```

- Preflights the cookie **from the VM**: a cookie issued to your laptop's browser may not work
  from another IP. If the preflight 302s, log in from inside the VM instead.
- Parts already in S3 are skipped; partials resume. Re-run freely.
- Peak scratch ≈ 300 GiB × `JOBS`. `JOBS=3` on a 1 TB volume is the sane middle.
- Rebuild the archive without materialising 2.8 TB:
  ```bash
  for i in $(seq 1 9); do aws s3 cp s3://<bucket>/spectral_earth/spectral_earth_part_$i.tar.gz -; done \
    | tar -xzv -C /data/spectral_earth
  ```

> `se_all.sh` still uses `--tries=20` (in-process retries). That predates gotcha 4, which was
> observed on WSL→NTFS; on a VM's native ext4 it hasn't been seen, but the MD5 check is what
> protects you either way.

## Numbers to plan with

| | |
|---|---|
| Per-connection throughput | ~20 MB/s (DLR cap; 1 Gbps link measured 55 Mbps on one connection) |
| SpectralEarth part size | 322,122,547,200 bytes (exactly 300 GiB) each, 9 parts, ~2.6 TiB total |
| One part | ~4.5 h → 2–3 re-logins per part |
| All 9 | ~39 h sequential, ~4–5 h with `JOBS=9` |
| Disk | 300 GiB per part to land + as much again to extract; this Windows box can't hold it |

Because a session expires before a single part finishes, downloading a 300 GiB file can't run
fully unattended: someone has to refresh `session.txt` every ~2 hours.
