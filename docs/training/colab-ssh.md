# Colab over SSH (for prototyping with Claude Code)

Gives Claude Code (or you) a shell on a Colab GPU runtime from your Windows machine, so
commands like `ssh colab "python scripts/train_foundation_model.py ..."` run on Colab.

```
your PC ──ssh──▶ cloudflared (local) ──▶ *.trycloudflare.com ──▶ cloudflared (Colab) ──▶ sshd :2222
```

No Cloudflare account is needed: a free "quick tunnel" gives each session a random hostname.

> Colab's terms restrict remote shells on managed runtimes (more strictly on the free tier).
> Check the current Colab FAQ for your plan; the runtime may be disconnected.

## One-time setup (Windows, PowerShell)

Already done on Vinoth's machine (2026-10-13). Repeat on any new machine.

1. **cloudflared**: `winget install Cloudflare.cloudflared` can hang indefinitely, so download
   the binary directly:
   ```powershell
   New-Item -ItemType Directory -Force "$env:USERPROFILE\bin" | Out-Null
   Invoke-WebRequest https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe -OutFile "$env:USERPROFILE\bin\cloudflared.exe" -UseBasicParsing
   ```
2. **A dedicated key with no passphrase**, so Claude can connect without prompting. It only
   grants access to your own temporary Colab runtimes.
   ```powershell
   ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\colab_ed25519" -N '""' -C "colab-claude"
   Get-Content "$env:USERPROFILE\.ssh\colab_ed25519.pub"   # paste into PUBKEY below
   ```
   (`'""'` is deliberate: PowerShell 5.1 drops a bare `''` argument.)
3. **`~/.ssh/config`**: one short `Host` entry per runtime, plus a shared `colab*` block.
   Replace `<you>` with your Windows username:
   ```
   Host colab-dl
     HostName REPLACE-ME.trycloudflare.com

   Host colab*
     User root
     IdentityFile ~/.ssh/colab_ed25519
     IdentitiesOnly yes
     ProxyCommand C:\Users\<you>\bin\cloudflared.exe access ssh --hostname %h
     StrictHostKeyChecking no
     UserKnownHostsFile NUL
     LogLevel ERROR
     ServerAliveInterval 30
   ```
   Host-key checking is off because every session is a fresh machine behind a new hostname.

## Every session

**1. In Colab**, connect a GPU runtime and run this cell. Set `PUBKEY` once and keep the
cell in your notebook.

```python
PUBKEY = "ssh-ed25519 AAAA...  colab-claude"   # contents of ~/.ssh/colab_ed25519.pub

import re, subprocess, time
sh = lambda c: subprocess.run(c, shell=True, check=True)
sh("apt-get -qq update > /dev/null 2>&1; apt-get -qq install -y openssh-server > /dev/null")
sh(f"mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '{PUBKEY}' > ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys")
sh("grep -q '^PermitRootLogin prohibit-password' /etc/ssh/sshd_config || echo 'PermitRootLogin prohibit-password' >> /etc/ssh/sshd_config")
sh("service ssh restart > /dev/null")
port = re.search(r"^port (\d+)", subprocess.run("sshd -T", shell=True, capture_output=True, text=True).stdout, re.M).group(1)
sh("[ -x /usr/local/bin/cloudflared ] || (wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared)")
subprocess.run("pkill cloudflared", shell=True)
# Popen, not `!nohup ... &`: a backgrounded `!` command can die when the cell ends.
cf = subprocess.Popen(["cloudflared", "tunnel", "--url", f"ssh://127.0.0.1:{port}"],
                      stdout=open("/content/cf.log", "w"), stderr=subprocess.STDOUT)
for _ in range(30):
    time.sleep(1)
    hosts = re.findall(r"https://([a-z0-9-]+\.trycloudflare\.com)", open("/content/cf.log").read())
    if hosts:
        break
print(f"sshd port {port}\nHOSTNAME: {hosts[0] if hosts else 'not found, see /content/cf.log'}")
```

**2. On your PC**, give the printed hostname to Claude with the runtime's name
("`colab-dl` is `xyz.trycloudflare.com`") and it updates that one `HostName` line. Or edit
`~/.ssh/config` yourself, saving it without a BOM (Windows OpenSSH rejects a UTF-8 BOM), and
test with `ssh colab-dl nvidia-smi`.

**3. Get the code onto the runtime.** `/content` is wiped when the runtime disconnects, so
clone (or `scp`) each session and push or copy back anything worth keeping, or mount Drive
from the notebook.

## Several runtimes at once

Each notebook tab is its own machine (own disk, own GPU) behind its own tunnel, so each gets
its own alias: `colab-dl` for downloads, `colab-exp` for an experiment, and so on. Name the
alias when asking Claude to do something. The one rule that bites: a runtime that was
restarted or replaced has a new hostname, and an alias left pointing at the wrong runtime
fails silently, e.g. a file "missing" because it was written on the other machine.

All runtimes see the same Google Drive once each one mounts it. Keep scratch work on the
runtime's local `/content` (faster, and the README's resume warnings apply to network mounts),
put results on Drive, and give each experiment its own Drive output folder. Two runtimes
writing the same Drive path can leave duplicate or conflicting files.

## Sending scripts from PowerShell

Windows PowerShell 5.1 adds a UTF-8 BOM and CRLF line endings to anything it pipes into
`ssh`, which breaks `bash -s` and `python3 -`. Strip both on the remote side:

```powershell
@'
echo hello from $(hostname)
'@ | ssh colab-dl "tr -d '\r' | sed '1s/^\xEF\xBB\xBF//' | bash -s"
```

For anything longer, write the script to a file and `scp` it over instead.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `websocket: bad handshake` | The tunnel can't reach sshd. Run `!pgrep -a cloudflared; ss -ltnp \| grep sshd; tail /content/cf.log` in Colab. A `connection refused` in the log means the tunnel port doesn't match sshd's: Colab's image sets `Port 2222` / `ListenAddress 127.0.0.1`, which the cell above detects. |
| No `cloudflared` in `pgrep` | It was killed when the cell ended. Re-run the cell (it uses `Popen`). |
| `Permission denied (publickey)` | `PUBKEY` in the cell doesn't match `~/.ssh/colab_ed25519.pub`. |
| `Could not resolve hostname` | The hostname is stale (new session) or was mistyped in `~/.ssh/config`. |
| `winget` never returns | Known on this machine; use the direct download in step 1. |
