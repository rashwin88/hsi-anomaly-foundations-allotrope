# H100 GPU Monitor — Flask Dashboard

A lightweight Flask server runs on the remote host, serving a live dashboard with real-time scrolling charts for GPU utilization, VRAM, temperature, and power draw. Accessed securely from your local machine via SSH tunnel.

---

## Prerequisites

On the **remote host**:

```bash
pip install flask
```

Chart.js is loaded from a CDN in the browser — no extra pip installs needed.

---

## Running the Server

```bash
# Basic
python monitor.py

# Keep alive after SSH disconnect (nohup)
nohup python monitor.py &

# Keep alive with tmux (lets you reattach)
tmux new -s monitor
python monitor.py
# Detach: Ctrl+B then D
# Reattach: tmux attach -t monitor
```

---

## What the Dashboard Shows

All metrics are pulled from `nvidia-smi` every 2 seconds via a background `fetch()` call — no page reload needed.

### Live Stat Boxes (top of each GPU card)

| Metric | Source |
|---|---|
| GPU Utilization % | `nvidia-smi --query-gpu=utilization.gpu` |
| VRAM Used (MB) | `nvidia-smi --query-gpu=memory.used` |
| Temperature °C | `nvidia-smi --query-gpu=temperature.gpu` |
| Power Draw W | `nvidia-smi --query-gpu=power.draw` |

### Live Progress Bars

| Bar | What it shows |
|---|---|
| GPU Utilization | Green→cyan gradient, updates every 2s |
| VRAM Usage | Red→orange gradient, shows used/total % |

### Real-Time Scrolling Charts (last 2 minutes)

| Chart | Y-axis range |
|---|---|
| GPU Utilization % | 0–100% |
| VRAM Usage % | 0–100% |
| Temperature °C | 0–110°C |
| Power Draw W | 0–700W |

Each chart keeps the last **60 data points** (2 min at 2s interval) and drops the oldest as new ones arrive. All charts update without any page refresh.

### Multi-GPU support

A full card with stat boxes, bars, and all 4 charts is created automatically for each GPU detected — no config needed.

---

## API Endpoint

The server exposes raw JSON at `/api/stats` — the same data the dashboard polls every 2 seconds:

```bash
curl http://localhost:8080/api/stats
```

Returns an array, one object per GPU:

```json
[
  {
    "index": "0",
    "name": "NVIDIA H100 SXM5 80GB",
    "utilization": "73",
    "mem_used": "40960",
    "mem_total": "81920",
    "mem_pct": 50,
    "temp": "72",
    "power": "350.2"
  }
]
```

---

## Connecting from Your Local Machine

### Step 1 — Verify port connectivity

```bash
# Preferred — netcat
nc -zv 103.42.50.245 2271

# Alternatives
telnet 103.42.50.245 2271
nmap -p 2271 103.42.50.245
```

Expected output: `Connection to 103.42.50.245 2271 port [tcp/*] succeeded!`

### Step 2 — SSH in via port 2271

```bash
ssh -p 2271 user57@103.42.50.245
```

### Step 3 — Set up the SSH tunnel

```bash
ssh -p 2271 -L 8080:localhost:8080 user57@103.42.50.245
```

This forwards local `localhost:8080` → through SSH → to `localhost:8080` on the remote host.

Open **http://localhost:8080** in your browser.

### Step 4 — Verify the tunnel

```bash
curl http://localhost:8080              # should return HTML
curl http://localhost:8080/api/stats   # should return JSON
```

---

## Shortcut — `~/.ssh/config`

Add this to `~/.ssh/config` on your local machine to avoid typing the full command each time:

```
Host gpu-server
    HostName 103.42.50.245
    User user57
    Port 2271
    LocalForward 8080 localhost:8080
```

Then just run:

```bash
ssh gpu-server
```

The tunnel is set up automatically on every connection.

---

## Passwordless Login — SSH Key Authentication

`~/.ssh/config` cannot store passwords — it's a deliberate security restriction.
The correct approach is **SSH key authentication**, which eliminates the password prompt entirely and is more secure.

### Step 1 — Generate a key pair (on your local machine, if you don't have one)

```bash
ssh-keygen -t ed25519 -C "raviashwin87@gmail.com"
# Accept the default path (~/.ssh/id_ed25519)
# Set a passphrase or leave empty for fully passwordless login
```

### Step 2 — Copy your public key to the remote host

```bash
ssh-copy-id -p 2271 user57@103.42.50.245
```

This appends your public key to `~/.ssh/authorized_keys` on the remote host.
You'll be asked for your password **one last time**.

### Step 3 — Update `~/.ssh/config` to use the key

```
Host gpu-server
    HostName 103.42.50.245
    User user57
    Port 2271
    LocalForward 8080 localhost:8080
    IdentityFile ~/.ssh/id_ed25519
```

From now on, `ssh gpu-server` connects with no password prompt.

### If `ssh-copy-id` is not available (Windows)

Manually copy the public key instead:

```bash
# Print your public key
cat ~/.ssh/id_ed25519.pub

# On the remote host, paste it into:
echo "YOUR_PUBLIC_KEY" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

---

## VS Code Remote-SSH Setup

VS Code's **Remote-SSH** extension lets you open the remote host as if it were a local folder — edit files, run terminals, and use the integrated terminal, all over SSH.

### Step 1 — Install the extension

In VS Code, open the Extensions panel (`Ctrl+Shift+X` / `Cmd+Shift+X`) and search for:

```
Remote - SSH
```

Install the one published by **Microsoft**.

### Step 2 — Connect to the remote host

1. Open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
2. Type `Remote-SSH: Connect to Host...` and select it
3. Choose **Add New SSH Host** and enter:
   ```
   ssh -p 2271 user57@103.42.50.245
   ```
4. Select `~/.ssh/config` to save it to

VS Code will update your `~/.ssh/config` automatically and connect.

> On subsequent connections, the host will appear directly in the **Remote-SSH: Connect to Host** list.

### Step 3 — Open a folder on the remote host

Once connected, VS Code will prompt you to open a folder.
Navigate to wherever your `monitor.py` lives (e.g. `~/gpu-monitor`) and open it.

### Step 4 — Forward port 8080 in VS Code

So you can access the Flask dashboard without a separate SSH tunnel command:

1. Open the **Ports** panel (bottom panel → **Ports** tab, or `Ctrl+Shift+P` → `Forward a Port`)
2. Click **Forward a Port** and enter `8080`
3. VS Code will tunnel it automatically — open **http://localhost:8080** in your browser

> This replaces the manual `ssh -L 8080:localhost:8080 ...` command — VS Code handles it for you.

### Your final `~/.ssh/config` (with everything)

```
Host gpu-server
    HostName 103.42.50.245
    User user57
    Port 2271
    LocalForward 8080 localhost:8080
    IdentityFile ~/.ssh/id_ed25519
```

---

## Connectivity Checklist

| Step | Command | Expected |
|---|---|---|
| Ping host | `ping 103.42.50.245` | Packets return |
| Port reachable | `nc -zv 103.42.50.245 2271` | `succeeded` |
| SSH in | `ssh -p 2271 user57@103.42.50.245` | Shell prompt |
| Start server | `python monitor.py` | `Running on http://0.0.0.0:8080` |
| Open tunnel | `ssh -p 2271 -L 8080:localhost:8080 user57@103.42.50.245` | — |
| Verify locally | `curl http://localhost:8080` | HTML response |