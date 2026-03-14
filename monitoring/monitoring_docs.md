# monitor.py — How It Works

A single-file Flask app that shows live CPU, RAM, and GPU stats in your browser. It reads system metrics via `psutil` and GPU metrics via `nvidia-smi`, then sends everything to a web dashboard that updates every 2 seconds.

---

## High-Level Flow

```
Browser (your laptop)
    |
    |  HTTP request every 2s
    v
Flask server (remote VM, port 8080)
    |
    |  psutil (CPU/RAM) + subprocess call (nvidia-smi)
    v
System hardware
    |
    |  numbers
    v
Flask bundles it -> JSON -> back to browser
    |
    v
Chart.js renders live charts
```

---

## Backend (Python)

### `get_system_stats()`

Uses `psutil` to read CPU and RAM usage. Returns a dict with:

| Field | What it is |
|---|---|
| `cpu_pct` | Overall CPU usage as a percentage (0-100) |
| `ram_used_gb` | RAM currently in use, in GB (rounded to 1 decimal) |
| `ram_total_gb` | Total RAM available, in GB (rounded to 1 decimal) |
| `ram_pct` | RAM usage as a percentage |

### `get_gpu_stats()`

Runs `nvidia-smi` as a subprocess and asks for these fields per GPU:

| Field | What it is |
|---|---|
| `index` | GPU number (0, 1, 2...) |
| `name` | GPU model name (e.g. "NVIDIA H100 SXM5 80GB") |
| `utilization.gpu` | How busy the GPU cores are (0-100%) |
| `memory.used` | VRAM currently in use (MB) |
| `memory.total` | Total VRAM available (MB) |
| `temperature.gpu` | GPU temperature in Celsius |
| `power.draw` | Current power draw in Watts |

The `--format=csv,noheader,nounits` flag tells nvidia-smi to return plain numbers in CSV format (no headers, no "MiB" or "%" suffixes).

The function parses each CSV line, calculates `mem_pct` (VRAM used as a percentage of total), and returns a list of dicts — one per GPU.

**Error handling:**
- If `nvidia-smi` is not installed, returns an error message instead of crashing.
- If `nvidia-smi` fails (e.g. driver issue), catches the error and returns the message.
- If power reports `[N/A]` (some GPUs do this), defaults to `"0"` so the frontend doesn't break.
- If memory values are unexpected, `mem_pct` defaults to `0`.

### Routes

| Route | What it does |
|---|---|
| `GET /` | Serves the HTML dashboard (the entire page is a single string in the Python file) |
| `GET /api/stats` | Returns system + GPU stats as JSON — this is what the dashboard polls every 2 seconds |

### API response format

`/api/stats` returns:

```json
{
  "system": {
    "cpu_pct": 23.5,
    "ram_used_gb": 12.3,
    "ram_total_gb": 64.0,
    "ram_pct": 19.2
  },
  "gpus": [
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
}
```

### Server startup

```python
app.run(host="0.0.0.0", port=8080)
```

- `0.0.0.0` means it listens on all network interfaces (needed so you can reach it via SSH tunnel).
- Port `8080` is where the dashboard lives.

---

## Frontend (HTML + JavaScript)

The entire frontend is a single HTML string embedded in the Python file (the `HTML` variable). No separate template files.

### Page layout

The page has two sections:

1. **CPU & Memory** — a fixed card at the top with stat boxes, progress bars, and 2 live charts (CPU %, RAM %).
2. **GPUs** — one card per GPU, created dynamically on first data fetch.

### On page load

1. Calls `fetchAndUpdate()` immediately.
2. Sets up `setInterval` to call it again every 2000ms (2 seconds).

### `fetchAndUpdate()`

1. Fetches `/api/stats` (the JSON endpoint).
2. Updates the CPU/RAM card via `updateSysCard()`.
3. **First time only:** creates a card for each GPU by calling `initGpuCard()`.
4. **Every time:** updates GPU numbers, progress bars, and charts by calling `updateGpuCard()`.

### `updateSysCard(sys)`

Updates the CPU & Memory card:

1. Sets CPU %, RAM used (GB), RAM total (GB), RAM % in the stat boxes.
2. Updates the CPU and RAM progress bar widths.
3. Pushes new data points onto the CPU and RAM charts.

### `initGpuCard(gpu)`

Builds the DOM for one GPU card:

- **4 stat boxes** at the top: utilization %, VRAM used (MB), temperature (C), power (W).
- **2 progress bars**: GPU utilization (blue gradient) and VRAM usage (red gradient).
- **4 live charts** (Chart.js line charts): utilization, VRAM %, temperature, power.

Each chart is configured with `lineConfig()` which sets the color, Y-axis range, and disables animations for smooth real-time updates.

### `updateGpuCard(gpu)`

Called every 2 seconds per GPU:

1. Updates the stat box numbers.
2. Sets the progress bar widths.
3. Pushes a new data point onto each chart via `pushPoint()`.

### `pushPoint(chart, label, value)`

Adds a data point to a Chart.js chart. Keeps a maximum of 60 points (the `MAX_POINTS` constant). That's 60 points x 2 seconds = **2 minutes of history** visible on each chart. Oldest points are dropped as new ones arrive.

### X-axis timestamps

All charts show timestamps on the x-axis (up to 6 time labels visible at once, no rotation). Labels are generated using `toLocaleTimeString()` in the browser.

---

## Styling

- Light theme — white cards (`#fff`) on light gray background (`#f5f6fa`), dark text.
- System font stack (`-apple-system`, `Segoe UI`, `Roboto`, etc.).
- Pulsing green dot next to the title indicates the page is live.
- Color scheme:
  - CPU: green (`#00b894`)
  - RAM: purple (`#6c5ce7`)
  - GPU utilization: blue (`#0984e3`)
  - VRAM: red (`#d63031`)
  - Temperature: orange (`#e17055`)
  - Power: purple (`#6c5ce7`)
- Cards auto-wrap for multi-GPU setups — one card per GPU, no config needed.
- Chart.js is loaded from a CDN (`cdn.jsdelivr.net`), so the remote host needs internet access (or you can bundle it).

---

## Dependencies

| Dependency | Where | Purpose |
|---|---|---|
| Flask | pip (remote host) | Web server |
| psutil | pip (remote host) | CPU and RAM metrics |
| nvidia-smi | NVIDIA driver (remote host) | GPU metrics |
| Chart.js | CDN (loaded in browser) | Renders live charts |

Install on the remote host:

```bash
pip install flask psutil
```

No database. No config files.
