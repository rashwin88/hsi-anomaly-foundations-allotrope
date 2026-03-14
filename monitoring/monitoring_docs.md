# monitor.py — How It Works

A single-file Flask app that shows live GPU stats in your browser. It talks to `nvidia-smi` on the server and sends the numbers to a web dashboard that updates every 2 seconds.

---

## High-Level Flow

```
Browser (your laptop)
    |
    |  HTTP request every 2s
    v
Flask server (remote VM, port 8080)
    |
    |  subprocess call
    v
nvidia-smi (reads GPU hardware)
    |
    |  CSV output
    v
Flask parses it -> JSON -> back to browser
    |
    v
Chart.js renders live charts
```

---

## Backend (Python)

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
| `GET /api/stats` | Returns the GPU stats as JSON — this is what the dashboard polls every 2 seconds |

### Server startup

```python
app.run(host="0.0.0.0", port=8080)
```

- `0.0.0.0` means it listens on all network interfaces (needed so you can reach it via SSH tunnel).
- Port `8080` is where the dashboard lives.

---

## Frontend (HTML + JavaScript)

The entire frontend is a single HTML string embedded in the Python file (the `HTML` variable). No separate template files.

### On page load

1. Calls `fetchAndUpdate()` immediately.
2. Sets up `setInterval` to call it again every 2000ms (2 seconds).

### `fetchAndUpdate()`

1. Fetches `/api/stats` (the JSON endpoint).
2. **First time only:** creates a card for each GPU by calling `initGpuCard()`.
3. **Every time:** updates the numbers, progress bars, and charts by calling `updateGpuCard()`.

### `initGpuCard(gpu)`

Builds the DOM for one GPU card:

- **4 stat boxes** at the top: utilization %, VRAM used (MB), temperature (C), power (W).
- **2 progress bars**: GPU utilization (green-to-cyan gradient) and VRAM usage (red-to-orange gradient).
- **4 live charts** (Chart.js line charts): utilization, VRAM %, temperature, power.

Each chart is configured with `lineConfig()` which sets the color, Y-axis range, and disables animations for smooth real-time updates.

### `updateGpuCard(gpu)`

Called every 2 seconds per GPU:

1. Updates the stat box numbers.
2. Sets the progress bar widths.
3. Pushes a new data point onto each chart via `pushPoint()`.

### `pushPoint(chart, label, value)`

Adds a data point to a Chart.js chart. Keeps a maximum of 60 points (the `MAX_POINTS` constant). That's 60 points x 2 seconds = **2 minutes of history** visible on each chart. Oldest points are dropped as new ones arrive.

---

## Styling

- Dark theme (`#0d0d0d` background, `#00ff88` green accent).
- Monospace font (`Courier New`).
- Pulsing green dot next to the title indicates the page is live.
- Cards auto-wrap for multi-GPU setups — one card per GPU, no config needed.
- Chart.js is loaded from a CDN (`cdn.jsdelivr.net`), so the remote host needs internet access (or you can bundle it).

---

## Dependencies

| Dependency | Where | Purpose |
|---|---|---|
| Flask | pip (remote host) | Web server |
| nvidia-smi | NVIDIA driver (remote host) | Reads GPU metrics |
| Chart.js | CDN (loaded in browser) | Renders live charts |

No other pip packages needed. No database. No config files.
