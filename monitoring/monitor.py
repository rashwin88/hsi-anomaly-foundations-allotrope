from flask import Flask, jsonify, render_template_string
import subprocess

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Allotrope Labs — GPU Monitor</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Courier New', monospace; background: #0d0d0d; color: #00ff88; padding: 1.5rem; }
    h1 { font-size: 1.4rem; margin-bottom: 1.5rem; letter-spacing: 2px; border-bottom: 1px solid #00ff8844; padding-bottom: 0.5rem; }
    .gpu-card { background: #111; border: 1px solid #00ff8833; border-radius: 8px; padding: 1.2rem; margin-bottom: 2rem; }
    .gpu-title { font-size: 1rem; font-weight: bold; margin-bottom: 1rem; color: #00ff88; }

    .stats-row { display: flex; gap: 1rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
    .stat-box { background: #0d0d0d; border: 1px solid #00ff8822; border-radius: 6px; padding: 0.7rem 1rem; flex: 1; min-width: 120px; }
    .stat-label { font-size: 0.65rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem; }
    .stat-value { font-size: 1.4rem; color: #00ff88; }
    .stat-unit  { font-size: 0.75rem; color: #555; margin-left: 2px; }

    .bar-row { margin-bottom: 0.8rem; }
    .bar-label { font-size: 0.7rem; color: #888; margin-bottom: 3px; display: flex; justify-content: space-between; }
    .bar-bg { background: #1a1a1a; border-radius: 4px; height: 14px; overflow: hidden; }
    .bar { height: 14px; border-radius: 4px; transition: width 0.6s ease; }
    .bar.util  { background: linear-gradient(90deg, #00ff88, #00ccff); }
    .bar.mem   { background: linear-gradient(90deg, #ff6b6b, #ff9f43); }

    .charts-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1.2rem; }
    .chart-box { flex: 1; min-width: 240px; background: #0d0d0d; border: 1px solid #00ff8822; border-radius: 6px; padding: 0.8rem; }
    .chart-title { font-size: 0.65rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem; }
    canvas { width: 100% !important; }

    .footer { font-size: 0.65rem; color: #333; margin-top: 2rem; text-align: center; }
    .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #00ff88; margin-right: 6px; animation: pulse 1.2s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.2} }
  </style>
</head>
<body>

<h1><span class="dot"></span>Allotrope Labs &mdash; GPU Monitor</h1>
<div id="dashboard"></div>
<div class="footer">Allotrope Labs &bull; Refreshing every 2s &mdash; <span id="last-update"></span></div>

<script>
const MAX_POINTS = 60;
const history = {};

const lineConfig = (label, color1, color2, yMax) => ({
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label,
      data: [],
      borderColor: color1,
      backgroundColor: color2,
      borderWidth: 1.5,
      pointRadius: 0,
      fill: true,
      tension: 0.3
    }]
  },
  options: {
    animation: false,
    responsive: true,
    scales: {
      x: { display: false },
      y: { min: 0, max: yMax, ticks: { color: '#555', font: { size: 10 } }, grid: { color: '#1a1a1a' } }
    },
    plugins: { legend: { display: false } }
  }
});

function initGpuCard(gpu) {
  const idx = gpu.index;
  history[idx] = {};

  const card = document.createElement('div');
  card.className = 'gpu-card';
  card.id = `gpu-${idx}`;
  card.innerHTML = `
    <div class="gpu-title">GPU ${idx} &mdash; ${gpu.name}</div>
    <div class="stats-row">
      <div class="stat-box"><div class="stat-label">GPU Util</div><div class="stat-value" id="util-${idx}">&#8212;<span class="stat-unit">%</span></div></div>
      <div class="stat-box"><div class="stat-label">VRAM Used</div><div class="stat-value" id="mem-${idx}">&#8212;<span class="stat-unit">MB</span></div></div>
      <div class="stat-box"><div class="stat-label">Temperature</div><div class="stat-value" id="temp-${idx}">&#8212;<span class="stat-unit">°C</span></div></div>
      <div class="stat-box"><div class="stat-label">Power Draw</div><div class="stat-value" id="power-${idx}">&#8212;<span class="stat-unit">W</span></div></div>
    </div>
    <div class="bar-row">
      <div class="bar-label"><span>GPU Utilization</span><span id="util-pct-${idx}">0%</span></div>
      <div class="bar-bg"><div class="bar util" id="bar-util-${idx}" style="width:0%"></div></div>
    </div>
    <div class="bar-row">
      <div class="bar-label"><span>VRAM</span><span id="mem-pct-${idx}">0%</span></div>
      <div class="bar-bg"><div class="bar mem" id="bar-mem-${idx}" style="width:0%"></div></div>
    </div>
    <div class="charts-row">
      <div class="chart-box"><div class="chart-title">GPU Utilization %</div><canvas id="chart-util-${idx}" height="80"></canvas></div>
      <div class="chart-box"><div class="chart-title">VRAM Usage %</div><canvas id="chart-mem-${idx}" height="80"></canvas></div>
      <div class="chart-box"><div class="chart-title">Temperature °C</div><canvas id="chart-temp-${idx}" height="80"></canvas></div>
      <div class="chart-box"><div class="chart-title">Power Draw W</div><canvas id="chart-power-${idx}" height="80"></canvas></div>
    </div>
  `;
  document.getElementById('dashboard').appendChild(card);

  history[idx].charts = {
    util:  new Chart(document.getElementById(`chart-util-${idx}`),  lineConfig('Util %',   '#00ff88', 'rgba(0,255,136,0.1)',   100)),
    mem:   new Chart(document.getElementById(`chart-mem-${idx}`),   lineConfig('VRAM %',   '#ff6b6b', 'rgba(255,107,107,0.1)', 100)),
    temp:  new Chart(document.getElementById(`chart-temp-${idx}`),  lineConfig('Temp °C',  '#00ccff', 'rgba(0,204,255,0.1)',   110)),
    power: new Chart(document.getElementById(`chart-power-${idx}`), lineConfig('Power W',  '#a29bfe', 'rgba(162,155,254,0.1)', 700))
  };
}

function pushPoint(chart, label, value) {
  chart.data.labels.push(label);
  chart.data.datasets[0].data.push(value);
  if (chart.data.labels.length > MAX_POINTS) {
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
  }
  chart.update('none');
}

function updateGpuCard(gpu) {
  const idx = gpu.index;
  const now = new Date().toLocaleTimeString();

  document.getElementById(`util-${idx}`).innerHTML  = `${gpu.utilization}<span class="stat-unit">%</span>`;
  document.getElementById(`mem-${idx}`).innerHTML   = `${gpu.mem_used}<span class="stat-unit">MB</span>`;
  document.getElementById(`temp-${idx}`).innerHTML  = `${gpu.temp}<span class="stat-unit">°C</span>`;
  document.getElementById(`power-${idx}`).innerHTML = `${gpu.power}<span class="stat-unit">W</span>`;

  document.getElementById(`bar-util-${idx}`).style.width = `${gpu.utilization}%`;
  document.getElementById(`bar-mem-${idx}`).style.width  = `${gpu.mem_pct}%`;
  document.getElementById(`util-pct-${idx}`).textContent = `${gpu.utilization}%`;
  document.getElementById(`mem-pct-${idx}`).textContent  = `${gpu.mem_pct}%`;

  const c = history[idx].charts;
  pushPoint(c.util,  now, parseFloat(gpu.utilization));
  pushPoint(c.mem,   now, gpu.mem_pct);
  pushPoint(c.temp,  now, parseFloat(gpu.temp));
  pushPoint(c.power, now, parseFloat(gpu.power));
}

let initialized = false;

async function fetchAndUpdate() {
  try {
    const res = await fetch('/api/stats');
    const gpus = await res.json();
    if (!initialized) {
      gpus.forEach(initGpuCard);
      initialized = true;
    }
    gpus.forEach(updateGpuCard);
    document.getElementById('last-update').textContent = 'Last update: ' + new Date().toLocaleTimeString();
  } catch (e) {
    console.error('Fetch failed:', e);
  }
}

fetchAndUpdate();
setInterval(fetchAndUpdate, 2000);
</script>
</body>
</html>
"""

def get_gpu_stats():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits"
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
    except FileNotFoundError:
        return [{"error": "nvidia-smi not found — is the NVIDIA driver installed?"}]
    except subprocess.CalledProcessError as e:
        return [{"error": f"nvidia-smi failed: {e.output.decode().strip()}"}]

    gpus = []
    for line in out.strip().split("\n"):
        idx, name, util, mem_used, mem_total, temp, power = [x.strip() for x in line.split(",")]
        try:
            mem_pct = round(int(mem_used) / int(mem_total) * 100)
        except (ValueError, ZeroDivisionError):
            mem_pct = 0
        power = power if power not in ("[N/A]", "[Not Supported]") else "0"
        gpus.append(dict(
            index=idx,
            name=name,
            utilization=util,
            mem_used=mem_used,
            mem_total=mem_total,
            mem_pct=mem_pct,
            temp=temp,
            power=power
        ))
    return gpus

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/stats")
def stats():
    return jsonify(get_gpu_stats())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)