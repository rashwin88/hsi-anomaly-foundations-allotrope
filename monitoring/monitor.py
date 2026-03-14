from flask import Flask, jsonify, render_template_string
import subprocess
import psutil

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Allotrope Labs — System Monitor</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #f5f6fa; color: #2d3436; padding: 1.5rem; }
    h1 { font-size: 1.4rem; margin-bottom: 1.5rem; letter-spacing: 1px; border-bottom: 2px solid #0984e3; padding-bottom: 0.5rem; color: #2d3436; }
    .section-title { font-size: 1.1rem; font-weight: 600; margin: 1.5rem 0 1rem; color: #2d3436; border-left: 4px solid #0984e3; padding-left: 0.6rem; }
    .card { background: #fff; border: 1px solid #dfe6e9; border-radius: 8px; padding: 1.2rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
    .card-title { font-size: 0.95rem; font-weight: 600; margin-bottom: 1rem; color: #2d3436; }

    .stats-row { display: flex; gap: 1rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
    .stat-box { background: #f5f6fa; border: 1px solid #dfe6e9; border-radius: 6px; padding: 0.7rem 1rem; flex: 1; min-width: 120px; }
    .stat-label { font-size: 0.65rem; color: #636e72; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem; }
    .stat-value { font-size: 1.4rem; color: #2d3436; font-weight: 600; }
    .stat-unit  { font-size: 0.75rem; color: #b2bec3; margin-left: 2px; }

    .bar-row { margin-bottom: 0.8rem; }
    .bar-label { font-size: 0.7rem; color: #636e72; margin-bottom: 3px; display: flex; justify-content: space-between; }
    .bar-bg { background: #dfe6e9; border-radius: 4px; height: 14px; overflow: hidden; }
    .bar { height: 14px; border-radius: 4px; transition: width 0.6s ease; }
    .bar.util  { background: linear-gradient(90deg, #0984e3, #74b9ff); }
    .bar.mem   { background: linear-gradient(90deg, #d63031, #ff7675); }
    .bar.cpu   { background: linear-gradient(90deg, #00b894, #55efc4); }
    .bar.ram   { background: linear-gradient(90deg, #6c5ce7, #a29bfe); }

    .charts-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1.2rem; }
    .chart-box { flex: 1; min-width: 240px; background: #f5f6fa; border: 1px solid #dfe6e9; border-radius: 6px; padding: 0.8rem; }
    .chart-title { font-size: 0.65rem; color: #636e72; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem; }
    canvas { width: 100% !important; }

    .footer { font-size: 0.7rem; color: #b2bec3; margin-top: 2rem; text-align: center; }
    .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #00b894; margin-right: 6px; animation: pulse 1.2s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.2} }
  </style>
</head>
<body>

<h1><span class="dot"></span>Allotrope Labs &mdash; System Monitor</h1>

<div class="section-title">CPU &amp; Memory</div>
<div class="card" id="sys-card">
  <div class="stats-row">
    <div class="stat-box"><div class="stat-label">CPU Usage</div><div class="stat-value" id="sys-cpu">&#8212;<span class="stat-unit">%</span></div></div>
    <div class="stat-box"><div class="stat-label">RAM Used</div><div class="stat-value" id="sys-ram-used">&#8212;<span class="stat-unit">GB</span></div></div>
    <div class="stat-box"><div class="stat-label">RAM Total</div><div class="stat-value" id="sys-ram-total">&#8212;<span class="stat-unit">GB</span></div></div>
    <div class="stat-box"><div class="stat-label">RAM Usage</div><div class="stat-value" id="sys-ram-pct">&#8212;<span class="stat-unit">%</span></div></div>
  </div>
  <div class="bar-row">
    <div class="bar-label"><span>CPU</span><span id="sys-cpu-pct-label">0%</span></div>
    <div class="bar-bg"><div class="bar cpu" id="bar-sys-cpu" style="width:0%"></div></div>
  </div>
  <div class="bar-row">
    <div class="bar-label"><span>RAM</span><span id="sys-ram-pct-label">0%</span></div>
    <div class="bar-bg"><div class="bar ram" id="bar-sys-ram" style="width:0%"></div></div>
  </div>
  <div class="charts-row">
    <div class="chart-box"><div class="chart-title">CPU Usage %</div><canvas id="chart-sys-cpu" height="80"></canvas></div>
    <div class="chart-box"><div class="chart-title">RAM Usage %</div><canvas id="chart-sys-ram" height="80"></canvas></div>
  </div>
</div>

<div class="section-title">GPUs</div>
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
      x: {
        display: true,
        ticks: { color: '#b2bec3', font: { size: 9 }, maxRotation: 0, maxTicksLimit: 6 },
        grid: { color: '#dfe6e9' }
      },
      y: { min: 0, max: yMax, ticks: { color: '#636e72', font: { size: 10 } }, grid: { color: '#dfe6e9' } }
    },
    plugins: { legend: { display: false } }
  }
});

// --- System (CPU/RAM) charts ---
const sysCharts = {
  cpu: new Chart(document.getElementById('chart-sys-cpu'), lineConfig('CPU %', '#00b894', 'rgba(0,184,148,0.1)', 100)),
  ram: new Chart(document.getElementById('chart-sys-ram'), lineConfig('RAM %', '#6c5ce7', 'rgba(108,92,231,0.1)', 100))
};

function updateSysCard(sys) {
  const now = new Date().toLocaleTimeString();
  document.getElementById('sys-cpu').innerHTML = `${sys.cpu_pct}<span class="stat-unit">%</span>`;
  document.getElementById('sys-ram-used').innerHTML = `${sys.ram_used_gb}<span class="stat-unit">GB</span>`;
  document.getElementById('sys-ram-total').innerHTML = `${sys.ram_total_gb}<span class="stat-unit">GB</span>`;
  document.getElementById('sys-ram-pct').innerHTML = `${sys.ram_pct}<span class="stat-unit">%</span>`;

  document.getElementById('bar-sys-cpu').style.width = `${sys.cpu_pct}%`;
  document.getElementById('bar-sys-ram').style.width = `${sys.ram_pct}%`;
  document.getElementById('sys-cpu-pct-label').textContent = `${sys.cpu_pct}%`;
  document.getElementById('sys-ram-pct-label').textContent = `${sys.ram_pct}%`;

  pushPoint(sysCharts.cpu, now, parseFloat(sys.cpu_pct));
  pushPoint(sysCharts.ram, now, parseFloat(sys.ram_pct));
}

// --- GPU cards ---
function initGpuCard(gpu) {
  const idx = gpu.index;
  history[idx] = {};

  const card = document.createElement('div');
  card.className = 'card';
  card.id = `gpu-${idx}`;
  card.innerHTML = `
    <div class="card-title">GPU ${idx} &mdash; ${gpu.name}</div>
    <div class="stats-row">
      <div class="stat-box"><div class="stat-label">GPU Util</div><div class="stat-value" id="util-${idx}">&#8212;<span class="stat-unit">%</span></div></div>
      <div class="stat-box"><div class="stat-label">VRAM Used</div><div class="stat-value" id="mem-${idx}">&#8212;<span class="stat-unit">MB</span></div></div>
      <div class="stat-box"><div class="stat-label">Temperature</div><div class="stat-value" id="temp-${idx}">&#8212;<span class="stat-unit">\u00b0C</span></div></div>
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
      <div class="chart-box"><div class="chart-title">Temperature \u00b0C</div><canvas id="chart-temp-${idx}" height="80"></canvas></div>
      <div class="chart-box"><div class="chart-title">Power Draw W</div><canvas id="chart-power-${idx}" height="80"></canvas></div>
    </div>
  `;
  document.getElementById('dashboard').appendChild(card);

  history[idx].charts = {
    util:  new Chart(document.getElementById(`chart-util-${idx}`),  lineConfig('Util %',   '#0984e3', 'rgba(9,132,227,0.1)',    100)),
    mem:   new Chart(document.getElementById(`chart-mem-${idx}`),   lineConfig('VRAM %',   '#d63031', 'rgba(214,48,49,0.1)',    100)),
    temp:  new Chart(document.getElementById(`chart-temp-${idx}`),  lineConfig('Temp \u00b0C', '#e17055', 'rgba(225,112,85,0.1)',   110)),
    power: new Chart(document.getElementById(`chart-power-${idx}`), lineConfig('Power W',  '#6c5ce7', 'rgba(108,92,231,0.1)',   700))
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
  document.getElementById(`temp-${idx}`).innerHTML  = `${gpu.temp}<span class="stat-unit">\u00b0C</span>`;
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

let gpuInitialized = false;

async function fetchAndUpdate() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();

    // System stats
    if (data.system) {
      updateSysCard(data.system);
    }

    // GPU stats
    if (data.gpus && data.gpus.length > 0 && !data.gpus[0].error) {
      if (!gpuInitialized) {
        data.gpus.forEach(initGpuCard);
        gpuInitialized = true;
      }
      data.gpus.forEach(updateGpuCard);
    }

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


def get_system_stats():
    cpu_pct = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    return dict(
        cpu_pct=cpu_pct,
        ram_used_gb=round(mem.used / (1024 ** 3), 1),
        ram_total_gb=round(mem.total / (1024 ** 3), 1),
        ram_pct=mem.percent
    )


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
    return jsonify(dict(
        system=get_system_stats(),
        gpus=get_gpu_stats()
    ))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
