// Live view (milestone M7, PLAN.md D17): signals with detected strides, the ensemble's stride
// probability with its spread, what every layer of the CNN sees, and the running step count
// with its conformal interval. Source: the board over MQTT, or a recorded session replayed.
import uPlot from "../vendor/uPlot.esm.js";
import { api } from "./api.js";

const RATE = 100;
const SPAN = 10 * RATE; // samples kept on screen
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

let ws = null;
let st = null; // per-connection state
let charts = [];

function freshState(hello) {
  return {
    hello, seq: [], gyr: [[], [], []], acc: [[], [], []], prob: [], spread: [],
    strides: [], finalSeq: null, count: null, activations: null, seq0: null,
  };
}

export async function mountLive(root) {
  root.innerHTML = `
    <div class="live-view">
      <div class="live-controls">
        <label>Source <select id="lv-source"><option value="board">Board (live, MQTT)</option></select></label>
        <label>Speed <select id="lv-speed"><option>1</option><option>2</option><option>4</option></select></label>
        <button class="primary" id="lv-start">Start</button>
        <span class="level-buttons" id="lv-levels"></span>
        <span class="lv-status" id="lv-status">idle</span>
      </div>
      <div class="live-grid">
        <section class="live-main">
          <div class="chart"><div class="chart-title">Gyro (deg/s), last 10 s. ▲ = final strides</div><div id="lv-imu"></div></div>
          <div class="chart"><div class="chart-title">Stride probability: ensemble mean, band = spread of the 5 CNNs, dashed = threshold, shaded = not final yet</div><div id="lv-prob"></div></div>
          <div class="chart"><div class="chart-title">Inside CNN #1: the latest 2.56 s window flowing through each layer (brighter = stronger activation)</div><canvas id="lv-net"></canvas></div>
        </section>
        <aside class="live-count" id="lv-count"><div class="empty">Press Start</div></aside>
      </div>
    </div>`;
  const sessions = await api.sessions();
  const sel = document.getElementById("lv-source");
  for (const s of sessions) sel.insertAdjacentHTML("beforeend", `<option value="replay:${s.session}">Replay ${s.session} (protocol ${s.protocol})</option>`);
  sel.value = sessions.length ? `replay:${sessions[0].session}` : "board";
  document.getElementById("lv-start").onclick = () => (ws ? stop() : start());
  const linked = location.hash.match(/^#live=(\w+)/);
  if (linked && sessions.some((s) => s.session === linked[1])) {
    sel.value = `replay:${linked[1]}`;
    start();
  }
  window.addEventListener("resize", resize);
}

function setStatus(text) {
  const el = document.getElementById("lv-status");
  if (el) el.textContent = text;
}

function start() {
  const [source, session] = document.getElementById("lv-source").value.split(":");
  const speed = document.getElementById("lv-speed").value;
  const q = source === "replay" ? `source=replay&session=${session}&speed=${speed}` : "source=board";
  ws = new WebSocket(`ws://${location.host}/ws/live?${q}`);
  document.getElementById("lv-start").textContent = "Stop";
  setStatus("connecting…");
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  ws.onclose = () => { ws = null; const b = document.getElementById("lv-start"); if (b) b.textContent = "Start"; };
}

function stop() {
  if (ws) ws.close();
  setStatus("stopped");
}

function onMessage(m) {
  if (m.type === "error") return setStatus(m.message);
  if (m.type === "end") return setStatus("replay finished");
  if (m.type === "hello") {
    st = freshState(m);
    buildCharts();
    buildLevels(m.levels, m.level);
    return setStatus(m.source === "replay" ? `replaying ${m.session}` : "live from the board");
  }
  if (m.calibrating) setStatus("calibrating: stand still for 3 s…");
  else if (st.hello.source === "board") setStatus("live from the board");
  if (m.imu) {
    if (st.seq0 == null) st.seq0 = m.imu.seq[0];
    for (let k = 0; k < m.imu.seq.length; k++) {
      st.seq.push(m.imu.seq[k]);
      for (let a = 0; a < 3; a++) { st.acc[a].push(m.imu.x[a][k]); st.gyr[a].push(m.imu.x[3 + a][k]); }
      st.prob.push(null); st.spread.push(null);
    }
    const extra = st.seq.length - SPAN;
    if (extra > 0) {
      st.seq.splice(0, extra); st.prob.splice(0, extra); st.spread.splice(0, extra);
      for (let a = 0; a < 3; a++) { st.acc[a].splice(0, extra); st.gyr[a].splice(0, extra); }
    }
  }
  if (m.prob) {
    const off = m.prob.start_seq - st.seq[0];
    m.prob.mean.forEach((v, k) => { const i = off + k; if (i >= 0 && i < st.seq.length) { st.prob[i] = v; st.spread[i] = m.prob.spread[k]; } });
  }
  if (m.strides) m.strides.seq.forEach((s, k) => st.strides.push({ seq: s, p: m.strides.p[k] }));
  if (m.final_seq != null) st.finalSeq = m.final_seq;
  if (m.count) st.count = m.count;
  if (m.activations) st.activations = m.activations;
  render();
}

function buildLevels(levels, current) {
  const box = document.getElementById("lv-levels");
  box.innerHTML = levels.map((l) => `<button data-level="${l}" class="${l === current ? "active" : ""}">${Math.round(l * 100)}%</button>`).join("");
  box.onclick = (e) => {
    const l = e.target.dataset.level;
    if (!l || !ws) return;
    ws.send(JSON.stringify({ level: Number(l) }));
    for (const b of box.querySelectorAll("button")) b.classList.toggle("active", b.dataset.level === l);
  };
}

const tOf = (seq) => (seq - st.seq0) / RATE;

function axis() {
  return { stroke: css("--text-muted"), grid: { stroke: css("--border"), width: 1 }, ticks: { stroke: css("--border") }, size: 30 };
}

function buildCharts() {
  for (const u of charts) u.destroy();
  const width = document.getElementById("lv-imu").clientWidth;
  const fmt = (dp) => (u, v) => (v == null ? "–" : v.toFixed(dp));
  const colors = ["--series-S", "--series-F", "--series-L"].map(css);
  const imu = new uPlot({
    width, height: 220, scales: { x: { time: false } }, legend: { live: true },
    axes: [axis(), { ...axis(), size: 50 }],
    series: [{ label: "t (s)", value: fmt(2) }, ...["S", "F", "L"].map((a, i) => ({ label: `${a} gyro`, stroke: colors[i], width: 1, value: fmt(0) }))],
    hooks: { draw: [drawStrides] },
  }, [[], [], [], []], document.getElementById("lv-imu"));
  const prob = new uPlot({
    width, height: 170, scales: { x: { time: false }, y: { range: [0, 1.05] } }, legend: { live: true },
    axes: [axis(), { ...axis(), size: 50 }],
    series: [{ label: "t (s)", value: fmt(2) },
      { label: "mean + spread", stroke: "transparent", value: fmt(2) },
      { label: "mean − spread", stroke: "transparent", value: fmt(2) },
      { label: "p(stride)", stroke: css("--text-primary"), width: 1.5, value: fmt(2) }],
    bands: [{ series: [1, 2], fill: "rgba(240, 239, 236, 0.16)" }],
    hooks: { draw: [drawProbExtras] },
  }, [[], [], [], []], document.getElementById("lv-prob"));
  charts = [imu, prob];
}

function resize() {
  const el = document.getElementById("lv-imu");
  if (!el) return;
  for (const u of charts) u.setSize({ width: el.clientWidth, height: u.height });
  drawNetwork();
}

function drawStrides(u) {
  const { ctx } = u;
  const r = uPlot.pxRatio;
  ctx.save();
  ctx.fillStyle = css("--text-primary");
  for (const s of st.strides) {
    const t = tOf(s.seq);
    if (t < u.scales.x.min || t > u.scales.x.max) continue;
    const x = u.valToPos(t, "x", true);
    const y = u.bbox.top + 4 * r;
    ctx.beginPath(); ctx.moveTo(x, y + 10 * r); ctx.lineTo(x - 5 * r, y); ctx.lineTo(x + 5 * r, y); ctx.closePath(); ctx.fill();
  }
  ctx.restore();
}

function drawProbExtras(u) {
  const { ctx } = u;
  const r = uPlot.pxRatio;
  ctx.save();
  // not final yet: the CNN hasn't seen what comes after these samples
  if (st.finalSeq != null && st.seq.length) {
    const x0 = u.valToPos(tOf(st.finalSeq), "x", true);
    const x1 = u.valToPos(tOf(st.seq[st.seq.length - 1]), "x", true);
    ctx.fillStyle = "rgba(240, 239, 236, 0.06)";
    ctx.fillRect(x0, u.bbox.top, x1 - x0, u.bbox.height);
  }
  const y = u.valToPos(st.hello ? 0.5 : 0.5, "y", true);
  ctx.strokeStyle = css("--text-muted"); ctx.setLineDash([4 * r, 4 * r]); ctx.lineWidth = r;
  ctx.beginPath(); ctx.moveTo(u.bbox.left, y); ctx.lineTo(u.bbox.left + u.bbox.width, y); ctx.stroke();
  ctx.restore();
}

function drawNetwork() {
  const canvas = document.getElementById("lv-net");
  if (!canvas || !st || !st.activations) return;
  const layers = st.activations;
  const names = ["input: accel S F L, gyro S F L", ...layers.map((l, i) =>
    `conv${i + 1}: ${l.channels} channels, dilation ${st.hello.layers[i]?.dilation ?? "?"}`), "output: p(stride)"];
  // the input rows: the last 256 samples, averaged down to 64 time points like the layers
  const n = st.seq.length;
  const win = st.hello.window;
  const input = [...st.acc, ...st.gyr].map((ch) => {
    const xs = ch.slice(Math.max(0, n - win));
    const out = [];
    for (let k = 0; k < 64; k++) out.push(xs.slice(k * 4, k * 4 + 4).reduce((a, b) => a + b, 0) / 4);
    const top = Math.max(...out.map(Math.abs)) || 1;
    return out.map((v) => Math.round(255 * Math.abs(v) / top));
  });
  const probRow = [st.prob.slice(Math.max(0, n - win))].map((xs) => {
    const out = [];
    for (let k = 0; k < 64; k++) out.push(Math.round(255 * ((xs[k * 4] ?? 0) || 0)));
    return out;
  });
  const rows = [input, ...layers.map((l) => l.values), probRow];
  const rowH = 2; // px per channel
  const labelW = 230;
  const width = canvas.parentElement.clientWidth - 2;
  const gap = 14;
  const height = rows.reduce((h, r) => h + Math.max(8, r.length * rowH) + gap, 0) + 4;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = width * ratio; canvas.height = height * ratio;
  canvas.style.width = `${width}px`; canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.font = "11px -apple-system, system-ui, sans-serif";
  ctx.textBaseline = "middle";
  const cellW = (width - labelW - 12) / 64;
  let y = 4;
  rows.forEach((row, r) => {
    const h = Math.max(8, row.length * rowH);
    const ch = h / row.length;
    ctx.fillStyle = css("--text-secondary");
    ctx.fillText(names[r], 8, y + h / 2);
    row.forEach((vals, c) => vals.forEach((v, k) => {
      // one hue, light -> strong: a sequential scale for magnitude
      ctx.fillStyle = `rgba(57, 135, 229, ${(v / 255).toFixed(3)})`;
      ctx.fillRect(labelW + k * cellW, y + c * ch, Math.ceil(cellW), Math.ceil(ch));
    }));
    y += h + gap;
  });
}

function renderCount() {
  const box = document.getElementById("lv-count");
  const c = st.count;
  if (!c) { box.innerHTML = `<div class="empty">Waiting for the first 2.56 s…</div>`; return; }
  const lvl = Math.round(c.level * 100);
  const recent = st.strides.filter((s) => st.seq.length && s.seq > st.seq[st.seq.length - 1] - SPAN);
  const cadence = recent.length * 2 * (60 / (SPAN / RATE));
  const max = Math.max(c.high, 10);
  box.innerHTML = `
    <div class="count-label">steps</div>
    <div class="count-big">${c.n_hat}</div>
    <div class="count-ci">${lvl}% interval: ${Math.floor(c.low)} – ${Math.ceil(c.high)}</div>
    <div class="ci-bar"><div class="ci-range" style="left:${100 * c.low / max}%; width:${100 * (c.high - c.low) / max}%"></div>
      <div class="ci-point" style="left:${100 * c.n_hat / max}%"></div></div>
    <dl class="count-facts">
      <dt>cadence (last 10 s)</dt><dd>${cadence.toFixed(0)} steps/min</dd>
      <dt>bouts</dt><dd>${c.bouts}</dd>
      <dt>σ, from its parts</dt><dd>√(4 × (${c.var_pb.toFixed(2)} + ${c.var_ens.toFixed(2)}) + ${c.var_dbl.toFixed(2)}) = ${c.sigma.toFixed(2)}</dd>
      <dt></dt><dd class="sub">Poisson-binomial + ensemble spread, ×4 for steps, + doubling</dd>
      <dt>interval</dt><dd>${c.n_hat} ± q × σ (q from held-out data)</dd>
    </dl>`;
}

function render() {
  if (!st || !charts.length) return;
  const t = st.seq.map(tOf);
  charts[0].setData([t, ...st.gyr]);
  const hi = st.prob.map((p, i) => (p == null ? null : Math.min(1, p + st.spread[i])));
  const lo = st.prob.map((p, i) => (p == null ? null : Math.max(0, p - st.spread[i])));
  charts[1].setData([t, hi, lo, st.prob]);
  renderCount();
  drawNetwork();
}
