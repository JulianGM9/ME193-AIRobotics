// Label view (milestone M3, PLAN.md D6-D7).
//
// One event per right-foot stride, at the mid-swing minimum of the L-axis gyro. The server
// proposes candidates with a deliberately loose detector; you accept, reject, nudge or add
// them. A session can only be saved when, in every bout and no-step block still in use,
// |2 x strides - counted steps| <= 1. Saving writes a new label version, never edits one.
import uPlot from "../vendor/uPlot.esm.js";
import { api } from "./api.js";

const RATE = 100; // samples per second
const SNAP_WINDOW = 25; // +/- samples searched when adding or snapping an event
const PICK_PX = 10; // click this close to a marker to select it
const MATCH_SAMPLES = 15; // a candidate this close to a saved stride is the same event

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const state = {
  sessions: [],
  id: null,
  data: null, // server payload for the open session
  t: [],
  events: [], // {seq, status: "accepted" | "rejected" | "proposed"}
  selected: -1,
  segmentOk: {}, // segment number -> false when excluded in this label version
  dirty: false,
  charts: [],
};

// ------------------------------------------------------------------ data helpers

function idxOf(seq) {
  // index of the first sample with seq >= the given seq (samples are sorted by seq)
  const s = state.data.seq;
  let lo = 0, hi = s.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (s[mid] < seq) lo = mid + 1; else hi = mid;
  }
  return lo;
}

const tOf = (seq) => (seq - state.data.seq[0]) / RATE;
const gyroL = () => state.data.gyr[2];

function snap(seq) {
  // the most negative L-gyro sample within +/- SNAP_WINDOW: the mid-swing minimum
  const g = gyroL();
  const c = idxOf(seq);
  let best = c;
  for (let i = Math.max(0, c - SNAP_WINDOW); i <= Math.min(g.length - 1, c + SNAP_WINDOW); i++) {
    if (g[i] < g[best]) best = i;
  }
  return state.data.seq[best];
}

function inCheckedBlock(seq) {
  // inside a bout or no-step block: only these candidates need a decision
  return state.data.meta.segments.some((seg) =>
    (seg.kind === "bout" || seg.kind === "hold") && seq >= seg.start_seq && seq < seg.end_seq);
}

function segmentOk(seg) {
  return state.segmentOk[seg.i] ?? seg.ok ?? true;
}

function segmentChecks() {
  return state.data.meta.segments
    .filter((seg) => seg.kind === "bout" || seg.kind === "hold")
    .map((seg) => {
      const inside = state.events.filter((e) => e.seq >= seg.start_seq && e.seq < seg.end_seq);
      const strides = inside.filter((e) => e.status === "accepted").length;
      const undecided = inside.filter((e) => e.status === "proposed").length;
      const ok = segmentOk(seg);
      let status;
      if (!ok) status = "excluded";
      else if (undecided) status = "undecided";
      else if (seg.steps == null) status = "no-count";
      else status = Math.abs(2 * strides - seg.steps) <= 1 ? "pass" : "fail";
      return { seg, strides, undecided, ok, status };
    });
}

const STATUS_TEXT = {
  pass: "✓ pass", fail: "✗ fail", undecided: "… undecided", excluded: "– excluded", "no-count": "? no count",
};

// ------------------------------------------------------------------ mounting

export async function mountLabel(root) {
  root.innerHTML = `
    <div class="label-view">
      <aside class="session-list"><h2>Sessions</h2><div id="sessions"></div></aside>
      <section class="label-main" id="label-main"><div class="empty">Loading…</div></section>
    </div>`;
  window.addEventListener("keydown", onKey);
  window.addEventListener("resize", resizeCharts);
  window.addEventListener("beforeunload", (e) => { if (state.dirty) e.preventDefault(); });
  await refreshSessions();
  const first = state.sessions.find((s) => s.label_version == null) || state.sessions[0];
  if (first) openSession(first.session);
  else document.getElementById("label-main").innerHTML =
    `<div class="empty">No sessions yet. Record one with <code>scripts/record.py</code>.</div>`;
}

async function refreshSessions() {
  state.sessions = await api.sessions();
  const list = document.getElementById("sessions");
  list.replaceChildren(...state.sessions.map((s) => {
    const b = document.createElement("button");
    b.className = "session-item" + (s.session === state.id ? " active" : "");
    b.innerHTML = `<div>${s.session} · protocol ${s.protocol}</div>
      <div class="sub">${s.label_version ? `labeled v${s.label_version} · ${s.strides} strides` : "not labeled"}</div>`;
    b.onclick = () => openSession(s.session);
    return b;
  }));
}

async function openSession(id) {
  if (state.dirty && !confirm("Discard unsaved label changes?")) return;
  const main = document.getElementById("label-main");
  main.innerHTML = `<div class="empty">Loading ${id}…</div>`;
  const data = await api.session(id);
  Object.assign(state, { id, data, selected: -1, dirty: false, segmentOk: {} });
  state.t = data.seq.map(tOf);

  if (data.labels) {
    // Saved strides are accepted; candidates that don't match one were rejected last time.
    const saved = data.labels.strides;
    state.events = saved.map((seq) => ({ seq, status: "accepted" }));
    for (const c of data.candidates) {
      if (!saved.some((s) => Math.abs(s - c) <= MATCH_SAMPLES)) state.events.push({ seq: c, status: "rejected" });
    }
    for (const [k, v] of Object.entries(data.labels.segment_ok || {})) state.segmentOk[k] = v;
  } else {
    state.events = data.candidates.map((seq) => ({ seq, status: "proposed" }));
  }
  sortEvents();
  render(main);
  refreshSessions();
}

function sortEvents() {
  const sel = state.events[state.selected];
  state.events.sort((a, b) => a.seq - b.seq);
  state.selected = sel ? state.events.indexOf(sel) : -1;
}

// ------------------------------------------------------------------ rendering

function render(main) {
  const m = state.data.meta;
  main.innerHTML = `
    <div class="label-header">
      <span class="title">${m.session}</span>
      <span class="info">protocol ${m.protocol} · ${m.recorded_at} · ${state.data.seq.length} samples · ${state.data.missing} missing
        · ${m.notes ? m.notes + " · " : ""}${state.data.labels ? `editing on top of v${state.data.labels.version}` : "not labeled yet"}</span>
      <span class="spacer"></span>
      <span class="save-msg" id="save-msg"></span>
      <button class="primary" id="save">Save labels</button>
    </div>
    <div class="chart"><div class="chart-title">L-axis gyro (sagittal, deg/s), the labeling channel. Time in seconds. Drag to zoom, double-click to reset</div><div id="c-main"></div></div>
    <div class="help">
      Click a marker to select it ·
      <kbd>a</kbd>/<kbd>Enter</kbd> accept · <kbd>r</kbd>/<kbd>⌫</kbd> reject · <kbd>←</kbd><kbd>→</kbd> nudge (<kbd>⇧</kbd> ×10) ·
      <kbd>s</kbd> snap to minimum · <kbd>n</kbd>/<kbd>⇧n</kbd> next/previous undecided · <kbd>z</kbd> zoom to it ·
      <kbd>⌥</kbd>-click add a stride
      <span class="marker-key"><span>▲ accepted</span><span>○ undecided</span><span>× rejected</span>
        <span>dim ○ outside any block (optional)</span></span>
    </div>
    <div class="chart"><div class="chart-title">Gyro, all axes (deg/s)</div><div id="c-gyr"></div></div>
    <div class="chart"><div class="chart-title">Accelerometer (g)</div><div id="c-acc"></div></div>
    <div class="panel"><h2>Blocks</h2><table class="segments" id="segments"></table></div>`;
  document.getElementById("save").onclick = save;
  buildCharts();
  renderTable();
}

function axisStyle() {
  return { stroke: css("--text-muted"), grid: { stroke: css("--border"), width: 1 }, ticks: { stroke: css("--border") } };
}

function buildCharts() {
  for (const u of state.charts) u.destroy();
  const width = document.getElementById("c-main").clientWidth;
  const d = state.data;
  const colors = ["--series-S", "--series-F", "--series-L"].map(css);
  const fmt = (dp) => (u, v) => (v == null ? "–" : v.toFixed(dp));
  let syncing = false;
  const syncX = (u, key) => {
    if (key !== "x" || syncing) return;
    syncing = true;
    for (const other of state.charts) {
      if (other !== u) other.setScale("x", { min: u.scales.x.min, max: u.scales.x.max });
    }
    syncing = false;
  };
  const common = (height) => ({
    width, height,
    cursor: { sync: { key: "label" }, drag: { x: true, y: false } },
    scales: { x: { time: false } },
    axes: [{ ...axisStyle(), size: 30 }, { ...axisStyle(), size: 56 }],
    legend: { live: true },
  });

  const main = new uPlot({
    ...common(320),
    series: [{ label: "t (s)", value: fmt(2) }, { label: "L gyro", stroke: colors[2], width: 1.25, value: fmt(1) }],
    hooks: { drawClear: [drawBands], draw: [drawSegmentLabels, drawMarkers], setScale: [syncX] },
  }, [state.t, d.gyr[2]], document.getElementById("c-main"));

  const axes3 = (unit, dp) => ["S", "F", "L"].map((a, i) => ({ label: `${a} ${unit}`, stroke: colors[i], width: 1, value: fmt(dp) }));
  const gyr = new uPlot({
    ...common(220),
    series: [{ label: "t (s)", value: fmt(2) }, ...axes3("gyro", 1)],
    hooks: { drawClear: [drawBands], draw: [drawEventLines], setScale: [syncX] },
  }, [state.t, ...d.gyr], document.getElementById("c-gyr"));

  const acc = new uPlot({
    ...common(220),
    series: [{ label: "t (s)", value: fmt(2) }, ...axes3("accel", 2)],
    hooks: { drawClear: [drawBands], draw: [drawEventLines], setScale: [syncX] },
  }, [state.t, ...d.acc], document.getElementById("c-acc"));

  state.charts = [main, gyr, acc];
  attachMouse(main);
}

function resizeCharts() {
  const el = document.getElementById("c-main");
  if (!el) return;
  for (const u of state.charts) u.setSize({ width: el.clientWidth, height: u.height });
}

function redraw() {
  for (const u of state.charts) u.redraw(false);
}

function drawBands(u) {
  // Segment backgrounds, under the signal: bouts darker than no-step blocks, excluded in red
  const { ctx } = u;
  const { top, height } = u.bbox;
  ctx.save();
  for (const seg of state.data.meta.segments) {
    if (seg.kind === "transition" || seg.kind === "calibrate") continue;
    const x0 = u.valToPos(tOf(seg.start_seq), "x", true);
    const x1 = u.valToPos(tOf(seg.end_seq), "x", true);
    ctx.fillStyle = !segmentOk(seg) ? css("--band-excluded") : seg.kind === "bout" ? css("--band-bout") : css("--band-hold");
    ctx.fillRect(x0, top, x1 - x0, height);
  }
  ctx.restore();
}

function drawSegmentLabels(u) {
  const { ctx } = u;
  const checks = new Map(segmentChecks().map((c) => [c.seg.i, c]));
  ctx.save();
  ctx.beginPath();
  ctx.rect(u.bbox.left, u.bbox.top, u.bbox.width, u.bbox.height);
  ctx.clip();
  ctx.font = `${11 * uPlot.pxRatio}px -apple-system, system-ui, sans-serif`;
  ctx.textAlign = "left"; // uPlot leaves the context right-aligned after drawing its axes
  ctx.textBaseline = "top";
  for (const seg of state.data.meta.segments) {
    const c = checks.get(seg.i);
    if (!c) continue;
    const x = u.valToPos(tOf(seg.start_seq), "x", true) + 4 * uPlot.pxRatio;
    const y = u.bbox.top + 4 * uPlot.pxRatio;
    ctx.fillStyle = css("--text-secondary");
    ctx.fillText(`#${seg.i} ${seg.label}`, x, y);
    const steps = seg.steps == null ? "?" : seg.steps;
    ctx.fillText(`${steps} steps · ${c.strides} strides`, x, y + 14 * uPlot.pxRatio);
    // status color only on the line that carries its icon + word
    ctx.fillStyle = c.status === "pass" ? css("--good") : c.status === "fail" ? css("--critical") : css("--text-muted");
    ctx.fillText(STATUS_TEXT[c.status], x, y + 28 * uPlot.pxRatio);
  }
  ctx.restore();
}

function drawMarkers(u) {
  const { ctx } = u;
  const r = uPlot.pxRatio;
  const g = gyroL();
  const [xmin, xmax] = [u.scales.x.min, u.scales.x.max];
  ctx.save();
  ctx.beginPath();
  ctx.rect(u.bbox.left, u.bbox.top, u.bbox.width, u.bbox.height);
  ctx.clip();
  state.events.forEach((e, k) => {
    const t = tOf(e.seq);
    if (t < xmin || t > xmax) return;
    const x = u.valToPos(t, "x", true);
    const y = u.valToPos(g[idxOf(e.seq)], "y", true);
    ctx.lineWidth = 1.5 * r;
    if (e.status === "accepted") {
      // filled triangle just under the dip, pointing up at it
      ctx.fillStyle = css("--text-primary");
      ctx.beginPath();
      ctx.moveTo(x, y + 3 * r);
      ctx.lineTo(x - 5 * r, y + 12 * r);
      ctx.lineTo(x + 5 * r, y + 12 * r);
      ctx.closePath();
      ctx.fill();
    } else if (e.status === "proposed") {
      ctx.strokeStyle = inCheckedBlock(e.seq) ? css("--text-secondary") : css("--text-muted");
      ctx.beginPath();
      ctx.arc(x, y, 5 * r, 0, 2 * Math.PI);
      ctx.stroke();
    } else {
      ctx.strokeStyle = css("--text-muted");
      ctx.beginPath();
      ctx.moveTo(x - 4 * r, y - 4 * r); ctx.lineTo(x + 4 * r, y + 4 * r);
      ctx.moveTo(x + 4 * r, y - 4 * r); ctx.lineTo(x - 4 * r, y + 4 * r);
      ctx.stroke();
    }
    if (k === state.selected) {
      ctx.strokeStyle = css("--text-primary");
      ctx.lineWidth = 2 * r;
      ctx.beginPath();
      ctx.arc(x, y, 10 * r, 0, 2 * Math.PI);
      ctx.stroke();
    }
  });
  ctx.restore();
}

function drawEventLines(u) {
  // faint vertical lines at accepted strides, to line the other channels up with them
  const { ctx } = u;
  ctx.save();
  ctx.strokeStyle = css("--text-muted");
  ctx.globalAlpha = 0.5;
  ctx.setLineDash([3 * uPlot.pxRatio, 3 * uPlot.pxRatio]);
  ctx.lineWidth = uPlot.pxRatio;
  for (const e of state.events) {
    if (e.status !== "accepted") continue;
    const t = tOf(e.seq);
    if (t < u.scales.x.min || t > u.scales.x.max) continue;
    const x = u.valToPos(t, "x", true);
    ctx.beginPath();
    ctx.moveTo(x, u.bbox.top);
    ctx.lineTo(x, u.bbox.top + u.bbox.height);
    ctx.stroke();
  }
  ctx.restore();
}

function renderTable() {
  const rows = segmentChecks().map(({ seg, strides, undecided, ok, status }) => `
    <tr class="${ok ? "" : "excluded"}">
      <td>#${seg.i}</td><td>${seg.label}</td><td>${seg.kind === "bout" ? "bout" : "no-step"}</td>
      <td>${seg.steps ?? "?"}</td><td>${strides}${undecided ? ` (+${undecided} undecided)` : ""}</td><td>${2 * strides}</td>
      <td class="status ${status}">${STATUS_TEXT[status]}</td>
      <td><label><input type="checkbox" data-use="${seg.i}" ${ok ? "checked" : ""}> use</label></td>
      <td>
        <button data-act="accept" data-i="${seg.i}">Accept all</button>
        <button data-act="reject" data-i="${seg.i}">Reject all</button>
        <button data-act="show" data-i="${seg.i}">Show</button>
      </td>
    </tr>`).join("");
  const table = document.getElementById("segments");
  table.innerHTML = `<tr><th></th><th>block</th><th>kind</th><th>counted steps</th><th>strides</th>
    <th>2 × strides</th><th>check</th><th></th><th></th></tr>${rows}`;
  table.onclick = onTableClick;
  table.onchange = (e) => {
    const i = e.target.dataset.use;
    if (i) { state.segmentOk[i] = e.target.checked; changed(); }
  };

  const checks = segmentChecks();
  const ready = checks.every((c) => c.status === "pass" || c.status === "excluded");
  const save = document.getElementById("save");
  save.disabled = !ready;
  save.title = ready ? "" : "Every block in use must pass the count check, with no undecided candidates";
}

function onTableClick(e) {
  const act = e.target.dataset.act;
  if (!act) return;
  const seg = state.data.meta.segments.find((s) => s.i === Number(e.target.dataset.i));
  const inside = (ev) => ev.seq >= seg.start_seq && ev.seq < seg.end_seq;
  if (act === "show") return zoomTo(tOf(seg.start_seq) - 2, tOf(seg.end_seq) + 2);
  for (const ev of state.events) {
    if (inside(ev) && ev.status === "proposed") ev.status = act === "accept" ? "accepted" : "rejected";
  }
  changed();
}

function changed() {
  state.dirty = true;
  document.getElementById("save-msg").textContent = "unsaved changes";
  renderTable();
  redraw();
}

// ------------------------------------------------------------------ interaction

function zoomTo(min, max) {
  state.charts[0].setScale("x", { min, max }); // syncX hook copies it to the other charts
}

function keepInView(k) {
  const u = state.charts[0];
  const t = tOf(state.events[k].seq);
  const { min, max } = u.scales.x;
  if (t < min || t > max) {
    const half = (max - min) / 2;
    zoomTo(t - half, t + half);
  }
}

function select(k) {
  state.selected = k;
  if (k >= 0) keepInView(k);
  redraw();
}

function attachMouse(u) {
  let down = null;
  u.over.addEventListener("mousedown", (e) => { down = [e.clientX, e.clientY]; });
  u.over.addEventListener("click", (e) => {
    if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 3) return; // that was a zoom drag
    const t = u.posToVal(e.offsetX, "x");
    if (e.altKey) {
      const seq = snap(state.data.seq[0] + Math.round(t * RATE));
      if (!state.events.some((ev) => ev.seq === seq)) {
        state.events.push({ seq, status: "accepted" });
        state.selected = state.events.length - 1;
        sortEvents();
        changed();
      }
      return;
    }
    let best = -1, bestPx = PICK_PX;
    state.events.forEach((ev, k) => {
      const px = Math.abs(u.valToPos(tOf(ev.seq), "x") - e.offsetX);
      if (px < bestPx) { best = k; bestPx = px; }
    });
    select(best);
  });
}

function nextUndecided(dir) {
  const n = state.events.length;
  for (let step = 1; step <= n; step++) {
    const k = (((state.selected < 0 ? (dir > 0 ? -1 : 0) : state.selected) + dir * step) % n + n) % n;
    if (state.events[k].status === "proposed" && inCheckedBlock(state.events[k].seq)) return k;
  }
  return -1;
}

function onKey(e) {
  if (!state.data || !document.getElementById("c-main")) return;
  if (["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
  const ev = state.events[state.selected];
  const key = e.key;
  if (key === "n" || key === "N") {
    select(nextUndecided(e.shiftKey ? -1 : 1));
  } else if (key === "Escape") {
    select(-1);
  } else if (!ev) {
    return;
  } else if (key === "a" || key === "Enter") {
    ev.status = "accepted";
    changed();
    const next = nextUndecided(1);
    if (next >= 0) select(next);
  } else if (key === "r" || key === "Backspace" || key === "Delete") {
    ev.status = "rejected";
    changed();
    const next = nextUndecided(1);
    if (next >= 0) select(next);
  } else if (key === "ArrowLeft" || key === "ArrowRight") {
    ev.seq += (key === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 10 : 1);
    sortEvents();
    changed();
  } else if (key === "s") {
    ev.seq = snap(ev.seq);
    sortEvents();
    changed();
  } else if (key === "z") {
    const t = tOf(ev.seq);
    zoomTo(t - 4, t + 4);
  } else {
    return;
  }
  e.preventDefault();
}

async function save() {
  const msg = document.getElementById("save-msg");
  const strides = state.events.filter((e) => e.status === "accepted").map((e) => e.seq).sort((a, b) => a - b);
  try {
    const saved = await api.saveLabels(state.id, strides, state.segmentOk);
    state.dirty = false;
    state.data.labels = saved;
    msg.textContent = `saved v${saved.version} · ${saved.strides.length} strides`;
    msg.style.color = css("--good");
    refreshSessions();
  } catch (err) {
    msg.textContent = err.message;
    msg.style.color = css("--critical");
  }
}
