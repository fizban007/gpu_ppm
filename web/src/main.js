// Interactive pulsar pulse-profile visualizer.
//
// Interaction:
//   - drag on sphere       → change observer phase
//   - wheel on sphere      → zoom camera
//   - drag curve-panel bar → reposition the panel
//   - sliders / dropdowns  → mutate hotspot/observer params, recompute + redraw
//   - preset picker        → load one of the OS1 cases into the sliders

import { createPulseEngine, OS1_DEFAULTS } from "./compute.js";
import {
  createSphereRenderer,
  CAM_DISTANCE_DEFAULT,
  CAM_DISTANCE_MIN,
  CAM_DISTANCE_MAX,
} from "./render.js";
import { createPlot } from "./plot.js";
import { OS1_SCENARIOS, scenarioByName } from "./scenarios.js";

const DEG = Math.PI / 180;
const DEFAULT_PRESET = "os1f";

const statusEl = document.getElementById("status");
const paramPanel = document.getElementById("param-panel");

// ------------ DOM helpers ------------

function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", isError);
}

function sizeCanvasToDisplay(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width * dpr));
  const h = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

// ------------ Lensing + validation plumbing ------------

async function loadLensing(device) {
  const header = await fetch("public/lensing.json").then((r) => r.json());
  const blob = await fetch("public/lensing.bin").then((r) => r.arrayBuffer());
  const buffers = {};
  for (const [name, { byte_offset, n_bytes }] of Object.entries(header.arrays)) {
    const buf = device.createBuffer({
      label: `lensing:${name}`,
      size: n_bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, blob, byte_offset, n_bytes);
    buffers[name] = buf;
  }
  return { header, buffers };
}

async function loadReference(name) {
  const res = await fetch(`public/reference/${name}_gpu.txt`);
  if (!res.ok) return null;
  const text = await res.text();
  return new Float32Array(text.trim().split(/\s+/).map(Number));
}

async function runValidationSweep(engine) {
  const lines = ["validation (max relative error to peak):"];
  for (const sc of OS1_SCENARIOS) {
    const t0 = performance.now();
    const got = await engine.computePulseProfile(sc, OS1_DEFAULTS);
    const ms = performance.now() - t0;
    const ref = await loadReference(sc.name);
    if (!ref) { lines.push(`  ${sc.name}  [no reference]`); continue; }
    let peak = 0;
    for (const v of ref) peak = Math.max(peak, Math.abs(v));
    let maxRel = 0;
    for (let i = 0; i < got.length; i++) {
      const rel = peak > 0 ? Math.abs(got[i] - ref[i]) / peak : 0;
      if (rel > maxRel) maxRel = rel;
    }
    const verdict = maxRel < 1e-2 ? "ok" : maxRel < 1e-1 ? "warn" : "FAIL";
    lines.push(
      `  ${sc.name}  rel=${(maxRel * 100).toFixed(4).padStart(8)}%  peak=${peak.toExponential(3)}  [${verdict}]  (${ms.toFixed(1)} ms)`,
    );
  }
  return lines.join("\n");
}

// ------------ Params model ------------
//
// Sliders operate in display units (degrees, Hz). Only `paramsToScenario`
// bridges to the compute-layer shape, which is in radians.

function defaultParams() {
  return {
    nu: 600,
    inc_deg: 80,
    spot_theta_deg: 20,
    angular_radius: 1.0,
    beaming: 0,
  };
}

function paramsToScenario(p) {
  return {
    name: "custom",
    nu: p.nu,
    spot_center_theta: p.spot_theta_deg * DEG,
    inc: p.inc_deg * DEG,
    angular_radius: p.angular_radius,
    beaming: p.beaming,
  };
}

function scenarioToParams(sc) {
  return {
    nu: sc.nu,
    inc_deg: sc.inc / DEG,
    spot_theta_deg: sc.spot_center_theta / DEG,
    angular_radius: sc.angular_radius,
    beaming: sc.beaming,
  };
}

// ------------ Control widgets ------------

function makeSlider({ label, min, max, step, value, format }) {
  const row = el("div", "slider-row");
  const lab = el("label", null, label);
  const input = el("input");
  input.type = "range";
  input.min = min; input.max = max; input.step = step;
  input.value = value;
  const readout = el("span", "slider-value");
  const render = (v) => { readout.textContent = format(v); };
  render(value);
  row.append(lab, input, readout);
  return {
    row, input,
    get value() { return parseFloat(input.value); },
    setValue(v) { input.value = v; render(v); },
    onInput(cb) { input.addEventListener("input", () => { const v = parseFloat(input.value); render(v); cb(v); }); },
  };
}

function makeSelect({ label, options, value }) {
  const row = el("div", "select-row");
  const lab = el("label", null, label);
  const sel = el("select", "control-select");
  for (const { value: v, text } of options) {
    const opt = el("option");
    opt.value = String(v);
    opt.textContent = text;
    sel.appendChild(opt);
  }
  sel.value = String(value);
  row.append(lab, sel);
  return {
    row, sel,
    get value() { return parseFloat(sel.value); },
    setValue(v) { sel.value = String(v); },
    onChange(cb) { sel.addEventListener("change", () => cb(parseFloat(sel.value))); },
  };
}

function makePresetPicker(initial) {
  const row = el("div", "select-row");
  const lab = el("label", null, "preset");
  const sel = el("select", "control-select");
  for (const sc of OS1_SCENARIOS) {
    const opt = el("option");
    opt.value = sc.name;
    const beam = { 0: "iso", 1: "cos²", 2: "1-cos²" }[sc.beaming];
    const deg = (r) => (r / DEG).toFixed(0);
    const arLbl = sc.angular_radius < 0.1 ? "pt" : `${sc.angular_radius.toFixed(2)}rad`;
    opt.textContent = `${sc.name}  ν=${sc.nu} θ=${deg(sc.spot_center_theta)}° i=${deg(sc.inc)}° ρ=${arLbl} ${beam}`;
    if (sc.name === initial) opt.selected = true;
    sel.appendChild(opt);
  }
  row.append(lab, sel);
  return {
    row, sel,
    onChange(cb) { sel.addEventListener("change", () => cb(sel.value)); },
  };
}

// ------------ Drag-to-move panel ------------

function makeDraggable(panel, handle) {
  let dragging = false, offX = 0, offY = 0;
  handle.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    dragging = true;
    const rect = panel.getBoundingClientRect();
    offX = e.clientX - rect.left;
    offY = e.clientY - rect.top;
    panel.style.left = `${rect.left}px`;
    panel.style.top = `${rect.top}px`;
    panel.style.right = "auto";
    panel.style.bottom = "auto";
    handle.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  handle.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const x = Math.max(0, Math.min(window.innerWidth - 40, e.clientX - offX));
    const y = Math.max(0, Math.min(window.innerHeight - 30, e.clientY - offY));
    panel.style.left = `${x}px`;
    panel.style.top = `${y}px`;
  });
  const end = (e) => {
    if (!dragging) return;
    dragging = false;
    try { handle.releasePointerCapture(e.pointerId); } catch {}
  };
  handle.addEventListener("pointerup", end);
  handle.addEventListener("pointercancel", end);
}

// ------------ Main ------------

async function main() {
  if (!("gpu" in navigator)) {
    setStatus("WebGPU not available. Use Chrome/Edge (≥113) or Firefox Nightly with WebGPU enabled.", true);
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { setStatus("No WebGPU adapter.", true); return; }
  const device = await adapter.requestDevice();
  device.lost.then((info) => setStatus(`Device lost: ${info.reason}\n${info.message}`, true));

  const sphereCanvas = document.getElementById("sphere-canvas");
  const curveCanvas  = document.getElementById("curve-canvas");
  const curvePanel   = document.getElementById("curve-panel");
  const curveHandle  = curvePanel.querySelector("[data-drag-handle]");

  const resizeAll = () => {
    sizeCanvasToDisplay(sphereCanvas);
    sizeCanvasToDisplay(curveCanvas);
  };
  resizeAll();

  setStatus("loading lensing tables…");
  const lensing = await loadLensing(device);

  setStatus("compiling pipelines…");
  const engine   = await createPulseEngine(device, lensing);
  const renderer = await createSphereRenderer(device, sphereCanvas);
  const plot     = createPlot(curveCanvas);

  makeDraggable(curvePanel, curveHandle);

  const url = new URL(window.location.href);
  if (url.searchParams.has("validate")) {
    setStatus("running validation sweep…");
    setStatus(await runValidationSweep(engine));
    return;
  }

  // ----- state -----
  const initialPreset = scenarioByName(DEFAULT_PRESET);
  const params = scenarioToParams(initialPreset);
  let presetName = DEFAULT_PRESET;
  let observerPhase = 0;
  const INITIAL_ZOOM = 0.35;  // default view is a touch zoomed out
  let cameraDistance = CAM_DISTANCE_DEFAULT / INITIAL_ZOOM;
  let lastComputeMs = 0;

  // ----- UI -----
  paramPanel.appendChild(el("div", "section-header", "preset"));
  const preset = makePresetPicker(DEFAULT_PRESET);
  paramPanel.appendChild(preset.row);

  paramPanel.appendChild(el("div", "section-header", "observer / star"));
  const nuSlider = makeSlider({
    label: "ν", min: 1, max: 700, step: 1,
    value: params.nu,
    format: (v) => `${v.toFixed(0)} Hz`,
  });
  paramPanel.appendChild(nuSlider.row);
  const incSlider = makeSlider({
    label: "i", min: 0, max: 180, step: 1,
    value: params.inc_deg,
    format: (v) => `${v.toFixed(0)}°`,
  });
  paramPanel.appendChild(incSlider.row);

  paramPanel.appendChild(el("div", "section-header", "hotspot"));
  const thetaSlider = makeSlider({
    label: "θ", min: 0, max: 180, step: 1,
    value: params.spot_theta_deg,
    format: (v) => `${v.toFixed(0)}°`,
  });
  paramPanel.appendChild(thetaSlider.row);
  const rhoSlider = makeSlider({
    label: "ρ", min: 0.01, max: 1.5, step: 0.01,
    value: params.angular_radius,
    format: (v) => `${v.toFixed(2)} rad`,
  });
  paramPanel.appendChild(rhoSlider.row);
  const beamSelect = makeSelect({
    label: "beam",
    options: [
      { value: 0, text: "iso (isotropic)" },
      { value: 1, text: "cos² (pencil)" },
      { value: 2, text: "1 - cos² (fan)" },
    ],
    value: params.beaming,
  });
  paramPanel.appendChild(beamSelect.row);

  const info = el("div");
  info.style.cssText = "margin-top:14px;padding-top:10px;border-top:1px solid var(--border);font-size:11px;color:var(--muted);line-height:1.55;";
  info.innerHTML = `
    <div style="margin-bottom:4px;font-weight:600;color:var(--fg);">interaction</div>
    <div>drag on sphere → rotate pulsar</div>
    <div>wheel on sphere → zoom</div>
    <div>drag curve header → move panel</div>
  `;
  paramPanel.appendChild(info);

  // ----- Latest-wins compute queue -----
  let computeBusy = false;
  let computeDirty = false;

  async function runComputeLoop() {
    if (computeBusy) { computeDirty = true; return; }
    computeBusy = true;
    try {
      do {
        computeDirty = false;
        const t0 = performance.now();
        const flux = await engine.computePulseProfile(paramsToScenario(params), OS1_DEFAULTS);
        lastComputeMs = performance.now() - t0;
        plot.setProfile(flux, presetName || "custom");
        redrawSphere();
        updateStatus();
      } while (computeDirty);
    } finally {
      computeBusy = false;
    }
  }

  function requestRecompute() { runComputeLoop(); }

  function redrawSphere() {
    renderer.draw({
      inc: params.inc_deg * DEG,
      spot_center_theta: params.spot_theta_deg * DEG,
      spot_center_phi: 0,
      angular_radius: params.angular_radius,
      observer_phase: observerPhase,
      aspect: sphereCanvas.width / sphereCanvas.height,
      distance: cameraDistance,
    });
    plot.setPhase(observerPhase);
  }

  function updateStatus() {
    const vendor = (adapter.info?.vendor ?? "?") + " " + (adapter.info?.architecture ?? "");
    setStatus([
      `adapter:  ${vendor.trim()}`,
      `preset:   ${presetName}`,
      `compute:  ${lastComputeMs.toFixed(1)} ms`,
      `zoom:     ${(CAM_DISTANCE_DEFAULT / cameraDistance).toFixed(2)}×`,
    ].join("\n"));
  }

  // ----- Wire sliders -----
  nuSlider.onInput((v)     => { params.nu = v;                 requestRecompute(); });
  incSlider.onInput((v)    => { params.inc_deg = v;            requestRecompute(); });
  thetaSlider.onInput((v)  => { params.spot_theta_deg = v;     requestRecompute(); });
  rhoSlider.onInput((v)    => { params.angular_radius = v;     requestRecompute(); });
  beamSelect.onChange((v)  => { params.beaming = v;            requestRecompute(); });

  preset.onChange((name) => {
    presetName = name;
    const p = scenarioToParams(scenarioByName(name));
    Object.assign(params, p);
    nuSlider.setValue(p.nu);
    incSlider.setValue(p.inc_deg);
    thetaSlider.setValue(p.spot_theta_deg);
    rhoSlider.setValue(p.angular_radius);
    beamSelect.setValue(p.beaming);
    requestRecompute();
  });

  // ----- Sphere drag -----
  const PHASE_PER_PIXEL = 1 / 600;
  let dragging = false, lastX = 0;
  sphereCanvas.addEventListener("pointerdown", (e) => {
    dragging = true;
    lastX = e.clientX;
    sphereCanvas.setPointerCapture(e.pointerId);
  });
  sphereCanvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX;
    lastX = e.clientX;
    observerPhase = ((observerPhase + dx * PHASE_PER_PIXEL) % 1 + 1) % 1;
    redrawSphere();
  });
  const endDrag = (e) => {
    if (!dragging) return;
    dragging = false;
    try { sphereCanvas.releasePointerCapture(e.pointerId); } catch {}
  };
  sphereCanvas.addEventListener("pointerup", endDrag);
  sphereCanvas.addEventListener("pointercancel", endDrag);

  // ----- Wheel zoom -----
  const ZOOM_SENSITIVITY = 0.0015;
  sphereCanvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const factor = Math.exp(e.deltaY * ZOOM_SENSITIVITY);
      cameraDistance = Math.min(CAM_DISTANCE_MAX, Math.max(CAM_DISTANCE_MIN, cameraDistance * factor));
      redrawSphere();
      updateStatus();
    },
    { passive: false },
  );

  // ----- Resize -----
  window.addEventListener("resize", () => {
    resizeAll();
    redrawSphere();
    plot.resize();
  });

  // ----- Initial compute + draw -----
  await runComputeLoop();

  window.__ppm = {
    device, adapter, lensing, engine, renderer, plot,
    get params() { return { ...params }; },
    get phase()  { return observerPhase; },
    get distance() { return cameraDistance; },
  };
}

main().catch((err) => {
  console.error(err);
  setStatus(`error: ${err.message}`, true);
});
