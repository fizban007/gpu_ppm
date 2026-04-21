// Interactive pulsar pulse-profile visualizer, multi-mode build.
//
// Emission models:
//   - spots   — user-placed ADD/SUBTRACT spherical-cap hotspots (kernel.wgsl)
//   - dipole  — physics-derived polar-cap temperature map from a centered
//               magnetic dipole (kernel_dipole.wgsl)
//
// The two kernels share the lensing table and observer geometry but run on
// independent compute pipelines.

import { createPulseEngine, OS1_DEFAULTS, findParent } from "./compute.js";
import { createDipoleEngine } from "./compute_dipole.js";
import {
  createSphereRenderer,
  CAM_DISTANCE_DEFAULT,
  CAM_DISTANCE_MIN,
  CAM_DISTANCE_MAX,
} from "./render.js";
import { createPlot } from "./plot.js";
import { OS1_SCENARIOS, PRESETS, presetByName } from "./scenarios.js";

const DEG = Math.PI / 180;
const DEFAULT_PRESET = "crescent + spot";
const MAX_SPOTS = 4;

// α₀ R/μ = √(3/2) (Ω R/c) (1 + sin²ι/5). c in km/s so Re (km) × ν (Hz) lines up.
const C_KM_S = 299792.458;
function alpha0Dim(nu_hz, Re_km, iota_rad) {
  const si = Math.sin(iota_rad);
  return Math.sqrt(1.5) * (2 * Math.PI * nu_hz * Re_km / C_KM_S) * (1 + (si * si) / 5);
}

const statusEl = document.getElementById("status");
const paramPanel = document.getElementById("param-panel");

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
  for (const p of PRESETS) {
    const opt = el("option");
    opt.value = p.name;
    const beam = { 0: "iso", 1: "cos²", 2: "1-cos²" }[p.beaming];
    if (p.emission_mode === "dipole") {
      opt.textContent = `${p.name}  ν=${p.nu} i=${p.inc_deg.toFixed(0)}° ${beam}`;
    } else {
      const n = p.spots.length;
      opt.textContent = `${p.name}  ν=${p.nu} i=${p.inc_deg.toFixed(0)}° ${beam} (${n} spot${n === 1 ? "" : "s"})`;
    }
    if (p.name === initial) opt.selected = true;
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

// ------------ Spot block (spots mode only) ------------

function buildSpotBlock(spot, index, ctx) {
  const block = el("div", "spot-block");
  const header = el("div", "spot-block-header");
  const title = el("span");
  title.innerHTML = `spot ${index + 1} &nbsp;<span class="tag">${spot.mode}</span>`;
  const rm = el("button", "spot-remove", "×");
  rm.title = "remove";
  rm.addEventListener("click", () => ctx.remove(index));
  header.append(title, rm);
  block.appendChild(header);

  const thetaSlider = makeSlider({
    label: "θ", min: 0, max: 180, step: 1,
    value: spot.theta_deg, format: (v) => `${v.toFixed(0)}°`,
  });
  const phiSlider = makeSlider({
    label: "φ", min: 0, max: 360, step: 1,
    value: spot.phi_deg, format: (v) => `${v.toFixed(0)}°`,
  });
  const rhoSlider = makeSlider({
    label: "ρ", min: 0.01, max: 1.5, step: 0.01,
    value: spot.rho, format: (v) => `${v.toFixed(2)} rad`,
  });
  const modeSelect = makeSelect({
    label: "mode",
    options: [{ value: 0, text: "ADD" }, { value: 1, text: "SUBTRACT" }],
    value: spot.mode === "ADD" ? 0 : 1,
  });
  const kTSlider = makeSlider({
    label: "kT", min: 0.05, max: 3.0, step: 0.01,
    value: spot.kT, format: (v) => `${v.toFixed(2)} keV`,
  });
  block.append(thetaSlider.row, phiSlider.row, rhoSlider.row, modeSelect.row, kTSlider.row);

  const applyMode = () => {
    title.innerHTML = `spot ${index + 1} &nbsp;<span class="tag">${spot.mode}</span>`;
    kTSlider.row.style.display = spot.mode === "ADD" ? "" : "none";
  };
  applyMode();

  thetaSlider.onInput((v) => { spot.theta_deg = v; ctx.changed(); });
  phiSlider.onInput((v)   => { spot.phi_deg = v;   ctx.changed(); });
  rhoSlider.onInput((v)   => { spot.rho = v;       ctx.changed(); });
  kTSlider.onInput((v)    => { spot.kT = v;        ctx.changed(); });
  modeSelect.onChange((v) => {
    spot.mode = v === 0 ? "ADD" : "SUB";
    applyMode();
    ctx.changed();
    ctx.rebuild();
  });

  if (spot.mode === "SUB") {
    const adds = ctx.allSpots.filter((s) => s.mode === "ADD")
      .map((s) => ({ theta: s.theta_deg * DEG, phi: s.phi_deg * DEG, rho: s.rho }));
    const self = { theta: spot.theta_deg * DEG, phi: spot.phi_deg * DEG };
    if (!findParent(adds, self)) block.classList.add("inert");
  }
  return block;
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
  const spotEngine   = await createPulseEngine(device, lensing);
  const dipoleEngine = await createDipoleEngine(device, lensing);
  const renderer     = await createSphereRenderer(device, sphereCanvas);
  const plot         = createPlot(curveCanvas);

  makeDraggable(curvePanel, curveHandle);

  // ---------- state ----------
  const observer = { nu: 600, inc_deg: 80, beaming: 0 };
  const spots = [];
  const dipole = { mag_incl_deg: 45, T0: 0.40, display_mode: 1 };   // 1 = T-map, 2 = |j|
  let emissionMode = "spots";
  let presetName = DEFAULT_PRESET;
  let observerPhase = 0;
  const INITIAL_ZOOM = 0.35;
  let cameraDistance = CAM_DISTANCE_DEFAULT / INITIAL_ZOOM;
  let lastComputeMs = 0;

  function loadPreset(name) {
    presetName = name;
    const p = presetByName(name);
    observer.nu = p.nu;
    observer.inc_deg = p.inc_deg;
    observer.beaming = p.beaming;
    emissionMode = p.emission_mode ?? "spots";
    if (emissionMode === "dipole") {
      dipole.mag_incl_deg = p.dipole.mag_incl_deg;
      dipole.T0 = p.dipole.T0;
    } else {
      spots.length = 0;
      for (const s of p.spots) spots.push({ ...s });
    }
  }
  loadPreset(DEFAULT_PRESET);

  // ---------- preset + shared controls ----------
  paramPanel.appendChild(el("div", "section-header", "preset"));
  const preset = makePresetPicker(DEFAULT_PRESET);
  paramPanel.appendChild(preset.row);

  paramPanel.appendChild(el("div", "section-header", "emission"));
  const modeSelect = makeSelect({
    label: "model",
    options: [
      { value: 0, text: "spots" },
      { value: 1, text: "dipole" },
    ],
    value: emissionMode === "dipole" ? 1 : 0,
  });
  paramPanel.appendChild(modeSelect.row);

  paramPanel.appendChild(el("div", "section-header", "observer / star"));
  const nuSlider = makeSlider({
    label: "ν", min: 1, max: 700, step: 1,
    value: observer.nu, format: (v) => `${v.toFixed(0)} Hz`,
  });
  const incSlider = makeSlider({
    label: "i", min: 0, max: 180, step: 1,
    value: observer.inc_deg, format: (v) => `${v.toFixed(0)}°`,
  });
  const beamSelect = makeSelect({
    label: "beam",
    options: [
      { value: 0, text: "iso (isotropic)" },
      { value: 1, text: "cos² (pencil)" },
      { value: 2, text: "1 - cos² (fan)" },
    ],
    value: observer.beaming,
  });
  paramPanel.append(nuSlider.row, incSlider.row, beamSelect.row);

  // ---------- mutex panels ----------
  const spotsHeader  = el("div", "section-header", "spots");
  const spotsBody    = el("div");
  const addBtn       = el("button", "add-spot-btn", "+ add spot");

  const dipoleHeader = el("div", "section-header", "dipole");
  const dipoleBody   = el("div");
  const iotaSlider   = makeSlider({
    label: "ι", min: 0, max: 90, step: 1,
    value: dipole.mag_incl_deg, format: (v) => `${v.toFixed(0)}°`,
  });
  const T0Slider     = makeSlider({
    label: "T₀", min: 0.05, max: 2.0, step: 0.01,
    value: dipole.T0, format: (v) => `${v.toFixed(2)} keV`,
  });
  const displaySelect = makeSelect({
    label: "show",
    options: [{ value: 1, text: "temperature" }, { value: 2, text: "|j| (current)" }],
    value: dipole.display_mode,
  });
  dipoleBody.append(iotaSlider.row, T0Slider.row, displaySelect.row);

  paramPanel.append(spotsHeader, spotsBody, addBtn, dipoleHeader, dipoleBody);

  const info = el("div");
  info.style.cssText = "margin-top:14px;padding-top:10px;border-top:1px solid var(--border);font-size:11px;color:var(--muted);line-height:1.55;";
  info.innerHTML = `
    <div style="margin-bottom:4px;font-weight:600;color:var(--fg);">interaction</div>
    <div>drag on sphere → rotate pulsar</div>
    <div>wheel on sphere → zoom</div>
    <div>drag curve header → move panel</div>
  `;
  paramPanel.appendChild(info);

  // ---------- Mutex panel visibility ----------
  function applyModeVisibility() {
    const showSpots  = emissionMode === "spots";
    spotsHeader.style.display  = showSpots ? "" : "none";
    spotsBody.style.display    = showSpots ? "" : "none";
    addBtn.style.display       = showSpots ? "" : "none";
    dipoleHeader.style.display = showSpots ? "none" : "";
    dipoleBody.style.display   = showSpots ? "none" : "";
  }

  // ---------- Spot list (rebuilt on add/remove/mode-change) ----------
  function rebuildSpots() {
    spotsBody.innerHTML = "";
    const ctx = {
      allSpots: spots,
      changed: requestRecompute,
      remove: (i) => { spots.splice(i, 1); rebuildSpots(); requestRecompute(); },
      rebuild: rebuildSpots,
    };
    spots.forEach((s, i) => spotsBody.appendChild(buildSpotBlock(s, i, ctx)));
    addBtn.disabled = spots.length >= MAX_SPOTS;
  }

  // ---------- Latest-wins compute queue ----------
  let computeBusy = false;
  let computeDirty = false;

  async function runComputeLoop() {
    if (computeBusy) { computeDirty = true; return; }
    computeBusy = true;
    try {
      do {
        computeDirty = false;
        const t0 = performance.now();
        let flux;
        if (emissionMode === "dipole") {
          flux = await dipoleEngine.computeDipoleProfile(
            {
              nu: observer.nu,
              mag_incl: dipole.mag_incl_deg * DEG,
              obs_incl: observer.inc_deg * DEG,
              T0: dipole.T0,
              beaming: observer.beaming,
            },
            OS1_DEFAULTS,
          );
        } else {
          const shared = {
            nu: observer.nu,
            inc: observer.inc_deg * DEG,
            beaming: observer.beaming,
            constants: OS1_DEFAULTS,
          };
          const engineSpots = spots.map((s) => ({
            theta: s.theta_deg * DEG,
            phi: s.phi_deg * DEG,
            rho: s.rho,
            mode: s.mode,
            kT: s.kT,
          }));
          flux = await spotEngine.computeMultiSpot(engineSpots, shared);
        }
        lastComputeMs = performance.now() - t0;
        plot.setProfile(flux, labelForStatus());
        redrawSphere();
        updateStatus();
      } while (computeDirty);
    } finally {
      computeBusy = false;
    }
  }
  function requestRecompute() { runComputeLoop(); }

  function labelForStatus() {
    if (emissionMode === "dipole") return `${presetName} • dipole`;
    return `${presetName} • ${spots.length} spot${spots.length === 1 ? "" : "s"}`;
  }

  function redrawSphere() {
    const p = {
      inc: observer.inc_deg * DEG,
      spots: spots.map((s) => ({
        theta: s.theta_deg * DEG,
        phi: s.phi_deg * DEG,
        rho: s.rho,
        mode: s.mode,
        kT: s.kT,
      })),
      observer_phase: observerPhase,
      aspect: sphereCanvas.width / sphereCanvas.height,
      distance: cameraDistance,
    };
    if (emissionMode === "dipole") {
      p.dipole = {
        mag_incl: dipole.mag_incl_deg * DEG,
        alpha0_dim: alpha0Dim(observer.nu, OS1_DEFAULTS.Re, dipole.mag_incl_deg * DEG),
        T0: dipole.T0,
        display_mode: dipole.display_mode,
      };
    }
    renderer.draw(p);
    plot.setPhase(observerPhase);
  }

  function updateStatus() {
    const vendor = (adapter.info?.vendor ?? "?") + " " + (adapter.info?.architecture ?? "");
    const lines = [
      `adapter:  ${vendor.trim()}`,
      `preset:   ${presetName}`,
      `model:    ${emissionMode}`,
    ];
    if (emissionMode === "dipole") {
      lines.push(`ι: ${dipole.mag_incl_deg.toFixed(0)}°  T₀: ${dipole.T0.toFixed(2)} keV`);
    } else {
      lines.push(`spots:    ${spots.length}`);
    }
    lines.push(`compute:  ${lastComputeMs.toFixed(1)} ms`);
    lines.push(`zoom:     ${(CAM_DISTANCE_DEFAULT / cameraDistance).toFixed(2)}×`);
    setStatus(lines.join("\n"));
  }

  // ---------- Wire shared controls ----------
  nuSlider.onInput((v)    => { observer.nu = v;       requestRecompute(); });
  incSlider.onInput((v)   => { observer.inc_deg = v;  requestRecompute(); });
  beamSelect.onChange((v) => { observer.beaming = v;  requestRecompute(); });

  iotaSlider.onInput((v)  => { dipole.mag_incl_deg = v; requestRecompute(); });
  T0Slider.onInput((v)    => { dipole.T0 = v;           requestRecompute(); });
  displaySelect.onChange((v) => {
    dipole.display_mode = v;
    redrawSphere();     // render-only change, no recompute
    updateStatus();
  });

  modeSelect.onChange((v) => {
    emissionMode = v === 1 ? "dipole" : "spots";
    applyModeVisibility();
    requestRecompute();
  });

  preset.onChange((name) => {
    loadPreset(name);
    nuSlider.setValue(observer.nu);
    incSlider.setValue(observer.inc_deg);
    beamSelect.setValue(observer.beaming);
    iotaSlider.setValue(dipole.mag_incl_deg);
    T0Slider.setValue(dipole.T0);
    modeSelect.setValue(emissionMode === "dipole" ? 1 : 0);
    applyModeVisibility();
    rebuildSpots();
    requestRecompute();
  });

  addBtn.addEventListener("click", () => {
    if (spots.length >= MAX_SPOTS) return;
    spots.push({ theta_deg: 60, phi_deg: 60, rho: 0.5, mode: "ADD", kT: 0.35 });
    rebuildSpots();
    requestRecompute();
  });

  applyModeVisibility();
  rebuildSpots();

  // ---------- Sphere drag ----------
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

  // ---------- Wheel zoom ----------
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

  window.addEventListener("resize", () => {
    resizeAll();
    redrawSphere();
    plot.resize();
  });

  await runComputeLoop();

  window.__ppm = {
    device, adapter, lensing, spotEngine, dipoleEngine, renderer, plot,
    get observer() { return { ...observer }; },
    get spots()    { return spots.map((s) => ({ ...s })); },
    get dipole()   { return { ...dipole }; },
    get mode()     { return emissionMode; },
    get phase()    { return observerPhase; },
  };
}

main().catch((err) => {
  console.error(err);
  setStatus(`error: ${err.message}`, true);
});
