// Canvas2D light curve + dashed phase marker.

const PAD = { left: 62, right: 18, top: 18, bottom: 38 };
const FONT = "14px ui-monospace, SF Mono, monospace";
const HEADER_FONT = "14px ui-monospace, SF Mono, monospace";
const AXIS = "#505566";
const GRID = "#1d2130";
const LINE = "#ffb86b";
const MARKER = "#e6e8ef";
const LABEL = "#8b93a7";

export function createPlot(canvas) {
  const ctx2d = canvas.getContext("2d");

  let fluxData = null;  // Float32Array of length N_output_phase
  let observerPhase = 0;
  let label = "";

  function setProfile(flux, lbl) {
    fluxData = flux;
    label = lbl ?? "";
    draw();
  }
  function setPhase(p) {
    observerPhase = ((p % 1) + 1) % 1;
    draw();
  }
  function resize() { draw(); }

  function draw() {
    const w = canvas.width;
    const h = canvas.height;
    ctx2d.fillStyle = "#141821";
    ctx2d.fillRect(0, 0, w, h);
    if (!fluxData) return;

    const plotX0 = PAD.left;
    const plotY0 = PAD.top;
    const plotW = w - PAD.left - PAD.right;
    const plotH = h - PAD.top - PAD.bottom;

    // y-axis anchored at zero; auto-scale only the upper bound so the
    // amplitude is readable across parameter changes.
    const fmin = 0;
    let fmax = 0;
    for (const v of fluxData) { if (v > fmax) fmax = v; }
    if (fmax <= 0) { fmax = 1; }
    fmax += fmax * 0.08;

    // Grid
    ctx2d.strokeStyle = GRID;
    ctx2d.lineWidth = 1;
    ctx2d.beginPath();
    for (let i = 0; i <= 4; i++) {
      const x = plotX0 + (plotW * i) / 4;
      ctx2d.moveTo(x, plotY0);
      ctx2d.lineTo(x, plotY0 + plotH);
    }
    for (let i = 0; i <= 4; i++) {
      const y = plotY0 + (plotH * i) / 4;
      ctx2d.moveTo(plotX0, y);
      ctx2d.lineTo(plotX0 + plotW, y);
    }
    ctx2d.stroke();

    // Axes
    ctx2d.strokeStyle = AXIS;
    ctx2d.beginPath();
    ctx2d.moveTo(plotX0, plotY0);
    ctx2d.lineTo(plotX0, plotY0 + plotH);
    ctx2d.lineTo(plotX0 + plotW, plotY0 + plotH);
    ctx2d.stroke();

    // Flux line (periodic — repeat first point at phase=1)
    const N = fluxData.length;
    ctx2d.strokeStyle = LINE;
    ctx2d.lineWidth = 2;
    ctx2d.beginPath();
    for (let i = 0; i <= N; i++) {
      const v = fluxData[i % N];
      const x = plotX0 + (plotW * i) / N;
      const y = plotY0 + plotH * (1 - (v - fmin) / (fmax - fmin));
      if (i === 0) ctx2d.moveTo(x, y); else ctx2d.lineTo(x, y);
    }
    ctx2d.stroke();

    // Dashed phase marker
    const xPhase = plotX0 + plotW * observerPhase;
    ctx2d.save();
    ctx2d.strokeStyle = MARKER;
    ctx2d.lineWidth = 1.5;
    ctx2d.setLineDash([5, 4]);
    ctx2d.beginPath();
    ctx2d.moveTo(xPhase, plotY0);
    ctx2d.lineTo(xPhase, plotY0 + plotH);
    ctx2d.stroke();
    ctx2d.restore();

    // Labels
    ctx2d.fillStyle = LABEL;
    ctx2d.font = FONT;
    ctx2d.textAlign = "right";
    ctx2d.textBaseline = "middle";
    for (let i = 0; i <= 4; i++) {
      const v = fmin + (fmax - fmin) * (1 - i / 4);
      const y = plotY0 + (plotH * i) / 4;
      ctx2d.fillText(v.toExponential(1), plotX0 - 8, y);
    }
    ctx2d.textAlign = "center";
    ctx2d.textBaseline = "top";
    for (let i = 0; i <= 4; i++) {
      const x = plotX0 + (plotW * i) / 4;
      ctx2d.fillText((i / 4).toFixed(2), x, plotY0 + plotH + 8);
    }

    ctx2d.font = HEADER_FONT;
    ctx2d.textAlign = "left";
    ctx2d.textBaseline = "top";
    ctx2d.fillStyle = LABEL;
    ctx2d.fillText(`phase: ${observerPhase.toFixed(3)}`, plotX0 + 4, plotY0 + 2);
    if (label) {
      ctx2d.textAlign = "right";
      ctx2d.fillText(label, plotX0 + plotW - 4, plotY0 + 2);
    }
  }

  return { setProfile, setPhase, resize, draw };
}
