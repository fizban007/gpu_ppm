// Sphere renderer — ray-casts a unit sphere at origin, paints up to 4 spots
// in painter's order (ADD blends hot, SUBTRACT blends cold).

export const CAM_DISTANCE_DEFAULT = 3.2;   // zoom = 1.0× reference (not the startup distance)
export const CAM_DISTANCE_MIN = 1.35;
export const CAM_DISTANCE_MAX = 20.0;
const FOV_Y = 35 * Math.PI / 180;

const MAX_SPOTS = 4;
// RenderParams layout: 6 x vec4 of chrome + 4 x vec4 of spots + 1 x vec4 of
// per-spot kTs = 11 * 16 = 176 bytes.
const PARAMS_SIZE = 176;

function normalize3(v) {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
}
function cross3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

export async function createSphereRenderer(device, canvas) {
  const shaderCode = await fetch("src/render.wgsl").then((r) => r.text());
  const module = device.createShaderModule({ label: "render.wgsl", code: shaderCode });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter((m) => m.type === "error");
  if (errs.length) {
    throw new Error("render.wgsl compile errors:\n" + errs.map((e) => `${e.lineNum}:${e.linePos}  ${e.message}`).join("\n"));
  }

  const ctx = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "opaque" });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }],
  });
  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: { module, entryPoint: "vs_main" },
    fragment: { module, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  const paramsBuffer = device.createBuffer({
    size: PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: paramsBuffer } }],
  });

  const scratch = new ArrayBuffer(PARAMS_SIZE);
  const sf = new Float32Array(scratch);

  // spots: array of {theta, phi, rho, mode: "ADD"|"SUB"}
  function writeParams({ inc, spots, observer_phase, aspect, distance }) {
    const d = distance ?? CAM_DISTANCE_DEFAULT;
    const si = Math.sin(inc);
    const ci = Math.cos(inc);
    const camPos = [d * si, 0, d * ci];
    const fwd = normalize3([-camPos[0], -camPos[1], -camPos[2]]);
    const worldUp = [0, 0, 1];
    const right = normalize3(cross3(fwd, worldUp));
    const up = cross3(right, fwd);

    // cam_pos / right / up / fwd
    sf[0]  = camPos[0]; sf[1]  = camPos[1]; sf[2]  = camPos[2]; sf[3] = 0;
    sf[4]  = right[0];  sf[5]  = right[1];  sf[6]  = right[2];  sf[7] = 0;
    sf[8]  = up[0];     sf[9]  = up[1];     sf[10] = up[2];     sf[11] = 0;
    sf[12] = fwd[0];    sf[13] = fwd[1];    sf[14] = fwd[2];    sf[15] = 0;

    // view: aspect, tan(fovy/2), observer_phase, spot_count
    sf[16] = aspect;
    sf[17] = Math.tan(FOV_Y * 0.5);
    sf[18] = observer_phase;
    sf[19] = Math.min(spots.length, MAX_SPOTS);

    // light_dir: reverse of the lit-side direction (fragment uses -light_dir sign).
    const ld = normalize3([
      fwd[0] + 0.3 * right[0] + 0.25 * up[0],
      fwd[1] + 0.3 * right[1] + 0.25 * up[1],
      fwd[2] + 0.3 * right[2] + 0.25 * up[2],
    ]);
    sf[20] = -ld[0]; sf[21] = -ld[1]; sf[22] = -ld[2]; sf[23] = 0;

    // spots: starting at offset 24 (= 96 bytes / 4), 4 vec4s.
    for (let k = 0; k < MAX_SPOTS; k++) {
      const base = 24 + k * 4;
      if (k < spots.length) {
        const s = spots[k];
        sf[base + 0] = s.theta;
        sf[base + 1] = s.phi;
        sf[base + 2] = Math.cos(s.rho);
        sf[base + 3] = s.mode === "ADD" ? 1.0 : (s.mode === "SUB" ? -1.0 : 0.0);
      } else {
        sf[base + 0] = 0;
        sf[base + 1] = 0;
        sf[base + 2] = 1.0;
        sf[base + 3] = 0.0;
      }
    }

    // spots_kt: one vec4 packing 4 scalar kTs.
    for (let k = 0; k < MAX_SPOTS; k++) {
      sf[40 + k] = k < spots.length ? (spots[k].kT ?? 0) : 0;
    }

    device.queue.writeBuffer(paramsBuffer, 0, scratch);
  }

  function draw(params) {
    writeParams(params);
    const encoder = device.createCommandEncoder({ label: "sphere-render" });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: ctx.getCurrentTexture().createView(),
        clearValue: { r: 0.02, g: 0.03, b: 0.05, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  return { draw };
}
