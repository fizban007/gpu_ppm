// Sphere renderer — ray-casts a unit sphere at origin, highlights the hotspot.

const PARAMS_SIZE = 112; // 7 vec4s = 7*16

// Camera always orbits at fixed distance in the plane containing the spin axis (z).
// The observer's inclination `inc` fixes the elevation; azimuth is user-controlled
// (though we hold azimuth fixed and advance `observer_phase` instead — physically
// equivalent, and keeps the camera relationship simple). Distance is now a
// runtime param so the mouse wheel can zoom in/out.
export const CAM_DISTANCE_DEFAULT = 3.2;
export const CAM_DISTANCE_MIN = 1.35;  // just outside the unit sphere
export const CAM_DISTANCE_MAX = 12.0;
const FOV_Y = 35 * Math.PI / 180;

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

  function writeParams({ inc, spot_center_theta, spot_center_phi, angular_radius, observer_phase, aspect, distance }) {
    // Camera in the star frame: sits in the (x,z) plane at elevation = inc
    // (measured from the spin axis +z). sin(inc) = projection onto the orbital
    // plane, cos(inc) = z component.
    const d = distance ?? CAM_DISTANCE_DEFAULT;
    const si = Math.sin(inc);
    const ci = Math.cos(inc);
    const camPos = [d * si, 0, d * ci];
    const fwd = normalize3([-camPos[0], -camPos[1], -camPos[2]]);
    const worldUp = [0, 0, 1];
    const right = normalize3(cross3(fwd, worldUp));
    const up = cross3(right, fwd);

    // cam_pos
    sf[0] = camPos[0]; sf[1] = camPos[1]; sf[2] = camPos[2]; sf[3] = 0;
    // cam_right
    sf[4] = right[0]; sf[5] = right[1]; sf[6] = right[2]; sf[7] = 0;
    // cam_up
    sf[8] = up[0]; sf[9] = up[1]; sf[10] = up[2]; sf[11] = 0;
    // cam_fwd
    sf[12] = fwd[0]; sf[13] = fwd[1]; sf[14] = fwd[2]; sf[15] = 0;
    // view: aspect, tan(fovy/2)
    sf[16] = aspect; sf[17] = Math.tan(FOV_Y * 0.5); sf[18] = 0; sf[19] = 0;
    // spot: theta, phi, cos(ang_rad), observer_phase
    sf[20] = spot_center_theta;
    sf[21] = spot_center_phi;
    sf[22] = Math.cos(angular_radius);
    sf[23] = observer_phase;
    // light_dir: place light a bit off from the camera for visible shading
    const ld = normalize3([fwd[0] + 0.3 * right[0] + 0.25 * up[0],
                           fwd[1] + 0.3 * right[1] + 0.25 * up[1],
                           fwd[2] + 0.3 * right[2] + 0.25 * up[2]]);
    // Light points from scene toward light source, so shader uses `light_dir` directly.
    sf[24] = -ld[0]; sf[25] = -ld[1]; sf[26] = -ld[2]; sf[27] = 0;

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
