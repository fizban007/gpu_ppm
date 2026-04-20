// WebGPU compute pipeline for pulse-profile calculation.
// Mirrors gpu/os1_gpu.cu:KernelFunc via web/src/kernel.wgsl.
//
// Exposes two levels:
//   - computePulseProfile(scenario, constants, clip?)  — single kernel dispatch
//   - computeMultiSpot(spots, shared)                  — signed sum across spots
//
// A "spot" is { theta, phi, rho, mode: "ADD"|"SUB", kT }. Each ADD contributes
// its own kernel dispatch at its own kT. Each SUBTRACT is clipped to the first
// ADD containing its center and dispatched at the parent's kT with sign -1.
// Per-spot phi is handled in JS by circular-shifting the 128-bin output.

const N_SIDE = 256;
const N_OUTPUT_PHASE = 128;
const N_RINGS_FULL = 4 * N_SIDE - 1;     // 1023
const PARAMS_SIZE = 112;                 // bytes; matches Params struct in kernel.wgsl

export const OS1_DEFAULTS = {
  M: 1.4,
  Re: 12.0,
  kT: 0.35,
  E_obs: 1.0,
  D: 0.2,
};

// Ibb_div_D2 = 2 / (h³ c² D_km²)  [computed in f64, narrowed to f32]
function ibbOverD2(D_kpc) {
  const H_KEV_S = 4.135668e-18;
  const C_CM_S = 2.99792458e10;
  const KPC_KM = 3.08567758e16;
  const D_km = D_kpc * KPC_KM;
  return 2.0 / (H_KEV_S * H_KEV_S * H_KEV_S * C_CM_S * C_CM_S * D_km * D_km);
}

// Circularly shift a light curve so that "spot at φ=0" output becomes the
// light curve for a spot at φ=phi (rad). Positive phi moves the light-curve
// peak to later phase.
export function phaseShift(flux, phi_rad) {
  const N = flux.length;
  const out = new Float32Array(N);
  const shiftIdx = -(phi_rad / (2 * Math.PI)) * N;  // may be negative
  for (let i = 0; i < N; i++) {
    let j = i - shiftIdx;
    j = ((j % N) + N) % N;
    const jf = Math.floor(j);
    const jc = (jf + 1) % N;
    const t = j - jf;
    out[i] = flux[jf] * (1 - t) + flux[jc] * t;
  }
  return out;
}

// For each SUBTRACT spot, pick its parent ADD (the first one whose cap
// contains the SUBTRACT's center). Returns undefined if no ADD contains it.
export function findParent(adds, sub) {
  for (const a of adds) {
    const cosRho =
      Math.sin(sub.theta) * Math.sin(a.theta) * Math.cos(sub.phi - a.phi)
      + Math.cos(sub.theta) * Math.cos(a.theta);
    if (cosRho > Math.cos(a.rho)) return a;
  }
  return undefined;
}

export async function createPulseEngine(device, lensing) {
  const shaderCode = await fetch("src/kernel.wgsl").then((r) => {
    if (!r.ok) throw new Error(`fetch kernel.wgsl: ${r.status}`);
    return r.text();
  });
  const module = device.createShaderModule({ label: "kernel.wgsl", code: shaderCode });

  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === "error");
  if (errors.length) {
    const msg = errors.map((e) => `${e.lineNum}:${e.linePos}  ${e.message}`).join("\n");
    throw new Error(`kernel.wgsl compile errors:\n${msg}`);
  }

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

  const mainPipeline = device.createComputePipeline({
    label: "pulse-profile main",
    layout: pipelineLayout,
    compute: { module, entryPoint: "main" },
  });
  const sumPipeline = device.createComputePipeline({
    label: "pulse-profile sum_rings",
    layout: pipelineLayout,
    compute: { module, entryPoint: "sum_rings" },
  });

  const paramsBuffer = device.createBuffer({
    label: "params",
    size: PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const perRingBuffer = device.createBuffer({
    label: "per_ring_flux",
    size: N_RINGS_FULL * N_OUTPUT_PHASE * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const totalFluxBuffer = device.createBuffer({
    label: "total_flux",
    size: N_OUTPUT_PHASE * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readbackBuffer = device.createBuffer({
    label: "readback",
    size: N_OUTPUT_PHASE * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: paramsBuffer } },
      { binding: 1, resource: { buffer: lensing.buffers.cos_alpha_of_u_cos_psi } },
      { binding: 2, resource: { buffer: lensing.buffers.lf_of_u_cos_psi } },
      { binding: 3, resource: { buffer: lensing.buffers.cdt_over_R_of_u_cos_alpha } },
      { binding: 4, resource: { buffer: perRingBuffer } },
      { binding: 5, resource: { buffer: totalFluxBuffer } },
    ],
  });

  // Layout matches Params in kernel.wgsl (112 bytes). Clip fields are zeroed
  // when clip is absent.
  function writeParams(dispatch, constants) {
    const h = lensing.header;
    const point_source = dispatch.angular_radius < 0.1 ? 1 : 0;
    const n_rings = point_source ? 1 : N_RINGS_FULL;
    const Ibb_div_D2 = ibbOverD2(constants.D);

    const ab = new ArrayBuffer(PARAMS_SIZE);
    const f = new Float32Array(ab);
    const u = new Uint32Array(ab);

    //   0 nu    4 spot_center_theta    8 inc    12 angular_radius
    f[0] = dispatch.nu;
    f[1] = dispatch.spot_center_theta;
    f[2] = dispatch.inc;
    f[3] = dispatch.angular_radius;
    //  16 M    20 Re    24 kT    28 E_obs
    f[4] = constants.M;
    f[5] = constants.Re;
    f[6] = dispatch.kT ?? constants.kT;
    f[7] = constants.E_obs;
    //  32 Ibb_div_D2    36 beaming    40 point_source    44 n_rings
    f[8] = Ibb_div_D2;
    u[9] = dispatch.beaming;
    u[10] = point_source;
    u[11] = n_rings;
    //  48 u_min   52 u_max   56 cos_psi_min   60 cos_psi_max
    f[12] = h.u.min;
    f[13] = h.u.max;
    f[14] = h.cos_psi.min;
    f[15] = h.cos_psi.max;
    //  64 cos_alpha_min   68 cos_alpha_max   72 N_u   76 N_cos_psi
    f[16] = h.cos_alpha.min;
    f[17] = h.cos_alpha.max;
    u[18] = h.u.n;
    u[19] = h.cos_psi.n;
    //  80 N_cos_alpha   84 clip_enabled   88 _pad0   92 _pad1
    u[20] = h.cos_alpha.n;
    u[21] = dispatch.clip ? 1 : 0;
    //  96 clip_center_theta   100 clip_center_phi   104 clip_cos_angular_radius   108 _pad2
    if (dispatch.clip) {
      f[24] = dispatch.clip.center_theta;
      f[25] = dispatch.clip.center_phi;  // relative to this spot's phi (spot is at φ=0)
      f[26] = Math.cos(dispatch.clip.angular_radius);
    }

    device.queue.writeBuffer(paramsBuffer, 0, ab);
    return { n_rings };
  }

  async function computePulseProfile(dispatch, constants) {
    const { n_rings } = writeParams(dispatch, constants);

    const encoder = device.createCommandEncoder({ label: "compute" });
    {
      const pass = encoder.beginComputePass({ label: "main" });
      pass.setPipeline(mainPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(n_rings);
      pass.end();
    }
    {
      const pass = encoder.beginComputePass({ label: "sum_rings" });
      pass.setPipeline(sumPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(1);
      pass.end();
    }
    encoder.copyBufferToBuffer(totalFluxBuffer, 0, readbackBuffer, 0, N_OUTPUT_PHASE * 4);
    device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();
    return result;
  }

  // spots: array of { theta, phi, rho, mode, kT }.  shared: { nu, inc, beaming, constants }.
  async function computeMultiSpot(spots, shared) {
    const adds = spots.filter((s) => s.mode === "ADD");
    const subs = spots.filter((s) => s.mode === "SUB");
    const constants = shared.constants ?? OS1_DEFAULTS;

    const total = new Float32Array(N_OUTPUT_PHASE);

    // --- ADDs ---
    for (const a of adds) {
      const flux = await computePulseProfile(
        {
          nu: shared.nu,
          inc: shared.inc,
          spot_center_theta: a.theta,
          angular_radius: a.rho,
          beaming: shared.beaming,
          kT: a.kT,
        },
        constants,
      );
      const shifted = phaseShift(flux, a.phi);
      for (let i = 0; i < N_OUTPUT_PHASE; i++) total[i] += shifted[i];
    }

    // --- SUBTRACTs ---
    for (const s of subs) {
      const parent = findParent(adds, s);
      if (!parent) continue;   // no parent → SUBTRACT is inert
      const flux = await computePulseProfile(
        {
          nu: shared.nu,
          inc: shared.inc,
          spot_center_theta: s.theta,
          angular_radius: s.rho,
          beaming: shared.beaming,
          kT: parent.kT,         // inherit parent kT
          clip: {
            center_theta: parent.theta,
            center_phi: parent.phi - s.phi,   // in the spot's frame (spot at φ=0)
            angular_radius: parent.rho,
          },
        },
        constants,
      );
      const shifted = phaseShift(flux, s.phi);
      for (let i = 0; i < N_OUTPUT_PHASE; i++) total[i] -= shifted[i];
    }
    return total;
  }

  return {
    computePulseProfile,
    computeMultiSpot,
    N_OUTPUT_PHASE,
  };
}
