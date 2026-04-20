// WebGPU compute pipeline for pulse-profile calculation.
// Mirrors gpu/os1_gpu.cu:KernelFunc via web/src/kernel.wgsl.

const N_SIDE = 256;
const N_OUTPUT_PHASE = 128;
const N_RINGS_FULL = 4 * N_SIDE - 1; // 1023
const PARAMS_SIZE = 96;              // bytes; matches Params struct in kernel.wgsl

export async function createPulseEngine(device, lensing) {
  const shaderCode = await fetch("src/kernel.wgsl").then((r) => {
    if (!r.ok) throw new Error(`fetch kernel.wgsl: ${r.status}`);
    return r.text();
  });
  const module = device.createShaderModule({ label: "kernel.wgsl", code: shaderCode });

  // Block until compilation info is available — surface WGSL errors clearly.
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

  // Buffers that are reused across dispatches:
  const paramsBuffer = device.createBuffer({
    label: "params",
    size: PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // per_ring_flux is sized for the worst case (full HEALPix tiling).
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

  function writeParams(scenario, constants) {
    const h = lensing.header;
    const point_source = scenario.angular_radius < 0.1 ? 1 : 0;
    const n_rings = point_source ? 1 : N_RINGS_FULL;

    // Ibb_div_D2 = 2 / (h³ c² D_km²)  [computed in f64, narrowed to f32]
    const H_KEV_S = 4.135668e-18;
    const C_CM_S = 2.99792458e10;
    const KPC_KM = 3.08567758e16;
    const D_km = constants.D * KPC_KM;
    const Ibb_div_D2 =
      2.0 / (H_KEV_S * H_KEV_S * H_KEV_S * C_CM_S * C_CM_S * D_km * D_km);

    const ab = new ArrayBuffer(PARAMS_SIZE);
    const f = new Float32Array(ab);
    const u = new Uint32Array(ab);

    //   0 nu
    //   4 spot_center_theta
    //   8 inc
    //  12 angular_radius
    f[0] = scenario.nu;
    f[1] = scenario.spot_center_theta;
    f[2] = scenario.inc;
    f[3] = scenario.angular_radius;
    //  16 M    20 Re    24 kT    28 E_obs
    f[4] = constants.M;
    f[5] = constants.Re;
    f[6] = constants.kT;
    f[7] = constants.E_obs;
    //  32 Ibb_div_D2    36 beaming    40 point_source    44 n_rings
    f[8] = Ibb_div_D2;
    u[9] = scenario.beaming;
    u[10] = point_source;
    u[11] = n_rings;
    //  48 u_min  52 u_max  56 cos_psi_min  60 cos_psi_max
    f[12] = h.u.min;
    f[13] = h.u.max;
    f[14] = h.cos_psi.min;
    f[15] = h.cos_psi.max;
    //  64 cos_alpha_min  68 cos_alpha_max  72 N_u  76 N_cos_psi
    f[16] = h.cos_alpha.min;
    f[17] = h.cos_alpha.max;
    u[18] = h.u.n;
    u[19] = h.cos_psi.n;
    //  80 N_cos_alpha  84..92 pad
    u[20] = h.cos_alpha.n;

    device.queue.writeBuffer(paramsBuffer, 0, ab);
    return { n_rings };
  }

  async function computePulseProfile(scenario, constants = OS1_DEFAULTS) {
    const { n_rings } = writeParams(scenario, constants);

    const encoder = device.createCommandEncoder({ label: `compute:${scenario.name}` });
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

  return {
    computePulseProfile,
    N_OUTPUT_PHASE,
  };
}

export const OS1_DEFAULTS = {
  M: 1.4,
  Re: 12.0,
  kT: 0.35,
  E_obs: 1.0,
  D: 0.2,
};
