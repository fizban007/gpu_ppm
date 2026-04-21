// Pipeline for the dipole-derived temperature-map kernel.
// Mirrors createPulseEngine in compute.js, specialized for the single-
// dispatch-per-update dipole model (no per-spot loop, no phase shift —
// the magnetic axis lives in the co-rotating frame at φ=0 by construction).

const N_SIDE = 256;
const N_OUTPUT_PHASE = 128;
const N_RINGS_FULL = 4 * N_SIDE - 1;
const PARAMS_SIZE = 80;    // matches Params layout in kernel_dipole.wgsl

function ibbOverD2(D_kpc) {
  const H_KEV_S = 4.135668e-18;
  const C_CM_S = 2.99792458e10;
  const KPC_KM = 3.08567758e16;
  const D_km = D_kpc * KPC_KM;
  return 2.0 / (H_KEV_S * H_KEV_S * H_KEV_S * C_CM_S * C_CM_S * D_km * D_km);
}

export async function createDipoleEngine(device, lensing) {
  const shaderCode = await fetch("src/kernel_dipole.wgsl").then((r) => {
    if (!r.ok) throw new Error(`fetch kernel_dipole.wgsl: ${r.status}`);
    return r.text();
  });
  const module = device.createShaderModule({ label: "kernel_dipole.wgsl", code: shaderCode });

  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === "error");
  if (errors.length) {
    const msg = errors.map((e) => `${e.lineNum}:${e.linePos}  ${e.message}`).join("\n");
    throw new Error(`kernel_dipole.wgsl compile errors:\n${msg}`);
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
    label: "dipole main",
    layout: pipelineLayout,
    compute: { module, entryPoint: "main" },
  });
  const sumPipeline = device.createComputePipeline({
    label: "dipole sum_rings",
    layout: pipelineLayout,
    compute: { module, entryPoint: "sum_rings" },
  });

  const paramsBuffer = device.createBuffer({
    label: "dipole-params",
    size: PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const perRingBuffer = device.createBuffer({
    label: "dipole-per-ring",
    size: N_RINGS_FULL * N_OUTPUT_PHASE * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const totalFluxBuffer = device.createBuffer({
    label: "dipole-total-flux",
    size: N_OUTPUT_PHASE * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readbackBuffer = device.createBuffer({
    label: "dipole-readback",
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

  function writeParams(dipole, constants) {
    const h = lensing.header;
    const Ibb_div_D2 = ibbOverD2(constants.D);

    const ab = new ArrayBuffer(PARAMS_SIZE);
    const f = new Float32Array(ab);
    const u = new Uint32Array(ab);

    //   0 nu    4 mag_incl    8 obs_incl    12 T0
    f[0] = dipole.nu;
    f[1] = dipole.mag_incl;
    f[2] = dipole.obs_incl;
    f[3] = dipole.T0;
    //  16 M    20 Re    24 E_obs    28 Ibb_div_D2
    f[4] = constants.M;
    f[5] = constants.Re;
    f[6] = constants.E_obs;
    f[7] = Ibb_div_D2;
    //  32 u_min   36 u_max   40 cos_psi_min   44 cos_psi_max
    f[8] = h.u.min;
    f[9] = h.u.max;
    f[10] = h.cos_psi.min;
    f[11] = h.cos_psi.max;
    //  48 cos_alpha_min  52 cos_alpha_max  56 N_u   60 N_cos_psi
    f[12] = h.cos_alpha.min;
    f[13] = h.cos_alpha.max;
    u[14] = h.u.n;
    u[15] = h.cos_psi.n;
    //  64 N_cos_alpha  68 beaming  72 n_rings  76 pad
    u[16] = h.cos_alpha.n;
    u[17] = dipole.beaming ?? 0;
    u[18] = N_RINGS_FULL;

    device.queue.writeBuffer(paramsBuffer, 0, ab);
  }

  async function computeDipoleProfile(dipole, constants) {
    writeParams(dipole, constants);

    const encoder = device.createCommandEncoder({ label: "compute-dipole" });
    {
      const pass = encoder.beginComputePass({ label: "main" });
      pass.setPipeline(mainPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(N_RINGS_FULL);
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

  return { computeDipoleProfile, N_OUTPUT_PHASE };
}
