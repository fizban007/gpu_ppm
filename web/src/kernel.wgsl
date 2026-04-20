// Port of gpu/os1_gpu.cu:KernelFunc to WGSL.
//
// Dispatch: (n_rings, 1, 1) workgroups of 256 threads.
//   n_rings = 1                              for point-source (angular_radius < 0.1)
//   n_rings = 4 * N_side - 1 = 1023          for full HEALPix tiling
//
// Each workgroup accumulates into its own 128-slot region of per_ring_flux.
// A second pass (sum_rings) reduces across rings into total_flux.

// ---------------- Constants ----------------

const PI: f32                = 3.141592653589793;
const TWO_PI: f32            = 6.283185307179586;
const C_KM_S: f32            = 299792.458;
const RS_SUN_KM: f32         = 2.9532501;
const SQRT_KM3_GMSUN: f32    = 2.745011592867327e-6;
// Ibb_constant / D² (see Ibb_div_D2 computation in compute.js) is passed in
// via the uniform, because h³·c²·kpc² under/overflows f32.

const N_SIDE: u32            = 256u;
const BLOCK_DIM: u32         = 256u;
const N_OUTPUT_PHASE: u32    = 128u;
const N_FINE_PHASE: u32      = 384u;    // N_OUTPUT_PHASE * 3
const N_FINE_PLUS_1: u32     = 385u;
const MAX_RING_SIZE: u32     = 1024u;   // 4 * N_SIDE

// ---------------- Bindings ----------------

struct Params {
    nu: f32,
    spot_center_theta: f32,
    inc: f32,
    angular_radius: f32,

    M: f32,
    Re: f32,
    kT: f32,
    E_obs: f32,

    Ibb_div_D2: f32,      // precomputed 2 / (h³ c² (D·kpc_in_km)²)
    beaming: u32,         // 0 = iso, 1 = cos², 2 = 1-cos²
    point_source: u32,    // 0 or 1
    n_rings: u32,

    u_min: f32,
    u_max: f32,
    cos_psi_min: f32,
    cos_psi_max: f32,

    cos_alpha_min: f32,
    cos_alpha_max: f32,
    N_u: u32,
    N_cos_psi: u32,

    N_cos_alpha: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform>                   P: Params;
@group(0) @binding(1) var<storage, read>             cos_alpha_of_u_cos_psi:   array<f32>;
@group(0) @binding(2) var<storage, read>             lf_of_u_cos_psi:          array<f32>;
@group(0) @binding(3) var<storage, read>             cdt_over_R_of_u_cos_alpha: array<f32>;
@group(0) @binding(4) var<storage, read_write>       per_ring_flux:            array<f32>;
@group(0) @binding(5) var<storage, read_write>       total_flux:               array<f32>;

// ---------------- Workgroup storage ----------------

var<workgroup> wg_cos_theta:          f32;
var<workgroup> wg_sin_theta:          f32;
var<workgroup> wg_cos_center_theta:   f32;
var<workgroup> wg_sin_center_theta:   f32;
var<workgroup> wg_cos_angular_radius: f32;
var<workgroup> wg_R:                  f32;
var<workgroup> wg_u:                  f32;
var<workgroup> wg_uu:                 f32;
var<workgroup> wg_cos_eta:            f32;
var<workgroup> wg_sin_eta:            f32;
var<workgroup> wg_dS:                 f32;
var<workgroup> wg_cos_i:              f32;
var<workgroup> wg_sin_i:              f32;
var<workgroup> wg_cc:                 f32;
var<workgroup> wg_ss:                 f32;
var<workgroup> wg_Rs:                 f32;

var<workgroup> wg_N_active_phi:       atomic<u32>;
var<workgroup> wg_active_phi:         array<f32, 1024>;       // MAX_RING_SIZE
var<workgroup> wg_fluxes_over_I:      array<f32, 385>;        // N_FINE_PHASE + 1
var<workgroup> wg_redshift_factors:   array<f32, 385>;
var<workgroup> wg_phase_o:            array<f32, 385>;
var<workgroup> wg_reduce:             array<i32, 256>;        // BLOCK_DIM

// ---------------- HEALPix ----------------

fn healpix_cos_theta_ring(i: u32) -> f32 {
    let Ns = i32(N_SIDE);
    var ii = i32(i);
    var minus: bool = false;
    if (ii > 2 * Ns) { ii = 4 * Ns - ii; minus = true; }
    var r: f32;
    if (ii < Ns) {
        r = 1.0 - f32(ii * ii) / (3.0 * f32(Ns * Ns));
    } else {
        r = 4.0 / 3.0 - 2.0 * f32(ii) / (3.0 * f32(Ns));
    }
    if (minus) { r = -r; }
    return r;
}

fn healpix_j_max(i: u32) -> u32 {
    let Ns = i32(N_SIDE);
    let ii = i32(i);
    var t = Ns;
    if (t > ii) { t = ii; }
    if (t > 4 * Ns - ii) { t = 4 * Ns - ii; }
    return u32(4 * t);
}

fn healpix_phi(i: u32, j: u32) -> f32 {
    let Ns = i32(N_SIDE);
    var ii = i32(i);
    if (ii > 4 * Ns - ii) { ii = 4 * Ns - ii; }
    if (ii < Ns) {
        return PI / (2.0 * f32(ii)) * (f32(j) - 0.5);
    } else {
        let s = (ii - Ns + 1) % 2;
        return PI / (2.0 * f32(N_SIDE)) * (f32(j) - 0.5 * f32(s));
    }
}

fn healpix_dOmega() -> f32 {
    return PI / (3.0 * f32(N_SIDE * N_SIDE));
}

// ---------------- Lensing table interpolation ----------------

struct CaLf { cos_alpha: f32, lf: f32 };

fn ca_lf_of_u_cos_psi(u_val: f32, cos_psi: f32) -> CaLf {
    let nu_m1  = f32(P.N_u - 1u);
    let ncp_m1 = f32(P.N_cos_psi - 1u);

    let tu = (u_val - P.u_min) / (P.u_max - P.u_min) * nu_m1;
    let tu_c = clamp(tu, 0.0, nu_m1 - 1e-5);
    let i_u = u32(tu_c);
    let a_u = tu_c - f32(i_u);
    let b_u = 1.0 - a_u;

    let tp = (cos_psi - P.cos_psi_min) / (P.cos_psi_max - P.cos_psi_min) * ncp_m1;
    let tp_c = clamp(tp, 0.0, ncp_m1 - 1e-5);
    let i_p = u32(tp_c);
    let a_p = tp_c - f32(i_p);
    let b_p = 1.0 - a_p;

    let row_stride = P.N_cos_psi;
    let idx00 = i_u * row_stride + i_p;
    let idx01 = i_u * row_stride + (i_p + 1u);
    let idx10 = (i_u + 1u) * row_stride + i_p;
    let idx11 = (i_u + 1u) * row_stride + (i_p + 1u);

    let ca = b_u * b_p * cos_alpha_of_u_cos_psi[idx00]
           + a_u * b_p * cos_alpha_of_u_cos_psi[idx10]
           + b_u * a_p * cos_alpha_of_u_cos_psi[idx01]
           + a_u * a_p * cos_alpha_of_u_cos_psi[idx11];
    let lf = b_u * b_p * lf_of_u_cos_psi[idx00]
           + a_u * b_p * lf_of_u_cos_psi[idx10]
           + b_u * a_p * lf_of_u_cos_psi[idx01]
           + a_u * a_p * lf_of_u_cos_psi[idx11];
    return CaLf(ca, lf);
}

fn cdt_over_R_of_u_ca(u_val: f32, cos_alpha: f32) -> f32 {
    let nu_m1  = f32(P.N_u - 1u);
    let nca_m1 = f32(P.N_cos_alpha - 1u);

    let tu = (u_val - P.u_min) / (P.u_max - P.u_min) * nu_m1;
    let tu_c = clamp(tu, 0.0, nu_m1 - 1e-5);
    let i_u = u32(tu_c);
    let a_u = tu_c - f32(i_u);
    let b_u = 1.0 - a_u;

    let tc = (cos_alpha - P.cos_alpha_min) / (P.cos_alpha_max - P.cos_alpha_min) * nca_m1;
    let tc_c = clamp(tc, 0.0, nca_m1 - 1e-5);
    let i_c = u32(tc_c);
    let a_c = tc_c - f32(i_c);
    let b_c = 1.0 - a_c;

    let row_stride = P.N_cos_alpha;
    let idx00 = i_u * row_stride + i_c;
    let idx01 = i_u * row_stride + (i_c + 1u);
    let idx10 = (i_u + 1u) * row_stride + i_c;
    let idx11 = (i_u + 1u) * row_stride + (i_c + 1u);

    return b_u * b_c * cdt_over_R_of_u_cos_alpha[idx00]
         + a_u * b_c * cdt_over_R_of_u_cos_alpha[idx10]
         + b_u * a_c * cdt_over_R_of_u_cos_alpha[idx01]
         + a_u * a_c * cdt_over_R_of_u_cos_alpha[idx11];
}

// ---------------- Physics helpers ----------------

fn cal_dt2(RR: f32, Re_km: f32, Rs_km: f32) -> f32 {
    let lg = log(1.0 + (Re_km - RR) / (RR - Rs_km));
    let cdt = Re_km - RR + Rs_km * lg;
    return cdt / C_KM_S;
}

fn blackbody_I(E: f32, kT: f32) -> f32 {
    return (E * E * E) / (exp(E / kT) - 1.0);
}

// ---------------- Hunt on wg_phase_o (assumes ascending) ----------------

struct HuntOut { i: i32, a: f32 };

fn hunt_phase_o(x: f32, jsav_ptr: ptr<function, i32>) -> HuntOut {
    let n = i32(N_FINE_PLUS_1);
    var jl: i32 = *jsav_ptr;
    var ju: i32;
    var jm: i32;
    var inc: i32 = 1;

    if (jl < 0 || jl > n - 1) {
        jl = 0;
        ju = n - 1;
    } else {
        if (x >= wg_phase_o[jl]) {
            loop {
                ju = jl + inc;
                if (ju >= n - 1) { ju = n - 1; break; }
                else if (x < wg_phase_o[ju]) { break; }
                else { jl = ju; inc = inc + inc; }
            }
        } else {
            ju = jl;
            loop {
                jl = jl - inc;
                if (jl <= 0) { jl = 0; break; }
                else if (x >= wg_phase_o[jl]) { break; }
                else { ju = jl; inc = inc + inc; }
            }
        }
    }
    loop {
        if (ju - jl <= 1) { break; }
        jm = (ju + jl) >> 1u;
        if (x >= wg_phase_o[jm]) { jl = jm; } else { ju = jm; }
    }
    *jsav_ptr = jl;
    let i = clamp(jl, 0, n - 2);
    let a = (x - wg_phase_o[i]) / (wg_phase_o[i + 1] - wg_phase_o[i]);
    return HuntOut(i, a);
}

// ---------------- Block reductions ----------------

fn reduce_min_i32(tid: u32, value: i32) -> i32 {
    wg_reduce[tid] = value;
    workgroupBarrier();
    var stride: u32 = BLOCK_DIM / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            let other = wg_reduce[tid + stride];
            if (other < wg_reduce[tid]) { wg_reduce[tid] = other; }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    return wg_reduce[0];
}

fn reduce_max_i32(tid: u32, value: i32) -> i32 {
    wg_reduce[tid] = value;
    workgroupBarrier();
    var stride: u32 = BLOCK_DIM / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            let other = wg_reduce[tid + stride];
            if (other > wg_reduce[tid]) { wg_reduce[tid] = other; }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    return wg_reduce[0];
}

fn clear_output_bins(tid: u32, wg_idx: u32) {
    var i = tid;
    loop {
        if (i >= N_OUTPUT_PHASE) { break; }
        per_ring_flux[wg_idx * N_OUTPUT_PHASE + i] = 0.0;
        i = i + BLOCK_DIM;
    }
}

// ---------------- Main kernel ----------------

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let wg_idx: u32 = wg_id.x;
    let tid: u32    = local_id.x;

    // ---- Thread-0 init of per-ring state ----
    if (tid == 0u) {
        atomicStore(&wg_N_active_phi, 0u);

        let healpix_i = wg_idx + 1u;
        let ct = select(
            healpix_cos_theta_ring(healpix_i),
            cos(P.spot_center_theta),
            P.point_source == 1u,
        );
        wg_cos_theta          = ct;
        wg_sin_theta          = sqrt(max(0.0, 1.0 - ct * ct));
        wg_cos_center_theta   = cos(P.spot_center_theta);
        wg_sin_center_theta   = sin(P.spot_center_theta);
        wg_cos_angular_radius = cos(P.angular_radius);

        let Rs = P.M * RS_SUN_KM;
        wg_Rs = Rs;
        let x_compact = 0.5 * Rs / P.Re;
        let Omega_bar = TWO_PI * P.nu * sqrt(P.Re * P.Re * P.Re / P.M) * SQRT_KM3_GMSUN;
        let o2 = Omega_bar * Omega_bar * (-0.788 + 1.030 * x_compact);
        let R_val = P.Re * (1.0 + o2 * ct * ct);
        let dRdtheta = -2.0 * P.Re * o2 * ct * wg_sin_theta;
        let u_val = Rs / R_val;
        wg_R  = R_val;
        wg_u  = u_val;
        wg_uu = sqrt(1.0 - u_val);

        let f_geom = dRdtheta / (R_val * wg_uu);
        let ff = sqrt(1.0 + f_geom * f_geom);
        wg_cos_eta = 1.0 / ff;
        wg_sin_eta = f_geom / ff;

        let sin_half_ar = sin(P.angular_radius * 0.5);
        let dOmega = select(
            healpix_dOmega(),
            TWO_PI * (2.0 * sin_half_ar * sin_half_ar),
            P.point_source == 1u,
        );
        wg_dS = R_val * R_val * dOmega * ff;

        wg_cos_i = cos(P.inc);
        wg_sin_i = sin(P.inc);
        wg_cc = wg_cos_i * ct;
        wg_ss = wg_sin_i * wg_sin_theta;
    }
    workgroupBarrier();

    // ---- Fill active_phi ----
    if (P.point_source == 1u) {
        if (tid == 0u) {
            wg_active_phi[0] = 0.0;
            atomicStore(&wg_N_active_phi, 1u);
        }
    } else {
        let healpix_i = wg_idx + 1u;
        let jmax = healpix_j_max(healpix_i);
        var hj = tid + 1u;
        loop {
            if (hj > jmax) { break; }
            let phi = healpix_phi(healpix_i, hj);
            let cos_rho = wg_sin_theta * wg_sin_center_theta * cos(phi)
                        + wg_cos_theta * wg_cos_center_theta;
            if (cos_rho > wg_cos_angular_radius) {
                let idx = atomicAdd(&wg_N_active_phi, 1u);
                wg_active_phi[idx] = phi;
            }
            hj = hj + BLOCK_DIM;
        }
    }
    workgroupBarrier();

    let N_active = atomicLoad(&wg_N_active_phi);

    if (N_active == 0u) {
        clear_output_bins(tid, wg_idx);
        return;
    }

    // ---- Fine-phase sweep ----
    var local_inv_min: i32 = i32(N_FINE_PHASE) + 1;
    var local_inv_max: i32 = -1;

    var ip = tid;
    loop {
        if (ip >= N_FINE_PHASE) { break; }
        let phase_ratio = f32(ip) / f32(N_FINE_PHASE);
        let phi_o = TWO_PI * phase_ratio;
        let cos_phi = cos(phi_o);
        let sin_phi = sin(phi_o);
        let cos_psi = wg_cc + wg_ss * cos_phi;

        var is_visible = true;
        var flux_val: f32     = 0.0;
        var redshift_val: f32 = -1.0;
        var phase_o_val: f32  = -1.0;

        if (cos_psi < P.cos_psi_min) { is_visible = false; }

        if (is_visible) {
            let lres = ca_lf_of_u_cos_psi(wg_u, cos_psi);
            let cos_alpha = lres.cos_alpha;
            let lf = lres.lf;
            let sin_alpha = sqrt(max(0.0, 1.0 - cos_alpha * cos_alpha));
            var sin_alpha_over_sin_psi: f32;
            if (cos_psi >= 1.0) {
                sin_alpha_over_sin_psi = sqrt(lf);
            } else {
                let sin_psi = sqrt(max(0.0, 1.0 - cos_psi * cos_psi));
                sin_alpha_over_sin_psi = sin_alpha / sin_psi;
            }
            let cos_sigma = cos_alpha * wg_cos_eta
                + sin_alpha_over_sin_psi * wg_sin_eta
                  * (wg_cos_i * wg_sin_theta - wg_sin_i * wg_cos_theta * cos_phi);

            if (cos_sigma <= 0.0) {
                is_visible = false;
            } else {
                let beta = TWO_PI * P.nu * wg_R * wg_sin_theta / wg_uu / C_KM_S;
                let gamma = 1.0 / sqrt(1.0 - beta * beta);
                let cos_xi = -sin_alpha_over_sin_psi * wg_sin_i * sin_phi;
                let delta = 1.0 / (gamma * (1.0 - beta * cos_xi));
                let delta3 = delta * delta * delta;

                let cdt_over_R = cdt_over_R_of_u_ca(wg_u, cos_alpha);
                let dt1 = cdt_over_R * wg_R / C_KM_S;

                var dt2: f32 = 0.0;
                if (cos_alpha >= 0.0) {
                    dt2 = cal_dt2(wg_R, P.Re, wg_Rs);
                } else {
                    let tmp = (2.0 * sin_alpha) / sqrt(3.0 * (1.0 - wg_u));
                    let arg = 3.0 * wg_u / tmp;
                    let p_over_R = -tmp * cos((acos(arg) + TWO_PI) / 3.0);
                    let p = p_over_R * wg_R;
                    dt2 = 2.0 * cal_dt2(p, P.Re, wg_Rs) - cal_dt2(wg_R, P.Re, wg_Rs);
                }
                let delta_phase = (dt1 + dt2) * P.nu;

                let cos_sigma_prime = cos_sigma * delta;
                var beaming_factor: f32 = 1.0;
                if (P.beaming == 1u) {
                    beaming_factor = cos_sigma_prime * cos_sigma_prime;
                } else if (P.beaming == 2u) {
                    beaming_factor = 1.0 - cos_sigma_prime * cos_sigma_prime;
                }

                flux_val = wg_uu * delta3 * cos_sigma_prime * lf * gamma
                         * wg_dS * P.Ibb_div_D2 * beaming_factor;
                redshift_val = 1.0 / (delta * wg_uu);
                phase_o_val  = phase_ratio + delta_phase;
            }
        }

        wg_fluxes_over_I[ip]    = flux_val;
        wg_redshift_factors[ip] = redshift_val;
        wg_phase_o[ip]          = phase_o_val;

        if (!is_visible) {
            if (i32(ip) < local_inv_min) { local_inv_min = i32(ip); }
            if (i32(ip) > local_inv_max) { local_inv_max = i32(ip); }
        }

        ip = ip + BLOCK_DIM;
    }
    workgroupBarrier();

    // ---- Reductions for invisible range ----
    let inv_min = reduce_min_i32(tid, local_inv_min);
    workgroupBarrier();
    let inv_max = reduce_max_i32(tid, local_inv_max);
    workgroupBarrier();

    if (inv_min == 0 || inv_max == i32(N_FINE_PHASE) - 1) {
        clear_output_bins(tid, wg_idx);
        return;
    }

    // ---- Fill invisible phases by linear interp ----
    if (inv_min <= inv_max) {
        let beg = inv_min - 1;
        let end = inv_max + 1;
        let phase0 = wg_phase_o[beg];
        let phase1 = wg_phase_o[end];
        let rf0 = wg_redshift_factors[beg];
        let rf1 = wg_redshift_factors[end];
        let denom = f32(end - beg);
        var ii = i32(tid) + inv_min;
        loop {
            if (ii > inv_max) { break; }
            let t = f32(ii - beg) / denom;
            wg_phase_o[ii]          = phase0 + (phase1 - phase0) * t;
            wg_redshift_factors[ii] = rf0    + (rf1    - rf0)    * t;
            ii = ii + i32(BLOCK_DIM);
        }
    }
    workgroupBarrier();

    // ---- Periodic closure ----
    if (tid == 0u) {
        wg_fluxes_over_I[N_FINE_PHASE]    = wg_fluxes_over_I[0];
        wg_redshift_factors[N_FINE_PHASE] = wg_redshift_factors[0];
        wg_phase_o[N_FINE_PHASE]          = 1.0 + wg_phase_o[0];
    }
    workgroupBarrier();

    // ---- Integrate into output phase bins ----
    // Threads 0..N_OUTPUT_PHASE-1 each own one output bin, summing over all active_phi.
    if (tid < N_OUTPUT_PHASE) {
        let target_phase = f32(tid) / f32(N_OUTPUT_PHASE);
        var sum: f32 = 0.0;
        var jsav: i32 = 0;

        for (var ia: u32 = 0u; ia < N_active; ia = ia + 1u) {
            let phase_shift = wg_active_phi[ia] / TWO_PI;
            let raw = target_phase + phase_shift + 1.0;
            var patch_phase = raw - floor(raw);
            if (patch_phase < wg_phase_o[0]) { patch_phase = patch_phase + 1.0; }
            let h = hunt_phase_o(patch_phase, &jsav);
            let flux_i = (1.0 - h.a) * wg_fluxes_over_I[h.i]
                       +        h.a  * wg_fluxes_over_I[h.i + 1];
            let rf     = (1.0 - h.a) * wg_redshift_factors[h.i]
                       +        h.a  * wg_redshift_factors[h.i + 1];
            let E_emit = P.E_obs * rf;
            sum = sum + flux_i * blackbody_I(E_emit, P.kT);
        }

        per_ring_flux[wg_idx * N_OUTPUT_PHASE + tid] = sum;
    }
}

// ---------------- Ring-sum reduction ----------------

@compute @workgroup_size(128)
fn sum_rings(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i: u32 = lid.x;
    if (i >= N_OUTPUT_PHASE) { return; }
    var s: f32 = 0.0;
    let n = P.n_rings;
    for (var r: u32 = 0u; r < n; r = r + 1u) {
        s = s + per_ring_flux[r * N_OUTPUT_PHASE + i];
    }
    total_flux[i] = s;
}
