// Physics-derived temperature-map kernel.
//
// Surface emission is set by a centered magnetic dipole at inclination ι:
//   α = −μ/r · sin²θ',   β = φ',
// with (θ', φ') the body-frame coordinates of a lab-frame point (θ, φ).
// The polar cap is the region |α| < α₀ with
//   α₀ = √(3/2) μΩ (1 + sin²ι/5)  (natural units, c=1).
// Inside it, we set kT = T₀ · |j_dim|^(1/4), where |j_dim| is the 3-current
// magnitude computed from Eqs. 4–6 of Huang & Chen (2025). Outside the cap,
// no emission. Return-current region (ρ_GJ·j_r < 0) is skipped for this
// Tier-1 implementation.
//
// Shares structure with kernel.wgsl — lensing / Doppler / time-delay math
// and the HEALPix ring iteration are copied verbatim; only the active-patch
// filter and the per-patch kT are new.

// ---------------- Constants ----------------

const PI: f32                = 3.141592653589793;
const TWO_PI: f32            = 6.283185307179586;
const C_KM_S: f32            = 299792.458;
const RS_SUN_KM: f32         = 2.9532501;
const SQRT_KM3_GMSUN: f32    = 2.745011592867327e-6;
const SQRT_3_HALVES: f32     = 1.224744871391589;

const N_SIDE: u32            = 256u;
const BLOCK_DIM: u32         = 256u;
const N_OUTPUT_PHASE: u32    = 128u;
const N_FINE_PHASE: u32      = 384u;
const N_FINE_PLUS_1: u32     = 385u;
const MAX_RING_SIZE: u32     = 1024u;

const N_PHI_ITER:   u32 = 4u;
const N_SWEEP_ITER: u32 = 2u;
const N_FILL_ITER:  u32 = 2u;

const FD_EPS: f32 = 1e-3;                 // finite-difference step for α, β derivatives

// ---------------- Bindings ----------------

struct Params {
    nu: f32,              // spin frequency, Hz
    mag_incl: f32,        // ι (rad) — magnetic-axis inclination
    obs_incl: f32,        // I (rad) — observer inclination
    T0: f32,              // peak temperature scale (keV)

    M: f32,               // solar masses
    Re: f32,              // km (equatorial radius)
    E_obs: f32,           // keV
    Ibb_div_D2: f32,

    u_min: f32,
    u_max: f32,
    cos_psi_min: f32,
    cos_psi_max: f32,

    cos_alpha_min: f32,
    cos_alpha_max: f32,
    N_u: u32,
    N_cos_psi: u32,

    N_cos_alpha: u32,
    beaming: u32,
    n_rings: u32,
    _pad0: u32,

    shift_x: f32,
    shift_y: f32,
    shift_z: f32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform>                   P: Params;
@group(0) @binding(1) var<storage, read>             cos_alpha_of_u_cos_psi:    array<f32>;
@group(0) @binding(2) var<storage, read>             lf_of_u_cos_psi:           array<f32>;
@group(0) @binding(3) var<storage, read>             cdt_over_R_of_u_cos_alpha: array<f32>;
@group(0) @binding(4) var<storage, read_write>       per_ring_flux:             array<f32>;
@group(0) @binding(5) var<storage, read_write>       total_flux:                array<f32>;

// ---------------- Workgroup storage ----------------

var<workgroup> wg_cos_theta:        f32;
var<workgroup> wg_sin_theta:        f32;
var<workgroup> wg_R:                f32;
var<workgroup> wg_u:                f32;
var<workgroup> wg_uu:               f32;
var<workgroup> wg_cos_eta:          f32;
var<workgroup> wg_sin_eta:          f32;
var<workgroup> wg_dS:               f32;
var<workgroup> wg_cos_i:            f32;
var<workgroup> wg_sin_i:            f32;
var<workgroup> wg_cc:               f32;
var<workgroup> wg_ss:               f32;
var<workgroup> wg_Rs:               f32;

// Dipole-specific per-ring state
var<workgroup> wg_cos_iota:         f32;
var<workgroup> wg_sin_iota:         f32;
var<workgroup> wg_Omega:            f32;   // 2πν, rad/s
var<workgroup> wg_alpha0_dim:       f32;   // α₀ R/μ = √(3/2) (ΩR/c) (1 + sin²ι/5)
var<workgroup> wg_cos_theta_cap:    f32;   // cos of polar-cap angular radius (for ring fast-path)

var<workgroup> wg_N_active_phi:     atomic<u32>;
var<workgroup> wg_active_phi:       array<f32, 1024>;
var<workgroup> wg_active_kT:        array<f32, 1024>;
var<workgroup> wg_fluxes_over_I:    array<f32, 385>;
var<workgroup> wg_redshift_factors: array<f32, 385>;
var<workgroup> wg_phase_o:          array<f32, 385>;
var<workgroup> wg_reduce:           array<i32, 256>;

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

// ---------------- Lensing interpolation (verbatim) ----------------

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

// ---------------- Physics helpers (verbatim) ----------------

fn cal_dt2(RR: f32, Re_km: f32, Rs_km: f32) -> f32 {
    let lg = log(1.0 + (Re_km - RR) / (RR - Rs_km));
    let cdt = Re_km - RR + Rs_km * lg;
    return cdt / C_KM_S;
}

fn blackbody_I(E: f32, kT: f32) -> f32 {
    return (E * E * E) / (exp(E / kT) - 1.0);
}

// ---------------- Dipole math ----------------

// Lab-frame spherical (r_s, θ_s, φ_s) → shifted Cartesian (x, y, z) relative
// to the magnetic-dipole center at (shift_x, shift_y, shift_z). Tier 2 extends
// Tier 1 by making every dipole quantity a function of these three lab coords
// rather than assuming the centered r=R=1 convention.
fn shifted_cart(r_s: f32, theta_s: f32, phi_s: f32) -> vec3<f32> {
    let st = sin(theta_s);
    return vec3<f32>(
        r_s * st * cos(phi_s) - P.shift_x,
        r_s * st * sin(phi_s) - P.shift_y,
        r_s * cos(theta_s)    - P.shift_z,
    );
}

// α(r_s, θ_s, φ_s) = −sin²θ' / r where r is the shifted radial distance and
// θ' is the body-frame colatitude (magnetic axis at (θ=ι, φ=0)).
fn alpha_at(r_s: f32, theta_s: f32, phi_s: f32) -> f32 {
    let cart = shifted_cart(r_s, theta_s, phi_s);
    let r = max(length(cart), 1e-4);
    let ct  = cart.z / r;
    let st  = sqrt(max(0.0, 1.0 - ct * ct));
    let cp  = cart.x / max(r * st, 1e-4);   // cos(phi_local)
    let cos_tp = ct * wg_cos_iota + st * cp * wg_sin_iota;
    let sin2_tp = max(0.0, 1.0 - cos_tp * cos_tp);
    return -sin2_tp / r;
}

// β = φ' (body-frame longitude). Atan2 of the shifted Cartesian projected
// onto the body-frame (x', y') plane — derived by substituting φ → φ+π in
// the paper's formula to match our +x magnetic-axis convention.
fn beta_at(r_s: f32, theta_s: f32, phi_s: f32) -> f32 {
    let cart = shifted_cart(r_s, theta_s, phi_s);
    let r = max(length(cart), 1e-4);
    let ct = cart.z / r;
    let st = sqrt(max(0.0, 1.0 - ct * ct));
    let cp = cart.x / max(r * st, 1e-4);
    let sp = cart.y / max(r * st, 1e-4);
    let num = -st * sp;
    let den = -st * cp * wg_cos_iota + ct * wg_sin_iota;
    return atan2(num, den);
}

// Bessel J₀ / J₁ via power series (accurate to ~1e-5 for |x| ≤ π, plenty
// for a visualizer).
fn bessel_J0(x: f32) -> f32 {
    let z = x * x * 0.25;
    var sum: f32 = 1.0;
    var term: f32 = 1.0;
    for (var k: u32 = 1u; k < 14u; k = k + 1u) {
        term = -term * z / (f32(k) * f32(k));
        sum = sum + term;
    }
    return sum;
}

fn bessel_J1(x: f32) -> f32 {
    let z = x * x * 0.25;
    var sum: f32 = 1.0;
    var term: f32 = 1.0;
    for (var k: u32 = 1u; k < 14u; k = k + 1u) {
        term = -term * z / (f32(k) * f32(k + 1u));
        sum = sum + term;
    }
    return sum * (x * 0.5);
}

// Λ(α, β) from Eq. 8 of Huang & Chen (2025), with the overall Ω prefactor
// absorbed into T₀ so |j| stays O(1) across the slider range — keeps the
// T₀ slider behaving as a direct peak-temperature dial.
fn lambda_func(abs_alpha: f32, beta_val: f32, cos_tp_sign: f32) -> f32 {
    let ratio = clamp(abs_alpha / wg_alpha0_dim, 0.0, 1.0);
    let arg   = 2.0 * asin(sqrt(ratio));
    let j0_   = bessel_J0(arg);
    let j1_   = bessel_J1(arg);
    let sign  = select(1.0, -1.0, cos_tp_sign > 0.0);
    return sign * 2.0 * (j0_ * wg_cos_iota - sign * j1_ * cos(beta_val) * wg_sin_iota);
}

// Goldreich-Julian charge density in shifted-dipole coordinates. The centered
// formula ρ ∝ -(3 cos θ' cos θ − cos ι) picks up a 1/r³ scaling from the
// non-relativistic dipole field at the shifted surface point.
fn rho_at(r_s: f32, theta_s: f32, phi_s: f32) -> f32 {
    let cart = shifted_cart(r_s, theta_s, phi_s);
    let r = max(length(cart), 1e-4);
    let ct = cart.z / r;
    let st = sqrt(max(0.0, 1.0 - ct * ct));
    let cp = cart.x / max(r * st, 1e-4);
    let cos_tp = ct * wg_cos_iota + st * cp * wg_sin_iota;
    return -(3.0 * cos_tp * ct - wg_cos_iota) / (r * r * r);
}

// Compute 3-current magnitude |j| at lab-frame (r_s, θ_s, φ_s). For shifted
// dipoles every derivative of α, β w.r.t. lab coords is non-trivial and we
// do them all via symmetric finite differences. Returns 0 outside the open
// polar cap or where |j| ≤ |ρ| (spacelike heating only, Tier 1 scope).
fn surface_current_mag(r_s: f32, theta_s: f32, phi_s: f32) -> f32 {
    let alpha = alpha_at(r_s, theta_s, phi_s);
    let abs_a = abs(alpha);
    if (abs_a >= wg_alpha0_dim) { return 0.0; }

    // Body-frame cos θ' at the evaluation point, for the Λ hemisphere sign.
    let cart = shifted_cart(r_s, theta_s, phi_s);
    let r = max(length(cart), 1e-4);
    let ctL = cart.z / r;
    let stL = sqrt(max(0.0, 1.0 - ctL * ctL));
    let cpL = cart.x / max(r * stL, 1e-4);
    let cos_tp = ctL * wg_cos_iota + stL * cpL * wg_sin_iota;

    let beta = beta_at(r_s, theta_s, phi_s);

    // FD for all six first derivatives of α, β w.r.t. (r_s, θ_s, φ_s).
    let a_r_p = alpha_at(r_s + FD_EPS, theta_s, phi_s);
    let a_r_m = alpha_at(r_s - FD_EPS, theta_s, phi_s);
    let a_t_p = alpha_at(r_s, theta_s + FD_EPS, phi_s);
    let a_t_m = alpha_at(r_s, theta_s - FD_EPS, phi_s);
    let a_p_p = alpha_at(r_s, theta_s, phi_s + FD_EPS);
    let a_p_m = alpha_at(r_s, theta_s, phi_s - FD_EPS);
    let b_r_p = beta_at (r_s + FD_EPS, theta_s, phi_s);
    let b_r_m = beta_at (r_s - FD_EPS, theta_s, phi_s);
    let b_t_p = beta_at (r_s, theta_s + FD_EPS, phi_s);
    let b_t_m = beta_at (r_s, theta_s - FD_EPS, phi_s);
    let b_p_p = beta_at (r_s, theta_s, phi_s + FD_EPS);
    let b_p_m = beta_at (r_s, theta_s, phi_s - FD_EPS);

    let da_dr     = (a_r_p - a_r_m) / (2.0 * FD_EPS);
    let da_dtheta = (a_t_p - a_t_m) / (2.0 * FD_EPS);
    let da_dphi   = (a_p_p - a_p_m) / (2.0 * FD_EPS);
    let db_dr     = wrap_pi(b_r_p - b_r_m) / (2.0 * FD_EPS);
    let db_dtheta = wrap_pi(b_t_p - b_t_m) / (2.0 * FD_EPS);
    let db_dphi   = wrap_pi(b_p_p - b_p_m) / (2.0 * FD_EPS);

    let lam = lambda_func(abs_a, beta, cos_tp);
    let sin_t = max(sin(theta_s), 1e-4);
    let inv_grr = 1.0 / max(wg_uu, 1e-4);        // 1/√(1−2M/r_s)

    // Eqs. 4–6 with r → r_s in the denominators. No assumption that ∂_r β = 0.
    let Jr  =  lam * (da_dtheta * db_dphi - da_dphi * db_dtheta) / (inv_grr * r_s * sin_t);
    let Jth =  lam * (da_dphi * db_dr    - da_dr * db_dphi)     / (r_s * sin_t);
    let Jph =  lam * (da_dr * db_dtheta  - da_dtheta * db_dr)   /  r_s;

    let j_sq = Jr * Jr + Jth * Jth + Jph * Jph;
    let j_mag = sqrt(max(j_sq, 0.0));
    let rho_mag = abs(rho_at(r_s, theta_s, phi_s));
    if (j_mag <= rho_mag) { return 0.0; }
    return j_mag;
}

// Helper: fold an angle difference into [-π, π]. Used to undo atan2 wrap
// artifacts inside the FD stencil for β = φ'.
fn wrap_pi(x: f32) -> f32 {
    var y = x;
    if (y >  PI) { y = y - TWO_PI; }
    if (y < -PI) { y = y + TWO_PI; }
    return y;
}

// Per-patch kT in keV for a lab-frame point (r_s, θ_s, φ_s) on the open cap.
// r_s = 1 in our R-unit convention on the stellar surface.
fn patch_temperature(r_s: f32, theta_s: f32, phi_s: f32) -> f32 {
    let jmag = surface_current_mag(r_s, theta_s, phi_s);
    if (jmag == 0.0) { return 0.0; }
    return P.T0 * pow(jmag, 0.25);
}

// ---------------- Hunt + reductions (verbatim from kernel.wgsl) ----------------

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

// ---------------- Main kernel ----------------

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let wg_idx: u32 = wg_id.x;
    let tid: u32    = local_id.x;

    // ---- Fast-path: skip rings that can't touch either polar cap ----
    // The centered-dipole caps sit at (θ=ι, any φ) and (θ=π−ι, any φ); both
    // have angular radius θ_cap with sin²θ_cap = α₀R/μ. When the dipole is
    // shifted the caps deform asymmetrically, so we only apply the fast-path
    // when the shift is negligible.
    let ring_ct_test = healpix_cos_theta_ring(wg_idx + 1u);
    let ring_st_test = sqrt(max(0.0, 1.0 - ring_ct_test * ring_ct_test));
    let ci_test = cos(P.mag_incl);
    let si_test = sin(P.mag_incl);
    let alpha0_test = SQRT_3_HALVES
        * (TWO_PI * P.nu * P.Re / C_KM_S)
        * (1.0 + si_test * si_test * 0.2);
    let cos_cap_test = sqrt(max(0.0, 1.0 - alpha0_test));
    let cos_N = ring_st_test * si_test + ring_ct_test * ci_test;
    let cos_S = ring_st_test * si_test - ring_ct_test * ci_test;
    let shift_mag = abs(P.shift_x) + abs(P.shift_y) + abs(P.shift_z);
    if (shift_mag < 1e-4
        && cos_N <= cos_cap_test
        && cos_S <= cos_cap_test) {
        if (tid < N_OUTPUT_PHASE) {
            per_ring_flux[wg_idx * N_OUTPUT_PHASE + tid] = 0.0;
        }
        return;
    }

    // ---- Thread-0 init of per-ring state ----
    if (tid == 0u) {
        atomicStore(&wg_N_active_phi, 0u);

        let healpix_i = wg_idx + 1u;
        let ct = healpix_cos_theta_ring(healpix_i);
        wg_cos_theta = ct;
        wg_sin_theta = sqrt(max(0.0, 1.0 - ct * ct));

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
        wg_dS = R_val * R_val * healpix_dOmega() * ff;

        wg_cos_i = cos(P.obs_incl);
        wg_sin_i = sin(P.obs_incl);
        wg_cc = wg_cos_i * ct;
        wg_ss = wg_sin_i * wg_sin_theta;

        // Dipole-specific per-ring constants
        wg_cos_iota = ci_test;
        wg_sin_iota = si_test;
        wg_Omega = TWO_PI * P.nu;
        wg_alpha0_dim = alpha0_test;
        wg_cos_theta_cap = cos_cap_test;
    }
    workgroupBarrier();

    // ---- Fill active_phi (dipole filter: |α| < α₀, store per-patch kT) ----
    // r_s = 1 on the surface in our R-unit convention; α and friends pick up
    // the shift through alpha_at/patch_temperature.
    {
        let healpix_i = wg_idx + 1u;
        let jmax = healpix_j_max(healpix_i);
        let theta_ring = acos(clamp(wg_cos_theta, -1.0, 1.0));
        for (var k: u32 = 0u; k < N_PHI_ITER; k = k + 1u) {
            let hj = k * BLOCK_DIM + tid + 1u;
            if (hj <= jmax) {
                let phi = healpix_phi(healpix_i, hj);
                let abs_a = abs(alpha_at(1.0, theta_ring, phi));
                if (abs_a < wg_alpha0_dim) {
                    let kT = patch_temperature(1.0, theta_ring, phi);
                    if (kT > 0.0) {
                        let idx = atomicAdd(&wg_N_active_phi, 1u);
                        wg_active_phi[idx] = phi;
                        wg_active_kT[idx]  = kT;
                    }
                }
            }
        }
    }
    workgroupBarrier();

    let N_active = atomicLoad(&wg_N_active_phi);

    // ---- Fine-phase sweep (verbatim from kernel.wgsl; kT is per-patch so
    //      we defer the Planck call to the final integration step) ----
    var local_inv_min: i32 = i32(N_FINE_PHASE) + 1;
    var local_inv_max: i32 = -1;

    for (var k: u32 = 0u; k < N_SWEEP_ITER; k = k + 1u) {
        let ip = k * BLOCK_DIM + tid;
        if (ip < N_FINE_PHASE) {
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

                    // Per-patch kT is folded in at integration time, so the
                    // "flux_val" here carries everything except Planck(E_emit, kT).
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
        }
    }
    workgroupBarrier();

    let inv_min = reduce_min_i32(tid, local_inv_min);
    workgroupBarrier();
    let inv_max = reduce_max_i32(tid, local_inv_max);
    workgroupBarrier();

    let skip = N_active == 0u
            || inv_min == 0
            || inv_max == i32(N_FINE_PHASE) - 1;

    let do_fill = !skip && inv_min <= inv_max;
    let fill_beg = select(0,  inv_min - 1, do_fill);
    let fill_end = select(1,  inv_max + 1, do_fill);
    let fill_lo  = select(0,  inv_min,     do_fill);
    let fill_hi  = select(-1, inv_max,     do_fill);
    let phase0 = wg_phase_o[fill_beg];
    let phase1 = wg_phase_o[fill_end];
    let rf0    = wg_redshift_factors[fill_beg];
    let rf1    = wg_redshift_factors[fill_end];
    let denom  = f32(fill_end - fill_beg);

    for (var k: u32 = 0u; k < N_FILL_ITER; k = k + 1u) {
        let ii = i32(k * BLOCK_DIM + tid) + fill_lo;
        if (ii >= fill_lo && ii <= fill_hi) {
            let t = f32(ii - fill_beg) / denom;
            wg_phase_o[ii]          = phase0 + (phase1 - phase0) * t;
            wg_redshift_factors[ii] = rf0    + (rf1    - rf0)    * t;
        }
    }
    workgroupBarrier();

    if (tid == 0u) {
        wg_fluxes_over_I[N_FINE_PHASE]    = wg_fluxes_over_I[0];
        wg_redshift_factors[N_FINE_PHASE] = wg_redshift_factors[0];
        wg_phase_o[N_FINE_PHASE]          = 1.0 + wg_phase_o[0];
    }
    workgroupBarrier();

    // ---- Integrate into output phase bins with per-patch kT ----
    if (tid < N_OUTPUT_PHASE) {
        var sum: f32 = 0.0;
        if (!skip) {
            let target_phase = f32(tid) / f32(N_OUTPUT_PHASE);
            var jsav: i32 = 0;
            for (var ia: u32 = 0u; ia < N_active; ia = ia + 1u) {
                let phase_shift = wg_active_phi[ia] / TWO_PI;
                let kT_patch    = wg_active_kT[ia];
                let raw = target_phase + phase_shift + 1.0;
                var patch_phase = raw - floor(raw);
                if (patch_phase < wg_phase_o[0]) { patch_phase = patch_phase + 1.0; }
                let h = hunt_phase_o(patch_phase, &jsav);
                let flux_i = (1.0 - h.a) * wg_fluxes_over_I[h.i]
                           +        h.a  * wg_fluxes_over_I[h.i + 1];
                let rf     = (1.0 - h.a) * wg_redshift_factors[h.i]
                           +        h.a  * wg_redshift_factors[h.i + 1];
                let E_emit = P.E_obs * rf;
                sum = sum + flux_i * blackbody_I(E_emit, kT_patch);
            }
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
