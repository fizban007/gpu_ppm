// Ray-cast a unit sphere at the origin, mark the hotspot region(s).
// The physics kernel uses the oblate R(theta) in its math, but the render
// is purely spherical per the project decision.

const PI: f32 = 3.141592653589793;
const TWO_PI: f32 = 6.283185307179586;
const MAX_SPOTS: u32 = 4u;

// Per-spot packing (stored as vec4):
//   x = center_theta
//   y = center_phi           (in the corotating frame)
//   z = cos(angular_radius)
//   w = mode                 (+1.0 = ADD, -1.0 = SUBTRACT, 0.0 = inactive)
struct RenderParams {
    cam_pos:   vec4<f32>,
    cam_right: vec4<f32>,
    cam_up:    vec4<f32>,
    cam_fwd:   vec4<f32>,
    view:      vec4<f32>,              // x=aspect  y=tan(fovy/2)  z=observer_phase  w=spot_count
    light_dir: vec4<f32>,
    spots:     array<vec4<f32>, 4>,    // (θ, φ, cos_ar, mode) per slot
    spots_kt:  vec4<f32>,              // kT per slot (keV); SUBTRACT slots' kT is unused
    dipole:    vec4<f32>,              // x=mag_incl  y=alpha0_dim  z=T0  w=display_mode (0|1|2 as f32)
};

@group(0) @binding(0) var<uniform> R: RenderParams;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VOut {
    let p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: VOut;
    out.pos = vec4<f32>(p[vid], 0.0, 1.0);
    out.ndc = p[vid];
    return out;
}

fn bg_color(dir: vec3<f32>) -> vec3<f32> {
    let t = 0.5 + 0.5 * dir.y;
    return mix(vec3<f32>(0.02, 0.03, 0.05), vec3<f32>(0.05, 0.05, 0.09), t);
}

// Map kT (keV) to an indicative hot color: deep red (cool) → orange
// (≈ 0.35 keV, the OS1 default) → near-white (hot). Log-scaled over the
// slider range [0.05, 3.0] so small kT changes at the low end are visible.
fn hot_color_for_kT(kT: f32) -> vec3<f32> {
    let log_lo = -4.321928;  // log2(0.05)
    let log_hi =  1.584963;  // log2(3.0)
    let t = clamp((log2(max(kT, 1e-3)) - log_lo) / (log_hi - log_lo), 0.0, 1.0);
    let c1 = vec3<f32>(0.80, 0.25, 0.12);  // deep red
    let c2 = vec3<f32>(1.00, 0.62, 0.20);  // warm orange (default)
    let c3 = vec3<f32>(1.00, 0.96, 0.85);  // off-white
    if (t < 0.5) { return mix(c1, c2, t * 2.0); }
    return mix(c2, c3, (t - 0.5) * 2.0);
}

fn spot_kT(index: u32) -> f32 {
    if (index == 0u) { return R.spots_kt.x; }
    if (index == 1u) { return R.spots_kt.y; }
    if (index == 2u) { return R.spots_kt.z; }
    return R.spots_kt.w;
}

// ---------------- Dipole surface math (for T and J modes) ----------------
//
// Uses the same centered-dipole prescription as kernel_dipole.wgsl: on the
// surface r = R (set to 1 in dimensionless units here),
//   α = −sin²θ'(θ, φ),    β = φ'(θ, φ),
// with (θ', φ') the magnetic-axis body-frame coordinates. Inside the open
// polar cap (sin²θ' < α₀R/μ stored in R.dipole.y), surface temperature is
// kT ≈ T₀ · |j|^(1/4) where |j| comes from Eqs. 4–6 via FD derivatives.

// Magnetic axis at (θ=ι, φ=0) — the inclined pole is on the camera side at
// default observer_phase. Same convention as kernel_dipole.wgsl.
fn dipole_cos_tp(theta: f32, phi: f32) -> f32 {
    let iota = R.dipole.x;
    return cos(theta) * cos(iota) + sin(theta) * cos(phi) * sin(iota);
}

fn dipole_sin2_tp(theta: f32, phi: f32) -> f32 {
    let c = dipole_cos_tp(theta, phi);
    return max(0.0, 1.0 - c * c);
}

fn dipole_phi_prime(theta: f32, phi: f32) -> f32 {
    let iota = R.dipole.x;
    let num = -sin(theta) * sin(phi);
    let den = -sin(theta) * cos(phi) * cos(iota) + cos(theta) * sin(iota);
    return atan2(num, den);
}

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

fn wrap_pi(x: f32) -> f32 {
    var y = x;
    if (y >  PI) { y = y - TWO_PI; }
    if (y < -PI) { y = y + TWO_PI; }
    return y;
}

// Returns dimensionless 3-current magnitude |j| at surface point (θ, φ).
// Ω is absorbed into T₀ for visualization — we skip it here and let the
// colormap scale handle magnitudes.
fn dipole_j_mag(theta: f32, phi: f32) -> f32 {
    let alpha0 = R.dipole.y;
    let sin2_tp = dipole_sin2_tp(theta, phi);
    if (sin2_tp >= alpha0) { return 0.0; }
    let cos_tp = dipole_cos_tp(theta, phi);
    let abs_a  = sin2_tp;

    let eps = 1e-3;
    let a_tp = dipole_sin2_tp(theta + eps, phi);
    let a_tm = dipole_sin2_tp(theta - eps, phi);
    let a_pp = dipole_sin2_tp(theta, phi + eps);
    let a_pm = dipole_sin2_tp(theta, phi - eps);
    // α = -sin²θ', so ∂α/∂x = -∂sin²θ'/∂x
    let da_dtheta = -(a_tp - a_tm) / (2.0 * eps);
    let da_dphi   = -(a_pp - a_pm) / (2.0 * eps);
    let db_dtheta = wrap_pi(dipole_phi_prime(theta + eps, phi) - dipole_phi_prime(theta - eps, phi)) / (2.0 * eps);
    let db_dphi   = wrap_pi(dipole_phi_prime(theta, phi + eps) - dipole_phi_prime(theta, phi - eps)) / (2.0 * eps);
    let da_dr     = abs_a;   // = -α at r=1

    // Λ(α, β), with the sign flip between hemispheres (cos θ'>0 is north).
    let ratio = clamp(abs_a / alpha0, 0.0, 1.0);
    let arg   = 2.0 * asin(sqrt(ratio));
    let j0_   = bessel_J0(arg);
    let j1_   = bessel_J1(arg);
    let iota  = R.dipole.x;
    let beta  = dipole_phi_prime(theta, phi);
    let hem_sign = select(1.0, -1.0, cos_tp > 0.0);
    let lam   = hem_sign * 2.0 * (j0_ * cos(iota) - hem_sign * j1_ * cos(beta) * sin(iota));

    // Eqs. 4-6 with r=1 and ∂_r β = 0.
    let sin_t = max(sin(theta), 1e-4);
    let Jr  = lam * (da_dtheta * db_dphi - da_dphi * db_dtheta) / sin_t;
    let Jth = -lam * da_dr * db_dphi / sin_t;
    let Jph =  lam * da_dr * db_dtheta;
    return sqrt(max(Jr * Jr + Jth * Jth + Jph * Jph, 0.0));
}

// Diverging colormap for signed J²: blue (negative), white (zero), red (positive).
// t is expected to lie in [-1, 1] after the caller's normalization.
fn diverging_color(t: f32) -> vec3<f32> {
    let tc = clamp(t, -1.0, 1.0);
    if (tc >= 0.0) {
        return mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.85, 0.20, 0.12), tc);
    } else {
        return mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.12, 0.35, 0.85), -tc);
    }
}

// Same GJ charge-density approximation used in kernel_dipole.wgsl.
fn dipole_rho_gj(theta: f32, phi: f32) -> f32 {
    let iota = R.dipole.x;
    let cos_tp = dipole_cos_tp(theta, phi);
    return -(3.0 * cos_tp * cos(theta) - cos(iota));
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let aspect = R.view.x;
    let tan_half_fovy = R.view.y;
    let observer_phase = R.view.z;
    let spot_count_f = R.view.w;

    let dx = in.ndc.x * aspect * tan_half_fovy;
    let dy = in.ndc.y * tan_half_fovy;
    let ray = normalize(R.cam_fwd.xyz + dx * R.cam_right.xyz + dy * R.cam_up.xyz);
    let origin = R.cam_pos.xyz;

    let b = dot(origin, ray);
    let c = dot(origin, origin) - 1.0;
    let disc = b * b - c;
    if (disc < 0.0) { return vec4<f32>(bg_color(ray), 1.0); }
    let sqd = sqrt(disc);
    let t = -b - sqd;
    if (t < 0.0) { return vec4<f32>(bg_color(ray), 1.0); }
    let p = origin + t * ray;

    let cos_theta = clamp(p.z, -1.0, 1.0);
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = atan2(p.y, p.x);

    // Base (cold) surface color, independent of spots.
    let surface_cold = vec3<f32>(0.22, 0.26, 0.38);
    var base = surface_cold;

    let display_mode = u32(R.dipole.w);
    let T0 = R.dipole.z;

    if (display_mode == 0u) {
        // --- Spots (painter's-order ADD/SUBTRACT) ---
        let spot_count = u32(spot_count_f);
        for (var k: u32 = 0u; k < MAX_SPOTS; k = k + 1u) {
            if (k >= spot_count) { break; }
            let s = R.spots[k];
            let mode = s.w;
            if (mode == 0.0) { continue; }
            let s_theta = s.x;
            let s_phi   = s.y + TWO_PI * observer_phase;
            let cos_ar  = s.z;
            let cos_rho = sin_theta * sin(s_theta) * cos(phi - s_phi)
                        + cos_theta * cos(s_theta);
            let edge = smoothstep(cos_ar - 0.004, cos_ar + 0.004, cos_rho);
            if (mode > 0.0) {
                base = mix(base, hot_color_for_kT(spot_kT(k)), edge);
            } else {
                base = mix(base, surface_cold,                 edge);
            }
        }
    } else {
        // --- Dipole modes: rotate lab φ back by observer_phase (the pattern
        //     is fixed in the corotating frame; phase advance rotates the
        //     star under the fixed observer). ---
        let phi_co = phi - TWO_PI * observer_phase;

        let theta_here = acos(clamp(cos_theta, -1.0, 1.0));
        let sin2_tp = dipole_sin2_tp(theta_here, phi_co);
        if (sin2_tp < R.dipole.y) {
            let jmag = dipole_j_mag(theta_here, phi_co);        // |j| inside cap (no ρ gate)
            let rho  = dipole_rho_gj(theta_here, phi_co);
            if (display_mode == 1u) {
                // T-map: heating only where |j| > |ρ|. Color picks up the
                // kT-dependent hot gradient; alpha fades smoothly with the
                // spacelike excess |j| − |ρ|, so the gate boundary doesn't
                // read as a hard edge.
                let excess = jmag - abs(rho);
                if (excess > 0.0) {
                    let kT = T0 * pow(jmag, 0.25);
                    let scale: f32 = 1.0;
                    let alpha = clamp(excess / scale, 0.0, 1.0);
                    base = mix(base, hot_color_for_kT(kT), alpha);
                }
            } else {
                // Signed J² = |j|² − ρ². Instead of blending toward white at
                // the zero crossing (which reads as grey over the cold base),
                // we always paint pure red (spacelike) or pure blue (timelike)
                // and let transparency alone encode the magnitude — that way
                // the transition sweeps smoothly from base color through tinted
                // base to saturated red/blue.
                let j_sq_signed = jmag * jmag - rho * rho;
                let scale: f32 = 5.0;
                let t = j_sq_signed / scale;
                let alpha = clamp(abs(t), 0.0, 1.0);
                let pure_red  = vec3<f32>(0.85, 0.20, 0.12);
                let pure_blue = vec3<f32>(0.12, 0.35, 0.85);
                let overlay = select(pure_blue, pure_red, t >= 0.0);
                base = mix(base, overlay, alpha);
            }
        }
    }

    // Lambertian shading + limb cue.
    let normal = p;
    let n_dot_l = max(0.0, dot(normal, normalize(R.light_dir.xyz)));
    let ambient = 0.15;
    let lit = ambient + (1.0 - ambient) * n_dot_l;
    let facing = max(0.0, dot(normal, -ray));
    var color = base * lit * (0.6 + 0.4 * facing);

    // Orientation reference marks — kept subtle.
    let eq_mask    = smoothstep(0.012, 0.005, abs(cos_theta));
    let pole_mask  = smoothstep(0.993, 0.999, abs(cos_theta));
    let mer_mask   = smoothstep(0.012, 0.005, abs(phi)) * sin_theta;
    let visibility = facing;
    let eq_color   = vec3<f32>(0.35, 0.60, 0.75);
    let pole_color = vec3<f32>(0.70, 0.72, 0.80);
    let mer_color  = vec3<f32>(0.45, 0.48, 0.58);
    color = mix(color, eq_color,   eq_mask   * 0.28 * visibility);
    color = mix(color, mer_color,  mer_mask  * 0.20 * visibility);
    color = mix(color, pole_color, pole_mask * 0.30 * visibility);

    return vec4<f32>(color, 1.0);
}
