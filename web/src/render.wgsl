// Ray-cast a unit sphere at the origin, mark the hotspot region.
// The physics kernel uses the oblate R(theta) in its math, but the render
// is purely spherical per the project decision.

const PI: f32 = 3.141592653589793;
const TWO_PI: f32 = 6.283185307179586;

struct RenderParams {
    cam_pos:   vec4<f32>,   // xyz = camera position in star frame; w unused
    cam_right: vec4<f32>,   // xyz
    cam_up:    vec4<f32>,
    cam_fwd:   vec4<f32>,
    view:      vec4<f32>,   // x=aspect  y=tan(fovy/2)  z,w unused
    spot:      vec4<f32>,   // x=spot_center_theta  y=spot_center_phi
                            // z=cos_angular_radius  w=observer_phase [0,1)
    light_dir: vec4<f32>,   // xyz = toward light (usually = -cam_fwd)
};

@group(0) @binding(0) var<uniform> R: RenderParams;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VOut {
    // Full-screen triangle
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

// Background: faint star-field gradient so empty pixels aren't pure black.
fn bg_color(dir: vec3<f32>) -> vec3<f32> {
    let t = 0.5 + 0.5 * dir.y;  // fake subtle vertical gradient
    return mix(vec3<f32>(0.02, 0.03, 0.05), vec3<f32>(0.05, 0.05, 0.09), t);
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let aspect = R.view.x;
    let tan_half_fovy = R.view.y;

    let dx = in.ndc.x * aspect * tan_half_fovy;
    let dy = in.ndc.y * tan_half_fovy;
    let ray = normalize(R.cam_fwd.xyz + dx * R.cam_right.xyz + dy * R.cam_up.xyz);
    let origin = R.cam_pos.xyz;

    // Ray-sphere intersection: unit sphere at origin.
    // |origin + t*ray|^2 = 1  =>  t^2 + 2(origin·ray)t + (origin·origin - 1) = 0
    let b = dot(origin, ray);
    let c = dot(origin, origin) - 1.0;
    let disc = b * b - c;
    if (disc < 0.0) {
        return vec4<f32>(bg_color(ray), 1.0);
    }
    let sqd = sqrt(disc);
    let t = -b - sqd;
    if (t < 0.0) {
        return vec4<f32>(bg_color(ray), 1.0);
    }
    let p = origin + t * ray;  // surface point; |p| = 1

    let cos_theta = clamp(p.z, -1.0, 1.0);
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = atan2(p.y, p.x);

    // Hotspot center moves with observer_phase: as phase advances, the star has
    // rotated so the spot's inertial-frame longitude = spot_phi + 2*pi*phase.
    let spot_theta = R.spot.x;
    let spot_phi   = R.spot.y + TWO_PI * R.spot.w;
    let cos_ar     = R.spot.z;

    let cos_rho = sin_theta * sin(spot_theta) * cos(phi - spot_phi)
                + cos_theta * cos(spot_theta);
    let in_spot = cos_rho > cos_ar;

    // Soft edge on the spot so it doesn't alias.
    let edge = smoothstep(cos_ar - 0.004, cos_ar + 0.004, cos_rho);

    // Lambertian shading.
    let normal = p;
    let n_dot_l = max(0.0, dot(normal, normalize(R.light_dir.xyz)));
    let ambient = 0.15;
    let lit = ambient + (1.0 - ambient) * n_dot_l;

    let surface_cold = vec3<f32>(0.22, 0.26, 0.38);
    let surface_hot  = vec3<f32>(1.00, 0.62, 0.20);
    let base = mix(surface_cold, surface_hot, edge);

    var color = base * lit;

    // Subtle limb-darkening cue.
    let facing = max(0.0, dot(normal, -ray));
    color = color * (0.6 + 0.4 * facing);

    // ----- Orientation reference marks -----
    //   equator ring       thin cyan band at z=0
    //   prime meridian     thin grey line at phi=0 (the spot's φ=0 reference)
    //   pole caps          bright white dots at ±z
    //
    // These make the rotation axis unambiguous. Each mark is additive and
    // respects limb-darkening (facing factor) so back-hemisphere hits don't
    // bleed through.
    let eq_mask    = smoothstep(0.012, 0.005, abs(cos_theta));
    let pole_mask  = smoothstep(0.993, 0.999, abs(cos_theta));
    let mer_mask   = smoothstep(0.012, 0.005, abs(phi)) * sin_theta;
    let visibility = facing;  // kill marks on the far hemisphere

    let eq_color   = vec3<f32>(0.35, 0.60, 0.75);
    let pole_color = vec3<f32>(0.70, 0.72, 0.80);
    let mer_color  = vec3<f32>(0.45, 0.48, 0.58);

    color = mix(color, eq_color,   eq_mask   * 0.28 * visibility);
    color = mix(color, mer_color,  mer_mask  * 0.20 * visibility);
    color = mix(color, pole_color, pole_mask * 0.30 * visibility);

    return vec4<f32>(color, 1.0);
}
