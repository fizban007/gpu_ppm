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
    spots:     array<vec4<f32>, 4>,    // MAX_SPOTS
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
    let surface_hot  = vec3<f32>(1.00, 0.62, 0.20);
    var base = surface_cold;

    // Painter's-order sweep over spots: ADD blends toward hot; SUBTRACT
    // blends back toward cold. Both use a smooth edge for antialiasing.
    let spot_count = u32(spot_count_f);
    for (var k: u32 = 0u; k < MAX_SPOTS; k = k + 1u) {
        if (k >= spot_count) { break; }
        let s = R.spots[k];
        let mode = s.w;
        if (mode == 0.0) { continue; }                       // inactive slot
        let s_theta = s.x;
        let s_phi   = s.y + TWO_PI * observer_phase;         // rotate with time
        let cos_ar  = s.z;
        let cos_rho = sin_theta * sin(s_theta) * cos(phi - s_phi)
                    + cos_theta * cos(s_theta);
        let edge = smoothstep(cos_ar - 0.004, cos_ar + 0.004, cos_rho);
        if (mode > 0.0) {
            base = mix(base, surface_hot,  edge);            // ADD
        } else {
            base = mix(base, surface_cold, edge);            // SUBTRACT
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
