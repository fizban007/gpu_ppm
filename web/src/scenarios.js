// OS1a-OS1j scenarios — mirrors StarData instances in gpu/os1_gpu.cu.
//
// Beaming codes: 0=iso ('i'), 1=cos² ('c'), 2=1-cos² ('s').

const DEG = Math.PI / 180;

// Star-wide physical constants, shared across all OS1 cases:
export const OS1_CONSTANTS = {
  M: 1.4,          // solar masses
  Re: 12.0,        // km (equatorial radius)
  kT: 0.35,        // keV
  E_obs: 1.0,      // keV
  D: 0.2,          // kpc
};

const make = (name, nu, theta_deg, inc_deg, ang_rad, beamChar) => ({
  name,
  nu,
  spot_center_theta: theta_deg * DEG,
  inc: inc_deg * DEG,
  angular_radius: ang_rad,
  beaming: { i: 0, c: 1, s: 2 }[beamChar],
});

export const OS1_SCENARIOS = [
  make("os1a", 600,  90,  90, 0.01, "i"),
  make("os1b", 600,  90,  90, 1.00, "i"),
  make("os1c", 200,  90,  90, 0.01, "i"),
  make("os1d",   1,  90,  90, 1.00, "i"),
  make("os1e", 600,  60,  30, 1.00, "i"),
  make("os1f", 600,  20,  80, 1.00, "i"),
  make("os1g", 600,  60,  30, 1.00, "c"),
  make("os1h", 600,  60,  30, 1.00, "s"),
  make("os1i", 600,  20,  80, 1.00, "c"),
  make("os1j", 600,  20,  80, 1.00, "s"),
];

export function scenarioByName(name) {
  const s = OS1_SCENARIOS.find((s) => s.name === name);
  if (!s) throw new Error(`unknown scenario ${name}`);
  return s;
}

// ----- Unified preset format for the multi-spot UI -----
// Each preset carries observer/star fields in display units plus an explicit
// `spots` array. The OS1 cases are trivially lifted into this shape so the
// picker can list them alongside true multi-spot presets.

export const MULTI_SPOT_PRESETS = [
  {
    name: "crescent + spot",
    nu: 400,
    inc_deg: 65,
    beaming: 0,
    emission_mode: "spots",
    spots: [
      // A crescent sitting near the equator: large ADD with an offset SUB inside it.
      { theta_deg: 60, phi_deg: 0,   rho: 0.50, mode: "ADD", kT: 0.40 },
      { theta_deg: 60, phi_deg: 20,  rho: 0.25, mode: "SUB", kT: 0.40 },
      // A small, hotter circular spot on the opposite hemisphere for a second pulse.
      { theta_deg: 40, phi_deg: 180, rho: 0.30, mode: "ADD", kT: 0.60 },
    ],
  },
];

export const DIPOLE_PRESETS = [
  {
    name: "dipole (centered, ι=45°)",
    nu: 400,
    inc_deg: 65,
    beaming: 0,
    emission_mode: "dipole",
    dipole: { mag_incl_deg: 45, T0: 0.40, shift_x: 0, shift_y: 0, shift_z: 0 },
  },
  {
    name: "dipole (aligned, ι=10°)",
    nu: 400,
    inc_deg: 65,
    beaming: 0,
    emission_mode: "dipole",
    dipole: { mag_incl_deg: 10, T0: 0.40, shift_x: 0, shift_y: 0, shift_z: 0 },
  },
  {
    name: "dipole (shifted, x₀=0.3R)",
    nu: 400,
    inc_deg: 65,
    beaming: 0,
    emission_mode: "dipole",
    dipole: { mag_incl_deg: 45, T0: 0.40, shift_x: 0.3, shift_y: 0, shift_z: 0 },
  },
];

function os1AsPreset(sc) {
  return {
    name: sc.name,
    nu: sc.nu,
    inc_deg: sc.inc / DEG,
    beaming: sc.beaming,
    emission_mode: "spots",
    spots: [{
      theta_deg: sc.spot_center_theta / DEG,
      phi_deg: 0,
      rho: sc.angular_radius,
      mode: "ADD",
      kT: 0.35,
    }],
  };
}

export const PRESETS = [
  ...MULTI_SPOT_PRESETS,
  ...DIPOLE_PRESETS,
  ...OS1_SCENARIOS.map(os1AsPreset),
];

export function presetByName(name) {
  const p = PRESETS.find((p) => p.name === name);
  if (!p) throw new Error(`unknown preset ${name}`);
  return p;
}
