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
