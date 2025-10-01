import numpy as np
from scipy.interpolate import CubicSpline
from pathlib import Path

rmf_fname = "ext/model_data/nicer-rmf6s-teamonly-array50_full_matrix.txt"
arf_fname = "ext/model_data/nicer-consim135p-teamonly-array50_arf.txt"
ism_fname = "ext/model_data/interstellar_phot_frac.txt"

output_folder = Path("tables/ism_inst")

used_i_E_min = 0
used_i_E_max = 1800
used_i_CH_min = 30
used_i_CH_max = 300

# setting = "ultra"
# setting = "high"
setting = "std"

if setting == "ultra":
    N_E_obs = 512
elif setting == "high":
    N_E_obs = 256
elif setting == "std":
    N_E_obs = 128

output_folder /= setting
output_folder.mkdir(exist_ok=True, parents=True)

rmf = np.loadtxt(rmf_fname, skiprows=3, usecols=-1)
print(f"{rmf.shape=}")

arf = np.loadtxt(arf_fname, skiprows=3)
print(f"{arf.shape=}")

N_E = arf.shape[0]
rmf = rmf.reshape((N_E, -1)).T
N_CH = rmf.shape[0]
print(f"{N_E=}, {N_CH=}")
print(f"{rmf.shape=}")

# trim the RMF and ARF to the used energy range
rmf = rmf[used_i_CH_min:used_i_CH_max, used_i_E_min:used_i_E_max]
arf = arf[used_i_E_min:used_i_E_max, :]
N_CH = rmf.shape[0]
N_E = rmf.shape[1]
print("After trim:")
print(f"{rmf.shape=}")
print(f"{arf.shape=}")
print(f"{N_CH=}, {N_E=}")

E_lo = arf[:, 0]
E_hi = arf[:, 1]
dE = E_hi - E_lo
E_mid = (E_lo + E_hi) * 0.5
E_edges = np.hstack((E_lo[0], E_hi))

response_matrix = arf[:, 2] * rmf
print(f"{response_matrix.shape=}")

E_obs = np.geomspace(E_edges[0], E_edges[-1], N_E_obs)
# E_obs = np.linspace(E_edges[0], E_edges[-1], N_E_obs)
E_obs_fname = output_folder / "E_obs.txt"
np.savetxt(E_obs_fname, E_obs)
print(f"E_obs saved to {E_obs_fname}")

fname = output_folder / "num.txt"
with open(fname, "w") as f:
    f.write(f"{N_CH} {N_E_obs}\n")
print(f"#channels and #ds_energy_bins saved to {fname}")

x_fine = E_mid
x_coarse = E_obs

C = np.zeros((len(x_fine), len(x_coarse)))

for i in range(len(x_coarse)):
    y_i = np.zeros(len(x_coarse))
    y_i[i] = 1.0
    cs = CubicSpline(x_coarse, y_i)
    for j in range(len(x_fine)):
        C[j, i] = cs.integrate(E_lo[j], E_hi[j])

################################### ISM ###################################
ism = np.loadtxt(ism_fname)
ism_e = ism[:, 0]
ism_frac = ism[:, 2]
ism_interp = CubicSpline(ism_e, ism_frac)
frac_fine = ism_interp(E_mid) ** 5

print(f"{response_matrix.shape=}")
print(f"{frac_fine.shape=}")
print(f"{C.shape=}")
################################### ISM ###################################

ds_response_matrix = (response_matrix * frac_fine) @ C
print(f"{ds_response_matrix.shape=}")

ds_response_matrix_fname = f"{output_folder}/rsp.txt"
ds_response_matrix[np.abs(ds_response_matrix) < 1e-300] = 0.0
np.savetxt(ds_response_matrix_fname, ds_response_matrix)
print(f"ds_response_matrix saved to {ds_response_matrix_fname}")
