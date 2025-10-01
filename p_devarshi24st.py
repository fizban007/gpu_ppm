import matplotlib.pyplot as plt
import numpy as np
import sys

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

choice = sys.argv[1]
match choice:
    case "a":
        case_name = "Ring_Eq"
    case "b":
        case_name = "Ring_Polar"
    case "c":
        case_name = "Crescent_Eq"
    case "d":
        case_name = "Crescent_Polar"
    case _:
        raise ValueError("Invalid choice")

data_XPSI_Ultra = (
    np.loadtxt(
        "ext/Pulse_profiles_by_diff_codes/XPSI/Ultra_res/"
        + case_name
        + "_expected_hreadable.dat",
        usecols=2,
    )
    .reshape(270, 32)
    .T
)
UltraXPSI_es = np.sum(data_XPSI_Ultra, axis=1)
UltraXPSI_ps = np.sum(data_XPSI_Ultra, axis=0)


def frac_diff(a, b):
    return (a - b) / b * 100


def cal_chi2(data, model):
    """Calculate reduced chi2 between data and model"""
    fac = 1e6 / np.sum(model)
    model_ = model * fac
    data_ = data * fac
    chi2 = np.sum((data_ - model_) ** 2 / model_)
    return chi2


fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(8, 5),
    gridspec_kw={"height_ratios": [3, 1]},
)


lw = 0.5
ms = 1

axes[0, 0].plot(
    np.linspace(0.0, 1.0, 32),
    UltraXPSI_es,
    "ko-",
    lw=lw,
    ms=ms,
    label="Ultra X-PSI",
)

axes[0, 1].plot(
    np.arange(30, 300),
    UltraXPSI_ps,
    "o-",
    color="k",
    lw=lw,
    ms=ms,
    label="Ultra X-PSI",
)

axes[1, 0].plot(
    np.linspace(0.0, 1.0, 32),
    (UltraXPSI_es) ** 0.5 / UltraXPSI_es * 100,
    "k--",
    lw=lw,
    ms=ms,
    label="poisson error",
)
axes[1, 0].plot(
    np.linspace(0.0, 1.0, 32),
    -((UltraXPSI_es) ** 0.5) / UltraXPSI_es * 100,
    "k--",
    lw=lw,
    ms=ms,
)

# axes[1, 1].plot(
#     np.arange(30, 300),
#     -((UltraXPSI_ps) ** 0.5) / UltraXPSI_ps * 100,
#     "k--",
#     lw=0.5,
#     label="poisson error",
# )
# axes[1, 1].plot(
#     np.arange(30, 300),
#     ((UltraXPSI_ps) ** 0.5) / UltraXPSI_ps * 100,
#     "k--",
#     lw=0.5,
# )


def plot_data(name, setting, ls, color, device):
    data = np.loadtxt(f"output/{name}_{setting}_counts_{device}.txt").T
    data_es = np.sum(data, axis=1)
    data_ps = np.sum(data, axis=0)
    chi2 = cal_chi2(data, data_XPSI_Ultra)
    print(f"{name} {setting} {device} chi2={chi2:.3g}")
    calc_time = np.loadtxt(f"output/{name}_{setting}_time_{device}.txt")

    axes[0, 0].plot(
        np.linspace(0.0, 1.0, 32),
        data_es,
        ls,
        color=color,
        lw=lw,
        ms=ms,
        label=f"{setting} {device.upper()} ($\\chi^2$={chi2:.3g})",
    )

    axes[1, 0].plot(
        np.linspace(0.0, 1.0, 32),
        frac_diff(data_es, UltraXPSI_es),
        ls,
        color=color,
        lw=lw,
        ms=ms,
    )

    time_suffix = "s" if device == "cpu" else "ms"

    axes[0, 1].plot(
        np.arange(30, 300),
        data_ps,
        ls,
        color=color,
        lw=lw,
        ms=ms,
        label=f"{setting} {device.upper()} (time={calc_time:.3g} {time_suffix})",
    )

    axes[1, 1].plot(
        np.arange(30, 300),
        frac_diff(data_ps, UltraXPSI_ps),
        ls,
        color=color,
        lw=lw,
        ms=ms / 2,
    )


for device, ls in [
    ("cpu", "o-"),
    ("gpu", "--"),
]:
    for setting, color in [
        ("std", "b"),
        ("high", "r"),
        # ("ultra", "g"),
    ]:
        plot_data(case_name, setting, ls, color, device)


axes[0, 0].set_ylabel("Counts")
axes[0, 0].set_title(f"{case_name} energy-summed waveform & diff.")
axes[0, 0].legend(fontsize=8)
axes[1, 0].set_xlabel("Rotational phase [cycles]")
axes[0, 1].set_ylabel("Counts")
axes[0, 1].set_title(f"{case_name} spectra & diff.")
axes[0, 1].legend(fontsize=8)
axes[1, 1].set_xlabel("Channels")


if choice in ["b"]:
    ylim_save = axes[0, 0].get_ylim()
    axes[0, 0].set_ylim(
        ylim_save[0], ylim_save[0] + (ylim_save[1] - ylim_save[0]) * 1.3
    )
    lp = 20
else:
    lp = None


axes[1, 0].set_ylabel("Fractional difference [%]", labelpad=lp)
axes[1, 1].set_ylabel("Fractional difference [%]")

fig.tight_layout()
plt.subplots_adjust(hspace=0)

fname = f"p_cmp1d_{choice}.pdf"
fig.savefig(fname)
print(f"Figure saved to {fname}")
