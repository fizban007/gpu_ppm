import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.ticker as ticker

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

for num in [1, 2]:
    datasets = [np.loadtxt(f"output/test{num}{i}_flux.txt").T for i in range(4)]

    fig, axs = plt.subplots(1, 4, figsize=(10, 3))
    plt.subplots_adjust(wspace=0.1, left=0.06, right=0.9)
    max_abs = max(-np.min(datasets), np.max(datasets))
    norm = colors.SymLogNorm(
        linthresh=1e-9, linscale=2, vmin=-max_abs, vmax=max_abs, base=10
    )

    images = []
    for ax, data in zip(axs.flat, datasets):
        images.append(
            ax.imshow(
                data,
                norm=norm,
                aspect=180 / (5 - 0.1) / 64,
                cmap="coolwarm",
                origin="lower",
                extent=(0, 0.999, 0.1, 5),
            )
        )
        ax.set_yscale("log")
        ax.set_yticks([])
        ax.set_xlabel("Rotational phase")

    axs[0].set_title("(a)")
    axs[1].set_title("(b)")
    axs[2].set_title("(c)")
    axs[3].set_title("(d)")

    axs[0].yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
    axs[0].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    axs[0].set_ylabel("Energy (keV)")

    fig.colorbar(
        images[3],
        ax=axs,
        fraction=0.011,
        pad=0.02,
        label="$f_E$ (cm$^{-2}$s$^{-1}$keV$^{-1}$)",
    )

    ofname = f"test{num}_flux.pdf"
    plt.savefig(ofname)
    print(f"Figure saved as ./{ofname}")
