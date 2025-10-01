import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(
    2,
    5,
    left=0.07,
    right=0.99,
    top=0.95,
    bottom=0.1,
    wspace=0.25,
    hspace=0.3,
)

for i, s in enumerate(list(map(chr, range(ord("a"), ord("j") + 1)))):
    subgs = gs[i].subgridspec(2, 1, height_ratios=[2, 1], hspace=0)
    ax0 = fig.add_subplot(subgs[0])
    ax1 = fig.add_subplot(subgs[1], sharex=ax0)

    ax0.tick_params(labelbottom=False)

    dat = np.loadtxt(f"ext/apjlab5968/OS1{s}_test_IM.txt")
    ax0.plot(dat[:, 0], dat[:, 1], "-", color="k", label="IM")
    ax0.set_title(f"OS1{s}")

    for dev, color in zip(["cpu", "gpu"], ["C0", "C1"]):
        dat1 = np.loadtxt(f"output/os1{s}_{dev}.txt")
        ax0.plot(dat[:, 0], dat1, "-", color=color, label=dev.upper())

        yhat = dat1
        y = dat[:, 1]
        rel_err = (yhat - y) / np.median(y) * 1e3
        ax1.plot(dat[:, 0], rel_err, "-", color=color, lw=1, label=dev.upper())

    ax1.set_ylim(-1.99, 1.99)
    ax1.hlines([-1, 1], 0, 1, color="r", lw=1, ls="--")

    if i == 0:
        ax0.legend(frameon=False, fontsize=10)

    if i > 4:
        ax1.set_xlabel("Rotational Phase", fontsize=12)

    if i % 5 == 0:
        ax0.set_ylabel("$f_E$ (cm$^{-2}$s$^{-1}$keV$^{-1}$)", fontsize=12)
        pad = 20 if i == 0 else 12
        ax1.set_ylabel("Diff./median (‰)", labelpad=pad, fontsize=12)

    ax0.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

plt.savefig("p_os1_all.pdf")
print("Saved ./p_os1_all.pdf")
