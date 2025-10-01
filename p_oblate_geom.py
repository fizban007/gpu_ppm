import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"


fig, ax = plt.subplots(figsize=(2.5, 2))
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

eye = np.array([3, 2, 1], dtype=float)
eye /= np.linalg.norm(eye)
x = np.array([1, 0, 0], dtype=float)
y = np.array([0, 1, 0], dtype=float)
z = np.array([0, 0, 1], dtype=float)
o = np.array([0, 0, 0], dtype=float)


def rotate_toward(a, b, angle):
    """Rotate unit vector a vector unit toward b by 'angle' [rad]."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    axis = np.cross(a, b)
    na = np.linalg.norm(axis)
    if na < 1e-12:
        return a
    axis /= na
    c, s = np.cos(angle), np.sin(angle)
    return a * c + np.cross(axis, a) * s + axis * np.dot(axis, a) * (1 - c)


eye_up = rotate_toward(eye, z, np.radians(90))
eye_right = np.cross(eye_up, eye)
eye_right /= np.linalg.norm(eye_right)


def proj(v):
    d = np.dot(eye, v)
    u = v - d * eye
    return np.array([np.dot(u, eye_right), np.dot(u, eye_up)])


# draw circle in xy plane
R = 2
theta1 = np.arctan(-eye[0] / eye[1])
theta2 = theta1 + np.radians(180)
theta = np.linspace(theta1, theta2, 100)
circle = np.array([R * np.cos(theta), R * np.sin(theta), np.zeros_like(theta)])
circle_proj = [proj(circle[:, i]) for i in range(circle.shape[1])]
ax.plot(*zip(*circle_proj), "k-", lw=1)
theta3 = theta2 + np.radians(180)
theta = np.linspace(theta2, theta3, 100)
circle = np.array([R * np.cos(theta), R * np.sin(theta), np.zeros_like(theta)])
circle_proj = [proj(circle[:, i]) for i in range(circle.shape[1])]
ax.plot(*zip(*circle_proj), "k--", lw=1)

# draw sphere cap
p_left = np.array([R * np.cos(theta1), R * np.sin(theta1), 0])
theta = np.linspace(0, np.radians(180), 100)
cap_dots = [rotate_toward(p_left, z, th) for th in theta]
oblate = 1 - 0.3 * np.sin(theta) ** 2
cap_dots_proj = [R * proj(v) * ob for v, ob in zip(cap_dots, oblate)]
ax.plot(*zip(*cap_dots_proj), "k-", lw=1)

theta_k = np.radians(30)
p_left = np.array([R, 0, 0])
theta = np.linspace(0, theta_k, 100)
cap_dots = [rotate_toward(p_left, z, th) for th in theta]
oblate = 1 - 0.3 * np.sin(theta) ** 2
cap_dots = [R * v * ob for v, ob in zip(cap_dots, oblate)]
k_mid = cap_dots[-1]
ax.plot(*zip(*[proj(v) for v in cap_dots]), "k-", lw=1)


def draw_dash(beg, end):
    ax.plot(*zip(proj(beg), proj(end)), "k--", lw=1)


def draw_arr(beg, end):
    ax.add_patch(
        FancyArrowPatch(
            proj(beg),
            proj(end),
            mutation_scale=20,
            arrowstyle="-|>,head_width=0.1",
            color="k",
            shrinkA=0,
            shrinkB=0,
        )
    )


# draw axes
draw_dash(o, R * x)
draw_arr(R * x, 3.5 * x)
draw_dash(o, R * y)
draw_arr(R * y, 3 * y)
min_ob = 0.7
draw_dash(o, R * z * min_ob)
draw_arr(R * z * min_ob, 2 * z)


# draw patch on sphere
theta1 = np.radians(50)
theta2 = np.radians(60)
phi1 = np.radians(70)
phi2 = np.radians(80)
p1 = R * np.array(
    [
        np.sin(theta1) * np.cos(phi1),
        np.sin(theta1) * np.sin(phi1),
        np.cos(theta1) * (1 - 0.3 * np.cos(theta1) ** 2),
    ]
)
p2 = R * np.array(
    [
        np.sin(theta1) * np.cos(phi2),
        np.sin(theta1) * np.sin(phi2),
        np.cos(theta1) * (1 - 0.3 * np.cos(theta1) ** 2),
    ]
)
p3 = R * np.array(
    [
        np.sin(theta2) * np.cos(phi2),
        np.sin(theta2) * np.sin(phi2),
        np.cos(theta2) * (1 - 0.3 * np.cos(theta2) ** 2),
    ]
)
p4 = R * np.array(
    [
        np.sin(theta2) * np.cos(phi1),
        np.sin(theta2) * np.sin(phi1),
        np.cos(theta2) * (1 - 0.3 * np.cos(theta2) ** 2),
    ]
)
patch = np.array([p1, p2, p3, p4])
ax.fill(
    *zip(*[proj(p) for p in patch]),
    facecolor="lightsalmon",
    edgecolor="orangered",
    linewidth=1,
)

p0 = (p1 + p2 + p3 + p4) / 4
p0_xy = p0 - np.dot(p0, z) * z
draw_dash(o, p0_xy)
draw_dash(p0_xy, p0)
draw_dash(o, p0)

r_hat = p0 / np.linalg.norm(p0)
n_hat = rotate_toward(r_hat, z, np.radians(25))
draw_arr(p0, p0 + 1.5 * r_hat)
draw_arr(p0, p0 + 1.05 * n_hat)

k = np.array([np.cos(theta_k), 0, np.sin(theta_k)], dtype=float)
k /= np.linalg.norm(k)
k_end = 4.8 * k
ax.add_patch(
    FancyArrowPatch(
        proj(k_mid),
        proj(k_end),
        mutation_scale=20,
        arrowstyle="-|>,head_width=0.1",
        color="k",
        shrinkA=0,
        shrinkB=0,
    )
)
ax.plot(*zip(proj(o), proj(k_mid)), "k--", lw=1)

k0 = rotate_toward(r_hat, k, np.radians(40))
draw_arr(p0, p0 + 2.1 * k0)


ax.annotate(r"$\hat{x}$", (-2.1, -0.8))
ax.annotate(r"$\hat{y}$", (2.5, -0.46))
ax.annotate(r"$\hat{z}$", (-0.17, 1.85))
ax.annotate(r"$\hat{k}$", (-2.2, 1.44))
ax.annotate(r"$\hat{r}$", (2.04, 1.1))
ax.annotate(r"$\hat{n}$", (1.48, 1.48))
ax.annotate(r"$\hat{k}_0$", (0.8, 1.5))


def draw_arc(center, a, b, radius, fac=1):
    theta1 = np.arctan2(a[1] - center[1], a[0] - center[0])
    theta2 = np.arctan2(b[1] - center[1], b[0] - center[0])
    if theta2 < theta1:
        theta2 += 2 * np.pi
    angle_arc = Arc(
        center,
        radius,
        radius * 0.7 * fac,
        theta1=np.degrees(theta1),
        theta2=np.degrees(theta2),
    )
    ax.add_patch(angle_arc)


draw_arc(proj(o), proj(z), proj(k), 0.6)
ax.annotate(r"$i$", (-0.22, 0.25))

draw_arc(proj(o), proj(r_hat), proj(z), 0.4)
ax.annotate(r"$\theta$", (0.1, 0.2))

draw_arc(proj(o), proj(x), proj(p0_xy), 0.5, fac=0.7)
ax.annotate(r"$\phi$", (-0.1, -0.3))

draw_arc(proj(o), proj(r_hat), proj(k), 1.3)
ax.annotate(r"$\psi$", (-0.5, 0.4))


draw_arc(proj(p0), proj(p0 + r_hat), proj(p0 + n_hat), 0.4)
ax.annotate(r"$\eta$", (1.22, 0.85))

draw_arc(proj(p0), proj(p0 + n_hat), proj(p0 + k0), 0.5)
ax.annotate(r"$\sigma$", (1.03, 0.88))

draw_arc(proj(p0), proj(p0 + r_hat), proj(p0 + k0), 1.05)
ax.annotate(r"$\alpha$", (0.98, 1.07))

ax.axis("off")
plt.savefig("oblate_geom.pdf")
print("Saved ./oblate_geom.pdf")
