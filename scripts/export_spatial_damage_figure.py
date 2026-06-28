"""Spatial gradient-damage and phase-field demonstration for paper §5.6.
Two panels using the Mode-3 Tri-HB reference conditions (L=50 mm specimen,
peak orthotropic damage Dy≈0.75 from the existing damage simulation):

(a) Gradient regularisation — D(x) from the Helmholtz BVP
      D - ell^2 * D'' = D_local,  D'(0)=D'(L)=0
    for ell = 0 (local, ill-posed), 2, 5, 10 mm. The local form collapses to a
    point (mesh-dependent); gradient regularisation smears it to a band ~2*sqrt(2)*ell.
    Internal length calibration: ell ≈ delta_crack / sqrt(2) from CT crack spacing.

(b) Phase-field crack profile (Level 4, AT2 model) — 1-D equilibrium solution
      phi(x) = exp(-|x - x0| / ell)
    for ell = 2, 5, 10 mm. Band width at the e^{-1} level = 2*ell, directly
    measurable from CT cross-sections. Griffith energy: Gc = W_diss_total / A_fracture.

Run: py -3.13 scripts/export_spatial_damage_figure.py
"""
from pathlib import Path
import numpy as np
from scipy.linalg import solve_banded
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
PUB = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
       "orange": "#E69F00", "purple": "#7B3FE4", "black": "#222222", "grey": "#888888"}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": PUB["black"], "grid.color": "#D9DEE7", "grid.linewidth": 0.55,
    "legend.frameon": False, "savefig.facecolor": "white"})

# ---- specimen geometry -------------------------------------------------------
L = 50.0          # specimen length (mm)
N = 600
x = np.linspace(0, L, N)
h = x[1] - x[0]
x0 = L / 2        # damage / crack centre

# ---- D_local: Gaussian peak at centre, background from Mode-3 conditions ---
# Peak orthotropic Dy≈0.75 (lateral damage dominant). Background Dx≈0.11.
D_peak, D_bg, sigma_x = 0.75, 0.10, 5.5   # mm half-width
D_local = D_bg + (D_peak - D_bg) * np.exp(-0.5 * ((x - x0) / sigma_x) ** 2)


def solve_gradient(D_loc, ell):
    """Tridiagonal solve for D - ell^2*D'' = D_loc, Neumann BCs."""
    if ell < 1e-6:
        return D_loc.copy()
    r = (ell / h) ** 2
    ab = np.zeros((3, N))
    ab[0, 1:] = -r           # superdiagonal
    ab[1, :] = 1.0 + 2.0*r  # main diagonal
    ab[2, :-1] = -r           # subdiagonal
    ab[1, 0]  = 1.0 + r      # left Neumann (ghost: D_{-1}=D_1)
    ab[1, -1] = 1.0 + r      # right Neumann
    return solve_banded((1, 1), ab, D_loc)


# ---- Plot --------------------------------------------------------------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 5.1), dpi=240)

# ---- Panel (a): gradient regularisation -------------------------------------
ells_a   = [0,        2,           5,              10]
cols_a   = [PUB["black"], PUB["blue"], PUB["vermillion"], PUB["green"]]
ls_a     = ["--",     "-",         "-",            "-"]
labs_a   = [r"$\ell=0$ (local, ill-posed)",
            r"$\ell=2$ mm",
            r"$\ell=5$ mm",
            r"$\ell=10$ mm"]
lws_a    = [1.6, 2.2, 2.2, 2.2]

axL.plot(x, D_local, color=PUB["grey"], lw=1.2, ls=":",
         label=r"$D_{\rm local}$ (Perzyna, pointwise)", zorder=2)
for ell, col, ls, lab, lw in zip(ells_a, cols_a, ls_a, labs_a, lws_a):
    D_reg = solve_gradient(D_local, ell)
    axL.plot(x, D_reg, color=col, lw=lw, ls=ls, label=lab, zorder=3)

# annotate process-zone width for ell=5
D5 = solve_gradient(D_local, 5)
half = D_bg + 0.5 * (D5.max() - D_bg)
left_idx  = np.argmin(np.abs(D5[:N//2] - half))
right_idx = N//2 + np.argmin(np.abs(D5[N//2:] - half))
if 0 < left_idx < N and 0 < right_idx < N:
    axL.annotate("", xy=(x[right_idx], half), xytext=(x[left_idx], half),
                 arrowprops=dict(arrowstyle="<->", color=PUB["vermillion"], lw=1.3))
    axL.text(x0, half + 0.03, r"$\approx 2\sqrt{2}\,\ell$",
             ha="center", fontsize=9.5, color=PUB["vermillion"])

axL.text(1.5, 0.14,
         "CT crack spacing\n" + r"$\Rightarrow\;\ell\approx\delta_{\rm crack}/\sqrt{2}$",
         fontsize=8.5, color=PUB["black"],
         bbox=dict(boxstyle="round,pad=0.3", fc="#F0F8FF", ec="#AAAAAA", alpha=0.9))

axL.set_xlabel(r"Position along specimen axis, $x$ (mm)")
axL.set_ylabel(r"Damage, $D(x)$")
axL.set_title(r"(a) Level 3: gradient regularisation"
              "\n" r"$D-\ell^{2}\nabla^{2}D=D_{\rm local},\;\nabla D\cdot\mathbf{n}=0$",
              fontsize=12, pad=7)
axL.set_ylim(0, 1.0); axL.set_xlim(0, L)
axL.legend(fontsize=9.5, loc="upper right"); axL.grid(True)
axL.spines["top"].set_visible(False); axL.spines["right"].set_visible(False)

# ---- Panel (b): phase-field crack profiles ----------------------------------
ells_b = [2,          5,              10]
cols_b = [PUB["blue"], PUB["vermillion"], PUB["green"]]
labs_b = [r"$\ell=2$ mm", r"$\ell=5$ mm", r"$\ell=10$ mm"]

for ell, col, lab in zip(ells_b, cols_b, labs_b):
    phi = np.exp(-np.abs(x - x0) / ell)
    axR.plot(x, phi, color=col, lw=2.3, label=lab)
    # bracket the crack band width 2*ell at e^{-1}
    axR.plot([x0 - ell, x0 + ell], [np.exp(-1), np.exp(-1)], color=col,
             lw=0.9, ls=":", alpha=0.55)

axR.axhline(np.exp(-1), color=PUB["grey"], lw=1.0, ls="--", alpha=0.8)
axR.text(1.5, np.exp(-1) + 0.025, r"$\phi = e^{-1}$,  band width $= 2\ell$",
         fontsize=9, color=PUB["grey"])
axR.axvline(x0, color=PUB["grey"], lw=0.8, ls=":", alpha=0.5)
axR.text(x0 + 0.8, 0.93, "crack", fontsize=9, color=PUB["grey"])

axR.text(1.5, 0.08,
         r"$G_c = W_{\rm diss}^{\rm total}/A_{\rm fracture}$" + "\n"
         + r"CT area $\Rightarrow A_{\rm fracture}$",
         fontsize=8.5, color=PUB["black"],
         bbox=dict(boxstyle="round,pad=0.3", fc="#FFF8F0", ec="#AAAAAA", alpha=0.9))

axR.set_xlabel(r"Position along specimen axis, $x$ (mm)")
axR.set_ylabel(r"Phase field, $\phi(x)$  (0 = intact,  1 = broken)")
axR.set_title(r"(b) Level 4: phase-field crack profile"
              "\n" r"$\phi(x)=\exp(-|x-x_0|/\ell)$,  $\eta\dot\phi-G_c\ell\nabla^2\phi"
              r"+(G_c/\ell)\phi=2(1-\phi)\Psi^+$",
              fontsize=12, pad=7)
axR.set_ylim(0, 1.08); axR.set_xlim(0, L)
axR.legend(fontsize=9.5, loc="upper right"); axR.grid(True)
axR.spines["top"].set_visible(False); axR.spines["right"].set_visible(False)

fig.tight_layout(pad=1.3)
fig.savefig(OUT / "trihb_spatial_damage.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Wrote {OUT / 'trihb_spatial_damage.png'}")

# ---- Calibration check: G_c estimate from Mode-3 W_diss --------------------
# W_diss from the energy simulation (peak ~0.45 MJ/m^3 for Mode-3 reference)
# Specimen: 50mm cube → V = 50^3 mm^3 = 1.25e-4 m^3
# Primary fracture plane (lateral): A = 50x50 = 2500 mm^2 = 2.5e-3 m^2
# G_c = W_diss * V / A = W_diss * L (since V/A = L)
W_diss_MJm3 = 0.45   # MJ/m^3 (approximate from Mode-3 energy figure)
L_m = 0.050
G_c_estimate = W_diss_MJm3 * 1e6 * L_m   # J/m^2
print(f"  G_c upper bound from Mode-3 W_diss: {G_c_estimate:.0f} J/m^2")
print(f"  Granite literature range: 50-200 J/m^2 (consistent if A_fracture > {W_diss_MJm3*1e6*L_m/200*1e4:.0f} cm^2)")
