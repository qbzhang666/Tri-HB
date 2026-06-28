"""True-triaxial failure criteria for paper section 5.2, at realistic Tri-HB
conditions: the strength is the DYNAMIC (rate-inflated, DIF) surface and the
principal stresses are TOTAL = static pre-stress (<=50 MPa) + dynamic confinement
developed during impact. Five criteria, all calibrated to the SAME dynamic UCS and
confinement ratio k_conf:
  Lode-MC (default), Mogi-Coulomb, modified Lade, Yu unified strength theory,
  modified Wiebols-Cook.

  (a) sigma_1 at failure vs the intermediate stress sigma_2 (fixed sigma_3): the
      sigma_2-blind Mohr-Coulomb is flat; the sigma_2-aware surfaces hump. sigma_2
      beyond the 50 MPa static cap is reached only by dynamic confinement.
  (b) The per-criterion failure index F (=1 at the surface) along the Mode-3
      stress path of Fig. 3; a single dynamic pulse is sub-critical (F<1) against
      the dynamic surface, so failure is cumulative (section 5.5).

NOTE: the Yu-UST and Wiebols-Cook coefficient forms are the standard published
ones to the author's best recollection and the DIF is representative; verify
against Yu (2002) and Wiebols & Cook (1968) before submission.
Run: py -3.13 scripts/export_paper_criteria_figure.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
PUB = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
       "orange": "#E69F00", "purple": "#7B3FE4", "black": "#222222"}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": PUB["black"], "grid.color": "#D9DEE7", "grid.linewidth": 0.55,
    "legend.frameon": False, "savefig.facecolor": "white"})

# --- Calibration: DYNAMIC strength (UCS x DIF) and confinement ratio ------------
UCS_static, DIF, k_conf, nu = 80.0, 1.4, 4.0, 0.20
UCS = UCS_static * DIF                              # dynamic UCS ~ 112 MPa
sphi = (k_conf - 1.0) / (k_conf + 1.0); cphi = np.sqrt(1.0 - sphi**2); tphi = sphi / cphi
c_coh = UCS * (1.0 - sphi) / (2.0 * cphi)
a_mc = (2.0 * np.sqrt(2.0) / 3.0) * c_coh * cphi
b_mc = (2.0 * np.sqrt(2.0) / 3.0) * sphi
S_lade = c_coh / tphi
eta_lade = 4.0 * tphi**2 * (9.0 - 7.0 * sphi) / (1.0 - sphi)
A_f = 3.0 * UCS / (k_conf + 2.0); B_f = 3.0 * (k_conf - 1.0) / (k_conf + 2.0); rho = 0.70
b_yu = 0.5                                          # Yu UST unified parameter
# modified Wiebols-Cook: sqrt(J2) = Aw + Bw*J1 + Cw*J1^2, fit to the compression
# meridian (UCS and two triaxial-compression points).
def _txc(s3):  # (J1, sqrtJ2) on the Coulomb TXC line
    s1 = UCS + k_conf * s3
    return s1 + 2 * s3, (s1 - s3) / np.sqrt(3.0)
_pts = [(UCS, UCS / np.sqrt(3.0)), _txc(10.0), _txc(80.0)]
_M = np.array([[1.0, j1, j1**2] for j1, _ in _pts]); _y = np.array([sj for _, sj in _pts])
Aw, Bw, Cw = np.linalg.solve(_M, _y)


def sqrtJ2(s1, s2, s3):
    return np.sqrt((1.0 / 6.0) * ((s1 - s2)**2 + (s2 - s3)**2 + (s3 - s1)**2))


def invariants(s1, s2, s3):
    p = (s1 + s2 + s3) / 3.0
    q = np.sqrt(0.5 * ((s1 - s2)**2 + (s2 - s3)**2 + (s3 - s1)**2))
    a1, a2, a3 = s1 - p, s2 - p, s3 - p
    J2 = 0.5 * (a1**2 + a2**2 + a3**2); J3 = a1 * a2 * a3
    arg = np.where(J2 > 1e-9, (3.0 * np.sqrt(3.0) / 2.0) * J3 / np.maximum(J2, 1e-9)**1.5, 0.0)
    return p, q, np.rad2deg((1.0 / 3.0) * np.arccos(np.clip(arg, -1.0, 1.0)))


def lode_h(th_deg):
    return 1.0 - (1.0 - rho) * (1.0 - np.cos(np.deg2rad(3.0 * th_deg))) / 2.0


def F_lode(s1, s2, s3):
    p, q, th = invariants(s1, s2, s3)
    return q / np.maximum((A_f + B_f * np.maximum(p, 0.0)) * lode_h(th), 1e-9)


def F_mogi(s1, s2, s3):
    return ((np.sqrt(2.0) / 3.0) * np.sqrt(0.5 * ((s1 - s2)**2 + (s2 - s3)**2 + (s3 - s1)**2))) \
        / np.maximum(a_mc + b_mc * 0.5 * (s1 + s3), 1e-9)


def F_lade(s1, s2, s3):
    I1 = (s1 + S_lade) + (s2 + S_lade) + (s3 + S_lade)
    I3 = (s1 + S_lade) * (s2 + S_lade) * (s3 + S_lade)
    return (I1**3 / np.maximum(I3, 1e-9)) / (27.0 + eta_lade)


def F_yu(s1, s2, s3):  # max of the two UST branches, normalised by sigma_c
    f1 = s1 - k_conf * (b_yu * s2 + s3) / (1.0 + b_yu)
    f2 = (s1 + b_yu * s2) / (1.0 + b_yu) - k_conf * s3
    return np.maximum(f1, f2) / UCS


def F_wc(s1, s2, s3):
    J1 = s1 + s2 + s3
    return sqrtJ2(s1, s2, s3) / np.maximum(Aw + Bw * J1 + Cw * J1**2, 1e-9)


def sigma1_fail(Ffun, s2, s3):
    s1g = np.linspace(max(s2, s3) + 1e-3, max(s2, s3) + 1500.0, 8000)
    g = np.array([Ffun(s, s2, s3) for s in s1g]) - 1.0
    cr = np.where(np.diff(np.sign(g)) > 0)[0]
    if len(cr) == 0:
        return np.nan
    i = cr[0]
    return s1g[i] - g[i] * (s1g[i + 1] - s1g[i]) / (g[i + 1] - g[i])


def mode3_path():
    E = 15e9; M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu)); cp0 = np.sqrt(M / 2650.0)
    L = 0.05; t = np.linspace(0, 360e-6, 4000); td = 200e-6; ctr = L / (2 * cp0)

    def hs(delay, dur):
        tau = t - delay; g = np.zeros_like(tau); m = (tau >= 0) & (tau <= dur)
        g[m] = np.sin(np.pi * tau[m] / dur); return g
    sx = 20.0 + 400.0 * hs(ctr, td)
    glat = hs(ctr + 0.9 * L / cp0, td * 1.15)
    sy = 15.0 + 0.18 * 400.0 * glat; sz = 10.0 + 0.15 * 400.0 * glat
    s1 = np.maximum.reduce([sx, sy, sz]); s3 = np.minimum.reduce([sx, sy, sz])
    return t * 1e6, s1, sx + sy + sz - s1 - s3, s3


CRIT = [(F_lode, PUB["blue"], "Lode--MC (default)"), (F_mogi, PUB["vermillion"], "Mogi--Coulomb"),
        (F_lade, PUB["green"], "modified Lade"), (F_yu, PUB["purple"], "Yu UST ($b{=}0.5$)"),
        (F_wc, PUB["orange"], "mod. Wiebols--Cook")]


def main():
    print(f"  dyn UCS={UCS:.0f} (DIF={DIF}); UCS check (s2=s3=0): " +
          " ".join(f"{lab.split()[0]}={sigma1_fail(F,0,0):.0f}" for F, _, lab in CRIT))
    print(f"  TXC (s2=s3=50, =312): " +
          " ".join(f"{lab.split()[0]}={sigma1_fail(F,50,50):.0f}" for F, _, lab in CRIT))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 5.2), dpi=240)
    s3 = 50.0; s2s = np.linspace(s3, 130.0, 180)
    for F, col, lab in CRIT:
        s1f = np.array([sigma1_fail(F, s2, s3) for s2 in s2s])
        ok = np.isfinite(s1f) & (s1f >= s2s)
        axL.plot(s2s[ok], s1f[ok], color=col, lw=2.2, label=lab)
    axL.axhline(UCS + k_conf * s3, color=PUB["black"], ls="--", lw=1.4)
    axL.text(75, UCS + k_conf * s3 + 6, r"classic Mohr--Coulomb ($\sigma_2$-blind)",
             fontsize=10, color=PUB["black"], ha="left")
    axL.set_xlabel(r"Intermediate stress, $\sigma_2$ (MPa, total = static + dynamic)")
    axL.set_ylabel(r"Dynamic strength, $\sigma_1$ at failure (MPa)")
    axL.set_title(r"(a) Dynamic envelope ($\sigma_3=50$ MPa, DIF$\approx1.4$)", fontsize=13.5, pad=8)
    axL.legend(fontsize=9.5, loc="upper left"); axL.grid(True)
    axL.spines["top"].set_visible(False); axL.spines["right"].set_visible(False)

    tus, s1, s2, s3p = mode3_path()
    for F, col, lab in CRIT:
        axR.plot(tus, np.maximum(F(s1, s2, s3p), 0.0), color=col, lw=2.0, label=lab)
    axR.axhline(1.0, color=PUB["black"], ls="--", lw=1.2)
    axR.text(150, 1.05, "failure surface, $F=1$", fontsize=10, color=PUB["black"])
    axR.set_ylim(0, 1.65)
    axR.set_xlabel(r"Time, $t$ ($\mu$s)"); axR.set_ylabel(r"Failure index, $F$ (=1 at surface)")
    axR.set_title(r"(b) Criteria along the Mode-3 path (Fig.~3)", fontsize=13.5, pad=8)
    axR.legend(fontsize=10, loc="upper right"); axR.grid(True)
    axR.spines["top"].set_visible(False); axR.spines["right"].set_visible(False)

    fig.tight_layout(pad=1.3)
    fig.savefig(OUT / "trihb_criteria_comparison.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  peak F (Mode-3): " +
          " ".join(f"{lab.split()[0]}={F(s1,s2,s3p).max():.2f}" for F, _, lab in CRIT))
    print(f"Wrote {OUT/'trihb_criteria_comparison.png'}")


if __name__ == "__main__":
    main()
