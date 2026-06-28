"""Dynamic increase factor (DIF) for the failure envelope, Zhu et al. (2024),
DOI 10.1016/j.ijrmms.2024.105948. Piecewise fit: a rate-independent static branch
below a transition strain rate and a power-law dynamic branch above it.

  DIF_c = 1                                  , edot < 10 s^-1   (compression)
        = 1 - 10^beta/alpha + edot^beta/alpha, edot >= 10
  DIF_t = 1                                  , edot < 1 s^-1    (tension)
        = 1 - 1/alpha + edot^(2 beta)/alpha  , edot >= 1

with the simplification k_c = k_t = 0, b = 1 (small sub-transition rate effect),
alpha = 10, beta = 0.57. Both branches are continuous at the transition. The
compression branch is the DIF_c that scales A_eff in the failure envelope
(Eq. qf); the tension branch scales the tensile strength used in the directional
damage of the orthotropic section.

Replaces the legacy single-slope semi-log DIF figure in paper Fig. 4(b).
Run: py -3.13 scripts/export_dif_figure.py
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

ALPHA, BETA = 10.0, 0.57          # Zhu et al. (2024) fitting constants
ETC, ETT = 10.0, 1.0              # transition (inflection) rates: compression, tension


def dif_c(ed):
    return np.where(ed < ETC, 1.0, 1.0 - ETC**BETA / ALPHA + ed**BETA / ALPHA)


def dif_t(ed):
    return np.where(ed < ETT, 1.0, 1.0 - 1.0 / ALPHA + ed**(2.0 * BETA) / ALPHA)


def main():
    ed = np.logspace(-1, 3, 500)
    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)

    ax.plot(ed, dif_c(ed), color=PUB["blue"], lw=2.4,
            label=r"$\mathrm{DIF}_c$ (compression, $\dot\varepsilon_t=10\ \mathrm{s^{-1}}$)")
    ax.plot(ed, dif_t(ed), color=PUB["vermillion"], lw=2.4, ls="--",
            label=r"$\mathrm{DIF}_t$ (tension, $\dot\varepsilon_t=1\ \mathrm{s^{-1}}$)")

    # transition rates
    ax.axvline(ETC, color=PUB["blue"], lw=1.0, ls=":", alpha=0.7)
    ax.axvline(ETT, color=PUB["vermillion"], lw=1.0, ls=":", alpha=0.7)

    # representative Mode-3 compression point used in section 5.2
    ed0 = 36.0
    ax.plot([ed0], [dif_c(ed0)], "o", color=PUB["blue"], ms=7, zorder=5)
    ax.annotate(r"Mode-3 pulse: $\mathrm{DIF}_c\approx1.4$" + f"\n($\\dot\\varepsilon\\approx{ed0:.0f}\\ \\mathrm{{s^{{-1}}}}$)",
                xy=(ed0, dif_c(ed0)), xytext=(ed0 * 1.5, 1.05),
                fontsize=9.5, color=PUB["blue"],
                arrowprops=dict(arrowstyle="->", color=PUB["blue"], lw=1.0))

    ax.text(2.2, 1.04, "static branch\n(rate-independent,\n$k=0,\\ b=1$)",
            fontsize=9, color=PUB["black"], ha="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Strain rate, $\dot{\varepsilon}$ (s$^{-1}$)")
    ax.set_ylabel(r"Dynamic increase factor, DIF")
    ax.set_ylim(0.9, 40.0)
    ax.set_title(r"Piecewise DIF (Zhu et al. 2024): $\alpha=10,\ \beta=0.57$",
                 fontsize=12.5, pad=8)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, which="both", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=1.1)
    fig.savefig(OUT / "trihb_dif.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for r in (1, 10, 36, 50, 100, 300):
        print(f"  edot={r:4d} s^-1:  DIF_c={float(dif_c(np.array(float(r)))):.3f}   "
              f"DIF_t={float(dif_t(np.array(float(r)))):.3f}")
    print(f"Wrote {OUT / 'trihb_dif.png'}")


if __name__ == "__main__":
    main()
