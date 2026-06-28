"""Cyclic (repeated) dynamic loading: cumulative damage model for the Tri-HB.

Each cycle k resets the specimen to the in-situ static pre-stress and re-applies
the SAME dynamic pulse. Damage carries over between cycles and degrades the
strength, so a sub-critical pulse that the rock survives once can accumulate to
failure over many cycles (dynamic fatigue). Cycle-jump map:

    q_f(D)   = q_{f,0} (1-D)^{c_d}                          (strength degradation)
    F_k      = q_peak / q_f(D_{k-1}) = F_1 (1-D_{k-1})^{-c_d}
    dD_k     = K <(1-D_{k-1}) F_k - F_th>^m (1-D_{k-1})^alpha
             = K <F_1 (1-D_{k-1})^{1-c_d} - F_th>^m (1-D_{k-1})^alpha
    D_k      = min(D_{k-1} + dD_k, 1),   fail when D_k >= D_crit

  c_d < 1 -> load-shedding wins  -> shakedown (D -> D_inf < D_crit, infinite life)
  c_d = 1 -> neutral             -> Miner-like accumulation
  c_d > 1 -> strength degrades   -> accelerating progressive failure
  F_1 <= F_th                    -> below the endurance limit -> infinite life

F_1 is the first-cycle peak failure index q_peak/q_f from the SELECTED criterion
(Lode envelope / Mogi-Coulomb / modified Lade); K sets the first-cycle damage
(in the workspace K is fixed so dD_1 equals the single-pulse damage).
Run:  py -3.13 scripts/export_cyclic_damage_figure.py
"""
from __future__ import annotations
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


def cycle_damage(F1, Fth=0.6, c_d=1.5, alpha=1.0, m=2.0, K=0.075,
                 D_crit=0.95, Nmax=200000):
    """Iterate the cumulative-damage cycle map; return (D_k array, N_f or None)."""
    if F1 <= Fth:
        return np.zeros(1), None                       # below endurance limit
    D = [0.0]
    for k in range(1, Nmax + 1):
        Dp = D[-1]
        eff = F1 * (1.0 - Dp) ** (1.0 - c_d)           # (1-D) F_k
        dDk = K * max(eff - Fth, 0.0) ** m * (1.0 - Dp) ** alpha
        Dk = min(1.0, Dp + dDk)
        D.append(Dk)
        if Dk >= D_crit:
            return np.array(D), k
        if dDk < 1e-10:                                # shakedown converged
            return np.array(D), None
    return np.array(D), None


def main():
    # Calibrated to the experimental observation: high dynamic levels fail the
    # rock within ~10 impacts, with a SUDDEN damage threshold (sharp sigmoid) set
    # by the strength-degradation feedback. Defaults: c_d=2.5 (sharp), K=0.13.
    Fth, c_d, K = 0.6, 2.5, 0.13
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 5.4), dpi=240)

    # (a) Cumulative damage vs impact number, three realistic dynamic levels.
    cases = [(1.45, PUB["vermillion"], "high"), (1.20, PUB["orange"], "medium"),
             (1.00, PUB["blue"], "moderate"), (0.80, PUB["green"], "low")]
    knee_done = False
    for F1, col, lab in cases:
        D, Nf = cycle_damage(F1, Fth=Fth, c_d=c_d, K=K)
        axL.plot(np.arange(len(D)), D, color=col, lw=2.3, marker="o", ms=4,
                 label=rf"$F_1={F1}$ ({lab})" + (rf", $N_f={Nf}$" if Nf else ", survives"))
        if Nf and Nf <= 12:
            axL.scatter([Nf], [D[Nf]], color=col, s=80, zorder=6, edgecolor=PUB["black"], lw=0.7)
        # annotate the "sudden threshold" knee on the medium curve
        if Nf and not knee_done and lab == "medium":
            kk = int(np.argmax(D > 0.25))          # first impact past the knee
            axL.annotate("sudden damage\nthreshold", (kk, D[kk]),
                         xytext=(kk - 3.4, D[kk] + 0.22), fontsize=11.5,
                         arrowprops=dict(arrowstyle="->", color=PUB["black"], lw=1.1))
            knee_done = True
    axL.axhline(0.95, color=PUB["black"], ls=":", lw=1.3)
    axL.text(8.4, 0.905, r"$D_{\mathrm{crit}}$", fontsize=12, color=PUB["black"])
    axL.set_xlabel(r"Dynamic impact number, $N$"); axL.set_ylabel(r"Cumulative damage, $D_N$")
    axL.set_title(r"(a) Cumulative damage vs impact ($c_d=2.5$)", fontsize=14.5, pad=10)
    axL.set_xlim(0, 12); axL.set_ylim(0, 1.03)
    axL.legend(fontsize=10.5, loc="center right")
    axL.grid(True); axL.spines["top"].set_visible(False); axL.spines["right"].set_visible(False)

    # (b) S-N (dynamic-fatigue) curve over the realistic few-cycle range.
    F1s = np.linspace(0.61, 1.8, 130)
    Nfs = [cycle_damage(f, Fth=Fth, c_d=c_d, K=K)[1] for f in F1s]
    Nfs = [n if n else np.nan for n in Nfs]
    axR.plot(F1s, Nfs, color=PUB["purple"], lw=2.5)
    axR.axhspan(0, 10, color=PUB["green"], alpha=0.08)
    axR.text(1.25, 8.6, "fails within ~10 impacts", fontsize=11, color="#1f7a4d")
    axR.axvline(Fth, color=PUB["black"], ls="--", lw=1.3)
    axR.text(Fth + 0.03, 19, r"endurance limit $F_{\mathrm{th}}$", rotation=90,
             fontsize=11.5, color=PUB["black"], va="top")
    axR.axhline(1, color=PUB["vermillion"], ls=":", lw=1.3)
    axR.text(1.45, 1.4, "single-pulse failure", fontsize=10.5, color=PUB["vermillion"])
    axR.set_xlabel(r"First-impact dynamic level, $F_1=q_{\mathrm{peak}}/q_f$")
    axR.set_ylabel(r"Impacts to failure, $N_f$")
    axR.set_title(r"(b) Dynamic fatigue ($S$--$N$) curve", fontsize=14.5, pad=10)
    axR.set_ylim(0, 22)
    axR.grid(True); axR.spines["top"].set_visible(False); axR.spines["right"].set_visible(False)

    fig.tight_layout(pad=1.4)
    fig.savefig(OUT / "trihb_cyclic_damage.png", bbox_inches="tight", facecolor="white")
    hb = Path(__file__).resolve().parents[1] / "Handbook" / "figures" / "handbook_steps"
    if hb.exists():
        fig.savefig(hb / "step4_cyclic_damage.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    for F1, _, lab in cases:
        D, Nf = cycle_damage(F1, Fth=Fth, c_d=c_d, K=K)
        print(f"  F1={F1} ({lab}): N_f={Nf}, D=[{', '.join(f'{d:.2f}' for d in D[:min(len(D),12)])}]")
    print(f"Wrote {OUT/'trihb_cyclic_damage.png'}")


if __name__ == "__main__":
    main()
