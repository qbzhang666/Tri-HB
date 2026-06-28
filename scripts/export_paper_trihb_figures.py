"""Mode-3 (gas-gun Tri-HB static-dynamic) figures for the paper: a dynamic X
half-sine pulse superposed on a true-triaxial static pre-stress, with a
Poisson-driven lateral output on the Y and Z bars. Matches the current
wave_damage.py physics (Lode envelope, corrected energy, orthotropic damage).
Run:  py -3.13 scripts/export_paper_trihb_figures.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
PUB = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
       "orange": "#E69F00", "purple": "#7B3FE4", "sky": "#56B4E9", "black": "#222222"}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": PUB["black"], "axes.labelcolor": PUB["black"], "axes.titlecolor": PUB["black"],
    "xtick.color": PUB["black"], "ytick.color": PUB["black"], "grid.color": "#D9DEE7",
    "grid.linewidth": 0.55, "legend.frameon": False, "savefig.facecolor": "white"})


def half_sine(t, td, delay=0.0):
    tau = t - delay
    g = np.zeros_like(tau)
    m = (tau >= 0) & (tau <= td)
    g[m] = np.sin(np.pi * tau[m] / td)
    return g


def cdiff(y, t):
    return np.gradient(y, t) if len(y) > 1 else np.zeros_like(y)


def ctrap(y, t):
    o = np.zeros_like(y)
    if len(y) > 1:
        o[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))
    return o


def invariants(sx, sy, sz):
    p = (sx + sy + sz) / 3.0
    q = np.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2))
    s1, s2, s3 = sx - p, sy - p, sz - p
    J2 = (s1**2 + s2**2 + s3**2) / 2.0
    J3 = s1 * s2 * s3
    th = np.zeros_like(p); m = J2 > 1e-12
    arg = np.zeros_like(p)
    arg[m] = (3*np.sqrt(3)/2) * J3[m] / (J2[m]**1.5)
    th[m] = (1/3) * np.arccos(np.clip(arg[m], -1, 1))
    return p, q, np.rad2deg(th)


def lode_shape(th_deg, rho=0.70):
    th = np.deg2rad(th_deg)
    return 1.0 - (1.0 - rho) * (1.0 - np.cos(3.0 * th)) / 2.0


def style(ax, title):
    ax.set_title(title, fontsize=16, pad=8)
    ax.tick_params(labelsize=12.5); ax.xaxis.label.set_size(14.5); ax.yaxis.label.set_size(14.5)
    ax.grid(True); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    lg = ax.get_legend()
    if lg:
        for tx in lg.get_texts(): tx.set_fontsize(12)


def save(fig, ax, fn, title):
    style(ax, title); fig.tight_layout(); fig.savefig(OUT / fn, bbox_inches="tight", facecolor="white"); plt.close(fig)


def compute():
    # Gas-gun Tri-HB (Mode 3): X dynamic, Y/Z static confinement (true-triaxial
    # pre-stress), with a Poisson-driven lateral output recorded on the Y/Z bars.
    E_GPa, nu, rho, L_mm = 15.0, 0.20, 2650.0, 50.0
    UCS, k_conf = 80.0, 4.0
    sx0, sy0, sz0 = 20.0, 15.0, 10.0     # true-triaxial static pre-stress (sigma1>sigma2>sigma3)
    Ax = 400.0; td_us = 200.0
    sig_t = UCS / 12.0
    tau_D_us, alpha, m_over, beta, eps0, F0 = 30.0, 1.0, 2.0, 0.15, 1.0, 1.0

    E = E_GPa * 1e9
    M = E * (1 - nu) / ((1 + nu) * max(1 - 2 * nu, 1e-9))
    cp0 = np.sqrt(M / rho); L_m = L_mm / 1000.0
    t_us = np.linspace(0, 360, 5000); t = t_us * 1e-6; td = td_us * 1e-6
    centre = L_m / (2 * cp0)
    gx = half_sine(t, td, centre)                     # dynamic X pulse at specimen centre
    g_lat = half_sine(t, td * 1.15, centre + 0.9 * L_m / cp0)  # lagged, broadened lateral output
    sx = sx0 + Ax * gx
    sy = sy0 + 0.18 * Ax * g_lat                      # Poisson-driven Y output
    sz = sz0 + 0.15 * Ax * g_lat                      # slightly different -> directional
    p, q, theta, = invariants(sx, sy, sz)
    A_f, B_f = 3*UCS/(k_conf+2), 3*(k_conf-1)/(k_conf+2)
    qf = (A_f + B_f * np.maximum(p, 0.0)) * lode_shape(theta)
    F_index = q / np.maximum(qf, 1e-9)

    E_MPa = E_GPa * 1000.0
    el_x = (sx - nu*(sy+sz))/E_MPa; el_y = (sy - nu*(sx+sz))/E_MPa; el_z = (sz - nu*(sx+sy))/E_MPa
    edeq = np.sqrt(2/3*((cdiff(el_x,t)-cdiff(el_y,t))**2+(cdiff(el_y,t)-cdiff(el_z,t))**2+(cdiff(el_z,t)-cdiff(el_x,t))**2)/2)
    rate = (np.maximum(np.abs(edeq), 1e-12)/eps0)**beta
    tau_D = tau_D_us*1e-6

    # isotropic D (for energy) + orthotropic tensor
    D = np.zeros_like(t)
    for i in range(1, len(t)):
        ov = max(((1-D[i-1])*F_index[i-1]-1)/F0, 0.0)
        D[i] = min(1.0, D[i-1] + ((1-D[i-1])**alpha)/tau_D*(ov**m_over)*rate[i-1]*(t[i]-t[i-1]))
    eps_t0 = sig_t/E_MPa
    # Directional damage drivers: a dominant tensile (extensile-strain) term that
    # opens cracks normal to each axis, PLUS a small isotropic shear/compaction
    # term from the deviatoric failure index F=q/qf. The shear term represents the
    # wing-crack, shear-microcrack and grain-crushing damage that occurs even
    # under net compression, so the axial component is small but NONZERO
    # (Dx << Dy,Dz) rather than exactly zero.
    F_s0, kappa_s = 0.5, 1.0
    F_shear = np.maximum((F_index - F_s0) / (1.0 - F_s0), 0.0)
    Fx = np.maximum(-el_x,0)/eps_t0 + kappa_s*F_shear
    Fy = np.maximum(-el_y,0)/eps_t0 + kappa_s*F_shear
    Fz = np.maximum(-el_z,0)/eps_t0 + kappa_s*F_shear
    Dx = np.zeros_like(t); Dy = np.zeros_like(t); Dz = np.zeros_like(t)
    for i in range(1, len(t)):
        dti = t[i]-t[i-1]
        for Da, Fa in ((Dx,Fx),(Dy,Fy),(Dz,Fz)):
            ov = max(((1-Da[i-1])*Fa[i-1]-1)/F0, 0.0)
            Da[i] = min(1.0, Da[i-1] + ((1-Da[i-1])**alpha)/tau_D*(ov**m_over)*rate[i-1]*dti)
    E_Dx, E_Dy, E_Dz = E_GPa*(1-Dx), E_GPa*(1-Dy), E_GPa*(1-Dz)

    omd = np.maximum(1-D, 1e-6)
    ex = (sx-nu*(sy+sz))/(E_MPa*omd); ey = (sy-nu*(sx+sz))/(E_MPa*omd); ez = (sz-nu*(sx+sy))/(E_MPa*omd)
    power = sx*cdiff(ex,t)+sy*cdiff(ey,t)+sz*cdiff(ez,t)
    W_in = ctrap(power, t)
    W_el = 0.5*(sx*ex+sy*ey+sz*ez); W_el = W_el - W_el[0]
    W_diss = np.maximum(W_in - W_el, 0.0)
    return locals()


def main():
    d = compute(); t_us = d["t_us"]
    # 1. Stress + invariants
    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    ax.plot(t_us, d["sx"], label=r"$\sigma_x$ (dynamic)", color=PUB["blue"], lw=2.0)
    ax.plot(t_us, d["sy"], label=r"$\sigma_y$ (confined)", color=PUB["green"], lw=2.0)
    ax.plot(t_us, d["sz"], label=r"$\sigma_z$ (confined)", color=PUB["purple"], lw=2.0)
    ax.plot(t_us, d["p"], label=r"$p$", color=PUB["orange"], ls="--", lw=1.8)
    ax.plot(t_us, d["q"], label=r"$q$", color=PUB["black"], ls=":", lw=2.0)
    ax.set_xlabel(r"Time, $t$ ($\mu$s)"); ax.set_ylabel(r"Stress (MPa)"); ax.legend(ncol=3, loc="upper right")
    save(fig, ax, "trihb_stress_invariants.png", "Tri-HB stress and invariant history")
    # 2. p-q trajectory, annotated with the key loading stages
    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    p_arr, q_arr = d["p"], d["q"]
    sc = ax.scatter(p_arr, q_arr, c=t_us, s=7, cmap="viridis")
    ipk = int(np.argmax(q_arr))
    # initial (static pre-stress) state
    ax.scatter([p_arr[0]], [q_arr[0]], s=85, facecolor="white", edgecolor=PUB["black"], lw=1.6, zorder=6)
    ax.annotate("initial state\n(static pre-stress)", (p_arr[0], q_arr[0]),
                xytext=(p_arr[0] + 22, q_arr[0] + 70), fontsize=11.5, ha="left",
                arrowprops=dict(arrowstyle="->", color=PUB["black"], lw=1.1))
    # peak deviatoric stress
    ax.scatter([p_arr[ipk]], [q_arr[ipk]], marker="*", s=230, color=PUB["vermillion"],
               edgecolor=PUB["black"], lw=0.8, zorder=6)
    ax.annotate(r"peak ($q_{\max}$)", (p_arr[ipk], q_arr[ipk]),
                xytext=(p_arr[ipk] - 70, q_arr[ipk] + 6), fontsize=12, ha="right",
                arrowprops=dict(arrowstyle="->", color=PUB["black"], lw=1.1))
    # loading vs unloading branches (the gap is the axial-lateral timing lag)
    ax.annotate("loading\n(axial-led)", xy=(62, 168), xytext=(20, 250), fontsize=11.5,
                color="#3b2f7a", arrowprops=dict(arrowstyle="->", color="#3b2f7a", lw=1.0))
    ax.annotate("unloading\n(lateral-output lag)", xy=(96, 86), xytext=(108, 22), fontsize=11.5,
                color="#11705f", arrowprops=dict(arrowstyle="->", color="#11705f", lw=1.0))
    ax.set_xlabel(r"Mean pressure, $p$ (MPa)"); ax.set_ylabel(r"Deviatoric stress, $q$ (MPa)")
    cb = fig.colorbar(sc, ax=ax); cb.set_label(r"Time, $t$ ($\mu$s)", fontsize=12.5); cb.ax.tick_params(labelsize=11)
    save(fig, ax, "trihb_pq_trajectory.png", r"Tri-HB $p$--$q$ stress trajectory")
    # 3. Energy
    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    ax.plot(t_us, d["W_in"], label=r"$W_{\mathrm{input}}$", color=PUB["blue"], lw=2.0)
    ax.plot(t_us, d["W_el"], label=r"$W_{\mathrm{el}}$ (recoverable)", color=PUB["green"], lw=2.0)
    ax.plot(t_us, d["W_diss"], label=r"$W_{\mathrm{diss}}=W_{\mathrm{input}}-W_{\mathrm{el}}$", color=PUB["vermillion"], lw=2.0)
    ax.set_xlabel(r"Time, $t$ ($\mu$s)"); ax.set_ylabel(r"Energy density (MJ m$^{-3}$)"); ax.legend(loc="upper right")
    save(fig, ax, "trihb_energy.png", "Tri-HB energy balance (first law closes)")
    # 4. Orthotropic damage
    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    ax.plot(t_us, d["Dx"], label=r"$D_x$ (axial)", color=PUB["vermillion"], lw=2.1)
    ax.plot(t_us, d["Dy"], label=r"$D_y$ (lateral)", color=PUB["green"], lw=2.1)
    ax.plot(t_us, d["Dz"], label=r"$D_z$ (lateral)", color=PUB["orange"], lw=2.1)
    ax.plot(t_us, np.maximum.reduce([d["Dx"],d["Dy"],d["Dz"]]), label=r"$D=\max_i D_i$", color=PUB["blue"], ls=":", lw=1.9)
    ax.set_xlabel(r"Time, $t$ ($\mu$s)"); ax.set_ylabel(r"Damage-tensor component, $D_i$"); ax.legend(loc="upper left", ncol=2)
    save(fig, ax, "trihb_orthotropic_damage.png", "Tri-HB orthotropic damage tensor")
    # 5. Directional stiffness
    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    ax.plot(t_us, d["E_Dx"], label=r"$E_x$", color=PUB["vermillion"], lw=2.1)
    ax.plot(t_us, d["E_Dy"], label=r"$E_y$", color=PUB["green"], lw=2.1)
    ax.plot(t_us, d["E_Dz"], label=r"$E_z$", color=PUB["orange"], lw=2.1)
    ax.set_xlabel(r"Time, $t$ ($\mu$s)"); ax.set_ylabel(r"Directional modulus, $E_i$ (GPa)"); ax.legend(loc="lower left")
    save(fig, ax, "trihb_directional_stiffness.png", r"Tri-HB directional stiffness, $E_i=E_0(1-D_i)$")
    print(f"Wrote Tri-HB (Mode 3) paper figures to {OUT}")
    print(f"  peak q={np.max(d['q']):.0f} MPa  peakF={np.max(d['F_index']):.2f}  "
          f"peak Dx/Dy/Dz={np.max(d['Dx']):.3f}/{np.max(d['Dy']):.3f}/{np.max(d['Dz']):.3f}")


if __name__ == "__main__":
    main()
