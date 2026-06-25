from __future__ import annotations

"""Handbook Step 2-4 figures, matched to the CURRENT wave_damage.py physics
(true-triaxial Lode envelope, optional rate-dependent envelope, corrected
energy balance W_input = W_el + W_diss, and the Stage-1 orthotropic damage
tensor). Regenerate with:  py -3.13 scripts/export_handbook_step_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Handbook" / "figures" / "handbook_steps"
PUB_COLORS = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "purple": "#7B3FE4",
    "sky": "#56B4E9",
    "magenta": "#CC79A7",
    "black": "#222222",
    "gray": "#6B7280",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Cambria", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.prop_cycle": plt.cycler(color=[
        PUB_COLORS["blue"],
        PUB_COLORS["vermillion"],
        PUB_COLORS["green"],
        PUB_COLORS["purple"],
        PUB_COLORS["orange"],
        PUB_COLORS["sky"],
    ]),
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": PUB_COLORS["black"],
    "axes.labelcolor": PUB_COLORS["black"],
    "axes.titlecolor": PUB_COLORS["black"],
    "xtick.color": PUB_COLORS["black"],
    "ytick.color": PUB_COLORS["black"],
    "grid.color": "#D9DEE7",
    "grid.linewidth": 0.55,
    "grid.alpha": 1.0,
    "legend.frameon": False,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
})


def half_sine_window(t, td):
    tau = np.asarray(t)
    g = np.zeros_like(tau, dtype=float)
    mask = (tau >= 0.0) & (tau <= td)
    g[mask] = np.sin(np.pi * tau[mask] / td)
    return g


def hann_window(t, td):
    tau = np.asarray(t)
    g = np.zeros_like(tau, dtype=float)
    mask = (tau >= 0.0) & (tau <= td)
    g[mask] = 0.5 * (1.0 - np.cos(2.0 * np.pi * tau[mask] / td))
    return g


def rectangular_window(t, td):
    tau = np.asarray(t)
    g = np.zeros_like(tau, dtype=float)
    mask = (tau >= 0.0) & (tau <= td)
    g[mask] = 1.0
    return g


def get_window(t, td, mode):
    if mode == "Hann":
        return hann_window(t, td)
    if mode == "Half-sine":
        return half_sine_window(t, td)
    return rectangular_window(t, td)


def pulse(t, A, td, delay, mode):
    return A * get_window(t - delay, td, mode)


def central_difference(y, t):
    if len(y) < 2:
        return np.zeros_like(y)
    return np.gradient(y, t)


def cumulative_trapezoid(y, t):
    out = np.zeros_like(y, dtype=float)
    if len(y) > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))
    return out


def trapz_safe(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 2 or x.size < 2:
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def invariants_from_diagonal(sx, sy, sz):
    p = (sx + sy + sz) / 3.0
    q = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2))
    s1, s2, s3 = sx - p, sy - p, sz - p
    J2 = (s1**2 + s2**2 + s3**2) / 2.0
    J3 = s1 * s2 * s3
    theta = np.zeros_like(p)
    mask = J2 > 1e-12
    arg = np.zeros_like(p)
    arg[mask] = (3.0 * np.sqrt(3.0) / 2.0) * J3[mask] / (J2[mask] ** 1.5)
    theta[mask] = (1.0 / 3.0) * np.arccos(np.clip(arg[mask], -1.0, 1.0))
    return p, q, np.rad2deg(theta), J2, J3


def derive_failure_surface(ucs_mpa, k_conf, dif=1.0):
    k = max(k_conf, 1.0 + 1e-6)
    ucs_eff = max(ucs_mpa, 1e-6) * max(dif, 1e-6)
    return 3.0 * ucs_eff / (k + 2.0), 3.0 * (k - 1.0) / (k + 2.0), 1.0


def lode_shape(theta_deg, mode, param):
    """Deviatoric strength multiplier h(theta) (matches wave_damage.lode_shape).
    theta in [0,60]: 0 = compression meridian, 60 = extension meridian."""
    th = np.deg2rad(np.asarray(theta_deg, dtype=float))
    if mode == "lode":
        e = float(np.clip(param, 0.5, 1.0))
        return 1.0 - (1.0 - e) * (1.0 - np.cos(3.0 * th)) / 2.0
    if mode == "willam":
        e = float(np.clip(param, 0.5 + 1e-6, 1.0))
        a = np.pi / 3.0 - th
        ca = np.cos(a)
        rad = np.sqrt(np.maximum(4.0 * (1.0 - e ** 2) * ca ** 2 + 5.0 * e ** 2 - 4.0 * e, 0.0))
        num = 2.0 * (1.0 - e ** 2) * ca + (2.0 * e - 1.0) * rad
        den = 4.0 * (1.0 - e ** 2) * ca ** 2 + (2.0 * e - 1.0) ** 2
        return num / np.maximum(den, 1e-12)
    return 1.0 + param * (1.0 - np.cos(3.0 * th))


def compute_case():
    # Sandstone, asynchronous XYZ (inherits the Mode-5 handbook example).
    E_GPa, nu, rho, L_mm = 15.0, 0.20, 2650.0, 50.0
    sx0, sy0, sz0 = 30.0, 20.0, 15.0
    Ax = Ay = Az = 400.0
    td_us = 200.0
    delay_y_us, delay_z_us = 80.0, 160.0
    pulse_type = "Half-sine"
    tmax_us, npts = 520.0, 6000

    # Current default failure surface: derived from UCS + confinement ratio, with
    # the physically-correct extension-weakened Lode shape (rho_t/rho_c = 0.70).
    UCS_MPa, k_conf = 80.0, 4.0
    A_fail, B_fail, n_fail = derive_failure_surface(UCS_MPa, k_conf, 1.0)
    lode_mode, lode_ratio = "lode", 0.70
    sigma_t_MPa = UCS_MPa / 12.0
    tau_D_us, alpha_sat, m_over, beta_rate, epsdot0, F0 = 30.0, 1.0, 2.0, 0.15, 1.0, 1.0
    central_width = 0.30

    E = E_GPa * 1e9
    M = E * (1.0 - nu) / ((1.0 + nu) * max(1.0 - 2.0 * nu, 1e-9))
    cp0 = np.sqrt(M / rho)
    L_m = L_mm / 1000.0
    t_travel = L_m / cp0
    t_eq_low, t_eq_high = 3.0 * t_travel, 5.0 * t_travel

    t_us = np.linspace(0.0, tmax_us, npts)
    t = t_us * 1e-6
    td = td_us * 1e-6
    delay_y, delay_z = delay_y_us * 1e-6, delay_z_us * 1e-6

    x_left = pulse(t, Ax, td, 0.0, pulse_type)
    x_right = np.zeros_like(t)
    y_drive = pulse(t, Ay, td, delay_y, pulse_type)
    z_drive = pulse(t, Az, td, delay_z, pulse_type)
    sx = sx0 + x_left + x_right
    sy = sy0 + y_drive
    sz = sz0 + z_drive
    p, q, theta_deg, J2, J3 = invariants_from_diagonal(sx, sy, sz)

    centre_delay = L_m / (2.0 * cp0)
    g_left_centre = get_window(t - centre_delay, td, pulse_type)
    g_y_centre = get_window(t - delay_y - centre_delay, td, pulse_type)
    g_z_centre = get_window(t - delay_z - centre_delay, td, pulse_type)
    sigma_centre = sx0 + Ax * g_left_centre + Ay * g_y_centre + Az * g_z_centre
    eta_sup = max(1.0, (np.max(sigma_centre) - sx0) / max(Ax, 1e-9))
    dt_star = delay_y / t_travel

    h_theta = lode_shape(theta_deg, lode_mode, lode_ratio)
    qf = (A_fail + B_fail * np.maximum(p, 0.0) ** n_fail) * h_theta
    F_index = q / np.maximum(qf, 1e-9)

    E_MPa = E_GPa * 1000.0
    # Applied (undamaged) directional elastic strains (Poisson-coupled).
    el_x = (sx - nu * (sy + sz)) / E_MPa
    el_y = (sy - nu * (sx + sz)) / E_MPa
    el_z = (sz - nu * (sx + sy)) / E_MPa
    epsdot_eq = np.sqrt(2.0 / 3.0 * ((central_difference(el_x, t) - central_difference(el_y, t)) ** 2
                        + (central_difference(el_y, t) - central_difference(el_z, t)) ** 2
                        + (central_difference(el_z, t) - central_difference(el_x, t)) ** 2) / 2.0)
    rate_factor = (np.maximum(np.abs(epsdot_eq), 1e-12) / epsdot0) ** beta_rate

    tau_D = tau_D_us * 1e-6
    # --- Isotropic scalar damage with load-shedding feedback (F_eff = (1-D)F) ---
    D = np.zeros_like(t)
    Ddot = np.zeros_like(t)
    F_eff = np.zeros_like(t)
    for i in range(1, len(t)):
        F_eff[i - 1] = (1.0 - D[i - 1]) * F_index[i - 1]
        ov = max((F_eff[i - 1] - 1.0) / F0, 0.0)
        Ddot[i - 1] = ((1.0 - D[i - 1]) ** alpha_sat) / tau_D * (ov ** m_over) * rate_factor[i - 1]
        D[i] = np.clip(D[i - 1] + Ddot[i - 1] * (t[i] - t[i - 1]), 0.0, 1.0)
    if len(Ddot) > 1:
        Ddot[-1] = Ddot[-2]
    E_D = E_GPa * (1.0 - D)
    cp_D = cp0 * np.sqrt(np.maximum(1.0 - D, 0.0))

    # --- Orthotropic (diagonal) damage tensor, tension-driven per axis ---
    eps_t0 = max(sigma_t_MPa, 1e-6) / E_MPa
    Fx = np.maximum(-el_x, 0.0) / eps_t0
    Fy = np.maximum(-el_y, 0.0) / eps_t0
    Fz = np.maximum(-el_z, 0.0) / eps_t0
    Dx = np.zeros_like(t); Dy = np.zeros_like(t); Dz = np.zeros_like(t)
    for i in range(1, len(t)):
        dti = t[i] - t[i - 1]
        for Darr, Farr in ((Dx, Fx), (Dy, Fy), (Dz, Fz)):
            feff = (1.0 - Darr[i - 1]) * Farr[i - 1]
            ov = max((feff - 1.0) / F0, 0.0)
            Darr[i] = min(1.0, Darr[i - 1] + ((1.0 - Darr[i - 1]) ** alpha_sat) / tau_D
                          * (ov ** m_over) * rate_factor[i - 1] * dti)
    E_Dx = E_GPa * (1.0 - Dx); E_Dy = E_GPa * (1.0 - Dy); E_Dz = E_GPa * (1.0 - Dz)

    # --- Corrected energy balance (damage-coupled strain; closes by construction) ---
    one_minus_D = np.maximum(1.0 - D, 1e-6)
    eps_x_d = (sx - nu * (sy + sz)) / (E_MPa * one_minus_D)
    eps_y_d = (sy - nu * (sx + sz)) / (E_MPa * one_minus_D)
    eps_z_d = (sz - nu * (sx + sy)) / (E_MPa * one_minus_D)
    power = (sx * central_difference(eps_x_d, t)
             + sy * central_difference(eps_y_d, t)
             + sz * central_difference(eps_z_d, t))
    W_input = cumulative_trapezoid(power, t)
    W_el_abs = 0.5 * (sx * eps_x_d + sy * eps_y_d + sz * eps_z_d)
    W_el = W_el_abs - W_el_abs[0]
    W_diss = np.maximum(W_input - W_el, 0.0)

    x = np.linspace(0.0, 1.0, 400)
    max_D = float(np.max(D))
    left_damage = np.exp(-((x - 0.20) / 0.12) ** 2) * max_D
    right_damage = np.exp(-((x - 0.80) / 0.12) ** 2) * max_D
    central_damage = np.exp(-((x - 0.50) / 0.16) ** 2) * max_D * min(eta_sup, 2.0) / 2.0
    delay_scatter = (1.0 / (1.0 + np.exp(-(dt_star - 5.0)))) * max_D * 0.35
    damage_profile = np.clip(left_damage + right_damage + central_damage + delay_scatter, 0.0, 1.0)
    central_mask = np.abs(x - 0.5) < central_width / 2.0
    total_damage_area = trapz_safe(damage_profile, x)
    central_damage_area = trapz_safe(damage_profile[central_mask], x[central_mask])
    D_c = central_damage_area / max(total_damage_area, 1e-12)
    D_left = trapz_safe(damage_profile[x < 0.5], x[x < 0.5])
    D_right = trapz_safe(damage_profile[x >= 0.5], x[x >= 0.5])
    S_x = 1.0 - abs(D_left - D_right) / max(D_left + D_right, 1e-12)

    return locals()


def style_axes(ax, title=None):
    if title is not None:
        ax.set_title(title, fontsize=17, fontweight="normal", pad=8)
    right_label = ax.yaxis.get_label_position() == "right"
    ax.tick_params(axis="both", labelsize=13, direction="out", length=4.2, width=0.9)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.grid(True, which="major")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(right_label)
    ax.spines["left"].set_visible(not right_label)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["right"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_frame_on(False)
        for text in legend.get_texts():
            text.set_fontsize(12.5)


def save_panel(fig, ax, filename, title):
    style_axes(ax, title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_step2(data):
    t_us = data["t_us"]
    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    ax.plot(t_us, data["x_left"], label=r"X pulse", color=PUB_COLORS["blue"], linewidth=2.05)
    ax.plot(t_us, data["y_drive"], label=r"Y pulse", color=PUB_COLORS["green"], linewidth=2.05)
    ax.plot(t_us, data["z_drive"], label=r"Z pulse", color=PUB_COLORS["purple"], linewidth=2.05)
    ax.axvspan(data["t_eq_low"] * 1e6, data["t_eq_high"] * 1e6, color=PUB_COLORS["orange"], alpha=0.14, label=r"$3$--$5$ travel times")
    ax.axvline(data["delay_y"] * 1e6, color=PUB_COLORS["green"], linestyle=":", linewidth=1.5)
    ax.axvline(data["delay_z"] * 1e6, color=PUB_COLORS["purple"], linestyle=":", linewidth=1.5)
    ax.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax.set_ylabel(r"Stress increment, $\Delta\sigma$ (MPa)")
    ax.legend(loc="upper right")
    save_panel(fig, ax, "step2_input_dynamic_pulses.png", "Input pulse timing")

    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    ax.plot(t_us, data["sigma_centre"], color=PUB_COLORS["blue"], linewidth=2.1, label=r"$\sigma_{\mathrm{centre}}(t)$")
    ax.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax.set_ylabel(r"Centre stress indicator (MPa)")
    ax.legend(loc="upper right")
    save_panel(fig, ax, "step2_centre_superposition.png", r"Specimen-centre superposition, $\eta_{\mathrm{sup}}=%.2f$" % data["eta_sup"])

    dt_grid = np.linspace(0.0, 20.0, 400)
    wave_control = np.exp(-(dt_grid / 1.2) ** 2)
    damage_control = 1.0 / (1.0 + np.exp(-(dt_grid - 5.0) / 1.3))
    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    ax.plot(dt_grid, wave_control, label="Wave-interaction control", color=PUB_COLORS["blue"], linewidth=2.0)
    ax.plot(dt_grid, damage_control, label="Damage-memory control", color=PUB_COLORS["vermillion"], linewidth=2.0)
    ax.axvline(data["dt_star"], linestyle="--", color=PUB_COLORS["black"], linewidth=1.25, label=r"sample $\Delta t^*$")
    ax.axvspan(0, 1, alpha=0.09, color=PUB_COLORS["sky"])
    ax.axvspan(1, 3, alpha=0.09, color=PUB_COLORS["green"])
    ax.axvspan(3, 10, alpha=0.07, color=PUB_COLORS["orange"])
    ax.axvspan(10, 20, alpha=0.07, color=PUB_COLORS["vermillion"])
    ax.set_xlabel(r"Normalised delay, $\Delta t^*$")
    ax.set_ylabel("Control indicator")
    ax.legend(loc="center right")
    save_panel(fig, ax, "step2_delay_regime_map.png", "Delay-regime interaction")

    labels = ["travel time", "eq low", "eq high", "Y delay", "Z delay"]
    values = [data["t_travel"] * 1e6, data["t_eq_low"] * 1e6, data["t_eq_high"] * 1e6,
              data["delay_y"] * 1e6, data["delay_z"] * 1e6]
    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    ax.bar(labels, values, color=[PUB_COLORS["gray"], PUB_COLORS["orange"], PUB_COLORS["vermillion"], PUB_COLORS["green"], PUB_COLORS["purple"]])
    ax.set_ylabel(r"Time, $t$ ($\mu$s)")
    ax.tick_params(axis="x", rotation=20)
    save_panel(fig, ax, "step2_timing_scales.png", "Timing-scale hierarchy")


def plot_step3(data):
    t_us = data["t_us"]
    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    ax.plot(t_us, data["sx"], label=r"$\sigma_x$", color=PUB_COLORS["blue"], linewidth=2.0)
    ax.plot(t_us, data["sy"], label=r"$\sigma_y$", color=PUB_COLORS["green"], linewidth=2.0)
    ax.plot(t_us, data["sz"], label=r"$\sigma_z$", color=PUB_COLORS["purple"], linewidth=2.0)
    ax.plot(t_us, data["p"], label=r"$p$", color=PUB_COLORS["orange"], linestyle="--", linewidth=1.8)
    ax.plot(t_us, data["q"], label=r"$q$", color=PUB_COLORS["black"], linestyle=":", linewidth=2.0)
    ax.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax.set_ylabel(r"Stress (MPa)")
    ax.legend(ncol=3, loc="upper right")
    save_panel(fig, ax, "step3_total_stresses_invariants.png", "Stress-invariant history")

    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    sc = ax.scatter(data["p"], data["q"], c=t_us, s=7, cmap="viridis")
    ax.set_xlabel(r"Mean pressure, $p$ (MPa)")
    ax.set_ylabel(r"Deviatoric stress, $q$ (MPa)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"Time, $t$ ($\mu$s)", fontsize=13)
    cb.ax.tick_params(labelsize=11)
    save_panel(fig, ax, "step3_pq_trajectory.png", r"$p$--$q$ stress trajectory")

    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    ax.plot(t_us, data["q"], label=r"$q(t)$", color=PUB_COLORS["blue"], linewidth=2.05)
    ax.plot(t_us, data["qf"], label=r"$q_f(p,\theta)$", color=PUB_COLORS["vermillion"], linestyle="--", linewidth=1.85)
    ax.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax.set_ylabel(r"Stress (MPa)")
    ax.legend(loc="upper right")
    save_panel(fig, ax, "step3_failure_surface_comparison.png", "Failure-surface comparison")

    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    ax.plot(t_us, data["F_index"], label=r"Failure index, $F(t)$", color=PUB_COLORS["blue"], linewidth=2.15)
    ax.plot(t_us, data["epsdot_eq"] / 100.0, label=r"$\dot{\varepsilon}_{\mathrm{eq}}/100$", color=PUB_COLORS["green"], alpha=0.82, linewidth=1.9)
    ax.axhline(1.0, color=PUB_COLORS["vermillion"], linestyle="--", linewidth=1.6, label=r"$F=1$")
    ax.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax.set_ylabel(r"Index / scaled rate")
    ax.legend(loc="upper right")
    save_panel(fig, ax, "step3_failure_index_strain_rate.png", "Failure-index interaction")

    # NEW (V6): true-triaxial Lode deviatoric shapes
    th = np.linspace(0.0, 60.0, 200)
    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    ax.plot(th, lode_shape(th, "legacy", 0.10), label=r"legacy $1+a_\theta(1-\cos3\theta)$", color=PUB_COLORS["gray"], linestyle=":", linewidth=2.0)
    ax.plot(th, lode_shape(th, "lode", 0.70), label=r"Lode (default), $\rho_t/\rho_c=0.70$", color=PUB_COLORS["blue"], linewidth=2.2)
    ax.plot(th, lode_shape(th, "willam", 0.70), label=r"Willam--Warnke, $\rho_t/\rho_c=0.70$", color=PUB_COLORS["vermillion"], linestyle="--", linewidth=2.0)
    ax.axvline(0, color=PUB_COLORS["black"], linewidth=0.8, alpha=0.5)
    ax.text(1.5, 0.62, "compression\nmeridian", fontsize=10.5, color=PUB_COLORS["black"])
    ax.text(46, 0.62, "extension\nmeridian", fontsize=10.5, color=PUB_COLORS["black"])
    ax.set_xlabel(r"Lode angle, $\theta$ (deg)")
    ax.set_ylabel(r"Deviatoric multiplier, $h(\theta)$")
    ax.set_ylim(0.55, 1.25)
    ax.legend(loc="upper center")
    save_panel(fig, ax, "step3_lode_shapes.png", "True-triaxial deviatoric (Lode) shapes")

    # NEW (V6): rate-dependent envelope DIF on q_f
    edot = np.logspace(0, 4, 200)
    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    for b_env, col in ((0.03, PUB_COLORS["blue"]), (0.05, PUB_COLORS["vermillion"]), (0.08, PUB_COLORS["green"])):
        ax.plot(edot, 1.0 + b_env * np.log10(np.clip(edot, 1.0, None)), label=fr"$b_{{\mathrm{{env}}}}={b_env}$", linewidth=2.1, color=col)
    ax.set_xscale("log")
    ax.set_xlabel(r"Equivalent strain rate, $\dot{\varepsilon}_{\mathrm{eq}}/\dot{\varepsilon}_0$")
    ax.set_ylabel(r"Envelope DIF on $q_f$")
    ax.legend(loc="upper left")
    save_panel(fig, ax, "step3_rate_dependent_envelope.png", r"Rate-dependent envelope, $\mathrm{DIF}=1+b\log_{10}(\dot\varepsilon/\dot\varepsilon_0)$")


def plot_step4_damage(data):
    t_us = data["t_us"]
    Ddot_norm = data["Ddot"] / max(float(np.max(data["Ddot"])), 1e-12)

    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    ax.plot(t_us, data["F_index"], color=PUB_COLORS["blue"], linewidth=2.15, label=r"Applied $F=q/q_f$")
    ax.plot(t_us, data["F_eff"], color=PUB_COLORS["green"], linestyle="-.", linewidth=1.9, label=r"Effective $(1-D)F$")
    ax.axhline(1.0, color=PUB_COLORS["vermillion"], linestyle="--", linewidth=1.6, label=r"$F=1$")
    ax.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax.set_ylabel(r"$F=q/q_f$")
    ax.legend(loc="upper right")
    save_panel(fig, ax, "step4_damage_trigger.png", "Failure-envelope interaction (load-shedding)")

    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    ax.plot(t_us, data["D"], label=r"$D(t)$", color=PUB_COLORS["blue"], linewidth=2.15)
    ax.plot(t_us, Ddot_norm, label=r"normalised $\dot{D}$", color=PUB_COLORS["vermillion"], linestyle="--", linewidth=1.8)
    ax.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax.set_ylabel(r"Damage variable")
    ax.legend(loc="upper right")
    save_panel(fig, ax, "step4_cumulative_damage.png", r"Damage accumulation (isotropic)")

    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    ax.plot(t_us, data["E_D"], color=PUB_COLORS["green"], linewidth=2.1, label=r"$E(D)$")
    ax2 = ax.twinx()
    ax2.plot(t_us, data["cp_D"], color=PUB_COLORS["purple"], linestyle="--", linewidth=1.9, label=r"$c_p(D)$")
    ax.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax.set_ylabel(r"$E(D)$ (GPa)")
    ax2.set_ylabel(r"$c_p(D)$ (m/s)")
    style_axes(ax, "Stiffness and wave-speed degradation")
    style_axes(ax2)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "step4_stiffness_wave_speed.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    ax.plot(data["x"], data["damage_profile"], color=PUB_COLORS["blue"], linewidth=2.15, label=r"$D(x)$")
    ax.axvspan(0.35, 0.65, color=PUB_COLORS["orange"], alpha=0.14, label="central region")
    ax.set_xlabel(r"Normalised specimen position, $x/L_s$")
    ax.set_ylabel(r"Damage profile, $D(x)$")
    ax.legend(loc="upper right")
    save_panel(fig, ax, "step4_damage_profile.png", r"Damage-profile descriptors, $D_c=%.2f$, $S_x=%.2f$" % (data["D_c"], data["S_x"]))

    # NEW (V6): orthotropic damage tensor components
    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    ax.plot(t_us, data["Dx"], label=r"$D_x$", color=PUB_COLORS["vermillion"], linewidth=2.1)
    ax.plot(t_us, data["Dy"], label=r"$D_y$", color=PUB_COLORS["green"], linewidth=2.1)
    ax.plot(t_us, data["Dz"], label=r"$D_z$", color=PUB_COLORS["orange"], linewidth=2.1)
    ax.plot(t_us, np.maximum.reduce([data["Dx"], data["Dy"], data["Dz"]]), label=r"$D=\max_i D_i$", color=PUB_COLORS["blue"], linestyle=":", linewidth=1.9)
    ax.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax.set_ylabel(r"Damage-tensor component, $D_i$")
    ax.legend(loc="upper left", ncol=2)
    save_panel(fig, ax, "step4_orthotropic_damage.png", "Orthotropic damage tensor (tension-driven)")

    # NEW (V6): directional stiffness degradation
    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    ax.plot(t_us, data["E_Dx"], label=r"$E_x$", color=PUB_COLORS["vermillion"], linewidth=2.1)
    ax.plot(t_us, data["E_Dy"], label=r"$E_y$", color=PUB_COLORS["green"], linewidth=2.1)
    ax.plot(t_us, data["E_Dz"], label=r"$E_z$", color=PUB_COLORS["orange"], linewidth=2.1)
    ax.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax.set_ylabel(r"Directional modulus, $E_i$ (GPa)")
    ax.legend(loc="lower left")
    save_panel(fig, ax, "step4_directional_stiffness.png", r"Directional stiffness, $E_i=E_0(1-D_i)$")


def plot_step4_energy(data):
    t_us = data["t_us"]
    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    ax.plot(t_us, data["W_input"], label=r"$W_{\mathrm{input}}$", color=PUB_COLORS["blue"], linewidth=2.05)
    ax.plot(t_us, data["W_el"], label=r"$W_{\mathrm{el}}$ (recoverable)", color=PUB_COLORS["green"], linewidth=2.0)
    ax.plot(t_us, data["W_diss"], label=r"$W_{\mathrm{diss}}=W_{\mathrm{input}}-W_{\mathrm{el}}$", color=PUB_COLORS["vermillion"], linewidth=2.0)
    ax.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax.set_ylabel(r"Energy density (MJ m$^{-3}$)")
    ax.legend(loc="upper right")
    save_panel(fig, ax, "step4_energy_density_histories.png", "Energy balance (first law closes)")

    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    ax.plot(t_us, data["power"], color=PUB_COLORS["purple"], linewidth=2.15)
    ax.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax.set_ylabel(r"Power density (MJ m$^{-3}$ s$^{-1}$)")
    save_panel(fig, ax, "step4_power_density.png", "Instantaneous power density")

    labels = ["Final D", "Dc", "Sx", "eta"]
    values = [data["D"][-1], data["D_c"], data["S_x"], min(data["eta_sup"] / 3.0, 1.0)]
    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    ax.bar(labels, values, color=[PUB_COLORS["blue"], PUB_COLORS["orange"], PUB_COLORS["green"], PUB_COLORS["purple"]])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Normalised value")
    save_panel(fig, ax, "step4_validation_descriptors.png", "Validation descriptors")

    dt_grid = np.linspace(0.0, 20.0, 400)
    wave_control = np.exp(-(dt_grid / 1.2) ** 2)
    damage_control = 1.0 / (1.0 + np.exp(-(dt_grid - 5.0) / 1.3))
    I_final = np.clip(0.15 + 0.35 * wave_control + 0.25 * damage_control, 0.0, 1.0)
    I_central = np.clip(0.75 * wave_control + 0.25 * (1.0 - damage_control), 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    ax.plot(dt_grid, I_final, label="Final damage trend", color=PUB_COLORS["vermillion"], linewidth=2.05)
    ax.plot(dt_grid, I_central, label=r"Central fraction, $D_c$", color=PUB_COLORS["blue"], linewidth=2.05)
    ax.axvline(data["dt_star"], color=PUB_COLORS["black"], linestyle="--", linewidth=1.25, label=r"sample $\Delta t^*$")
    ax.set_xlabel(r"Normalised delay, $\Delta t^*$")
    ax.set_ylabel("Indicator")
    ax.legend(loc="upper right")
    save_panel(fig, ax, "step4_delay_sensitivity_indicators.png", "Delay-sensitivity indicators")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = compute_case()
    plot_step2(data)
    plot_step3(data)
    plot_step4_damage(data)
    plot_step4_energy(data)
    print(f"Wrote Step 2-4 handbook figures (V6 physics) to {OUT_DIR}")
    print(f"  peak F={np.max(data['F_index']):.2f}  peak D={np.max(data['D']):.3f}  "
          f"peak Dx/Dy/Dz={np.max(data['Dx']):.3f}/{np.max(data['Dy']):.3f}/{np.max(data['Dz']):.3f}")


if __name__ == "__main__":
    main()
