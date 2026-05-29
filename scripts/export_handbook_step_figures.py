from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures" / "handbook_steps"


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


def compute_case():
    # One deliberately nontrivial sample: sandstone, asynchronous XYZ, inherited
    # from the Step 1 Mode 5 example used in the handbook figures.
    E_GPa = 15.0
    nu = 0.20
    rho = 2650.0
    L_mm = 50.0
    sx0, sy0, sz0 = 30.0, 20.0, 15.0
    Ax = Ay = Az = 400.0
    td_us = 200.0
    delay_y_us = 80.0
    delay_z_us = 160.0
    pulse_type = "Half-sine"
    tmax_us = 520.0
    npts = 6000

    A_fail = 15.0
    B_fail = 1.3
    n_fail = 0.75
    lode_amp = 0.10
    tau_D_us = 30.0
    alpha_sat = 1.0
    m_over = 2.0
    beta_rate = 0.15
    epsdot0 = 1.0
    F0 = 1.0
    central_width = 0.30

    E = E_GPa * 1e9
    G = E / (2.0 * (1.0 + nu))
    M = E * (1.0 - nu) / ((1.0 + nu) * max(1.0 - 2.0 * nu, 1e-9))
    cp0 = np.sqrt(M / rho)
    cs = np.sqrt(G / rho)
    L_m = L_mm / 1000.0
    t_travel = L_m / cp0
    t_eq_low = 3.0 * t_travel
    t_eq_high = 5.0 * t_travel

    t_us = np.linspace(0.0, tmax_us, npts)
    t = t_us * 1e-6
    td = td_us * 1e-6
    delay_y = delay_y_us * 1e-6
    delay_z = delay_z_us * 1e-6

    x_left = pulse(t, Ax, td, 0.0, pulse_type)
    x_right = np.zeros_like(t)
    y_drive = pulse(t, Ay, td, delay_y, pulse_type)
    z_drive = pulse(t, Az, td, delay_z, pulse_type)

    sx_dyn = x_left + x_right
    sx = sx0 + sx_dyn
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

    theta_rad = np.deg2rad(theta_deg)
    h_theta = 1.0 + lode_amp * (1.0 - np.cos(3.0 * theta_rad))
    qf = (A_fail + B_fail * np.maximum(p, 0.0) ** n_fail) * h_theta
    F_index = q / np.maximum(qf, 1e-9)

    E_MPa = E_GPa * 1000.0
    eps_x = sx / E_MPa
    eps_y = sy / E_MPa
    eps_z = sz / E_MPa
    epsdot_x = central_difference(eps_x, t)
    epsdot_y = central_difference(eps_y, t)
    epsdot_z = central_difference(eps_z, t)
    epsdot_eq = np.sqrt(
        2.0
        / 3.0
        * ((epsdot_x - epsdot_y) ** 2 + (epsdot_y - epsdot_z) ** 2 + (epsdot_z - epsdot_x) ** 2)
        / 2.0
    )
    rate_factor = (np.maximum(np.abs(epsdot_eq), 1e-12) / epsdot0) ** beta_rate

    tau_D = tau_D_us * 1e-6
    D = np.zeros_like(t)
    Ddot = np.zeros_like(t)
    for i in range(1, len(t)):
        overstress = max((F_index[i - 1] - 1.0) / F0, 0.0)
        Ddot[i - 1] = ((1.0 - D[i - 1]) ** alpha_sat) / tau_D * (overstress**m_over) * rate_factor[i - 1]
        D[i] = np.clip(D[i - 1] + Ddot[i - 1] * (t[i] - t[i - 1]), 0.0, 1.0)
    if len(Ddot) > 1:
        Ddot[-1] = Ddot[-2]

    E_D = E_GPa * (1.0 - D)
    cp_D = cp0 * np.sqrt(np.maximum(1.0 - D, 0.0))

    W_el = (1.0 / (2.0 * E_MPa)) * (
        sx**2 + sy**2 + sz**2 - 2.0 * nu * (sx * sy + sy * sz + sz * sx)
    )
    power = sx * epsdot_x + sy * epsdot_y + sz * epsdot_z
    W_input = cumulative_trapezoid(power, t)
    W_diss = W_input - W_el + W_el[0]

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


def style_axes(ax):
    ax.grid(True, alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_step2(data):
    t_us = data["t_us"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), dpi=220)
    fig.suptitle("Step 2: wave timing and interaction regime", fontsize=14, fontweight="bold")

    axes[0, 0].plot(t_us, data["x_left"], label="X pulse", color="#0f6fb6", linewidth=1.9)
    axes[0, 0].plot(t_us, data["y_drive"], label="Y pulse", color="#059669", linewidth=1.7)
    axes[0, 0].plot(t_us, data["z_drive"], label="Z pulse", color="#7c3aed", linewidth=1.7)
    axes[0, 0].axvspan(data["t_eq_low"] * 1e6, data["t_eq_high"] * 1e6, color="#f59e0b", alpha=0.15, label="3-5 travel times")
    axes[0, 0].axvline(data["delay_y"] * 1e6, color="#059669", linestyle=":", linewidth=1.2)
    axes[0, 0].axvline(data["delay_z"] * 1e6, color="#7c3aed", linestyle=":", linewidth=1.2)
    axes[0, 0].set_title("Input dynamic pulses")
    axes[0, 0].set_xlabel("Time (us)")
    axes[0, 0].set_ylabel("Stress increment (MPa)")
    axes[0, 0].legend(fontsize=7, frameon=False)
    style_axes(axes[0, 0])

    axes[0, 1].plot(t_us, data["sigma_centre"], color="#334155", linewidth=1.9)
    axes[0, 1].set_title(f"Centre superposition factor eta = {data['eta_sup']:.2f}")
    axes[0, 1].set_xlabel("Time (us)")
    axes[0, 1].set_ylabel("Centre stress indicator (MPa)")
    style_axes(axes[0, 1])

    dt_grid = np.linspace(0.0, 20.0, 400)
    wave_control = np.exp(-(dt_grid / 1.2) ** 2)
    damage_control = 1.0 / (1.0 + np.exp(-(dt_grid - 5.0) / 1.3))
    axes[1, 0].plot(dt_grid, wave_control, label="Wave-interaction control", color="#0f6fb6")
    axes[1, 0].plot(dt_grid, damage_control, label="Damage-memory control", color="#dc2626")
    axes[1, 0].axvline(data["dt_star"], linestyle="--", color="#111827", label="sample dt*")
    axes[1, 0].axvspan(0, 1, alpha=0.10, color="#60a5fa")
    axes[1, 0].axvspan(1, 3, alpha=0.10, color="#34d399")
    axes[1, 0].axvspan(3, 10, alpha=0.08, color="#f59e0b")
    axes[1, 0].axvspan(10, 20, alpha=0.08, color="#f87171")
    axes[1, 0].set_title("Normalised delay regime map")
    axes[1, 0].set_xlabel("dt* = delay / travel time")
    axes[1, 0].set_ylabel("Control indicator")
    axes[1, 0].legend(fontsize=7, frameon=False)
    style_axes(axes[1, 0])

    labels = ["travel time", "eq low", "eq high", "Y delay", "Z delay"]
    values = [
        data["t_travel"] * 1e6,
        data["t_eq_low"] * 1e6,
        data["t_eq_high"] * 1e6,
        data["delay_y"] * 1e6,
        data["delay_z"] * 1e6,
    ]
    axes[1, 1].bar(labels, values, color=["#64748b", "#fbbf24", "#f59e0b", "#059669", "#7c3aed"])
    axes[1, 1].set_title("Timing scales")
    axes[1, 1].set_ylabel("Time (us)")
    axes[1, 1].tick_params(axis="x", rotation=20)
    style_axes(axes[1, 1])

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "step2_wave_timing_regime.png", bbox_inches="tight")
    plt.close(fig)


def plot_step3(data):
    t_us = data["t_us"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), dpi=220)
    fig.suptitle("Step 3: stress path and failure index", fontsize=14, fontweight="bold")

    axes[0, 0].plot(t_us, data["sx"], label="sigma_x", color="#0f6fb6")
    axes[0, 0].plot(t_us, data["sy"], label="sigma_y", color="#059669")
    axes[0, 0].plot(t_us, data["sz"], label="sigma_z", color="#7c3aed")
    axes[0, 0].plot(t_us, data["p"], label="p", color="#f59e0b", linestyle="--")
    axes[0, 0].plot(t_us, data["q"], label="q", color="#334155", linestyle=":")
    axes[0, 0].set_title("Total stresses and invariants")
    axes[0, 0].set_xlabel("Time (us)")
    axes[0, 0].set_ylabel("MPa")
    axes[0, 0].legend(fontsize=7, ncol=3, frameon=False)
    style_axes(axes[0, 0])

    sc = axes[0, 1].scatter(data["p"], data["q"], c=t_us, s=7, cmap="viridis")
    axes[0, 1].set_title("p-q trajectory")
    axes[0, 1].set_xlabel("Mean pressure p (MPa)")
    axes[0, 1].set_ylabel("Deviatoric stress q (MPa)")
    style_axes(axes[0, 1])
    cb = fig.colorbar(sc, ax=axes[0, 1])
    cb.set_label("Time (us)")

    axes[1, 0].plot(t_us, data["q"], label="q", color="#334155")
    axes[1, 0].plot(t_us, data["qf"], label="qf", color="#dc2626", linestyle="--")
    axes[1, 0].set_title("Failure surface comparison")
    axes[1, 0].set_xlabel("Time (us)")
    axes[1, 0].set_ylabel("MPa")
    axes[1, 0].legend(fontsize=7, frameon=False)
    style_axes(axes[1, 0])

    axes[1, 1].plot(t_us, data["F_index"], label="F=q/qf", color="#dc2626")
    axes[1, 1].plot(t_us, data["epsdot_eq"] / 100.0, label="eq. strain rate / 100", color="#0f6fb6", alpha=0.75)
    axes[1, 1].axhline(1.0, color="#111827", linestyle=":", linewidth=1.2)
    axes[1, 1].set_title("Failure index and equivalent strain rate")
    axes[1, 1].set_xlabel("Time (us)")
    axes[1, 1].set_ylabel("Index / scaled rate")
    axes[1, 1].legend(fontsize=7, frameon=False)
    style_axes(axes[1, 1])

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "step3_stress_path_failure.png", bbox_inches="tight")
    plt.close(fig)


def plot_step4_damage(data):
    t_us = data["t_us"]
    Ddot_norm = data["Ddot"] / max(float(np.max(data["Ddot"])), 1e-12)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), dpi=220)
    fig.suptitle("Step 4: damage evolution and stiffness degradation", fontsize=14, fontweight="bold")

    axes[0, 0].plot(t_us, data["F_index"], color="#dc2626", label="F(t)")
    axes[0, 0].axhline(1.0, color="#111827", linestyle=":", label="damage threshold")
    axes[0, 0].set_title("Damage trigger")
    axes[0, 0].set_xlabel("Time (us)")
    axes[0, 0].set_ylabel("Failure index")
    axes[0, 0].legend(fontsize=7, frameon=False)
    style_axes(axes[0, 0])

    axes[0, 1].plot(t_us, data["D"], label="D(t)", color="#0f6fb6")
    axes[0, 1].plot(t_us, Ddot_norm, label="normalised Ddot", color="#f59e0b", linestyle="--")
    axes[0, 1].set_title(f"Cumulative damage, final D={data['D'][-1]:.2f}")
    axes[0, 1].set_xlabel("Time (us)")
    axes[0, 1].set_ylabel("Damage variable")
    axes[0, 1].legend(fontsize=7, frameon=False)
    style_axes(axes[0, 1])

    axes[1, 0].plot(t_us, data["E_D"], color="#059669", label="E(D)")
    ax2 = axes[1, 0].twinx()
    ax2.plot(t_us, data["cp_D"], color="#7c3aed", linestyle="--", label="cp(D)")
    axes[1, 0].set_title("Stiffness and P-wave speed loss")
    axes[1, 0].set_xlabel("Time (us)")
    axes[1, 0].set_ylabel("E(D) (GPa)")
    ax2.set_ylabel("cp(D) (m/s)")
    style_axes(axes[1, 0])

    axes[1, 1].plot(data["x"], data["damage_profile"], color="#334155", linewidth=2)
    axes[1, 1].axvspan(0.35, 0.65, color="#f59e0b", alpha=0.15, label="central region")
    axes[1, 1].set_title(f"Damage profile: Dc={data['D_c']:.2f}, Sx={data['S_x']:.2f}")
    axes[1, 1].set_xlabel("Normalised specimen position")
    axes[1, 1].set_ylabel("D(x)")
    axes[1, 1].legend(fontsize=7, frameon=False)
    style_axes(axes[1, 1])

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "step4_damage_evolution.png", bbox_inches="tight")
    plt.close(fig)


def plot_step4_energy(data):
    t_us = data["t_us"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), dpi=220)
    fig.suptitle("Step 4: energy-density and validation indicators", fontsize=14, fontweight="bold")

    axes[0, 0].plot(t_us, data["W_input"], label="W_input", color="#0f6fb6")
    axes[0, 0].plot(t_us, data["W_el"], label="W_el", color="#059669")
    axes[0, 0].plot(t_us, data["W_diss"], label="W_diss estimate", color="#dc2626")
    axes[0, 0].set_title("Energy density histories")
    axes[0, 0].set_xlabel("Time (us)")
    axes[0, 0].set_ylabel("MJ/m^3")
    axes[0, 0].legend(fontsize=7, frameon=False)
    style_axes(axes[0, 0])

    axes[0, 1].plot(t_us, data["power"], color="#7c3aed")
    axes[0, 1].set_title("Instantaneous power density")
    axes[0, 1].set_xlabel("Time (us)")
    axes[0, 1].set_ylabel("MJ m^-3 s^-1")
    style_axes(axes[0, 1])

    labels = ["Final D", "Dc", "Sx", "eta"]
    values = [data["D"][-1], data["D_c"], data["S_x"], min(data["eta_sup"] / 3.0, 1.0)]
    axes[1, 0].bar(labels, values, color=["#0f6fb6", "#f59e0b", "#059669", "#7c3aed"])
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_title("Validation descriptors")
    axes[1, 0].set_ylabel("Normalised value")
    style_axes(axes[1, 0])

    dt_grid = np.linspace(0.0, 20.0, 400)
    wave_control = np.exp(-(dt_grid / 1.2) ** 2)
    damage_control = 1.0 / (1.0 + np.exp(-(dt_grid - 5.0) / 1.3))
    I_final = np.clip(0.15 + 0.35 * wave_control + 0.25 * damage_control, 0.0, 1.0)
    I_central = np.clip(0.75 * wave_control + 0.25 * (1.0 - damage_control), 0.0, 1.0)
    axes[1, 1].plot(dt_grid, I_final, label="final damage trend", color="#dc2626")
    axes[1, 1].plot(dt_grid, I_central, label="central damage trend", color="#0f6fb6")
    axes[1, 1].axvline(data["dt_star"], color="#111827", linestyle="--", label="sample dt*")
    axes[1, 1].set_title("Delay-sensitivity indicators")
    axes[1, 1].set_xlabel("dt*")
    axes[1, 1].set_ylabel("Indicator")
    axes[1, 1].legend(fontsize=7, frameon=False)
    style_axes(axes[1, 1])

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "step4_energy_density_validation.png", bbox_inches="tight")
    plt.close(fig)


def write_summary(data):
    rows = {
        "sample": "Sandstone asynchronous XYZ, Ax=Ay=Az=400 MPa, td=200 us, delays 80/160 us",
        "t_travel_us": data["t_travel"] * 1e6,
        "dt_star_y": data["dt_star"],
        "eta_sup": data["eta_sup"],
        "peak_p_MPa": float(np.max(data["p"])),
        "peak_q_MPa": float(np.max(data["q"])),
        "peak_failure_index": float(np.max(data["F_index"])),
        "final_D": float(data["D"][-1]),
        "central_damage_fraction": data["D_c"],
        "symmetry_index": data["S_x"],
        "final_W_input_MJ_m3": float(data["W_input"][-1]),
        "final_W_elastic_MJ_m3": float(data["W_el"][-1]),
        "final_W_diss_MJ_m3": float(data["W_diss"][-1]),
    }
    (OUT_DIR / "step_analysis_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    pd.DataFrame([rows]).to_csv(OUT_DIR / "step_analysis_summary.csv", index=False)


def write_latex_snippets():
    snippets = r"""% Handbook Step 2-4 figure snippets generated by scripts/export_handbook_step_figures.py

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{figures/handbook_steps/step2_wave_timing_regime.png}
  \caption{Step 2 sample wave model for the asynchronous XYZ case: input pulse timing, specimen-centre superposition, normalised delay regime map, and the travel-time/equilibrium scales used by the Streamlit workflow.}
  \label{fig:step2-wave-timing-regime}
\end{figure}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{figures/handbook_steps/step3_stress_path_failure.png}
  \caption{Step 3 sample stress-path analysis: total principal stresses, $p$--$q$ trajectory, comparison of $q$ with the pressure- and Lode-angle-dependent failure surface $q_f$, and the resulting failure index $F=q/q_f$.}
  \label{fig:step3-stress-path-failure}
\end{figure}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{figures/handbook_steps/step4_damage_evolution.png}
  \caption{Step 4 sample damage analysis: failure-index trigger, cumulative damage and damage rate, stiffness and P-wave-speed degradation, and the synthetic damage-profile descriptors used for DEM/experimental comparison.}
  \label{fig:step4-damage-evolution}
\end{figure}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{figures/handbook_steps/step4_energy_density_validation.png}
  \caption{Step 4 sample energy-density and validation view: input, recoverable elastic, and dissipated energy-density estimates, instantaneous power density, scalar validation descriptors, and delay-sensitivity guide curves.}
  \label{fig:step4-energy-density-validation}
\end{figure}
"""
    (OUT_DIR / "latex_include_snippets.tex").write_text(snippets, encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = compute_case()
    plot_step2(data)
    plot_step3(data)
    plot_step4_damage(data)
    plot_step4_energy(data)
    write_summary(data)
    write_latex_snippets()
    print(f"Wrote Step 2-4 handbook figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
