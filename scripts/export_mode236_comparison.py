from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Handbook" / "figures" / "handbook_steps"
CSV_PATH = OUT_DIR / "mode236_controlled_comparison.csv"

PUB_COLORS = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "purple": "#7B3FE4",
    "black": "#222222",
    "gray": "#6B7280",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "Cambria", "DejaVu Serif"],
        "mathtext.fontset": "stix",
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
    }
)


def half_sine(t: np.ndarray, amplitude: float, duration: float, delay: float = 0.0) -> np.ndarray:
    y = np.zeros_like(t)
    mask = (t >= delay) & (t <= delay + duration)
    y[mask] = amplitude * np.sin(np.pi * (t[mask] - delay) / duration)
    return y


def central_difference(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.gradient(y, t)


def cumulative_trapezoid(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))
    return out


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def invariants(sx: np.ndarray, sy: np.ndarray, sz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = (sx + sy + sz) / 3.0
    q = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2))
    s1, s2, s3 = sx - p, sy - p, sz - p
    j2 = (s1**2 + s2**2 + s3**2) / 2.0
    j3 = s1 * s2 * s3
    theta = np.zeros_like(p)
    mask = j2 > 1e-12
    arg = np.zeros_like(p)
    arg[mask] = (3.0 * np.sqrt(3.0) / 2.0) * j3[mask] / (j2[mask] ** 1.5)
    theta[mask] = np.rad2deg(np.arccos(np.clip(arg[mask], -1.0, 1.0)) / 3.0)
    return p, q, theta


def damage_profile(final_damage: float, kind: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    x = np.linspace(0.0, 1.0, 500)
    if final_damage <= 1e-10:
        return x, np.zeros_like(x), 0.0, 1.0

    left = np.exp(-((x - 0.22) / 0.14) ** 2) * final_damage
    if kind == "Mode 2":
        central = 0.35 * np.exp(-((x - 0.50) / 0.18) ** 2) * final_damage
        right = 0.08 * np.exp(-((x - 0.82) / 0.14) ** 2) * final_damage
    elif kind == "Mode 3":
        central = 0.78 * np.exp(-((x - 0.50) / 0.20) ** 2) * final_damage
        right = 0.36 * np.exp(-((x - 0.78) / 0.16) ** 2) * final_damage
    else:
        central = np.zeros_like(x)
        right = np.zeros_like(x)

    profile = np.clip(left + central + right, 0.0, 1.0)
    centre_mask = np.abs(x - 0.5) < 0.15
    total_area = trapezoid(profile, x)
    central_area = trapezoid(profile[centre_mask], x[centre_mask])
    left_area = trapezoid(profile[x < 0.5], x[x < 0.5])
    right_area = trapezoid(profile[x >= 0.5], x[x >= 0.5])
    dc = central_area / max(total_area, 1e-12)
    sx_sym = 1.0 - abs(left_area - right_area) / max(left_area + right_area, 1e-12)
    return x, profile, dc, sx_sym


def compute_case(kind: str) -> dict:
    # Controlled comparison used in the presentation:
    # sandstone, same static prestress, same half-sine pulse scale.
    e_gpa = 15.0
    nu = 0.20
    rho = 2650.0
    sx0, sy0, sz0 = 30.0, 20.0, 20.0
    amplitude = 400.0
    duration = 200e-6
    t = np.linspace(0.0, 500e-6, 6000)
    g = half_sine(t, amplitude, duration)

    if kind == "Mode 2":
        sx = sx0 + g
        sy = np.full_like(t, sy0)
        sz = np.full_like(t, sz0)
        model_note = "axisymmetric chamber; lateral stresses locked"
    elif kind == "Mode 3":
        sx = sx0 + g
        # Small delayed lateral wave response mirrors the active true-triaxial
        # boundary while preserving the same pre-static state and input pulse.
        sy = sy0 + 0.22 * half_sine(t, amplitude, duration, 12e-6)
        sz = sz0 + 0.18 * half_sine(t, amplitude, duration, 18e-6)
        model_note = "true-triaxial bars; lateral response included"
    elif kind == "Mode 6":
        sx = sx0 + 2.0 * g
        sy = sy0 + 2.0 * g
        sz = sz0 + 2.0 * g
        model_note = "symmetric XYZ; opposed pulses double each axis"
    else:
        raise ValueError(kind)

    p, q, theta = invariants(sx, sy, sz)

    a_fail = 15.0
    b_fail = 1.3
    n_fail = 0.75
    lode_amp = 0.10
    theta_rad = np.deg2rad(theta)
    h_theta = 1.0 + lode_amp * (1.0 - np.cos(3.0 * theta_rad))
    qf = (a_fail + b_fail * np.maximum(p, 0.0) ** n_fail) * h_theta
    failure_index = q / np.maximum(qf, 1e-9)

    e_mpa = e_gpa * 1000.0
    eps_x = sx / e_mpa
    eps_y = sy / e_mpa
    eps_z = sz / e_mpa
    epsdot_x = central_difference(eps_x, t)
    epsdot_y = central_difference(eps_y, t)
    epsdot_z = central_difference(eps_z, t)
    epsdot_eq = np.sqrt(
        2.0
        / 3.0
        * ((epsdot_x - epsdot_y) ** 2 + (epsdot_y - epsdot_z) ** 2 + (epsdot_z - epsdot_x) ** 2)
        / 2.0
    )

    tau_d = 30e-6
    alpha_sat = 1.0
    m_over = 2.0
    beta_rate = 0.15
    epsdot0 = 1.0
    rate_factor = (np.maximum(np.abs(epsdot_eq), 1e-12) / epsdot0) ** beta_rate
    damage = np.zeros_like(t)
    damage_rate = np.zeros_like(t)
    for i in range(1, len(t)):
        overstress = max(failure_index[i - 1] - 1.0, 0.0)
        damage_rate[i - 1] = ((1.0 - damage[i - 1]) ** alpha_sat) / tau_d * (overstress**m_over) * rate_factor[i - 1]
        damage[i] = np.clip(damage[i - 1] + damage_rate[i - 1] * (t[i] - t[i - 1]), 0.0, 1.0)
    damage_rate[-1] = damage_rate[-2]

    bulk_longitudinal = e_gpa * 1e9 * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    cp0 = np.sqrt(bulk_longitudinal / rho)
    e_d = e_gpa * (1.0 - damage)
    cp_d = cp0 * np.sqrt(np.maximum(1.0 - damage, 0.0))

    w_el = (1.0 / (2.0 * e_mpa)) * (
        sx**2 + sy**2 + sz**2 - 2.0 * nu * (sx * sy + sy * sz + sz * sx)
    )
    power = sx * epsdot_x + sy * epsdot_y + sz * epsdot_z
    w_input = cumulative_trapezoid(power, t)
    w_diss = w_input - w_el + w_el[0]

    x, profile, dc, sx_sym = damage_profile(float(damage[-1]), kind)
    onset = float(t[np.argmax(failure_index > 1.0)] * 1e6) if np.any(failure_index > 1.0) else float("nan")

    return {
        "kind": kind,
        "note": model_note,
        "t_us": t * 1e6,
        "sx": sx,
        "sy": sy,
        "sz": sz,
        "p": p,
        "q": q,
        "theta": theta,
        "qf": qf,
        "failure_index": failure_index,
        "damage": damage,
        "damage_rate": damage_rate,
        "e_d": e_d,
        "cp_d": cp_d,
        "w_diss": w_diss,
        "x": x,
        "profile": profile,
        "dc": dc,
        "sx_sym": sx_sym,
        "peak_sx": float(np.max(sx)),
        "peak_sy": float(np.max(sy)),
        "peak_sz": float(np.max(sz)),
        "peak_p": float(np.max(p)),
        "peak_q": float(np.max(q)),
        "peak_theta": float(theta[np.argmax(q)]),
        "peak_f": float(np.max(failure_index)),
        "onset_us": onset,
        "final_damage": float(damage[-1]),
        "max_w_diss": float(np.max(w_diss)),
        "final_e": float(e_d[-1]),
        "final_cp": float(cp_d[-1]),
    }


def style_axis(ax, title: str | None = None) -> None:
    if title:
        ax.set_title(title, fontsize=15, pad=7)
    ax.tick_params(axis="both", labelsize=11, direction="out", length=4.0, width=0.9)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.grid(True, which="major")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_frame_on(False)
        for text in legend.get_texts():
            text.set_fontsize(10.5)


def export_csv(cases: list[dict]) -> None:
    rows = []
    for case in cases:
        rows.append(
            {
                "mode": case["kind"],
                "controlled_condition": "sigma0=30/20/20 MPa; A=400 MPa; tau=200 us; half-sine",
                "model_note": case["note"],
                "peak_sigma_x_MPa": f"{case['peak_sx']:.1f}",
                "peak_sigma_y_MPa": f"{case['peak_sy']:.1f}",
                "peak_sigma_z_MPa": f"{case['peak_sz']:.1f}",
                "peak_p_MPa": f"{case['peak_p']:.1f}",
                "peak_q_MPa": f"{case['peak_q']:.1f}",
                "theta_at_peak_q_deg": f"{case['peak_theta']:.1f}",
                "peak_failure_index": f"{case['peak_f']:.2f}",
                "damage_onset_us": "" if np.isnan(case["onset_us"]) else f"{case['onset_us']:.1f}",
                "final_damage": f"{case['final_damage']:.2f}",
                "central_damage_fraction": f"{case['dc']:.2f}",
                "damage_symmetry_index": f"{case['sx_sym']:.2f}",
                "max_dissipated_energy_MJ_m3": f"{case['max_w_diss']:.2f}",
            }
        )

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_stress_comparison(cases: list[dict]) -> None:
    colors = [PUB_COLORS["blue"], PUB_COLORS["green"], PUB_COLORS["vermillion"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=240)
    ax = axes[0]
    for case, color in zip(cases, colors):
        ax.plot(case["t_us"], case["p"], label=f"{case['kind']} $p$", color=color, linewidth=1.95)
        ax.plot(case["t_us"], case["q"], label=f"{case['kind']} $q$", color=color, linestyle="--", linewidth=1.65)
    ax.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax.set_ylabel(r"Stress invariant (MPa)")
    ax.legend(ncol=2, loc="upper right")
    style_axis(ax, r"Controlled $p$ and $q$ histories")

    ax = axes[1]
    for case, color in zip(cases, colors):
        ax.plot(case["p"], case["q"], label=case["kind"], color=color, linewidth=2.05)
        ax.scatter([case["p"][np.argmax(case["q"])]], [np.max(case["q"])], color=color, s=22)
    ax.set_xlabel(r"Mean pressure, $p$ (MPa)")
    ax.set_ylabel(r"Deviatoric stress, $q$ (MPa)")
    ax.legend(loc="upper right")
    style_axis(ax, r"Stress-path comparison")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mode236_controlled_stress_comparison.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_damage_comparison(cases: list[dict]) -> None:
    colors = [PUB_COLORS["blue"], PUB_COLORS["green"], PUB_COLORS["vermillion"]]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), dpi=240)

    ax = axes[0]
    for case, color in zip(cases, colors):
        ax.plot(case["t_us"], case["failure_index"], label=case["kind"], color=color, linewidth=2.0)
    ax.axhline(1.0, color=PUB_COLORS["black"], linestyle=":", linewidth=1.25, label=r"$F=1$")
    ax.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax.set_ylabel(r"Failure index, $F$")
    ax.legend(loc="upper right")
    style_axis(ax, r"Damage trigger")

    ax = axes[1]
    for case, color in zip(cases, colors):
        ax.plot(case["t_us"], case["damage"], label=case["kind"], color=color, linewidth=2.0)
    ax.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax.set_ylabel(r"Scalar damage, $D$")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="lower right")
    style_axis(ax, r"Damage accumulation")

    ax = axes[2]
    width = 0.24
    xpos = np.arange(len(cases))
    ax.bar(xpos - width, [c["final_damage"] for c in cases], width, label=r"$D_f$", color=PUB_COLORS["blue"])
    ax.bar(xpos, [c["dc"] for c in cases], width, label=r"$D_c$", color=PUB_COLORS["orange"])
    ax.bar(xpos + width, [c["sx_sym"] for c in cases], width, label=r"$S_x$", color=PUB_COLORS["green"])
    ax.set_xticks(xpos, [c["kind"] for c in cases])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Descriptor value")
    ax.legend(loc="upper right")
    style_axis(ax, "Step 4 descriptors")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "step4_mode236_damage_model_comparison.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [compute_case(kind) for kind in ["Mode 2", "Mode 3", "Mode 6"]]
    export_csv(cases)
    plot_stress_comparison(cases)
    plot_damage_comparison(cases)
    print(f"Wrote Mode 2/3/6 comparison figures and data to {OUT_DIR}")


if __name__ == "__main__":
    main()
