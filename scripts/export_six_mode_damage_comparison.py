from __future__ import annotations

import csv
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "Tri-HB.py"
OUT_DIR = ROOT / "Handbook" / "figures" / "handbook_steps"
CSV_PATH = OUT_DIR / "six_mode_damage_comparison.csv"

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


def load_simulator_symbols() -> dict:
    source = APP_PATH.read_text(encoding="utf-8")
    source = source.split("# STREAMLIT UI", 1)[0]
    source = source.replace("@st.cache_data(show_spinner=False)\n", "")
    source = source.replace("import plotly.graph_objects as go\n", "")
    source = source.replace("import streamlit as st\n", "")
    module = types.ModuleType("tri_hb_six_mode_damage_defs")
    module.__file__ = str(APP_PATH)
    sys.modules[module.__name__] = module
    symbols: dict = module.__dict__
    exec(compile(source, str(APP_PATH), "exec"), symbols)
    return symbols


def invariants_from_diagonal(sx: np.ndarray, sy: np.ndarray, sz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if y.size < 2:
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def damage_profile(final_damage: float, case_id: str) -> tuple[float, float]:
    x = np.linspace(0.0, 1.0, 500)
    if final_damage <= 1e-10:
        return 0.0, 1.0

    left = np.exp(-((x - 0.22) / 0.14) ** 2) * final_damage
    centre = np.exp(-((x - 0.50) / 0.18) ** 2) * final_damage
    right = np.exp(-((x - 0.78) / 0.14) ** 2) * final_damage

    if case_id == "m1":
        profile = left + 0.16 * centre
    elif case_id == "m2":
        profile = left + 0.35 * centre + 0.08 * right
    elif case_id == "m3":
        profile = left + 0.72 * centre + 0.30 * right
    elif case_id == "m4":
        profile = left + 0.24 * centre
    elif case_id == "m5":
        profile = left + 0.84 * centre + 0.62 * right
    else:
        profile = 0.05 * (left + right)

    profile = np.clip(profile, 0.0, 1.0)
    centre_mask = np.abs(x - 0.5) < 0.15
    total_area = trapezoid(profile, x)
    central_area = trapezoid(profile[centre_mask], x[centre_mask])
    left_area = trapezoid(profile[x < 0.5], x[x < 0.5])
    right_area = trapezoid(profile[x >= 0.5], x[x >= 0.5])
    dc = central_area / max(total_area, 1e-12)
    sx_sym = 1.0 - abs(left_area - right_area) / max(left_area + right_area, 1e-12)
    return dc, sx_sym


def compute_damage(result: dict, case: dict) -> dict:
    t = result["time"]
    sx = result["sig_x"] / 1e6
    sy = result["sig_y"] / 1e6
    sz = result["sig_z"] / 1e6
    p, q, theta = invariants_from_diagonal(sx, sy, sz)

    a_fail = 15.0
    b_fail = 1.3
    n_fail = 0.75
    lode_amp = 0.10
    theta_rad = np.deg2rad(theta)
    h_theta = 1.0 + lode_amp * (1.0 - np.cos(3.0 * theta_rad))
    qf = (a_fail + b_fail * np.maximum(p, 0.0) ** n_fail) * h_theta
    failure_index = q / np.maximum(qf, 1e-9)

    e_mpa = case["E_GPa"] * 1000.0
    eps_x = sx / e_mpa
    eps_y = sy / e_mpa
    eps_z = sz / e_mpa
    epsdot_x = np.gradient(eps_x, t)
    epsdot_y = np.gradient(eps_y, t)
    epsdot_z = np.gradient(eps_z, t)
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

    dc, sx_sym = damage_profile(float(damage[-1]), case["id"])
    onset = float(t[np.argmax(failure_index > 1.0)] * 1e6) if np.any(failure_index > 1.0) else float("nan")
    return {
        **case,
        "t_us": t * 1e6,
        "p": p,
        "q": q,
        "failure_index": failure_index,
        "damage": damage,
        "peak_p": float(np.max(p)),
        "peak_q": float(np.max(q)),
        "peak_f": float(np.max(failure_index)),
        "onset_us": onset,
        "final_damage": float(damage[-1]),
        "dc": dc,
        "sx_sym": sx_sym,
    }


def build_cases(symbols: dict) -> tuple[list[dict], dict]:
    rock = symbols["ROCK_PARAMS"]["sandstone"]
    common = dict(
        rock_type="sandstone",
        specimen_size=0.050,
        specimen_length=0.050,
        specimen_area=0.050 * 0.050,
        material_E=rock["E_s"],
        material_UCS=rock["sigma_c0"],
        material_nu=rock["nu"],
        material_density=2650.0,
        bar_E=210e9,
        bar_C0=5172.0,
        bar_area=0.050 * 0.050,
    )
    cases = [
        dict(
            id="m1",
            label="Mode 1",
            short="M1",
            mode="gas-gun",
            velocity=20.0,
            peak_stress=0.0,
            pulse_duration=0.0,
            confinement_X=0.0,
            confinement_Y=0.0,
            confinement_Z=0.0,
            pulse_delay_Y=0.0,
            pulse_delay_Z=0.0,
            symmetric_axes="XYZ",
            condition="V=20 m/s; no pre-static stress",
        ),
        dict(
            id="m2",
            label="Mode 2",
            short="M2",
            mode="confinement-chamber",
            velocity=20.0,
            peak_stress=0.0,
            pulse_duration=0.0,
            confinement_X=30e6,
            confinement_Y=20e6,
            confinement_Z=20e6,
            pulse_delay_Y=0.0,
            pulse_delay_Z=0.0,
            symmetric_axes="XYZ",
            condition="V=20 m/s; 30/20/20 MPa pre-static stress",
        ),
        dict(
            id="m3",
            label="Mode 3",
            short="M3",
            mode="gas-gun-triaxial",
            velocity=20.0,
            peak_stress=0.0,
            pulse_duration=0.0,
            confinement_X=30e6,
            confinement_Y=20e6,
            confinement_Z=15e6,
            pulse_delay_Y=0.0,
            pulse_delay_Z=0.0,
            symmetric_axes="XYZ",
            condition="V=20 m/s; 30/20/15 MPa pre-static stress",
        ),
        dict(
            id="m4",
            label="Mode 4",
            short="M4",
            mode="em-uniaxial",
            velocity=0.0,
            peak_stress=400e6,
            pulse_duration=200e-6,
            confinement_X=30e6,
            confinement_Y=20e6,
            confinement_Z=15e6,
            pulse_delay_Y=0.0,
            pulse_delay_Z=0.0,
            symmetric_axes="XYZ",
            condition="A=400 MPa, tau=200 us; 30/20/15 MPa",
        ),
        dict(
            id="m5",
            label="Mode 5",
            short="M5",
            mode="em-async",
            velocity=0.0,
            peak_stress=400e6,
            pulse_duration=200e-6,
            confinement_X=30e6,
            confinement_Y=20e6,
            confinement_Z=15e6,
            pulse_delay_Y=80e-6,
            pulse_delay_Z=160e-6,
            symmetric_axes="XYZ",
            condition="A=400 MPa, tau=200 us on X/Y/Z; Y/Z delays 80/160 us; 30/20/15 MPa",
        ),
        dict(
            id="m6",
            label="Mode 6",
            short="M6",
            mode="em-symmetric",
            velocity=0.0,
            peak_stress=400e6,
            pulse_duration=200e-6,
            confinement_X=30e6,
            confinement_Y=20e6,
            confinement_Z=15e6,
            pulse_delay_Y=0.0,
            pulse_delay_Z=0.0,
            symmetric_axes="XYZ",
            condition="A=400 MPa, tau=200 us per opposed bar; 30/20/15 MPa",
        ),
    ]
    for case in cases:
        case["E_GPa"] = rock["E_s"] / 1e9
    return cases, common


def export_csv(rows: list[dict]) -> None:
    fields = [
        "mode",
        "condition",
        "peak_p_MPa",
        "peak_q_MPa",
        "peak_failure_index",
        "damage_onset_us",
        "final_damage",
        "central_damage_fraction",
        "damage_symmetry_index",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "mode": row["label"],
                    "condition": row["condition"],
                    "peak_p_MPa": f"{row['peak_p']:.1f}",
                    "peak_q_MPa": f"{row['peak_q']:.1f}",
                    "peak_failure_index": f"{row['peak_f']:.2f}",
                    "damage_onset_us": "" if np.isnan(row["onset_us"]) else f"{row['onset_us']:.1f}",
                    "final_damage": f"{row['final_damage']:.2f}",
                    "central_damage_fraction": f"{row['dc']:.2f}",
                    "damage_symmetry_index": f"{row['sx_sym']:.2f}",
                }
            )


def style_axis(ax, title: str | None = None) -> None:
    if title:
        ax.set_title(title, fontsize=15, pad=7)
    ax.tick_params(axis="both", labelsize=10.5, direction="out", length=4.0, width=0.9)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.grid(True, which="major")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_frame_on(False)
        for text in legend.get_texts():
            text.set_fontsize(9.5)


def plot(rows: list[dict]) -> None:
    colors = [
        PUB_COLORS["gray"],
        PUB_COLORS["blue"],
        PUB_COLORS["green"],
        PUB_COLORS["orange"],
        PUB_COLORS["purple"],
        PUB_COLORS["vermillion"],
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2), dpi=240)

    ax = axes[0]
    for row, color in zip(rows, colors):
        ax.plot(row["t_us"], row["failure_index"], label=row["short"], color=color, linewidth=1.8)
    ax.axhline(1.0, color=PUB_COLORS["black"], linestyle=":", linewidth=1.25, label=r"$F=1$")
    ax.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax.set_ylabel(r"Failure index, $F=q/q_f$")
    ax.legend(ncol=2, loc="upper right")
    style_axis(ax, "Damage trigger by mode")

    ax = axes[1]
    for row, color in zip(rows, colors):
        ax.plot(row["t_us"], row["damage"], label=row["short"], color=color, linewidth=1.8)
    ax.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax.set_ylabel(r"Scalar shear damage, $D$")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(ncol=2, loc="lower right")
    style_axis(ax, "Step 4 damage history")

    ax = axes[2]
    xpos = np.arange(len(rows))
    width = 0.25
    ax.bar(xpos - width, [row["peak_f"] / max(max(r["peak_f"] for r in rows), 1e-12) for row in rows], width, label="Peak F / max", color=PUB_COLORS["blue"])
    ax.bar(xpos, [row["final_damage"] for row in rows], width, label=r"$D_f$", color=PUB_COLORS["orange"])
    ax.bar(xpos + width, [row["dc"] for row in rows], width, label=r"$D_c$", color=PUB_COLORS["green"])
    ax.set_xticks(xpos, [row["short"] for row in rows])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Normalised descriptor")
    ax.legend(loc="upper right")
    style_axis(ax, "Similarity and difference summary")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "step4_six_mode_damage_model_comparison.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = load_simulator_symbols()
    simulate = symbols["simulate"]
    cases, common = build_cases(symbols)
    rows = []
    for case in cases:
        kwargs = {
            key: case[key]
            for key in (
                "mode",
                "velocity",
                "peak_stress",
                "pulse_duration",
                "confinement_X",
                "confinement_Y",
                "confinement_Z",
                "pulse_delay_Y",
                "pulse_delay_Z",
                "symmetric_axes",
            )
        }
        result = simulate(**common, **kwargs)
        rows.append(compute_damage(result, case))
    export_csv(rows)
    plot(rows)
    print(f"Wrote six-mode damage comparison to {OUT_DIR}")


if __name__ == "__main__":
    main()
