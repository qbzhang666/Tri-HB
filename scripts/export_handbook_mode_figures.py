from __future__ import annotations

import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "Tri-HB.py"
OUT_DIR = ROOT / "Handbook" / "figures" / "handbook_modes"
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


def load_simulator_symbols() -> dict:
    source = APP_PATH.read_text(encoding="utf-8")
    source = source.split("# STREAMLIT UI", 1)[0]
    source = source.replace("@st.cache_data(show_spinner=False)\n", "")
    source = source.replace("import plotly.graph_objects as go\n", "")
    source = source.replace("import streamlit as st\n", "")
    module_name = "tri_hb_export_defs"
    module = types.ModuleType(module_name)
    module.__file__ = str(APP_PATH)
    sys.modules[module_name] = module
    symbols: dict = module.__dict__
    exec(compile(source, str(APP_PATH), "exec"), symbols)
    return symbols


def stress_invariants(sig_x, sig_y, sig_z):
    p = (sig_x + sig_y + sig_z) / 3.0
    q = np.sqrt(
        0.5
        * (
            (sig_x - sig_y) ** 2
            + (sig_y - sig_z) ** 2
            + (sig_z - sig_x) ** 2
        )
    )
    return p, q


def loading_branch(result):
    eps = result["eps_x"] * 100.0
    sig = result["sig_x"] / 1e6
    keep_x = []
    keep_y = []
    last = -1.0
    plateau = 0
    for e, s in zip(eps, sig):
        if e <= 0:
            continue
        if e > last + 1e-5:
            keep_x.append(e)
            keep_y.append(s)
            last = e
            plateau = 0
        else:
            plateau += 1
            if plateau > 5:
                break
    return np.asarray(keep_x), np.asarray(keep_y)


def plot_bar_signals(ax, result, mode):
    t_us = result["time"] * 1e6
    candidates = [
        ("epsI_x_pos", r"$\varepsilon_I$ X+", PUB_COLORS["blue"], "-"),
        ("epsR_x_pos", r"$\varepsilon_R$ X+", PUB_COLORS["orange"], "--"),
        ("epsT_x", r"$\varepsilon_T$ X", PUB_COLORS["green"], "-"),
        ("epsI_y_pos", r"$\varepsilon_I$ Y+", PUB_COLORS["green"], "-"),
        ("epsR_y_pos", r"$\varepsilon_R$ Y+", PUB_COLORS["vermillion"], "--"),
        ("epsT_y", r"$\varepsilon_T$ Y", "#047857", ":"),
        ("epsI_z_pos", r"$\varepsilon_I$ Z+", PUB_COLORS["purple"], "-"),
        ("epsR_z_pos", r"$\varepsilon_R$ Z+", PUB_COLORS["magenta"], "--"),
        ("epsT_z", r"$\varepsilon_T$ Z", "#6d28d9", ":"),
        ("epsI_x_neg", r"$\varepsilon_I$ X-", PUB_COLORS["sky"], ":"),
        ("epsR_x_neg", r"$\varepsilon_R$ X-", "#fbbf24", ":"),
        ("epsI_y_neg", r"$\varepsilon_I$ Y-", "#34d399", ":"),
        ("epsR_y_neg", r"$\varepsilon_R$ Y-", "#fb7185", ":"),
        ("epsI_z_neg", r"$\varepsilon_I$ Z-", "#a78bfa", ":"),
        ("epsR_z_neg", r"$\varepsilon_R$ Z-", "#f0abfc", ":"),
    ]
    active = []
    for key, label, color, style in candidates:
        y = result[key] * 1e6
        if np.nanmax(np.abs(y)) > 0.05:
            active.append((np.nanmax(np.abs(y)), key, label, color, style))

    if mode == "em-symmetric":
        preferred = {
            "epsI_x_pos",
            "epsI_y_pos",
            "epsI_z_pos",
            "epsR_x_pos",
            "epsR_y_pos",
            "epsR_z_pos",
        }
        active = [item for item in active if item[1] in preferred]
    elif mode == "em-async":
        preferred = {
            "epsI_x_pos",
            "epsI_y_pos",
            "epsI_z_pos",
            "epsT_x",
            "epsT_y",
            "epsT_z",
        }
        active = [item for item in active if item[1] in preferred]
    else:
        active = sorted(active, reverse=True)[:6]

    for _, key, label, color, style in active:
        ax.plot(t_us, result[key] * 1e6, label=label, color=color, linestyle=style, linewidth=1.75)

    ax.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax.set_ylabel(r"Bar strain, $\varepsilon$ ($\mu\varepsilon$)")
    ax.legend(ncol=2, loc="upper right")


def _style_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=17, fontweight="normal", pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", labelsize=13, direction="out", length=4.2, width=0.9)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.grid(True, which="major")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_frame_on(False)
        for text in legend.get_texts():
            text.set_fontsize(12)


def _save_single_panel(fig, ax, case, panel, summary_text):
    _style_axis(ax, panel["title"], ax.get_xlabel(), ax.get_ylabel())
    fig.tight_layout()
    out_path = OUT_DIR / panel["filename"]
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_mode_panels(case, result):
    t_us = result["time"] * 1e6
    sig_x = result["sig_x"] / 1e6
    sig_y = result["sig_y"] / 1e6
    sig_z = result["sig_z"] / 1e6
    p, q = stress_invariants(sig_x, sig_y, sig_z)
    summary_text = ""

    generated = []

    panel = {
        "key": "bar_signals",
        "title": "Bar-gauge waveform interaction",
        "filename": f"{case['id']}_bar_signals.png",
    }
    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    plot_bar_signals(ax, result, case["mode"])
    generated.append(_save_single_panel(fig, ax, case, panel, summary_text))

    panel = {
        "key": "stress_history",
        "title": "Specimen stress history",
        "filename": f"{case['id']}_stress_history.png",
    }
    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    ax.plot(t_us, sig_x, label=r"$\sigma_X$", color=PUB_COLORS["blue"], linewidth=2.05)
    ax.plot(t_us, sig_y, label=r"$\sigma_Y$", color=PUB_COLORS["green"], linewidth=1.95)
    ax.plot(t_us, sig_z, label=r"$\sigma_Z$", color=PUB_COLORS["purple"], linewidth=1.95)
    if case["mode"] == "em-symmetric":
        ax.plot(t_us, p, label=r"$p$", color=PUB_COLORS["orange"], linestyle="--", linewidth=1.75)
    ax.axhline(1000, color=PUB_COLORS["magenta"], linestyle=":", linewidth=1.4, label="bar limit")
    ax.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax.set_ylabel(r"Stress, $\sigma$ (MPa)")
    ax.legend(loc="upper right")
    generated.append(_save_single_panel(fig, ax, case, panel, summary_text))

    panel = {
        "key": "stress_strain",
        "title": "Axial stress-strain response",
        "filename": f"{case['id']}_stress_strain.png",
    }
    eps_branch, sig_branch = loading_branch(result)
    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    if len(eps_branch):
        ax.plot(eps_branch, sig_branch, color=PUB_COLORS["blue"], linewidth=2.15, label=r"$\sigma_X(\varepsilon_X)$")
    ax.set_xlabel(r"Axial strain, $\varepsilon_X$ (%)")
    ax.set_ylabel(r"Axial stress, $\sigma_X$ (MPa)")
    if len(eps_branch):
        ax.legend(loc="upper left")
    generated.append(_save_single_panel(fig, ax, case, panel, summary_text))

    panel = {
        "key": "stress_path",
        "title": "Stress-path interaction",
        "filename": f"{case['id']}_stress_path.png",
    }
    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=240)
    if case["mode"] == "em-symmetric" and case["symmetric_axes"] == "XYZ":
        ax.plot(result["eps_vol"] * 100.0, p, color=PUB_COLORS["blue"], linewidth=2.15, label=r"$p(\varepsilon_v)$")
        ax.set_xlabel(r"Volumetric strain, $\varepsilon_v$ (%)")
        ax.set_ylabel(r"Mean pressure, $p$ (MPa)")
        ax.legend(loc="upper left")
    else:
        ax.plot(p, q, color=PUB_COLORS["blue"], linewidth=2.15, label=r"$p$--$q$ path")
        ax.scatter([p[np.argmax(q)]], [np.max(q)], color=PUB_COLORS["vermillion"], s=34, zorder=3, label="peak q")
        ax.set_xlabel(r"Mean pressure, $p$ (MPa)")
        ax.set_ylabel(r"Deviatoric stress, $q$ (MPa)")
        ax.legend(loc="upper left")
    generated.append(_save_single_panel(fig, ax, case, panel, summary_text))

    return generated


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = load_simulator_symbols()
    simulate = symbols["simulate"]
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
            id="mode_1_gas_gun_uniaxial",
            short="M1 gas",
            title="Mode 1: Gas-gun uniaxial SHPB",
            caption="Sandstone, V=20 m/s, sigma1=sigma2=sigma3=0 MPa.",
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
        ),
        dict(
            id="mode_2_confinement_chamber",
            short="M2 chamber",
            title="Mode 2: Confinement-chamber triaxial SHPB",
            caption="Sandstone, V=20 m/s, sigma1=30 MPa, pc=sigma2=sigma3=20 MPa.",
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
        ),
        dict(
            id="mode_3_monash_tri_hb",
            short="M3 Tri-HB",
            title="Mode 3: Monash Tri-HB gas-gun triaxial",
            caption="Sandstone, V=20 m/s, sigma1=30 MPa, sigma2=20 MPa, sigma3=15 MPa.",
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
        ),
        dict(
            id="mode_4_em_uniaxial",
            short="M4 EM X",
            title="Mode 4: EM uniaxial half-sine",
            caption="Sandstone, EM peak=400 MPa, tau=200 us, sigma1=30 MPa, sigma2=20 MPa, sigma3=15 MPa.",
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
        ),
        dict(
            id="mode_5_em_async_triaxial",
            short="M5 async",
            title="Mode 5: EM asynchronous triaxial",
            caption="Sandstone, EM peak=400 MPa, tau=200 us, Y delay=0 us, Z delay=0 us.",
            mode="em-async",
            velocity=0.0,
            peak_stress=400e6,
            pulse_duration=200e-6,
            confinement_X=30e6,
            confinement_Y=20e6,
            confinement_Z=15e6,
            pulse_delay_Y=0.0,
            pulse_delay_Z=0.0,
            symmetric_axes="XYZ",
        ),
        dict(
            id="mode_6_em_symmetric_xyz",
            short="M6 sym XYZ",
            title="Mode 6: EM symmetric full XYZ",
            caption="Sandstone, full XYZ symmetric, EM peak=400 MPa per bar, tau=200 us, sigma1=30 MPa, sigma2=20 MPa, sigma3=15 MPa.",
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
        ),
    ]

    panel_caption_templates = {
        "bar_signals": "Mode~{mode_no} default sandstone case: separated incident, reflected, and transmitted bar-gauge strain histories for the active bars.",
        "stress_history": "Mode~{mode_no} default sandstone case: total specimen stress histories on the principal axes, including the applied static pre-stress.",
        "stress_strain": "Mode~{mode_no} default sandstone case: axial stress--strain loading branch extracted from the simulated Hopkinson-bar records.",
        "stress_path": "Mode~{mode_no} default sandstone case: stress-path diagnostic used for confinement, Lode-angle, or hydrostatic-compaction interpretation.",
    }
    for mode_no, case in enumerate(cases, start=1):
        case["panels"] = [
            {
                "key": "bar_signals",
                "filename": f"{case['id']}_bar_signals.png",
                "caption": panel_caption_templates["bar_signals"].format(mode_no=mode_no),
            },
            {
                "key": "stress_history",
                "filename": f"{case['id']}_stress_history.png",
                "caption": panel_caption_templates["stress_history"].format(mode_no=mode_no),
            },
            {
                "key": "stress_strain",
                "filename": f"{case['id']}_stress_strain.png",
                "caption": panel_caption_templates["stress_strain"].format(mode_no=mode_no),
            },
            {
                "key": "stress_path",
                "filename": f"{case['id']}_stress_path.png",
                "caption": panel_caption_templates["stress_path"].format(mode_no=mode_no),
            },
        ]

    generated_count = 0
    for case in cases:
        kwargs = {k: case[k] for k in (
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
        )}
        result = simulate(**common, **kwargs)
        generated_files = plot_mode_panels(case, result)
        generated_count += len(generated_files)

    print(f"Wrote {generated_count} PNG files to {OUT_DIR}")


if __name__ == "__main__":
    main()
