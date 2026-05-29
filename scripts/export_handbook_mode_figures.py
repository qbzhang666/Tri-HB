from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "Tri-HB.py"
OUT_DIR = ROOT / "figures" / "handbook_modes"


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
        ("epsI_x_pos", "I X+", "#0f6fb6", "-"),
        ("epsR_x_pos", "R X+", "#f59e0b", "-"),
        ("epsT_x", "T X", "#10b981", "-"),
        ("epsI_y_pos", "I Y+", "#059669", "--"),
        ("epsR_y_pos", "R/out Y+", "#dc2626", "--"),
        ("epsT_y", "T Y", "#047857", ":"),
        ("epsI_z_pos", "I Z+", "#7c3aed", "--"),
        ("epsR_z_pos", "R/out Z+", "#be185d", "--"),
        ("epsT_z", "T Z", "#6d28d9", ":"),
        ("epsI_x_neg", "I X-", "#38bdf8", ":"),
        ("epsR_x_neg", "R X-", "#fbbf24", ":"),
        ("epsI_y_neg", "I Y-", "#34d399", ":"),
        ("epsR_y_neg", "R Y-", "#fb7185", ":"),
        ("epsI_z_neg", "I Z-", "#a78bfa", ":"),
        ("epsR_z_neg", "R Z-", "#f0abfc", ":"),
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
        ax.plot(t_us, result[key] * 1e6, label=label, color=color, linestyle=style, linewidth=1.3)

    ax.set_title("Bar gauge signals")
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Strain (microstrain)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2, frameon=False)


def plot_mode_figure(case, result, out_path):
    t_us = result["time"] * 1e6
    sig_x = result["sig_x"] / 1e6
    sig_y = result["sig_y"] / 1e6
    sig_z = result["sig_z"] / 1e6
    p, q = stress_invariants(sig_x, sig_y, sig_z)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), dpi=220)
    fig.suptitle(case["title"], fontsize=14, fontweight="bold")

    plot_bar_signals(axes[0, 0], result, case["mode"])

    axes[0, 1].plot(t_us, sig_x, label="sigma_X total", color="#0f6fb6", linewidth=1.8)
    axes[0, 1].plot(t_us, sig_y, label="sigma_Y total", color="#059669", linewidth=1.6)
    axes[0, 1].plot(t_us, sig_z, label="sigma_Z total", color="#7c3aed", linewidth=1.6)
    if case["mode"] == "em-symmetric":
        axes[0, 1].plot(t_us, p, label="p mean", color="#f59e0b", linestyle="--", linewidth=1.5)
    axes[0, 1].axhline(1000, color="#db2777", linestyle=":", linewidth=1.2, label="bar limit")
    axes[0, 1].set_title("Specimen total stresses")
    axes[0, 1].set_xlabel("Time (us)")
    axes[0, 1].set_ylabel("Stress (MPa)")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend(fontsize=7, frameon=False)

    eps_branch, sig_branch = loading_branch(result)
    if len(eps_branch):
        axes[1, 0].plot(eps_branch, sig_branch, color="#0f6fb6", linewidth=2.0)
    axes[1, 0].set_title("Axial stress-strain branch")
    axes[1, 0].set_xlabel("Axial strain (%)")
    axes[1, 0].set_ylabel("sigma_X total (MPa)")
    axes[1, 0].grid(True, alpha=0.25)

    if case["mode"] == "em-symmetric" and case["symmetric_axes"] == "XYZ":
        axes[1, 1].plot(result["eps_vol"] * 100.0, p, color="#f59e0b", linewidth=2.0)
        axes[1, 1].set_title("Hydrostatic compaction")
        axes[1, 1].set_xlabel("Volumetric strain (%)")
        axes[1, 1].set_ylabel("Mean pressure p (MPa)")
    else:
        axes[1, 1].plot(p, q, color="#334155", linewidth=2.0)
        axes[1, 1].scatter([p[np.argmax(q)]], [np.max(q)], color="#dc2626", s=18, zorder=3)
        axes[1, 1].set_title("Stress path")
        axes[1, 1].set_xlabel("Mean pressure p (MPa)")
        axes[1, 1].set_ylabel("Deviatoric stress q (MPa)")
    axes[1, 1].grid(True, alpha=0.25)

    text = (
        f"{case['caption']}\n"
        f"Peak specimen: {result['summary']['peak_specimen_stress_MPa']:.0f} MPa; "
        f"peak bar: {result['summary']['peak_bar_stress_MPa']:.0f} MPa; "
        f"mean rate: {result['summary']['avg_strain_rate']:.0f} /s"
    )
    fig.text(0.5, 0.018, text, ha="center", va="bottom", fontsize=8.2)
    fig.tight_layout(rect=[0, 0.055, 1, 0.94])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_summary(cases, results, out_path):
    names = [case["short"] for case in cases]
    peak_specimen = [results[case["id"]]["summary"]["peak_specimen_stress_MPa"] for case in cases]
    peak_bar = [results[case["id"]]["summary"]["peak_bar_stress_MPa"] for case in cases]
    rate = [results[case["id"]]["summary"]["avg_strain_rate"] for case in cases]

    x = np.arange(len(cases))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=220)
    width = 0.36
    axes[0].bar(x - width / 2, peak_specimen, width, label="Peak specimen stress", color="#0f6fb6")
    axes[0].bar(x + width / 2, peak_bar, width, label="Peak bar stress", color="#f59e0b")
    axes[0].axhline(1000, color="#db2777", linestyle=":", linewidth=1.2, label="bar limit")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=25, ha="right")
    axes[0].set_ylabel("Stress (MPa)")
    axes[0].set_title("Six-mode stress comparison")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(fontsize=8, frameon=False)

    axes[1].bar(x, rate, color="#059669")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25, ha="right")
    axes[1].set_ylabel("Mean loading strain rate (/s)")
    axes[1].set_title("Mean strain-rate comparison")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_latex_snippets(cases):
    lines = [
        "% Handbook figure snippets generated by scripts/export_handbook_mode_figures.py",
        "% Paths are relative to the repository root.",
        "",
    ]
    for case in cases:
        lines.extend(
            [
                "\\begin{figure}[htbp]",
                "  \\centering",
                f"  \\includegraphics[width=\\linewidth]{{figures/handbook_modes/{case['filename']}}}",
                f"  \\caption{{{case['latex_caption']}}}",
                f"  \\label{{fig:{case['id'].replace('_', '-')}}}",
                "\\end{figure}",
                "",
            ]
        )
    (OUT_DIR / "latex_include_snippets.tex").write_text("\n".join(lines), encoding="utf-8")


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
            filename="mode_1_gas_gun_uniaxial.png",
            latex_caption="Mode 1 sample run: sandstone, gas-gun uniaxial SHPB, V=20 m/s, no static confinement.",
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
            filename="mode_2_confinement_chamber.png",
            latex_caption="Mode 2 sample run: sandstone in confinement chamber, V=20 m/s, sigma1=30 MPa and pc=20 MPa.",
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
            filename="mode_3_monash_tri_hb.png",
            latex_caption="Mode 3 sample run: sandstone with independent static triaxial prestress, V=20 m/s.",
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
            filename="mode_4_em_uniaxial.png",
            latex_caption="Mode 4 sample run: sandstone, single EM half-sine pulse on +X, 400 MPa and 200 microseconds.",
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
            filename="mode_5_em_async_triaxial.png",
            latex_caption="Mode 5 sample run: sandstone, EM pulses on +X, +Y, +Z with 80 and 160 microsecond delays.",
            caption="Sandstone, EM peak=400 MPa, tau=200 us, Y delay=80 us, Z delay=160 us.",
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
        ),
        dict(
            id="mode_6_em_symmetric_xyz",
            short="M6 sym XYZ",
            title="Mode 6: EM symmetric full XYZ",
            filename="mode_6_em_symmetric_xyz.png",
            latex_caption="Mode 6 sample run: sandstone, full symmetric XYZ operation, 400 MPa single-bar pulse and 200 microseconds.",
            caption="Sandstone, full XYZ symmetric, EM peak=400 MPa per bar, tau=200 us, 50 MPa hydrostatic prestress.",
            mode="em-symmetric",
            velocity=0.0,
            peak_stress=400e6,
            pulse_duration=200e-6,
            confinement_X=50e6,
            confinement_Y=50e6,
            confinement_Z=50e6,
            pulse_delay_Y=0.0,
            pulse_delay_Z=0.0,
            symmetric_axes="XYZ",
        ),
    ]

    results = {}
    rows = []
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
        results[case["id"]] = result
        plot_mode_figure(case, result, OUT_DIR / case["filename"])
        rows.append(
            {
                "mode": case["title"],
                "file": case["filename"],
                "peak_specimen_stress_MPa": round(result["summary"]["peak_specimen_stress_MPa"], 3),
                "peak_bar_stress_MPa": round(result["summary"]["peak_bar_stress_MPa"], 3),
                "peak_pressure_MPa": round(result["summary"]["peak_pressure_MPa"], 3),
                "peak_strain_pct": round(result["summary"]["peak_strain_pct"], 4),
                "avg_strain_rate_s-1": round(result["summary"]["avg_strain_rate"], 3),
                "warning": "" if result["warning"] is None else result["warning"]["level"],
            }
        )

    plot_summary(cases, results, OUT_DIR / "mode_comparison_summary.png")
    pd.DataFrame(rows).to_csv(OUT_DIR / "sample_case_summary.csv", index=False)
    (OUT_DIR / "sample_case_summary.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    write_latex_snippets(cases)
    print(f"Wrote {len(cases) + 1} PNG files to {OUT_DIR}")


if __name__ == "__main__":
    main()
