"""
Integrated Tri-HB Streamlit app.

Run with:
    streamlit run tri_hb_integrated.py

This file keeps the existing specialist apps available while adding one
navigation shell and an experimental SHPB/Tri-HB data-reduction workspace.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


APP_DIR = Path(__file__).resolve().parent

# High-resolution publication defaults. On-screen previews stay light (150 dpi)
# while every downloaded / bundled PNG is rendered at 600 dpi.
PUB_DPI = 600

matplotlib.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": PUB_DPI,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "lines.linewidth": 1.8,
    "axes.linewidth": 0.9,
    "savefig.bbox": "tight",
    # Keep the default (tight) layout engine. constrained_layout must stay off
    # because wave_damage.py calls fig.tight_layout() on figures that have a
    # colorbar, and mixing the two engines raises a RuntimeError. savefig with
    # bbox_inches="tight" already prevents label clipping in the saved PNGs.
    "figure.constrained_layout.use": False,
})

PLOT_FONT = "STIX Two Text, Cambria, Times New Roman, serif"
PLOT_COLORS = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#7A5195",
    "magenta": "#CC79A7",
    "black": "#222222",
    "gray": "#6B7280",
}
DEFAULT_GAUGE_TRIGGER_US = 250.0
DEFAULT_GAUGE_DISTANCE_M = 1.0


def _gauge_time_offsets_us(cfg: dict) -> dict[str, float]:
    c0 = max(float(cfg.get("bar_C0", 5172.0)), 1.0)
    trigger = float(cfg.get("gauge_trigger_offset_us", DEFAULT_GAUGE_TRIGGER_US))
    incident_distance = float(cfg.get("incident_gauge_distance_m", DEFAULT_GAUGE_DISTANCE_M))
    transmission_distance = float(cfg.get("transmission_gauge_distance_m", DEFAULT_GAUGE_DISTANCE_M))
    return {
        "incident": trigger,
        "reflected": trigger + 2.0 * incident_distance / c0 * 1e6,
        "transmitted": trigger + (incident_distance + transmission_distance) / c0 * 1e6,
    }


def _waveform_trace_window(t_us: np.ndarray, y: np.ndarray,
                           offset_us: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(t_us, dtype=float) + float(offset_us)
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return x, y
    peak = float(np.nanmax(np.abs(y))) if np.any(np.isfinite(y)) else 0.0
    if peak <= 0.0:
        return x, y
    threshold = max(peak * 1e-4, 1e-12)
    nonzero = np.flatnonzero(np.abs(y) >= threshold)
    if nonzero.size == 0:
        return x, y
    start = max(int(nonzero[0]) - 1, 0)
    stop = min(int(nonzero[-1]) + 2, y.size)
    return x[start:stop], y[start:stop]


def _fig_to_png(fig, dpi: int = PUB_DPI) -> bytes:
    """Render a matplotlib figure to PNG bytes at the given resolution."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def show_pub_figure(fig, caption: str, filename: str, registry: list | None = None,
                    notes: str | None = None, equations: list[str] | None = None,
                    width_px: int = 480) -> None:
    """Show a compact publication figure with explanatory notes beside it.

    The figure fills its own column (never overflows into the notes column);
    equations render in a full-width expander below so wide LaTeX never clips.

    Parameters
    ----------
    fig : matplotlib Figure
    caption : short caption shown under the image
    filename : download name; also the key under which the 600 dpi PNG is
        stored in ``registry`` for the bundled ZIP
    registry : optional list collecting ``(filename, png_bytes)`` pairs
    notes : optional markdown shown to the right of the figure
    equations : optional list of LaTeX strings rendered full width below
    width_px : unused (kept for backward compatibility)
    """
    png = _fig_to_png(fig)
    if registry is not None:
        # Store the figure plus its caption, notes and equations so the Step 6
        # document generators (report and presentation) can reuse the exact
        # same descriptions and governing equations shown here.
        registry.append((filename, png, caption, notes, list(equations or [])))
    fig_col, note_col = st.columns([1.05, 0.95], gap="large")
    with fig_col:
        st.image(png, use_container_width=True, caption=caption)
        st.download_button(
            f"Download {filename} (600 dpi)",
            data=png,
            file_name=filename,
            mime="image/png",
            key=f"dl_{filename}",
        )
    with note_col:
        if notes:
            st.markdown(notes)
    if equations:
        with st.expander("Governing equations", expanded=True):
            for eq in equations:
                st.latex(eq)
    plt.close(fig)


st.set_page_config(
    page_title="Tri-HB Integrated",
    page_icon="Tri-HB",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1280px;
        padding-top: 1.7rem;
        padding-bottom: 2.2rem;
    }
    /* When the sidebar is collapsed, use the freed space so the figures grow
       instead of leaving empty margins. Targets the collapsed sidebar state. */
    .stApp:has(section[data-testid="stSidebar"][aria-expanded="false"]) .block-container,
    .stApp:has(section[data-testid="stSidebar"][aria-collapsed="true"]) .block-container {
        max-width: 1800px;
    }
    h1 {
        font-size: clamp(1.75rem, 2.1vw, 2.45rem) !important;
        line-height: 1.12 !important;
        letter-spacing: 0 !important;
        margin-bottom: 0.35rem !important;
    }
    h2 {
        font-size: clamp(1.25rem, 1.45vw, 1.55rem) !important;
        line-height: 1.2 !important;
        letter-spacing: 0 !important;
        margin-top: 1.05rem !important;
        margin-bottom: 0.55rem !important;
    }
    h3 {
        font-size: 1.05rem !important;
        line-height: 1.25 !important;
        letter-spacing: 0 !important;
    }
    p, li, label, div[data-testid="stMarkdownContainer"] {
        font-size: 0.94rem;
        line-height: 1.48;
    }
    div[data-testid="stCaptionContainer"] {
        font-size: 0.84rem !important;
        line-height: 1.42 !important;
    }
    div[data-testid="stMetric"] {
        padding: 0.2rem 0;
    }
    div[data-testid="stMetricLabel"] {
        color: #a8afbb !important;
        font-size: 0.74rem !important;
        line-height: 1.12 !important;
        letter-spacing: 0 !important;
        text-transform: none !important;
        white-space: normal !important;
    }
    div[data-testid="stMetricValue"] {
        font-family: "JetBrains Mono", Consolas, monospace;
        font-size: clamp(1.28rem, 1.7vw, 1.95rem) !important;
        line-height: 1.05 !important;
        letter-spacing: 0 !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.72rem !important;
        line-height: 1.1 !important;
    }
    div[data-testid="stAlert"] {
        padding: 0.55rem 0.75rem;
    }
    div[data-testid="stAlert"] p {
        font-size: 0.88rem;
        line-height: 1.42;
    }
    button[role="tab"] {
        font-size: 0.86rem !important;
        padding: 0.55rem 0.75rem !important;
    }
    section[data-testid="stSidebar"]:not([aria-expanded="false"]):not([aria-collapsed="true"]) {
        width: 20rem !important;
        min-width: 20rem !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-size: 1.02rem !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
        font-size: 0.88rem !important;
        line-height: 1.42;
    }
    @media (max-width: 1100px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.35rem !important;
        }
        section[data-testid="stSidebar"]:not([aria-expanded="false"]):not([aria-collapsed="true"]) {
            width: 18.5rem !important;
            min-width: 18.5rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def publication_plot_layout(x_title: str, y_title: str, height: int = 420, right_title: str | None = None) -> dict:
    layout = dict(
        template="plotly_white",
        height=height,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#222222", family=PLOT_FONT, size=13),
        xaxis=dict(
            title=dict(text=x_title, font=dict(color="#222222", family=PLOT_FONT, size=14)),
            tickfont=dict(color="#222222", family=PLOT_FONT, size=12),
            gridcolor="#E5E7EB",
            zeroline=False,
            showline=True,
            linecolor="#222222",
            ticks="outside",
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(color="#222222", family=PLOT_FONT, size=14)),
            tickfont=dict(color="#222222", family=PLOT_FONT, size=12),
            gridcolor="#E5E7EB",
            zeroline=False,
            showline=True,
            linecolor="#222222",
            ticks="outside",
        ),
        margin=dict(l=66, r=30 if right_title is None else 66, t=24, b=58),
        legend=dict(
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#D1D5DB",
            borderwidth=1,
            font=dict(color="#222222", family=PLOT_FONT, size=12),
        ),
    )
    if right_title is not None:
        layout["yaxis2"] = dict(
            title=dict(text=right_title, font=dict(color="#222222", family=PLOT_FONT, size=14)),
            tickfont=dict(color="#222222", family=PLOT_FONT, size=12),
            overlaying="y",
            side="right",
            gridcolor="#FFFFFF",
            zeroline=False,
            showline=True,
            linecolor="#222222",
            ticks="outside",
        )
    return layout


def strip_page_config(source: str) -> str:
    """Remove st.set_page_config blocks from legacy Streamlit scripts."""
    lines = source.splitlines()
    kept: list[str] = []
    skipping = False
    depth = 0

    for line in lines:
        if not skipping and "st.set_page_config(" in line:
            skipping = True
            depth = line.count("(") - line.count(")")
            if depth <= 0:
                skipping = False
            continue

        if skipping:
            depth += line.count("(") - line.count(")")
            if depth <= 0:
                skipping = False
            continue

        kept.append(line)

    return "\n".join(kept)


def run_legacy_app(filename: str, extra_globals: dict | None = None) -> None:
    """Execute an existing app inside this integrated shell."""
    path = APP_DIR / filename
    source = strip_page_config(path.read_text(encoding="utf-8"))
    globals_dict = {
        "__file__": str(path),
        "__name__": "__main__",
    }
    if extra_globals:
        globals_dict.update(extra_globals)
    exec(compile(source, str(path), "exec"), globals_dict)


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=float)
    if len(y) > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def close_energy_budget(
    incident: np.ndarray,
    reflected_target: np.ndarray,
    transmitted_target: np.ndarray,
    absorbed_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create monotonic reflected/transmitted/absorbed histories that do not exceed incident energy."""
    reflected = np.zeros_like(incident, dtype=float)
    transmitted = np.zeros_like(incident, dtype=float)
    absorbed = np.zeros_like(incident, dtype=float)

    for i in range(1, len(incident)):
        reflected[i] = reflected[i - 1]
        transmitted[i] = transmitted[i - 1]
        absorbed[i] = absorbed[i - 1]

        remaining = max(float(incident[i] - reflected[i] - transmitted[i] - absorbed[i]), 0.0)

        for target, history in (
            (absorbed_target, absorbed),
            (reflected_target, reflected),
            (transmitted_target, transmitted),
        ):
            requested = max(float(target[i] - history[i]), 0.0)
            addition = min(requested, remaining)
            history[i] += addition
            remaining -= addition

    return reflected, transmitted, absorbed


def choose_column(label: str, columns: Iterable[str], key: str, optional: bool = False) -> str | None:
    options = list(columns)
    if optional:
        options = ["None"] + options
    selected = st.selectbox(label, options, key=key)
    if selected == "None":
        return None
    return selected


def experimental_analysis_page() -> None:
    st.markdown("## Experimental Data Analysis")
    st.caption(
        "Reduce bar-gauge data into stress, strain, strain-rate and energy histories, "
        "or import already reduced stress-strain data for comparison."
    )

    with st.expander("Equations used by the reducer", expanded=False):
        st.markdown("**Symbols.** "
                    r"$E_b$ bar Young's modulus, $C_0=\sqrt{E_b/\rho_b}$ bar wave speed, "
                    r"$A_b$ bar cross-section, $A_s$, $L_s$ specimen area and length, "
                    r"$\varepsilon_I,\varepsilon_R,\varepsilon_T$ incident / reflected / transmitted bar strains.")

        st.markdown("**Specimen stress (three-wave average):**")
        st.latex(r"\sigma_s(t)=\frac{E_b A_b}{2 A_s}\,\bigl[\varepsilon_I(t)+\varepsilon_R(t)+\varepsilon_T(t)\bigr]")

        st.markdown("**Specimen stress (transmitted-wave only, assumes force equilibrium):**")
        st.latex(r"\sigma_s(t)=\frac{E_b A_b}{A_s}\,\varepsilon_T(t)")

        st.markdown("**Specimen strain rate and strain:**")
        st.latex(r"\dot\varepsilon_s(t)=\frac{C_0}{L_s}\,\bigl[\varepsilon_I(t)-\varepsilon_R(t)-\varepsilon_T(t)\bigr]")
        st.latex(r"\varepsilon_s(t)=\int_0^t \dot\varepsilon_s(\tau)\,\mathrm{d}\tau\quad\text{(trapezoidal)}")
        st.caption("Under perfect equilibrium $\\varepsilon_I+\\varepsilon_R=\\varepsilon_T$ the rate "
                   "simplifies to $\\dot\\varepsilon_s=-2C_0\\varepsilon_R/L_s$; this app uses the "
                   "general 3-wave form so pre-equilibrium data are still handled.")

        st.markdown("**Energies (per bar wave):**")
        st.latex(r"W_I(t)=E_b A_b C_0\!\int_0^t\!\varepsilon_I^2\,\mathrm{d}\tau,\quad "
                 r"W_R(t)=E_b A_b C_0\!\int_0^t\!\varepsilon_R^2\,\mathrm{d}\tau,\quad "
                 r"W_T(t)=E_b A_b C_0\!\int_0^t\!\varepsilon_T^2\,\mathrm{d}\tau")
        st.latex(r"W_{\rm abs}(t)=W_I(t)-W_R(t)-W_T(t)")
        st.caption("Units: $A_b\\,[\\mathrm{m}^2]\\cdot E_b\\,[\\mathrm{Pa}]\\cdot C_0\\,[\\mathrm{m/s}]"
                   "\\cdot\\int\\varepsilon^2\\,\\mathrm{d}t\\,[\\mathrm{s}]\\;=\\;\\mathrm{J}$.")

        st.markdown("**Symmetric-mode (Mode 6) energy folding** — opposing bar pair contributes both incident pulses; "
                    "both reflected bars are retained explicitly:")
        st.latex(r"W_I=E_b A_b C_0\!\int(\varepsilon_{I,+}^2+\varepsilon_{I,-}^2)\,\mathrm{d}t,\qquad "
                 r"W_R=E_b A_b C_0\!\int(\varepsilon_{R,+}^2+\varepsilon_{R,-}^2)\,\mathrm{d}t")

        st.markdown("**Absorbed energy from simulator stress–strain:**")
        st.latex(r"W_{\rm abs}(t)=\int_0^t \sigma_s(\tau)\,\dot\varepsilon_s(\tau)\,V_s\,\mathrm{d}\tau,"
                 r"\quad V_s=A_sL_s")
        st.caption("The simulator stores $\\sigma_s$ as the **total** axial stress (static pre-load + dynamic). "
                   "Reflected, transmitted, and absorbed energies are then closed against the available "
                   "incident wave energy so their cumulative sum never exceeds $W_I$; this prevents the "
                   "static-preload contribution from appearing as spurious wave energy.")

        st.markdown("**Direct stress–strain branch** — only unit conversion and rate estimate:")
        st.latex(r"\dot\varepsilon_s(t)=\frac{\mathrm{d}\varepsilon_s}{\mathrm{d}t}\quad(\text{central difference})")
        st.caption("Strain unit options: strain (1), percent ($\\div100$), microstrain ($\\times10^{-6}$). "
                   "Stress unit options: MPa (passthrough), Pa ($\\div10^6$). The compression-convention "
                   "selector applies a single global sign so the formulas above use the textbook "
                   "convention $\\varepsilon_I>0,\\;\\varepsilon_R<0,\\;\\varepsilon_T>0$ for a compression test.")

    latest_result = st.session_state.get("tri_hb_latest_result")
    latest_config = st.session_state.get("tri_hb_latest_config", {})
    default_bar_E_GPa = float(latest_config.get("bar_E", 210e9)) / 1e9
    default_bar_C0 = float(latest_config.get("bar_C0", 5172.0))
    default_bar_area_m2 = float(latest_config.get("bar_area", 0.050 * 0.050))
    default_specimen_size_mm = float(latest_config.get("specimen_size", 0.050)) * 1000.0
    default_specimen_length_mm = float(latest_config.get("specimen_length", latest_config.get("specimen_size", 0.050))) * 1000.0
    default_specimen_area_m2 = float(latest_config.get(
        "specimen_area",
        (default_specimen_size_mm * 1e-3) ** 2,
    ))

    # Defaults used by branches where duplicate geometry controls are hidden.
    bar_E_GPa = default_bar_E_GPa
    bar_C0 = default_bar_C0
    bar_d_mm = float(np.sqrt(4.0 * default_bar_area_m2 / np.pi) * 1000.0)
    square_bar = bool(np.isclose(default_bar_area_m2, 0.050 * 0.050, rtol=0.0, atol=1e-12))
    specimen_side_mm = default_specimen_size_mm
    specimen_length_mm = default_specimen_length_mm
    specimen_shape = "Square/cube"

    default_square_specimen_area = (default_specimen_size_mm * 1e-3) ** 2
    default_circular_specimen_area = np.pi * (default_specimen_size_mm * 1e-3 / 2.0) ** 2
    default_specimen_shape_index = 0
    if abs(default_specimen_area_m2 - default_circular_specimen_area) < abs(default_specimen_area_m2 - default_square_specimen_area):
        default_specimen_shape_index = 1

    with st.sidebar:
        st.header("Experimental analysis")
        source_options = []
        if latest_result is not None:
            source_options.append("Latest simulator result")
        source_options.extend(["Upload bar strain gauges", "Upload direct stress-strain"])
        data_source = st.radio(
            "Data source",
            source_options,
            help="Choose the source first; the column mapping and energy workflow update to match it.",
        )

        if data_source == "Upload bar strain gauges":
            st.divider()
            st.subheader("Specimen and bars")
            st.caption("Required only for reducing uploaded bar-gauge histories.")
            bar_E_GPa = st.number_input("Bar Young's modulus, Eb (GPa)", value=default_bar_E_GPa, min_value=1.0, step=5.0)
            bar_C0 = st.number_input("Bar wave speed, C0 (m/s)", value=default_bar_C0, min_value=1000.0, step=50.0)
            bar_d_mm = st.number_input("Round bar diameter (mm)", value=bar_d_mm, min_value=1.0, step=1.0)
            square_bar = st.checkbox("Use square 50 x 50 mm bar area", value=square_bar)
            specimen_side_mm = st.number_input("Specimen side/diameter (mm)", value=default_specimen_size_mm, min_value=1.0, step=1.0)
            specimen_length_mm = st.number_input("Specimen length (mm)", value=default_specimen_length_mm, min_value=1.0, step=1.0)
            specimen_shape = st.radio(
                "Specimen area",
                ["Square/cube", "Circular cylinder"],
                horizontal=True,
                index=default_specimen_shape_index,
            )
        elif data_source == "Latest simulator result":
            st.divider()
            st.caption(
                "The analysis inherits the bar, specimen geometry, and material settings from the Step 1 test setup. "
                "Duplicate Specimen and bars inputs are hidden."
            )
        else:
            st.divider()
            st.caption(
                "Direct stress-strain files already contain reduced stress and strain, so bar/specimen inputs are not required."
            )

    use_simulator_result = data_source == "Latest simulator result"
    if data_source == "Upload bar strain gauges":
        analysis_mode = "Bar strain gauges"
    elif data_source == "Upload direct stress-strain":
        analysis_mode = "Direct stress-strain"
    else:
        analysis_mode = "Simulator result"

    uploaded = st.file_uploader(
        "Upload CSV or Excel data",
        type=["csv", "xlsx", "xls"],
        help="Expected columns include time and incident/reflected/transmitted strains, or direct stress and strain.",
        disabled=use_simulator_result,
    )

    if use_simulator_result and latest_result is not None:
        time_s = latest_result["time"]
        eps_i_pos = latest_result["epsI_x_pos"]
        eps_i_neg = latest_result.get("epsI_x_neg", np.zeros_like(eps_i_pos))
        eps_r_pos = latest_result.get("epsR_x_pos", np.zeros_like(eps_i_pos))
        eps_r_neg = latest_result.get("epsR_x_neg", np.zeros_like(eps_i_pos))
        eps_t = latest_result.get("epsT_x", np.zeros_like(eps_i_pos))

        sim_cfg = st.session_state.get("tri_hb_latest_config", latest_config)
        Ab = float(sim_cfg.get("bar_area", default_bar_area_m2))
        Eb = float(sim_cfg.get("bar_E", default_bar_E_GPa * 1e9))
        C0 = float(sim_cfg.get("bar_C0", default_bar_C0))
        As = float(sim_cfg.get("specimen_area", default_specimen_area_m2))
        Ls = float(sim_cfg.get("specimen_length", default_specimen_length_mm * 1e-3))
        sim_volume = As * Ls

        energy_i = Ab * Eb * C0 * cumulative_trapezoid(eps_i_pos**2 + eps_i_neg**2, time_s)
        energy_r_target = Ab * Eb * C0 * cumulative_trapezoid(eps_r_pos**2 + eps_r_neg**2, time_s)
        energy_t_target = Ab * Eb * C0 * cumulative_trapezoid(eps_t**2, time_s)
        absorbed_target = cumulative_trapezoid(latest_result["sig_x"] * latest_result["rate_x"] * sim_volume, time_s)
        energy_r, energy_t, absorbed = close_energy_budget(
            energy_i,
            energy_r_target,
            energy_t_target,
            absorbed_target,
        )
        out = pd.DataFrame(
            {
                "time_us": time_s * 1e6,
                "stress_MPa": latest_result["sig_x"] / 1e6,
                "strain": latest_result["eps_x"],
                "strain_rate_s-1": latest_result["rate_x"],
                "energy_incident_J": energy_i,
                "energy_reflected_J": energy_r,
                "energy_transmitted_J": energy_t,
                "energy_absorbed_J": absorbed,
            }
        )
        st.success(
            "Using the latest result from Test Design and Simulator. "
            "Peak stress and strain are read from the same Step 1 arrays."
        )
        df_raw = out.copy()
    elif uploaded is None:
        if analysis_mode == "Bar strain gauges":
            st.info("Upload bar-gauge data with time, incident strain, reflected strain, and transmitted strain columns.")
            example = pd.DataFrame(
                {
                    "time_us": np.linspace(0, 250, 6),
                    "eps_I_ue": [0, 50, 250, 120, 20, 0],
                    "eps_R_ue": [0, -10, -80, -40, -5, 0],
                    "eps_T_ue": [0, 20, 160, 90, 10, 0],
                }
            )
        else:
            st.info("Upload reduced stress-strain data with time, stress, and strain columns.")
            example = pd.DataFrame(
                {
                    "time_us": np.linspace(0, 250, 6),
                    "stress_MPa": [0, 35, 120, 165, 130, 80],
                    "strain_percent": [0, 0.08, 0.28, 0.52, 0.78, 0.95],
                }
            )
        st.dataframe(example, width="stretch")
        return
    else:
        df_raw = read_uploaded_table(uploaded)
        df_raw = df_raw.dropna(how="all")
        st.dataframe(df_raw.head(20), width="stretch")

        if df_raw.empty:
            st.error("The uploaded file did not contain any readable rows.")
            return

    if analysis_mode == "Simulator result":
        out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_us", "stress_MPa", "strain"])
        columns = list(df_raw.columns)
        col_map, col_plot = st.columns([0.9, 1.1])
        with col_map:
            st.subheader("Data source")
            st.write("Latest simulated axial stress-strain response.")
    else:
        columns = list(df_raw.columns)
        col_map, col_plot = st.columns([0.9, 1.1])

        with col_map:
            st.subheader("Column mapping")
            time_col = choose_column("Time column", columns, "exp_time")
            time_unit = st.selectbox("Time unit", ["microsecond", "second", "millisecond"], key="exp_time_unit")
            compression_sign = st.radio(
                "Compression convention",
                ["Positive compression", "Negative compression"],
                horizontal=True,
                key="compression_sign",
            )

            if analysis_mode == "Bar strain gauges":
                eps_i_col = choose_column("Incident strain column", columns, "eps_i")
                eps_r_col = choose_column("Reflected strain column", columns, "eps_r")
                eps_t_col = choose_column("Transmitted strain column", columns, "eps_t")
                strain_unit = st.selectbox("Gauge strain unit", ["microstrain", "strain"], key="strain_unit")
                reduction = st.selectbox(
                    "Stress reduction",
                    ["Three-wave average", "Transmitted-wave only"],
                    key="reduction_mode",
                )
            else:
                strain_col = choose_column("Strain column", columns, "direct_strain")
                stress_col = choose_column("Stress column", columns, "direct_stress")
                stress_unit = st.selectbox("Stress unit", ["MPa", "Pa"], key="direct_stress_unit")
                strain_unit = st.selectbox("Strain unit", ["strain", "percent", "microstrain"], key="direct_strain_unit")

        time = pd.to_numeric(df_raw[time_col], errors="coerce").to_numpy(dtype=float)
        if time_unit == "microsecond":
            time_s = time * 1e-6
            time_us = time
        elif time_unit == "millisecond":
            time_s = time * 1e-3
            time_us = time * 1e3
        else:
            time_s = time
            time_us = time * 1e6

        order = np.argsort(time_s)
        time_s = time_s[order]
        time_us = time_us[order]

        sign = 1.0 if compression_sign == "Positive compression" else -1.0
        Eb = bar_E_GPa * 1e9
        Ab = (0.050 * 0.050) if square_bar else np.pi * (bar_d_mm * 1e-3 / 2.0) ** 2
        if specimen_shape == "Square/cube":
            As = (specimen_side_mm * 1e-3) ** 2
        else:
            As = np.pi * (specimen_side_mm * 1e-3 / 2.0) ** 2
        Ls = specimen_length_mm * 1e-3

        if analysis_mode == "Bar strain gauges":
            scale = 1e-6 if strain_unit == "microstrain" else 1.0
            eps_i = sign * pd.to_numeric(df_raw[eps_i_col], errors="coerce").to_numpy(dtype=float)[order] * scale
            eps_r = sign * pd.to_numeric(df_raw[eps_r_col], errors="coerce").to_numpy(dtype=float)[order] * scale
            eps_t = sign * pd.to_numeric(df_raw[eps_t_col], errors="coerce").to_numpy(dtype=float)[order] * scale

            if reduction == "Three-wave average":
                stress_pa = Eb * Ab / (2.0 * As) * (eps_i + eps_r + eps_t)
            else:
                stress_pa = Eb * Ab / As * eps_t

            strain_rate = bar_C0 / Ls * (eps_i - eps_r - eps_t)
            strain = cumulative_trapezoid(strain_rate, time_s)

            energy_i = Ab * Eb * bar_C0 * cumulative_trapezoid(eps_i**2, time_s)
            energy_r = Ab * Eb * bar_C0 * cumulative_trapezoid(eps_r**2, time_s)
            energy_t = Ab * Eb * bar_C0 * cumulative_trapezoid(eps_t**2, time_s)
            absorbed = energy_i - energy_r - energy_t

            out = pd.DataFrame(
                {
                    "time_us": time_us,
                    "eps_incident": eps_i,
                    "eps_reflected": eps_r,
                    "eps_transmitted": eps_t,
                    "stress_MPa": stress_pa / 1e6,
                    "strain": strain,
                    "strain_rate_s-1": strain_rate,
                    "energy_incident_J": energy_i,
                    "energy_reflected_J": energy_r,
                    "energy_transmitted_J": energy_t,
                    "energy_absorbed_J": absorbed,
                }
            )
        else:
            raw_strain = sign * pd.to_numeric(df_raw[strain_col], errors="coerce").to_numpy(dtype=float)[order]
            if strain_unit == "percent":
                strain = raw_strain / 100.0
            elif strain_unit == "microstrain":
                strain = raw_strain * 1e-6
            else:
                strain = raw_strain
            raw_stress = sign * pd.to_numeric(df_raw[stress_col], errors="coerce").to_numpy(dtype=float)[order]
            stress_mpa = raw_stress if stress_unit == "MPa" else raw_stress / 1e6
            strain_rate = np.gradient(strain, time_s)
            out = pd.DataFrame(
                {
                    "time_us": time_us,
                    "stress_MPa": stress_mpa,
                    "strain": strain,
                    "strain_rate_s-1": strain_rate,
                }
            )

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_us", "stress_MPa", "strain"])
    st.session_state["tri_hb_reduced_data"] = out
    st.session_state["tri_hb_reduced_source"] = analysis_mode

    if out.empty:
        st.error("No valid reduced rows were produced. Check the selected columns and units.")
        return

    with col_plot:
        st.subheader("Reduced curves")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Peak stress", f"{out['stress_MPa'].max():.1f} MPa")
        m2.metric("Peak strain", f"{100.0 * out['strain'].max():.3f} %")
        m3.metric("Peak strain rate", f"{out['strain_rate_s-1'].abs().max():.0f} /s")
        if "energy_absorbed_J" in out:
            m4.metric("Absorbed energy", f"{out['energy_absorbed_J'].iloc[-1]:.2f} J")
        else:
            m4.metric("Rows", f"{len(out):,}")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=100.0 * out["strain"],
                y=out["stress_MPa"],
                mode="lines",
                name="σ-ε",
                line=dict(color=PLOT_COLORS["blue"], width=2.6),
            )
        )
        fig.update_layout(**publication_plot_layout("Strain, ε (%)", "Stress, σ (MPa)", height=390))
        st.plotly_chart(fig, width="stretch")

    tabs = st.tabs(["Time histories", "Energy", "Export"])
    with tabs[0]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=out["time_us"],
            y=out["stress_MPa"],
            name="σ(t)",
            line=dict(color=PLOT_COLORS["blue"], width=2.2),
        ))
        fig.add_trace(go.Scatter(
            x=out["time_us"],
            y=out["strain_rate_s-1"],
            name="ε̇(t)",
            yaxis="y2",
            line=dict(color=PLOT_COLORS["vermillion"], width=2.0, dash="dash"),
        ))
        fig.update_layout(**publication_plot_layout("Time, t (μs)", "Stress, σ (MPa)", height=430, right_title="Strain rate, ε̇ (s⁻¹)"))
        st.plotly_chart(fig, width="stretch")

    with tabs[1]:
        if "energy_absorbed_J" not in out:
            st.info("Energy histories require incident, reflected and transmitted bar strain gauges.")
        else:
            if analysis_mode == "Simulator result":
                st.caption(
                    "Simulator energy histories use a closed budget: reflected, transmitted, and absorbed "
                    "energy are constrained by the available incident energy."
                )
            else:
                transmitted_excess = float(np.nanmax(
                    out["energy_transmitted_J"] - out["energy_incident_J"]
                ))
                reflected_excess = float(np.nanmax(
                    out["energy_reflected_J"] - out["energy_incident_J"]
                ))
                absorbed_min = float(np.nanmin(out["energy_absorbed_J"]))
                issues: list[str] = []
                if transmitted_excess > 1e-9:
                    issues.append(f"transmitted exceeds incident by up to {transmitted_excess:.3g} J")
                if reflected_excess > 1e-9:
                    issues.append(f"reflected exceeds incident by up to {reflected_excess:.3g} J")
                if absorbed_min < -1e-9:
                    issues.append(
                        f"absorbed = W_I - W_R - W_T goes negative (min {absorbed_min:.3g} J), "
                        "which means W_R + W_T > W_I"
                    )
                if issues:
                    st.warning(
                        "Energy balance is inconsistent — "
                        + "; ".join(issues)
                        + ". Check gauge signs, units, bar area, wave-speed setting, "
                        "and time alignment of the incident/reflected/transmitted windows."
                    )
            fig = go.Figure()
            for col, label, color in [
                ("energy_incident_J", "W<sub>I</sub> incident", PLOT_COLORS["blue"]),
                ("energy_reflected_J", "W<sub>R</sub> reflected", PLOT_COLORS["vermillion"]),
                ("energy_transmitted_J", "W<sub>T</sub> transmitted", PLOT_COLORS["green"]),
                ("energy_absorbed_J", "W<sub>abs</sub> absorbed", PLOT_COLORS["black"]),
            ]:
                fig.add_trace(go.Scatter(x=out["time_us"], y=out[col], name=label, line=dict(color=color, width=2.2)))
            fig.update_layout(**publication_plot_layout("Time, t (μs)", "Energy, W (J)", height=430))
            st.plotly_chart(fig, width="stretch")

    with tabs[2]:
        st.dataframe(out.head(50), width="stretch")
        st.download_button(
            "Download reduced data",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="tri_hb_experimental_reduced.csv",
            mime="text/csv",
        )
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            out.to_excel(writer, index=False, sheet_name="Reduced data")
            df_raw.to_excel(writer, index=False, sheet_name="Raw upload")
        st.download_button(
            "Download workbook",
            data=buf.getvalue(),
            file_name="tri_hb_experimental_reduced.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def _overview_workflow_dot() -> str:
    """Graphviz DOT for the workspace schematic shown on the Overview page.

    Compact left-to-right flow (landscape) so it scales to one screen-height.
    Step 1 offers two input routes; Steps 6-7 are shown as a single export
    node because both are run-document generators.
    """
    return r"""
    digraph trihb_workflow {
        rankdir=LR;
        bgcolor="transparent";
        ranksep=0.40; nodesep=0.22;
        node [shape=box, style="rounded,filled", fontname="Helvetica",
              fontsize=10, margin="0.16,0.10", penwidth=0, fontcolor="white"];
        edge [color="#6B7280", penwidth=1.4, arrowsize=0.7];

        subgraph cluster_s1 {
            label="STEP 1 · setup";
            labelloc="t"; fontname="Helvetica"; fontsize=9; fontcolor="#505050";
            color="#c7ccd6"; style="rounded";
            sim [label="Simulator", fillcolor="#0072B2"];
            red [label="Data reducer", fillcolor="#0072B2"];
        }

        s2 [label="STEP 2\nWave model", fillcolor="#56B4E9"];
        s3 [label="STEP 3\nStress path\np · q · θ", fillcolor="#009E73"];
        s4 [label="STEP 4\nDamage &\nenergy", fillcolor="#E69F00"];
        s5 [label="STEP 5\nSummary\n(figures)", fillcolor="#D55E00"];
        s67 [label="STEP 6\nReport &\npresentation\n(LaTeX + PDF)", fillcolor="#7A5195"];

        sim -> s2;
        red -> s2;
        s2 -> s3 -> s4 -> s5 -> s67;
        { rank=same; sim; red; }
    }
    """


def _trihb_animation_html(theme: str = "light") -> str:
    """Self-contained isometric Tri-HB schematic + synchronized strain-gauge scope.

    Returns a standalone HTML fragment for ``st.components.v1.html``. The
    component renders inside a sandboxed iframe that does NOT inherit Streamlit's
    theme, so the palette is injected here as CSS custom properties on the root
    div (``--t`` text, ``--t2/t3`` muted, ``--bd/bd2`` borders, ``--pbg`` panel
    fill). The drawing scales to the container width but is capped so it stays
    roughly one screen tall on a wide page.
    """
    if str(theme).lower() == "dark":
        pal = ("--t:#e9e9e6;--t2:#b4b2a9;--t3:#83827c;"
               "--bd:#444441;--bd2:#5f5e5a;--pbg:#1f1f1d;")
    else:
        pal = ("--t:#2c2c2a;--t2:#5f5e5a;--t3:#9a988f;"
               "--bd:#cdccc6;--bd2:#b4b2a9;--pbg:#ffffff;")
    btn = ("border:1px solid var(--bd2);background:var(--pbg);color:var(--t);"
           "border-radius:8px;padding:6px 14px;cursor:pointer;"
           "font-family:system-ui,sans-serif;font-size:14px;")
    return (
"<div style=\"" + pal + "max-width:780px;margin:0 auto;font-family:system-ui,sans-serif;\">"
"<div style=\"display:flex;flex-wrap:wrap;gap:10px 14px;align-items:center;margin-bottom:8px;\">"
"<button id=\"playBtn\" style=\"" + btn + "\">&#9654; Play</button>"
"<button id=\"resetBtn\" style=\"" + btn + "\">&#8635; Reset</button>"
"<label style=\"display:flex;align-items:center;gap:8px;font-size:13px;color:var(--t2);\">striker velocity"
"<input id=\"vel\" type=\"range\" min=\"10\" max=\"50\" step=\"5\" value=\"20\" style=\"width:120px;\">"
"<span id=\"velVal\" style=\"font-family:monospace;color:var(--t);min-width:48px;\">20 m/s</span></label>"
"<label style=\"display:flex;align-items:center;gap:6px;font-size:13px;color:var(--t2);cursor:pointer;user-select:none;\">"
"<input id=\"confChk\" type=\"checkbox\" checked> confinement (X/Y/Z)</label></div>"
"<div style=\"display:flex;flex-wrap:wrap;gap:6px 18px;margin-bottom:6px;font-size:12.5px;color:var(--t2);\">"
"<span>bar speed C&#8320; &asymp; <b style=\"color:var(--t);\">5200 m/s</b></span>"
"<span>pulse width 2L/C&#8320; &asymp; <b style=\"color:var(--t);\">192 &micro;s</b></span>"
"<span>incident strain &epsilon;&#8321; = v/2C&#8320; &asymp; <b id=\"epsVal\" style=\"color:var(--t);\">1923 &micro;&epsilon;</b></span></div>"
"<svg id=\"trihb\" viewBox=\"0 0 680 600\" width=\"100%\" role=\"img\" aria-label=\"Monash Tri-HB isometric schematic with synchronized strain-gauge scope\">"
"<g id=\"scene\"></g><g id=\"overlay\"></g>"
"<text id=\"phase\" x=\"345\" y=\"392\" text-anchor=\"middle\" style=\"font-size:13px;fill:var(--t);\">press play &mdash; striker impacts the incident bar</text>"
"<g id=\"plotaxes\"></g><g id=\"ghost\"></g><g id=\"live\"></g></svg></div>"
"<script>(function(){"
"var S=50,OX=235,OY=185,C30=0.8660254,S30=0.5,r=0.12,c=0.17;"
"var GRAY=['#B4B2A9','#6d6c66','#8d8c84'],DGRAY=['#9a9890','#54534e','#6e6d67'];"
"var AMBER=['#FAC775','#BA7517','#EF9F27'],BLUE=['#85B7EB','#185FA5','#378ADD'],CORAL=['#F0997B','#993C1D','#D85A30'];"
"var EDGE='#4a4944',COL_I='#D85A30',COL_R='#378ADD',COL_T='#1D9E75',COL_Y='#8E5AB8',COL_Z='#C75D93';"
"var d=document,scene=d.getElementById('scene'),overlay=d.getElementById('overlay'),phase=d.getElementById('phase');"
"var plotaxes=d.getElementById('plotaxes'),ghost=d.getElementById('ghost'),live=d.getElementById('live');"
"var playBtn=d.getElementById('playBtn'),resetBtn=d.getElementById('resetBtn'),velEl=d.getElementById('vel');"
"var velVal=d.getElementById('velVal'),epsVal=d.getElementById('epsVal'),confChk=d.getElementById('confChk');"
"function P(x,y,z){return [OX+(x-z)*C30*S,OY+(x+z)*S30*S-y*S];}"
"function fp(x,y,z){var p=P(x,y,z);return p[0].toFixed(1)+','+p[1].toFixed(1);}"
"function quad(a,b,c2,e,fill,op){return '<polygon points=\"'+a+' '+b+' '+c2+' '+e+'\" fill=\"'+fill+'\"'+(op!=null?' opacity=\"'+op+'\"':'')+' stroke=\"'+EDGE+'\" stroke-width=\"0.7\" stroke-linejoin=\"round\"/>';}"
"function boxStr(x0,y0,z0,x1,y1,z1,k){return quad(fp(x0,y1,z0),fp(x1,y1,z0),fp(x1,y1,z1),fp(x0,y1,z1),k[0])+quad(fp(x0,y0,z1),fp(x1,y0,z1),fp(x1,y1,z1),fp(x0,y1,z1),k[1])+quad(fp(x1,y0,z0),fp(x1,y0,z1),fp(x1,y1,z1),fp(x1,y1,z0),k[2]);}"
"function topband(xa,xb,fill,op){return '<polygon points=\"'+fp(xa,r,-r)+' '+fp(xb,r,-r)+' '+fp(xb,r,r)+' '+fp(xa,r,r)+'\" fill=\"'+fill+'\" opacity=\"'+op+'\"/>';}"
"function bandY(ya,yb,fill,op){return '<polygon points=\"'+fp(r,ya,-r)+' '+fp(r,yb,-r)+' '+fp(r,yb,r)+' '+fp(r,ya,r)+'\" fill=\"'+fill+'\" opacity=\"'+op+'\"/>';}"
"function bandZ(za,zb,fill,op){return '<polygon points=\"'+fp(-r,r,za)+' '+fp(r,r,za)+' '+fp(r,r,zb)+' '+fp(-r,r,zb)+'\" fill=\"'+fill+'\" opacity=\"'+op+'\"/>';}"
"var confOn=true;"
"function buildScene(t){var solids=[];"
"function add(x0,y0,z0,x1,y1,z1,k,op){var s=boxStr(x0,y0,z0,x1,y1,z1,k);if(op!=null)s='<g opacity=\"'+op+'\">'+s+'</g>';solids.push({k:x0+x1+y0+y1+z0+z1,s:s});}"
"var sShift=(t<0.18)?(t/0.18)*0.55:0.55,sx0=-2.95+sShift;add(sx0,-r,-r,sx0+0.36,r,r,CORAL);"
"add(-2.25,-r,-r,-c,r,r,GRAY);add(c,-r,-r,1.75,r,r,GRAY);add(1.75,-r*0.8,-r*0.8,2.05,r*0.8,r*0.8,DGRAY);"
"var dim=confOn?null:0.16,rc=r*1.5;"
"add(-r,c,-r,r,1.25,r,GRAY,dim);add(-r,-1.25,-r,r,-c,r,GRAY,dim);add(-r,-r,c,r,r,1.25,GRAY,dim);add(-r,-r,-1.25,r,r,-c,GRAY,dim);"
"add(-rc,1.25,-rc,rc,1.55,rc,BLUE,dim);add(-rc,-1.55,-rc,rc,-1.25,rc,BLUE,dim);add(-rc,-rc,1.25,rc,rc,1.55,BLUE,dim);add(-rc,-rc,-1.55,rc,rc,-1.25,BLUE,dim);"
"add(2.05,-rc,-rc,2.4,rc,rc,BLUE,dim);"
"var comp=0;if(t>=0.5){var fr=(t-0.5)/0.36;comp=Math.sin(Math.min(fr,1)*Math.PI);}var cx=c*(1-0.32*comp),cl=c*(1+0.20*comp);add(-cx,-cl,-cl,cx,cl,cl,AMBER);"
"solids.sort(function(a,b){return a.k-b.k;});var html=solids.map(function(o){return o.s;}).join('');"
"if(t>=0.18&&t<0.5){var k=(t-0.18)/0.32,x=-2.1+k*((-c+0.05)-(-2.1));html+=topband(x,x+0.22,COL_I,0.95);}"
"if(t>=0.5){var k2=(t-0.5)/0.36;var xr=(-c)-k2*(2.1-c);html+=topband(xr-0.22,xr,COL_R,0.92);var xt=c+k2*1.55;html+=topband(xt,xt+0.22,COL_T,0.92);}"
"if(confOn&&t>=0.56){var ko=Math.min((t-0.56)/0.36,1),seg=0.2,reach=1.2-c;var yT=c+ko*reach;html+=bandY(yT-seg,yT,COL_Y,0.9);var yB=-c-ko*reach;html+=bandY(yB,yB+seg,COL_Y,0.9);var zB=c+ko*reach;html+=bandZ(zB-seg,zB,COL_Z,0.9);var zF=-c-ko*reach;html+=bandZ(zF,zF+seg,COL_Z,0.9);}"
"scene.innerHTML=html;phase.textContent=phaseText(t);}"
"function phaseText(t){if(t<0.18)return 'striker accelerates toward the incident bar';if(t<0.5)return 'incident pulse travels down the incident bar (C0 \\u2248 5200 m/s)';if(t<0.62)return 'at the specimen: part reflects (\\u2190 incident), part transmits (\\u2192 transmission)';return 'Poisson expansion drives output waves outward along the Y and Z bars (Eqs. 4\\u20137)';}"
"function badge(x,y,z,n){var p=P(x,y,z);return '<circle cx=\"'+p[0].toFixed(1)+'\" cy=\"'+p[1].toFixed(1)+'\" r=\"9\" fill=\"var(--pbg)\" stroke=\"var(--bd2)\" stroke-width=\"1\"/><text x=\"'+p[0].toFixed(1)+'\" y=\"'+(p[1]+3.5).toFixed(1)+'\" text-anchor=\"middle\" style=\"font-size:11px;fill:var(--t);\">'+n+'</text>';}"
"function gauge(x,lbl){return gmark(P(x,r,0),lbl);}"
"function gmark(p,lbl){return '<circle cx=\"'+p[0].toFixed(1)+'\" cy=\"'+p[1].toFixed(1)+'\" r=\"3.4\" fill=\"#444441\"/><text x=\"'+p[0].toFixed(1)+'\" y=\"'+(p[1]-7).toFixed(1)+'\" text-anchor=\"middle\" style=\"font-size:10px;fill:var(--t2);\">'+lbl+'</text>';}"
"function buildOverlay(){var h='';h+=badge(-2.7,0,0,'1')+badge(-1.45,0,0,'2')+badge(0,c+0.32,0,'3')+badge(0.95,0,0,'4')+badge(1.95,0,0,'5')+badge(0,1.4,0,'6')+badge(0,0,1.4,'7');h+=gauge(-1.2,'G1')+gauge(0.8,'G2')+gmark(P(r,0.7,0),'G3')+gmark(P(r,-0.7,0),'G4')+gmark(P(0,r,0.7),'G5')+gmark(P(0,r,-0.7),'G6');"
"var Tx=70,Ty=350;function arr(dx,dy,lab){return '<line x1=\"'+Tx+'\" y1=\"'+Ty+'\" x2=\"'+(Tx+dx)+'\" y2=\"'+(Ty+dy)+'\" stroke=\"var(--t3)\" stroke-width=\"1.4\"/><text x=\"'+(Tx+dx*1.25)+'\" y=\"'+(Ty+dy*1.25+3)+'\" text-anchor=\"middle\" style=\"font-size:11px;fill:var(--t2);\">'+lab+'</text>';}"
"h+=arr(C30*S*0.55,S30*S*0.55,'X')+arr(0,-S*0.55,'Y')+arr(-C30*S*0.55,S30*S*0.55,'Z');"
"var items=['Striker (\\u00d840 mm, 42CrMo)','Incident bar (2.5 m)','Specimen \\u2014 6 interfaces','Transmission bar (2 m)','Absorption, trap + X load cyl.','Y bars + confinement','Z bars + confinement'];"
"var lx=448,ly=60;for(var i=0;i<items.length;i++){var yy=ly+i*23;h+='<circle cx=\"'+lx+'\" cy=\"'+(yy-4)+'\" r=\"8.5\" fill=\"var(--pbg)\" stroke=\"var(--bd2)\" stroke-width=\"1\"/><text x=\"'+lx+'\" y=\"'+(yy-0.5)+'\" text-anchor=\"middle\" style=\"font-size:11px;fill:var(--t);\">'+(i+1)+'</text><text x=\"'+(lx+15)+'\" y=\"'+(yy-0.5)+'\" style=\"font-size:12px;fill:var(--t);\">'+items[i]+'</text>';}overlay.innerHTML=h;}"
"var N=240,EI=[],ER=[],ET=[],EY=[],EZ=[];function bump(tau,cc,w,a){var dd=(tau-cc)/w;if(Math.abs(dd)>=0.5)return 0;return a*Math.cos(dd*Math.PI);}"
"function outpulse(tau,cc,w,a){return bump(tau,cc,w,a)-bump(tau,cc+1.4*w,0.6*w,0.16*a);}"
"for(var i=0;i<=N;i++){var tau=i/N;EI.push(bump(tau,0.33,0.085,1.0));ER.push(bump(tau,0.69,0.11,-0.6));ET.push(bump(tau,0.64,0.12,0.45));EY.push(outpulse(tau,0.72,0.10,0.26));EZ.push(outpulse(tau,0.75,0.10,0.28));}"
"var PX0=70,PX1=620,YB=505,AMP=50;function PXt(tau){return PX0+tau*(PX1-PX0);}function PYt(v){return YB-v*AMP;}"
"function pl(arr,upto,color,w,op){var pts='';for(var j=0;j<=upto&&j<arr.length;j++){pts+=(j?' ':'')+PXt(j/N).toFixed(1)+','+PYt(arr[j]).toFixed(1);}return '<polyline points=\"'+pts+'\" fill=\"none\" stroke=\"'+color+'\" stroke-width=\"'+w+'\" opacity=\"'+op+'\" stroke-linejoin=\"round\"/>';}"
"function buildAxes(){var h='';h+='<line x1=\"'+PX0+'\" y1=\"'+YB+'\" x2=\"'+PX1+'\" y2=\"'+YB+'\" stroke=\"var(--bd)\" stroke-width=\"1\"/>';"
"h+='<line x1=\"'+PX0+'\" y1=\"'+(YB-AMP-12)+'\" x2=\"'+PX0+'\" y2=\"'+(YB+AMP+12)+'\" stroke=\"var(--bd)\" stroke-width=\"1\"/>';"
"h+='<text x=\"'+PX0+'\" y=\"'+(YB-AMP-18)+'\" text-anchor=\"middle\" style=\"font-size:11px;fill:var(--t3);\">strain</text>';"
"h+='<text x=\"'+(PX1-2)+'\" y=\"'+(YB+22)+'\" text-anchor=\"end\" style=\"font-size:11px;fill:var(--t3);\">time \\u2192</text>';"
"h+='<text x=\"'+(PX0-6)+'\" y=\"'+(YB-AMP+2)+'\" text-anchor=\"end\" style=\"font-size:10px;fill:var(--t3);\">+ comp.</text>';"
"h+='<text x=\"'+(PX0-6)+'\" y=\"'+(YB+AMP+2)+'\" text-anchor=\"end\" style=\"font-size:10px;fill:var(--t3);\">\\u2212 tens.</text>';"
"h+='<text x=\"345\" y=\"418\" text-anchor=\"middle\" style=\"font-size:12.5px;fill:var(--t);\">synchronized strain-gauge signals</text>';"
"var lg=[['incident',COL_I],['reflected',COL_R],['transmitted',COL_T],['Y output',COL_Y],['Z output',COL_Z]];var st=[95,100,108,92,92],gx=120;for(var m=0;m<lg.length;m++){h+='<rect x=\"'+gx+'\" y=\"430\" width=\"13\" height=\"11\" rx=\"2\" fill=\"'+lg[m][1]+'\"/><text x=\"'+(gx+18)+'\" y=\"439.5\" style=\"font-size:11px;fill:var(--t2);\">'+lg[m][0]+'</text>';gx+=st[m];}"
"plotaxes.innerHTML=h;ghost.innerHTML=pl(EI,N,COL_I,1.2,0.22)+pl(ER,N,COL_R,1.2,0.22)+pl(ET,N,COL_T,1.2,0.22)+pl(EY,N,COL_Y,1.2,0.22)+pl(EZ,N,COL_Z,1.2,0.22);}"
"function dot(arr,idx,color){return '<circle cx=\"'+PXt(idx/N).toFixed(1)+'\" cy=\"'+PYt(arr[Math.min(idx,N)]).toFixed(1)+'\" r=\"3.3\" fill=\"'+color+'\"/>';}"
"function updatePlot(t){var idx=Math.min(N,Math.floor(t*N));var h=pl(EI,idx,COL_I,2.4,1)+pl(ER,idx,COL_R,2.4,1)+pl(ET,idx,COL_T,2.4,1)+pl(EY,idx,COL_Y,2.4,1)+pl(EZ,idx,COL_Z,2.4,1);h+=dot(EI,idx,COL_I)+dot(ER,idx,COL_R)+dot(ET,idx,COL_T)+dot(EY,idx,COL_Y)+dot(EZ,idx,COL_Z);"
"var cx=PXt(t).toFixed(1);h+='<line x1=\"'+cx+'\" y1=\"'+(YB-AMP-10)+'\" x2=\"'+cx+'\" y2=\"'+(YB+AMP+10)+'\" stroke=\"var(--t3)\" stroke-width=\"1\" stroke-dasharray=\"3 3\"/>';live.innerHTML=h;}"
"var t=0,playing=false,raf=null,vel=20;"
"function render(){buildScene(t);updatePlot(t);}"
"function frame(){t+=0.0016*(vel/20);if(t>1)t=0;render();if(playing)raf=requestAnimationFrame(frame);}"
"function setPlay(p){playing=p;playBtn.innerHTML=p?'\\u23f8 Pause':'\\u25b6 Play';if(p)raf=requestAnimationFrame(frame);else cancelAnimationFrame(raf);}"
"playBtn.addEventListener('click',function(){setPlay(!playing);});"
"resetBtn.addEventListener('click',function(){cancelAnimationFrame(raf);playing=false;playBtn.innerHTML='\\u25b6 Play';t=0;render();});"
"velEl.addEventListener('input',function(){vel=+velEl.value;velVal.textContent=vel+' m/s';epsVal.innerHTML=Math.round(vel/(2*5200)*1e6)+' \\u00b5\\u03b5';});"
"confChk.addEventListener('change',function(){confOn=confChk.checked;if(!playing)buildScene(t);});"
"buildOverlay();buildAxes();render();"
"})();</script>"
    )


def overview_page() -> None:
    st.markdown("# Tri-HB Integrated Workspace")
    st.caption("Testing design, experimental data reduction, stress-wave interpretation, and DEM-oriented damage validation.")

    st.markdown("### Workflow at a glance")
    try:
        # Natural (compact) size, not stretched to the column, so a landscape
        # flow stays one-screen tall.
        st.graphviz_chart(_overview_workflow_dot(), use_container_width=False)
    except Exception:  # noqa: BLE001 - fall back to text if Graphviz is unavailable
        st.markdown(
            "Step 1 (setup + simulator / reducer) -> Step 2 wave -> Step 3 stress "
            "path -> Step 4 damage -> Step 5 summary -> Step 6 (report and "
            "presentation)."
        )
    st.caption(
        "Step 1 creates the shared setup (via the simulator **or** the data reducer). "
        "Steps 2-5 reuse it, so the wave, stress-path, damage and summary views stay "
        "connected. Step 5 captures the figures that Step 6 turns into a report and a "
        "presentation deck."
    )

    with st.expander("Step-by-step detail", expanded=False):
        st.markdown(
            """
            1. **Test setup, simulator and data analysis** defines the shared material, specimen, bar, loading mode, and experimental data source.
            2. **Wave model** checks pulse timing, wave travel, equilibrium, and interaction regime.
            3. **Stress path and analysis** reviews p-q-theta paths, stress histories, and reduced-data comparison.
            4. **Damage model and validation** evaluates damage evolution, energy indicators, DEM descriptors, and exports.
            5. **Summary of Results** collects every step's key figure in one publication-styled view (600 dpi), each shown beside its governing equations and a short reading guide, with per-figure PNG downloads and a one-click ZIP of the whole set.
            6. **Report and Presentation** generates run-specific documents (LaTeX + PDF): a short report and a Handbook-styled slide deck, both built from this run's Steps 1-5.
            """
        )

    st.caption(
        "Step 1 uses four loading families: Gas-gun uniaxial SHPB; "
        "Confinement-chamber SHPB; Gas-gun Tri-HB static-dynamic loading; "
        "and Electromagnetic programmable loading with three internal topologies."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Loading families", "4")
    c2.metric("Workflow steps", "6")
    c3.metric("Experimental reducer", "CSV/XLSX")

    st.info(
        "Start with Step 1 to create the shared setup and either simulate or reduce data. "
        "Steps 2-5 reuse those settings so the wave, stress-path, damage and summary views "
        "stay connected; Step 6 exports a run report and a presentation deck."
    )

    st.markdown("### How the Tri-HB works — wave propagation")
    st.caption(
        "Isometric view of the Monash triaxial Hopkinson bar. **Static confinement "
        "(pre-stress) is applied on all three axes** by hydraulic cylinders (blue): "
        "the X hydraulic load cylinder at the transmission end, plus the Y and Z "
        "cylinders — this is the triaxial static–dynamic coupling. Press **Play**: "
        "the striker then launches a dynamic stress pulse down the X incident bar; at "
        "the cubic specimen it partly reflects (← incident bar) and partly transmits "
        "(→ transmission bar). The specimen expands laterally (Poisson effect) and "
        "drives **output waves outward along the four Y and Z bars**, recorded by "
        "their gauges (G3–G6). The scope below shows all six gauge signals — "
        "incident, reflected, transmitted, and the Y/Z outputs (Eqs. 4–7) — sweeping "
        "in step with the animation."
    )
    try:
        theme = st.get_option("theme.base") or "light"
    except Exception:  # noqa: BLE001 - default to light if the option is unavailable
        theme = "light"
    components.html(_trihb_animation_html(theme), height=720, scrolling=False)
    st.caption(
        "Schematic for orientation; bar lengths and the gas-gun pulse follow the "
        "Monash 42CrMo system (C₀ ≈ 5200 m/s, striker up to 50 m/s). The Y/Z output "
        "waves are Poisson-driven lateral output (peaks ≈ 0.2× the axial pulse, "
        "arriving just after the transmitted peak), matching the gas-gun Tri-HB "
        "branch in Tri-HB.py. The same signals are reduced quantitatively in Step 2 "
        "(wave model) and Step 3 (stress path)."
    )


def setup_simulator_and_data_page() -> None:
    st.sidebar.markdown("### Step 1 task")
    task = st.sidebar.radio(
        "Choose workspace",
        ["Test design and simulator", "Experimental data analysis"],
        help="Both tasks share the same Step 1 setup. Run the simulator first when you want the analysis page to inherit simulated signals.",
    )
    if task == "Test design and simulator":
        run_legacy_app("Tri-HB.py")
    else:
        experimental_analysis_page()


_MODE_LABELS = {
    "gas-gun": "Gas-gun uniaxial SHPB",
    "confinement-chamber": "Confinement-chamber SHPB",
    "gas-gun-triaxial": "Gas-gun Tri-HB static-dynamic loading",
    "em-uniaxial": "Electromagnetic programmable loading - single-axis one-sided",
    "em-async": "Electromagnetic programmable loading - multi-axis one-sided",
    "em-symmetric": "Electromagnetic programmable loading - symmetric opposing pairs",
}


def _wave_linked_signature(cfg: dict, res: dict | None) -> str:
    """Return the Step 1 setup signature used by the linked wave/damage page."""
    cfg = cfg or {}
    res = res or {}
    mode = str(cfg.get("mode", ""))
    axes = str(cfg.get("symmetric_axes", ""))
    active_axes = str(cfg.get("active_axes", axes or "XYZ"))
    if mode == "em-symmetric" and axes == "XYZ":
        default_path = "Symmetric XYZ"
    elif mode == "em-symmetric" and axes == "XY":
        default_path = "Symmetric XY"
    elif mode == "em-symmetric":
        default_path = "Symmetric X"
    elif mode == "em-async":
        default_path = "Asynchronous XY" if active_axes == "XY" else "Asynchronous XYZ"
        if active_axes == "X":
            default_path = "Single-sided X"
    else:
        default_path = "Single-sided X"

    linked_peak_mpa = float(cfg.get("peak_stress", 0.0)) / 1e6
    if linked_peak_mpa <= 0.0:
        linked_peak_mpa = float(res.get("summary", {}).get("peak_incident_MPa", 0.0))
    peak_x_mpa = float(cfg.get("peak_stress_X", cfg.get("peak_stress", 0.0))) / 1e6
    peak_y_mpa = float(cfg.get("peak_stress_Y", cfg.get("peak_stress", 0.0))) / 1e6
    peak_z_mpa = float(cfg.get("peak_stress_Z", cfg.get("peak_stress", 0.0))) / 1e6
    default_ax = peak_x_mpa if peak_x_mpa > 0 else linked_peak_mpa
    default_ay = peak_y_mpa if peak_y_mpa > 0 else linked_peak_mpa
    default_az = peak_z_mpa if peak_z_mpa > 0 else linked_peak_mpa

    return (
        f"{mode}|{axes}|{active_axes}|{default_path}|"
        f"{default_ax:.6g}|{default_ay:.6g}|{default_az:.6g}|"
        f"{float(cfg.get('pulse_duration', 0.0)) * 1e6:.6g}|"
        f"{float(cfg.get('confinement_X', 0.0)) / 1e6:.6g}|"
        f"{float(cfg.get('confinement_Y', 0.0)) / 1e6:.6g}|"
        f"{float(cfg.get('confinement_Z', 0.0)) / 1e6:.6g}|"
        f"{float(cfg.get('pulse_delay_Y', 0.0)) * 1e6:.6g}|"
        f"{float(cfg.get('pulse_delay_Z', 0.0)) * 1e6:.6g}|"
        f"{float(cfg.get('material_E', 0.0)) / 1e9:.6g}|"
        f"{float(cfg.get('material_nu', 0.0)):.6g}|"
        f"{float(cfg.get('material_density', 0.0)):.6g}|"
        f"{float(cfg.get('specimen_length', cfg.get('specimen_size', 0.0))) * 1000.0:.6g}"
    )


def _summary_input_waveform_details(res, cfg, wd, have_sim, have_summary) -> None:
    """Detailed breakdown of the input stress-wave loading for the Summary page."""
    st.markdown("## Input stress-waveform details")
    st.caption(
        "Exactly how the specimen was loaded - pulse shape, amplitude, duration, "
        "timing, and the static pre-stress on each axis - so the figures below can "
        "be read against their inputs."
    )

    mode = str(cfg.get("mode", "-"))
    mode_label = _MODE_LABELS.get(mode, mode)
    is_gas = mode in ("gas-gun", "confinement-chamber", "gas-gun-triaxial")
    is_async = mode == "em-async"
    is_sym = mode == "em-symmetric"
    is_em = mode.startswith("em-")

    bar_E = float(cfg.get("bar_E", 210e9))
    bar_C0 = float(cfg.get("bar_C0", 5172.0))
    peak_MPa = float(cfg.get("peak_stress", 0.0)) / 1e6
    peak_x_MPa = float(cfg.get("peak_stress_X", cfg.get("peak_stress", 0.0))) / 1e6
    peak_y_MPa = float(cfg.get("peak_stress_Y", cfg.get("peak_stress", 0.0))) / 1e6
    peak_z_MPa = float(cfg.get("peak_stress_Z", cfg.get("peak_stress", 0.0))) / 1e6
    dur_us = float(cfg.get("pulse_duration", 0.0)) * 1e6
    vel = float(cfg.get("velocity", 0.0))

    left, right = st.columns([1.0, 1.0])

    with left:
        st.markdown("**Loading configuration**")
        info = [("Loading mode", mode_label)]
        if is_em:
            info += [
                ("Loading family", str(cfg.get("loading_family", "electromagnetic"))),
                ("EM topology", str(cfg.get("em_topology", "-"))),
                ("Active axes", str(cfg.get("active_axes", "-"))),
                ("Legacy branch", str(cfg.get("legacy_mode", mode))),
            ]
        if is_gas:
            gas_peak = 0.5 * (bar_E / bar_C0) * vel / 1e6
            info += [
                ("Pulse shape", "Half-sine (pulse-shaped)"),
                ("Striker velocity", f"{vel:.0f} m/s"),
                ("Peak incident stress", f"{gas_peak:.0f} MPa (= 0.5 Eb/C0 . V)"),
                ("Pulse duration", "~193 us (set by striker length)"),
            ]
        else:
            info.append(("Pulse shape", "Half-sine"))
            if is_async:
                info += [
                    ("A_X pulse amplitude", f"{peak_x_MPa:.0f} MPa"),
                    ("A_Y pulse amplitude", f"{peak_y_MPa:.0f} MPa"),
                    ("A_Z pulse amplitude", f"{peak_z_MPa:.0f} MPa"),
                ]
            else:
                info.append(("Single-pulse peak amplitude", f"{peak_MPa:.0f} MPa"))
            info.append(("Pulse duration tau", f"{dur_us:.0f} us"))
        if is_sym:
            axes = str(cfg.get("symmetric_axes", "XYZ"))
            info += [
                ("Active symmetric axes", axes),
                ("Effective specimen drive", f"{2 * peak_MPa:.0f} MPa per active axis (2x superposition)"),
            ]
        if is_async:
            info += [
                ("Y-pulse delay", f"{float(cfg.get('pulse_delay_Y', 0.0)) * 1e6:.0f} us"),
                ("Z-pulse delay", f"{float(cfg.get('pulse_delay_Z', 0.0)) * 1e6:.0f} us"),
            ]
        st.dataframe(pd.DataFrame(info, columns=["Parameter", "Value"]),
                     use_container_width=True, hide_index=True)

        st.markdown("**Static pre-stress (held by the loading frame before the pulse)**")
        pre = [
            ("sigma_1 axial (X)", f"{float(cfg.get('confinement_X', 0.0)) / 1e6:.0f} MPa"),
            ("sigma_2 confining (Y)", f"{float(cfg.get('confinement_Y', 0.0)) / 1e6:.0f} MPa"),
            ("sigma_3 confining (Z)", f"{float(cfg.get('confinement_Z', 0.0)) / 1e6:.0f} MPa"),
        ]
        st.dataframe(pd.DataFrame(pre, columns=["Axis", "Pre-stress"]),
                     use_container_width=True, hide_index=True)

    with right:
        st.markdown("**Input pulse equation**")
        if is_gas:
            st.latex(r"\sigma_I(t)=\sigma_{\mathrm{peak}}\,\sin\!\left(\frac{\pi t}{\tau}\right),\quad \tau=\frac{2 L_{\mathrm{striker}}}{C_0}")
            st.latex(r"\sigma_{\mathrm{peak}}=\tfrac12\,\rho_b C_0 V=\tfrac12\,(E_b/C_0)\,V")
            st.caption("Pulse-shaped half-sine (a soft-metal pulse shaper smooths the striker pulse, as in real SHPB tests). Striker impact sets the amplitude; striker length sets the duration.")
        else:
            st.latex(r"\sigma_I(t)=\sigma_{\mathrm{peak}}\,\sin\!\left(\frac{\pi(t-\Delta t_i)}{\tau}\right),\ 0\le t-\Delta t_i\le\tau")
            if is_async:
                st.latex(r"\Delta t_X=0,\quad \Delta t_Y,\ \Delta t_Z\ \text{set the sequence}")
                st.caption("Each axis carries its own delayed half-sine; amplitude and tau are independent of the delays.")
            elif is_sym:
                st.latex(r"\sigma_{\mathrm{specimen}}\approx 2\,\sigma_{\mathrm{peak}},\qquad \sigma_{\mathrm{bar}}\approx \sigma_{\mathrm{peak}}")
                st.caption("Opposing bars superpose in the specimen (2x) while each bar still carries only one pulse.")
            else:
                st.caption("Peak amplitude and duration tau are tuned independently of each other.")

        st.markdown("**Total stress decomposition**")
        st.latex(r"\sigma_i^{\mathrm{total}}(t)=\sigma_i^{0}\ (\text{static}) + \sigma_i^{\mathrm{dyn}}(t)\ (\text{wave})")

        if have_summary:
            st.markdown("**Wave timing (from Steps 2-3)**")
            timing = [
                ("Travel time t_travel", f"{wd.get('t_travel_us', float('nan')):.2f} us"),
                ("Equilibrium window", f"{wd.get('t_eq_low_us', float('nan')):.1f}-{wd.get('t_eq_high_us', float('nan')):.1f} us"),
                ("Normalised delay dt*", f"{wd.get('dt_star', float('nan')):.2f}"),
                ("Regime", str(wd.get("regime", "-"))),
                ("Superposition factor eta", f"{wd.get('eta_sup', float('nan')):.2f}"),
            ]
            st.dataframe(pd.DataFrame(timing, columns=["Quantity", "Value"]),
                         use_container_width=True, hide_index=True)
        else:
            st.info("Open Steps 2-3 once to capture wave-timing details (travel time, equilibrium window, regime).")

    st.divider()


def summary_page() -> None:
    """Step 5 - Summary of Results: publication-ready figures for every step."""
    st.markdown("# Step 5: Summary of Results")
    st.caption(
        "One place to review every step's key result. Each figure sits beside its "
        "governing equations and a short reading guide. Download any figure as a "
        f"{PUB_DPI} dpi PNG, or grab the whole set as a single ZIP at the bottom."
    )

    res = st.session_state.get("tri_hb_latest_result")
    cfg = st.session_state.get("tri_hb_latest_config", {}) or {}
    reduced = st.session_state.get("tri_hb_reduced_data")
    reduced_src = st.session_state.get("tri_hb_reduced_source", "")
    wd = st.session_state.get("tri_hb_wave_damage_summary")
    wdf = st.session_state.get("tri_hb_step3_damage_data")

    have_sim = isinstance(res, dict) and "time" in res
    have_reduced = reduced is not None and hasattr(reduced, "empty") and not reduced.empty
    have_wave = wdf is not None and hasattr(wdf, "empty") and not wdf.empty
    have_summary = isinstance(wd, dict)

    if have_summary:
        linked_signature = wd.get("linked_signature")
        if linked_signature and linked_signature != _wave_linked_signature(cfg, res):
            st.warning(
                "Step 1 has changed since the Step 2-4 wave/damage results were "
                "generated. Open Step 2 or Step 3 once to refresh the stress path "
                "before using the Summary figures."
            )
            have_wave = False
            have_summary = False

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Step 1 simulator", "ready" if have_sim else "not run")
    s2.metric("Optional experimental reducer", "ready" if have_reduced else "not run")
    s3.metric("Steps 2-3 wave/stress", "ready" if have_wave else "not run")
    s4.metric("Step 4 damage", "ready" if have_wave else "not run")

    if not (have_sim or have_reduced or have_wave):
        st.info(
            "No results yet. Run Step 1 (simulator or data reduction), then open "
            "Steps 2-4 at least once so their figures are captured here."
        )
        return

    registry: list = []  # (filename, png_bytes) collected by show_pub_figure

    # ---- Combined key-metric table ----
    rows = []
    if have_sim:
        sm = res.get("summary", {})
        rows += [
            ("Step 1 sim", "Loading mode", _MODE_LABELS.get(str(cfg.get("mode", "-")), str(cfg.get("mode", "-")))),
            ("Step 1 sim", "Peak specimen stress (MPa)", f"{sm.get('peak_specimen_stress_MPa', float('nan')):.1f}"),
            ("Step 1 sim", "Peak axial strain (%)", f"{sm.get('peak_strain_pct', float('nan')):.3f}"),
            ("Step 1 sim", "Avg strain rate (1/s)", f"{sm.get('avg_strain_rate', float('nan')):.0f}"),
            ("Step 1 sim", "Max bar stress (MPa)", f"{sm.get('peak_bar_stress_MPa', float('nan')):.0f}"),
            ("Step 1 sim", "Peak pressure (MPa)", f"{sm.get('peak_pressure_MPa', float('nan')):.1f}"),
        ]
        if str(cfg.get("mode", "")).startswith("em-"):
            rows += [
                ("Step 1 sim", "EM topology", str(cfg.get("em_topology", "-"))),
                ("Step 1 sim", "Active axes", str(cfg.get("active_axes", "-"))),
            ]
    if have_reduced:
        rows += [
            ("Optional experimental reducer", "Source", str(reduced_src)),
            ("Optional experimental reducer", "Peak stress (MPa)", f"{reduced['stress_MPa'].max():.1f}"),
            ("Optional experimental reducer", "Peak strain (%)", f"{100.0 * reduced['strain'].max():.3f}"),
        ]
        if "energy_absorbed_J" in reduced.columns:
            rows.append(("Optional experimental reducer", "Absorbed energy (J)", f"{reduced['energy_absorbed_J'].iloc[-1]:.2f}"))
    if have_summary:
        rows += [
            ("Step 2 wave", "Loading path", str(wd.get("loading_path", "-"))),
            ("Step 2 wave", "Regime", str(wd.get("regime", "-"))),
            ("Step 2 wave", "Travel time (us)", f"{wd.get('t_travel_us', float('nan')):.2f}"),
            ("Step 2 wave", "Normalised delay dt*", f"{wd.get('dt_star', float('nan')):.2f}"),
            ("Step 2 wave", "Superposition factor", f"{wd.get('eta_sup', float('nan')):.2f}"),
            ("Step 3 stress", "Peak p (MPa)", f"{wd.get('peak_p_MPa', float('nan')):.1f}"),
            ("Step 3 stress", "Peak q (MPa)", f"{wd.get('peak_q_MPa', float('nan')):.1f}"),
            ("Step 3 stress", "Peak failure index F", f"{wd.get('peak_F', float('nan')):.2f}"),
            ("Step 4 damage", "Final damage D", f"{wd.get('D_final', float('nan')):.3f}"),
            ("Step 4 damage", "Central fraction Dc", f"{wd.get('D_c', float('nan')):.3f}"),
            ("Step 4 damage", "Symmetry index Sx", f"{wd.get('S_x', float('nan')):.3f}"),
        ]
    if rows:
        st.markdown("## Key metrics")
        st.dataframe(pd.DataFrame(rows, columns=["Step", "Metric", "Value"]),
                     use_container_width=True, hide_index=True)

    # ---- Input stress-waveform details (so users can read the input clearly) ----
    if have_sim or have_summary:
        _summary_input_waveform_details(res, cfg, wd, have_sim, have_summary)

    # ---- Section A: Step 1 simulator figures ----
    if have_sim:
        st.markdown("## Step 1 - simulator output")
        t_us = np.asarray(res["time"]) * 1e6

        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        ax.plot(np.asarray(res["eps_x"]) * 100.0, np.asarray(res["sig_x"]) / 1e6,
                color=PLOT_COLORS["blue"])
        ax.set_xlabel("Axial strain (%)")
        ax.set_ylabel("Axial stress (MPa)")
        ax.set_title("Dynamic stress-strain (axial)")
        show_pub_figure(
            fig, "Step 1: axial dynamic stress-strain curve",
            "step1_stress_strain.png", registry,
            notes=(
                "**What it shows.** The specimen's dynamic constitutive response: "
                "axial stress against axial strain, with rate hardening already "
                "baked in.\n\n"
                "- **Slope at origin** = dynamic Young's modulus.\n"
                "- **Peak** = dynamic compressive strength.\n"
                "- **Post-peak drop** = softening / damage.\n\n"
                "Reduced from the bar gauges by the three-wave method:"
            ),
            equations=[
                r"\sigma_s(t)=\frac{E_b A_b}{2 A_s}\,[\varepsilon_I+\varepsilon_R+\varepsilon_T]",
                r"\varepsilon_s(t)=\int_0^t \dot\varepsilon_s\,d\tau,\quad "
                r"\dot\varepsilon_s=\frac{C_0}{L_s}[\varepsilon_I-\varepsilon_R-\varepsilon_T]",
            ],
        )

        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        gauge_offsets = _gauge_time_offsets_us(cfg)
        x_i, y_i = _waveform_trace_window(t_us, np.asarray(res["epsI_x_pos"]) * 1e6, gauge_offsets["incident"])
        x_r, y_r = _waveform_trace_window(t_us, np.asarray(res["epsR_x_pos"]) * 1e6, gauge_offsets["reflected"])
        x_t, y_t = _waveform_trace_window(t_us, np.asarray(res["epsT_x"]) * 1e6, gauge_offsets["transmitted"])
        ax.plot(x_i, y_i, color=PLOT_COLORS["blue"], label="incident")
        ax.plot(x_r, y_r, color=PLOT_COLORS["orange"], label="reflected")
        ax.plot(x_t, y_t, color=PLOT_COLORS["green"], label="transmitted")
        ax.set_xlabel(r"Acquisition time ($\mu$s)")
        ax.set_ylabel(r"Bar strain ($\mu\varepsilon$)")
        ax.set_title("Raw bar gauge waveforms (X axis)")
        ax.margins(y=0.18)  # headroom so the legend box clears the plateau
        ax.legend(loc="upper right", ncol=3, fontsize=8)
        show_pub_figure(
            fig, "Step 1: incident / reflected / transmitted bar waveforms",
            "step1_bar_waveforms.png", registry,
            notes=(
                "**What it shows.** The three raw bar-gauge strain signals before "
                "time shifting for reduction.\n\n"
                "- **Incident** $\\varepsilon_I$ (loading pulse into the bar).\n"
                "- **Reflected** $\\varepsilon_R$ (negative for a softening specimen) "
                "encodes the strain rate.\n"
                "- **Transmitted** $\\varepsilon_T$ scales with the specimen stress.\n\n"
                "The stress calculation aligns these windows in the next reduction "
                "step; this figure preserves the acquisition-time delays.\n\n"
                "Each gauge reads a bar stress $\\sigma_b=E_b\\varepsilon$; that must "
                "stay below the bar limit (930 MPa)."
            ),
            equations=[
                r"\sigma_b(t)=E_b\,\varepsilon(t)\le \sigma_{\mathrm{prop}}=930\ \mathrm{MPa}",
            ],
        )

        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        ax.plot(t_us, np.asarray(res["sig_x"]) / 1e6, color=PLOT_COLORS["blue"], label=r"$\sigma_X$")
        ax.plot(t_us, np.asarray(res["sig_y"]) / 1e6, color=PLOT_COLORS["green"], label=r"$\sigma_Y$")
        ax.plot(t_us, np.asarray(res["sig_z"]) / 1e6, color=PLOT_COLORS["purple"], label=r"$\sigma_Z$")
        ax.plot(t_us, np.asarray(res["pressure"]) / 1e6, color=PLOT_COLORS["gray"],
                linestyle="--", label=r"mean $p$")
        ax.set_xlabel(r"Time ($\mu$s)")
        ax.set_ylabel("Stress (MPa)")
        ax.set_title("Specimen stresses")
        ax.margins(y=0.18)
        ax.legend(loc="upper right", ncol=2, fontsize=8)
        show_pub_figure(
            fig, "Step 1: specimen stress histories",
            "step1_specimen_stresses.png", registry,
            notes=(
                "**What it shows.** The three principal specimen stresses and the mean "
                "pressure through the event.\n\n"
                "- Each axis is static pre-stress plus its dynamic increment.\n"
                "- $\\sigma_X$ is the driven (axial) load; $\\sigma_Y,\\sigma_Z$ are the "
                "lateral/confining response.\n"
                "- The dashed mean pressure $p$ is what feeds the Step 3 stress path."
            ),
            equations=[
                r"\sigma_i^{\mathrm{total}}=\sigma_i^{0}+\sigma_i^{\mathrm{dyn}}(t)",
                r"p=\tfrac{1}{3}(\sigma_X+\sigma_Y+\sigma_Z)",
            ],
        )

        if res.get("summary", {}).get("is_full_hydrostatic"):
            fig, ax = plt.subplots(figsize=(5.4, 3.6))
            ax.plot(np.asarray(res["eps_vol"]) * 100.0, np.asarray(res["pressure"]) / 1e6,
                    color=PLOT_COLORS["vermillion"])
            ax.set_xlabel("Volumetric strain (%)")
            ax.set_ylabel("Mean pressure (MPa)")
            ax.set_title("Pressure - volumetric strain (hydrostatic)")
            show_pub_figure(
                fig, "Step 1: pressure-volume compaction curve",
                "step1_pressure_volume.png", registry,
                notes=(
                    "**What it shows.** The compaction (cap) response under fully "
                    "hydrostatic loading, where the deviatoric stress $q\\approx0$.\n\n"
                    "- **Initial slope** = dynamic bulk modulus $K$.\n"
                    "- **Plateau** = cap pressure (pore collapse / grain crushing)."
                ),
                equations=[
                    r"\frac{dp}{d\varepsilon_v}=K,\qquad "
                    r"\varepsilon_v=\varepsilon_X+\varepsilon_Y+\varepsilon_Z",
                ],
            )

    # ---- Section B: Step 1 experimental reducer figures ----
    if have_reduced:
        st.markdown("## Step 1 - experimental data reduction")
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        ax.plot(reduced["strain"] * 100.0, reduced["stress_MPa"], color=PLOT_COLORS["vermillion"])
        ax.set_xlabel("Strain (%)")
        ax.set_ylabel("Stress (MPa)")
        ax.set_title(f"Reduced stress-strain ({reduced_src})")
        show_pub_figure(
            fig, "Step 1: reduced experimental stress-strain",
            "step1_reduced_stress_strain.png", registry,
            notes=(
                "**What it shows.** The stress-strain curve reduced from your uploaded "
                "bar-gauge or direct data, on the same axes as the simulator so you can "
                "overlay measurement against prediction."
            ),
            equations=[
                r"\sigma_s=\frac{E_b A_b}{2A_s}[\varepsilon_I+\varepsilon_R+\varepsilon_T]",
            ],
        )

        if "energy_absorbed_J" in reduced.columns:
            fig, ax = plt.subplots(figsize=(5.6, 3.6))
            for col, lbl, c in [
                ("energy_incident_J", "incident", PLOT_COLORS["blue"]),
                ("energy_reflected_J", "reflected", PLOT_COLORS["orange"]),
                ("energy_transmitted_J", "transmitted", PLOT_COLORS["green"]),
                ("energy_absorbed_J", "absorbed", PLOT_COLORS["vermillion"]),
            ]:
                if col in reduced.columns:
                    ax.plot(reduced["time_us"], reduced[col], label=lbl, color=c)
            ax.set_xlabel(r"Time ($\mu$s)")
            ax.set_ylabel("Energy (J)")
            ax.set_title("Reduced energy histories")
            ax.legend()
            show_pub_figure(
                fig, "Step 1: reduced energy histories",
                "step1_reduced_energy.png", registry,
                notes=(
                    "**What it shows.** Cumulative bar-wave energies. The absorbed "
                    "energy is what the specimen actually dissipated.\n\n"
                    "A physical check: absorbed must stay non-negative and below "
                    "incident."
                ),
                equations=[
                    r"W_j=E_b A_b C_0\!\int_0^t\!\varepsilon_j^2\,d\tau,\ j\in\{I,R,T\}",
                    r"W_{\mathrm{abs}}=W_I-W_R-W_T",
                ],
            )

    # ---- Sections C-E: Steps 2-4 from the wave/damage dataframe ----
    if have_wave:
        d = wdf
        tw = np.asarray(d["time_us"])

        st.markdown("## Step 2 - wave model")
        fig, ax = plt.subplots(figsize=(5.8, 3.6))
        ax.plot(tw, np.asarray(d["x_left_MPa"]), color=PLOT_COLORS["blue"], label="X reference")
        if np.any(np.abs(np.asarray(d["x_right_MPa"])) > 0):
            ax.plot(tw, np.asarray(d["x_right_MPa"]), color=PLOT_COLORS["gray"],
                    linestyle="--", label="X opposing/secondary")
        if np.any(np.abs(np.asarray(d["y_drive_MPa"])) > 0):
            ax.plot(tw, np.asarray(d["y_drive_MPa"]), color=PLOT_COLORS["green"], label="Y drive")
        if np.any(np.abs(np.asarray(d["z_drive_MPa"])) > 0):
            ax.plot(tw, np.asarray(d["z_drive_MPa"]), color=PLOT_COLORS["purple"], label="Z drive")
        if have_summary:
            ax.axvspan(wd.get("t_eq_low_us", 0.0), wd.get("t_eq_high_us", 0.0),
                       color="#9ec7ff", alpha=0.30, label="3-5 travel times")
        ax.set_xlabel(r"Time ($\mu$s)")
        ax.set_ylabel(r"$\Delta\sigma$ (MPa)")
        ax.set_title("Input pulses and equilibrium window")
        ax.margins(y=0.18)
        ax.legend(loc="upper right", ncol=2, fontsize=8)
        show_pub_figure(
            fig, "Step 2: input pulses with equilibrium window",
            "step2_input_pulses.png", registry,
            notes=(
                "**What it shows.** The dynamic stress pulses entering the specimen on "
                "each active axis, with the shaded equilibrium window.\n\n"
                "- The pulse must be **longer** than the window for valid analysis.\n"
                "- Travel time and the window come from the P-wave speed."
            ),
            equations=[
                r"c_p=\sqrt{M/\rho},\quad t_{\mathrm{travel}}=L_s/c_p",
                r"t_{\mathrm{eq}}\approx 3\text{--}5\,t_{\mathrm{travel}}",
            ],
        )

        st.markdown("## Step 3 - stress path")
        p_arr = np.asarray(d["p_MPa"]); q_arr = np.asarray(d["q_MPa"])
        fig, ax = plt.subplots(figsize=(5.2, 3.9))
        sc = ax.scatter(p_arr, q_arr, c=tw, s=10, cmap="viridis")
        if "qf_MPa" in d.columns:
            order = np.argsort(p_arr)
            ax.plot(p_arr[order], np.asarray(d["qf_MPa"])[order], color=PLOT_COLORS["vermillion"],
                    linestyle="--", label=r"failure surface $q_f$")
            ax.legend()
        ax.set_xlabel("Mean stress, p (MPa)")
        ax.set_ylabel("Deviatoric stress, q (MPa)")
        ax.set_title("p-q stress path")
        fig.colorbar(sc, ax=ax, label=r"Time ($\mu$s)")
        show_pub_figure(
            fig, "Step 3: p-q stress path coloured by time",
            "step3_pq_path.png", registry,
            notes=(
                "**What it shows.** The loading trajectory in mean-stress / "
                "deviatoric-stress space, coloured by time.\n\n"
                "- The dashed line is the pressure-dependent **failure surface**.\n"
                "- The path **touching** $q_f$ is when damage starts to grow."
            ),
            equations=[
                r"p=\tfrac{1}{3}(\sigma_X+\sigma_Y+\sigma_Z)",
                r"q=\sqrt{\tfrac{1}{2}[(\sigma_X-\sigma_Y)^2+(\sigma_Y-\sigma_Z)^2+(\sigma_Z-\sigma_X)^2]}",
                r"q_f=(A+B\,p^{n})\,h(\theta)",
            ],
        )

        theta_arr = np.asarray(d["theta_deg"])
        fig, ax = plt.subplots(figsize=(5.8, 3.6))
        line_p, = ax.plot(tw, p_arr, color=PLOT_COLORS["blue"], label="p")
        line_q, = ax.plot(tw, q_arr, color=PLOT_COLORS["vermillion"], label="q")
        ax.set_xlabel(r"Time ($\mu$s)")
        ax.set_ylabel("p and q (MPa)")
        ax.set_title("Stress invariant histories")
        ax_theta = ax.twinx()
        line_theta, = ax_theta.plot(
            tw, theta_arr, color=PLOT_COLORS["green"], label=r"$\theta$ (deg)"
        )
        theta_max = float(np.nanmax(theta_arr)) if theta_arr.size else 0.0
        ax_theta.set_ylim(0.0, max(5.0, min(60.0, theta_max * 1.15)))
        ax_theta.set_ylabel(r"Lode angle, $\theta$ (deg)")
        ax_theta.grid(False)
        ax.legend([line_p, line_q, line_theta], ["p", "q", r"$\theta$ (deg)"], loc="upper right")
        show_pub_figure(
            fig, "Step 3: invariant histories (p, q, theta)",
            "step3_invariants.png", registry,
            notes=(
                "**What it shows.** The three stress invariants through time. "
                "**p** and **q** use the left axis in MPa; **theta** uses the "
                "right axis in degrees.\n\n"
                "- **p** mean (hydrostatic) stress.\n"
                "- **q** deviatoric (shear) stress drives shear failure.\n"
                "- **theta** classifies the shear mode: **0 deg** = triaxial "
                "compression, **30 deg** = pure shear, **60 deg** = triaxial "
                "extension.\n\n"
                "A near-zero theta is expected for single-sided / axisymmetric "
                "compression paths. It only moves strongly away from zero when "
                "the three principal stresses become genuinely unequal, such as "
                "in an asynchronous true-triaxial case."
            ),
            equations=[
                r"\cos(3\theta)=\frac{3\sqrt{3}}{2}\,\frac{J_3}{J_2^{3/2}}",
            ],
        )

        st.markdown("## Step 4 - damage and energy")
        fig, ax = plt.subplots(figsize=(5.8, 3.6))
        ax.plot(tw, np.asarray(d["D"]), color=PLOT_COLORS["vermillion"], label="Damage D")
        ax.set_xlabel(r"Time ($\mu$s)")
        ax.set_ylabel("Damage variable D")
        ax.set_ylim(0, 1.02)
        ax2 = ax.twinx()
        ax2.plot(tw, np.asarray(d["failure_index"]), color=PLOT_COLORS["blue"],
                 linestyle="--", label="Failure index F")
        ax2.axhline(1.0, color=PLOT_COLORS["gray"], linewidth=1.0, linestyle=":")
        ax2.set_ylabel("Failure index F = q / q_f")
        ax2.grid(False)
        l1, lab1 = ax.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lab1 + lab2, loc="best")
        ax.set_title("Damage growth and failure index")
        show_pub_figure(
            fig, "Step 4: damage growth and failure index",
            "step4_damage_failure.png", registry,
            notes=(
                "**What it shows.** Damage $D$ (0 = intact, 1 = fully failed) and the "
                "failure index $F=q/q_f$.\n\n"
                "- Damage is driven by the **effective** index $(1-D)F$, not $F$ itself.\n"
                "- As $D$ grows the material **sheds load**, so the rate self-limits and "
                "$D$ rises smoothly toward $\\approx 1-1/F_{\\mathrm{peak}}$ instead of "
                "snapping to 1."
            ),
            equations=[
                r"F_{\mathrm{eff}}=(1-D)\,F",
                r"\dot D=\frac{(1-D)^{\alpha}}{\tau_D}\,\langle F_{\mathrm{eff}}-1\rangle^{m}\,"
                r"\Big(\frac{\dot\varepsilon_{\mathrm{eq}}}{\dot\varepsilon_0}\Big)^{\beta}",
            ],
        )

        if "W_input_MJ_m3" in d.columns and "W_diss_estimate_MJ_m3" in d.columns:
            fig, ax = plt.subplots(figsize=(5.8, 3.6))
            ax.plot(tw, np.asarray(d["W_input_MJ_m3"]), color=PLOT_COLORS["blue"], label="input energy")
            ax.plot(tw, np.asarray(d["W_diss_estimate_MJ_m3"]), color=PLOT_COLORS["vermillion"],
                    label="dissipated energy")
            ax.set_xlabel(r"Time ($\mu$s)")
            ax.set_ylabel(r"Energy density (MJ/m$^3$)")
            ax.set_title("Energy balance")
            ax.legend()
            show_pub_figure(
                fig, "Step 4: energy balance",
                "step4_energy_balance.png", registry,
                notes=(
                    "**What it shows.** Cumulative input work versus the energy "
                    "dissipated by damage.\n\n"
                    "- $W_{\\mathrm{diss}}$ is the continuum-damage dissipation "
                    "$\\int Y\\,dD$ (energy released as the modulus degrades).\n"
                    "- It stays at zero until damage grows, then climbs monotonically - "
                    "it is **not** the old near-zero elastic-cancellation estimate."
                ),
                equations=[
                    r"W_{\mathrm{in}}=\int(\sigma_X\dot\varepsilon_X+"
                    r"\sigma_Y\dot\varepsilon_Y+\sigma_Z\dot\varepsilon_Z)\,d\tau",
                    r"W_{\mathrm{diss}}=\int Y\,\dot D\,d\tau,\quad Y=W_{\mathrm{el}}",
                ],
            )

    # ---- Stash the metrics and figures so Step 6 (Report and Presentation)
    # can reuse exactly what Step 5 rendered, without recomputing the figures. ----
    st.session_state["tri_hb_summary_rows"] = rows
    st.session_state["tri_hb_summary_registry"] = registry

    # ---- Bundle everything ----
    st.divider()
    st.markdown("## Download the full figure set")
    if registry:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, png, *_ in registry:
                zf.writestr(fname, png)
            if rows:
                zf.writestr(
                    "summary_metrics.csv",
                    pd.DataFrame(rows, columns=["Step", "Metric", "Value"]).to_csv(index=False),
                )
        st.download_button(
            f"Download all {len(registry)} figures + metrics (ZIP)",
            data=zip_buf.getvalue(),
            file_name="tri_hb_summary_figures.zip",
            mime="application/zip",
        )
    else:
        st.info("No figures were generated for the current session data.")


def _find_latex_engine(name: str) -> str | None:
    """Locate a LaTeX engine by name, even if it is not on the process PATH.

    Streamlit is often launched from an environment whose PATH does not include
    the TeX install directory, so ``shutil.which`` alone can fail even when a
    TeX distribution is present. Fall back to common MiKTeX / TeX Live install
    locations (Windows, macOS, Linux).
    """
    found = shutil.which(name)
    if found:
        return found
    exe = name + (".exe" if os.name == "nt" else "")
    import glob
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64" / exe,
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "MiKTeX" / "miktex" / "bin" / exe,
        Path("C:/Program Files/MiKTeX/miktex/bin/x64") / exe,
        Path("C:/Program Files/MiKTeX/miktex/bin") / exe,
        Path("C:/Program Files (x86)/MiKTeX/miktex/bin/x64") / exe,
        Path("/Library/TeX/texbin") / name,
        Path("/usr/local/texlive") / "**" / "bin" / "**" / name,
        Path("/usr/bin") / name,
        Path("/usr/local/bin") / name,
    ]
    for cand in candidates:
        cstr = str(cand)
        if "**" in cstr:
            hits = sorted(glob.glob(cstr, recursive=True))
            if hits:
                return hits[-1]
        elif cand.exists():
            return str(cand)
    return None


def _compile_latex_to_pdf(tex_source: str, registry: list, jobname: str,
                          prefer_xelatex: bool = False):
    """Compile LaTeX to PDF in a temp dir with the figures alongside.

    Returns (pdf_bytes_or_None, message). The figure PNGs from ``registry``
    (list of (filename, bytes)) are written next to the .tex so
    \\includegraphics resolves. Tries xelatex first when ``prefer_xelatex``
    (needed for beamer/fontspec decks), else pdflatex.
    """
    order = ("xelatex", "pdflatex") if prefer_xelatex else ("pdflatex", "xelatex")
    engine = next((e for e in (_find_latex_engine(n) for n in order) if e), None)
    if not engine:
        return None, ("No LaTeX engine (pdflatex/xelatex) was found. The PDF was "
                      "not built here. Either (a) install a TeX distribution "
                      "(MiKTeX or TeX Live) and restart Streamlit from a shell "
                      "where `pdflatex` is on PATH, or (b) download the LaTeX "
                      "source / ZIP bundle below and compile it elsewhere.")
    try:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            (tdp / f"{jobname}.tex").write_text(tex_source, encoding="utf-8")
            for fname, png, *_ in registry:
                (tdp / fname).write_bytes(png)
            out_pdf = tdp / f"{jobname}.pdf"
            # Build the engine command. For MiKTeX, allow silent on-the-fly
            # package installation so a first cold run does not stall on a
            # prompt; this flag is ignored by TeX Live engines.
            base = [engine, "-interaction=nonstopmode"]
            if "miktex" in engine.lower():
                base += ["--enable-installer"]
            cmd = base + [f"{jobname}.tex"]
            # Note: -halt-on-error is intentionally omitted so non-fatal
            # warnings (e.g. underfull hboxes) never abort the build.
            proc = None
            last_exc = None
            for i in range(2):  # two passes for tables / cross-references
                # Generous timeout: the first run may download MiKTeX packages.
                timeout = 600 if i == 0 else 300
                try:
                    proc = subprocess.run(cmd, cwd=td, capture_output=True, timeout=timeout)
                except subprocess.TimeoutExpired as exc:
                    last_exc = exc
                    break  # keep any PDF produced so far
            if out_pdf.exists():
                return out_pdf.read_bytes(), ""
            # No PDF: assemble the most useful diagnostic available.
            log_path = tdp / f"{jobname}.log"
            detail = ""
            if log_path.exists():
                log = log_path.read_text(encoding="utf-8", errors="replace")
                err_lines = [ln for ln in log.splitlines()
                             if ln.startswith("!") or "Fatal error" in ln
                             or "Emergency stop" in ln or "not found" in ln.lower()]
                detail = "\n".join(err_lines[-25:]) or log[-1800:]
            elif proc is not None and proc.stdout:
                detail = proc.stdout.decode("utf-8", "replace")[-1800:]
            elif last_exc is not None:
                detail = (f"The LaTeX engine timed out after {last_exc.timeout:.0f}s. "
                          "On a first run MiKTeX may be downloading packages; try "
                          "again once, or pre-install packages by compiling any "
                          "document manually.")
            else:
                detail = "LaTeX produced no PDF (no log file was created)."
            return None, detail
    except Exception as exc:  # noqa: BLE001
        return None, f"LaTeX compilation failed: {type(exc).__name__}: {exc}"


def _offer_doc_downloads(tex_source: str, pdf_bytes, registry: list,
                         jobname: str, pdf_label: str, tex_label: str,
                         zip_label: str, key_prefix: str) -> None:
    """Shared download UI: PDF (if built), LaTeX source, and a tex+figures ZIP."""
    c1, c2 = st.columns(2)
    with c1:
        if pdf_bytes:
            st.download_button(pdf_label, data=pdf_bytes,
                               file_name=f"{jobname}.pdf", mime="application/pdf",
                               key=f"{key_prefix}_pdf")
        else:
            st.caption("PDF not built here — use the LaTeX source or ZIP below.")
    with c2:
        st.download_button(tex_label, data=tex_source.encode("utf-8"),
                           file_name=f"{jobname}.tex", mime="text/x-tex",
                           key=f"{key_prefix}_tex")
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{jobname}.tex", tex_source)
        for fname, png, *_ in registry:
            zf.writestr(fname, png)
    st.download_button(zip_label, data=zbuf.getvalue(),
                       file_name=f"{jobname}_bundle.zip", mime="application/zip",
                       key=f"{key_prefix}_zip")


def _latex_escape(text: str) -> str:
    """Escape plain text for safe inclusion in a LaTeX document."""
    if text is None:
        return ""
    s = str(text)
    repl = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def _md_inline_to_latex(text: str) -> str:
    """Convert a line of the figure 'notes' markdown to LaTeX.

    Preserves inline math written as ``$...$`` (passed through verbatim),
    escapes the surrounding prose, and renders ``**bold**`` as \\textbf{}.
    """
    if not text:
        return ""
    # split on $...$ math spans; even indices are prose, odd are math
    parts = re.split(r"(\$[^$]*\$)", str(text))
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # math span - keep as-is (already valid LaTeX)
            out.append(part)
            continue
        esc = _latex_escape(part)
        # restore bold markers: _latex_escape leaves '*' untouched
        esc = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", esc)
        out.append(esc)
    return "".join(out)


def _notes_to_beamer(notes: str):
    """Turn the markdown figure-notes block into (intro_lines, bullet_items).

    Returns a short intro paragraph (non-bullet lines) and a list of bullet
    strings, all already converted to LaTeX. Keeps slides compact.
    """
    intro: list[str] = []
    bullets: list[str] = []
    for raw in (notes or "").split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("- "):
            bullets.append(_md_inline_to_latex(line[2:].strip()))
        else:
            intro.append(_md_inline_to_latex(line))
    return intro, bullets


def _build_report_tex(rows: list, registry: list, cfg: dict, fig_names: list) -> str:
    """Assemble a short LaTeX report from this session's Step 1-5 results."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    mode = _MODE_LABELS.get(str(cfg.get("mode", "-")), str(cfg.get("mode", "-")))

    # group metric rows by their step label, preserving order
    grouped: dict[str, list] = {}
    for step, metric, value in rows:
        grouped.setdefault(step, []).append((metric, value))

    lines: list[str] = []
    A = lines.append
    A(r"\documentclass[11pt,a4paper]{article}")
    A(r"\usepackage[margin=2.4cm]{geometry}")
    A(r"\usepackage{graphicx}")
    A(r"\usepackage{booktabs}")
    A(r"\usepackage{longtable}")
    A(r"\usepackage{amsmath}")
    A(r"\usepackage{xcolor}")
    A(r"\usepackage[colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black]{hyperref}")
    A(r"\usepackage{titlesec}")
    A(r"\setlength{\parskip}{6pt plus 2pt minus 1pt}\setlength{\parindent}{0pt}")
    A(r"\title{\textbf{Tri-HB Run Report}\\[2pt]\large Dynamic Triaxial Hopkinson Bar --- Session Summary}")
    A(r"\author{Generated by the Virtual Tri-HB Integrated Workspace}")
    A(rf"\date{{{_latex_escape(now)}}}")
    A(r"\begin{document}")
    A(r"\maketitle")

    A(r"\section*{1. Overview}")
    A("This short report summarises a single run of the Virtual Tri-HB workspace "
      "(Steps 1--5). It uses the same metrics and figures shown on the Step~5 "
      "Summary page. Full theory and background are in the Tri-HB Handbook; "
      "this is a condensed, run-specific record.")
    A(rf"\textbf{{Loading mode:}} {_latex_escape(mode)}. \\")
    A(rf"\textbf{{Generated:}} {_latex_escape(now)}.")

    if cfg:
        A(r"\section*{2. Test configuration}")
        cfg_rows = [
            ("Loading mode", mode),
            ("Material UCS (MPa)", f"{float(cfg.get('material_UCS', 0.0))/1e6:.1f}"),
            ("Young's modulus E (GPa)", f"{float(cfg.get('material_E', 0.0))/1e9:.1f}"),
            ("Poisson's ratio", f"{float(cfg.get('material_nu', 0.0)):.2f}"),
            ("Specimen length (mm)", f"{float(cfg.get('specimen_length', 0.0))*1e3:.1f}"),
            ("Static pre-stress sx0/sy0/sz0 (MPa)",
             f"{float(cfg.get('confinement_X',0.0))/1e6:.0f}/"
             f"{float(cfg.get('confinement_Y',0.0))/1e6:.0f}/"
             f"{float(cfg.get('confinement_Z',0.0))/1e6:.0f}"),
        ]
        if str(cfg.get("mode", "")).startswith("em-"):
            cfg_rows.append(("Pulse peak / duration",
                             f"{float(cfg.get('peak_stress',0.0))/1e6:.0f} MPa / "
                             f"{float(cfg.get('pulse_duration',0.0))*1e6:.0f} us"))
        else:
            cfg_rows.append(("Striker velocity (m/s)", f"{float(cfg.get('velocity',0.0)):.0f}"))
        A(r"\begin{longtable}{@{}p{0.55\textwidth}p{0.35\textwidth}@{}}")
        A(r"\toprule \textbf{Parameter} & \textbf{Value} \\ \midrule \endhead")
        for k, v in cfg_rows:
            A(rf"{_latex_escape(k)} & {_latex_escape(v)} \\")
        A(r"\bottomrule")
        A(r"\end{longtable}")

    if rows:
        A(r"\section*{3. Key results}")
        A(r"\begin{longtable}{@{}p{0.24\textwidth}p{0.46\textwidth}p{0.22\textwidth}@{}}")
        A(r"\toprule \textbf{Step} & \textbf{Metric} & \textbf{Value} \\ \midrule \endhead")
        for step, metrics in grouped.items():
            for metric, value in metrics:
                A(rf"{_latex_escape(step)} & {_latex_escape(metric)} & {_latex_escape(value)} \\")
            A(r"\midrule")
        A(r"\bottomrule")
        A(r"\end{longtable}")

    if fig_names:
        A(r"\section*{4. Figures}")
        reg_meta = {item[0]: item for item in registry}
        for fname in fig_names:
            item = reg_meta.get(fname)
            caption = (item[2] if item and len(item) > 2 and item[2]
                       else fname.replace("_", " ").replace(".png", ""))
            notes = item[3] if item and len(item) > 3 else None
            equations = item[4] if item and len(item) > 4 else []
            intro, bullets = _notes_to_beamer(notes)
            A(r"\begin{figure}[htbp]\centering")
            A(rf"\includegraphics[width=0.74\textwidth]{{{fname}}}")
            A(rf"\caption{{{_latex_escape(caption)}}}")
            A(r"\end{figure}")
            for para in intro[:2]:
                A(para + r"\par")
            if bullets:
                A(r"\begin{itemize}\setlength{\itemsep}{1pt}")
                for b in bullets[:5]:
                    A(rf"  \item {b}")
                A(r"\end{itemize}")
            if equations:
                A(r"\textbf{Governing equations:}")
                for eq in equations[:4]:
                    A(rf"\[{eq}\]")
            A(r"\clearpage")

    A(r"\section*{5. Notes and scope}")
    A("This is a planning- and design-grade digital-twin report. The wave/stress "
      "histories are reduced from prescribed pulse envelopes; the failure surface "
      "is derived consistently with the Step~1 strength model; the spatial damage "
      "descriptors are model indicators. For quantitative validation, calibrate "
      "the material and damage parameters against measured Tri-HB data (see the "
      "validation and calibration plan).")
    A(r"\end{document}")
    return "\n".join(lines)


def _build_presentation_tex(rows: list, registry: list, cfg: dict, fig_names: list) -> str:
    """Assemble a beamer presentation from this run, in the Handbook deck style."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    mode = _MODE_LABELS.get(str(cfg.get("mode", "-")), str(cfg.get("mode", "-")))

    grouped: dict[str, list] = {}
    for step, metric, value in rows:
        grouped.setdefault(step, []).append((metric, value))

    lines: list[str] = []
    A = lines.append
    # Preamble mirrors Handbook_V4_Presentation.tex: Metropolis if available,
    # graceful fallback; Monash-inspired colour theme.
    A(r"\documentclass[11pt,aspectratio=169]{beamer}")
    A(r"\newif\ifmetropolisavailable")
    A(r"\IfFileExists{beamerthememetropolis.sty}{\metropolisavailabletrue}{\metropolisavailablefalse}")
    A(r"\ifmetropolisavailable")
    A(r"  \usetheme[progressbar=frametitle, sectionpage=progressbar]{metropolis}")
    A(r"\else")
    A(r"  \usetheme{default}")
    A(r"\fi")
    A(r"\usepackage{graphicx}")
    A(r"\usepackage{booktabs}")
    A(r"\usepackage{amsmath}")
    A(r"\usepackage{xcolor}")
    A(r"\definecolor{TUSTnavy}{HTML}{00739D}")
    A(r"\definecolor{TUSTteal}{HTML}{006DAE}")
    A(r"\definecolor{TUSTmidgray}{HTML}{505050}")
    A(r"\definecolor{TUSToffwhite}{HTML}{F6F6F6}")
    A(r"\setbeamercolor{frametitle}{fg=TUSTnavy}")
    A(r"\setbeamercolor{title}{fg=TUSTnavy}")
    A(r"\setbeamercolor{structure}{fg=TUSTteal}")
    A(r"\setbeamertemplate{navigation symbols}{}")
    A(rf"\title{{Tri-HB Run Presentation}}")
    A(rf"\subtitle{{Dynamic Triaxial Hopkinson Bar --- Session Results}}")
    A(rf"\date{{{_latex_escape(now)}}}")
    A(r"\author{Virtual Tri-HB Integrated Workspace}")
    A(r"\begin{document}")
    A(r"\maketitle")

    # Overview slide
    A(r"\begin{frame}{Overview}")
    A(r"\begin{itemize}")
    A(rf"  \item \textbf{{Loading mode:}} {_latex_escape(mode)}")
    A(rf"  \item \textbf{{Generated:}} {_latex_escape(now)}")
    A(r"  \item Auto-generated from this session's Steps 1--5 results.")
    A(r"  \item Theory and full background: Tri-HB Handbook.")
    A(r"\end{itemize}")
    A(r"\end{frame}")

    # Configuration slide
    if cfg:
        cfg_rows = [
            ("Loading mode", mode),
            ("Material UCS (MPa)", f"{float(cfg.get('material_UCS', 0.0))/1e6:.1f}"),
            ("Young's modulus E (GPa)", f"{float(cfg.get('material_E', 0.0))/1e9:.1f}"),
            ("Poisson's ratio", f"{float(cfg.get('material_nu', 0.0)):.2f}"),
            ("Specimen length (mm)", f"{float(cfg.get('specimen_length', 0.0))*1e3:.1f}"),
            ("Pre-stress sx0/sy0/sz0 (MPa)",
             f"{float(cfg.get('confinement_X',0.0))/1e6:.0f}/"
             f"{float(cfg.get('confinement_Y',0.0))/1e6:.0f}/"
             f"{float(cfg.get('confinement_Z',0.0))/1e6:.0f}"),
        ]
        if str(cfg.get("mode", "")).startswith("em-"):
            cfg_rows.append(("Pulse peak / duration",
                             f"{float(cfg.get('peak_stress',0.0))/1e6:.0f} MPa / "
                             f"{float(cfg.get('pulse_duration',0.0))*1e6:.0f} us"))
        else:
            cfg_rows.append(("Striker velocity (m/s)", f"{float(cfg.get('velocity',0.0)):.0f}"))
        A(r"\begin{frame}{Test configuration}")
        A(r"\small\begin{tabular}{@{}p{0.55\textwidth}p{0.35\textwidth}@{}}")
        A(r"\toprule \textbf{Parameter} & \textbf{Value} \\ \midrule")
        for k, v in cfg_rows:
            A(rf"{_latex_escape(k)} & {_latex_escape(v)} \\")
        A(r"\bottomrule\end{tabular}")
        A(r"\end{frame}")

    # Key-results slide(s): one frame per step group to avoid overflow
    for step, metrics in grouped.items():
        A(rf"\begin{{frame}}{{Key results: {_latex_escape(step)}}}")
        A(r"\small\begin{tabular}{@{}p{0.6\textwidth}p{0.3\textwidth}@{}}")
        A(r"\toprule \textbf{Metric} & \textbf{Value} \\ \midrule")
        for metric, value in metrics:
            A(rf"{_latex_escape(metric)} & {_latex_escape(value)} \\")
        A(r"\bottomrule\end{tabular}")
        A(r"\end{frame}")

    # One figure per slide: figure on the left, short description and the
    # governing equations on the right (mirrors the Step 5 layout).
    reg_meta = {item[0]: item for item in registry}
    for fname in fig_names:
        item = reg_meta.get(fname)
        caption = (item[2] if item and len(item) > 2 and item[2]
                   else fname.replace("_", " ").replace(".png", ""))
        notes = item[3] if item and len(item) > 3 else None
        equations = item[4] if item and len(item) > 4 else []
        intro, bullets = _notes_to_beamer(notes)

        A(rf"\begin{{frame}}{{{_latex_escape(caption)}}}")
        A(r"\begin{columns}[T,onlytextwidth]")
        A(r"\begin{column}{0.56\textwidth}")
        A(r"\centering")
        A(rf"\includegraphics[width=\linewidth,height=0.74\textheight,keepaspectratio]{{{fname}}}")
        A(r"\end{column}")
        A(r"\begin{column}{0.42\textwidth}")
        A(r"\scriptsize")
        for para in intro[:2]:  # keep it short
            A(para + r"\\[2pt]")
        if bullets:
            A(r"\begin{itemize}")
            for b in bullets[:4]:
                A(rf"  \item {b}")
            A(r"\end{itemize}")
        if equations:
            A(r"\vspace{2pt}\textbf{\color{TUSTteal} Governing equations}\\[1pt]")
            for eq in equations[:3]:
                A(rf"\[{eq}\]")
        A(r"\end{column}")
        A(r"\end{columns}")
        A(r"\end{frame}")

    # Closing
    A(r"\begin{frame}{Scope}")
    A(r"\footnotesize")
    A("Planning- and design-grade digital-twin results. Wave/stress histories "
      "are reduced from prescribed pulse envelopes; the failure surface is "
      "derived consistently with the Step~1 strength model; spatial damage "
      "descriptors are model indicators. Calibrate against measured Tri-HB data "
      "for quantitative validation.")
    A(r"\end{frame}")

    A(r"\end{document}")
    return "\n".join(lines)


def _render_doc_generator(kind: str, rows: list, registry: list, cfg: dict) -> None:
    """Render one document generator (report or presentation) in the combined page."""
    fig_names = [fname for fname, *_ in registry]
    if kind == "report":
        tex_source = _build_report_tex(rows, registry, cfg, fig_names)
        jobname, prefer_xe = "tri_hb_report", False
        pdf_label = "Download report PDF"
        tex_label = "Download report LaTeX"
        zip_label = "Download report bundle (LaTeX + figures, ZIP)"
        key_prefix = "dl_report"
        preview_label = "Preview report LaTeX source"
    else:
        tex_source = _build_presentation_tex(rows, registry, cfg, fig_names)
        jobname, prefer_xe = "tri_hb_presentation", True
        pdf_label = "Download presentation PDF"
        tex_label = "Download presentation LaTeX"
        zip_label = "Download presentation bundle (LaTeX + figures, ZIP)"
        key_prefix = "dl_pres"
        preview_label = "Preview presentation LaTeX source"

    pdf_bytes, compile_msg = _compile_latex_to_pdf(
        tex_source, registry, jobname, prefer_xelatex=prefer_xe
    )
    _offer_doc_downloads(tex_source, pdf_bytes, registry, jobname,
                         pdf_label=pdf_label, tex_label=tex_label,
                         zip_label=zip_label, key_prefix=key_prefix)
    if compile_msg and not pdf_bytes:
        st.warning("The PDF could not be built on the server. See the build "
                   "details below; you can still download the LaTeX source or "
                   "the ZIP bundle and compile it yourself.")
    if compile_msg:
        with st.expander("PDF build details", expanded=not pdf_bytes):
            st.code(compile_msg)
    with st.expander(preview_label):
        st.code(tex_source, language="latex")


def report_presentation_page() -> None:
    """Step 6 - Report and Presentation: generate both run documents (LaTeX + PDF)."""
    st.markdown("# Step 6: Report and Presentation")
    st.caption(
        "Generate run-specific documents from this session's Steps 1-5: a short "
        "report and a presentation deck, each as LaTeX + PDF. Both use the "
        "metrics and figures captured on the Step 5 summary page; the Handbook "
        "deck is used only as the visual template."
    )

    rows = st.session_state.get("tri_hb_summary_rows")
    registry = st.session_state.get("tri_hb_summary_registry")
    cfg = st.session_state.get("tri_hb_latest_config", {}) or {}

    if not rows and not registry:
        st.info(
            "No run results captured yet. Open **Step 5: Summary of Results** "
            "first (after running Steps 1-4) so the metrics and figures are "
            "available for the documents."
        )
        return

    rows = rows or []
    registry = registry or []
    st.markdown(f"**Captured from this run:** {len(rows)} metrics, "
                f"{len(registry)} figures.")

    tab_report, tab_pres = st.tabs(["Report", "Presentation"])
    with tab_report:
        _render_doc_generator("report", rows, registry, cfg)
    with tab_pres:
        _render_doc_generator("presentation", rows, registry, cfg)


def validation_plan_page() -> None:
    """Render the validation/calibration/publication plan live from the markdown
    file so the app and the source document never drift apart."""
    st.markdown("# Validation & publication plan")
    st.caption(
        "Rendered live from docs/validation_calibration_plan.md — the single "
        "source of truth for the model's validation status, the honest novelty "
        "assessment, and the publication strategy (calibrate → blind-predict)."
    )
    doc_path = APP_DIR / "docs" / "validation_calibration_plan.md"
    try:
        text = doc_path.read_text(encoding="utf-8")
    except OSError as exc:  # file missing/unreadable
        st.error(
            f"Could not read {doc_path}. The plan lives in the repo at "
            f"docs/validation_calibration_plan.md.\n\n{exc}"
        )
        return
    st.download_button(
        "Download this plan (Markdown)",
        data=text,
        file_name="validation_calibration_plan.md",
        mime="text/markdown",
        key="dl_validation_plan",
    )
    st.divider()
    st.markdown(text)


page = st.sidebar.radio(
    "Tri-HB workspace",
    [
        "Overview",
        "Step 1: Setup, simulator and data",
        "Step 2: Wave model",
        "Step 3: Stress path and analysis",
        "Step 4: Damage model and validation",
        "Step 5: Summary of Results",
        "Step 6: Report and Presentation",
        "Validation & publication plan",
    ],
    key="tri_hb_workspace_page",
)

if st.session_state.get("tri_hb_last_rendered_page") != page:
    st.session_state["tri_hb_last_rendered_page"] = page
    st.rerun()

page_slot = st.empty()
with page_slot.container():
    if page == "Overview":
        overview_page()
    elif page == "Step 1: Setup, simulator and data":
        setup_simulator_and_data_page()
    elif page == "Step 2: Wave model":
        run_legacy_app("wave_damage.py", {"TRI_HB_WORKFLOW_VIEW": "wave"})
    elif page == "Step 3: Stress path and analysis":
        run_legacy_app("wave_damage.py", {"TRI_HB_WORKFLOW_VIEW": "stress"})
    elif page == "Step 4: Damage model and validation":
        run_legacy_app("wave_damage.py", {"TRI_HB_WORKFLOW_VIEW": "damage"})
    elif page == "Step 5: Summary of Results":
        summary_page()
    elif page == "Step 6: Report and Presentation":
        report_presentation_page()
    elif page == "Validation & publication plan":
        validation_plan_page()
