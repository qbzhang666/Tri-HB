"""
Stress-wave, stress-path, damage evolution and DEM validation workspace.

Optimised Streamlit layout for the integrated Tri-HB workflow. The page keeps
all of the original calculations, but the controls are arranged in a Guided /
Advanced workflow so users can work section by section without scrolling
through every model constant.
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Tri-HB Waves, Stress Path and Damage",
    layout="wide",
)

PUB_COLORS = {
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
    "axes.edgecolor": "#222222",
    "axes.labelcolor": "#222222",
    "axes.titlecolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "grid.color": "#D9DEE7",
    "grid.linewidth": 0.55,
    "grid.alpha": 1.0,
    "legend.frameon": False,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
})

# =============================================================================
# Utility functions
# =============================================================================
def hann_window(t, td):
    tau = np.asarray(t)
    g = np.zeros_like(tau, dtype=float)
    mask = (tau >= 0.0) & (tau <= td)
    g[mask] = 0.5 * (1.0 - np.cos(2.0 * np.pi * tau[mask] / max(td, 1e-15)))
    return g


def half_sine_window(t, td):
    tau = np.asarray(t)
    g = np.zeros_like(tau, dtype=float)
    mask = (tau >= 0.0) & (tau <= td)
    g[mask] = np.sin(np.pi * tau[mask] / max(td, 1e-15))
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


def pulse(t, A, td, delay, mode, sign=1.0):
    """Finite-duration compressive pulse, in MPa."""
    tau = t - delay
    return sign * A * get_window(tau, td, mode)


def central_difference(y, t):
    if len(y) < 2:
        return np.zeros_like(y)
    return np.gradient(y, t)


def invariants_from_diagonal(sx, sy, sz):
    p = (sx + sy + sz) / 3.0
    q = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2))
    s1, s2, s3 = sx - p, sy - p, sz - p
    J2 = (s1 ** 2 + s2 ** 2 + s3 ** 2) / 2.0
    J3 = s1 * s2 * s3
    theta = np.zeros_like(p)
    mask = J2 > 1e-12
    arg = np.zeros_like(p)
    arg[mask] = (3.0 * np.sqrt(3.0) / 2.0) * J3[mask] / (J2[mask] ** 1.5)
    arg = np.clip(arg, -1.0, 1.0)
    theta[mask] = (1.0 / 3.0) * np.arccos(arg[mask])
    return p, q, np.rad2deg(theta), J2, J3


def cumulative_trapezoid(y, t):
    out = np.zeros_like(y, dtype=float)
    if len(y) > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))
    return out


def trapz_safe(y, x):
    """NumPy 2.x safe trapezoidal integration.

    np.trapz was removed in newer NumPy releases used by Streamlit Cloud
    on Python 3.14.  Use np.trapezoid when available and keep a manual
    fallback for older/lightweight environments.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 2 or x.size < 2:
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


PUB_DPI = 600  # downloaded / bundled PNGs render at this resolution


def fig_to_bytes(fig, fmt="png", dpi=PUB_DPI):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf


def format_mpa(value):
    return f"{value:.0f} MPa"


def apply_publication_axes(ax, title=None):
    if title is not None:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    right_label = ax.yaxis.get_label_position() == "right"
    ax.tick_params(axis="both", labelsize=10, direction="out", length=3.5, width=0.8)
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)
    ax.grid(True, which="major")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(right_label)
    ax.spines["left"].set_visible(not right_label)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["right"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_frame_on(False)
        for text in legend.get_texts():
            text.set_fontsize(9.5)


def show_publication_figure(fig):
    for ax in fig.axes:
        apply_publication_axes(ax)
    fig.tight_layout()
    st.pyplot(fig)


def show_step_figure(fig, filename, notes=None, equations=None, caption=None):
    """Step-5 style layout for a Steps 2-4 figure.

    Left column: the publication figure (fills the column) with a 600 dpi PNG
    download beneath it. Right column: a short plain-language "What it shows"
    guide. Full width below: an expandable "Governing equations" panel.

    Parameters
    ----------
    fig : matplotlib Figure (already drawn)
    filename : download name for the 600 dpi PNG
    notes : optional markdown shown to the right of the figure
    equations : optional list of LaTeX strings rendered in the equations expander
    caption : optional caption shown directly under the figure
    """
    for ax in fig.axes:
        apply_publication_axes(ax)
    fig.tight_layout()
    png = fig_to_bytes(fig).getvalue()

    fig_col, note_col = st.columns([1.05, 0.95], gap="large")
    with fig_col:
        st.pyplot(fig)
        if caption:
            st.caption(caption)
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
        with st.expander("Governing equations", expanded=False):
            for eq in equations:
                st.latex(eq)
    plt.close(fig)


def annotate_path_direction(ax, xs, ys):
    """Make a time-coloured stress path readable as an out-and-back trajectory.

    Overlays a faded connecting line (so the loading and unloading legs are seen
    as one retraced path), explicit start / peak / end markers, and a small
    arrow showing the loading direction. This removes the visual ambiguity where
    the start point gets overpainted by late-time points at the same location.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 2:
        return
    # faded path line under the coloured scatter
    ax.plot(xs, ys, color="#888888", linewidth=0.8, alpha=0.45, zorder=1)
    # peak point = furthest from the start in (x, y)
    i_peak = int(np.argmax((xs - xs[0]) ** 2 + (ys - ys[0]) ** 2))
    ax.scatter([xs[0]], [ys[0]], s=70, facecolor="white", edgecolor="#1f3864",
               linewidth=1.6, zorder=6, label="start (t=0)")
    ax.scatter([xs[i_peak]], [ys[i_peak]], s=70, marker="^", facecolor="white",
               edgecolor="#c0392b", linewidth=1.6, zorder=6, label="peak")
    ax.scatter([xs[-1]], [ys[-1]], s=80, marker="X", facecolor="white",
               edgecolor="#117a3d", linewidth=1.6, zorder=6, label="end")
    # loading-direction arrow on the first quarter of the rising leg
    j = max(1, i_peak // 4)
    if j < i_peak:
        ax.annotate(
            "", xy=(xs[j + 1], ys[j + 1]), xytext=(xs[j], ys[j]),
            arrowprops=dict(arrowstyle="-|>", color="#1f3864", lw=1.6),
            zorder=6,
        )
    ax.legend(loc="best", fontsize=8, framealpha=0.9)


# =============================================================================
# Page header and style
# =============================================================================
st.markdown(
    """
    <style>
    h1 {
        font-size: clamp(1.75rem, 2.1vw, 2.45rem) !important;
        line-height: 1.12 !important;
        letter-spacing: 0 !important;
        margin-bottom: 0.35rem !important;
    }
    h2 {
        font-size: clamp(1.25rem, 1.45vw, 1.55rem) !important;
        line-height: 1.2 !important;
    }
    h3 {
        font-size: 1.05rem !important;
        line-height: 1.25 !important;
    }
    .step3-card {
        border: 1px solid rgba(120,130,150,0.25);
        border-radius: 0.55rem;
        padding: 0.65rem 0.8rem;
        background: rgba(80, 95, 120, 0.10);
        margin-bottom: 0.55rem;
        font-size: 0.88rem;
        line-height: 1.45;
    }
    .step3-card b { color: #ffffff; }
    .step3-muted { color: #9aa4b2; font-size: 0.84rem; line-height: 1.42; }
    .step3-status {
        border-radius: 0.5rem;
        padding: 0.55rem 0.7rem;
        margin: 0.15rem 0 0.6rem 0;
        font-size: 0.88rem;
        line-height: 1.42;
    }
    .step3-status b { color: #ffffff; }
    .step3-status-success {
        color: #55ef8a;
        background: rgba(34, 139, 82, 0.25);
        border: 1px solid rgba(85, 239, 138, 0.22);
    }
    .step3-status-info {
        color: #a9c7ff;
        background: rgba(64, 105, 170, 0.18);
        border: 1px solid rgba(140, 174, 230, 0.20);
    }
    .step3-status-muted {
        color: #9aa4b2;
        background: rgba(80, 95, 120, 0.08);
        border: 1px solid rgba(120,130,150,0.18);
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', Consolas, monospace;
        font-size: clamp(1.28rem, 1.7vw, 1.95rem) !important;
        line-height: 1.05 !important;
        letter-spacing: 0 !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.74rem !important;
        line-height: 1.12 !important;
        letter-spacing: 0 !important;
        text-transform: none !important;
        color: #a8afbb !important;
        white-space: normal !important;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.72rem !important;
        line-height: 1.1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

workflow_view = globals().get("TRI_HB_WORKFLOW_VIEW", "combined")
if workflow_view not in {"combined", "wave", "stress", "damage"}:
    workflow_view = "combined"

view_titles = {
    "combined": "Stress waves, damage and DEM validation",
    "wave": "Step 2 - Wave model",
    "stress": "Step 3 - Stress path and analysis",
    "damage": "Step 4 - Damage model and validation",
}
view_captions = {
    "combined": (
        "Guided review of wave timing, stress path, damage growth, energy balance, "
        "and DEM/experimental descriptors using the shared Tri-HB setup."
    ),
    "wave": "Check pulse timing, travel time, equilibrium window, wave superposition, and loading regime.",
    "stress": "Review p-q-theta stress paths, stress histories, and comparison with reduced test data.",
    "damage": "Evaluate damage growth, stiffness loss, energy indicators, DEM descriptors, and exportable model data.",
}

st.title(view_titles[workflow_view])
st.caption(view_captions[workflow_view])
if workflow_view in {"wave", "stress"}:
    st.markdown(
        """
        <style>
        div[data-testid="stTabs"] div[role="tablist"] button[role="tab"]:not(:first-child) {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
elif workflow_view == "damage":
    st.markdown(
        """
        <style>
        div[data-testid="stTabs"] div[role="tablist"] button[role="tab"]:nth-child(n+4) {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# Linked setup from the integrated workflow
# =============================================================================
linked_cfg = st.session_state.get("tri_hb_latest_config", {})
linked_result = st.session_state.get("tri_hb_latest_result", {})
linked_reduced = st.session_state.get("tri_hb_reduced_data")
has_linked_design = bool(linked_cfg)

# Defaults inherited from Step 1; sensible standalone values when this page is run alone.
default_E_GPa = float(linked_cfg.get("material_E", 50e9)) / 1e9
default_nu = float(linked_cfg.get("material_nu", 0.25))
default_density = float(linked_cfg.get("material_density", 2650.0))
default_L_mm = float(linked_cfg.get("specimen_length", linked_cfg.get("specimen_size", 0.050))) * 1000.0
default_sx0 = float(linked_cfg.get("confinement_X", 20e6)) / 1e6
default_sy0 = float(linked_cfg.get("confinement_Y", 15e6)) / 1e6
default_sz0 = float(linked_cfg.get("confinement_Z", 10e6)) / 1e6
default_td_us = float(linked_cfg.get("pulse_duration", 60e-6)) * 1e6
default_tmax_us = max(250.0, default_td_us * 1.30)

# ---------------------------------------------------------------------------
# Consistent failure surface derived from the Step-1 strength model.
#
# Step 1 uses an axial dynamic strength
#     sigma_1^peak = (UCS + k_conf * sigma_conf) * DIF
# for triaxial compression with sigma_2 = sigma_3 = sigma_conf.  In invariant
# space (theta = 0, h(theta) = 1) this is q = sigma_1 - sigma_conf and
# p = (sigma_1 + 2 sigma_conf)/3.  Eliminating sigma_conf gives a LINEAR
# Mohr-Coulomb-type surface that reproduces the Step-1 strength exactly:
#     q_f = A + B p,   with   A = 3 UCS_eff/(k_conf+2),  B = 3(k_conf-1)/(k_conf+2),  n = 1,
# where UCS_eff = UCS * DIF carries the same rate hardening as Step 1.  Using
# these derived defaults keeps the Step-3 failure index F = q/q_f consistent
# with the Step-1 stress-strain strength instead of an unrelated hardcoded set.
# Advanced mode still lets the user override A, B, n, a_theta for calibration.
default_UCS_MPa = float(linked_cfg.get("material_UCS", 80e6)) / 1e6
default_k_conf = float(linked_cfg.get("material_k_conf", 4.5))
default_b_rate = float(linked_cfg.get("material_b_rate", 0.18))
default_epsdot_ref = float(linked_cfg.get("material_epsdot_ref", 1e-5))

def derive_failure_surface(ucs_mpa, k_conf, dif=1.0):
    """Return (A_fail, B_fail, n_fail) consistent with the Step-1 strength."""
    k = max(k_conf, 1.0 + 1e-6)
    ucs_eff = max(ucs_mpa, 1e-6) * max(dif, 1e-6)
    A = 3.0 * ucs_eff / (k + 2.0)
    B = 3.0 * (k - 1.0) / (k + 2.0)
    return A, B, 1.0

# DIF at a nominal rate so the static surface anchor carries Step-1 rate
# hardening; the Step-4 strain-rate factor handles the rest dynamically.
_dif_nominal = 1.0 + default_b_rate * np.log10(
    max(1.0, default_epsdot_ref) / default_epsdot_ref
) if default_epsdot_ref > 0 else 1.0
default_A_fail, default_B_fail, default_n_fail = derive_failure_surface(
    default_UCS_MPa, default_k_conf, _dif_nominal
)

linked_peak_mpa = float(linked_cfg.get("peak_stress", 0.0)) / 1e6
linked_peak_x_mpa = float(linked_cfg.get("peak_stress_X", linked_cfg.get("peak_stress", 0.0))) / 1e6
linked_peak_y_mpa = float(linked_cfg.get("peak_stress_Y", linked_cfg.get("peak_stress", 0.0))) / 1e6
linked_peak_z_mpa = float(linked_cfg.get("peak_stress_Z", linked_cfg.get("peak_stress", 0.0))) / 1e6
if linked_peak_mpa <= 0.0 and linked_result:
    linked_peak_mpa = float(linked_result.get("summary", {}).get("peak_incident_MPa", 0.0))
default_A = linked_peak_mpa if has_linked_design and linked_peak_mpa > 0 else 60.0
default_Ax = linked_peak_x_mpa if has_linked_design and linked_peak_x_mpa > 0 else default_A
default_Ay = linked_peak_y_mpa if has_linked_design and linked_peak_y_mpa > 0 else default_A
default_Az = linked_peak_z_mpa if has_linked_design and linked_peak_z_mpa > 0 else default_A

default_delay_y_us = float(linked_cfg.get("pulse_delay_Y", 0.0)) * 1e6
default_delay_z_us = float(linked_cfg.get("pulse_delay_Z", default_delay_y_us * 1e-6)) * 1e6

linked_mode = linked_cfg.get("mode", "")
linked_axes = linked_cfg.get("symmetric_axes", "")
linked_active_axes = str(linked_cfg.get("active_axes", linked_axes or "XYZ"))
path_options = ["Single-sided X", "Symmetric X", "Symmetric XY", "Symmetric XYZ", "Asynchronous XY", "Asynchronous XYZ"]
if linked_mode == "em-symmetric" and linked_axes == "XYZ":
    default_path = "Symmetric XYZ"
elif linked_mode == "em-symmetric" and linked_axes == "XY":
    default_path = "Symmetric XY"
elif linked_mode == "em-symmetric":
    default_path = "Symmetric X"
elif linked_mode == "em-async":
    default_path = "Asynchronous XY" if linked_active_axes == "XY" else "Asynchronous XYZ"
    if linked_active_axes == "X":
        default_path = "Single-sided X"
else:
    default_path = "Single-sided X"

linked_signature = (
    f"{linked_mode}|{linked_axes}|{linked_active_axes}|{default_path}|"
    f"{default_Ax:.6g}|{default_Ay:.6g}|{default_Az:.6g}|{default_td_us:.6g}|"
    f"{default_sx0:.6g}|{default_sy0:.6g}|{default_sz0:.6g}|"
    f"{default_delay_y_us:.6g}|{default_delay_z_us:.6g}|"
    f"{default_E_GPa:.6g}|{default_nu:.6g}|{default_density:.6g}|{default_L_mm:.6g}"
)

# =============================================================================
# Sidebar - simplified control hierarchy
# =============================================================================
with st.sidebar:
    st.header("Shared model controls")
    control_level = st.radio(
        "Control level",
        ["Guided", "Advanced"],
        horizontal=True,
        help="Guided hides model constants. Advanced exposes the full failure and damage law.",
    )

    sync_to_step1 = False
    if has_linked_design:
        sync_to_step1 = st.checkbox("Use Step 1 setup", value=True)
        if sync_to_step1:
            st.success("Material, geometry, prestress and pulse defaults are inherited from Step 1.")
    else:
        st.warning("No Step 1 design is available. Manual setup is shown below.")

    analysis_goal = st.selectbox(
        "Analysis goal",
        [
            "Quick validation",
            "Stress-path focus",
            "Damage calibration",
            "DEM comparison",
        ],
        help="Changes which guidance is highlighted; the calculations remain the same.",
    )

    # Initialise all variables from Step 1 defaults.  The UI below overrides them.
    E_GPa = default_E_GPa
    nu = default_nu
    rho = default_density
    L_mm = default_L_mm
    sx0 = default_sx0
    sy0 = default_sy0
    sz0 = default_sz0
    loading_path = default_path
    td_us = default_td_us
    Ax = default_Ax
    Ay = default_Ay if default_path in ("Symmetric XY", "Symmetric XYZ", "Asynchronous XY", "Asynchronous XYZ") else 0.0
    Az = default_Az if default_path in ("Symmetric XYZ", "Asynchronous XYZ") else 0.0
    pulse_type = "Half-sine" if has_linked_design else "Hann"
    amplitude_ratio = 1.0
    duration_ratio = 1.0
    delay_us = default_delay_y_us
    delay_y_us = default_delay_y_us
    delay_z_us = default_delay_z_us
    central_width = 0.30
    damage_threshold = 0.15

    # Failure surface defaults, derived from the Step-1 strength model so the
    # Step-3 failure index F = q/q_f is consistent with the Step-1 axial
    # strength (see derive_failure_surface above). Advanced mode can override.
    A_fail = default_A_fail
    B_fail = default_B_fail
    n_fail = default_n_fail
    lode_amp = 0.10

    # Damage law defaults; guided presets can change these.
    tau_D_us = 30.0
    alpha_sat = 1.0
    m_over = 2.0
    beta_rate = 0.15
    epsdot0 = 1.0
    F0 = 1.0

    if control_level == "Guided":
        st.divider()
        st.subheader("Essential inputs")
        loading_path = st.selectbox(
            "Loading path",
            path_options,
            index=path_options.index(default_path),
            key=f"guided_loading_path_{linked_signature}" if sync_to_step1 else "guided_loading_path_manual",
            help="Inherited from Step 1 by default; change it here for quick what-if review.",
        )
        if loading_path in ("Symmetric XY", "Symmetric XYZ", "Asynchronous XY", "Asynchronous XYZ") and Ay <= 0.0:
            Ay = default_Ay
        if loading_path in ("Symmetric XYZ", "Asynchronous XYZ") and Az <= 0.0:
            Az = default_Az
        pulse_type = st.selectbox(
            "Pulse shape",
            ["Half-sine", "Hann", "Rectangular"],
            index=0 if has_linked_design else 1,
            key=f"guided_pulse_shape_{linked_signature}" if sync_to_step1 else "guided_pulse_shape_manual",
        )
        delay_us = st.number_input(
            "Secondary-pulse delay, Δt (μs)",
            value=float(default_delay_y_us),
            min_value=0.0,
            step=5.0,
            key=f"guided_delay_{linked_signature}" if sync_to_step1 else "guided_delay_manual",
            help="For symmetric loading this delays the opposing X pulse; for asynchronous loading it is the Y-pulse delay.",
        )
        delay_y_us = delay_us
        delay_z_us = st.number_input(
            "Z-pulse delay for async mode (μs)",
            value=float(default_delay_z_us),
            min_value=0.0,
            step=5.0,
            key=f"guided_delay_z_{linked_signature}" if sync_to_step1 else "guided_delay_z_manual",
        ) if loading_path == "Asynchronous XYZ" else 0.0

        damage_preset = st.selectbox(
            "Damage response preset",
            ["Default rock", "Slower / tougher", "Faster / more brittle", "Calibration-ready"],
            key="guided_damage_preset",
            help="Use Advanced mode to edit individual damage-law constants.",
        )
        if damage_preset == "Slower / tougher":
            tau_D_us, alpha_sat, m_over, beta_rate, F0 = 60.0, 1.2, 2.2, 0.10, 1.15
        elif damage_preset == "Faster / more brittle":
            tau_D_us, alpha_sat, m_over, beta_rate, F0 = 15.0, 0.8, 1.6, 0.20, 0.85
        elif damage_preset == "Calibration-ready":
            tau_D_us, alpha_sat, m_over, beta_rate, F0 = 30.0, 1.0, 2.0, 0.15, 1.0

        resolution = st.select_slider("Resolution", options=["Fast", "Normal", "High"], value="Normal")
        npts = {"Fast": 2000, "Normal": 5000, "High": 10000}[resolution]
        tmax_us = max(default_tmax_us, td_us * 1.3, delay_us + td_us * 1.15, delay_z_us + td_us * 1.15)

        # A compact manual override for guided mode. When linked to Step 1, keep
        # these widgets inactive unless the user explicitly unlocks them; closed
        # Streamlit widgets still retain old values and would otherwise hide a
        # stale Ax/prestress behind a fresh Step 1 setup.
        manual_override = not sync_to_step1
        if sync_to_step1:
            manual_override = st.checkbox(
                "Unlock manual overrides",
                value=False,
                key=f"guided_unlock_overrides_{linked_signature}",
                help="Leave off to use Step 1 material, pulse and prestress exactly.",
            )
        with st.expander("Manual setup / quick overrides", expanded=manual_override):
            if manual_override:
                override_key = linked_signature if sync_to_step1 else "manual"
                E_GPa = st.number_input("Young's modulus, E (GPa)", value=E_GPa, min_value=1.0, step=1.0, key=f"guided_E_{override_key}")
                nu = st.number_input("Poisson's ratio, ν", value=nu, min_value=0.0, max_value=0.49, step=0.01, key=f"guided_nu_{override_key}")
                rho = st.number_input("Density, ρ (kg/m³)", value=rho, min_value=1000.0, step=50.0, key=f"guided_rho_{override_key}")
                L_mm = st.number_input("Specimen length, L (mm)", value=L_mm, min_value=1.0, step=1.0, key=f"guided_L_{override_key}")
                sx0 = st.number_input("σx0 (MPa)", value=sx0, min_value=0.0, step=1.0, key=f"guided_sx0_{override_key}")
                sy0 = st.number_input("σy0 (MPa)", value=sy0, min_value=0.0, step=1.0, key=f"guided_sy0_{override_key}")
                sz0 = st.number_input("σz0 (MPa)", value=sz0, min_value=0.0, step=1.0, key=f"guided_sz0_{override_key}")
                td_us = st.number_input("Pulse duration, td (μs)", value=td_us, min_value=1.0, step=5.0, key=f"guided_td_{override_key}")
                Ax = st.number_input("Ax single-pulse amplitude (MPa)", value=Ax, min_value=0.0, step=5.0, key=f"guided_Ax_{override_key}")
                Ay = st.number_input("Ay single-pulse amplitude (MPa)", value=Ay, min_value=0.0, step=5.0, key=f"guided_Ay_{override_key}")
                Az = st.number_input("Az single-pulse amplitude (MPa)", value=Az, min_value=0.0, step=5.0, key=f"guided_Az_{override_key}")
                tmax_us = max(default_tmax_us, td_us * 1.3, delay_us + td_us * 1.15, delay_z_us + td_us * 1.15)
            else:
                st.caption("Linked Step 1 values are active. Unlock overrides to edit these values for a what-if run.")
                st.dataframe(
                    pd.DataFrame(
                        [
                            ("Ax pulse amplitude", f"{Ax:.1f} MPa"),
                            ("Prestress σx0/σy0/σz0", f"{sx0:.1f}/{sy0:.1f}/{sz0:.1f} MPa"),
                            ("Pulse duration", f"{td_us:.1f} μs"),
                        ],
                        columns=["Linked input", "Value"],
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

        with st.expander("Descriptor options", expanded=False):
            damage_threshold = st.slider("Damage threshold", 0.01, 0.9, damage_threshold, step=0.01)
            central_width = st.slider("Central region fraction", 0.1, 0.8, central_width, step=0.05)

    else:
        st.divider()
        with st.expander("1. Specimen, material and prestress", expanded=True):
            E_GPa = st.number_input("Young's modulus, E (GPa)", value=E_GPa, min_value=1.0, step=1.0)
            nu = st.number_input("Poisson's ratio, ν", value=nu, min_value=0.0, max_value=0.49, step=0.01)
            rho = st.number_input("Density, ρ (kg/m³)", value=rho, min_value=1000.0, step=50.0)
            L_mm = st.number_input("Specimen length, L (mm)", value=L_mm, min_value=1.0, step=1.0)
            sx0 = st.number_input("σx0 (MPa)", value=sx0, min_value=0.0, step=1.0)
            sy0 = st.number_input("σy0 (MPa)", value=sy0, min_value=0.0, step=1.0)
            sz0 = st.number_input("σz0 (MPa)", value=sz0, min_value=0.0, step=1.0)

        with st.expander("2. Loading and wave timing", expanded=True):
            loading_path = st.selectbox("Loading path", path_options, index=path_options.index(default_path))
            pulse_type = st.selectbox("Pulse envelope", ["Half-sine", "Hann", "Rectangular"], index=0 if has_linked_design else 1)
            td_us = st.number_input("Pulse duration, td (μs)", value=td_us, min_value=1.0, step=5.0)
            Ax = st.number_input("Ax single-pulse amplitude (MPa)", value=Ax, min_value=0.0, step=5.0)
            Ay = st.number_input("Ay single-pulse amplitude (MPa)", value=Ay, min_value=0.0, step=5.0)
            Az = st.number_input("Az single-pulse amplitude (MPa)", value=Az, min_value=0.0, step=5.0)
            amplitude_ratio = st.number_input("Right/left amplitude ratio in X", value=1.0, min_value=0.0, step=0.1)
            duration_ratio = st.number_input("Right/left pulse-duration ratio in X", value=1.0, min_value=0.1, step=0.1)
            delay_us = st.number_input("Right/secondary X delay, Δt (μs)", value=float(default_delay_y_us), min_value=0.0, step=5.0)
            delay_y_us = st.number_input("Y-pulse delay for async mode (μs)", value=float(default_delay_y_us), min_value=0.0, step=5.0)
            delay_z_us = st.number_input("Z-pulse delay for async mode (μs)", value=float(default_delay_z_us), min_value=0.0, step=5.0)
            tmax_us = st.number_input("Simulation window (μs)", value=max(default_tmax_us, delay_z_us + td_us * 1.15), min_value=20.0, step=10.0)
            npts = st.slider("Time points", 1000, 30000, 5000, step=1000)

        with st.expander("3. Failure surface", expanded=False):
            st.caption(
                f"Defaults are derived from the Step-1 strength model "
                f"(UCS={default_UCS_MPa:.0f} MPa, k_conf={default_k_conf:.1f}) so that "
                f"F=q/q_f is consistent with the Step-1 axial strength: "
                f"A={default_A_fail:.1f} MPa, B={default_B_fail:.2f}, n=1. "
                f"Override only for independent calibration."
            )
            use_derived_surface = st.checkbox(
                "Use Step-1-consistent surface", value=True,
                help="Keep on so the failure index matches the Step-1 stress-strain strength. "
                     "Turn off to enter calibrated A, B, n by hand.",
            )
            if use_derived_surface:
                A_fail, B_fail, n_fail = default_A_fail, default_B_fail, default_n_fail
                st.latex(rf"q_f = {A_fail:.1f} + {B_fail:.2f}\,p\quad(\mathrm{{n}}=1)")
            else:
                A_fail = st.number_input("A in qf = (A+Bpⁿ)h(θ) (MPa)", value=A_fail, min_value=0.0, step=1.0)
                B_fail = st.number_input("B in qf", value=B_fail, min_value=0.0, step=0.1)
                n_fail = st.number_input("n in qf", value=n_fail, min_value=0.1, step=0.05)
            lode_amp = st.number_input("Lode-angle amplitude, aθ", value=lode_amp, min_value=0.0, step=0.02)

        with st.expander("4. Damage law and descriptors", expanded=False):
            tau_D_us = st.number_input("Damage time scale, τD (μs)", value=tau_D_us, min_value=0.1, step=5.0)
            alpha_sat = st.number_input("Saturation exponent, α", value=alpha_sat, min_value=0.0, step=0.2)
            m_over = st.number_input("Overstress exponent, m", value=m_over, min_value=0.1, step=0.2)
            beta_rate = st.number_input("Rate exponent, β", value=beta_rate, min_value=0.0, step=0.05)
            epsdot0 = st.number_input("Reference strain rate, ε̇0 (s⁻¹)", value=epsdot0, min_value=1e-6, step=1.0)
            F0 = st.number_input("Failure-index normalisation, F0", value=F0, min_value=0.1, step=0.1)
            damage_threshold = st.slider("Damage threshold", 0.01, 0.9, damage_threshold, step=0.01)
            central_width = st.slider("Central region fraction", 0.1, 0.8, central_width, step=0.05)

# Make sure runtime variables exist in all branches.
tmax_us = float(locals().get("tmax_us", max(default_tmax_us, td_us * 1.3, delay_us + td_us * 1.15, delay_z_us + td_us * 1.15)))
npts = int(locals().get("npts", 5000))

# =============================================================================
# Calculations
# =============================================================================
E = E_GPa * 1e9
G = E / (2.0 * (1.0 + nu))
M = E * (1.0 - nu) / ((1.0 + nu) * max(1.0 - 2.0 * nu, 1e-9))
cp0 = np.sqrt(M / rho)
cs = np.sqrt(G / rho)
L_m = L_mm / 1000.0
t_travel = L_m / cp0 if cp0 > 0 else np.nan
t_eq_low = 3.0 * t_travel
t_eq_high = 5.0 * t_travel

t_us = np.linspace(0.0, tmax_us, npts)
t = t_us * 1e-6
td = max(td_us, 1e-9) * 1e-6
delay = delay_us * 1e-6
delay_y = delay_y_us * 1e-6
delay_z = delay_z_us * 1e-6
td_right = td * duration_ratio

x_left = pulse(t, Ax, td, 0.0, pulse_type)
x_right = np.zeros_like(t)
y_drive = np.zeros_like(t)
z_drive = np.zeros_like(t)

if loading_path in ["Symmetric X", "Symmetric XY", "Symmetric XYZ"]:
    x_right = pulse(t, Ax * amplitude_ratio, td_right, delay, pulse_type)
elif loading_path in ["Asynchronous XY", "Asynchronous XYZ"]:
    # X is the reference pulse. Y and Z arrive later and control the sequential path.
    x_right = np.zeros_like(t)

if loading_path in ["Symmetric XY", "Symmetric XYZ"]:
    y_drive = 2.0 * pulse(t, Ay, td, 0.0, pulse_type)
elif loading_path in ["Asynchronous XY", "Asynchronous XYZ"]:
    y_drive = pulse(t, Ay, td, delay_y, pulse_type)

if loading_path == "Symmetric XYZ":
    z_drive = 2.0 * pulse(t, Az, td, 0.0, pulse_type)
elif loading_path == "Asynchronous XYZ":
    z_drive = pulse(t, Az, td, delay_z, pulse_type)

sx_dyn = x_left + x_right if loading_path != "Single-sided X" else x_left
sx = sx0 + sx_dyn
sy = sy0 + y_drive
sz = sz0 + z_drive

p, q, theta_deg, J2, J3 = invariants_from_diagonal(sx, sy, sz)

# Superposition factor.  For asynchronous cases this is the X-pair factor; the
# delay regime still uses the first secondary pulse arrival.
g_left_centre = get_window(t - (L_m / (2.0 * cp0)), td, pulse_type)
if loading_path in ["Symmetric X", "Symmetric XY", "Symmetric XYZ"]:
    g_right_centre = get_window(t - delay - (L_m / (2.0 * cp0)), td_right, pulse_type)
    sigma_centre = sx0 + Ax * g_left_centre + Ax * amplitude_ratio * g_right_centre
    eta_sup = (np.max(sigma_centre) - sx0) / max(Ax, 1e-9)
    regime_delay = delay
elif loading_path in ["Asynchronous XY", "Asynchronous XYZ"]:
    g_y_centre = get_window(t - delay_y - (L_m / (2.0 * cp0)), td, pulse_type)
    sigma_centre = sx0 + Ax * g_left_centre + Ay * g_y_centre
    active_delays = [delay_y] if delay_y > 0 else []
    if loading_path == "Asynchronous XYZ":
        g_z_centre = get_window(t - delay_z - (L_m / (2.0 * cp0)), td, pulse_type)
        sigma_centre = sigma_centre + Az * g_z_centre
        if delay_z > 0:
            active_delays.append(delay_z)
    eta_sup = max(1.0, (np.max(sigma_centre) - sx0) / max(Ax, 1e-9))
    regime_delay = min(active_delays) if active_delays else 0.0
else:
    sigma_centre = sx0 + Ax * g_left_centre
    eta_sup = (np.max(sigma_centre) - sx0) / max(Ax, 1e-9)
    regime_delay = 0.0

dt_star = regime_delay / t_travel if t_travel and t_travel > 0 else np.nan
if dt_star < 1:
    regime = "Synchronous wave superposition"
    regime_short = "Synchronous"
elif dt_star <= 3:
    regime = "Reverberation-coupled interaction"
    regime_short = "Reverberation"
elif dt_star <= 10:
    regime = "Transitional / decaying reverberations"
    regime_short = "Transitional"
else:
    regime = "Sequential, damage-memory controlled"
    regime_short = "Sequential"

# Failure envelope and damage law.
# F_index is the *undamaged* failure index q / q_f (what the applied stress would
# reach with no degradation).  Damage growth, however, is driven by the EFFECTIVE
# stress carried by the intact material fraction, (1 - D) q.  This load-shedding
# feedback makes the damage self-limit instead of snapping to 1: once the material
# softens, the effective overstress drops and the damage rate falls.
theta_rad = np.deg2rad(theta_deg)
h_theta = 1.0 + lode_amp * (1.0 - np.cos(3.0 * theta_rad))
qf = (A_fail + B_fail * np.maximum(p, 0.0) ** n_fail) * h_theta
qf_safe = np.maximum(qf, 1e-9)
F_index = q / qf_safe

E_MPa = E_GPa * 1000.0
eps_x = sx / E_MPa
eps_y = sy / E_MPa
eps_z = sz / E_MPa
epsdot_x = central_difference(eps_x, t)
epsdot_y = central_difference(eps_y, t)
epsdot_z = central_difference(eps_z, t)
epsdot_eq = np.sqrt(
    2.0 / 3.0 * ((epsdot_x - epsdot_y) ** 2 + (epsdot_y - epsdot_z) ** 2 + (epsdot_z - epsdot_x) ** 2) / 2.0
)
rate_factor = (np.maximum(np.abs(epsdot_eq), 1e-12) / epsdot0) ** beta_rate

# Recoverable elastic energy density of the intact material (energy release rate Y).
W_el = (1.0 / (2.0 * E_MPa)) * (
    sx ** 2 + sy ** 2 + sz ** 2 - 2.0 * nu * (sx * sy + sy * sz + sz * sx)
)

tau_D = tau_D_us * 1e-6
D = np.zeros_like(t)
Ddot = np.zeros_like(t)
F_eff = np.zeros_like(t)            # damage-coupled (effective) failure index
W_diss = np.zeros_like(t)          # cumulative damage dissipation, integral of Y dD
for i in range(1, len(t)):
    # Effective failure index uses the stress carried by the intact fraction.
    F_eff[i - 1] = (1.0 - D[i - 1]) * F_index[i - 1]
    overstress = max((F_eff[i - 1] - 1.0) / F0, 0.0)
    Ddot[i - 1] = ((1.0 - D[i - 1]) ** alpha_sat) / tau_D * (overstress ** m_over) * rate_factor[i - 1]
    dD = Ddot[i - 1] * (t[i] - t[i - 1])
    D[i] = np.clip(D[i - 1] + dD, 0.0, 1.0)
    # Continuum-damage-mechanics dissipation: energy released as the modulus
    # degrades, W_diss = integral of Y dD with Y = W_el (>= 0 by construction).
    W_diss[i] = W_diss[i - 1] + W_el[i - 1] * (D[i] - D[i - 1])
if len(Ddot) > 1:
    Ddot[-1] = Ddot[-2]
    F_eff[-1] = (1.0 - D[-1]) * F_index[-1]

E_D = E_GPa * (1.0 - D)
cp_D = cp0 * np.sqrt(np.maximum(1.0 - D, 0.0))

# Input work done by the applied stress on the (elastic) strain field.
power = sx * epsdot_x + sy * epsdot_y + sz * epsdot_z
W_input = cumulative_trapezoid(power, t)
# Damage dissipation is the physically meaningful dissipated energy here.
W_diss_estimate = W_diss

# Synthetic descriptors: planning indicators for DEM / experimental validation.
x = np.linspace(0, 1, 400)
centre = 0.5
left_damage = np.exp(-((x - 0.20) / 0.12) ** 2) * np.max(D)
right_scale = amplitude_ratio if loading_path in ["Symmetric X", "Symmetric XY", "Symmetric XYZ"] else 1.0
right_damage = np.exp(-((x - 0.80) / 0.12) ** 2) * np.max(D) * right_scale
central_damage = np.exp(-((x - 0.50) / 0.16) ** 2) * np.max(D) * min(eta_sup, 2.0) / 2.0
delay_scatter = (1.0 / (1.0 + np.exp(-(dt_star - 5.0)))) * np.max(D) * 0.35 if np.isfinite(dt_star) else 0.0
damage_profile = np.clip(left_damage + right_damage + central_damage + delay_scatter, 0, 1)

central_mask = np.abs(x - centre) < central_width / 2.0
total_damage_area = trapz_safe(damage_profile, x)
central_damage_area = trapz_safe(damage_profile[central_mask], x[central_mask]) if np.any(central_mask) else 0.0
D_c = central_damage_area / max(total_damage_area, 1e-12)
D_left = trapz_safe(damage_profile[x < 0.5], x[x < 0.5])
D_right = trapz_safe(damage_profile[x >= 0.5], x[x >= 0.5])
S_x = 1.0 - abs(D_left - D_right) / max(D_left + D_right, 1e-12)
neutral_width_estimate = np.clip(
    1.0 / (1.0 + 0.5 * max(dt_star, 0.0)) * (1.0 / max(1.0, abs(amplitude_ratio - 1.0) + 1.0)),
    0,
    1,
)

# Dataframe used for export and optional downstream use.
df_results = pd.DataFrame({
    "time_us": t_us,
    "x_left_MPa": x_left,
    "x_right_MPa": x_right,
    "y_drive_MPa": y_drive,
    "z_drive_MPa": z_drive,
    "sigma_x_MPa": sx,
    "sigma_y_MPa": sy,
    "sigma_z_MPa": sz,
    "p_MPa": p,
    "q_MPa": q,
    "theta_deg": theta_deg,
    "qf_MPa": qf,
    "failure_index": F_index,
    "failure_index_eff": F_eff,
    "epsdot_eq_s-1": epsdot_eq,
    "D": D,
    "Ddot_s-1": Ddot,
    "E_D_GPa": E_D,
    "cp_D_m_s": cp_D,
    "W_input_MJ_m3": W_input,
    "W_elastic_MJ_m3": W_el,
    "W_diss_estimate_MJ_m3": W_diss_estimate,
})
st.session_state["tri_hb_step3_damage_data"] = df_results

# Scalar summary for the Step 5 publication-figure page.
st.session_state["tri_hb_wave_damage_summary"] = {
    "linked_signature": linked_signature if sync_to_step1 else None,
    "loading_path": loading_path,
    "pulse_Ax_MPa": float(Ax),
    "prestress_x_MPa": float(sx0),
    "prestress_y_MPa": float(sy0),
    "prestress_z_MPa": float(sz0),
    "regime": regime,
    "t_travel_us": float(t_travel * 1e6),
    "t_eq_low_us": float(t_eq_low * 1e6),
    "t_eq_high_us": float(t_eq_high * 1e6),
    "dt_star": float(dt_star),
    "eta_sup": float(eta_sup),
    "peak_p_MPa": float(np.max(p)),
    "peak_q_MPa": float(np.max(q)),
    "peak_F": float(np.max(F_index)),
    "D_final": float(D[-1]),
    "D_c": float(D_c),
    "S_x": float(S_x),
}

# =============================================================================
# Top status and summary
# =============================================================================
link_c1, link_c2 = st.columns([1.05, 0.95])
with link_c1:
    if has_linked_design and sync_to_step1:
        st.markdown(
            f"""
            <div class="step3-status step3-status-success">
            Step 1 linked: prestress <b>{default_sx0:.0f}/{default_sy0:.0f}/{default_sz0:.0f} MPa</b>;
            pulse X/Y/Z <b>{default_Ax:.0f}/{default_Ay:.0f}/{default_Az:.0f} MPa</b>,
            <b>{default_td_us:.0f} &micro;s</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif has_linked_design:
        st.markdown(
            """
            <div class="step3-status step3-status-info">
            Step 1 result exists; visible override values are active.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="step3-status step3-status-info">
            Standalone mode: use the manual setup panel in the sidebar.
            </div>
            """,
            unsafe_allow_html=True,
        )

with link_c2:
    if linked_reduced is not None and len(linked_reduced) > 0:
        st.markdown(
            f"""
            <div class="step3-status step3-status-info">
            Reduced data ready: peak stress <b>{linked_reduced['stress_MPa'].max():.1f} MPa</b>;
            peak strain <b>{100.0 * linked_reduced['strain'].max():.3f}%</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="step3-status step3-status-muted">
            Reduced stress-strain data has not been generated in Step 1 yet.
            </div>
            """,
            unsafe_allow_html=True,
        )

m1, m2, m3, m4, m5 = st.columns([1.05, 1.25, 0.75, 0.75, 0.85])
m1.metric("Travel time", f"{t_travel * 1e6:.2f} μs", help=f"cp = {cp0:.0f} m/s")
m2.metric("Regime", regime_short, help=regime)
m3.metric("Δt*", f"{dt_star:.2f}")
m4.metric("ηsup", f"{eta_sup:.2f}")
m5.metric("Final D", f"{D[-1]:.3f}", help=f"Central fraction Dc = {D_c:.3f}")

st.markdown(
    """
    <div class="step3-card">
    <b>Workflow link:</b> Step 1 defines the shared test setup and reduced data.
    Step 2 checks the wave model, Step 3 reviews the stress path, and Step 4 evaluates
    damage, energy, and DEM/experimental validation descriptors. Detailed constants
    are hidden unless <b>Advanced</b> is selected in the sidebar.
    </div>
    """,
    unsafe_allow_html=True,
)

equation_title = {
    "wave": "Equations used in Step 2: wave timing and regime",
    "stress": "Equations used in Step 3: stress path and failure index",
    "damage": "Equations used in Step 4: damage, energy and validation",
}.get(workflow_view, "Equations used by this workflow")

with st.expander(equation_title, expanded=False):
    if workflow_view in {"wave", "combined"}:
        st.markdown("**Step 2: elastic wave speeds and equilibrium window**")
        st.latex(r"M=\frac{E(1-\nu)}{(1+\nu)(1-2\nu)},\quad c_p=\sqrt{M/\rho},\quad t_{travel}=L/c_p")
        st.latex(r"t_{eq}\approx 3\text{--}5\,t_{travel},\qquad \Delta t^*=\Delta t/t_{travel}")
        st.latex(r"\sigma_{\mathrm{dyn}}(t)=A\,g(t-\Delta t)")
        st.caption("The pulse envelope g(t) is the selected half-sine, Hann, or rectangular waveform.")

    if workflow_view in {"stress", "combined"}:
        st.markdown("**Step 3: total stress histories, invariants and failure index**")
        st.latex(r"\sigma_x(t)=\sigma_{x0}+\Delta\sigma_x(t),\quad \sigma_y(t)=\sigma_{y0}+\Delta\sigma_y(t),\quad \sigma_z(t)=\sigma_{z0}+\Delta\sigma_z(t)")
        st.latex(r"\Delta\sigma_x=x_{\mathrm{left}}(t)+x_{\mathrm{right}}(t),\quad \Delta\sigma_y=y_{\mathrm{drive}}(t),\quad \Delta\sigma_z=z_{\mathrm{drive}}(t)")
        st.latex(r"p=\tfrac{1}{3}(\sigma_x+\sigma_y+\sigma_z),\qquad q=\sqrt{\tfrac{1}{2}[(\sigma_x-\sigma_y)^2+(\sigma_y-\sigma_z)^2+(\sigma_z-\sigma_x)^2]}")
        st.latex(r"J_2=\tfrac{1}{2}(s_x^2+s_y^2+s_z^2),\quad J_3=s_xs_ys_z,\quad s_i=\sigma_i-p")
        st.latex(r"\cos(3\theta)=\frac{3\sqrt{3}}{2}\frac{J_3}{J_2^{3/2}}")
        st.latex(r"q_f=(A+Bp^n)\,[1+a_\theta(1-\cos 3\theta)],\qquad F=q/q_f")
        st.caption("Step 3 uses total stresses: static prestress plus dynamic pulse increments.")
        st.markdown("**How to choose failure-surface parameters**")
        st.markdown(
            """
            - **A, B, n** define the pressure-dependent failure envelope and should be fitted from triaxial compression/extension strengths or a calibrated DEM failure surface.
            - **aθ** controls Lode-angle sensitivity; set it to zero if no true-triaxial or Lode-angle calibration is available.
            """
        )

    if workflow_view in {"damage", "combined"}:
        st.markdown("**Step 4: damage variable and rate law (with load-shedding feedback)**")
        st.latex(r"F(t)=q(t)/q_f(t),\qquad F_{\mathrm{eff}}=(1-D)\,F,\qquad \langle x\rangle=\max(x,0)")
        st.latex(r"\dot D=\frac{(1-D)^\alpha}{\tau_D}\left\langle\frac{F_{\mathrm{eff}}-1}{F_0}\right\rangle^m\left(\frac{|\dot\varepsilon_{eq}|}{\dot\varepsilon_0}\right)^\beta,\qquad 0\le D\le 1")
        st.latex(r"D_i=\operatorname{clip}\!\left[D_{i-1}+\dot D_{i-1}(t_i-t_{i-1}),\,0,\,1\right]")
        st.latex(r"E(D)=E_0(1-D),\qquad c_p(D)=c_{p0}\sqrt{\max(1-D,0)}")
        st.caption(
            "Damage is driven by the EFFECTIVE failure index (1-D)F, i.e. the stress "
            "carried by the intact fraction. This load-shedding makes D rise smoothly "
            "and self-limit near 1 - 1/F_peak instead of snapping to 1."
        )
        st.markdown("**Step 4: energy-density indicators**")
        st.latex(r"\varepsilon_i=\sigma_i/E_0,\qquad P(t)=\sigma_x\dot\varepsilon_x+\sigma_y\dot\varepsilon_y+\sigma_z\dot\varepsilon_z")
        st.latex(r"W_{\mathrm{input}}(t)=\int_0^t P(\tau)\,d\tau")
        st.latex(r"W_{\mathrm{el}}=\frac{1}{2E_0}\left[\sigma_x^2+\sigma_y^2+\sigma_z^2-2\nu(\sigma_x\sigma_y+\sigma_y\sigma_z+\sigma_z\sigma_x)\right]")
        st.latex(r"W_{\mathrm{diss}}(t)=\int_0^t Y\,\dot D\,d\tau,\qquad Y=W_{\mathrm{el}}\ \ (\text{energy release rate})")
        st.markdown("**Step 4: validation descriptors and delay sensitivity**")
        st.latex(r"D_c=\frac{\int_{\mathrm{centre}}D(x)\,dx}{\int_0^1D(x)\,dx},\qquad S_x=1-\frac{|D_{\mathrm{left}}-D_{\mathrm{right}}|}{D_{\mathrm{left}}+D_{\mathrm{right}}}")
        st.latex(r"\eta_{\mathrm{sup}}(\Delta t^*)=\frac{\max_t\sigma_{\mathrm{centre}}(t;\Delta t^*)-\sigma_{x0}}{A_x}")
        st.latex(r"C_{\mathrm{wave}}=\exp[-(\Delta t^*/1.2)^2],\qquad C_{\mathrm{damage}}=\frac{1}{1+\exp[-(\Delta t^*-5.0)/1.3]}")
        st.latex(r"I_{\mathrm{final}}=\operatorname{clip}(0.15+0.35C_{\mathrm{wave}}+0.25C_{\mathrm{damage}},0,1)")
        st.latex(r"I_{\mathrm{central}}=\operatorname{clip}(0.75C_{\mathrm{wave}}+0.25(1-C_{\mathrm{damage}}),0,1)")
        st.markdown(
            """
            - **τD, F0, α, m, β, εdot0** are damage-evolution calibration parameters.
            - **I_final** and **I_central** are normalised heuristic trends used in the delay-sensitivity plot, not independent damage-law solutions.
            - **ηsup** is a pulse-superposition ratio; it can exceed 1 and is not itself a damage variable.
            """
        )

# =============================================================================
# Main tabs - ordered by the selected top-level workflow step
# =============================================================================
tab_order = {
    "combined": [
        ("overview", "Wave model"),
        ("stress", "Stress path"),
        ("damage", "Damage evolution"),
        ("validation", "Energy + validation"),
        ("export", "Sensitivity + export"),
    ],
    "wave": [
        ("overview", "Wave model"),
        ("stress", "Stress path"),
        ("damage", "Damage evolution"),
        ("validation", "Energy + validation"),
        ("export", "Sensitivity + export"),
    ],
    "stress": [
        ("stress", "Stress path"),
        ("overview", "Wave model"),
        ("damage", "Damage evolution"),
        ("validation", "Energy + validation"),
        ("export", "Sensitivity + export"),
    ],
    "damage": [
        ("damage", "Damage evolution"),
        ("validation", "Energy + validation"),
        ("export", "Sensitivity + export"),
        ("overview", "Wave model"),
        ("stress", "Stress path"),
    ],
}[workflow_view]
created_tabs = st.tabs([label for _, label in tab_order])
tabs_by_key = {key: tab for (key, _), tab in zip(tab_order, created_tabs)}
tab_overview = tabs_by_key["overview"]
tab_stress = tabs_by_key["stress"]
tab_damage = tabs_by_key["damage"]
tab_validation = tabs_by_key["validation"]
tab_export = tabs_by_key["export"]

with tab_overview:
    st.subheader("Wave timing and regime check")
    st.markdown("**Input pulses**")
    fig1, ax1 = plt.subplots(figsize=(12, 5.4))
    ax1.plot(t_us, x_left, label="X reference pulse")
    if np.any(np.abs(x_right) > 0):
        ax1.plot(t_us, x_right, label="X opposing/secondary pulse")
    if np.any(np.abs(y_drive) > 0):
        ax1.plot(t_us, y_drive, label="Y dynamic drive")
    if np.any(np.abs(z_drive) > 0):
        ax1.plot(t_us, z_drive, label="Z dynamic drive")
    ax1.axvline(t_travel * 1e6, color="#888", linestyle="--", linewidth=1.2, label="travel time")
    ax1.axvspan(t_eq_low * 1e6, t_eq_high * 1e6, color="#9ec7ff", alpha=0.35, label="3-5 travel times")
    if regime_delay > 0:
        ax1.axvline(regime_delay * 1e6, linestyle=":", linewidth=1.5, label="secondary delay")
    ax1.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax1.set_ylabel(r"Dynamic stress increment, $\Delta\sigma$ (MPa)")
    ax1.set_title("Input pulses")
    ax1.legend(ncol=2, fontsize=9)
    ax1.grid(True, alpha=0.3)
    show_step_figure(
        fig1, "tri_hb_step2_input_pulses.png",
        notes=(
            "**What it shows.** The dynamic stress pulse(s) entering the specimen "
            "on each active axis, with the shaded equilibrium window.\n\n"
            "- The pulse must be **longer** than the equilibrium window for valid "
            "wave analysis.\n"
            "- The dashed line marks the single bar **travel time**; the band is "
            "3-5 travel times (when the specimen reaches dynamic equilibrium).\n"
            "- Wave speed and the window come from the P-wave modulus M."
        ),
        equations=[
            r"M=\frac{E(1-\nu)}{(1+\nu)(1-2\nu)},\qquad c_p=\sqrt{M/\rho}",
            r"t_{\mathrm{travel}}=\frac{L}{c_p},\qquad t_{\mathrm{eq}}\approx 3\text{--}5\,t_{\mathrm{travel}}",
            r"\Delta t^\ast=\frac{\Delta t}{t_{\mathrm{travel}}}",
            r"\sigma_{\mathrm{dyn}}(t)=A\,g(t-\Delta t)\quad(g=\text{half-sine, Hann or rectangular})",
        ],
    )

    st.markdown("**Current interpretation**")
    ic1, ic2, ic3, ic4 = st.columns(4)
    ic1.metric("Loading path", loading_path)
    ic2.metric("Regime", regime)
    ic3.metric("Equilibrium window", f"{t_eq_low * 1e6:.1f}-{t_eq_high * 1e6:.1f} µs")
    ic4.metric("Peak dyn. stress X", f"{np.max(sx_dyn):.1f} MPa")
    st.write(
        f"Peak dynamic stress inputs: X **{np.max(sx_dyn):.1f} MPa**, "
        f"Y **{np.max(y_drive):.1f} MPa**, Z **{np.max(z_drive):.1f} MPa**"
    )
    if analysis_goal == "Quick validation":
        st.info("Check that the main pulse length is longer than the equilibrium window before interpreting stress-strain or damage.")
    elif analysis_goal == "Stress-path focus":
        st.info("Use the Stress path tab to verify whether the case is hydrostatic, deviatoric, or sequential-path dominated.")
    elif analysis_goal == "Damage calibration":
        st.info("Use Advanced mode only after the wave timing and reduced stress curve look reasonable.")
    else:
        st.info("Compare final D, Dc, Sx, and the energy trend against DEM bond-breakage and dissipated-energy outputs.")

    fig2, ax2 = plt.subplots(figsize=(10, 4.1))
    dt_grid = np.linspace(0, 20, 400)
    wave_control = np.exp(-(dt_grid / 1.2) ** 2)
    damage_control = 1.0 / (1.0 + np.exp(-(dt_grid - 5.0) / 1.3))
    ax2.plot(dt_grid, wave_control, label="Wave-interaction control")
    ax2.plot(dt_grid, damage_control, label="Damage-memory control")
    ax2.axvline(dt_star, linestyle="--", linewidth=1.5, label="Current Δt*")
    ax2.axvspan(0, 1, alpha=0.12)
    ax2.axvspan(1, 3, alpha=0.10)
    ax2.axvspan(3, 10, alpha=0.08)
    ax2.axvspan(10, 20, alpha=0.08)
    ax2.text(0.5, 1.03, "Superposition", ha="center", fontsize=9)
    ax2.text(2.0, 1.03, "Reverberation", ha="center", fontsize=9)
    ax2.text(6.5, 1.03, "Transitional", ha="center", fontsize=9)
    ax2.text(15, 1.03, "Sequential", ha="center", fontsize=9)
    ax2.set_xlim(0, 20)
    ax2.set_ylim(0, 1.1)
    ax2.set_xlabel(r"Normalised delay, $\Delta t^*=\Delta t/t_{\mathrm{travel}}$")
    ax2.set_ylabel("Control indicator (0-1)")
    ax2.set_title("Wave-to-damage transition map")
    ax2.grid(True, alpha=0.35)
    ax2.legend()
    show_step_figure(
        fig2, "tri_hb_step2_regime_map.png",
        caption=("The two smooth curves are heuristic guide curves, not calibrated "
                 "damage laws. The dashed line is the current normalised delay marker."),
        notes=(
            "**What it shows.** Which mechanism is expected to control the response "
            "as a function of the normalised delay between pulses, Δt*.\n\n"
            "- **Δt* < 1** -> waves still overlap (superposition).\n"
            "- **1-3** -> reverberation-coupled.\n"
            "- **3-10** -> transitional.\n"
            "- **> 10** -> sequential, damage-memory controlled.\n\n"
            "The dashed vertical line is your current case. Only the first two "
            "legend items are curves; the y-axis is a unitless 0-1 dominance "
            "indicator."
        ),
        equations=[
            r"\Delta t^\ast=\Delta t/t_{\mathrm{travel}}",
            r"C_{\mathrm{wave}}=\exp[-(\Delta t^\ast/1.2)^2]",
            r"C_{\mathrm{damage}}=\frac{1}{1+\exp[-(\Delta t^\ast-5.0)/1.3]}",
        ],
    )

with tab_stress:
    st.subheader("Stress-path interpretation")
    st.info(
        "The stress path uses total stresses: static prestress plus dynamic pulse increments. "
        "Prestress sets the starting point; the moving p-q-theta history is driven by the dynamic loading."
    )

    with st.expander("How to read this", expanded=False):
        st.markdown(
            """
            **What the three quantities mean**

            - **p (mean stress)** = how hard the specimen is squeezed all-round
              (the hydrostatic / confining part). Higher p raises strength and
              drives compaction.
            - **q (deviatoric stress)** = how *unequal* the three stresses are
              (the shear part). q drives shear failure and cracking; **q = 0**
              means all three stresses are equal (pure hydrostatic).
            - **θ (Lode angle, 0-60°)** = the *kind* of shear: **~0° = triaxial
              compression**, ~30° = pure shear, **~60° = triaxial extension**.

            **The path is driven by the pulse.** Before the pulse the point sits
            at the static pre-stress state. When the dynamic pulse arrives it
            pushes one or more axes up, so p, q and θ move together - that moving
            trace is the stress path. Because the half-sine pulse loads then
            unloads, the path runs **out and back along the same line**: start
            (○) → peak (△) at mid-pulse → back to start (✕). Colour = time.

            **Reading checklist**

            1. **Where does it start?** -> your confinement. Low-p start =
               little/no confinement; further right = more confined.
            2. **How steep is the p-q climb?** -> *failure mode.* Steep (q rises
               faster than p) = shear-dominated (cracking); shallow (p rises, q
               low) = compaction-dominated (pore collapse).
            3. **How far does the peak (△) reach?** -> compare against the
               failure surface q_f in the Damage tab (F = q / q_f). Peak near or
               above q_f -> damage grows; well below -> specimen survives.
            4. **What is θ at the peak?** -> failure *style*: 0° compression,
               30° shear, 60° extension.
            5. **Does it return to the start?** -> yes = elastic recovery; a
               path that does not close indicates permanent change.

            **Mode signatures**

            - *Single-sided X / gas-gun*: one axis driven -> large q, steep p-q
              line, θ -> 0 (compression). Shear-failure prone.
            - *Symmetric XYZ (hydrostatic)*: all axes driven equally -> q ~ 0,
              path runs almost flat along the p-axis; the specimen can only
              compact, not shear-fail.
            - *Asynchronous XY/XYZ*: axes driven at different times -> the path
              **bends** as each delayed pulse arrives; this is where stress
              path-dependence shows up.
            """
        )

    fig3, ax3 = plt.subplots(figsize=(7, 5.2))
    sc = ax3.scatter(p, q, c=t_us, s=8, zorder=3)
    annotate_path_direction(ax3, p, q)
    ax3.set_xlabel(r"Mean stress, $p$ (MPa)")
    ax3.set_ylabel(r"Deviatoric stress, $q$ (MPa)")
    ax3.set_title(r"$p$-$q$ stress path")
    ax3.grid(True, alpha=0.35)
    cb = fig3.colorbar(sc, ax=ax3)
    cb.set_label("Time (μs)")
    show_step_figure(
        fig3, "tri_hb_step3_pq_path.png",
        caption=("Colour = time since pulse start (purple -> yellow). The pulse "
                 "loads then unloads, so the path runs out and back along the same "
                 "line: start (○) -> peak (△) at mid-pulse -> back to start (✕)."),
        notes=(
            "**What it shows.** The loading trajectory in mean-stress / "
            "deviatoric-stress space.\n\n"
            "- **Start (○)** = your static pre-stress; further right = more "
            "confined.\n"
            "- **Steep climb** (q rises faster than p) = shear-dominated; "
            "**shallow** = compaction-dominated.\n"
            "- **Peak (△)**: compare against the failure surface q_f in the "
            "Damage tab. Reaching q_f is when damage starts."
        ),
        equations=[
            r"p=\tfrac{1}{3}(\sigma_x+\sigma_y+\sigma_z)",
            r"q=\sqrt{\tfrac{1}{2}[(\sigma_x-\sigma_y)^2+(\sigma_y-\sigma_z)^2+(\sigma_z-\sigma_x)^2]}",
        ],
    )

    fig4, ax4 = plt.subplots(figsize=(7, 5.2))
    sc2 = ax4.scatter(q, theta_deg, c=t_us, s=8, zorder=3)
    annotate_path_direction(ax4, q, theta_deg)
    ax4.set_xlabel(r"Deviatoric stress, $q$ (MPa)")
    ax4.set_ylabel(r"Lode angle, $\theta$ (degrees)")
    ax4.set_title(r"$q$-$\theta$ projection")
    ax4.grid(True, alpha=0.35)
    cb2 = fig4.colorbar(sc2, ax=ax4)
    cb2.set_label("Time (μs)")
    show_step_figure(
        fig4, "tri_hb_step3_qtheta.png",
        notes=(
            "**What it shows.** How the *character* of the shear evolves with the "
            "load.\n\n"
            "- **θ ~ 0°** = triaxial compression, **~30°** = pure shear, "
            "**~60°** = triaxial extension.\n"
            "- A path that sweeps θ -> 0 as q grows (single-sided loading) means "
            "the state moves toward triaxial compression."
        ),
        equations=[
            r"J_2=\tfrac{1}{2}(s_1^2+s_2^2+s_3^2),\quad J_3=s_1 s_2 s_3,\quad s_i=\sigma_i-p",
            r"\cos(3\theta)=\frac{3\sqrt{3}}{2}\,\frac{J_3}{J_2^{3/2}}",
        ],
    )

    fig5, ax5 = plt.subplots(figsize=(10, 4.1))
    ax5.plot(t_us, sx, label=r"$\sigma_x$")
    ax5.plot(t_us, sy, label=r"$\sigma_y$")
    ax5.plot(t_us, sz, label=r"$\sigma_z$")
    ax5.plot(t_us, p, linestyle="--", color=PUB_COLORS["black"], label=r"$p$")
    ax5.plot(t_us, q, linestyle=":", color=PUB_COLORS["orange"], label=r"$q$")
    ax5.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax5.set_ylabel("Stress / invariant (MPa)")
    ax5.set_title("Stress and invariant histories")
    ax5.grid(True, alpha=0.35)
    ax5.legend(ncol=3)
    show_step_figure(
        fig5, "tri_hb_step3_stress_histories.png",
        notes=(
            "**What it shows.** The three total principal stresses and the two "
            "invariants p, q through time.\n\n"
            "- Each σ is static pre-stress plus its dynamic increment.\n"
            "- p (dashed) and q (dotted) are what drive the p-q path above."
        ),
        equations=[
            r"\sigma_i^{\mathrm{total}}(t)=\sigma_i^{0}+\sigma_i^{\mathrm{dyn}}(t)",
        ],
    )

    if loading_path == "Symmetric XYZ" and np.nanmax(q) < 0.05 * max(np.nanmax(p), 1.0):
        st.success("This case is close to hydrostatic: q remains small relative to p.")
    elif np.nanmax(q) > np.nanmax(p):
        st.warning("The deviatoric component q is large relative to p; shear/failure-index effects may dominate.")
    else:
        st.info("The path has both pressure and deviatoric components. Use q/qf in the Damage tab to judge damage onset.")

with tab_damage:
    st.subheader("Damage onset, accumulation and stiffness loss")

    fig6, ax6 = plt.subplots(figsize=(7, 4.6))
    ax6.plot(t_us, F_index, color=PUB_COLORS["blue"], label=r"Applied $F=q/q_f$")
    ax6.plot(t_us, F_eff, color=PUB_COLORS["green"], linestyle="-.", label=r"Effective $(1-D)\,F$")
    ax6.axhline(1.0, color=PUB_COLORS["vermillion"], linestyle="--", linewidth=1.2, label=r"$F=1$")
    ax6.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax6.set_ylabel(r"Failure index")
    ax6.set_title("Failure-envelope interaction")
    ax6.grid(True, alpha=0.35)
    ax6.legend()
    show_step_figure(
        fig6, "tri_hb_step4_failure_index.png",
        caption=("Damage is driven by the effective index (1-D)F, not the applied "
                 "F. As D grows the material sheds load, so the effective index "
                 "falls back toward 1 and the damage rate self-limits."),
        notes=(
            "**What it shows.** How close the stress state is to failure.\n\n"
            "- **F = q / q_f**; damage grows only when **F > 1** (above the dashed "
            "line).\n"
            "- The **effective** index (1-D)F is the real driver - it is the "
            "applied index reduced by the load the damaged material can no longer "
            "carry."
        ),
        equations=[
            r"F=q/q_f,\qquad F_{\mathrm{eff}}=(1-D)\,F",
            r"q_f=(A+B\,p^{n})\,[1+a_\theta(1-\cos 3\theta)]",
        ],
    )

    fig7, ax7 = plt.subplots(figsize=(7, 4.6))
    ax7.plot(t_us, D, color=PUB_COLORS["blue"], label=r"$D(t)$")
    if np.max(Ddot) > 0:
        ax7.plot(t_us, Ddot / max(np.max(Ddot), 1e-12), color=PUB_COLORS["orange"], linestyle="--", label=r"normalised $\dot{D}$")
    ax7.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax7.set_ylabel(r"Damage variable, $D$")
    ax7.set_title("Cumulative damage and damage rate")
    ax7.grid(True, alpha=0.35)
    ax7.legend()
    show_step_figure(
        fig7, "tri_hb_step4_damage_evolution.png",
        notes=(
            "**What it shows.** Cumulative damage D (0 = intact, 1 = fully failed) "
            "and its rate.\n\n"
            "- D rises while F_eff > 1 and **self-limits** as the material softens "
            "(it does not snap straight to 1).\n"
            "- The rate (dashed) peaks when the stress most exceeds the failure "
            "surface."
        ),
        equations=[
            r"\dot D=\frac{(1-D)^{\alpha}}{\tau_D}\,\Big\langle\frac{F_{\mathrm{eff}}-1}{F_0}\Big\rangle^{m}\,\Big(\frac{|\dot\varepsilon_{\mathrm{eq}}|}{\dot\varepsilon_0}\Big)^{\beta}",
            r"D_i=\operatorname{clip}[\,D_{i-1}+\dot D_{i-1}\,\Delta t,\,0,\,1\,]",
        ],
    )

    fig8, ax8 = plt.subplots(figsize=(10, 4.1))
    ax8.plot(t_us, E_D, color=PUB_COLORS["blue"], label=r"$E(D)$")
    ax8b = ax8.twinx()
    ax8b.plot(t_us, cp_D, color=PUB_COLORS["orange"], linestyle="--", label=r"$c_p(D)$")
    ax8.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax8.set_ylabel(r"Damaged Young's modulus, $E(D)$ (GPa)")
    ax8b.set_ylabel(r"Damaged P-wave speed, $c_p(D)$ (m/s)")
    ax8.set_title("Stiffness and wave-speed degradation")
    ax8.grid(True, alpha=0.35)
    lines, labels = ax8.get_legend_handles_labels()
    lines2, labels2 = ax8b.get_legend_handles_labels()
    ax8.legend(lines + lines2, labels + labels2)
    show_step_figure(
        fig8, "tri_hb_step4_stiffness_degradation.png",
        notes=(
            "**What it shows.** How the growing damage degrades the material's "
            "elastic properties.\n\n"
            "- Young's modulus falls linearly with damage.\n"
            "- P-wave speed falls with the square root of (1-D) - the quantity a "
            "real test measures via wave-speed/time-of-flight."
        ),
        equations=[
            r"E(D)=E_0\,(1-D),\qquad c_p(D)=c_{p0}\sqrt{\max(1-D,\,0)}",
        ],
    )

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Peak F", f"{np.nanmax(F_index):.2f}")
    d2.metric("Final D", f"{D[-1]:.3f}")
    d3.metric("Peak damage rate", f"{np.nanmax(Ddot):.2e} /s")
    d4.metric("cp loss", f"{100.0 * (1.0 - cp_D[-1] / cp0):.1f} %")

with tab_validation:
    st.subheader("Energy balance and validation descriptors")

    fig9, ax9 = plt.subplots(figsize=(7, 4.6))
    ax9.plot(t_us, W_input, color=PUB_COLORS["blue"], label=r"$W_{\mathrm{input}}$")
    ax9.plot(t_us, W_el, color=PUB_COLORS["green"], label=r"$W_{\mathrm{el}}$ (recoverable)")
    ax9.plot(t_us, W_diss_estimate, color=PUB_COLORS["vermillion"], label=r"$W_{\mathrm{diss}}$ (damage)")
    ax9.set_xlabel(r"Time, $t$ ($\mu$s)")
    ax9.set_ylabel(r"Energy density (MJ m$^{-3}$)")
    ax9.set_title("Energy indicators")
    ax9.grid(True, alpha=0.35)
    ax9.legend()
    show_step_figure(
        fig9, "tri_hb_step4_energy_balance.png",
        caption=("W_diss is the continuum-damage dissipation, zero until damage "
                 "grows then increasing monotonically."),
        notes=(
            "**What it shows.** Where the input work goes.\n\n"
            "- **W_input** total work done on the specimen.\n"
            "- **W_el** recoverable elastic energy (returned on unloading).\n"
            "- **W_diss** energy dissipated by damage = the continuum-damage "
            "integral ∫Y dD with Y the elastic energy release rate."
        ),
        equations=[
            r"W_{\mathrm{input}}(t)=\int_0^t(\sigma_x\dot\varepsilon_x+\sigma_y\dot\varepsilon_y+\sigma_z\dot\varepsilon_z)\,d\tau",
            r"W_{\mathrm{el}}=\frac{1}{2E_0}\big[\sigma_x^2+\sigma_y^2+\sigma_z^2-2\nu(\sigma_x\sigma_y+\sigma_y\sigma_z+\sigma_z\sigma_x)\big]",
            r"W_{\mathrm{diss}}(t)=\int_0^t Y\,\dot D\,d\tau,\qquad Y=W_{\mathrm{el}}",
        ],
    )

    st.markdown("**Validation descriptors**")
    descriptors = pd.DataFrame({
        "Descriptor": [
            "Final damage D",
            "Central damage fraction Dc",
            "Symmetry index Sx",
            "Neutral-zone estimate χn",
            "Superposition factor ηsup",
            "Normalised delay Δt*",
            "Peak failure index Fmax",
        ],
        "Value": [
            D[-1],
            D_c,
            S_x,
            neutral_width_estimate,
            eta_sup,
            dt_star,
            np.nanmax(F_index),
        ],
        "Use for": [
            "Overall degradation",
            "Central damage concentration",
            "Left-right damage symmetry",
            "Low-velocity zone estimate",
            "Constructive overlap check",
            "Regime classifier",
            "Damage-onset calibration",
        ],
    })
    st.dataframe(descriptors, width="stretch")

    fig10, ax10 = plt.subplots(figsize=(7, 4.5))
    ax10.plot(x, damage_profile, color=PUB_COLORS["blue"], label="Estimated damage profile")
    ax10.axvspan(0.5 - central_width / 2.0, 0.5 + central_width / 2.0, alpha=0.15, label="Central region")
    ax10.axhline(damage_threshold, linestyle="--", label="Threshold")
    ax10.set_xlabel(r"Normalised specimen position, $x/L$")
    ax10.set_ylabel("Damage intensity")
    ax10.set_title("Damage-zone migration / central concentration")
    ax10.grid(True, alpha=0.35)
    ax10.legend()
    show_step_figure(
        fig10, "tri_hb_step4_damage_profile.png",
        notes=(
            "**What it shows.** Where damage concentrates along the specimen (a "
            "planning indicator, to be replaced by DEM/CT/DIC when available).\n\n"
            "- A peak at the **centre** (x/L = 0.5) indicates central concentration "
            "from wave superposition.\n"
            "- Off-centre or asymmetric peaks indicate sequential / one-sided "
            "loading. The shaded band is the central region used for D_c."
        ),
        equations=[
            r"D_c=\frac{\int_{\mathrm{centre}}D(x)\,dx}{\int_0^1 D(x)\,dx},\qquad S_x=1-\frac{|D_{\mathrm{left}}-D_{\mathrm{right}}|}{D_{\mathrm{left}}+D_{\mathrm{right}}}",
        ],
    )

    if linked_reduced is not None and len(linked_reduced) > 0:
        fig11, ax11 = plt.subplots(figsize=(7, 4.5))
        ax11.plot(t_us, sx, color=PUB_COLORS["blue"], label=r"Model $\sigma_x$")
        red = linked_reduced.dropna(subset=["time_us", "stress_MPa"]) if isinstance(linked_reduced, pd.DataFrame) else pd.DataFrame()
        if not red.empty:
            ax11.plot(red["time_us"], red["stress_MPa"], color=PUB_COLORS["vermillion"], linestyle="--", label="Reduced stress")
        ax11.set_xlabel(r"Time, $t$ ($\mu$s)")
        ax11.set_ylabel(r"Stress, $\sigma$ (MPa)")
        ax11.set_title("Reduced-data / model stress comparison")
        ax11.grid(True, alpha=0.35)
        ax11.legend()
        show_step_figure(
            fig11, "tri_hb_step4_model_vs_reduced.png",
            notes=(
                "**What it shows.** The model axial stress overlaid on the Step-1 "
                "reduced data (simulator or uploaded experiment), so you can judge "
                "how well the model reproduces the measured response."
            ),
        )
    else:
        st.info(
            "Generate reduced data in Step 1 to overlay stress and energy histories here. "
            "Use DEM/CT/DIC outputs to replace the synthetic damage-profile descriptor when available."
        )

    st.markdown(
        """
        **Validation checklist:** bar strain-gauge amplitude and delay, reduced stress curve, DIC localisation,
        CT damage volume/crack orientation, and DEM bond breakage plus dissipated energy.
        """
    )

with tab_export:
    st.subheader("Delay sensitivity and export")
    dt_star_grid = np.linspace(0, 20, 120)
    eta_grid = []
    Dfinal_grid = []
    Dc_grid = []
    for dts in dt_star_grid:
        d = dts * t_travel
        gl = get_window(t - (L_m / (2.0 * cp0)), td, pulse_type)
        gr = get_window(t - d - (L_m / (2.0 * cp0)), td_right, pulse_type)
        sig_c = sx0 + Ax * gl + Ax * amplitude_ratio * gr
        eta = (np.max(sig_c) - sx0) / max(Ax, 1e-9)
        eta_grid.append(eta)
        wc = np.exp(-(dts / 1.2) ** 2)
        dm = 1.0 / (1.0 + np.exp(-(dts - 5.0) / 1.3))
        Dfinal_grid.append(np.clip(0.15 + 0.35 * wc + 0.25 * dm, 0, 1))
        Dc_grid.append(np.clip(0.75 * wc + 0.25 * (1 - dm), 0, 1))

    fig12, ax12 = plt.subplots(figsize=(10, 4.5))
    ax12.plot(dt_star_grid, eta_grid, color=PUB_COLORS["blue"], label=r"$\eta_{\mathrm{sup}}$")
    ax12.plot(dt_star_grid, Dfinal_grid, color=PUB_COLORS["vermillion"], label="Final damage trend")
    ax12.plot(dt_star_grid, Dc_grid, color=PUB_COLORS["green"], label=r"Central fraction, $D_c$")
    ax12.axvline(dt_star, color=PUB_COLORS["black"], linestyle="--", label=r"Current $\Delta t^*$")
    ax12.set_xlabel(r"Normalised delay, $\Delta t^*$")
    ax12.set_ylabel("Delay-sensitivity indicator")
    ax12.set_title("Delay-controlled transition from central superposition to sequential damage")
    ax12.grid(True, alpha=0.35)
    ax12.legend()
    show_step_figure(
        fig12, "tri_hb_step4_delay_sensitivity.png",
        caption=("ηsup is the centre-stress superposition ratio. The final and "
                 "central damage curves are 0-1 heuristic sensitivity indicators."),
        notes=(
            "**What it shows.** How the damage outcome is expected to vary as you "
            "sweep the inter-pulse delay Δt*, with your current case marked "
            "(dashed line).\n\n"
            "- At **small Δt*** waves superpose -> high central damage.\n"
            "- At **large Δt*** pulses act sequentially -> damage spreads / shifts.\n\n"
            "The trend curves are planning indicators; use calibrated DEM/CT/DIC "
            "data for final damage quantification."
        ),
        equations=[
            r"\eta_{\mathrm{sup}}(\Delta t^*)=\frac{\max_t\sigma_{\mathrm{centre}}(t;\Delta t^*)-\sigma_{x0}}{A_x}",
            r"C_{\mathrm{wave}}=\exp[-(\Delta t^*/1.2)^2],\qquad C_{\mathrm{damage}}=\frac{1}{1+\exp[-(\Delta t^*-5.0)/1.3]}",
            r"I_{\mathrm{final}}=\operatorname{clip}(0.15+0.35C_{\mathrm{wave}}+0.25C_{\mathrm{damage}},0,1)",
            r"I_{\mathrm{central}}=\operatorname{clip}(0.75C_{\mathrm{wave}}+0.25(1-C_{\mathrm{damage}}),0,1)",
        ],
    )

    st.markdown("**Calculated data preview**")
    st.dataframe(df_results.head(40), width="stretch")
    col_csv, col_xlsx = st.columns(2)
    with col_csv:
        st.download_button(
            "Download model results CSV",
            data=df_results.to_csv(index=False).encode("utf-8"),
            file_name="tri_hb_wave_stress_damage_results.csv",
            mime="text/csv",
        )
    with col_xlsx:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_results.to_excel(writer, index=False, sheet_name="Model results")
            pd.DataFrame({
                "parameter": [
                    "control_level", "analysis_goal", "loading_path", "pulse_type", "E_GPa", "nu", "rho_kg_m3", "L_mm",
                    "sx0_MPa", "sy0_MPa", "sz0_MPa", "Ax_MPa", "Ay_MPa", "Az_MPa", "td_us",
                    "dt_star", "eta_sup", "final_D", "Dc", "Sx",
                ],
                "value": [
                    control_level, analysis_goal, loading_path, pulse_type, E_GPa, nu, rho, L_mm,
                    sx0, sy0, sz0, Ax, Ay, Az, td_us,
                    dt_star, eta_sup, D[-1], D_c, S_x,
                ],
            }).to_excel(writer, index=False, sheet_name="Summary")
        st.download_button(
            "Download model workbook",
            data=buf.getvalue(),
            file_name="tri_hb_wave_stress_damage_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

st.divider()
st.caption(
    "The workflow uses a simplified diagonal-stress interpretation model for rapid validation. "
    "Use calibrated DEM, CT, and bar-gauge data to replace the synthetic descriptor trends for final reporting."
)
