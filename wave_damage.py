
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Wave–damage transition under multidirectional and sequential impact",
    layout="wide"
)

st.title("Instability evolution under multidirectional and sequential impact loading")
st.caption(
    "Updated for Paper 2: DEM–experimental interpretation of wave–damage transition, "
    "normalised delay, loading-path dimensionality, wave-superposition factor, "
    "damage evolution, stiffness degradation, energy balance and experimental observables."
)

# =============================================================================
# Utility functions
# =============================================================================
def hann_window(t, td):
    tau = np.asarray(t)
    g = np.zeros_like(tau, dtype=float)
    mask = (tau >= 0.0) & (tau <= td)
    g[mask] = 0.5 * (1.0 - np.cos(2.0 * np.pi * tau[mask] / td))
    return g

def half_sine_window(t, td):
    tau = np.asarray(t)
    g = np.zeros_like(tau, dtype=float)
    mask = (tau >= 0.0) & (tau <= td)
    g[mask] = np.sin(np.pi * tau[mask] / td)
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
    """A finite-duration compressive pulse. sign allows opposite-direction plotting if needed."""
    tau = t - delay
    return sign * A * get_window(tau, td, mode)

def central_difference(y, t):
    return np.gradient(y, t)

def invariants_from_diagonal(sx, sy, sz):
    p = (sx + sy + sz) / 3.0
    q = np.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2))
    s1, s2, s3 = sx - p, sy - p, sz - p
    J2 = (s1**2 + s2**2 + s3**2) / 2.0
    J3 = s1 * s2 * s3
    theta = np.zeros_like(p)
    mask = J2 > 1e-12
    arg = np.zeros_like(p)
    arg[mask] = (3.0 * np.sqrt(3.0) / 2.0) * J3[mask] / (J2[mask] ** 1.5)
    arg = np.clip(arg, -1.0, 1.0)
    theta[mask] = (1.0 / 3.0) * np.arccos(arg[mask])
    return p, q, np.rad2deg(theta), J2, J3

def cumulative_trapezoid(y, t):
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))
    return out

def fig_to_bytes(fig, fmt="png", dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf

# =============================================================================
# Sidebar inputs
# =============================================================================
st.sidebar.header("Material and specimen")

E_GPa = st.sidebar.number_input("Young's modulus, E (GPa)", value=50.0, min_value=1.0, step=1.0)
nu = st.sidebar.number_input("Poisson's ratio, ν", value=0.25, min_value=0.0, max_value=0.49, step=0.01)
rho = st.sidebar.number_input("Density, ρ (kg/m³)", value=2650.0, min_value=1000.0, step=50.0)
L_mm = st.sidebar.number_input("Specimen side length, L (mm)", value=50.0, min_value=1.0, step=1.0)

E = E_GPa * 1e9
G = E / (2.0 * (1.0 + nu))
M = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
cp0 = np.sqrt(M / rho)
cs = np.sqrt(G / rho)
L_m = L_mm / 1000.0
t_travel = L_m / cp0
t_eq_low = 3.0 * t_travel
t_eq_high = 5.0 * t_travel

st.sidebar.header("Initial stresses")
sx0 = st.sidebar.number_input("σx0 (MPa)", value=30.0, min_value=0.0, step=1.0)
sy0 = st.sidebar.number_input("σy0 (MPa)", value=20.0, min_value=0.0, step=1.0)
sz0 = st.sidebar.number_input("σz0 (MPa)", value=15.0, min_value=0.0, step=1.0)

st.sidebar.header("Loading configuration")
loading_path = st.sidebar.selectbox(
    "Impact path dimensionality",
    ["Single-sided X", "Symmetric X", "Symmetric XY", "Symmetric XYZ"],
    index=1
)

pulse_type = st.sidebar.selectbox("Pulse envelope", ["Hann", "Half-sine", "Rectangular"], index=0)
td_us = st.sidebar.number_input("Pulse duration, td (μs)", value=60.0, min_value=1.0, step=5.0)
tmax_us = st.sidebar.number_input("Simulation time (μs)", value=250.0, min_value=20.0, step=10.0)
npts = st.sidebar.slider("Number of time points", 1000, 30000, 5000, step=1000)

st.sidebar.subheader("Pulse amplitudes")
Ax = st.sidebar.number_input("Ax (MPa)", value=60.0, min_value=0.0, step=5.0)
Ay = st.sidebar.number_input("Ay (MPa)", value=60.0, min_value=0.0, step=5.0)
Az = st.sidebar.number_input("Az (MPa)", value=60.0, min_value=0.0, step=5.0)

st.sidebar.subheader("Waveform matching")
amplitude_ratio = st.sidebar.number_input("Right/left amplitude ratio in X", value=1.0, min_value=0.0, step=0.1)
duration_ratio = st.sidebar.number_input("Right/left pulse-duration ratio in X", value=1.0, min_value=0.1, step=0.1)
delay_us = st.sidebar.number_input("Time delay Δt for right/secondary pulse (μs)", value=100.0, min_value=0.0, step=5.0)

st.sidebar.header("Failure and damage")
A_fail = st.sidebar.number_input("A in qf = (A+Bpⁿ)h(θ) (MPa)", value=15.0, min_value=0.0, step=1.0)
B_fail = st.sidebar.number_input("B in qf = (A+Bpⁿ)h(θ)", value=1.3, min_value=0.0, step=0.1)
n_fail = st.sidebar.number_input("n in qf = (A+Bpⁿ)h(θ)", value=0.75, min_value=0.1, step=0.05)
lode_amp = st.sidebar.number_input("Lode-angle factor amplitude, aθ", value=0.10, min_value=0.0, step=0.02)

tau_D_us = st.sidebar.number_input("Damage time scale, τD (μs)", value=30.0, min_value=0.1, step=5.0)
alpha_sat = st.sidebar.number_input("Damage saturation exponent, α", value=1.0, min_value=0.0, step=0.2)
m_over = st.sidebar.number_input("Overstress exponent, m", value=2.0, min_value=0.1, step=0.2)
beta_rate = st.sidebar.number_input("Rate exponent, β", value=0.15, min_value=0.0, step=0.05)
epsdot0 = st.sidebar.number_input("Reference strain rate, ε̇0 (s⁻¹)", value=1.0, min_value=1e-6, step=1.0)
F0 = st.sidebar.number_input("Failure-index normalisation, F0", value=1.0, min_value=0.1, step=0.1)

st.sidebar.header("Descriptors")
damage_threshold = st.sidebar.slider("Damage threshold for central damage fraction", 0.01, 0.9, 0.15, step=0.01)
central_width = st.sidebar.slider("Central region fraction of specimen length", 0.1, 0.8, 0.30, step=0.05)

# =============================================================================
# Time and loading histories
# =============================================================================
t_us = np.linspace(0.0, tmax_us, npts)
t = t_us * 1e-6
td = td_us * 1e-6
delay = delay_us * 1e-6
td_right = td * duration_ratio

# Impact pulses. Positive values are compressive dynamic stress increments.
# x has left and right pulses; right may be delayed/mismatched.
x_left = pulse(t, Ax, td, 0.0, pulse_type)
x_right = pulse(t, Ax * amplitude_ratio, td_right, delay, pulse_type)

# yz activation depends on loading dimensionality
if loading_path in ["Symmetric XY", "Symmetric XYZ"]:
    y_pair = pulse(t, Ay, td, 0.0, pulse_type) + pulse(t, Ay, td, 0.0, pulse_type)
else:
    y_pair = np.zeros_like(t)

if loading_path == "Symmetric XYZ":
    z_pair = pulse(t, Az, td, 0.0, pulse_type) + pulse(t, Az, td, 0.0, pulse_type)
else:
    z_pair = np.zeros_like(t)

if loading_path == "Single-sided X":
    sx_dyn = x_left
elif loading_path == "Symmetric X":
    sx_dyn = x_left + x_right
else:
    sx_dyn = x_left + x_right

# Simplified invariant stress representation:
# symmetric pair loading increases normal compressive stress in active directions.
sx = sx0 + sx_dyn
sy = sy0 + y_pair
sz = sz0 + z_pair

p, q, theta_deg, J2, J3 = invariants_from_diagonal(sx, sy, sz)

# Wave-superposition factor for X pair at specimen centre
# centre stress proxy uses delayed envelopes only, as in the paper.
g_left_centre = get_window(t - (L_m / (2.0 * cp0)), td, pulse_type)
g_right_centre = get_window(t - delay - (L_m / (2.0 * cp0)), td_right, pulse_type)
sigma_centre = sx0 + Ax * g_left_centre + Ax * amplitude_ratio * g_right_centre
eta_sup = (np.max(sigma_centre) - sx0) / max(Ax, 1e-9)

# Normalised delay and regime
dt_star = delay / t_travel if t_travel > 0 else np.nan
if dt_star < 1:
    regime = "Synchronous wave superposition"
elif dt_star <= 3:
    regime = "Reverberation-coupled interaction"
elif dt_star <= 10:
    regime = "Transitional / decaying reverberations"
else:
    regime = "Sequential, damage-memory controlled"

# Failure envelope and damage law
theta_rad = np.deg2rad(theta_deg)
h_theta = 1.0 + lode_amp * (1.0 - np.cos(3.0 * theta_rad))
qf = (A_fail + B_fail * np.maximum(p, 0.0) ** n_fail) * h_theta
F_index = q / np.maximum(qf, 1e-9)

# equivalent strain-rate proxy from elastic normal strains
E_MPa = E_GPa * 1000.0
eps_x = sx / E_MPa
eps_y = sy / E_MPa
eps_z = sz / E_MPa
epsdot_x = central_difference(eps_x, t)
epsdot_y = central_difference(eps_y, t)
epsdot_z = central_difference(eps_z, t)
epsdot_eq = np.sqrt(2.0 / 3.0 * ((epsdot_x - epsdot_y)**2 + (epsdot_y - epsdot_z)**2 + (epsdot_z - epsdot_x)**2) / 2.0)
rate_factor = (np.maximum(np.abs(epsdot_eq), 1e-12) / epsdot0) ** beta_rate

tau_D = tau_D_us * 1e-6
D = np.zeros_like(t)
Ddot = np.zeros_like(t)
for i in range(1, len(t)):
    overstress = max((F_index[i-1] - 1.0) / F0, 0.0)
    Ddot[i-1] = ((1.0 - D[i-1]) ** alpha_sat) / tau_D * (overstress ** m_over) * rate_factor[i-1]
    D[i] = np.clip(D[i-1] + Ddot[i-1] * (t[i] - t[i-1]), 0.0, 1.0)
Ddot[-1] = Ddot[-2] if len(t) > 1 else 0.0

E_D = E_GPa * (1.0 - D)
cp_D = cp0 * np.sqrt(np.maximum(1.0 - D, 0.0))

# Energy indicators
W_el = (1.0 / (2.0 * E_MPa)) * (
    sx**2 + sy**2 + sz**2 - 2.0 * nu * (sx * sy + sy * sz + sz * sx)
)
power = sx * epsdot_x + sy * epsdot_y + sz * epsdot_z
W_input = cumulative_trapezoid(power, t)
W_diss_proxy = W_input - W_el + W_el[0]

# Synthetic descriptors reflecting expected DEM/experimental observables
# These are model indicators for planning and interpretation, not actual DEM outputs.
x = np.linspace(0, 1, 400)
centre = 0.5
left_damage = np.exp(-((x - 0.20) / 0.12)**2) * np.max(D)
right_damage = np.exp(-((x - 0.80) / 0.12)**2) * np.max(D) * amplitude_ratio
central_damage = np.exp(-((x - 0.50) / 0.16)**2) * np.max(D) * min(eta_sup, 2.0) / 2.0
delay_scatter = (1.0 / (1.0 + np.exp(-(dt_star - 5.0)))) * np.max(D) * 0.35
damage_profile = np.clip(left_damage + right_damage + central_damage + delay_scatter, 0, 1)

central_mask = np.abs(x - centre) < central_width / 2.0
total_damage_area = np.trapz(damage_profile, x)
central_damage_area = np.trapz(damage_profile[central_mask], x[central_mask]) if np.any(central_mask) else 0
D_c = central_damage_area / max(total_damage_area, 1e-12)

D_left = np.trapz(damage_profile[x < 0.5], x[x < 0.5])
D_right = np.trapz(damage_profile[x >= 0.5], x[x >= 0.5])
S_x = 1.0 - abs(D_left - D_right) / max(D_left + D_right + 1e-12, 1e-12)

neutral_width_proxy = np.clip(1.0 / (1.0 + 0.5 * dt_star) * (1.0 / max(1.0, abs(amplitude_ratio - 1.0) + 1.0)), 0, 1)

# =============================================================================
# Summary metrics
# =============================================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("P-wave speed, cp", f"{cp0:.0f} m/s")
col2.metric("Travel time", f"{t_travel*1e6:.2f} μs")
col3.metric("Equilibrium time", f"{t_eq_low*1e6:.1f}–{t_eq_high*1e6:.1f} μs")
col4.metric("Normalised delay, Δt*", f"{dt_star:.2f}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Regime", regime)
col6.metric("Superposition factor, ηsup", f"{eta_sup:.2f}")
col7.metric("Final damage, D", f"{D[-1]:.3f}")
col8.metric("Central damage fraction, Dc", f"{D_c:.3f}")

st.info(
    "This app is intended as a theoretical/interpretive demo for Paper 2. "
    "The stress paths and damage variables are analytical proxies; DEM outputs and experimental data should be imported later for validation."
)

# =============================================================================
# Tabs
# =============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Loading and regimes",
    "Stress path",
    "Damage evolution",
    "Energy balance",
    "DEM/experimental descriptors",
    "Parametric study",
    "Export"
])

with tab1:
    st.subheader("Input pulses and normalised-delay regime")

    fig1, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.plot(t_us, x_left, label="X left pulse")
    ax1.plot(t_us, x_right, label="X right/secondary pulse")
    if np.any(y_pair):
        ax1.plot(t_us, y_pair, label="Y pair")
    if np.any(z_pair):
        ax1.plot(t_us, z_pair, label="Z pair")
    ax1.axvline(t_travel * 1e6, linestyle="--", linewidth=1.0, label=r"$t_{\rm travel}$")
    ax1.axvspan(t_eq_low * 1e6, t_eq_high * 1e6, alpha=0.15, label=r"$t_{\rm eq}\approx3$--$5t_{\rm travel}$")
    ax1.axvline(delay_us, linestyle=":", linewidth=1.4, label=r"$\Delta t$")
    ax1.set_xlabel("Time (μs)")
    ax1.set_ylabel("Dynamic stress increment (MPa)")
    ax1.set_title("Finite-duration input pulses")
    ax1.grid(True, alpha=0.35)
    ax1.legend(ncol=2)
    st.pyplot(fig1)
    st.download_button("Download pulse figure", fig_to_bytes(fig1), "paper2_input_pulses.png", "image/png")

    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    dt_grid = np.linspace(0, 20, 400)
    wave_control = np.exp(-(dt_grid / 1.2)**2)
    damage_control = 1.0 / (1.0 + np.exp(-(dt_grid - 5.0) / 1.3))
    ax2.plot(dt_grid, wave_control, label="Wave-interaction control")
    ax2.plot(dt_grid, damage_control, label="Damage-memory control")
    ax2.axvline(dt_star, linestyle="--", linewidth=1.5, label="Current case")
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
    ax2.set_xlabel(r"Normalised delay, $\Delta t^*=\Delta t/t_{\rm travel}$")
    ax2.set_ylabel("Relative controlling mechanism")
    ax2.set_title("Wave–damage transition map")
    ax2.grid(True, alpha=0.35)
    ax2.legend()
    st.pyplot(fig2)
    st.download_button("Download regime map", fig_to_bytes(fig2), "paper2_wave_damage_regime_map.png", "image/png")

    fig3, ax3 = plt.subplots(figsize=(10, 3.8))
    ax3.plot(t_us, sigma_centre, label="Central stress proxy")
    ax3.set_xlabel("Time (μs)")
    ax3.set_ylabel("Central stress proxy (MPa)")
    ax3.set_title(r"Wave-superposition factor $\eta_{\rm sup}$")
    ax3.grid(True, alpha=0.35)
    ax3.legend()
    st.pyplot(fig3)

with tab2:
    st.subheader("Stress-path interpretation")

    c1, c2 = st.columns(2)
    with c1:
        fig4, ax4 = plt.subplots(figsize=(7, 5.4))
        sc = ax4.scatter(p, q, c=t_us, s=8)
        ax4.set_xlabel("Mean stress, p (MPa)")
        ax4.set_ylabel("Deviatoric stress, q (MPa)")
        ax4.set_title("p–q stress path")
        ax4.grid(True, alpha=0.35)
        cb = fig4.colorbar(sc, ax=ax4)
        cb.set_label("Time (μs)")
        st.pyplot(fig4)
        st.download_button("Download p–q path", fig_to_bytes(fig4), "paper2_pq_path.png", "image/png")

    with c2:
        fig5, ax5 = plt.subplots(figsize=(7, 5.4))
        sc2 = ax5.scatter(q, theta_deg, c=t_us, s=8)
        ax5.set_xlabel("q (MPa)")
        ax5.set_ylabel("Lode angle, θ (degrees)")
        ax5.set_title("q–θ projection")
        ax5.grid(True, alpha=0.35)
        cb2 = fig5.colorbar(sc2, ax=ax5)
        cb2.set_label("Time (μs)")
        st.pyplot(fig5)

    fig6, ax6 = plt.subplots(figsize=(10, 4.2))
    ax6.plot(t_us, p, label="p")
    ax6.plot(t_us, q, label="q")
    ax6.plot(t_us, theta_deg, label="θ (degrees)")
    ax6.set_xlabel("Time (μs)")
    ax6.set_ylabel("Value")
    ax6.set_title("Invariant histories")
    ax6.grid(True, alpha=0.35)
    ax6.legend()
    st.pyplot(fig6)

with tab3:
    st.subheader("Rate-sensitive, saturating damage evolution")

    c1, c2 = st.columns(2)
    with c1:
        fig7, ax7 = plt.subplots(figsize=(7, 4.8))
        ax7.plot(t_us, F_index, label="Failure index F(t)")
        ax7.axhline(1.0, linestyle="--", linewidth=1.2, label="Damage threshold")
        ax7.set_xlabel("Time (μs)")
        ax7.set_ylabel("F(t)")
        ax7.set_title("Failure-envelope interaction")
        ax7.grid(True, alpha=0.35)
        ax7.legend()
        st.pyplot(fig7)

    with c2:
        fig8, ax8 = plt.subplots(figsize=(7, 4.8))
        ax8.plot(t_us, D, label="D(t)")
        ax8.plot(t_us, Ddot / max(np.max(Ddot), 1e-12), label="Normalised Ddot")
        ax8.set_xlabel("Time (μs)")
        ax8.set_ylabel("Damage variable")
        ax8.set_title("Cumulative damage and damage rate")
        ax8.grid(True, alpha=0.35)
        ax8.legend()
        st.pyplot(fig8)
        st.download_button("Download damage figure", fig_to_bytes(fig8), "paper2_damage_evolution.png", "image/png")

    fig9, ax9 = plt.subplots(figsize=(10, 4.2))
    ax9.plot(t_us, E_D, label="E(D)")
    ax9b = ax9.twinx()
    ax9b.plot(t_us, cp_D, linestyle="--", label="cp(D)")
    ax9.set_xlabel("Time (μs)")
    ax9.set_ylabel("Damaged Young's modulus (GPa)")
    ax9b.set_ylabel("Damaged P-wave speed (m/s)")
    ax9.set_title("Stiffness and wave-speed degradation")
    ax9.grid(True, alpha=0.35)
    lines, labels = ax9.get_legend_handles_labels()
    lines2, labels2 = ax9b.get_legend_handles_labels()
    ax9.legend(lines + lines2, labels + labels2)
    st.pyplot(fig9)

with tab4:
    st.subheader("Energy-balance indicators")

    fig10, ax10 = plt.subplots(figsize=(10, 4.8))
    ax10.plot(t_us, W_input, label="Input energy proxy")
    ax10.plot(t_us, W_el, label="Recoverable elastic energy")
    ax10.plot(t_us, W_diss_proxy, label="Dissipated-energy proxy")
    ax10.set_xlabel("Time (μs)")
    ax10.set_ylabel("Energy density (MJ/m³)")
    ax10.set_title("Energy indicators")
    ax10.grid(True, alpha=0.35)
    ax10.legend()
    st.pyplot(fig10)
    st.download_button("Download energy figure", fig_to_bytes(fig10), "paper2_energy_balance.png", "image/png")

    st.markdown(
        """
        In the actual DEM/experimental workflow, replace these analytical proxies with:
        - bar-wave energy: \(E_{I,R,T}=A_bE_bc_b\\int\\varepsilon_{I,R,T}^2dt\);
        - DEM absorbed energy from stress power and contact dissipation;
        - kinetic-energy residual \(E_K\) to check energy closure.
        """
    )

with tab5:
    st.subheader("DEM and experimental descriptors")

    c1, c2 = st.columns(2)
    with c1:
        fig11, ax11 = plt.subplots(figsize=(7, 4.6))
        ax11.plot(x, damage_profile, label="Synthetic damage profile")
        ax11.axvspan(0.5 - central_width / 2.0, 0.5 + central_width / 2.0, alpha=0.15, label="Central region")
        ax11.axhline(damage_threshold, linestyle="--", label="Damage threshold")
        ax11.set_xlabel("Normalised specimen position, x/L")
        ax11.set_ylabel("Damage intensity")
        ax11.set_title("Damage-zone migration / central concentration")
        ax11.grid(True, alpha=0.35)
        ax11.legend()
        st.pyplot(fig11)

    with c2:
        descriptors = pd.DataFrame({
            "Descriptor": [
                "Final damage D",
                "Central damage fraction Dc",
                "Symmetry index Sx",
                "Neutral-zone proxy χn",
                "Superposition factor ηsup",
                "Normalised delay Δt*",
            ],
            "Value": [
                D[-1],
                D_c,
                S_x,
                neutral_width_proxy,
                eta_sup,
                dt_star,
            ],
            "Interpretation": [
                "Cumulative material degradation",
                "Degree of central damage concentration",
                "Left-right damage symmetry",
                "Low-velocity / low-strain-rate zone proxy",
                "Constructive wave-overlap indicator",
                "Regime classifier",
            ],
        })
        st.dataframe(descriptors, use_container_width=True)

    st.markdown(
        """
        **Suggested validation hierarchy:**  
        1. Bar strain gauges: amplitude, duration, delay and energy.  
        2. High-speed imaging / DIC: surface strain localisation and damage-zone migration.  
        3. CT scanning: internal damage volume, damage band width and crack orientation.  
        4. DEM: stress path, bond-breakage ratio, central damage fraction and energy dissipation.
        """
    )

with tab6:
    st.subheader("Quick parametric study of delay effect")

    dt_star_grid = np.linspace(0, 20, 120)
    eta_grid = []
    Dfinal_grid = []
    Dc_grid = []
    for dts in dt_star_grid:
        d = dts * t_travel
        gr = get_window(t - d - (L_m / (2.0 * cp0)), td_right, pulse_type)
        gl = get_window(t - (L_m / (2.0 * cp0)), td, pulse_type)
        sig_c = sx0 + Ax * gl + Ax * amplitude_ratio * gr
        eta = (np.max(sig_c) - sx0) / max(Ax, 1e-9)
        eta_grid.append(eta)

        # conceptual trends for final damage and central fraction
        wave_control = np.exp(-(dts / 1.2)**2)
        damage_memory = 1.0 / (1.0 + np.exp(-(dts - 5.0) / 1.3))
        Dfinal_grid.append(np.clip(0.15 + 0.35 * wave_control + 0.25 * damage_memory, 0, 1))
        Dc_grid.append(np.clip(0.75 * wave_control + 0.25 * (1 - damage_memory), 0, 1))

    fig12, ax12 = plt.subplots(figsize=(10, 4.8))
    ax12.plot(dt_star_grid, eta_grid, label=r"$\eta_{\rm sup}$")
    ax12.plot(dt_star_grid, Dfinal_grid, label="Final damage trend")
    ax12.plot(dt_star_grid, Dc_grid, label="Central damage fraction trend")
    ax12.axvline(dt_star, linestyle="--", label="Current case")
    ax12.set_xlabel(r"Normalised delay, $\Delta t^*$")
    ax12.set_ylabel("Normalised indicator")
    ax12.set_title("Delay-controlled transition from central superposition to sequential damage")
    ax12.grid(True, alpha=0.35)
    ax12.legend()
    st.pyplot(fig12)
    st.download_button("Download parametric delay figure", fig_to_bytes(fig12), "paper2_delay_parametric_study.png", "image/png")

with tab7:
    st.subheader("Export data")

    df = pd.DataFrame({
        "time_us": t_us,
        "x_left_MPa": x_left,
        "x_right_MPa": x_right,
        "y_pair_MPa": y_pair,
        "z_pair_MPa": z_pair,
        "sigma_x_MPa": sx,
        "sigma_y_MPa": sy,
        "sigma_z_MPa": sz,
        "p_MPa": p,
        "q_MPa": q,
        "theta_deg": theta_deg,
        "qf_MPa": qf,
        "failure_index": F_index,
        "epsdot_eq_s-1": epsdot_eq,
        "D": D,
        "Ddot_s-1": Ddot,
        "E_D_GPa": E_D,
        "cp_D_m_s": cp_D,
        "W_input_MJ_m3": W_input,
        "W_elastic_MJ_m3": W_el,
        "W_diss_proxy_MJ_m3": W_diss_proxy,
    })
    st.dataframe(df.head(30), use_container_width=True)
    st.download_button(
        "Download full results as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="paper2_wave_damage_transition_results.csv",
        mime="text/csv",
    )

    st.markdown("### Suggested manuscript wording")
    st.code(
        """The instability response was interpreted using the normalised delay Δt*=Δt/ttravel. 
For Δt*<1, the response is governed by direct stress-wave superposition, quantified by ηsup. 
For 1≤Δt*≤3, the response remains coupled through wave reverberations. 
For 3<Δt*≤10, the system enters a transitional regime in which reverberations decay and damage begins to influence the subsequent response. 
For Δt*>10, the second pulse acts primarily on a stiffness-degraded material state, and the instability mechanism becomes damage-memory controlled.""",
        language="text",
    )
