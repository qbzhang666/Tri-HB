
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Tri-HB Multidirectional Stress-Wave Superposition",
    layout="wide"
)

st.title("Multidirectional stress-wave superposition and stress-path evolution")
st.caption(
    "Updated according to the final Paper 1 formulation: finite-duration windowed pulses, "
    "P-wave travel time, p–q–θ invariants, failure index with Lode-angle/rate effects, "
    "principal-axis rotation, and energy input."
)

# =============================================================================
# Utility functions
# =============================================================================
def hann_window(t, td):
    """Finite-support Hann window, active for 0 <= t <= td."""
    tau = np.asarray(t)
    g = np.zeros_like(tau, dtype=float)
    mask = (tau >= 0.0) & (tau <= td)
    g[mask] = 0.5 * (1.0 - np.cos(2.0 * np.pi * tau[mask] / td))
    return g

def half_sine_window(t, td):
    """Finite-support half-sine pulse window, active for 0 <= t <= td."""
    tau = np.asarray(t)
    g = np.zeros_like(tau, dtype=float)
    mask = (tau >= 0.0) & (tau <= td)
    g[mask] = np.sin(np.pi * tau[mask] / td)
    return g

def rectangular_window(t, td):
    """Finite-support rectangular window."""
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

def central_difference(y, t):
    return np.gradient(y, t)

def compute_invariants(stress_tensor):
    """
    stress_tensor shape: (n, 3, 3), MPa.
    Returns p, q, theta_deg, J2, J3, principal stresses.
    """
    n = stress_tensor.shape[0]
    p = np.trace(stress_tensor, axis1=1, axis2=2) / 3.0
    I = np.eye(3)[None, :, :]
    s = stress_tensor - p[:, None, None] * I
    J2 = 0.5 * np.sum(s * s, axis=(1, 2))
    J3 = np.linalg.det(s)
    q = np.sqrt(np.maximum(3.0 * J2, 0.0))

    # Lode angle: cos(3θ) = (3√3/2) J3 / J2^(3/2)
    theta = np.zeros_like(p)
    mask = J2 > 1e-12
    arg = np.zeros_like(p)
    arg[mask] = (3.0 * np.sqrt(3.0) / 2.0) * J3[mask] / (J2[mask] ** 1.5)
    arg = np.clip(arg, -1.0, 1.0)
    theta[mask] = (1.0 / 3.0) * np.arccos(arg[mask])
    theta_deg = np.rad2deg(theta)

    # Principal stresses, sorted descending
    eigs = np.linalg.eigvalsh(stress_tensor)
    eigs = np.sort(eigs, axis=1)[:, ::-1]

    return p, q, theta_deg, J2, J3, eigs

def rotation_angle(sigma_a, sigma_b, tau_ab):
    """Principal-axis rotation in a coordinate plane, degrees."""
    return 0.5 * np.rad2deg(np.arctan2(2.0 * tau_ab, sigma_a - sigma_b))

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
L_mm = st.sidebar.number_input("Specimen length, L (mm)", value=50.0, min_value=1.0, step=1.0)

E = E_GPa * 1e9
G = E / (2.0 * (1.0 + nu))
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
M = lam + 2.0 * G
cp = np.sqrt(M / rho)
cs = np.sqrt(G / rho)
L_m = L_mm / 1000.0
t_travel = L_m / cp
t_eq_low = 3.0 * t_travel
t_eq_high = 5.0 * t_travel

st.sidebar.header("Initial true triaxial stresses")
sx0 = st.sidebar.number_input("σx0 (MPa)", value=30.0, min_value=0.0, step=1.0)
sy0 = st.sidebar.number_input("σy0 (MPa)", value=20.0, min_value=0.0, step=1.0)
sz0 = st.sidebar.number_input("σz0 (MPa)", value=15.0, min_value=0.0, step=1.0)

st.sidebar.header("Finite-duration pulses")
pulse_type = st.sidebar.selectbox("Pulse envelope", ["Hann", "Half-sine", "Rectangular"], index=0)
t_duration_us = st.sidebar.number_input("Pulse duration, td (μs)", value=60.0, min_value=1.0, step=5.0)
tmax_us = st.sidebar.number_input("Simulation time (μs)", value=100.0, min_value=5.0, step=5.0)
npts = st.sidebar.slider("Number of time points", 500, 20000, 3000, step=500)

st.sidebar.subheader("Amplitudes")
Ax = st.sidebar.number_input("Ax (MPa)", value=6.0, min_value=0.0, step=0.5)
Ay = st.sidebar.number_input("Ay (MPa)", value=5.0, min_value=0.0, step=0.5)
Az = st.sidebar.number_input("Az (MPa)", value=4.0, min_value=0.0, step=0.5)

st.sidebar.subheader("Carrier frequencies")
fx = st.sidebar.number_input("fx (kHz)", value=50.0, min_value=0.0, step=5.0)
fy = st.sidebar.number_input("fy (kHz)", value=70.0, min_value=0.0, step=5.0)
fz = st.sidebar.number_input("fz (kHz)", value=90.0, min_value=0.0, step=5.0)

st.sidebar.subheader("Phase angles")
phix = st.sidebar.number_input("φx (degrees)", value=0.0, step=5.0)
phiy = st.sidebar.number_input("φy (degrees)", value=45.0, step=5.0)
phiz = st.sidebar.number_input("φz (degrees)", value=90.0, step=5.0)

st.sidebar.subheader("Pulse delays")
delay_x_us = st.sidebar.number_input("Delay x (μs)", value=0.0, min_value=0.0, step=1.0)
delay_y_us = st.sidebar.number_input("Delay y (μs)", value=0.0, min_value=0.0, step=1.0)
delay_z_us = st.sidebar.number_input("Delay z (μs)", value=0.0, min_value=0.0, step=1.0)

st.sidebar.header("Optional shear / principal-axis rotation")
include_shear = st.sidebar.checkbox("Include small shear components", value=True)
tau_amp = st.sidebar.number_input("Shear amplitude scale, τamp (MPa)", value=1.0, min_value=0.0, step=0.2)
shear_phase = np.deg2rad(st.sidebar.number_input("Shear phase shift (degrees)", value=30.0, step=5.0))

st.sidebar.header("Failure envelope")
A_fail = st.sidebar.number_input("A in qf = (A + B pⁿ) h(θ) DIF (MPa)", value=3.0, step=0.5)
B_fail = st.sidebar.number_input("B in qf = (A + B pⁿ) h(θ) DIF", value=0.80, step=0.05)
n_fail = st.sidebar.number_input("n in qf = (A + B pⁿ) h(θ) DIF", value=0.80, step=0.05)
lode_strength_factor = st.sidebar.number_input("Lode-angle factor amplitude, aθ", value=0.10, step=0.02)
use_dif = st.sidebar.checkbox("Include strain-rate DIF", value=True)

st.sidebar.subheader("DIF parameters")
epsdot0 = st.sidebar.number_input("Reference strain rate, ε̇0 (s⁻¹)", value=1.0, min_value=1e-6, step=1.0)
epsdot_tr = st.sidebar.number_input("Transition strain rate, ε̇tr (s⁻¹)", value=50.0, min_value=1e-6, step=10.0)
A1 = st.sidebar.number_input("Low-rate DIF coefficient, A1", value=0.02, min_value=0.0, step=0.01)
B1 = st.sidebar.number_input("High-rate DIF coefficient, B1", value=1.0, min_value=0.0, step=0.05)

# =============================================================================
# Calculation
# =============================================================================
t_us = np.linspace(0.0, tmax_us, npts)
t = t_us * 1e-6
td = t_duration_us * 1e-6

delays = np.array([delay_x_us, delay_y_us, delay_z_us]) * 1e-6
amps = np.array([Ax, Ay, Az], dtype=float)
freqs = np.array([fx, fy, fz], dtype=float) * 1e3
phases = np.deg2rad(np.array([phix, phiy, phiz], dtype=float))

def pulse_component(A, f, phi, delay):
    tau = t - delay
    g = get_window(tau, td, pulse_type)
    if f == 0:
        carrier = np.ones_like(t)
    else:
        carrier = np.sin(2.0 * np.pi * f * tau + phi)
    return A * g * carrier, g

sx_dyn, gx = pulse_component(Ax, fx * 1e3, np.deg2rad(phix), delays[0])
sy_dyn, gy = pulse_component(Ay, fy * 1e3, np.deg2rad(phiy), delays[1])
sz_dyn, gz = pulse_component(Az, fz * 1e3, np.deg2rad(phiz), delays[2])

sx = sx0 + sx_dyn
sy = sy0 + sy_dyn
sz = sz0 + sz_dyn

# Optional shear terms to demonstrate principal-axis rotation
if include_shear and tau_amp > 0:
    # bounded, finite-window shear tied to combined active envelopes
    gxy = np.minimum(gx + gy, 1.0)
    gyz = np.minimum(gy + gz, 1.0)
    gzx = np.minimum(gz + gx, 1.0)
    base_freq = max((fx + fy + fz) / 3.0, 1.0) * 1e3
    tau_xy = tau_amp * gxy * np.sin(2.0 * np.pi * base_freq * t + shear_phase)
    tau_yz = 0.7 * tau_amp * gyz * np.sin(2.0 * np.pi * 1.15 * base_freq * t + 1.2 * shear_phase)
    tau_zx = 0.5 * tau_amp * gzx * np.sin(2.0 * np.pi * 0.85 * base_freq * t + 0.8 * shear_phase)
else:
    tau_xy = np.zeros_like(t)
    tau_yz = np.zeros_like(t)
    tau_zx = np.zeros_like(t)

stress = np.zeros((len(t), 3, 3), dtype=float)
stress[:, 0, 0] = sx
stress[:, 1, 1] = sy
stress[:, 2, 2] = sz
stress[:, 0, 1] = stress[:, 1, 0] = tau_xy
stress[:, 1, 2] = stress[:, 2, 1] = tau_yz
stress[:, 2, 0] = stress[:, 0, 2] = tau_zx

p, q, theta_deg, J2, J3, eigs = compute_invariants(stress)

# Equivalent strain rate estimate using elastic relation; useful for DIF demonstration
# strain approx = stress / E (MPa converted to Pa)
eps_x = sx * 1e6 / E
eps_y = sy * 1e6 / E
eps_z = sz * 1e6 / E
epsdot_x = central_difference(eps_x, t)
epsdot_y = central_difference(eps_y, t)
epsdot_z = central_difference(eps_z, t)
epsdot_eq = np.sqrt(2.0 / 3.0 * ((epsdot_x - epsdot_y)**2 + (epsdot_y - epsdot_z)**2 + (epsdot_z - epsdot_x)**2) / 2.0)
epsdot_abs = np.maximum(np.abs(epsdot_eq), 1e-12)

if use_dif:
    DIF = np.ones_like(t)
    low = epsdot_abs < epsdot_tr
    DIF[low] = 1.0 + A1 * np.log10(np.maximum(epsdot_abs[low] / epsdot0, 1e-12))
    DIF[~low] = B1 * (epsdot_abs[~low] / epsdot0) ** (1.0 / 3.0)
    DIF = np.maximum(DIF, 0.2)
else:
    DIF = np.ones_like(t)

# h(theta), normalised h(0)=1. Simple bounded illustrative factor.
theta_rad = np.deg2rad(theta_deg)
h_theta = 1.0 + lode_strength_factor * (1.0 - np.cos(3.0 * theta_rad))
qf = (A_fail + B_fail * np.maximum(p, 0.0) ** n_fail) * h_theta * DIF
F_index = q / np.maximum(qf, 1e-9)

alpha_xy = rotation_angle(sx, sy, tau_xy)
alpha_yz = rotation_angle(sy, sz, tau_yz)
alpha_zx = rotation_angle(sz, sx, tau_zx)

# Boundary particle velocities implied by acoustic impedance: A = rho cp vp
vp_x = (Ax * 1e6) / (rho * cp) if rho * cp > 0 else 0.0
vp_y = (Ay * 1e6) / (rho * cp) if rho * cp > 0 else 0.0
vp_z = (Az * 1e6) / (rho * cp) if rho * cp > 0 else 0.0

# Elastic energy density and work-like energy proxy
# Use MPa strain -> MJ/m3 because MPa = MJ/m3 for stress * strain
we_elastic = 0.5 / E_GPa / 1000.0 * 0  # placeholder not used
# Rigorous isotropic elastic energy density in MPa (MJ/m3): 1/(2E) [sigma^2 - 2nu cross], with E in MPa
E_MPa = E_GPa * 1000.0
W_el = (1.0 / (2.0 * E_MPa)) * (
    sx**2 + sy**2 + sz**2
    - 2.0 * nu * (sx * sy + sy * sz + sz * sx)
)

# input energy density proxy: integrate sigma * epsdot in each direction
power = sx * epsdot_x + sy * epsdot_y + sz * epsdot_z  # MPa/s = MJ/m3/s
dt = np.gradient(t)
W_input = np.cumsum(power * dt)
W_diss_proxy = W_input - W_el + W_el[0]

# =============================================================================
# Summary
# =============================================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("P-wave speed, cp", f"{cp:.0f} m/s")
col2.metric("S-wave speed, cs", f"{cs:.0f} m/s")
col3.metric("P-wave travel time", f"{t_travel*1e6:.2f} μs")
col4.metric("Equilibration time", f"{t_eq_low*1e6:.1f}–{t_eq_high*1e6:.1f} μs")

col5, col6, col7, col8 = st.columns(4)
col5.metric("vp,x from Ax", f"{vp_x:.3f} m/s")
col6.metric("vp,y from Ay", f"{vp_y:.3f} m/s")
col7.metric("vp,z from Az", f"{vp_z:.3f} m/s")
col8.metric("Peak failure index", f"{np.nanmax(F_index):.2f}")

st.info(
    "For Paper 1, keep the pulse delays in the synchronous/short-delay range "
    r"0 ≤ Δt ≤ 2 ttravel. Long-delay sequential loading (Δt* > 10) belongs to the companion wave–damage paper."
)

# =============================================================================
# Tabs
# =============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Stress waves",
    "p–q–θ invariants",
    "3D stress path",
    "Failure and energy",
    "Principal-axis rotation",
    "Export"
])

with tab1:
    st.subheader("Finite-duration stress waves")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(t_us, sx, label=r"$\sigma_x(t)$")
    ax1.plot(t_us, sy, label=r"$\sigma_y(t)$")
    ax1.plot(t_us, sz, label=r"$\sigma_z(t)$")
    ax1.axvline(t_travel * 1e6, linestyle="--", linewidth=1.0, label=r"$t_{\rm travel}$")
    ax1.axvspan(t_eq_low * 1e6, t_eq_high * 1e6, alpha=0.15, label=r"$t_{\rm eq}\approx 3$--$5t_{\rm travel}$")
    ax1.set_xlabel("Time (μs)")
    ax1.set_ylabel("Stress (MPa)")
    ax1.set_title("Windowed orthogonal stress pulses")
    ax1.grid(True, alpha=0.35)
    ax1.legend(ncol=2)
    st.pyplot(fig1)
    st.download_button("Download stress-wave figure", fig_to_bytes(fig1), "paper1_stress_waves.png", "image/png")

    fig1b, ax1b = plt.subplots(figsize=(10, 3))
    ax1b.plot(t_us, gx, label="gx")
    ax1b.plot(t_us, gy, label="gy")
    ax1b.plot(t_us, gz, label="gz")
    ax1b.set_xlabel("Time (μs)")
    ax1b.set_ylabel("Pulse envelope")
    ax1b.set_title(f"{pulse_type} envelope")
    ax1b.grid(True, alpha=0.35)
    ax1b.legend()
    st.pyplot(fig1b)

with tab2:
    st.subheader("Invariant stress paths")
    c1, c2 = st.columns([1.1, 1.0])

    with c1:
        fig2, ax2 = plt.subplots(figsize=(7, 5.5))
        sc = ax2.scatter(p, q, c=t_us, s=8)
        pp = np.linspace(max(0.0, np.nanmin(p) * 0.95), max(np.nanmax(p) * 1.05, 1.0), 300)
        # show static envelope without theta/rate, and dynamic-like median correction
        q_env_static = A_fail + B_fail * pp ** n_fail
        ax2.plot(pp, q_env_static, linestyle="--", linewidth=1.6, label=r"Base envelope $A+Bp^n$")
        for idx in np.linspace(0, len(p) - 2, 12, dtype=int):
            ax2.annotate("", xy=(p[idx + 1], q[idx + 1]), xytext=(p[idx], q[idx]),
                         arrowprops=dict(arrowstyle="->", lw=0.7))
        ax2.set_xlabel("Mean stress, p (MPa)")
        ax2.set_ylabel("Deviatoric stress, q (MPa)")
        ax2.set_title("p–q stress path")
        ax2.grid(True, alpha=0.35)
        ax2.legend()
        cb = fig2.colorbar(sc, ax=ax2)
        cb.set_label("Time (μs)")
        st.pyplot(fig2)
        st.download_button("Download p–q path", fig_to_bytes(fig2), "paper1_pq_path.png", "image/png")

    with c2:
        fig3, ax3 = plt.subplots(figsize=(7, 5.5))
        sc2 = ax3.scatter(q, theta_deg, c=t_us, s=8)
        ax3.set_xlabel("q (MPa)")
        ax3.set_ylabel("Lode angle, θ (degrees)")
        ax3.set_title("q–θ projection")
        ax3.grid(True, alpha=0.35)
        cb2 = fig3.colorbar(sc2, ax=ax3)
        cb2.set_label("Time (μs)")
        st.pyplot(fig3)
        st.download_button("Download q–theta path", fig_to_bytes(fig3), "paper1_q_theta_path.png", "image/png")

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(t_us, p, label="p")
    ax4.plot(t_us, q, label="q")
    ax4.plot(t_us, theta_deg, label="θ (degrees)")
    ax4.set_xlabel("Time (μs)")
    ax4.set_ylabel("Value")
    ax4.set_title("Invariant histories")
    ax4.grid(True, alpha=0.35)
    ax4.legend()
    st.pyplot(fig4)

with tab3:
    st.subheader("3D principal-stress trajectory")
    fig5 = plt.figure(figsize=(7, 6))
    ax5 = fig5.add_subplot(111, projection="3d")
    ax5.plot(eigs[:, 0], eigs[:, 1], eigs[:, 2], linewidth=1.5)
    ax5.scatter([eigs[0, 0]], [eigs[0, 1]], [eigs[0, 2]], s=40, label="Start")
    ax5.scatter([eigs[-1, 0]], [eigs[-1, 1]], [eigs[-1, 2]], s=40, marker="^", label="End")
    ax5.set_xlabel(r"$\sigma_1$ (MPa)")
    ax5.set_ylabel(r"$\sigma_2$ (MPa)")
    ax5.set_zlabel(r"$\sigma_3$ (MPa)")
    ax5.set_title("Principal-stress path")
    ax5.legend()
    st.pyplot(fig5)
    st.download_button("Download 3D stress path", fig_to_bytes(fig5), "paper1_3d_principal_stress_path.png", "image/png")

    fig5b = plt.figure(figsize=(7, 6))
    ax5b = fig5b.add_subplot(111, projection="3d")
    ax5b.plot(sx, sy, sz, linewidth=1.5)
    ax5b.set_xlabel(r"$\sigma_x$ (MPa)")
    ax5b.set_ylabel(r"$\sigma_y$ (MPa)")
    ax5b.set_zlabel(r"$\sigma_z$ (MPa)")
    ax5b.set_title("Coordinate stress path")
    st.pyplot(fig5b)

with tab4:
    st.subheader("Failure-envelope interaction and energy")
    c1, c2 = st.columns(2)

    with c1:
        fig6, ax6 = plt.subplots(figsize=(7, 4.5))
        ax6.plot(t_us, F_index, label=r"$F(t)=q/q_f$")
        ax6.axhline(1.0, linestyle="--", linewidth=1.2, label="Failure threshold")
        ax6.set_xlabel("Time (μs)")
        ax6.set_ylabel("Failure index")
        ax6.set_title("Failure-envelope interaction")
        ax6.grid(True, alpha=0.35)
        ax6.legend()
        st.pyplot(fig6)
        st.download_button("Download failure-index figure", fig_to_bytes(fig6), "paper1_failure_index.png", "image/png")

    with c2:
        fig7, ax7 = plt.subplots(figsize=(7, 4.5))
        ax7.plot(t_us, W_input, label=r"$w_{\rm input}$")
        ax7.plot(t_us, W_el, label=r"$w_e$")
        ax7.plot(t_us, W_diss_proxy, label=r"$w_{\rm diss}$ proxy")
        ax7.set_xlabel("Time (μs)")
        ax7.set_ylabel("Energy density (MJ/m³)")
        ax7.set_title("Energy-density indicators")
        ax7.grid(True, alpha=0.35)
        ax7.legend()
        st.pyplot(fig7)
        st.download_button("Download energy figure", fig_to_bytes(fig7), "paper1_energy_indicators.png", "image/png")

    st.markdown(
        """
        **Interpretation:**  
        The failure index follows the paper formulation  
        \(F(t)=q(t)/q_f[p(t),\\theta(t),\\dot\\varepsilon(t)]\).  
        The energy plot is an analytical proxy, not a DEM fracture-energy calculation. 
        In DEM, use contact-bond breakage, frictional slip and damping work for rigorous energy partition.
        """
    )

with tab5:
    st.subheader("Principal-axis rotation")
    fig8, ax8 = plt.subplots(figsize=(10, 4.8))
    ax8.plot(t_us, alpha_xy, label=r"$\alpha_{xy}$")
    ax8.plot(t_us, alpha_yz, label=r"$\alpha_{yz}$")
    ax8.plot(t_us, alpha_zx, label=r"$\alpha_{zx}$")
    ax8.set_xlabel("Time (μs)")
    ax8.set_ylabel("Rotation angle (degrees)")
    ax8.set_title("Principal-axis rotation induced by shear components")
    ax8.grid(True, alpha=0.35)
    ax8.legend()
    st.pyplot(fig8)
    st.download_button("Download rotation figure", fig_to_bytes(fig8), "paper1_principal_axis_rotation.png", "image/png")

    if not include_shear:
        st.warning(
            "Shear components are disabled. For purely diagonal stresses aligned with x, y and z, "
            "the eigenvectors remain fixed; only the principal stress magnitudes and ordering change."
        )

with tab6:
    st.subheader("Export calculated data")
    df = pd.DataFrame({
        "time_us": t_us,
        "sigma_x_MPa": sx,
        "sigma_y_MPa": sy,
        "sigma_z_MPa": sz,
        "tau_xy_MPa": tau_xy,
        "tau_yz_MPa": tau_yz,
        "tau_zx_MPa": tau_zx,
        "p_MPa": p,
        "q_MPa": q,
        "theta_deg": theta_deg,
        "J2_MPa2": J2,
        "J3_MPa3": J3,
        "sigma1_MPa": eigs[:, 0],
        "sigma2_MPa": eigs[:, 1],
        "sigma3_MPa": eigs[:, 2],
        "alpha_xy_deg": alpha_xy,
        "alpha_yz_deg": alpha_yz,
        "alpha_zx_deg": alpha_zx,
        "epsdot_eq_s-1": epsdot_eq,
        "DIF": DIF,
        "qf_MPa": qf,
        "failure_index": F_index,
        "W_input_MJ_m3": W_input,
        "W_elastic_MJ_m3": W_el,
        "W_diss_proxy_MJ_m3": W_diss_proxy,
    })
    st.dataframe(df.head(30), use_container_width=True)
    st.download_button(
        "Download full results as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="paper1_final_streamlit_results.csv",
        mime="text/csv",
    )

    st.markdown("### Suggested manuscript wording")
    st.code(
        """The analytical stress histories were generated using finite-duration windowed pulses. 
The resulting stress tensor was transformed into the invariant quantities p(t), q(t) and θ(t), 
and the transient failure index was evaluated using a pressure-, Lode-angle- and rate-dependent envelope. 
Principal-axis rotation was quantified from the shear components of the stress tensor, while the DEM stress 
calculation can be mapped to the same invariant framework using the symmetrised Weber--Love expression.""",
        language="text",
    )
