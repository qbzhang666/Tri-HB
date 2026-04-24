
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Anisotropic cumulative damage under multidirectional harmonic loading",
    layout="wide"
)

st.title("Anisotropic cumulative damage under multidirectional harmonic wave superposition")
st.caption(
    "Concise Streamlit demo for the cumulative damage model: stress-path invariants, "
    "tensorial damage evolution, damage anisotropy, CT crack-density mapping and wave-speed degradation."
)

# -----------------------------
# Helper functions
# -----------------------------
def window(t, td, kind):
    g = np.zeros_like(t)
    mask = (t >= 0) & (t <= td)
    if kind == "Hann":
        g[mask] = 0.5 * (1 - np.cos(2 * np.pi * t[mask] / td))
    elif kind == "Half-sine":
        g[mask] = np.sin(np.pi * t[mask] / td)
    else:
        g[mask] = 1.0
    return g

def pulse(t, A, f, phi, td, kind):
    g = window(t, td, kind)
    return A * g * np.sin(2 * np.pi * f * t + phi), g

def invariants(sx, sy, sz):
    p = (sx + sy + sz) / 3.0
    s1, s2, s3 = sx - p, sy - p, sz - p
    J2 = 0.5 * (s1**2 + s2**2 + s3**2)
    J3 = s1 * s2 * s3
    q = np.sqrt(np.maximum(3 * J2, 0.0))

    theta = np.zeros_like(p)
    m = J2 > 1e-12
    arg = np.zeros_like(p)
    arg[m] = (3 * np.sqrt(3) / 2) * J3[m] / (J2[m] ** 1.5)
    arg = np.clip(arg, -1, 1)
    theta[m] = np.rad2deg((1 / 3) * np.arccos(arg[m]))
    return p, q, theta, J2, J3

def damage_anisotropy(Dx, Dy, Dz):
    trD = Dx + Dy + Dz
    Diso = trD / 3.0
    dev = np.array([Dx - Diso, Dy - Diso, Dz - Diso])
    J2D = 0.5 * np.sum(dev**2)
    return np.sqrt(3 * J2D) / max(trD, 1e-12)

def trapezoid_cumulative(y, t):
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))
    return out

def fig_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Material and specimen")
E_GPa = st.sidebar.number_input("Young's modulus E (GPa)", value=50.0, min_value=1.0, step=1.0)
nu = st.sidebar.number_input("Poisson's ratio ν", value=0.25, min_value=0.0, max_value=0.49, step=0.01)
rho = st.sidebar.number_input("Density ρ (kg/m³)", value=2650.0, min_value=1000.0, step=50.0)
L_mm = st.sidebar.number_input("Specimen size L (mm)", value=50.0, min_value=1.0, step=1.0)

E = E_GPa * 1e9
M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
cp0 = np.sqrt(M / rho)
t_travel = (L_mm / 1000) / cp0

st.sidebar.header("Initial true triaxial stress")
sx0 = st.sidebar.number_input("σx0 (MPa)", value=30.0, min_value=0.0, step=1.0)
sy0 = st.sidebar.number_input("σy0 (MPa)", value=20.0, min_value=0.0, step=1.0)
sz0 = st.sidebar.number_input("σz0 (MPa)", value=15.0, min_value=0.0, step=1.0)

st.sidebar.header("Loading configuration")
case = st.sidebar.selectbox("Dynamic loading path", ["Uniaxial X", "Biaxial XY", "Triaxial XYZ"], index=1)
window_type = st.sidebar.selectbox("Pulse envelope", ["Hann", "Half-sine", "Rectangular"], index=0)
tmax_us = st.sidebar.number_input("Simulation time (μs)", value=120.0, min_value=10.0, step=10.0)
td_us = st.sidebar.number_input("Pulse duration (μs)", value=80.0, min_value=1.0, step=5.0)
npts = st.sidebar.slider("Time points", 1000, 20000, 4000, step=500)

st.sidebar.subheader("Wave amplitudes")
Ax = st.sidebar.number_input("Ax (MPa)", value=60.0, min_value=0.0, step=5.0)
Ay = st.sidebar.number_input("Ay (MPa)", value=60.0, min_value=0.0, step=5.0)
Az = st.sidebar.number_input("Az (MPa)", value=60.0, min_value=0.0, step=5.0)

st.sidebar.subheader("Frequency and phase")
fx = st.sidebar.number_input("fx (kHz)", value=50.0, min_value=0.0, step=5.0)
fy = st.sidebar.number_input("fy (kHz)", value=50.0, min_value=0.0, step=5.0)
fz = st.sidebar.number_input("fz (kHz)", value=50.0, min_value=0.0, step=5.0)
phix = np.deg2rad(st.sidebar.number_input("φx (deg)", value=0.0, step=5.0))
phiy = np.deg2rad(st.sidebar.number_input("φy (deg)", value=0.0, step=5.0))
phiz = np.deg2rad(st.sidebar.number_input("φz (deg)", value=0.0, step=5.0))

st.sidebar.header("Failure and damage evolution")
A_fail = st.sidebar.number_input("A in qf=(A+Bpⁿ)h(θ)DIF", value=15.0, min_value=0.0, step=1.0)
B_fail = st.sidebar.number_input("B in qf", value=1.3, min_value=0.0, step=0.1)
n_fail = st.sidebar.number_input("n in qf", value=0.75, min_value=0.1, step=0.05)
atheta = st.sidebar.number_input("Lode-angle factor aθ", value=0.10, min_value=0.0, step=0.02)

tau_D_us = st.sidebar.number_input("Damage time scale τD (μs)", value=25.0, min_value=0.1, step=2.5)
alpha = st.sidebar.number_input("Saturation exponent α", value=1.0, min_value=0.0, step=0.2)
mexp = st.sidebar.number_input("Overstress exponent m", value=2.0, min_value=0.1, step=0.2)
beta = st.sidebar.number_input("Rate exponent β", value=0.15, min_value=0.0, step=0.05)
epsdot0 = st.sidebar.number_input("Reference strain rate ε̇0 (s⁻¹)", value=1.0, min_value=1e-6, step=1.0)

st.sidebar.header("CT mapping")
k_ct = st.sidebar.number_input("CT mapping Dii = k αii", value=5.0, min_value=0.1, step=0.1)

# -----------------------------
# Calculations
# -----------------------------
t_us = np.linspace(0, tmax_us, npts)
t = t_us * 1e-6
td = td_us * 1e-6

active_x = True
active_y = case in ["Biaxial XY", "Triaxial XYZ"]
active_z = case == "Triaxial XYZ"

sx_dyn, gx = pulse(t, Ax if active_x else 0.0, fx * 1e3, phix, td, window_type)
sy_dyn, gy = pulse(t, Ay if active_y else 0.0, fy * 1e3, phiy, td, window_type)
sz_dyn, gz = pulse(t, Az if active_z else 0.0, fz * 1e3, phiz, td, window_type)

sx = sx0 + sx_dyn
sy = sy0 + sy_dyn
sz = sz0 + sz_dyn

p, q, theta, J2, J3 = invariants(sx, sy, sz)

E_MPa = E_GPa * 1000
eps_x = sx / E_MPa
eps_y = sy / E_MPa
eps_z = sz / E_MPa
edotx = np.gradient(eps_x, t)
edoty = np.gradient(eps_y, t)
edotz = np.gradient(eps_z, t)
epsdot_eq = np.sqrt(
    np.maximum(2/3 * ((edotx-edoty)**2 + (edoty-edotz)**2 + (edotz-edotx)**2) / 2, 0)
)

# failure index
h_theta = 1 + atheta * (1 - np.cos(3 * np.deg2rad(theta)))
DIF = np.maximum(1.0 + 0.02 * np.log10(np.maximum(epsdot_eq, 1e-12) / epsdot0), 0.2)
qf = (A_fail + B_fail * np.maximum(p, 0)**n_fail) * h_theta * DIF
F = q / np.maximum(qf, 1e-9)

# Directional thermodynamic-force proxy and tensorial damage evolution
# Use positive overstress and directional stress work proxy.
Yx = sx**2 / (2 * E_MPa)
Yy = sy**2 / (2 * E_MPa)
Yz = sz**2 / (2 * E_MPa)
Ynorm = np.sqrt(Yx**2 + Yy**2 + Yz**2) + 1e-12
Nx, Ny, Nz = Yx/Ynorm, Yy/Ynorm, Yz/Ynorm

Dx = np.zeros_like(t)
Dy = np.zeros_like(t)
Dz = np.zeros_like(t)
Ddotx = np.zeros_like(t)
Ddoty = np.zeros_like(t)
Ddotz = np.zeros_like(t)
tau_D = tau_D_us * 1e-6

for i in range(1, len(t)):
    dt = t[i] - t[i-1]
    overstress = max(F[i-1] - 1, 0.0)
    rate_factor = (max(epsdot_eq[i-1], 1e-12) / epsdot0) ** beta
    scale = (overstress ** mexp) * rate_factor / tau_D

    Ddotx[i-1] = ((1-Dx[i-1])**(2*alpha)) * scale * Nx[i-1]
    Ddoty[i-1] = ((1-Dy[i-1])**(2*alpha)) * scale * Ny[i-1]
    Ddotz[i-1] = ((1-Dz[i-1])**(2*alpha)) * scale * Nz[i-1]

    Dx[i] = np.clip(Dx[i-1] + Ddotx[i-1] * dt, 0, 0.999)
    Dy[i] = np.clip(Dy[i-1] + Ddoty[i-1] * dt, 0, 0.999)
    Dz[i] = np.clip(Dz[i-1] + Ddotz[i-1] * dt, 0, 0.999)

xi = np.array([damage_anisotropy(Dx[i], Dy[i], Dz[i]) for i in range(len(t))])
trD = Dx + Dy + Dz
Diso = trD / 3

cpx = cp0 * np.sqrt(np.maximum(1 - Dx, 0))
cpy = cp0 * np.sqrt(np.maximum(1 - Dy, 0))
cpz = cp0 * np.sqrt(np.maximum(1 - Dz, 0))

alphax = Dx / k_ct
alphay = Dy / k_ct
alphaz = Dz / k_ct

# Energy proxy
power = sx * edotx + sy * edoty + sz * edotz  # MPa/s = MJ/m3/s
W_input = trapezoid_cumulative(power, t)
W_el = (1/(2*E_MPa)) * (sx**2 + sy**2 + sz**2 - 2*nu*(sx*sy + sy*sz + sz*sx))
W_diss = W_input - W_el + W_el[0]

# -----------------------------
# Summary metrics
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("P-wave speed", f"{cp0:.0f} m/s")
c2.metric("Travel time", f"{t_travel*1e6:.2f} μs")
c3.metric("Final tr(D)", f"{trD[-1]:.3f}")
c4.metric("Final ξD", f"{xi[-1]:.3f}")

c5, c6, c7 = st.columns(3)
c5.metric("Final Dxx", f"{Dx[-1]:.3f}")
c6.metric("Final Dyy", f"{Dy[-1]:.3f}")
c7.metric("Final Dzz", f"{Dz[-1]:.3f}")

st.info(
    "Use this app as a model-demonstration and figure generator. "
    "For the final paper, replace the synthetic damage histories with calibrated T-HPB, DEM and CT-derived values."
)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Stress waves and stress path",
    "Damage tensor",
    "CT and wave-speed validation",
    "Energy and thermodynamics",
    "Comparison of loading paths",
    "Export"
])

with tab1:
    st.subheader("Stress histories and invariant stress path")
    colA, colB = st.columns(2)
    with colA:
        fig, ax = plt.subplots(figsize=(7,4.6))
        ax.plot(t_us, sx, label=r"$\sigma_x$")
        ax.plot(t_us, sy, label=r"$\sigma_y$")
        ax.plot(t_us, sz, label=r"$\sigma_z$")
        ax.axvline(t_travel*1e6, ls="--", lw=1, label=r"$t_{\rm travel}$")
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Stress (MPa)")
        ax.set_title("Dynamic stress histories")
        ax.grid(True, alpha=0.35)
        ax.legend()
        st.pyplot(fig)
        st.download_button("Download stress histories", fig_bytes(fig), "damage_stress_histories.png", "image/png")
    with colB:
        fig, ax = plt.subplots(figsize=(7,4.6))
        sc = ax.scatter(p, q, c=t_us, s=8)
        ax.set_xlabel("p (MPa)")
        ax.set_ylabel("q (MPa)")
        ax.set_title("p–q trajectory")
        ax.grid(True, alpha=0.35)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("Time (μs)")
        st.pyplot(fig)
        st.download_button("Download p-q path", fig_bytes(fig), "damage_pq_path.png", "image/png")

    fig, ax = plt.subplots(figsize=(10,3.8))
    ax.plot(t_us, p, label="p")
    ax.plot(t_us, q, label="q")
    ax.plot(t_us, theta, label="θ (degrees)")
    ax.set_xlabel("Time (μs)")
    ax.set_ylabel("Invariant value")
    ax.set_title("Stress-path invariant histories")
    ax.grid(True, alpha=0.35)
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.subheader("Tensorial cumulative damage evolution")
    colA, colB = st.columns(2)
    with colA:
        fig, ax = plt.subplots(figsize=(7,4.6))
        ax.plot(t_us, Dx, label=r"$D_{xx}$")
        ax.plot(t_us, Dy, label=r"$D_{yy}$")
        ax.plot(t_us, Dz, label=r"$D_{zz}$")
        ax.plot(t_us, Diso, "--", label=r"$D_{\rm iso}$")
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Damage")
        ax.set_ylim(0, max(0.05, min(1, np.nanmax([Dx.max(), Dy.max(), Dz.max()])*1.15)))
        ax.set_title("Damage tensor components")
        ax.grid(True, alpha=0.35)
        ax.legend()
        st.pyplot(fig)
        st.download_button("Download damage components", fig_bytes(fig), "damage_tensor_components.png", "image/png")
    with colB:
        fig, ax = plt.subplots(figsize=(7,4.6))
        ax.plot(t_us, xi, label=r"$\xi_D$")
        ax.plot(t_us, trD, label=r"$\mathrm{tr}(D)$")
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Scalar damage indicator")
        ax.set_title("Damage anisotropy and total damage")
        ax.grid(True, alpha=0.35)
        ax.legend()
        st.pyplot(fig)
        st.download_button("Download anisotropy", fig_bytes(fig), "damage_anisotropy.png", "image/png")

    st.latex(r"\xi_D=\frac{\sqrt{3J_2(D_{\rm dev})}}{\operatorname{tr}(D)}")

with tab3:
    st.subheader("CT crack-density mapping and wave-speed degradation")
    colA, colB = st.columns(2)
    with colA:
        fig, ax = plt.subplots(figsize=(7,4.6))
        ax.plot(t_us, alphax, label=r"$\alpha_{xx}=D_{xx}/k$")
        ax.plot(t_us, alphay, label=r"$\alpha_{yy}=D_{yy}/k$")
        ax.plot(t_us, alphaz, label=r"$\alpha_{zz}=D_{zz}/k$")
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Crack-density tensor component")
        ax.set_title("CT-equivalent crack density")
        ax.grid(True, alpha=0.35)
        ax.legend()
        st.pyplot(fig)
    with colB:
        fig, ax = plt.subplots(figsize=(7,4.6))
        ax.plot(t_us, cpx/cp0, label=r"$c_{p,x}/c_{p,0}$")
        ax.plot(t_us, cpy/cp0, label=r"$c_{p,y}/c_{p,0}$")
        ax.plot(t_us, cpz/cp0, label=r"$c_{p,z}/c_{p,0}$")
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Normalised wave speed")
        ax.set_title("Directional wave-speed degradation")
        ax.grid(True, alpha=0.35)
        ax.legend()
        st.pyplot(fig)
        st.download_button("Download wave-speed degradation", fig_bytes(fig), "wave_speed_degradation.png", "image/png")

    st.latex(r"D_{ii}\approx k\alpha_{ii},\qquad c_{p,i}=c_{p,0}\sqrt{1-D_{ii}}")

with tab4:
    st.subheader("Failure index, energy and thermodynamic force proxy")
    colA, colB = st.columns(2)
    with colA:
        fig, ax = plt.subplots(figsize=(7,4.6))
        ax.plot(t_us, F, label="F(t)")
        ax.axhline(1, ls="--", label="damage threshold")
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Failure index")
        ax.set_title("Failure-envelope interaction")
        ax.grid(True, alpha=0.35)
        ax.legend()
        st.pyplot(fig)
    with colB:
        fig, ax = plt.subplots(figsize=(7,4.6))
        ax.plot(t_us, W_input, label=r"$w_{\rm input}$")
        ax.plot(t_us, W_el, label=r"$w_e$")
        ax.plot(t_us, W_diss, label=r"$w_{\rm diss}$ proxy")
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Energy density (MJ/m³)")
        ax.set_title("Energy indicators")
        ax.grid(True, alpha=0.35)
        ax.legend()
        st.pyplot(fig)

with tab5:
    st.subheader("Quick comparison of loading-path dimensionality")

    cases = ["Uniaxial X", "Biaxial XY", "Triaxial XYZ"]
    summary = []
    for cc in cases:
        # indicative patterns from the paper; can be replaced by batch simulations
        if cc == "Uniaxial X":
            vals = (0.15, 0.03, 0.03)
        elif cc == "Biaxial XY":
            vals = (0.10, 0.10, 0.03)
        else:
            vals = (0.08, 0.08, 0.08)
        xii = damage_anisotropy(*vals)
        summary.append([cc, vals[0], vals[1], vals[2], sum(vals), xii])

    df_summary = pd.DataFrame(summary, columns=["Configuration", "Dxx", "Dyy", "Dzz", "tr(D)", "ξD"])
    st.dataframe(df_summary, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8,4.6))
    xpos = np.arange(len(df_summary))
    width = 0.22
    ax.bar(xpos - width, df_summary["Dxx"], width, label="Dxx")
    ax.bar(xpos, df_summary["Dyy"], width, label="Dyy")
    ax.bar(xpos + width, df_summary["Dzz"], width, label="Dzz")
    ax.set_xticks(xpos)
    ax.set_xticklabels(df_summary["Configuration"])
    ax.set_ylabel("Representative damage")
    ax.set_title("Expected damage morphology by loading path")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8,4.6))
    ax.plot(df_summary["Configuration"], df_summary["ξD"], marker="o", label=r"$\xi_D$")
    ax.plot(df_summary["Configuration"], df_summary["tr(D)"], marker="s", label=r"$\mathrm{tr}(D)$")
    ax.set_ylabel("Indicator")
    ax.set_title("Anisotropy decreases as loading dimensionality increases")
    ax.grid(True, alpha=0.35)
    ax.legend()
    st.pyplot(fig)

with tab6:
    st.subheader("Export results")
    df = pd.DataFrame({
        "time_us": t_us,
        "sigma_x_MPa": sx,
        "sigma_y_MPa": sy,
        "sigma_z_MPa": sz,
        "p_MPa": p,
        "q_MPa": q,
        "theta_deg": theta,
        "failure_index_F": F,
        "Dxx": Dx,
        "Dyy": Dy,
        "Dzz": Dz,
        "trD": trD,
        "Diso": Diso,
        "xi_D": xi,
        "alpha_xx": alphax,
        "alpha_yy": alphay,
        "alpha_zz": alphaz,
        "cp_x_over_cp0": cpx/cp0,
        "cp_y_over_cp0": cpy/cp0,
        "cp_z_over_cp0": cpz/cp0,
        "W_input_MJ_m3": W_input,
        "W_elastic_MJ_m3": W_el,
        "W_diss_proxy_MJ_m3": W_diss
    })
    st.dataframe(df.head(30), use_container_width=True)
    st.download_button(
        "Download results as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="anisotropic_cumulative_damage_results.csv",
        mime="text/csv"
    )
