
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Tri-HB Stress-Wave Superposition Demo",
    layout="wide"
)

st.title("Multidirectional Stress-Wave Superposition in True Triaxial Impact")
st.caption("Paper 1 demo: analytical stress histories, p–q stress path, failure envelope, and 3D stress trajectory.")

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Initial true triaxial stresses")
sx0 = st.sidebar.number_input("σx0 (MPa)", value=30.0, min_value=0.0, step=1.0)
sy0 = st.sidebar.number_input("σy0 (MPa)", value=20.0, min_value=0.0, step=1.0)
sz0 = st.sidebar.number_input("σz0 (MPa)", value=15.0, min_value=0.0, step=1.0)

st.sidebar.header("Dynamic wave amplitudes")
Ax = st.sidebar.number_input("Ax (MPa)", value=6.0, min_value=0.0, step=0.5)
Ay = st.sidebar.number_input("Ay (MPa)", value=5.0, min_value=0.0, step=0.5)
Az = st.sidebar.number_input("Az (MPa)", value=4.0, min_value=0.0, step=0.5)

st.sidebar.header("Frequencies")
fx = st.sidebar.number_input("fx (kHz)", value=50.0, min_value=1.0, step=5.0)
fy = st.sidebar.number_input("fy (kHz)", value=70.0, min_value=1.0, step=5.0)
fz = st.sidebar.number_input("fz (kHz)", value=90.0, min_value=1.0, step=5.0)

st.sidebar.header("Phase angles")
phix = st.sidebar.number_input("φx (degrees)", value=0.0, step=5.0)
phiy = st.sidebar.number_input("φy (degrees)", value=45.0, step=5.0)
phiz = st.sidebar.number_input("φz (degrees)", value=90.0, step=5.0)

st.sidebar.header("Specimen and time scale")
L_mm = st.sidebar.number_input("Specimen length L (mm)", value=50.0, min_value=1.0, step=1.0)
c_ms = st.sidebar.number_input("Wave speed c (m/s)", value=5000.0, min_value=100.0, step=100.0)
tmax_us = st.sidebar.number_input("Simulation time (μs)", value=60.0, min_value=1.0, step=5.0)
npts = st.sidebar.slider("Number of time points", 500, 10000, 2000, step=500)

st.sidebar.header("Failure envelope")
A_fail = st.sidebar.number_input("A in qf = A + B p^n (MPa)", value=3.0, step=0.5)
B_fail = st.sidebar.number_input("B in qf = A + B p^n", value=0.80, step=0.05)
n_fail = st.sidebar.number_input("n in qf = A + B p^n", value=0.80, step=0.05)

# -----------------------------
# Calculations
# -----------------------------
t_us = np.linspace(0.0, tmax_us, npts)
t = t_us * 1e-6

wx, wy, wz = 2*np.pi*fx*1e3, 2*np.pi*fy*1e3, 2*np.pi*fz*1e3
px, py, pz = np.deg2rad(phix), np.deg2rad(phiy), np.deg2rad(phiz)

sx = sx0 + Ax*np.sin(wx*t + px)
sy = sy0 + Ay*np.sin(wy*t + py)
sz = sz0 + Az*np.sin(wz*t + pz)

p = (sx + sy + sz) / 3.0
q = np.sqrt(0.5*((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2))
eta = q / np.maximum(p, 1e-9)
qf = A_fail + B_fail*np.maximum(p, 0)**n_fail
F = q / np.maximum(qf, 1e-9)

L_m = L_mm / 1000.0
t_travel_us = L_m / c_ms * 1e6
n_crossings = tmax_us / t_travel_us if t_travel_us > 0 else np.nan

df = pd.DataFrame({
    "time_us": t_us,
    "sigma_x_MPa": sx,
    "sigma_y_MPa": sy,
    "sigma_z_MPa": sz,
    "p_MPa": p,
    "q_MPa": q,
    "eta": eta,
    "failure_index": F
})

# -----------------------------
# Summary metrics
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Wave travel time", f"{t_travel_us:.2f} μs")
col2.metric("Simulated crossings", f"{n_crossings:.1f}")
col3.metric("Initial p0", f"{(sx0+sy0+sz0)/3:.2f} MPa")
q0 = np.sqrt(0.5*((sx0-sy0)**2 + (sy0-sz0)**2 + (sz0-sx0)**2))
col4.metric("Initial q0", f"{q0:.2f} MPa")

col5, col6, col7 = st.columns(3)
col5.metric("Peak q", f"{np.max(q):.2f} MPa")
col6.metric("Peak failure index", f"{np.max(F):.2f}")
col7.metric("Failure crossing?", "Yes" if np.any(F >= 1.0) else "No")

# -----------------------------
# Plot helpers
# -----------------------------
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf

# -----------------------------
# Stress histories
# -----------------------------
st.subheader("1. Stress histories")
fig1, ax1 = plt.subplots(figsize=(9, 4.8))
ax1.plot(t_us, sx, label=r"$\sigma_x$")
ax1.plot(t_us, sy, label=r"$\sigma_y$")
ax1.plot(t_us, sz, label=r"$\sigma_z$")
ax1.set_xlabel("Time (μs)")
ax1.set_ylabel("Stress (MPa)")
ax1.set_title("Principal stress histories under multidirectional sinusoidal impact")
ax1.grid(True, alpha=0.35)
ax1.legend()
st.pyplot(fig1)
st.download_button("Download stress-history figure", data=fig_to_bytes(fig1), file_name="paper1_stress_histories.png", mime="image/png")

# -----------------------------
# p-q stress path
# -----------------------------
st.subheader("2. p–q stress path with failure envelope")
fig2, ax2 = plt.subplots(figsize=(7.2, 5.8))
sc = ax2.scatter(p, q, c=t_us, s=8)
p_env = np.linspace(max(0, np.min(p)*0.95), np.max(p)*1.05, 300)
q_env = A_fail + B_fail*np.maximum(p_env, 0)**n_fail
ax2.plot(p_env, q_env, linewidth=2.0, label=r"$q_f=A+Bp^n$")
# arrows along path
for idx in np.linspace(0, len(p)-2, 12, dtype=int):
    ax2.annotate("", xy=(p[idx+1], q[idx+1]), xytext=(p[idx], q[idx]),
                 arrowprops=dict(arrowstyle="->", lw=0.8))
ax2.set_xlabel("Mean stress, p (MPa)")
ax2.set_ylabel("Deviatoric stress, q (MPa)")
ax2.set_title("Time-dependent stress path in p–q space")
ax2.grid(True, alpha=0.35)
ax2.legend()
cbar = fig2.colorbar(sc, ax=ax2)
cbar.set_label("Time (μs)")
st.pyplot(fig2)
st.download_button("Download p–q figure", data=fig_to_bytes(fig2), file_name="paper1_pq_path.png", mime="image/png")

# -----------------------------
# 3D stress path
# -----------------------------
st.subheader("3. 3D principal stress trajectory")
fig3 = plt.figure(figsize=(7.0, 6.0))
ax3 = fig3.add_subplot(111, projection="3d")
ax3.plot(sx, sy, sz, linewidth=1.5)
ax3.scatter([sx[0]], [sy[0]], [sz[0]], marker="o", s=40, label="Start")
ax3.scatter([sx[-1]], [sy[-1]], [sz[-1]], marker="^", s=40, label="End")
ax3.set_xlabel(r"$\sigma_x$ (MPa)")
ax3.set_ylabel(r"$\sigma_y$ (MPa)")
ax3.set_zlabel(r"$\sigma_z$ (MPa)")
ax3.set_title("3D stress trajectory")
ax3.legend()
st.pyplot(fig3)
st.download_button("Download 3D stress-path figure", data=fig_to_bytes(fig3), file_name="paper1_3d_stress_path.png", mime="image/png")

# -----------------------------
# Failure index
# -----------------------------
st.subheader("4. Failure index")
fig4, ax4 = plt.subplots(figsize=(9, 4.5))
ax4.plot(t_us, F, label=r"$F(t)=q/q_f$")
ax4.axhline(1.0, linestyle="--", linewidth=1.2, label="Failure threshold")
ax4.set_xlabel("Time (μs)")
ax4.set_ylabel("Failure index")
ax4.set_title("Transient failure-envelope interaction")
ax4.grid(True, alpha=0.35)
ax4.legend()
st.pyplot(fig4)
st.download_button("Download failure-index figure", data=fig_to_bytes(fig4), file_name="paper1_failure_index.png", mime="image/png")

# -----------------------------
# Data export
# -----------------------------
st.subheader("5. Data export")
st.dataframe(df.head(20), use_container_width=True)
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download computed data as CSV", data=csv, file_name="paper1_wave_superposition_data.csv", mime="text/csv")

st.markdown("""
### Notes for interpretation

- If the three frequencies are equal and the phase angles are similar, the stress path tends to be simpler and may close after one period.
- If the frequencies differ, the p–q trajectory becomes a Lissajous-type path.
- If the phase angles differ, peak mean stress and peak deviatoric stress may occur at different times.
- A failure-index value greater than one indicates transient crossing of the selected failure envelope.
""")
