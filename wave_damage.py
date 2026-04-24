import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Wave–Damage Impact Demo", layout="wide")

st.title("Multidirectional Impact: Stress Path and Wave–Damage Transition")
st.markdown("Interactive demo for Tri-HB / DEM interpretation of wave superposition, delayed interaction, and sequential damage-controlled loading.")

with st.sidebar:
    st.header("Specimen and wave scale")
    L_mm = st.number_input("Specimen side length L (mm)", value=50.0, min_value=1.0)
    c = st.number_input("Wave speed c (m/s)", value=5000.0, min_value=100.0)
    t_end_us = st.number_input("Simulation time (µs)", value=200.0, min_value=10.0)
    npts = st.slider("Number of time points", 500, 5000, 2000, step=500)

    st.header("Initial stresses (MPa)")
    sx0 = st.number_input("σx0", value=30.0)
    sy0 = st.number_input("σy0", value=20.0)
    sz0 = st.number_input("σz0", value=15.0)

    st.header("Dynamic amplitudes (MPa)")
    Ax = st.number_input("Ax", value=6.0)
    Ay = st.number_input("Ay", value=5.0)
    Az = st.number_input("Az", value=4.0)

    st.header("Frequencies (kHz)")
    fx = st.number_input("fx", value=50.0)
    fy = st.number_input("fy", value=70.0)
    fz = st.number_input("fz", value=90.0)

    st.header("Phase angles (degrees)")
    phix = st.number_input("φx", value=0.0)
    phiy = st.number_input("φy", value=45.0)
    phiz = st.number_input("φz", value=90.0)

    st.header("Delay settings")
    delay_y_us = st.number_input("Delay of y-wave (µs)", value=0.0, min_value=0.0)
    delay_z_us = st.number_input("Delay of z-wave (µs)", value=0.0, min_value=0.0)

    st.header("Failure envelope")
    A_env = st.number_input("A in qf = A + Bp^n", value=5.0)
    B_env = st.number_input("B", value=0.75)
    n_env = st.number_input("n", value=0.85)

L = L_mm / 1000
travel_us = L / c * 1e6

t_us = np.linspace(0, t_end_us, npts)
t = t_us * 1e-6

def delayed_sine(A, f_khz, phase_deg, delay_us):
    td = (t_us - delay_us) * 1e-6
    H = (t_us >= delay_us).astype(float)
    return A * np.sin(2*np.pi*f_khz*1000*td + np.deg2rad(phase_deg)) * H

sx = sx0 + delayed_sine(Ax, fx, phix, 0.0)
sy = sy0 + delayed_sine(Ay, fy, phiy, delay_y_us)
sz = sz0 + delayed_sine(Az, fz, phiz, delay_z_us)

p = (sx + sy + sz) / 3
q = np.sqrt(0.5*((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2))
qf = A_env + B_env * np.maximum(p, 0)**n_env
F = q / qf

# Simple cumulative damage index for demonstration
C = 0.015
m = 1.2
Ddot = C * np.maximum(F - 1, 0)**m
D = np.cumsum(Ddot) * (t_us[1] - t_us[0])
D = np.clip(D, 0, 1)

st.subheader("Characteristic time scale")
col1, col2, col3 = st.columns(3)
col1.metric("Wave travel time", f"{travel_us:.2f} µs")
max_delay = max(delay_y_us, delay_z_us)
col2.metric("Maximum delay Δt", f"{max_delay:.2f} µs")
col3.metric("Normalised delay Δt*", f"{max_delay/travel_us:.2f}")

if max_delay/travel_us < 1:
    regime = "Near-synchronous wave superposition"
elif max_delay/travel_us <= 5:
    regime = "Delayed wave interaction / reverberation"
elif max_delay/travel_us <= 10:
    regime = "Stress redistribution / equilibrium-development stage"
else:
    regime = "Sequential disturbance / damage-memory-controlled loading"
st.info(f"Regime interpretation: **{regime}**")

# Figure 1 stress histories
fig1, ax1 = plt.subplots(figsize=(8, 4.5), dpi=140)
ax1.plot(t_us, sx, label="σx")
ax1.plot(t_us, sy, label="σy")
ax1.plot(t_us, sz, label="σz")
ax1.set_xlabel("Time (µs)")
ax1.set_ylabel("Stress (MPa)")
ax1.set_title("Stress histories")
ax1.grid(True, alpha=0.4)
ax1.legend()
st.pyplot(fig1)

colA, colB = st.columns(2)
with colA:
    fig2, ax2 = plt.subplots(figsize=(6, 5), dpi=140)
    sc = ax2.scatter(p, q, c=t_us, s=8)
    p_line = np.linspace(max(0, p.min()*0.9), p.max()*1.1, 200)
    q_line = A_env + B_env * p_line**n_env
    ax2.plot(p_line, q_line, linestyle="--", label="Failure envelope")
    ax2.set_xlabel("p (MPa)")
    ax2.set_ylabel("q (MPa)")
    ax2.set_title("p–q stress path")
    ax2.grid(True, alpha=0.4)
    ax2.legend()
    cbar = fig2.colorbar(sc, ax=ax2)
    cbar.set_label("Time (µs)")
    st.pyplot(fig2)

with colB:
    fig3 = plt.figure(figsize=(6, 5), dpi=140)
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.plot(sx, sy, sz, linewidth=1.5)
    ax3.set_xlabel("σx (MPa)")
    ax3.set_ylabel("σy (MPa)")
    ax3.set_zlabel("σz (MPa)")
    ax3.set_title("3D stress path")
    st.pyplot(fig3)

fig4, ax4 = plt.subplots(figsize=(8, 4), dpi=140)
ax4.plot(t_us, F, label="Failure index F=q/qf")
ax4.plot(t_us, D, label="Cumulative damage index D")
ax4.axhline(1, linestyle="--", linewidth=1)
ax4.set_xlabel("Time (µs)")
ax4.set_ylabel("Index")
ax4.set_title("Failure index and cumulative damage")
ax4.grid(True, alpha=0.4)
ax4.legend()
st.pyplot(fig4)

st.subheader("Exported data")
df = pd.DataFrame({"time_us": t_us, "sigma_x_MPa": sx, "sigma_y_MPa": sy, "sigma_z_MPa": sz, "p_MPa": p, "q_MPa": q, "q_failure_MPa": qf, "failure_index": F, "damage_index": D})
st.dataframe(df.head(20))
st.download_button("Download CSV", df.to_csv(index=False), file_name="wave_damage_demo_output.csv")
