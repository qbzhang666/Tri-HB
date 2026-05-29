"""
Virtual Tri-HB Simulator — Streamlit version
============================================

Streamlit workspace for Triaxial Hopkinson Bar test design and analysis.
Four loading modes:

  1. Gas-Gun uniaxial impact (classical SHPB)
  2. EM uniaxial impact (clean half-sine pulse)
  3. EM asynchronous triaxial impact (stress-path control)
  4. EM symmetric multidirectional impact (wave superposition)

Run with:  streamlit run virtual_tri_hb_streamlit.py

Author: Monash Tri-HB Group (Virtual Tri-HB v3, Streamlit edition)
"""

import io
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# PHYSICAL CONSTANTS — 42CrMo steel bars
# =============================================================================
@dataclass(frozen=True)
class BarProps:
    E: float = 210e9            # Pa, Young's modulus
    rho: float = 7850.0         # kg/m^3
    C0: float = 5172.0          # m/s (= sqrt(E/rho))
    sigma_prop: float = 1000e6  # Pa, proportional limit
    sigma_yield: float = 1080e6 # Pa, yield strength
    Ab_round: float = np.pi * (0.040 / 2) ** 2  # gas-gun bar (Φ40)
    Ab_square: float = 0.050 * 0.050             # square bar (50×50)


BAR = BarProps()


# =============================================================================
# ROCK CONSTITUTIVE MODELS
# =============================================================================
ROCK_PARAMS = {
    "sandstone": dict(E_s=15e9, sigma_c0=80e6, nu=0.20, b_rate=0.18,
                      epsdot_ref=1e-5, k_conf=4.5, eps_peak=0.008, soften=80.0),
    "granite":   dict(E_s=50e9, sigma_c0=180e6, nu=0.25, b_rate=0.12,
                      epsdot_ref=1e-5, k_conf=5.5, eps_peak=0.005, soften=120.0),
    "concrete":  dict(E_s=30e9, sigma_c0=40e6, nu=0.20, b_rate=0.22,
                      epsdot_ref=1e-5, k_conf=4.0, eps_peak=0.006, soften=100.0),
}


def rock_response(strain: float, strain_rate: float, confinement: float,
                  params: Dict) -> Dict:
    """Holmquist–Johnson–Cook-style response with rate hardening and
    confinement-dependent peak strength.

    Parameters
    ----------
    strain : float (axial strain, positive in compression)
    strain_rate : float (1/s)
    confinement : float (Pa, lateral pre-stress)
    params : dict from ROCK_PARAMS

    Returns dict with stress, sigma_peak, modulus.
    """
    rate_ref = params.get("epsdot_ref", 1e-5)
    DIF = 1.0 + params["b_rate"] * np.log10(max(strain_rate, rate_ref) / rate_ref)
    sigma_peak = (params["sigma_c0"] + params["k_conf"] * confinement) * DIF

    if strain <= 0:
        return dict(stress=0.0, modulus=params["E_s"], sigma_peak=sigma_peak)

    if strain <= params["eps_peak"]:
        r = strain / params["eps_peak"]
        stress = sigma_peak * (2 * r - r * r)
        modulus = sigma_peak * 2 * (1 - r) / params["eps_peak"]
        return dict(stress=stress, modulus=modulus, sigma_peak=sigma_peak)

    # Post-peak softening, bounded below by 5% residual
    eps_excess = strain - params["eps_peak"]
    stress = sigma_peak * np.exp(-params["soften"] * eps_excess)
    return dict(stress=max(stress, sigma_peak * 0.05),
                modulus=-params["soften"] * stress, sigma_peak=sigma_peak)


def rock_response_hydrostatic(strain_vol: float, strain_rate: float,
                              params: Dict) -> Dict:
    """Volumetric (cap) response for fully hydrostatic loading.
    Deviator is zero; specimen deforms by pore collapse / cataclasis.
    """
    rate_ref = params.get("epsdot_ref", 1e-5)
    DIF = 1.0 + params["b_rate"] * np.log10(max(strain_rate, rate_ref) / rate_ref)
    p_cap = 4.0 * params["sigma_c0"] * DIF

    if strain_vol <= 0:
        return dict(stress=0.0, sigma_peak=p_cap)

    r = strain_vol / (params["eps_peak"] * 1.5)
    if r < 1.0:
        return dict(stress=p_cap * (1 - np.exp(-3 * r)), sigma_peak=p_cap)
    return dict(stress=p_cap * (1 + 0.3 * (r - 1)), sigma_peak=p_cap)


# =============================================================================
# PULSE GENERATORS
# =============================================================================
def gas_gun_pulse(t: float, velocity: float, striker_length: float = 0.5,
                  bar_E: float = BAR.E, bar_C0: float = BAR.C0) -> float:
    """Trapezoidal gas-gun pulse with 20 μs rise/fall (pulse-shaper effect)."""
    tau = 2.0 * striker_length / bar_C0
    peak = 0.5 * (bar_E / bar_C0) * velocity
    rise = 20e-6
    if t < 0:
        return 0.0
    if t < rise:
        return peak * (t / rise)
    if t < tau - rise:
        return peak
    if t < tau:
        return peak * (tau - t) / rise
    return 0.0


def em_half_sine(t: float, peak_stress: float, duration: float,
                 delay: float = 0.0) -> float:
    """Electromagnetic half-sine pulse with optional time delay."""
    t_shift = t - delay
    if t_shift < 0 or t_shift > duration:
        return 0.0
    return peak_stress * np.sin(np.pi * t_shift / duration)


# =============================================================================
# CORE SIMULATION ENGINE
# =============================================================================
@dataclass
class SimConfig:
    mode: str                  # 'gas-gun' | 'em-uniaxial' | 'em-async' | 'em-symmetric'
    rock_type: str
    velocity: float            # m/s, gas-gun
    peak_stress: float         # Pa, EM modes
    pulse_duration: float      # s, EM modes
    confinement_X: float       # Pa, σ_1 axial pre-stress
    confinement_Y: float       # Pa, σ_2 confining
    confinement_Z: float       # Pa, σ_3 confining
    specimen_size: float       # m, cube edge
    specimen_length: float     # m, loading length
    specimen_area: float       # m^2, loaded cross-section
    material_E: float          # Pa
    material_UCS: float        # Pa
    material_nu: float
    material_density: float    # kg/m^3
    bar_E: float               # Pa
    bar_C0: float              # m/s
    bar_area: float            # m^2
    pulse_delay_Y: float       # s, async mode
    pulse_delay_Z: float       # s, async mode
    symmetric_axes: str        # 'X' | 'XY' | 'XYZ'


@st.cache_data(show_spinner=False)
def simulate(mode: str, rock_type: str, velocity: float, peak_stress: float,
             pulse_duration: float, confinement_X: float, confinement_Y: float,
             confinement_Z: float, specimen_size: float, specimen_length: float,
             specimen_area: float, material_E: float, material_UCS: float,
             material_nu: float, material_density: float,
             bar_E: float, bar_C0: float, bar_area: float,
             pulse_delay_Y: float, pulse_delay_Z: float, symmetric_axes: str) -> Dict:
    """Run the time-domain wave simulation and return all signals.

    All arguments are hashable primitives so Streamlit's @st.cache_data
    correctly invalidates the cache whenever any input changes.
    """
    cfg = SimConfig(
        mode=mode, rock_type=rock_type, velocity=velocity,
        peak_stress=peak_stress, pulse_duration=pulse_duration,
        confinement_X=confinement_X, confinement_Y=confinement_Y,
        confinement_Z=confinement_Z, specimen_size=specimen_size,
        specimen_length=specimen_length, specimen_area=specimen_area,
        material_E=material_E, material_UCS=material_UCS, material_nu=material_nu,
        material_density=material_density,
        bar_E=bar_E, bar_C0=bar_C0, bar_area=bar_area,
        pulse_delay_Y=pulse_delay_Y, pulse_delay_Z=pulse_delay_Z,
        symmetric_axes=symmetric_axes,
    )
    is_gas_gun = cfg.mode == "gas-gun"
    is_symmetric = cfg.mode == "em-symmetric"
    is_async = cfg.mode == "em-async"

    Ab = cfg.bar_area
    As = cfg.specimen_area
    Ls = cfg.specimen_length

    rock_params = dict(ROCK_PARAMS[cfg.rock_type])
    rock_params.update(E_s=cfg.material_E, sigma_c0=cfg.material_UCS, nu=cfg.material_nu)

    # Time grid
    dt = 1e-6
    t_end = 600e-6 if is_gas_gun else 500e-6
    N = int(t_end / dt)
    time = np.arange(N) * dt

    # State vectors
    eps_x = np.zeros(N); eps_y = np.zeros(N); eps_z = np.zeros(N)
    sig_x = np.zeros(N); sig_y = np.zeros(N); sig_z = np.zeros(N)
    rate_x = np.zeros(N); rate_y = np.zeros(N); rate_z = np.zeros(N)

    # Bar gauge signals (strain; positive compression for incident waves).
    # Symmetric XYZ has six incident bars and six reflected bars.  The plot uses
    # a compact 9-trace view: 3 incident pair traces + 6 reflected traces.
    epsI_x_pos = np.zeros(N)
    epsI_x_neg = np.zeros(N)
    epsI_y_pos = np.zeros(N)
    epsI_y_neg = np.zeros(N)
    epsI_z_pos = np.zeros(N)
    epsI_z_neg = np.zeros(N)

    epsR_x_pos = np.zeros(N)
    epsR_x_neg = np.zeros(N)
    epsR_y_pos = np.zeros(N)
    epsR_y_neg = np.zeros(N)
    epsR_z_pos = np.zeros(N)
    epsR_z_neg = np.zeros(N)

    # One-sided SHPB transmission signal.  In symmetric mode there is no separate
    # transmission bar; all six bars are drive/reflection bars.
    epsT_x = np.zeros(N)

    # Backward-compatible dynamic output aliases used elsewhere in the workspace.
    epsY_dyn = np.zeros(N)
    epsZ_dyn = np.zeros(N)

    # Active axes for symmetric mode
    sym_X = is_symmetric and cfg.symmetric_axes in ("X", "XY", "XYZ")
    sym_Y = is_symmetric and cfg.symmetric_axes in ("XY", "XYZ")
    sym_Z = is_symmetric and cfg.symmetric_axes == "XYZ"
    is_full_hydro = is_symmetric and cfg.symmetric_axes == "XYZ"

    # Static pre-stress (held by hydraulic cylinders, not by bars)
    sigX_static = cfg.confinement_X
    sigY_static = cfg.confinement_Y
    sigZ_static = cfg.confinement_Z

    # ---- Main time loop ----
    for i in range(1, N):
        t = time[i]

        # === INCIDENT PULSES ===
        inc_x_pos = inc_x_neg = 0.0
        inc_y_pos = inc_y_neg = 0.0
        inc_z_pos = inc_z_neg = 0.0

        if cfg.mode == "gas-gun":
            inc_x_pos = gas_gun_pulse(t, cfg.velocity, bar_E=cfg.bar_E, bar_C0=cfg.bar_C0)
        elif cfg.mode == "em-uniaxial":
            inc_x_pos = em_half_sine(t, cfg.peak_stress, cfg.pulse_duration)
        elif cfg.mode == "em-async":
            inc_x_pos = em_half_sine(t, cfg.peak_stress, cfg.pulse_duration, 0.0)
            inc_y_pos = em_half_sine(t, cfg.peak_stress, cfg.pulse_duration, cfg.pulse_delay_Y)
            inc_z_pos = em_half_sine(t, cfg.peak_stress, cfg.pulse_duration, cfg.pulse_delay_Z)
        elif cfg.mode == "em-symmetric":
            if sym_X:
                inc_x_pos = inc_x_neg = em_half_sine(t, cfg.peak_stress, cfg.pulse_duration)
            if sym_Y:
                inc_y_pos = inc_y_neg = em_half_sine(t, cfg.peak_stress, cfg.pulse_duration)
            if sym_Z:
                inc_z_pos = inc_z_neg = em_half_sine(t, cfg.peak_stress, cfg.pulse_duration)

        epsI_x_pos[i] = inc_x_pos / cfg.bar_E
        epsI_x_neg[i] = inc_x_neg / cfg.bar_E
        epsI_y_pos[i] = inc_y_pos / cfg.bar_E
        epsI_y_neg[i] = inc_y_neg / cfg.bar_E
        epsI_z_pos[i] = inc_z_pos / cfg.bar_E
        epsI_z_neg[i] = inc_z_neg / cfg.bar_E

        # === SPECIMEN AXIAL (X) RESPONSE ===
        # Lateral confinement strengthens the rock (Mohr-Coulomb).
        # σ₁ is the axial baseline — it does NOT strengthen, it pre-loads.
        lateral_conf = max(cfg.confinement_Y, cfg.confinement_Z)

        if is_full_hydro:
            eps_vol = (eps_x[i-1] + eps_y[i-1] + eps_z[i-1]) / 3.0
            r = rock_response_hydrostatic(eps_vol, abs(rate_x[i-1]) + 1.0, rock_params)
            sigma_dyn = r["stress"]
            sigma_peak_x = r["sigma_peak"]
        else:
            r = rock_response(eps_x[i-1], abs(rate_x[i-1]) + 1.0,
                              lateral_conf, rock_params)
            sigma_dyn = r["stress"]
            sigma_peak_x = r["sigma_peak"]

        sigma_total = sigX_static + sigma_dyn

        # Wave analysis equations operate on the dynamic component only.
        # The simulator keeps the existing compression-positive bar-output
        # convention: the reflected arrays are per-bar dynamic reflected/output
        # amplitudes, not the hydraulic static preload.
        sig_x_dyn = sigma_dyn
        if sym_X:
            epsT_x[i] = 0.0  # no transmission bar in symmetric mode
            refl_x_pos = max(0.0, inc_x_pos - sig_x_dyn * As / (2.0 * Ab))
            refl_x_neg = max(0.0, inc_x_neg - sig_x_dyn * As / (2.0 * Ab))
            epsR_x_pos[i] = refl_x_pos / cfg.bar_E
            epsR_x_neg[i] = refl_x_neg / cfg.bar_E
        else:
            epsT_x[i] = sigma_dyn * As / (cfg.bar_E * Ab)
            epsR_x_pos[i] = epsT_x[i] - epsI_x_pos[i]
            epsR_x_neg[i] = 0.0

        sig_x[i] = sigma_total

        # === STRAIN RATE (clamped to non-negative to prevent rewind) ===
        if sym_X:
            raw_rate = (2.0 * cfg.bar_C0 / Ls) * epsI_x_pos[i]
        else:
            raw_rate = (cfg.bar_C0 / Ls) * (epsI_x_pos[i] - epsR_x_pos[i] - epsT_x[i])

        rate_x[i] = max(0.0, raw_rate)
        eps_x[i] = eps_x[i-1] + rate_x[i] * dt

        # === LATERAL Y, Z RESPONSE ===
        damage_ratio = sigma_dyn / sigma_peak_x if sigma_peak_x > 0 else 0.0
        dilation = 1.0 + 3.0 * (damage_ratio - 0.6) if damage_ratio > 0.6 else 1.0
        d_eps_x = rate_x[i] * dt
        passive_lat = -rock_params["nu"] * d_eps_x * dilation

        lat_stiffness = rock_params["E_s"] / (1 - rock_params["nu"])

        if is_symmetric:
            if sym_Y:
                drive_Y = inc_y_pos + inc_y_neg
                eps_y[i] = eps_y[i-1] + (2 * cfg.bar_C0 / Ls) * (epsI_y_pos[i] - 0.3 * epsI_y_pos[i]) * dt + passive_lat * 0.3
                sig_y[i] = sigma_total if is_full_hydro else sigY_static + drive_Y * 0.85
            else:
                eps_y[i] = eps_y[i-1] + passive_lat
                sig_y[i] = sigY_static + lat_stiffness * (-eps_y[i])
            if sym_Z:
                drive_Z = inc_z_pos + inc_z_neg
                eps_z[i] = eps_z[i-1] + (2 * cfg.bar_C0 / Ls) * (epsI_z_pos[i] - 0.3 * epsI_z_pos[i]) * dt + passive_lat * 0.3
                sig_z[i] = sigma_total if is_full_hydro else sigZ_static + drive_Z * 0.85
            else:
                eps_z[i] = eps_z[i-1] + passive_lat
                sig_z[i] = sigZ_static + lat_stiffness * (-eps_z[i])
        elif is_async:
            eps_y[i] = eps_y[i-1] + passive_lat
            eps_z[i] = eps_z[i-1] + passive_lat
            sig_y[i] = sigY_static + lat_stiffness * (-eps_y[i]) + inc_y_pos
            sig_z[i] = sigZ_static + lat_stiffness * (-eps_z[i]) + inc_z_pos
        else:
            eps_y[i] = eps_y[i-1] + passive_lat
            eps_z[i] = eps_z[i-1] + passive_lat
            sig_y[i] = sigY_static + lat_stiffness * (-eps_y[i])
            sig_z[i] = sigZ_static + lat_stiffness * (-eps_z[i])

        # Dynamic component on lateral bar gauges.  Static pre-stress is
        # carried by the hydraulic loading frame, so bar gauges show wave-only
        # increments.  In symmetric mode, each active axis has a + and - bar;
        # both reflected components are retained explicitly.
        sig_y_dyn = sig_y[i] - sigY_static
        sig_z_dyn = sig_z[i] - sigZ_static

        Y_share = 2.0 if (is_symmetric and sym_Y) else 1.0
        Z_share = 2.0 if (is_symmetric and sym_Z) else 1.0
        epsY_dyn[i] = sig_y_dyn * As / (Y_share * cfg.bar_E * Ab)
        epsZ_dyn[i] = sig_z_dyn * As / (Z_share * cfg.bar_E * Ab)

        if sym_Y:
            refl_y_pos = max(0.0, inc_y_pos - sig_y_dyn * As / (2.0 * Ab))
            refl_y_neg = max(0.0, inc_y_neg - sig_y_dyn * As / (2.0 * Ab))
            epsR_y_pos[i] = refl_y_pos / cfg.bar_E
            epsR_y_neg[i] = refl_y_neg / cfg.bar_E
            rate_y[i] = max(0.0, (cfg.bar_C0 / Ls) * (
                (epsI_y_pos[i] - epsR_y_pos[i]) +
                (epsI_y_neg[i] - epsR_y_neg[i])
            ))
        else:
            # Passive lateral output or one-sided async Y pulse.
            epsR_y_pos[i] = epsY_dyn[i] if abs(inc_y_pos) < 1e-12 else (sig_y_dyn * As / (cfg.bar_E * Ab) - epsI_y_pos[i])
            epsR_y_neg[i] = 0.0
            rate_y[i] = max(0.0, (cfg.bar_C0 / Ls) * (epsI_y_pos[i] - epsR_y_pos[i]))

        if sym_Z:
            refl_z_pos = max(0.0, inc_z_pos - sig_z_dyn * As / (2.0 * Ab))
            refl_z_neg = max(0.0, inc_z_neg - sig_z_dyn * As / (2.0 * Ab))
            epsR_z_pos[i] = refl_z_pos / cfg.bar_E
            epsR_z_neg[i] = refl_z_neg / cfg.bar_E
            rate_z[i] = max(0.0, (cfg.bar_C0 / Ls) * (
                (epsI_z_pos[i] - epsR_z_pos[i]) +
                (epsI_z_neg[i] - epsR_z_neg[i])
            ))
        else:
            # Passive lateral output or one-sided async Z pulse.
            epsR_z_pos[i] = epsZ_dyn[i] if abs(inc_z_pos) < 1e-12 else (sig_z_dyn * As / (cfg.bar_E * Ab) - epsI_z_pos[i])
            epsR_z_neg[i] = 0.0
            rate_z[i] = max(0.0, (cfg.bar_C0 / Ls) * (epsI_z_pos[i] - epsR_z_pos[i]))

    # ---- Bar plasticity check ----
    # A strain gauge on a bar records ONE signal at any instant — incident and
    # reflected waves pass through the gauge at well-separated times. So the
    # peak bar stress on each bar is just the max |signal| at that bar, not
    # the sum of |incident| + |reflected|.
    #
    # In symmetric mode the +X and -X bars each carry their own incident plus
    # reflected; the same gauge would see whichever is larger.
    bar_signal_arrays = [
        epsI_x_pos, epsI_x_neg, epsR_x_pos, epsR_x_neg, epsT_x,
        epsI_y_pos, epsI_y_neg, epsR_y_pos, epsR_y_neg,
        epsI_z_pos, epsI_z_neg, epsR_z_pos, epsR_z_neg,
    ]
    bar_signals = [np.abs(arr * cfg.bar_E) for arr in bar_signal_arrays]
    max_bar_stress = float(max(np.max(s) for s in bar_signals))

    warning = None
    if max_bar_stress > BAR.sigma_yield:
        warning = dict(level="critical", stress=max_bar_stress,
                       message=f"BAR YIELDING: {max_bar_stress/1e6:.0f} MPa exceeds yield strength {BAR.sigma_yield/1e6:.0f} MPa. Bars deform plastically — invalid test.")
    elif max_bar_stress > BAR.sigma_prop:
        warning = dict(level="warning", stress=max_bar_stress,
                       message=f"NEAR LIMIT: {max_bar_stress/1e6:.0f} MPa exceeds proportional limit {BAR.sigma_prop/1e6:.0f} MPa. Wave analysis becomes nonlinear.")
    elif max_bar_stress > 0.85 * BAR.sigma_prop:
        warning = dict(level="caution", stress=max_bar_stress,
                       message=f"Approaching limit: {max_bar_stress/1e6:.0f} MPa = {100*max_bar_stress/BAR.sigma_prop:.0f}% of proportional limit.")

    # ---- Build output ----
    pressure = (sig_x + sig_y + sig_z) / 3.0
    eps_vol = eps_x + eps_y + eps_z

    # Average strain rate during the loading portion
    loading_mask = rate_x > 0
    avg_rate = float(np.mean(rate_x[loading_mask])) if loading_mask.any() else 0.0

    summary = dict(
        peak_incident_MPa=float(np.max(epsI_x_pos * cfg.bar_E)) / 1e6,
        peak_specimen_stress_MPa=float(np.max(sig_x)) / 1e6,
        peak_bar_stress_MPa=max_bar_stress / 1e6,
        peak_strain_pct=float(np.max(eps_x)) * 100,
        peak_pressure_MPa=float(np.max(pressure)) / 1e6,
        avg_strain_rate=avg_rate,
        pulse_duration_us=192.0 if is_gas_gun else cfg.pulse_duration * 1e6,
        is_full_hydrostatic=is_full_hydro,
    )

    return dict(
        time=time, eps_x=eps_x, eps_y=eps_y, eps_z=eps_z,
        sig_x=sig_x, sig_y=sig_y, sig_z=sig_z,
        pressure=pressure, eps_vol=eps_vol,
        rate_x=rate_x, rate_y=rate_y, rate_z=rate_z,
        epsI_x_pos=epsI_x_pos, epsI_x_neg=epsI_x_neg,
        epsI_y_pos=epsI_y_pos, epsI_y_neg=epsI_y_neg,
        epsI_z_pos=epsI_z_pos, epsI_z_neg=epsI_z_neg,
        epsR_x_pos=epsR_x_pos, epsR_x_neg=epsR_x_neg,
        epsR_y_pos=epsR_y_pos, epsR_y_neg=epsR_y_neg,
        epsR_z_pos=epsR_z_pos, epsR_z_neg=epsR_z_neg,
        epsT_x=epsT_x,
        epsY_dyn=epsY_dyn, epsZ_dyn=epsZ_dyn,
        warning=warning, summary=summary,
        config=dict(
            mode=mode, rock_type=rock_type, velocity=velocity,
            peak_stress=peak_stress, pulse_duration=pulse_duration,
            confinement_X=confinement_X, confinement_Y=confinement_Y,
            confinement_Z=confinement_Z, specimen_size=specimen_size,
            specimen_length=specimen_length, specimen_area=specimen_area,
            material_E=material_E, material_UCS=material_UCS, material_nu=material_nu,
            material_density=material_density,
            bar_E=bar_E, bar_C0=bar_C0, bar_area=bar_area,
            pulse_delay_Y=pulse_delay_Y, pulse_delay_Z=pulse_delay_Z,
            symmetric_axes=symmetric_axes,
        ),
    )


# =============================================================================
# DATA EXPORT HELPERS
# =============================================================================
def build_signals_csv(result: Dict) -> bytes:
    """All time-series signals as a single CSV."""
    df = pd.DataFrame({
        "time_us":             result["time"] * 1e6,

        # Full six-bar incident/reflected set for symmetric multi-axis mode.
        "eps_I_x_pos_uS":      result["epsI_x_pos"] * 1e6,
        "eps_I_x_neg_uS":      result["epsI_x_neg"] * 1e6,
        "eps_I_y_pos_uS":      result["epsI_y_pos"] * 1e6,
        "eps_I_y_neg_uS":      result["epsI_y_neg"] * 1e6,
        "eps_I_z_pos_uS":      result["epsI_z_pos"] * 1e6,
        "eps_I_z_neg_uS":      result["epsI_z_neg"] * 1e6,
        "eps_R_x_pos_uS":      result["epsR_x_pos"] * 1e6,
        "eps_R_x_neg_uS":      result["epsR_x_neg"] * 1e6,
        "eps_R_y_pos_uS":      result["epsR_y_pos"] * 1e6,
        "eps_R_y_neg_uS":      result["epsR_y_neg"] * 1e6,
        "eps_R_z_pos_uS":      result["epsR_z_pos"] * 1e6,
        "eps_R_z_neg_uS":      result["epsR_z_neg"] * 1e6,
        "eps_T_x_uS":          result["epsT_x"] * 1e6,

        # Legacy aliases kept so existing notebooks/workflows still open.
        "eps_I_pos_uS":        result["epsI_x_pos"] * 1e6,
        "eps_I_neg_uS":        result["epsI_x_neg"] * 1e6,
        "eps_R_uS":            result["epsR_x_pos"] * 1e6,
        "eps_T_uS":            result["epsT_x"] * 1e6,
        "eps_Y_uS":            result["epsY_dyn"] * 1e6,
        "eps_Z_uS":            result["epsZ_dyn"] * 1e6,

        "sigma_X_MPa":         result["sig_x"] / 1e6,
        "sigma_Y_MPa":         result["sig_y"] / 1e6,
        "sigma_Z_MPa":         result["sig_z"] / 1e6,
        "pressure_MPa":        result["pressure"] / 1e6,
        "strain_X_pct":        result["eps_x"] * 100,
        "strain_Y_pct":        result["eps_y"] * 100,
        "strain_Z_pct":        result["eps_z"] * 100,
        "strain_vol_pct":      result["eps_vol"] * 100,
        "strain_rate_X":       result["rate_x"],
        "strain_rate_Y":       result["rate_y"],
        "strain_rate_Z":       result["rate_z"],
    })
    return df.to_csv(index=False).encode("utf-8")


def build_stress_strain_csv(result: Dict) -> bytes:
    """σ-ε loading branch only (monotonically increasing strain)."""
    eps = result["eps_x"]
    sig = result["sig_x"] / 1e6
    rows = []
    last = -1.0
    plateau = 0
    for i in range(len(eps)):
        e = eps[i]
        if e <= 0:
            continue
        if e > last + 1e-7:
            rows.append((e * 100, sig[i],
                         result["pressure"][i] / 1e6,
                         result["eps_vol"][i] * 100))
            last = e
            plateau = 0
        else:
            plateau += 1
            if plateau > 5:
                break
    df = pd.DataFrame(rows, columns=[
        "axial_strain_pct", "axial_stress_MPa", "pressure_MPa", "volumetric_strain_pct"
    ])
    return df.to_csv(index=False).encode("utf-8")


def build_summary_json(result: Dict) -> bytes:
    """Configuration + summary metrics + warning, all in one JSON."""
    payload = {
        "metadata": {
            "tool": "Virtual Tri-HB Streamlit",
            "version": "v3",
            "exported_at": datetime.utcnow().isoformat() + "Z",
        },
        "configuration": result["config"],
        "summary": result["summary"],
        "bar_properties": asdict(BAR),
        "rock_parameters": ROCK_PARAMS[result["config"]["rock_type"]],
        "warning": result["warning"],
    }
    return json.dumps(payload, indent=2, default=float).encode("utf-8")


def build_combined_xlsx(result: Dict) -> bytes:
    """Multi-sheet Excel workbook: signals, σ-ε curve, configuration."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Sheet 1: time-series signals
        signals_df = pd.read_csv(io.BytesIO(build_signals_csv(result)))
        signals_df.to_excel(writer, sheet_name="Signals", index=False)
        # Sheet 2: σ-ε curve (loading branch)
        ss_df = pd.read_csv(io.BytesIO(build_stress_strain_csv(result)))
        ss_df.to_excel(writer, sheet_name="Stress-Strain", index=False)
        # Sheet 3: configuration + summary
        cfg_rows = [("parameter", "value", "unit")]
        cfg = result["config"]
        cfg_rows.extend([
            ("mode", cfg["mode"], "-"),
            ("rock_type", cfg["rock_type"], "-"),
            ("velocity", cfg["velocity"], "m/s"),
            ("peak_stress", cfg["peak_stress"] / 1e6, "MPa"),
            ("pulse_duration", cfg["pulse_duration"] * 1e6, "us"),
            ("confinement_X (sigma_1)", cfg["confinement_X"] / 1e6, "MPa"),
            ("confinement_Y (sigma_2)", cfg["confinement_Y"] / 1e6, "MPa"),
            ("confinement_Z (sigma_3)", cfg["confinement_Z"] / 1e6, "MPa"),
            ("specimen_size", cfg["specimen_size"] * 1000, "mm"),
            ("pulse_delay_Y", cfg["pulse_delay_Y"] * 1e6, "us"),
            ("pulse_delay_Z", cfg["pulse_delay_Z"] * 1e6, "us"),
            ("symmetric_axes", cfg["symmetric_axes"], "-"),
        ])
        cfg_rows.append(("", "", ""))
        cfg_rows.append(("--- Summary ---", "", ""))
        for k, v in result["summary"].items():
            unit = ""
            if "MPa" in k: unit = "MPa"
            elif "pct" in k: unit = "%"
            elif "us" in k: unit = "us"
            elif "rate" in k: unit = "1/s"
            cfg_rows.append((k, v, unit))
        cfg_df = pd.DataFrame(cfg_rows[1:], columns=cfg_rows[0])
        cfg_df.to_excel(writer, sheet_name="Configuration", index=False)

    buf.seek(0)
    return buf.read()


# =============================================================================
# STREAMLIT UI
# =============================================================================
st.set_page_config(
    page_title="Virtual Tri-HB",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---- Custom CSS for cleaner appearance ----
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff 0%, #ff6b9d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6a7287;
        font-size: 0.95rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
    }
    .metric-good { color: #06d6a0 !important; }
    .metric-warn { color: #ffa032 !important; }
    .metric-crit { color: #ff3c50 !important; }
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }
    div[data-testid="stMetricLabel"] { text-transform: uppercase; letter-spacing: 1px; font-size: 0.75rem; }
    .stSlider > div[data-baseweb="slider"] { padding: 0 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ---- Header ----
st.markdown('<div class="main-header">Virtual Tri-HB</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Triaxial Hopkinson Bar Simulator · Monash · v3 (Streamlit)</div>',
            unsafe_allow_html=True)
st.markdown(
    "Four loading modes: gas-gun uniaxial, EM uniaxial, EM asynchronous triaxial, "
    "and EM symmetric multidirectional impact. Configure the test, run the "
    "simulation, and export results for further analysis."
)


# ---- Sidebar controls ----
with st.sidebar:
    exp_tab, test_tab = st.tabs(["Experimental setup", "Test loading"])

    exp_tab.header("Specimen material")
    rock_type = exp_tab.selectbox(
        "Material preset",
        ["sandstone", "granite", "concrete"],
        format_func=lambda r: {
            "sandstone": "Sandstone",
            "granite":   "Granite",
            "concrete":  "Concrete",
        }[r],
    )
    preset = ROCK_PARAMS[rock_type]
    material_E_GPa = exp_tab.number_input(
        "Young's modulus, E (GPa)",
        value=float(preset["E_s"] / 1e9),
        min_value=1.0,
        step=1.0,
    )
    material_UCS_MPa = exp_tab.number_input(
        "Uniaxial compressive strength, UCS (MPa)",
        value=float(preset["sigma_c0"] / 1e6),
        min_value=1.0,
        step=5.0,
    )
    material_nu = exp_tab.number_input(
        "Poisson's ratio, ν",
        value=float(preset["nu"]),
        min_value=0.0,
        max_value=0.49,
        step=0.01,
    )
    material_density = exp_tab.number_input(
        "Density, ρ (kg/m³)",
        value=2650.0,
        min_value=1000.0,
        step=50.0,
    )

    exp_tab.header("Specimen geometry")
    specimen_shape = exp_tab.radio("Cross-section", ["Square/cube", "Circular cylinder"], horizontal=True)
    specimen_side_mm = exp_tab.number_input("Side length / diameter (mm)", value=50.0, min_value=1.0, step=1.0)
    specimen_length_mm = exp_tab.number_input("Loading length, Ls (mm)", value=50.0, min_value=1.0, step=1.0)
    if specimen_shape == "Square/cube":
        specimen_area_m2 = (specimen_side_mm * 1e-3) ** 2
    else:
        specimen_area_m2 = np.pi * (specimen_side_mm * 1e-3 / 2.0) ** 2
    exp_tab.caption(f"Specimen area: {specimen_area_m2 * 1e6:.1f} mm²")

    exp_tab.header("Bar properties")
    bar_E_GPa = exp_tab.number_input("Bar Young's modulus, Eb (GPa)", value=210.0, min_value=1.0, step=5.0)
    bar_C0 = exp_tab.number_input("Bar wave speed, C0 (m/s)", value=5172.0, min_value=1000.0, step=50.0)
    bar_section = exp_tab.radio("Bar cross-section", ["Square 50 x 50 mm", "Round"], horizontal=True)
    if bar_section == "Square 50 x 50 mm":
        bar_area_m2 = 0.050 * 0.050
        exp_tab.caption("Bar area: 2500.0 mm²")
    else:
        bar_diameter_mm = exp_tab.number_input("Round bar diameter (mm)", value=50.0, min_value=1.0, step=1.0)
        bar_area_m2 = np.pi * (bar_diameter_mm * 1e-3 / 2.0) ** 2
        exp_tab.caption(f"Bar area: {bar_area_m2 * 1e6:.1f} mm²")

    test_tab.header("Test Configuration")

    mode_label_map = {
        "Gas-Gun Uniaxial": "gas-gun",
        "EM Uniaxial": "em-uniaxial",
        "EM Async Triaxial": "em-async",
        "EM Symmetric Multi-axis": "em-symmetric",
    }
    mode_label = test_tab.selectbox("Loading mode", list(mode_label_map.keys()))
    mode = mode_label_map[mode_label]
    is_em = mode != "gas-gun"
    is_symmetric = mode == "em-symmetric"
    is_async = mode == "em-async"

    mode_descriptions = {
        "gas-gun": "Single striker → incident bar → specimen → transmission bar (classical SHPB).",
        "em-uniaxial": "Single EM half-sine pulse along +X. Y/Z carry only Poisson reactions.",
        "em-async": "3 EM pulses (+X, +Y, +Z) with adjustable time delays. Stress-path-dependent.",
        "em-symmetric": "6 EM pulses fire SIMULTANEOUSLY on opposing bars. Constructive superposition at specimen → 2× stress per axis.",
    }
    test_tab.caption(mode_descriptions[mode])

    if is_symmetric:
        sym_axes_label = test_tab.radio(
            "Active symmetric axes",
            ["±X only", "±X ±Y", "Full hydrostatic (±X ±Y ±Z)"],
            help="Number of opposing pulse pairs that fire simultaneously.",
        )
        symmetric_axes = {"±X only": "X", "±X ±Y": "XY",
                          "Full hydrostatic (±X ±Y ±Z)": "XYZ"}[sym_axes_label]
    else:
        symmetric_axes = "XYZ"  # placeholder, unused

    test_tab.divider()
    test_tab.subheader("Pulse settings")

    if mode == "gas-gun":
        velocity = test_tab.slider("Striker velocity (m/s)", 5, 50, 20, step=1)
        peak_stress_MPa = 0
        pulse_duration_us = 2.0 * 0.5 / bar_C0 * 1e6
        gas_peak = 0.5 * ((bar_E_GPa * 1e9) / bar_C0) * velocity / 1e6
        test_tab.caption(f"Peak incident stress: {gas_peak:.0f} MPa "
                         f"(set by 0.5·Eb/C₀·V)")
    else:
        velocity = 0
        peak_stress_MPa = test_tab.slider("Single-pulse peak amplitude (MPa)",
                                           50, 900, 400, step=25)
        pulse_duration_us = test_tab.slider("Pulse duration τ (μs)",
                                            50, 400, 200, step=10)
        if is_symmetric:
            test_tab.caption(f"Effective specimen drive (with superposition): "
                             f"{2 * peak_stress_MPa} MPa per active axis.")

    test_tab.divider()
    test_tab.subheader("Static pre-stress")
    test_tab.caption("Applied by hydraulic cylinders before the dynamic pulse arrives.")

    max_conf = 50
    conf_X = test_tab.slider("σ₁ axial (X) — MPa", 0, max_conf, 20, step=1)
    conf_Y = test_tab.slider("σ₂ confining (Y) — MPa", 0, max_conf, 15, step=1)
    conf_Z = test_tab.slider("σ₃ confining (Z) — MPa", 0, max_conf, 10, step=1)

    if is_async:
        test_tab.divider()
        test_tab.subheader("Asynchronous timing")
        delay_Y_us = test_tab.slider("Y-pulse delay (μs)", 0, 200, 0, step=5)
        delay_Z_us = test_tab.slider("Z-pulse delay (μs)", 0, 200, 0, step=5)
    else:
        delay_Y_us = 0
        delay_Z_us = 0


# ---- Build config and run simulation ----
config = {
    "mode": mode,
    "rock_type": rock_type,
    "velocity": float(velocity),
    "peak_stress": float(peak_stress_MPa) * 1e6,
    "pulse_duration": float(pulse_duration_us) * 1e-6,
    "confinement_X": float(conf_X) * 1e6,
    "confinement_Y": float(conf_Y) * 1e6,
    "confinement_Z": float(conf_Z) * 1e6,
    "specimen_size": float(specimen_side_mm) * 1e-3,
    "specimen_length": float(specimen_length_mm) * 1e-3,
    "specimen_area": float(specimen_area_m2),
    "material_E": float(material_E_GPa) * 1e9,
    "material_UCS": float(material_UCS_MPa) * 1e6,
    "material_nu": float(material_nu),
    "material_density": float(material_density),
    "bar_E": float(bar_E_GPa) * 1e9,
    "bar_C0": float(bar_C0),
    "bar_area": float(bar_area_m2),
    "pulse_delay_Y": float(delay_Y_us) * 1e-6,
    "pulse_delay_Z": float(delay_Z_us) * 1e-6,
    "symmetric_axes": symmetric_axes,
}

result = simulate(**config)
st.session_state["tri_hb_latest_result"] = result
st.session_state["tri_hb_latest_config"] = config


# =============================================================================
# WARNINGS BANNER
# =============================================================================
if result["warning"]:
    w = result["warning"]
    if w["level"] == "critical":
        st.error(f"⚠ **CRITICAL — Bar Yielding:** {w['message']}")
    elif w["level"] == "warning":
        st.warning(f"⚠ **WARNING — Near Limit:** {w['message']}")
    else:
        st.info(f"⚠ **Caution:** {w['message']}")


# =============================================================================
# METRIC CARDS
# =============================================================================
s = result["summary"]
col1, col2, col3, col4, col5 = st.columns(5)

drive_label = "Effective drive" if is_symmetric else "Peak incident"
drive_value = 2 * peak_stress_MPa if is_symmetric else s["peak_incident_MPa"]
col1.metric(drive_label, f"{drive_value:.0f} MPa")
col2.metric("Specimen σ peak", f"{s['peak_specimen_stress_MPa']:.1f} MPa")
col3.metric("Max bar stress", f"{s['peak_bar_stress_MPa']:.1f} MPa",
            delta=f"limit {BAR.sigma_prop/1e6:.0f}",
            delta_color="inverse")
if is_symmetric and s["is_full_hydrostatic"]:
    col4.metric("Peak pressure", f"{s['peak_pressure_MPa']:.1f} MPa")
else:
    col4.metric("Strain rate", f"{s['avg_strain_rate']:.0f} /s")
col5.metric("Peak strain", f"{s['peak_strain_pct']:.3f} %")


# =============================================================================
# TABBED PLOTS
# =============================================================================
tab_labels = ["Bar Waveforms", "Specimen Stresses", "σ–ε Curve"]
if is_symmetric:
    tab_labels.append("p–εᵥ (volumetric)")
tab_labels.append("Equations")

tabs = st.tabs(tab_labels)


# ---- Common plot styling ----
def base_layout(xtitle: str, ytitle: str, height: int = 420) -> dict:
    return dict(
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c0c8d8", family="JetBrains Mono, monospace", size=12),
        xaxis=dict(gridcolor="rgba(80,90,110,0.3)"),
        yaxis=dict(gridcolor="rgba(80,90,110,0.3)"),
        margin=dict(l=60, r=20, t=20, b=50),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2a3348", borderwidth=1),
    )


# ---- Tab 1: Bar Waveforms ----
with tabs[0]:
    fig = go.Figure()
    t_us = result["time"] * 1e6

    if is_symmetric:
        st.markdown(
            "**Symmetric compact view:** 3 incident pair waveforms "
            "(+/− bars are identical within each active axis) plus the 6 "
            "individual reflected bar waveforms. Full XYZ therefore shows 9 traces."
        )

        active_axes = []
        if symmetric_axes in ("X", "XY", "XYZ"):
            active_axes.append(("x", "X", "#00d4ff"))
        if symmetric_axes in ("XY", "XYZ"):
            active_axes.append(("y", "Y", "#06d6a0"))
        if symmetric_axes == "XYZ":
            active_axes.append(("z", "Z", "#a78bfa"))

        for axis_key, axis_label, color in active_axes:
            fig.add_trace(go.Scatter(
                x=t_us,
                y=result[f"epsI_{axis_key}_pos"] * 1e6,
                name=f"ε_I (±{axis_label} pair)",
                line=dict(color=color, width=2),
            ))

        reflected_styles = {
            "x": ("#ffd166", "dash"),
            "y": ("#ff6b9d", "dot"),
            "z": ("#c084fc", "dashdot"),
        }
        for axis_key, axis_label, _ in active_axes:
            color, dash = reflected_styles[axis_key]
            fig.add_trace(go.Scatter(
                x=t_us,
                y=result[f"epsR_{axis_key}_pos"] * 1e6,
                name=f"ε_R (+{axis_label} bar)",
                line=dict(color=color, width=1.5, dash=dash),
            ))
            fig.add_trace(go.Scatter(
                x=t_us,
                y=result[f"epsR_{axis_key}_neg"] * 1e6,
                name=f"ε_R (−{axis_label} bar)",
                line=dict(color=color, width=1.5, dash="longdash"),
            ))
    else:
        fig.add_trace(go.Scatter(x=t_us, y=result["epsI_x_pos"] * 1e6,
                                 name="ε_I (+X incident)", line=dict(color="#00d4ff", width=2)))
        fig.add_trace(go.Scatter(x=t_us, y=result["epsR_x_pos"] * 1e6,
                                 name="ε_R (+X reflected)",
                                 line=dict(color="#ffd166", width=1.5)))
        fig.add_trace(go.Scatter(x=t_us, y=result["epsT_x"] * 1e6,
                                 name="ε_T (X transmitted)",
                                 line=dict(color="#06d6a0", width=2)))
        if np.any(np.abs(result["epsI_y_pos"]) > 0.0):
            fig.add_trace(go.Scatter(x=t_us, y=result["epsI_y_pos"] * 1e6,
                                     name="ε_I (+Y incident)",
                                     line=dict(color="#06d6a0", width=1.5, dash="dot")))
            fig.add_trace(go.Scatter(x=t_us, y=result["epsR_y_pos"] * 1e6,
                                     name="ε_R (+Y reflected/output)",
                                     line=dict(color="#ff6b9d", width=1.5, dash="dot")))
        if np.any(np.abs(result["epsI_z_pos"]) > 0.0):
            fig.add_trace(go.Scatter(x=t_us, y=result["epsI_z_pos"] * 1e6,
                                     name="ε_I (+Z incident)",
                                     line=dict(color="#a78bfa", width=1.5, dash="dot")))
            fig.add_trace(go.Scatter(x=t_us, y=result["epsR_z_pos"] * 1e6,
                                     name="ε_R (+Z reflected/output)",
                                     line=dict(color="#c084fc", width=1.5, dash="dot")))

    fig.update_layout(**base_layout("Time (μs)", "Strain (μstrain)"))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Bar gauge signals only show the **dynamic** wave component. Static "
        "pre-stress is held by the hydraulic cylinders and is added separately "
        "to the specimen stress. In symmetric mode the ± incident traces for "
        "each axis overlap, so one incident pair trace is shown per active axis."
    )


# ---- Tab 2: Specimen Stresses ----
with tabs[1]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_us, y=result["sig_x"] / 1e6,
                             name="σ_X (total)", line=dict(color="#00d4ff", width=2)))
    fig.add_trace(go.Scatter(x=t_us, y=result["sig_y"] / 1e6,
                             name="σ_Y (total)", line=dict(color="#06d6a0", width=2)))
    fig.add_trace(go.Scatter(x=t_us, y=result["sig_z"] / 1e6,
                             name="σ_Z (total)", line=dict(color="#a78bfa", width=2)))
    if is_symmetric:
        fig.add_trace(go.Scatter(x=t_us, y=result["pressure"] / 1e6,
                                 name="p (mean)",
                                 line=dict(color="#ffd166", width=2.5, dash="dash")))
    # Bar-limit reference line
    fig.add_hline(y=BAR.sigma_prop / 1e6, line_dash="dash", line_color="#ff6b9d",
                  annotation_text=f"bar limit ({BAR.sigma_prop / 1e6:.0f} MPa)",
                  annotation_position="top right",
                  annotation_font=dict(color="#ff6b9d", size=10))
    fig.update_layout(**base_layout("Time (μs)", "Stress (MPa)"))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"σ_X = σ₁(static) + σ_X^dyn  (σ₁ = {conf_X} MPa). "
        f"σ_Y = σ₂(static) + σ_Y^dyn  (σ₂ = {conf_Y} MPa). "
        f"σ_Z = σ₃(static) + σ_Z^dyn  (σ₃ = {conf_Z} MPa)."
    )


# ---- Tab 3: σ–ε Curve (loading branch only) ----
with tabs[2]:
    eps = result["eps_x"]
    sig = result["sig_x"] / 1e6
    pts = []
    last = -1.0
    plateau = 0
    for i in range(len(eps)):
        e = eps[i]
        if e <= 0:
            continue
        if e > last + 1e-7:
            pts.append((e * 100, sig[i]))
            last = e
            plateau = 0
        else:
            plateau += 1
            if plateau > 5:
                break

    if pts:
        ss_x, ss_y = zip(*pts)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ss_x, y=ss_y, mode="lines",
            line=dict(color="#ff6b9d" if is_symmetric else "#00d4ff", width=3),
            name="σ–ε",
        ))
        fig.update_layout(**base_layout("Axial Strain (%)", "Axial Stress (MPa)"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No measurable strain accumulated (specimen unloaded). Increase pulse amplitude or duration.")

    st.caption(
        "Loading branch only — the curve traces hardening up to peak strength "
        "and post-peak softening, then stops when strain plateaus. Elastic "
        "unloading after the pulse is omitted to keep the constitutive curve clean."
    )


# ---- Tab 4: Pressure-volumetric (symmetric mode only) ----
if is_symmetric:
    with tabs[3]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result["eps_vol"] * 100, y=result["pressure"] / 1e6,
            mode="lines", line=dict(color="#ffd166", width=3),
            name="p–εᵥ",
        ))
        fig.update_layout(**base_layout("Volumetric Strain (%)", "Mean Pressure (MPa)"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Compaction curve. Initial slope = bulk modulus K. Plateau = cap pressure. "
            "Under full hydrostatic loading, the deviatoric stress q ≈ 0, so the "
            "specimen cannot fail by shear — it deforms only volumetrically."
        )


# ---- Equations tab ----
with tabs[-1]:
    st.subheader("Wave Analysis Equations")

    st.markdown("**Standard one-sided drive** (gas-gun, EM uniaxial, EM async):")
    st.latex(r"\sigma_i^{\rm dyn}(t) = \frac{E_b A_b}{2 A_s}\left[\varepsilon_{I,i}+\varepsilon_{R,i}+\varepsilon_{T,i}\right],\qquad i\in\{X,Y,Z\}")
    st.latex(r"\dot{\varepsilon}_i(t) = \frac{C_0}{L_i}\left[\varepsilon_{I,i}-\varepsilon_{R,i}-\varepsilon_{T,i}\right]")
    st.caption("For the classical one-sided X test, i = X. For EM async, the same dynamic equation is applied to each driven one-sided axis.")

    st.markdown("**Symmetric superposition** (Mode 4, opposing bars fire identically):")
    st.latex(r"\sigma_i^{\rm dyn}(t) = \frac{E_b A_b}{2 A_s}\left[(\varepsilon_{I,i}^{+}+\varepsilon_{R,i}^{+})+(\varepsilon_{I,i}^{-}+\varepsilon_{R,i}^{-})\right],\qquad i\in\{X,Y,Z\}")
    st.latex(r"\dot{\varepsilon}_i(t) = \frac{C_0}{L_i}\left[(\varepsilon_{I,i}^{+}-\varepsilon_{R,i}^{+})+(\varepsilon_{I,i}^{-}-\varepsilon_{R,i}^{-})\right]")
    st.markdown("When the two opposing pulses in an axis are matched, this reduces to:")
    st.latex(r"\sigma_{i,\rm sym}^{\rm dyn}(t)=\frac{E_b A_b}{A_s}\left[\varepsilon_{I,i}+\varepsilon_{R,i}\right]")
    st.latex(r"\dot{\varepsilon}_{i,\rm sym}(t)=\frac{2C_0}{L_i}\left[\varepsilon_{I,i}-\varepsilon_{R,i}\right]")
    st.markdown(
        "The specimen receives the superposed pair load on every active axis, "
        "while each individual bar carries only its own incident/reflected wave. "
        "For full XYZ symmetric operation this means 6 incident bar waves and 6 reflected bar waves; "
        "the Bar Waveforms tab displays them compactly as 3 incident pair curves + 6 reflected curves."
    )

    st.markdown("**Total stress = static hydraulic preload + dynamic bar-wave increment**:")
    st.latex(r"\sigma_X^{\rm total}(t)=\sigma_1^{\rm static}+\sigma_X^{\rm dyn}(t)")
    st.latex(r"\sigma_Y^{\rm total}(t)=\sigma_2^{\rm static}+\sigma_Y^{\rm dyn}(t)")
    st.latex(r"\sigma_Z^{\rm total}(t)=\sigma_3^{\rm static}+\sigma_Z^{\rm dyn}(t)")
    st.caption("The static term is imposed before the pulse by the hydraulic system. It is not a travelling gauge signal, so the bar-wave plots show only the dynamic increment.")

    st.markdown("**Hydrostatic mode** (full XYZ symmetric):")
    st.latex(r"p(t)=\tfrac{1}{3}(\sigma_X+\sigma_Y+\sigma_Z),\qquad q(t)\approx0\quad\text{when }\sigma_X^{\rm dyn}=\sigma_Y^{\rm dyn}=\sigma_Z^{\rm dyn}")
    st.latex(r"\varepsilon_v(t)=\varepsilon_X+\varepsilon_Y+\varepsilon_Z")

    st.markdown("**Individual-bar dynamic stress and plasticity bound** (always required):")
    st.latex(r"\sigma_{b,i}^{I,\pm}(t)=E_b\varepsilon_{I,i}^{\pm}(t),\qquad \sigma_{b,i}^{R,\pm}(t)=E_b\varepsilon_{R,i}^{\pm}(t)")
    st.latex(
        r"\max_{i\in\{X,Y,Z\},\;\pm}\;\max_t\left(|\sigma_{b,i}^{I,\pm}(t)|,|\sigma_{b,i}^{R,\pm}(t)|\right)"
        + rf"\leq\sigma_{{\rm prop}}={BAR.sigma_prop / 1e6:.0f}\ \text{{MPa}}"
    )

    with st.expander("Constitutive model details"):
        st.latex(r"\sigma_{\text{peak}} = (\sigma_{c0} + k_{\text{conf}} \cdot \sigma_{\text{conf}}) \cdot \text{DIF}")
        st.latex(r"\text{DIF} = 1 + b_{\text{rate}} \log_{10}\!\left(\dot{\varepsilon}/\dot{\varepsilon}_{\text{ref}}\right)")
        st.markdown(
            "Pre-peak: $\\sigma = \\sigma_{\\text{peak}} \\cdot (2r - r^2)$, "
            "where $r = \\varepsilon / \\varepsilon_{\\text{peak}}$."
        )
        st.markdown(
            "Post-peak: $\\sigma = \\sigma_{\\text{peak}} \\cdot \\exp(-\\gamma (\\varepsilon - \\varepsilon_{\\text{peak}}))$, "
            "bounded below by $0.05 \\sigma_{\\text{peak}}$ (residual frictional strength)."
        )


# =============================================================================
# DATA EXPORT SECTION
# =============================================================================
st.divider()
st.subheader("📥 Export Data")
st.markdown(
    "Download the simulation inputs and outputs for further analysis in MATLAB, "
    "Python, Excel, or any other tool."
)

# Build filename stem from configuration
filename_stem = (
    f"tri_hb_{mode.replace('-', '_')}_{rock_type}"
    f"_v{int(velocity)}_p{int(peak_stress_MPa)}"
    f"_s{int(conf_X)}-{int(conf_Y)}-{int(conf_Z)}"
)

ec1, ec2, ec3, ec4 = st.columns(4)

with ec1:
    st.download_button(
        label="📊 All Signals (CSV)",
        data=build_signals_csv(result),
        file_name=f"{filename_stem}_signals.csv",
        mime="text/csv",
        help="Time-series of all bar gauge signals, specimen stresses and strains.",
    )

with ec2:
    st.download_button(
        label="📈 σ–ε Curve (CSV)",
        data=build_stress_strain_csv(result),
        file_name=f"{filename_stem}_stress_strain.csv",
        mime="text/csv",
        help="Loading branch of the dynamic stress-strain curve.",
    )

with ec3:
    st.download_button(
        label="⚙️ Configuration (JSON)",
        data=build_summary_json(result),
        file_name=f"{filename_stem}_config.json",
        mime="application/json",
        help="Full input configuration, summary metrics, and bar/rock properties.",
    )

with ec4:
    try:
        xlsx_bytes = build_combined_xlsx(result)
        st.download_button(
            label="📦 Full workbook (XLSX)",
            data=xlsx_bytes,
            file_name=f"{filename_stem}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="All data + configuration in a single multi-sheet workbook.",
        )
    except ImportError:
        st.caption("Install `openpyxl` for Excel export")


# Quick preview of the export contents
with st.expander("📋 Preview export contents"):
    sub_a, sub_b = st.columns(2)
    with sub_a:
        st.markdown("**Configuration & summary (JSON):**")
        st.json(json.loads(build_summary_json(result).decode("utf-8")))
    with sub_b:
        st.markdown("**Signals CSV (first 8 rows):**")
        df = pd.read_csv(io.BytesIO(build_signals_csv(result)))
        st.dataframe(df.head(8), use_container_width=True, height=280)


# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.caption(
    "Virtual Tri-HB Streamlit · v3 · 1-D wave-analysis workspace. "
    "Check dispersion correction, equilibrium verification, pulse shaping, and bar plasticity bounds before interpreting experimental results."
)
