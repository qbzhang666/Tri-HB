"""
Virtual Tri-HB Simulator — Streamlit version
============================================

Streamlit workspace for Triaxial Hopkinson Bar test design and analysis.
Four top-level loading families:

  1. Gas-gun uniaxial SHPB
  2. Confinement-chamber SHPB
  3. Gas-gun Tri-HB static-dynamic loading
  4. Electromagnetic programmable loading

The electromagnetic family branches internally into three calculation
topologies: single-axis one-sided, multi-axis one-sided, and symmetric
opposing pairs.

Run with:  streamlit run virtual_tri_hb_streamlit.py

Author: Monash Tri-HB Group (Virtual Tri-HB v3, Streamlit edition)
"""

import io
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
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


def gauge_time_offsets_us(cfg: Dict) -> Dict[str, float]:
    """Display-time offsets for raw bar-gauge waveforms.

    The simulator stores waves in specimen/reduction time so the equations can
    use the already-windowed incident, reflected and transmitted histories.  The
    measurement plot should instead show acquisition time: incident first, then
    reflected/transmitted after the gauge-to-specimen travel times.
    """
    c0 = max(float(cfg.get("bar_C0", BAR.C0)), 1.0)
    trigger = float(cfg.get("gauge_trigger_offset_us", DEFAULT_GAUGE_TRIGGER_US))
    incident_distance = float(cfg.get("incident_gauge_distance_m", DEFAULT_GAUGE_DISTANCE_M))
    transmission_distance = float(cfg.get("transmission_gauge_distance_m", DEFAULT_GAUGE_DISTANCE_M))
    return {
        "incident": trigger,
        "reflected": trigger + 2.0 * incident_distance / c0 * 1e6,
        "transmitted": trigger + (incident_distance + transmission_distance) / c0 * 1e6,
        "output": trigger + (incident_distance + transmission_distance) / c0 * 1e6,
    }


def waveform_trace_window(t_us: np.ndarray, y: np.ndarray,
                          offset_us: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Return the visible non-zero waveform window, shifted for plotting."""
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
    """Pulse-shaped gas-gun incident pulse.

    In a real gas-gun SHPB test a pulse shaper (a thin soft-metal disc at the
    impact face) is used to smooth the otherwise near-rectangular striker pulse
    into a half-sine shape. This reduces Pochhammer-Chree dispersion and lets the
    specimen reach dynamic stress equilibrium before failure. We therefore model
    the shaped incident pulse as a half-sine of the same peak amplitude and the
    same striker round-trip duration as the unshaped pulse:

        sigma_I(t) = peak * sin(pi * t / tau),   0 <= t <= tau

    Peak amplitude is the elastic-impact value 0.5 * (E_b / C_0) * V and the
    duration is the striker round-trip time tau = 2 L_striker / C_0.
    """
    tau = 2.0 * striker_length / bar_C0
    peak = 0.5 * (bar_E / bar_C0) * velocity
    if t < 0.0 or t > tau:
        return 0.0
    return peak * np.sin(np.pi * t / tau)


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
    mode: str                  # 'gas-gun' | 'confinement-chamber' | 'gas-gun-triaxial' | 'em-uniaxial' | 'em-async' | 'em-symmetric'
    rock_type: str
    velocity: float            # m/s, gas-gun
    peak_stress: float         # Pa, representative EM pulse amplitude
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
    loading_family: str = ""
    em_topology: str = ""
    active_axes: str = "XYZ"   # one-sided EM axes: 'X' | 'XY' | 'XYZ'
    peak_stress_X: float = 0.0 # Pa, EM X-axis pulse amplitude
    peak_stress_Y: float = 0.0 # Pa, EM Y-axis pulse amplitude
    peak_stress_Z: float = 0.0 # Pa, EM Z-axis pulse amplitude


@st.cache_data(show_spinner=False)
def simulate(mode: str, rock_type: str, velocity: float, peak_stress: float,
             pulse_duration: float, confinement_X: float, confinement_Y: float,
             confinement_Z: float, specimen_size: float, specimen_length: float,
             specimen_area: float, material_E: float, material_UCS: float,
             material_nu: float, material_density: float,
             bar_E: float, bar_C0: float, bar_area: float,
             pulse_delay_Y: float, pulse_delay_Z: float, symmetric_axes: str,
             loading_family: str = "", em_topology: str = "",
             active_axes: str = "XYZ",
             peak_stress_X: float | None = None,
             peak_stress_Y: float | None = None,
             peak_stress_Z: float | None = None) -> Dict:
    """Run the time-domain wave simulation and return all signals.

    All arguments are hashable primitives so Streamlit's @st.cache_data
    correctly invalidates the cache whenever any input changes.
    """
    peak_stress_X = peak_stress if peak_stress_X is None else peak_stress_X
    peak_stress_Y = peak_stress if peak_stress_Y is None else peak_stress_Y
    peak_stress_Z = peak_stress if peak_stress_Z is None else peak_stress_Z
    loading_family = loading_family or {
        "gas-gun": "gas_gun_uniaxial",
        "confinement-chamber": "confinement_chamber_shpb",
        "gas-gun-triaxial": "gas_gun_tri_hb_static_dynamic",
        "em-uniaxial": "electromagnetic",
        "em-async": "electromagnetic",
        "em-symmetric": "electromagnetic",
    }.get(mode, mode)
    em_topology = em_topology or {
        "em-uniaxial": "single_axis",
        "em-async": "one_sided_multiaxis",
        "em-symmetric": "symmetric_pairs",
    }.get(mode, "")

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
        loading_family=loading_family, em_topology=em_topology,
        active_axes=active_axes,
        peak_stress_X=peak_stress_X, peak_stress_Y=peak_stress_Y,
        peak_stress_Z=peak_stress_Z,
    )
    is_gas_drive = cfg.mode in ("gas-gun", "confinement-chamber", "gas-gun-triaxial")
    is_confinement_chamber = cfg.mode == "confinement-chamber"
    is_gas_triaxial = cfg.mode == "gas-gun-triaxial"
    is_gas_uniaxial = cfg.mode == "gas-gun"
    is_symmetric = cfg.mode == "em-symmetric"
    is_async = cfg.mode == "em-async"

    Ab = cfg.bar_area
    As = cfg.specimen_area
    Ls = cfg.specimen_length

    rock_params = dict(ROCK_PARAMS[cfg.rock_type])
    rock_params.update(E_s=cfg.material_E, sigma_c0=cfg.material_UCS, nu=cfg.material_nu)

    # Time grid
    dt = 1e-6
    t_end = 600e-6 if is_gas_drive else max(500e-6, cfg.pulse_duration + cfg.pulse_delay_Y + 120e-6, cfg.pulse_duration + cfg.pulse_delay_Z + 120e-6)
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
    # transmission bar; all six bars are drive/reflection bars.  In asynchronous
    # mode the -Y and -Z bars act as transmission bars on their respective axes,
    # so epsT_y and epsT_z carry the per-axis transmitted wave.
    epsT_x = np.zeros(N)
    epsT_y = np.zeros(N)
    epsT_z = np.zeros(N)

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

    # Largest strain reached so far on each driven axis.  Damage is irreversible,
    # so the constitutive backbone (hardening / softening / residual) is a
    # function of this running peak, while the current point unloads elastically
    # below it.  Y/Z peaks are only used by the async per-axis treatment.
    eps_x_max = 0.0
    eps_y_max = 0.0
    eps_z_max = 0.0

    # Smoothed (EMA) loading strain rate used for the rate-hardening DIF.  The
    # raw step-to-step strain rate develops a small sign-alternating ripple at
    # the loading -> unloading transition; feeding that through log10(rate) in
    # the DIF would amplify it into a spurious oscillation of the strength (and
    # hence of the transmitted wave).  Smoothing the rate that drives the DIF
    # removes the feedback while leaving the physical rate hardening intact.
    dif_rate_x = 0.0
    dif_rate_y = 0.0
    dif_rate_z = 0.0

    # ---- Main time loop ----
    for i in range(1, N):
        t = time[i]

        # === INCIDENT PULSES ===
        inc_x_pos = inc_x_neg = 0.0
        inc_y_pos = inc_y_neg = 0.0
        inc_z_pos = inc_z_neg = 0.0

        if is_gas_drive:
            inc_x_pos = gas_gun_pulse(t, cfg.velocity, bar_E=cfg.bar_E, bar_C0=cfg.bar_C0)
        elif cfg.mode == "em-uniaxial":
            inc_x_pos = em_half_sine(t, cfg.peak_stress_X, cfg.pulse_duration)
        elif cfg.mode == "em-async":
            if cfg.active_axes in ("X", "XY", "XYZ"):
                inc_x_pos = em_half_sine(t, cfg.peak_stress_X, cfg.pulse_duration, 0.0)
            if cfg.active_axes in ("XY", "XYZ"):
                inc_y_pos = em_half_sine(t, cfg.peak_stress_Y, cfg.pulse_duration, cfg.pulse_delay_Y)
            if cfg.active_axes == "XYZ":
                inc_z_pos = em_half_sine(t, cfg.peak_stress_Z, cfg.pulse_duration, cfg.pulse_delay_Z)
        elif cfg.mode == "em-symmetric":
            if sym_X:
                inc_x_pos = inc_x_neg = em_half_sine(t, cfg.peak_stress_X, cfg.pulse_duration)
            if sym_Y:
                inc_y_pos = inc_y_neg = em_half_sine(t, cfg.peak_stress_Y, cfg.pulse_duration)
            if sym_Z:
                inc_z_pos = inc_z_neg = em_half_sine(t, cfg.peak_stress_Z, cfg.pulse_duration)

        epsI_x_pos[i] = inc_x_pos / cfg.bar_E
        epsI_x_neg[i] = inc_x_neg / cfg.bar_E
        epsI_y_pos[i] = inc_y_pos / cfg.bar_E
        epsI_y_neg[i] = inc_y_neg / cfg.bar_E
        epsI_z_pos[i] = inc_z_pos / cfg.bar_E
        epsI_z_neg[i] = inc_z_neg / cfg.bar_E

        # === SPECIMEN AXIAL (X) RESPONSE ===
        # Lateral confinement strengthens the rock (Mohr-Coulomb).
        # σ₁ is the axial baseline — it does NOT strengthen, it pre-loads.
        # Confinement uses the STATIC perpendicular pre-stress.  Using the
        # dynamic perpendicular stress here would create a positive-feedback
        # loop in async mode (each axis would strengthen because of the
        # others' dynamic stress, mutually).  Path-dependence in async mode
        # is therefore expressed through the damage / strain state of the
        # specimen when each pulse arrives, not through dynamic confinement.
        lateral_conf = 0.0 if is_gas_uniaxial else max(cfg.confinement_Y, cfg.confinement_Z)

        if is_full_hydro:
            eps_vol = (eps_x[i-1] + eps_y[i-1] + eps_z[i-1]) / 3.0
            r = rock_response_hydrostatic(eps_vol, abs(rate_x[i-1]) + 1.0, rock_params)
            sigma_dyn = r["stress"]
            sigma_peak_x = r["sigma_peak"]
        else:
            # Constitutive backbone evaluated at the running peak strain so that
            # damage (hardening -> softening -> residual) is irreversible.  The
            # DIF uses the smoothed loading rate (see dif_rate_x) to avoid the
            # transition-ripple feedback.
            r = rock_response(eps_x_max, dif_rate_x + 1.0,
                              lateral_conf, rock_params)
            sigma_backbone = r["stress"]
            sigma_peak_x = r["sigma_peak"]

            # Secant unloading to the origin (scalar elastic-damage model):
            # below the running peak strain the stress relaxes linearly with the
            # current strain along the damaged secant, so the transmitted wave
            # returns to zero after the pulse passes instead of being frozen at
            # the residual floor.  On the loading path (eps = eps_max) this
            # reproduces the backbone exactly.
            if eps_x_max > 1e-12:
                sigma_dyn = sigma_backbone * (max(eps_x[i-1], 0.0) / eps_x_max)
            else:
                sigma_dyn = 0.0
            sigma_dyn = max(0.0, min(sigma_dyn, sigma_backbone))

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

        # === STRAIN RATE ===
        if sym_X:
            # Symmetric mode keeps the non-negative clamp: opposing pulses
            # compact the specimen and the simplified driver has no unloading arm.
            rate_x[i] = max(0.0, (2.0 * cfg.bar_C0 / Ls) * epsI_x_pos[i])
            eps_x[i] = eps_x[i-1] + rate_x[i] * dt
        else:
            # One-sided drive (gas-gun modes, EM uniaxial, EM async X axis): the
            # rate may go NEGATIVE so the specimen springs back elastically once
            # the incident pulse decays.  Strain itself is clamped at zero (the
            # specimen cannot go into net tension here).
            raw_rate = (cfg.bar_C0 / Ls) * (epsI_x_pos[i] - epsR_x_pos[i] - epsT_x[i])
            rate_x[i] = raw_rate
            eps_x[i] = max(0.0, eps_x[i-1] + rate_x[i] * dt)

        # Track the irreversible peak strain that the constitutive backbone uses.
        if eps_x[i] > eps_x_max:
            eps_x_max = eps_x[i]

        # Update the smoothed loading rate that drives the DIF.  Only positive
        # (loading) rates contribute; the smoothing constant damps the
        # transition ripple.  (EMA: tau ~ 10 steps = 10 us.)
        ema = 0.1
        dif_rate_x = (1.0 - ema) * dif_rate_x + ema * max(rate_x[i], 0.0)

        # === LATERAL Y, Z RESPONSE ===
        damage_ratio = sigma_dyn / sigma_peak_x if sigma_peak_x > 0 else 0.0
        dilation = 1.0 + 3.0 * (damage_ratio - 0.6) if damage_ratio > 0.6 else 1.0
        # Use the actual axial strain increment (strain is now clamped at zero on
        # spring-back, so this can differ from rate_x[i]*dt) for the Poisson term.
        d_eps_x = eps_x[i] - eps_x[i-1]
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
            # === Per-axis three-wave SHPB for asynchronous triaxial mode ===
            # Each axis (Y, Z) runs its own constitutive evaluation and three-
            # wave reduction, mirroring the X-axis treatment above. The +Y / +Z
            # bars are incident bars and the -Y / -Z bars are pure transmission
            # bars (epsR_*_neg therefore stays zero on those axes).
            #
            # Cross-axis coupling enters through:
            #   (a) confinement-dependent strengthening of the rock
            #       (lateral_conf_y uses sig_x, sig_z; analogous for Z),
            #   (b) Poisson lateral expansion that piggybacks on each axis's
            #       strain rate (passive_lat from X plus a Y-driven term).

            # --- Y axis ---
            # Confinement uses STATIC perpendicular pre-stress (same convention
            # as the X axis above) to avoid the runaway feedback that would
            # arise from using dynamic perpendicular stresses.
            lateral_conf_y = max(cfg.confinement_X, cfg.confinement_Z)
            r_y = rock_response(eps_y_max, dif_rate_y + 1.0,
                                lateral_conf_y, rock_params)
            sigma_y_backbone = r_y["stress"]
            sigma_y_peak = r_y["sigma_peak"]
            sigma_y_dyn = (sigma_y_backbone * (max(eps_y[i-1], 0.0) / eps_y_max)
                           if eps_y_max > 1e-12 else 0.0)
            sigma_y_dyn = max(0.0, min(sigma_y_dyn, sigma_y_backbone))
            epsT_y[i] = sigma_y_dyn * As / (cfg.bar_E * Ab)
            epsR_y_pos[i] = epsT_y[i] - epsI_y_pos[i]
            epsR_y_neg[i] = 0.0
            # Rate may go negative so the Y specimen springs back after its pulse.
            bar_rate_y = (cfg.bar_C0 / Ls) * (
                epsI_y_pos[i] - epsR_y_pos[i] - epsT_y[i]
            )
            rate_y[i] = bar_rate_y
            # Bar-driven compression + Poisson expansion from the X loading
            eps_y[i] = max(0.0, eps_y[i-1] + bar_rate_y * dt + passive_lat)
            if eps_y[i] > eps_y_max:
                eps_y_max = eps_y[i]
            dif_rate_y = (1.0 - 0.1) * dif_rate_y + 0.1 * max(bar_rate_y, 0.0)
            sig_y[i] = sigY_static + sigma_y_dyn

            # Poisson contribution from Y to Z (so Z sees the effect of an
            # active Y pulse on the same step)
            damage_ratio_y = (sigma_y_dyn / sigma_y_peak) if sigma_y_peak > 0 else 0.0
            dilation_y = 1.0 + 3.0 * (damage_ratio_y - 0.6) if damage_ratio_y > 0.6 else 1.0
            passive_lat_y = -rock_params["nu"] * max(bar_rate_y, 0.0) * dt * dilation_y

            # --- Z axis ---
            lateral_conf_z = max(cfg.confinement_X, cfg.confinement_Y)
            r_z = rock_response(eps_z_max, dif_rate_z + 1.0,
                                lateral_conf_z, rock_params)
            sigma_z_backbone = r_z["stress"]
            sigma_z_dyn = (sigma_z_backbone * (max(eps_z[i-1], 0.0) / eps_z_max)
                           if eps_z_max > 1e-12 else 0.0)
            sigma_z_dyn = max(0.0, min(sigma_z_dyn, sigma_z_backbone))
            epsT_z[i] = sigma_z_dyn * As / (cfg.bar_E * Ab)
            epsR_z_pos[i] = epsT_z[i] - epsI_z_pos[i]
            epsR_z_neg[i] = 0.0
            bar_rate_z = (cfg.bar_C0 / Ls) * (
                epsI_z_pos[i] - epsR_z_pos[i] - epsT_z[i]
            )
            rate_z[i] = bar_rate_z
            eps_z[i] = max(0.0, eps_z[i-1] + bar_rate_z * dt + passive_lat + passive_lat_y)
            if eps_z[i] > eps_z_max:
                eps_z_max = eps_z[i]
            dif_rate_z = (1.0 - 0.1) * dif_rate_z + 0.1 * max(bar_rate_z, 0.0)
            sig_z[i] = sigZ_static + sigma_z_dyn
        else:
            eps_y[i] = eps_y[i-1] + passive_lat
            eps_z[i] = eps_z[i-1] + passive_lat
            if is_confinement_chamber:
                sig_y[i] = sigY_static
                sig_z[i] = sigZ_static
            else:
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
        elif is_async:
            # Async Y axis is fully handled by the per-axis three-wave block
            # above (epsT_y, epsR_y_pos, rate_y already set). Skip overwriting.
            pass
        else:
            # Passive lateral output (gas-gun modes, em-uniaxial).
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
        elif is_async:
            # Async Z axis is fully handled by the per-axis three-wave block
            # above (epsT_z, epsR_z_pos, rate_z already set). Skip overwriting.
            pass
        else:
            # Passive lateral output (gas-gun modes, em-uniaxial).
            epsR_z_pos[i] = epsZ_dyn[i] if abs(inc_z_pos) < 1e-12 else (sig_z_dyn * As / (cfg.bar_E * Ab) - epsI_z_pos[i])
            epsR_z_neg[i] = 0.0
            rate_z[i] = max(0.0, (cfg.bar_C0 / Ls) * (epsI_z_pos[i] - epsR_z_pos[i]))

    if is_gas_triaxial:
        # Passive Y/Z output bars measure travelling lateral waves, not the
        # cumulative Poisson strain. With equal gauge distances, their raw gauge
        # arrival is only slightly later than the X transmitted wave; the extra
        # delay is a specimen lateral-response delay, not another bar-length
        # travel delay.
        axial_dyn = np.maximum(sig_x - sigX_static, 0.0)
        axial_peak = float(np.max(axial_dyn))
        if axial_peak > 0.0:
            gas_tau = 2.0 * 0.5 / cfg.bar_C0
            m_rock = (
                cfg.material_E * (1.0 - cfg.material_nu)
                / ((1.0 + cfg.material_nu) * max(1.0 - 2.0 * cfg.material_nu, 1e-9))
            )
            cp_rock = np.sqrt(max(m_rock / max(cfg.material_density, 1.0), 1.0))
            specimen_transit = Ls / cp_rock
            transmitted_peak_time = time[int(np.argmax(np.abs(epsT_x)))]
            lateral_delay = 1.25 * specimen_transit
            lateral_stagger = 0.35 * specimen_transit
            width = max(1.6 * specimen_transit, 0.12 * gas_tau)

            def lateral_output_pulse(peak: float, center: float, rebound_scale: float = 0.16) -> np.ndarray:
                main = peak * np.exp(-0.5 * ((time - center) / width) ** 2)
                shoulder = 0.22 * peak * np.exp(-0.5 * ((time - (center + 0.55 * width)) / (0.65 * width)) ** 2)
                rebound = rebound_scale * peak * np.exp(-0.5 * ((time - (center + 1.85 * width)) / (0.45 * width)) ** 2)
                return main + shoulder - rebound

            peak_y = min(0.22 * axial_peak, 0.12 * axial_peak + 0.35 * sigY_static)
            peak_z = min(0.24 * axial_peak, 1.05 * (0.12 * axial_peak + 0.35 * sigZ_static))

            y1_dyn = lateral_output_pulse(0.94 * peak_y, transmitted_peak_time + lateral_delay)
            y2_dyn = lateral_output_pulse(1.04 * peak_y, transmitted_peak_time + lateral_delay + lateral_stagger, rebound_scale=0.14)
            z1_dyn = lateral_output_pulse(1.06 * peak_z, transmitted_peak_time + lateral_delay + 0.55 * lateral_stagger, rebound_scale=0.18)
            z2_dyn = lateral_output_pulse(0.98 * peak_z, transmitted_peak_time + lateral_delay + 1.15 * lateral_stagger, rebound_scale=0.15)

            epsR_y_pos = y1_dyn * As / (cfg.bar_E * Ab)
            epsR_y_neg = y2_dyn * As / (cfg.bar_E * Ab)
            epsR_z_pos = z1_dyn * As / (cfg.bar_E * Ab)
            epsR_z_neg = z2_dyn * As / (cfg.bar_E * Ab)
            epsY_dyn = 0.5 * (epsR_y_pos + epsR_y_neg)
            epsZ_dyn = 0.5 * (epsR_z_pos + epsR_z_neg)

            sig_y_dyn = 0.5 * (y1_dyn + y2_dyn)
            sig_z_dyn = 0.5 * (z1_dyn + z2_dyn)
            sig_y = sigY_static + sig_y_dyn
            sig_z = sigZ_static + sig_z_dyn

            lateral_modulus = max(0.45 * rock_params["E_s"], 1.0)
            eps_y = sig_y_dyn / lateral_modulus
            eps_z = sig_z_dyn / lateral_modulus
            rate_y = np.gradient(eps_y, dt)
            rate_z = np.gradient(eps_z, dt)

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
        epsI_y_pos, epsI_y_neg, epsR_y_pos, epsR_y_neg, epsT_y,
        epsI_z_pos, epsI_z_neg, epsR_z_pos, epsR_z_neg, epsT_z,
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

    incident_arrays = [
        epsI_x_pos, epsI_x_neg, epsI_y_pos, epsI_y_neg, epsI_z_pos, epsI_z_neg,
    ]
    peak_incident_MPa = float(max(np.max(arr * cfg.bar_E) for arr in incident_arrays)) / 1e6

    summary = dict(
        peak_incident_MPa=peak_incident_MPa,
        peak_specimen_stress_MPa=float(np.max(sig_x)) / 1e6,
        peak_bar_stress_MPa=max_bar_stress / 1e6,
        peak_strain_pct=float(np.max(eps_x)) * 100,
        peak_pressure_MPa=float(np.max(pressure)) / 1e6,
        avg_strain_rate=avg_rate,
        pulse_duration_us=2.0 * 0.5 / cfg.bar_C0 * 1e6 if is_gas_drive else cfg.pulse_duration * 1e6,
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
        epsT_x=epsT_x, epsT_y=epsT_y, epsT_z=epsT_z,
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
            loading_family=cfg.loading_family,
            em_topology=cfg.em_topology,
            active_axes=cfg.active_axes,
            legacy_mode={
                "em-uniaxial": "em-uniaxial",
                "em-async": "em-async",
                "em-symmetric": "em-symmetric",
            }.get(mode, mode),
            peak_stress_X=cfg.peak_stress_X,
            peak_stress_Y=cfg.peak_stress_Y,
            peak_stress_Z=cfg.peak_stress_Z,
        ),
    )


# =============================================================================
# DATA EXPORT HELPERS
# =============================================================================
def build_signals_csv(result: Dict) -> bytes:
    """All time-series signals as a single CSV."""
    cfg = result.get("config", {})
    t_us = result["time"] * 1e6
    offsets = gauge_time_offsets_us(cfg)
    df = pd.DataFrame({
        "time_us":             t_us,
        "time_aligned_us":     t_us,
        "time_gauge_incident_us": t_us + offsets["incident"],
        "time_gauge_reflected_us": t_us + offsets["reflected"],
        "time_gauge_transmitted_us": t_us + offsets["transmitted"],

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
        "eps_T_y_uS":          result["epsT_y"] * 1e6,
        "eps_T_z_uS":          result["epsT_z"] * 1e6,

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
            "exported_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
            ("loading_family", cfg.get("loading_family", ""), "-"),
            ("em_topology", cfg.get("em_topology", ""), "-"),
            ("active_axes", cfg.get("active_axes", ""), "-"),
            ("legacy_mode", cfg.get("legacy_mode", cfg["mode"]), "-"),
            ("rock_type", cfg["rock_type"], "-"),
            ("velocity", cfg["velocity"], "m/s"),
            ("peak_stress", cfg["peak_stress"] / 1e6, "MPa"),
            ("peak_stress_X", cfg.get("peak_stress_X", cfg["peak_stress"]) / 1e6, "MPa"),
            ("peak_stress_Y", cfg.get("peak_stress_Y", cfg["peak_stress"]) / 1e6, "MPa"),
            ("peak_stress_Z", cfg.get("peak_stress_Z", cfg["peak_stress"]) / 1e6, "MPa"),
            ("pulse_duration", cfg["pulse_duration"] * 1e6, "us"),
            ("confinement_X (sigma_1)", cfg["confinement_X"] / 1e6, "MPa"),
            ("confinement_Y (sigma_2)", cfg["confinement_Y"] / 1e6, "MPa"),
            ("confinement_Z (sigma_3)", cfg["confinement_Z"] / 1e6, "MPa"),
            ("specimen_size", cfg["specimen_size"] * 1000, "mm"),
            ("pulse_delay_Y", cfg["pulse_delay_Y"] * 1e6, "us"),
            ("pulse_delay_Z", cfg["pulse_delay_Z"] * 1e6, "us"),
            ("symmetric_axes", cfg["symmetric_axes"], "-"),
            ("gauge_trigger_offset", cfg.get("gauge_trigger_offset_us", DEFAULT_GAUGE_TRIGGER_US), "us"),
            ("incident_gauge_distance", cfg.get("incident_gauge_distance_m", DEFAULT_GAUGE_DISTANCE_M), "m"),
            ("transmission_gauge_distance", cfg.get("transmission_gauge_distance_m", DEFAULT_GAUGE_DISTANCE_M), "m"),
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
        font-size: clamp(1.75rem, 2.1vw, 2.4rem);
        line-height: 1.12;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff 0%, #ff6b9d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6a7287;
        font-size: 0.82rem;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .metric-good { color: #06d6a0 !important; }
    .metric-warn { color: #ffa032 !important; }
    .metric-crit { color: #ff3c50 !important; }
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
        text-transform: none !important;
        letter-spacing: 0 !important;
        font-size: 0.74rem !important;
        line-height: 1.12 !important;
        color: #a8afbb !important;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.72rem !important;
        line-height: 1.1 !important;
    }
    .stSlider > div[data-baseweb="slider"] { padding: 0 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ---- Header ----
st.markdown('<div class="main-header">Dynamic Triaxial Hopkinson Bar Workspace</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Test design, wave analysis, stress paths, and damage validation</div>',
            unsafe_allow_html=True)
st.markdown(
    "Four loading families: gas-gun uniaxial SHPB, confinement-chamber SHPB, "
    "gas-gun Tri-HB static-dynamic loading, and electromagnetic programmable "
    "loading. The EM family keeps single-axis, multi-axis one-sided, and "
    "symmetric opposing-pair topologies as distinct calculation branches."
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

    with exp_tab.expander("Gauge timing display", expanded=False):
        gauge_trigger_offset_us = st.number_input(
            "Acquisition pre-trigger offset (μs)",
            value=DEFAULT_GAUGE_TRIGGER_US,
            min_value=0.0,
            step=10.0,
        )
        incident_gauge_distance_m = st.number_input(
            "Incident gauge distance to specimen, Li (m)",
            value=DEFAULT_GAUGE_DISTANCE_M,
            min_value=0.0,
            step=0.05,
        )
        transmission_gauge_distance_m = st.number_input(
            "Transmission/output gauge distance, Lt (m)",
            value=DEFAULT_GAUGE_DISTANCE_M,
            min_value=0.0,
            step=0.05,
        )

    test_tab.header("Test Configuration")

    loading_family_options = {
        "Gas-gun uniaxial SHPB": ("gas_gun_uniaxial", "gas-gun"),
        "Confinement-chamber SHPB": ("confinement_chamber_shpb", "confinement-chamber"),
        "Gas-gun Tri-HB static-dynamic loading": ("gas_gun_tri_hb_static_dynamic", "gas-gun-triaxial"),
        "Electromagnetic programmable loading": ("electromagnetic", None),
    }
    loading_family_label = test_tab.selectbox("Loading family", list(loading_family_options.keys()))
    loading_family, mode = loading_family_options[loading_family_label]
    em_topology = ""
    em_topology_label = ""
    active_axes = "X"

    if loading_family == "electromagnetic":
        em_topology_options = {
            "Single-axis, one-sided": ("single_axis", "em-uniaxial"),
            "Multi-axis, one-sided": ("one_sided_multiaxis", "em-async"),
            "Symmetric opposing pairs": ("symmetric_pairs", "em-symmetric"),
        }
        em_topology_label = test_tab.selectbox("EM topology", list(em_topology_options.keys()))
        em_topology, mode = em_topology_options[em_topology_label]

    is_gas_drive = mode in ("gas-gun", "confinement-chamber", "gas-gun-triaxial")
    is_confinement_chamber = mode == "confinement-chamber"
    is_gas_triaxial = mode == "gas-gun-triaxial"
    is_symmetric = mode == "em-symmetric"
    is_async = mode == "em-async"

    mode_descriptions = {
        "gas-gun": "Single striker → incident bar → specimen → transmission bar (classical SHPB). Pulse-shaped half-sine incident wave.",
        "confinement-chamber": "Axisymmetric confinement-chamber triaxial mode: σ₂ = σ₃ = pc is held constant by a pressure vessel; no Y/Z output bars.",
        "gas-gun-triaxial": "Monash Tri-HB system: gas-gun X pulse after quasi-static σ₁/σ₂/σ₃ pre-stress; total stress = static preload + dynamic increment.",
        "em-uniaxial": "EM topology: one +X half-sine pulse. Y/Z carry only Poisson reactions.",
        "em-async": "EM topology: one-sided +X/+Y/+Z pulse control with optional Y/Z delays.",
        "em-symmetric": "EM topology: opposing pulse pairs fire symmetrically. Constructive superposition gives 2× specimen drive per active axis.",
    }
    test_tab.caption(mode_descriptions[mode])

    if is_async:
        active_axes_label = test_tab.radio(
            "Active axes",
            ["X", "X + Y", "X + Y + Z"],
            index=2,
            horizontal=True,
            help="One-sided EM drive axes. X is always the reference axis.",
        )
        active_axes = {"X": "X", "X + Y": "XY", "X + Y + Z": "XYZ"}[active_axes_label]

    if is_symmetric:
        sym_axes_label = test_tab.radio(
            "Active opposing pairs",
            ["±X only", "±X ±Y", "Full hydrostatic (±X ±Y ±Z)"],
            help="Number of opposing pulse pairs that fire simultaneously.",
        )
        symmetric_axes = {"±X only": "X", "±X ±Y": "XY",
                          "Full hydrostatic (±X ±Y ±Z)": "XYZ"}[sym_axes_label]
        active_axes = symmetric_axes
    else:
        symmetric_axes = "XYZ"  # placeholder, unused

    test_tab.divider()
    test_tab.subheader("Pulse settings")
    peak_stress_X_MPa = 0
    peak_stress_Y_MPa = 0
    peak_stress_Z_MPa = 0

    if is_gas_drive:
        velocity = test_tab.slider("Striker velocity (m/s)", 5, 50, 20, step=1)
        peak_stress_MPa = 0
        pulse_duration_us = 2.0 * 0.5 / bar_C0 * 1e6
        gas_peak = 0.5 * ((bar_E_GPa * 1e9) / bar_C0) * velocity / 1e6
        test_tab.caption(f"Peak incident stress: {gas_peak:.0f} MPa "
                         f"(set by 0.5·Eb/C₀·V). Pulse-shaped to a half-sine "
                         f"(sin πt/τ), τ = 2·Lstriker/C₀, as in real SHPB tests.")
    else:
        velocity = 0
        pulse_duration_us = test_tab.slider("Pulse duration τ (μs)",
                                            50, 400, 200, step=10)
        if is_async:
            peak_stress_X_MPa = test_tab.slider("A_X pulse amplitude (MPa)", 50, 900, 400, step=25)
            peak_stress_Y_MPa = (
                test_tab.slider("A_Y pulse amplitude (MPa)", 50, 900, 400, step=25)
                if active_axes in ("XY", "XYZ") else 0
            )
            peak_stress_Z_MPa = (
                test_tab.slider("A_Z pulse amplitude (MPa)", 50, 900, 400, step=25)
                if active_axes == "XYZ" else 0
            )
            peak_stress_MPa = max(peak_stress_X_MPa, peak_stress_Y_MPa, peak_stress_Z_MPa)
        else:
            amp_label = "Per-bar pulse amplitude (MPa)" if is_symmetric else "Pulse amplitude (MPa)"
            peak_stress_MPa = test_tab.slider(amp_label, 50, 900, 400, step=25)
            peak_stress_X_MPa = peak_stress_MPa
            peak_stress_Y_MPa = peak_stress_MPa if is_symmetric and symmetric_axes in ("XY", "XYZ") else 0
            peak_stress_Z_MPa = peak_stress_MPa if is_symmetric and symmetric_axes == "XYZ" else 0
        if is_symmetric:
            test_tab.caption(
                f"Ideal paired incident drive (with superposition): "
                f"{2 * peak_stress_MPa} MPa per active axis before specimen response."
            )
            test_tab.success(
                f"Symmetry check: {sym_axes_label} uses matched +/− pulses "
                f"({peak_stress_MPa} MPa, τ = {pulse_duration_us} μs)."
            )
            if symmetric_axes == "XYZ":
                test_tab.caption("Full ±X±Y±Z selection enables the hydrostatic p-εv view when static pre-stress is also hydrostatic.")

    test_tab.divider()
    test_tab.subheader("Static pre-stress")
    if mode == "gas-gun":
        prestress_defaults = (0, 0, 0)
        prestress_disabled = True
        test_tab.caption("Classical gas-gun mode is unconfined. Select Mode 2 for chamber confinement or the Monash Tri-HB System for active true-triaxial pre-stress.")
    elif is_confinement_chamber:
        prestress_defaults = (30, 20, 20)
        prestress_disabled = False
        test_tab.caption("Mode 2: a chamber holds σ₂ = σ₃ = pc constant; only the X-axis bars carry dynamic wave data.")
    elif is_gas_triaxial:
        prestress_defaults = (30, 20, 15)
        prestress_disabled = False
        test_tab.caption("Monash Tri-HB System: σ₁, σ₂, and σ₃ are applied independently before the striker fires.")
    elif is_symmetric and symmetric_axes == "XYZ":
        prestress_defaults = (30, 30, 30)
        prestress_disabled = False
        test_tab.caption("Full symmetric XYZ: hydrostatic static pre-stress is applied before the matched opposing EM pulses.")
    else:
        prestress_defaults = (30, 20, 15)
        prestress_disabled = False
        test_tab.caption("Applied by hydraulic cylinders before the dynamic pulse arrives.")

    max_conf = 50
    if is_symmetric and symmetric_axes == "XYZ":
        hydrostatic_p0 = test_tab.slider(
            "Hydrostatic pre-stress p₀ = σ₁ = σ₂ = σ₃ — MPa",
            0,
            max_conf,
            prestress_defaults[0],
            step=1,
        )
        conf_X = hydrostatic_p0
        conf_Y = hydrostatic_p0
        conf_Z = hydrostatic_p0
    else:
        conf_X = test_tab.slider("σ₁ axial (X) — MPa", 0, max_conf, prestress_defaults[0], step=1, disabled=prestress_disabled)
    if is_confinement_chamber:
        chamber_pc = test_tab.slider("p_c chamber pressure = σ₂ = σ₃ — MPa", 0, max_conf, prestress_defaults[1], step=1)
        conf_Y = chamber_pc
        conf_Z = chamber_pc
    elif not (is_symmetric and symmetric_axes == "XYZ"):
        conf_Y = test_tab.slider("σ₂ confining (Y) — MPa", 0, max_conf, prestress_defaults[1], step=1, disabled=prestress_disabled)
        conf_Z = test_tab.slider("σ₃ confining (Z) — MPa", 0, max_conf, prestress_defaults[2], step=1, disabled=prestress_disabled)

    if is_confinement_chamber or is_gas_triaxial:
        rate_ref = float(preset.get("epsdot_ref", 1e-5))
        dif_nominal = 1.0 + float(preset["b_rate"]) * np.log10(max(150.0, rate_ref) / rate_ref)
        planned_peak = (float(material_UCS_MPa) + float(preset["k_conf"]) * max(conf_Y, conf_Z)) * dif_nominal
        planning_label = "rock dynamic peak" if is_confinement_chamber else "rock peak"
        test_tab.caption(
            f"Planning estimate at εdot ≈ 150 /s: {planning_label} ≈ {planned_peak:.0f} MPa; "
            f"incident pulse ≈ {gas_peak:.0f} MPa; bar proportional limit = {BAR.sigma_prop / 1e6:.0f} MPa."
        )

    if is_async:
        test_tab.divider()
        test_tab.subheader("Asynchronous timing")
        if active_axes == "X":
            delay_Y_us = 0
            delay_Z_us = 0
            test_tab.caption("Only X is active; Y/Z delays are not used.")
        else:
            synchronous = test_tab.checkbox("Synchronous pulses (zero delays)", value=True)
            if synchronous:
                delay_Y_us = 0
                delay_Z_us = 0
                test_tab.caption("Y/Z delays are fixed at zero for synchronous one-sided EM loading.")
            else:
                delay_Y_us = test_tab.slider("Y-pulse delay (μs)", 0, 500, 0, step=5)
                delay_Z_us = (
                    test_tab.slider("Z-pulse delay (μs)", 0, 500, 0, step=5)
                    if active_axes == "XYZ" else 0
                )
    else:
        delay_Y_us = 0
        delay_Z_us = 0


# ---- Build config and run simulation ----
config = {
    "mode": mode,
    "loading_family": loading_family,
    "em_topology": em_topology,
    "active_axes": active_axes,
    "rock_type": rock_type,
    "velocity": float(velocity),
    "peak_stress": float(peak_stress_MPa) * 1e6,
    "peak_stress_X": float(peak_stress_X_MPa) * 1e6,
    "peak_stress_Y": float(peak_stress_Y_MPa) * 1e6,
    "peak_stress_Z": float(peak_stress_Z_MPa) * 1e6,
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
result["config"].update(
    gauge_trigger_offset_us=float(gauge_trigger_offset_us),
    incident_gauge_distance_m=float(incident_gauge_distance_m),
    transmission_gauge_distance_m=float(transmission_gauge_distance_m),
    # Strength-model constants (from the active rock preset) exported so the
    # Step-3 failure surface can be derived consistently with the Step-1
    # axial strength sigma_peak = (UCS + k_conf*sigma_conf)*DIF. These are
    # added to the shared config here (not passed to simulate(), which does
    # not accept them).
    material_k_conf=float(preset["k_conf"]),
    material_b_rate=float(preset["b_rate"]),
    material_epsdot_ref=float(preset.get("epsdot_ref", 1e-5)),
)
st.session_state["tri_hb_latest_result"] = result
st.session_state["tri_hb_latest_config"] = result["config"]


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
if is_symmetric and symmetric_axes == "XYZ":
    tab_labels.append("p–εᵥ (volumetric)")
tab_labels.append("Equations")

tabs = st.tabs(tab_labels)


# ---- Common plot styling ----
def base_layout(xtitle: str, ytitle: str, height: int = 420) -> dict:
    return dict(
        height=height,
        template="plotly_white",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#222222", family=PLOT_FONT, size=13),
        xaxis=dict(
            title=dict(text=xtitle, font=dict(color="#222222", family=PLOT_FONT, size=14)),
            tickfont=dict(color="#222222", family=PLOT_FONT, size=12),
            gridcolor="#E5E7EB",
            zeroline=False,
            showline=True,
            linecolor="#222222",
            ticks="outside",
        ),
        yaxis=dict(
            title=dict(text=ytitle, font=dict(color="#222222", family=PLOT_FONT, size=14)),
            tickfont=dict(color="#222222", family=PLOT_FONT, size=12),
            gridcolor="#E5E7EB",
            zeroline=False,
            showline=True,
            linecolor="#222222",
            ticks="outside",
        ),
        margin=dict(l=66, r=24, t=24, b=58),
        legend=dict(
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#D1D5DB",
            borderwidth=1,
            font=dict(color="#222222", family=PLOT_FONT, size=12),
        ),
    )


# ---- Tab 1: Bar Waveforms ----
with tabs[0]:
    fig = go.Figure()
    t_us = result["time"] * 1e6
    time_basis = st.radio(
        "Time basis",
        ["Raw gauge time", "Aligned reduction time"],
        horizontal=True,
        help="Raw gauge time shows acquisition arrivals. Aligned reduction time is the shifted view used for three-wave stress analysis.",
    )
    if time_basis == "Raw gauge time":
        plot_offsets = gauge_time_offsets_us(result["config"])
        st.caption(
            "Raw gauge-time display: the incident, reflected, and transmitted/output "
            "pulses keep their acquisition-time travel delays. The stress and strain "
            "tabs use the aligned reduction-time histories."
        )
    else:
        plot_offsets = {"incident": 0.0, "reflected": 0.0, "transmitted": 0.0, "output": 0.0}
        st.caption(
            "Aligned reduction-time display: the three waves are shifted to the "
            "specimen/reduction origin for stress analysis."
        )

    def add_wave_trace(key: str, name: str, color: str, time_kind: str,
                       width: float = 1.8, dash: str | None = None,
                       scale: float = 1e6) -> None:
        y = result.get(key)
        if y is None or not np.any(np.abs(y) > 0.0):
            return
        x_plot, y_plot = waveform_trace_window(t_us, np.asarray(y) * scale, plot_offsets[time_kind])
        line = dict(color=color, width=width)
        if dash:
            line["dash"] = dash
        fig.add_trace(go.Scatter(x=x_plot, y=y_plot, name=name, line=line))

    if is_symmetric:
        st.markdown(
            "**Symmetric compact view:** 3 incident pair waveforms "
            "(+/− bars are identical within each active axis) plus the 6 "
            "individual reflected bar waveforms. Full XYZ therefore shows 9 traces."
        )

        active_axes = []
        if symmetric_axes in ("X", "XY", "XYZ"):
            active_axes.append(("x", "X", PLOT_COLORS["blue"]))
        if symmetric_axes in ("XY", "XYZ"):
            active_axes.append(("y", "Y", PLOT_COLORS["green"]))
        if symmetric_axes == "XYZ":
            active_axes.append(("z", "Z", PLOT_COLORS["purple"]))

        for axis_key, axis_label, color in active_axes:
            add_wave_trace(
                f"epsI_{axis_key}_pos",
                f"ε<sub>I</sub> (±{axis_label} pair)",
                color,
                "incident",
                width=2,
            )

        reflected_styles = {
            "x": (PLOT_COLORS["orange"], "dash"),
            "y": (PLOT_COLORS["vermillion"], "dot"),
            "z": (PLOT_COLORS["magenta"], "dashdot"),
        }
        for axis_key, axis_label, _ in active_axes:
            color, dash = reflected_styles[axis_key]
            add_wave_trace(
                f"epsR_{axis_key}_pos",
                f"ε<sub>R</sub> (+{axis_label} bar)",
                color,
                "reflected",
                width=1.5,
                dash=dash,
            )
            add_wave_trace(
                f"epsR_{axis_key}_neg",
                f"ε<sub>R</sub> (−{axis_label} bar)",
                color,
                "reflected",
                width=1.5,
                dash="longdash",
            )
    else:
        add_wave_trace(
            "epsI_x_pos",
            "ε<sub>I</sub> (+X incident)",
            PLOT_COLORS["blue"],
            "incident",
            width=2.2,
        )
        add_wave_trace(
            "epsR_x_pos",
            "ε<sub>R</sub> (+X reflected)",
            PLOT_COLORS["orange"],
            "reflected",
            width=1.8,
        )
        add_wave_trace(
            "epsT_x",
            "ε<sub>T</sub> (X transmitted)",
            PLOT_COLORS["green"],
            "transmitted",
            width=2.2,
        )
        has_y_incident = np.any(np.abs(result["epsI_y_pos"]) > 0.0)
        has_y_output_signal = (
            np.any(np.abs(result["epsR_y_pos"]) > 0.0)
            or np.any(np.abs(result.get("epsY_dyn", np.array([0.0]))) > 0.0)
        )
        has_y_output = has_y_output_signal and (has_y_incident or is_gas_triaxial)
        if has_y_incident:
            add_wave_trace(
                "epsI_y_pos",
                "ε<sub>I</sub> (+Y incident)",
                PLOT_COLORS["green"],
                "incident",
                width=1.8,
                dash="dot",
            )
        if has_y_output:
            if is_async:
                add_wave_trace(
                    "epsR_y_pos",
                    "ε<sub>R</sub> (+Y reflected)",
                    PLOT_COLORS["vermillion"],
                    "reflected",
                    width=1.8,
                    dash="dot",
                )
            else:
                add_wave_trace(
                    "epsR_y_pos",
                    "ε<sub>out</sub> (Y1 output bar)",
                    PLOT_COLORS["vermillion"],
                    "output",
                    width=1.8,
                    dash="dot",
                )
                if np.any(np.abs(result["epsR_y_neg"]) > 0.0):
                    add_wave_trace(
                        "epsR_y_neg",
                        "ε<sub>out</sub> (Y2 output bar)",
                        PLOT_COLORS["magenta"],
                        "output",
                        width=1.8,
                        dash="longdash",
                    )
        if np.any(np.abs(result.get("epsT_y", np.array([0.0]))) > 0.0):
            add_wave_trace(
                "epsT_y",
                "ε<sub>T</sub> (−Y transmitted)",
                PLOT_COLORS["green"],
                "transmitted",
                width=1.8,
                dash="dashdot",
            )

        has_z_incident = np.any(np.abs(result["epsI_z_pos"]) > 0.0)
        has_z_output_signal = (
            np.any(np.abs(result["epsR_z_pos"]) > 0.0)
            or np.any(np.abs(result.get("epsZ_dyn", np.array([0.0]))) > 0.0)
        )
        has_z_output = has_z_output_signal and (has_z_incident or is_gas_triaxial)
        if has_z_incident:
            add_wave_trace(
                "epsI_z_pos",
                "ε<sub>I</sub> (+Z incident)",
                PLOT_COLORS["purple"],
                "incident",
                width=1.8,
                dash="dot",
            )
        if has_z_output:
            if is_async:
                add_wave_trace(
                    "epsR_z_pos",
                    "ε<sub>R</sub> (+Z reflected)",
                    PLOT_COLORS["magenta"],
                    "reflected",
                    width=1.8,
                    dash="dot",
                )
            else:
                add_wave_trace(
                    "epsR_z_pos",
                    "ε<sub>out</sub> (Z1 output bar)",
                    PLOT_COLORS["magenta"],
                    "output",
                    width=1.8,
                    dash="dot",
                )
                if np.any(np.abs(result["epsR_z_neg"]) > 0.0):
                    add_wave_trace(
                        "epsR_z_neg",
                        "ε<sub>out</sub> (Z2 output bar)",
                        PLOT_COLORS["purple"],
                        "output",
                        width=1.8,
                        dash="longdash",
                    )
        if np.any(np.abs(result.get("epsT_z", np.array([0.0]))) > 0.0):
            add_wave_trace(
                "epsT_z",
                "ε<sub>T</sub> (−Z transmitted)",
                PLOT_COLORS["purple"],
                "transmitted",
                width=1.8,
                dash="dashdot",
            )

    waveform_xtitle = (
        "Acquisition time, t (μs)"
        if time_basis == "Raw gauge time"
        else "Aligned reduction time, t (μs)"
    )
    fig.update_layout(**base_layout(waveform_xtitle, "Bar strain, ε (με)"))
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "Bar gauge signals only show the **dynamic** wave component. Static "
        "pre-stress is held by the hydraulic cylinders and is added separately "
        "to the specimen stress. For the Monash Tri-HB system, Y and Z have no "
        "incident pulse; their curves are passive output-bar measurements from "
        "the opposing lateral bars used in the corrected two-bar average. If the "
        "Y/Z gauge distances match the X transmission gauge distance, their raw "
        "arrival is only slightly later than εT because the extra delay is the "
        "specimen lateral-response time. In "
        "symmetric mode the ± incident traces for each axis overlap, so one "
        "incident pair trace is shown per active axis."
    )
    if is_confinement_chamber:
        st.info(
            "Confinement-Chamber Triaxial has no Y/Z output bars. "
            "The chamber holds σ₂ = σ₃ = pc, so only X-axis bar waveforms are plotted."
        )

    if is_async:
        st.info(
            "Multi-axis one-sided EM uses the same three-wave layout on each active axis: "
            "+axis incident/reflected bars and the opposing -axis transmitted bar. "
            "Waveform shapes can still differ because A_X, A_Y, A_Z and pulse delays "
            "change the specimen damage state each axis sees."
        )

    if is_gas_triaxial:
        st.markdown("**Y/Z passive output bars — dynamic stress view**")
        fig_lat = go.Figure()
        lateral_outputs = [
            ("epsR_y_pos", "σ<sub>y1</sub> output", PLOT_COLORS["vermillion"], "solid"),
            ("epsR_y_neg", "σ<sub>y2</sub> output", PLOT_COLORS["orange"], "solid"),
            ("epsR_z_pos", "σ<sub>z1</sub> output", PLOT_COLORS["blue"], "solid"),
            ("epsR_z_neg", "σ<sub>z2</sub> output", PLOT_COLORS["purple"], "solid"),
        ]
        for key, label, color, dash in lateral_outputs:
            if np.any(np.abs(result[key]) > 0.0):
                x_plot, y_plot = waveform_trace_window(
                    t_us,
                    result[key] * config["bar_E"] / 1e6,
                    plot_offsets["output"],
                )
                fig_lat.add_trace(go.Scatter(
                    x=x_plot,
                    y=y_plot,
                    name=label,
                    line=dict(color=color, width=2, dash=dash),
                ))
        fig_lat.update_layout(**base_layout(waveform_xtitle, "Dynamic stress, Δσ (MPa)", height=330))
        st.plotly_chart(fig_lat, width="stretch")
        st.caption(
            "This separate stress-scale plot matches the experimental convention for the Y/Z output bars. "
            "The traces are delayed transient lateral waves, not static pre-stress and not cumulative lateral strain."
        )


# ---- Tab 2: Specimen Stresses ----
with tabs[1]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_us, y=result["sig_x"] / 1e6,
                             name="σ<sub>x</sub> (total)", line=dict(color=PLOT_COLORS["blue"], width=2.2)))
    fig.add_trace(go.Scatter(x=t_us, y=result["sig_y"] / 1e6,
                             name="σ<sub>y</sub> (total)", line=dict(color=PLOT_COLORS["green"], width=2.2)))
    fig.add_trace(go.Scatter(x=t_us, y=result["sig_z"] / 1e6,
                             name="σ<sub>z</sub> (total)", line=dict(color=PLOT_COLORS["purple"], width=2.2)))
    if is_symmetric:
        fig.add_trace(go.Scatter(x=t_us, y=result["pressure"] / 1e6,
                                 name="p (mean)",
                                 line=dict(color=PLOT_COLORS["black"], width=2.4, dash="dash")))
    # Bar-limit reference line
    fig.add_hline(y=BAR.sigma_prop / 1e6, line_dash="dash", line_color=PLOT_COLORS["vermillion"],
                  annotation_text=f"bar limit ({BAR.sigma_prop / 1e6:.0f} MPa)",
                  annotation_position="top right",
                  annotation_font=dict(color=PLOT_COLORS["vermillion"], size=10))
    fig.update_layout(**base_layout("Time, t (μs)", "Stress, σ (MPa)"))
    st.plotly_chart(fig, width="stretch")

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
            line=dict(color=PLOT_COLORS["vermillion"] if is_symmetric else PLOT_COLORS["blue"], width=2.6),
            name="σ-ε",
        ))
        fig.update_layout(**base_layout("Axial strain, ε<sub>x</sub> (%)", "Axial stress, σ<sub>x</sub> (MPa)"))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No measurable strain accumulated (specimen unloaded). Increase pulse amplitude or duration.")

    st.caption(
        "Loading branch only — the curve traces hardening up to peak strength "
        "and post-peak softening, then stops when strain plateaus. Elastic "
        "unloading after the pulse is omitted to keep the constitutive curve clean."
    )


# ---- Tab 4: Pressure-volumetric (full symmetric XYZ only) ----
if is_symmetric and symmetric_axes == "XYZ":
    with tabs[3]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result["eps_vol"] * 100, y=result["pressure"] / 1e6,
            mode="lines", line=dict(color=PLOT_COLORS["black"], width=2.6),
            name="p-ε<sub>v</sub>",
        ))
        fig.update_layout(**base_layout("Volumetric strain, ε<sub>v</sub> (%)", "Mean pressure, p (MPa)"))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Compaction curve. Initial slope = bulk modulus K. Plateau = cap pressure. "
            "Under full hydrostatic loading, the deviatoric stress q ≈ 0, so the "
            "specimen cannot fail by shear — it deforms only volumetrically."
        )


# ---- Equations tab ----
with tabs[-1]:
    st.subheader("Wave Analysis Equations")
    st.caption(
        "The equations below use aligned reduction-time wave windows. The Bar "
        "Waveforms tab defaults to raw gauge/acquisition time, where incident, "
        "reflected, and transmitted pulses arrive at different times."
    )

    st.markdown("**Standard one-sided drive** (gas-gun modes, EM uniaxial, EM async):")
    st.latex(r"\sigma_i^{\rm dyn}(t) = \frac{E_b A_b}{2 A_s}\left[\varepsilon_{I,i}+\varepsilon_{R,i}+\varepsilon_{T,i}\right],\qquad i\in\{X,Y,Z\}")
    st.latex(r"\dot{\varepsilon}_i(t) = \frac{C_0}{L_i}\left[\varepsilon_{I,i}-\varepsilon_{R,i}-\varepsilon_{T,i}\right]")
    st.caption("For gas-gun uniaxial, confinement-chamber, and Monash system X-axis loading, i = X. For EM async, the same dynamic equation is applied to each driven one-sided axis.")

    st.markdown("**Confinement-Chamber Triaxial:**")
    st.latex(r"\sigma_I^{\rm peak}=\frac{1}{2}\rho_b C_0 V=\frac{1}{2}\frac{E_b}{C_0}V")
    st.latex(r"\sigma_X^{\rm total}(t)=\sigma_1+\sigma_X^{\rm dyn}(t),\qquad "
             r"\sigma_Y^{\rm total}(t)=p_c,\qquad \sigma_Z^{\rm total}(t)=p_c")
    st.latex(r"p(t)=\frac{\sigma_X^{\rm total}(t)+2p_c}{3},\qquad "
             r"q(t)=\left|\sigma_X^{\rm total}(t)-p_c\right|,\qquad "
             r"\theta=0")
    st.latex(r"\sigma_{\rm peak}^{\rm rock,dyn}=\left(\sigma_{c0}+k_{\rm conf}p_c\right)\mathrm{DIF}")
    st.caption(
        "The chamber mode is axisymmetric: σ2 = σ3 = pc is held by the pressure vessel, "
        "and no lateral output-bar wave data are available. Only the X-axis incident, reflected, "
        "and transmitted bars carry dynamic wave information."
    )

    st.markdown("**Mode 3 Monash Tri-HB system gas-gun coupled static-dynamic triaxial loading:**")
    st.latex(r"\sigma_I^{\rm peak}=\frac{1}{2}\rho_b C_0 V=\frac{1}{2}\frac{E_b}{C_0}V")
    st.latex(r"\sigma_X^{\rm total}(t)=\sigma_1+\sigma_X^{\rm dyn}(t),\quad "
             r"\sigma_Y^{\rm total}(t)=\sigma_2+\sigma_Y^{\rm dyn}(t),\quad "
             r"\sigma_Z^{\rm total}(t)=\sigma_3+\sigma_Z^{\rm dyn}(t)")
    st.latex(r"\sigma_Y^{\rm dyn}(t)=\frac{E_bA_b}{2A_s}\left[\varepsilon_{y1}(t)+\varepsilon_{y2}(t)\right],\qquad "
             r"\sigma_Z^{\rm dyn}(t)=\frac{E_bA_b}{2A_s}\left[\varepsilon_{z1}(t)+\varepsilon_{z2}(t)\right]")
    st.latex(r"\varepsilon_Y(t)=\frac{C_0}{L_s}\int_0^t\left[\varepsilon_{y1}(\tau)+\varepsilon_{y2}(\tau)\right]d\tau,\qquad "
             r"\varepsilon_Z(t)=\frac{C_0}{L_s}\int_0^t\left[\varepsilon_{z1}(\tau)+\varepsilon_{z2}(\tau)\right]d\tau")
    st.latex(r"\sigma_{\rm peak}^{\rm rock,dyn}=\left[\sigma_{c0}+k_{\rm conf}\max(\sigma_2,\sigma_3)\right]\mathrm{DIF}")
    st.caption(
        "The Monash Tri-HB system keeps the gas-gun X-axis wave equations unchanged. The bar gauges are baseline-subtracted "
        "dynamic signals; the two opposing Y bars and two opposing Z bars are averaged with the 1/2 factor shown above. "
        "The hydraulic pre-stresses are added back when plotting/reporting total stress. In this app the confinement-strengthening "
        "term uses the static perpendicular stresses, while σ1 is only an axial baseline."
    )

    st.markdown("**EM asynchronous triaxial detail.** Each active axis has its own incident, "
                "reflected, and transmitted bar signal and its own constitutive evaluation. "
                "The incident pulse on axis $i$ is shifted by $\\Delta t_i$:")
    st.latex(r"\varepsilon_{I,i}(t)=\frac{\sigma_{\rm peak}}{E_b}\sin\!\left(\frac{\pi(t-\Delta t_i)}{\tau}\right),\qquad \Delta t_x=0")
    st.markdown("Bar-to-bar wave superposition does **not** occur — the three bars are physically "
                "orthogonal and only share the specimen. Cross-axis coupling enters through (i) the "
                "Mohr–Coulomb confinement-dependent strength on each axis (using the *static* "
                "perpendicular pre-stress to avoid feedback runaway between dynamically loaded axes),")
    st.latex(r"\sigma_{\rm peak,Y}=\bigl(\sigma_{c0}+k_{\rm conf}\,\max(\sigma_1^{\rm static},\sigma_3^{\rm static})\bigr)\cdot \mathrm{DIF},\quad "
             r"\sigma_{\rm peak,Z}=\bigl(\sigma_{c0}+k_{\rm conf}\,\max(\sigma_1^{\rm static},\sigma_2^{\rm static})\bigr)\cdot \mathrm{DIF}")
    st.markdown("and (ii) Poisson lateral expansion carried between axes,")
    st.latex(r"\Delta\varepsilon_i^{\rm Poisson}=-\nu\,\bigl(\dot\varepsilon_x+\dot\varepsilon_y\bigr)\,dt\,\cdot\,\text{dilation factor}")
    st.markdown("So the equations on every axis remain the standard three-wave form above; only the "
                "damage / strain state of the specimen each pulse meets differs from the synchronous case.")

    st.markdown("**Symmetric opposing-pairs topology** (legacy Mode 6, opposing bars fire identically):")
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
        rate_ref_text = f"{preset.get('epsdot_ref', 1e-5):.1g}".replace("e-0", "e-").replace("e+0", "e+")
        st.caption(
            f"For the selected {rock_type} preset, b_rate = {preset['b_rate']:.2f} "
            f"and εdot_ref = {rate_ref_text} s⁻¹. "
            "They are calibration constants for the dynamic increase factor, not quantities derived from bar geometry."
        )
        st.markdown(
            "To calibrate them, run or collect tests at the same confinement but different strain rates, "
            "compute $\\mathrm{DIF}=\\sigma_{\\text{peak}}/(\\sigma_{c0}+k_{\\text{conf}}\\sigma_{\\text{conf}})$, "
            "then fit the slope of $\\mathrm{DIF}$ against $\\log_{10}(\\dot\\varepsilon/\\dot\\varepsilon_{\\text{ref}})$. "
            "$\\dot\\varepsilon_{\\text{ref}}$ is the reference rate where $\\mathrm{DIF}=1$; "
            "use the value associated with the fitted $b_{\\text{rate}}$ dataset."
        )
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
        st.dataframe(df.head(8), width="stretch", height=280)


# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.caption(
    "Virtual Tri-HB Streamlit · v3 · 1-D wave-analysis workspace. "
    "Check dispersion correction, equilibrium verification, pulse shaping, and bar plasticity bounds before interpreting experimental results."
)
