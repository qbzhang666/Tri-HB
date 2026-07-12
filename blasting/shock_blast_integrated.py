"""Integrated Chapter 17 shock and blast computation workspace.

Run from the repository root with:

    py -3.13 -m streamlit run shock_blast_integrated.py

The interface joins virtual plate-impact configuration, auditable shock
reduction, experimental velocity/Hugoniot comparison, an idealised C-J source
screen, empirical scaled-distance fields, established constitutive evidence,
and engineering consequences. It is a teaching and pre-analysis workspace, not
a substitute for a site-calibrated blast design or a hydrocode solution.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from models.shock_blast import (
    HugoniotMaterial,
    acoustic_pressure_transmission,
    cj_state,
    compare_velocity_histories,
    cubic_root_scaled_distance,
    empirical_peak_pressure_pa,
    empirical_ppv_mm_s,
    fit_linear_hugoniot,
    hugoniot_curve,
    impedance_match,
    impedance_ppv_mm_s,
    hjc_normalized_strength,
    regime_label,
    jh2_normalized_strength,
    rht_normalized_failure_strength,
    spall_strength_pa,
    square_root_scaled_distance,
    symmetric_impact_state,
    threshold_radius_m,
    virtual_free_surface_velocity_history,
)


st.set_page_config(
    page_title="Shock and Blast Workspace",
    page_icon="SB",
    layout="wide",
    initial_sidebar_state="expanded",
)


COLORS = {
    "ink": "#18212b",
    "muted": "#697586",
    "blue": "#1769aa",
    "cyan": "#00a6a6",
    "orange": "#d97706",
    "red": "#c43d4b",
    "green": "#168568",
    "paper": "#f7f9fb",
    "line": "#d8dee7",
}

PLOT_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Aptos, Segoe UI, Arial, sans-serif", size=13, color=COLORS["ink"]),
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    xaxis=dict(gridcolor=COLORS["line"], zerolinecolor=COLORS["line"]),
    yaxis=dict(gridcolor=COLORS["line"], zerolinecolor=COLORS["line"]),
    margin=dict(l=58, r=24, t=52, b=52),
    hoverlabel=dict(bgcolor="white"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)


st.markdown(
    f"""
    <style>
    .block-container {{
        width: 100% !important; max-width: 1200px; box-sizing: border-box;
        overflow-x: hidden; padding-top: 1.4rem; padding-bottom: 2.4rem;
    }}
    h1 {{ font-size: 2.05rem !important; line-height: 1.12 !important; letter-spacing: 0 !important; }}
    h2 {{ font-size: 1.42rem !important; letter-spacing: 0 !important; margin-top: 1.15rem !important; }}
    h3 {{ font-size: 1.08rem !important; letter-spacing: 0 !important; }}
    p, li, label, div[data-testid="stMarkdownContainer"] {{ font-size: 0.95rem; line-height: 1.48; }}
    div[data-testid="stMetric"] {{
        border-top: 2px solid {COLORS['line']}; padding: 0.45rem 0.15rem 0.2rem 0.15rem;
    }}
    div[data-testid="stMetricLabel"] {{ color: {COLORS['muted']}; }}
    .chain {{
        display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
        gap: 8px; margin: 0.5rem 0 1.1rem 0;
    }}
    .chain > div {{
        min-height: 74px; border-left: 3px solid {COLORS['blue']};
        background: {COLORS['paper']}; color: {COLORS['ink']}; padding: 10px 11px;
    }}
    .chain b {{ display: block; color: {COLORS['ink']}; font-size: 0.84rem; margin-bottom: 4px; }}
    .chain span {{ display: block; color: {COLORS['muted']}; font-size: 0.78rem; line-height: 1.3; }}
    .chapter-tag {{
        display: inline-block; border: 1px solid {COLORS['line']}; border-radius: 4px;
        padding: 2px 6px; margin: 1px 3px 1px 0; color: {COLORS['muted']}; font-size: 0.76rem;
    }}
    .callout {{
        border-left: 4px solid {COLORS['orange']}; background: #fff9ee;
        padding: 10px 13px; margin: 0.6rem 0 1rem 0; color: {COLORS['ink']};
    }}
    @media (max-width: 900px) {{
        .chain {{ grid-template-columns: 1fr; }}
        .chain > div {{ min-height: auto; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


MATERIAL_PRESETS = {
    "Competent rock example": dict(
        density=2700.0, c0=4300.0, slope=1.35, cl=5200.0,
        hel=3.0, tensile=12.0, ucs=150.0,
    ),
    "Dense concrete example": dict(
        density=2400.0, c0=3600.0, slope=1.45, cl=4100.0,
        hel=1.4, tensile=4.0, ucs=55.0,
    ),
    "Porous brittle example": dict(
        density=2250.0, c0=3000.0, slope=1.75, cl=3500.0,
        hel=0.75, tensile=5.0, ucs=75.0,
    ),
}

FLYER_PRESETS = {
    "Identical material": None,
    "Aluminium-like flyer": dict(
        name="Aluminium-like flyer", density_kg_m3=2700.0,
        bulk_sound_speed_m_s=5350.0, hugoniot_slope=1.34,
        longitudinal_speed_m_s=6300.0, hel_pa=0.3e9,
        tensile_strength_pa=0.2e9, uniaxial_compressive_strength_pa=0.3e9,
    ),
    "Steel-like flyer": dict(
        name="Steel-like flyer", density_kg_m3=7850.0,
        bulk_sound_speed_m_s=4570.0, hugoniot_slope=1.49,
        longitudinal_speed_m_s=5900.0, hel_pa=1.5e9,
        tensile_strength_pa=0.6e9, uniaxial_compressive_strength_pa=1.0e9,
    ),
    "Custom flyer": "custom",
}


def _load_material_preset() -> None:
    preset = MATERIAL_PRESETS[st.session_state.material_preset]
    for key, value in preset.items():
        st.session_state[f"mat_{key}"] = value


def _initialise_state() -> None:
    defaults = MATERIAL_PRESETS["Competent rock example"]
    st.session_state.setdefault("material_preset", "Competent rock example")
    for key, value in defaults.items():
        st.session_state.setdefault(f"mat_{key}", value)
    state_defaults = {
        "case_name": "Chapter 17 demonstration",
        "impact_velocity": 900.0,
        "target_thickness": 12.0,
        "pullback_velocity": 70.0,
        "flyer_preset": "Identical material",
        "flyer_density": 2700.0,
        "flyer_c0": 5350.0,
        "flyer_slope": 1.34,
        "flyer_cl": 6300.0,
        "flyer_hel": 0.3,
        "flyer_tensile": 200.0,
        "flyer_ucs": 300.0,
        "flyer_thickness": 4.0,
        "explosive_density": 1200.0,
        "vod": 4500.0,
        "product_gamma": 3.0,
        "coupling_factor": 0.42,
        "charge_mass": 4.0,
        "observation_distance": 20.0,
        "reference_pressure_mpa": 120.0,
        "reference_z": 1.0,
        "pressure_exponent": 1.55,
        "ppv_k": 850.0,
        "ppv_n": 1.55,
        "borehole_radius_mm": 45.0,
        "radial_exponent": 1.6,
    }
    for key, value in state_defaults.items():
        st.session_state.setdefault(key, value)


_initialise_state()


def _material_from_state() -> HugoniotMaterial:
    return HugoniotMaterial(
        name=st.session_state.material_preset,
        density_kg_m3=float(st.session_state.mat_density),
        bulk_sound_speed_m_s=float(st.session_state.mat_c0),
        hugoniot_slope=float(st.session_state.mat_slope),
        longitudinal_speed_m_s=float(st.session_state.mat_cl),
        hel_pa=float(st.session_state.mat_hel) * 1.0e9,
        tensile_strength_pa=float(st.session_state.mat_tensile) * 1.0e6,
        uniaxial_compressive_strength_pa=float(st.session_state.mat_ucs) * 1.0e6,
    )


def _flyer_material(target: HugoniotMaterial) -> HugoniotMaterial:
    selected = st.session_state.flyer_preset
    if selected == "Identical material":
        return target
    if selected == "Custom flyer":
        return HugoniotMaterial(
            name="Custom flyer",
            density_kg_m3=float(st.session_state.flyer_density),
            bulk_sound_speed_m_s=float(st.session_state.flyer_c0),
            hugoniot_slope=float(st.session_state.flyer_slope),
            longitudinal_speed_m_s=float(st.session_state.flyer_cl),
            hel_pa=float(st.session_state.flyer_hel) * 1.0e9,
            tensile_strength_pa=float(st.session_state.flyer_tensile) * 1.0e6,
            uniaxial_compressive_strength_pa=float(st.session_state.flyer_ucs) * 1.0e6,
        )
    return HugoniotMaterial(**FLYER_PRESETS[selected])


def _plot(fig: go.Figure, height: int = 470) -> None:
    fig.update_layout(**PLOT_LAYOUT, height=height)
    st.plotly_chart(
        fig,
        width="stretch",
        theme=None,
        config={"displaylogo": False},
    )


def _case_values() -> dict:
    material = _material_from_state()
    flyer = _flyer_material(material)
    impact_velocity = float(st.session_state.impact_velocity)
    if flyer is material:
        shock = symmetric_impact_state(material, impact_velocity)
        flyer_state = shock
        impact_mode = "symmetric identical-material impact"
    else:
        flyer_state, shock = impedance_match(flyer, material, impact_velocity)
        impact_mode = f"asymmetric impact with {flyer.name}"

    cj = cj_state(
        float(st.session_state.explosive_density),
        float(st.session_state.vod),
        float(st.session_state.product_gamma),
    )
    coupled_wall_pressure = float(st.session_state.coupling_factor) * cj.pressure_pa
    source_impedance = float(st.session_state.explosive_density) * float(st.session_state.vod)
    rock_impedance = material.density_kg_m3 * material.longitudinal_speed_m_s
    acoustic_wall_pressure = acoustic_pressure_transmission(
        cj.pressure_pa, source_impedance, rock_impedance
    )

    distance = float(st.session_state.observation_distance)
    mass = float(st.session_state.charge_mass)
    field_pressure = empirical_peak_pressure_pa(
        distance,
        mass,
        float(st.session_state.reference_pressure_mpa) * 1.0e6,
        float(st.session_state.reference_z),
        float(st.session_state.pressure_exponent),
        pressure_cap_pa=coupled_wall_pressure,
    )
    ppv = empirical_ppv_mm_s(
        distance,
        mass,
        float(st.session_state.ppv_k),
        float(st.session_state.ppv_n),
    )
    ppv_impedance = impedance_ppv_mm_s(
        field_pressure, material.density_kg_m3, material.longitudinal_speed_m_s
    )
    spall = spall_strength_pa(
        material.density_kg_m3,
        material.longitudinal_speed_m_s,
        float(st.session_state.pullback_velocity),
    )
    return {
        "material": material,
        "flyer": flyer,
        "shock": shock,
        "flyer_state": flyer_state,
        "impact_mode": impact_mode,
        "cj": cj,
        "coupled_wall_pressure_pa": coupled_wall_pressure,
        "acoustic_wall_pressure_pa": acoustic_wall_pressure,
        "field_pressure_pa": field_pressure,
        "ppv_mm_s": ppv,
        "ppv_impedance_mm_s": ppv_impedance,
        "spall_pa": spall,
        "z_cubic": cubic_root_scaled_distance(distance, mass),
        "sd_sqrt": square_root_scaled_distance(distance, mass),
    }


def _virtual_plate_impact(case: dict) -> dict:
    """Return the current idealised free-surface virtual test."""

    return virtual_free_surface_velocity_history(
        case["shock"],
        case["flyer_state"],
        float(st.session_state.target_thickness) * 1.0e-3,
        float(st.session_state.flyer_thickness) * 1.0e-3,
        float(st.session_state.pullback_velocity),
    )


def _read_uploaded_table(uploaded_file) -> pd.DataFrame:
    """Read a CSV or Excel plate-impact record."""

    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def _finite_numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce selected columns to finite numeric rows."""

    clean = frame.loc[:, columns].copy()
    for column in columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
    return clean


with st.sidebar:
    st.title("Shock and blast")
    st.caption("Chapter 17 computational workspace")
    st.text_input("Case name", key="case_name")

    st.subheader("Material")
    st.selectbox(
        "Illustrative preset",
        list(MATERIAL_PRESETS),
        key="material_preset",
    )
    st.button("Load material preset", on_click=_load_material_preset, width="stretch")
    with st.expander("Hugoniot and strength inputs", expanded=False):
        st.number_input("Initial density (kg/m3)", min_value=500.0, max_value=8000.0, step=10.0, key="mat_density")
        st.number_input("Hugoniot intercept c0 (m/s)", min_value=500.0, max_value=10000.0, step=50.0, key="mat_c0")
        st.number_input("Hugoniot slope s", min_value=0.2, max_value=4.0, step=0.05, key="mat_slope")
        st.number_input("Longitudinal speed cL (m/s)", min_value=500.0, max_value=12000.0, step=50.0, key="mat_cl")
        st.number_input("HEL (GPa)", min_value=0.01, max_value=30.0, step=0.1, key="mat_hel")
        st.number_input("Tensile strength (MPa)", min_value=0.1, max_value=500.0, step=0.5, key="mat_tensile")
        st.number_input("UCS (MPa)", min_value=1.0, max_value=2000.0, step=5.0, key="mat_ucs")

    st.subheader("Plate impact")
    st.selectbox("Flyer", list(FLYER_PRESETS), key="flyer_preset")
    if st.session_state.flyer_preset == "Custom flyer":
        with st.expander("Custom flyer Hugoniot", expanded=True):
            st.number_input("Flyer density (kg/m3)", min_value=500.0, max_value=20000.0, step=10.0, key="flyer_density")
            st.number_input("Flyer Hugoniot intercept c0 (m/s)", min_value=500.0, max_value=15000.0, step=50.0, key="flyer_c0")
            st.number_input("Flyer Hugoniot slope s", min_value=0.2, max_value=4.0, step=0.05, key="flyer_slope")
            st.number_input("Flyer longitudinal speed cL (m/s)", min_value=500.0, max_value=15000.0, step=50.0, key="flyer_cl")
            st.number_input("Flyer HEL (GPa)", min_value=0.01, max_value=100.0, step=0.1, key="flyer_hel")
            st.number_input("Flyer tensile strength (MPa)", min_value=0.1, max_value=5000.0, step=1.0, key="flyer_tensile")
            st.number_input("Flyer compressive strength (MPa)", min_value=1.0, max_value=10000.0, step=5.0, key="flyer_ucs")
    st.number_input("Impact velocity (m/s)", min_value=10.0, max_value=6000.0, step=25.0, key="impact_velocity")
    st.number_input("Flyer thickness (mm)", min_value=0.1, max_value=100.0, step=0.5, key="flyer_thickness")
    st.number_input("Target thickness (mm)", min_value=1.0, max_value=200.0, step=1.0, key="target_thickness")
    st.number_input("Pullback velocity (m/s)", min_value=0.0, max_value=1000.0, step=5.0, key="pullback_velocity")

    st.subheader("Blast source")
    with st.expander("C-J screening inputs", expanded=False):
        st.number_input("Explosive density (kg/m3)", min_value=200.0, max_value=3000.0, step=25.0, key="explosive_density")
        st.number_input("Measured or assumed VOD (m/s)", min_value=500.0, max_value=12000.0, step=100.0, key="vod")
        st.number_input("Product gamma", min_value=1.05, max_value=8.0, step=0.05, key="product_gamma")
        st.slider("Wall-pressure coupling factor", min_value=0.05, max_value=1.0, step=0.01, key="coupling_factor")

    st.subheader("Field point")
    st.number_input("Charge mass per delay (kg)", min_value=0.01, max_value=100000.0, step=0.5, key="charge_mass")
    st.number_input("Observation distance (m)", min_value=0.05, max_value=10000.0, step=1.0, key="observation_distance")
    with st.expander("Site attenuation inputs", expanded=False):
        st.number_input("Pressure at reference Z (MPa)", min_value=0.001, max_value=10000.0, step=5.0, key="reference_pressure_mpa")
        st.number_input("Reference cubic-root Z", min_value=0.01, max_value=100.0, step=0.1, key="reference_z")
        st.number_input("Pressure exponent m", min_value=0.1, max_value=5.0, step=0.05, key="pressure_exponent")
        st.number_input("PPV site constant K", min_value=0.01, max_value=100000.0, step=10.0, key="ppv_k")
        st.number_input("PPV exponent n", min_value=0.1, max_value=5.0, step=0.05, key="ppv_n")

    st.divider()
    page = st.radio(
        "Workspace",
        [
            "Case overview",
            "Plate-impact animation",
            "Virtual plate-impact test",
            "Shock state and EOS",
            "Experimental validation",
            "C-J blast source",
            "Pressure and PPV",
            "Established model scope",
            "Engineering consequences",
            "Summary and export",
        ],
        label_visibility="collapsed",
    )


values = _case_values()
material: HugoniotMaterial = values["material"]
shock = values["shock"]
cj = values["cj"]
virtual_test = _virtual_plate_impact(values)

st.title("Shock and blast computation")
st.caption(
    f"{st.session_state.case_name} | {material.name} | pressure positive in compression | SI computation"
)


def render_workflow() -> None:
    st.markdown(
        """
        <div class="chain">
          <div><b>Virtual plate impact</b><span>Flyer, target and geometry to Us, up and a diagnostic trace</span></div>
          <div><b>EOS and HEL</b><span>Compression, energy, precursor and release evidence</span></div>
          <div><b>Experiment comparison</b><span>Upload velocity or Hugoniot data and report residuals</span></div>
          <div><b>C-J source</b><span>Ideal source screen, coupling and separate rock impedance</span></div>
          <div><b>Field transfer</b><span>Scaled distance, pressure and site-calibrated PPV</span></div>
          <div><b>Decision output</b><span>Model evidence, damage zones and validation measurements</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview() -> None:
    render_workflow()
    top_metrics = st.columns(3)
    top_metrics[0].metric("Impact pressure", f"{shock.pressure_pa / 1e9:.2f} GPa")
    top_metrics[1].metric("C-J screen", f"{cj.pressure_pa / 1e9:.2f} GPa")
    top_metrics[2].metric("Coupled wall screen", f"{values['coupled_wall_pressure_pa'] / 1e9:.2f} GPa")
    lower_metrics = st.columns(2)
    lower_metrics[0].metric("Field pressure", f"{values['field_pressure_pa'] / 1e6:.2f} MPa")
    lower_metrics[1].metric("Empirical PPV", f"{values['ppv_mm_s']:.1f} mm/s")

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.subheader("Evidence chain")
        evidence = pd.DataFrame(
            [
                ["Initial stress and disturbance", "Pressure/rate path and unloading state", "Chapters 3 and 7"],
                ["Fracture and cumulative damage", "Tension, shear, crack orientation and history", "Sections 13.6 and 14.6"],
                ["Classical model card", "EOS, strength surface, rate, damage and residual response", "Section 14.2"],
                ["Shock behaviour", "Us-up, HEL, release, spall and recovered damage", "Chapter 15"],
                ["Excavation transfer", "Geometry, source, drilling, free surfaces and acceptance", "Chapter 20"],
                ["Blast consequences", "Crushed zone, EDZ, PPV, overbreak and cumulative damage", "Chapter 22"],
            ],
            columns=["Evidence object", "Workspace use", "Book handoff"],
        )
        st.dataframe(evidence, hide_index=True, width="stretch")
    with right:
        st.subheader("Current interpretation")
        st.markdown(
            f"""
            **Impact configuration:** {values['impact_mode']}  
            **Shock regime:** {'above' if shock.hel_exceeded else 'below'} the declared HEL  
            **Field-point regime:** {regime_label(values['field_pressure_pa'], material)}  
            **Spall estimate:** {values['spall_pa'] / 1e6:.1f} MPa from the declared pullback
            """
        )
        st.markdown(
            """
            <div class="callout"><b>Keep the three pressure objects separate.</b><br>
            The C-J state is in the detonation products, the wall pressure is a coupled boundary
            estimate, and the field pressure is an empirical attenuation result. None is an
            automatic substitute for another.</div>
            """,
            unsafe_allow_html=True,
        )


def _animation_html() -> str:
    impact = float(st.session_state.impact_velocity)
    us = shock.shock_velocity_m_s
    up = shock.particle_velocity_m_s
    pressure = shock.pressure_pa / 1.0e9
    target_s = float(st.session_state.target_thickness) * 1.0e-3 / us
    return f"""
    <div id="stage" style="background:#111923;color:#eef3f8;border:1px solid #2c3a49;border-radius:6px;padding:12px;font-family:Segoe UI,Arial,sans-serif;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;gap:8px;flex-wrap:wrap;">
        <div><b>Planar impact sequence</b><span id="phase" style="color:#9fb0c2;margin-left:10px;">ready</span></div>
        <div>
          <button id="play" title="Play or pause" style="width:38px;height:30px;background:#1769aa;color:white;border:0;border-radius:4px;">&#9654;</button>
          <button id="reset" title="Reset" style="width:38px;height:30px;background:#344457;color:white;border:0;border-radius:4px;">&#8635;</button>
        </div>
      </div>
      <canvas id="canvas" width="1180" height="390" style="display:block;width:100%;max-width:100%;height:auto;background:#0d141d;border-radius:4px;"></canvas>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(125px,1fr));gap:8px;margin-top:9px;font-size:12px;">
        <div><span style="color:#9fb0c2">Impact velocity</span><br><b>{impact:.0f} m/s</b></div>
        <div><span style="color:#9fb0c2">Particle velocity</span><br><b>{up:.0f} m/s</b></div>
        <div><span style="color:#9fb0c2">Shock velocity</span><br><b>{us:.0f} m/s</b></div>
        <div><span style="color:#9fb0c2">Target transit</span><br><b>{target_s * 1e6:.2f} microseconds</b></div>
      </div>
    </div>
    <script>
    (() => {{
      const canvas = document.getElementById('canvas');
      const ctx = canvas.getContext('2d');
      const play = document.getElementById('play');
      const reset = document.getElementById('reset');
      const phaseLabel = document.getElementById('phase');
      let t = 0, running = false, last = 0;
      const impactP = {pressure:.2f};
      function line(x1,y1,x2,y2,color,w=2,dash=[]) {{
        ctx.beginPath(); ctx.setLineDash(dash); ctx.strokeStyle=color; ctx.lineWidth=w;
        ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke(); ctx.setLineDash([]);
      }}
      function text(s,x,y,color='#dce5ee',size=14,align='left') {{
        ctx.fillStyle=color; ctx.font=size+'px Segoe UI'; ctx.textAlign=align; ctx.fillText(s,x,y);
      }}
      function draw() {{
        ctx.clearRect(0,0,canvas.width,canvas.height);
        ctx.fillStyle='#0d141d'; ctx.fillRect(0,0,canvas.width,canvas.height);
        const contact=250, y=70, h=150, targetW=420, flyerW=150;
        const pre = Math.min(t/0.18,1), flyerX=45+(contact-flyerW-45)*pre;
        ctx.fillStyle='#74889d'; ctx.fillRect(flyerX,y,flyerW,h);
        ctx.strokeStyle='#b7c5d3'; ctx.strokeRect(flyerX,y,flyerW,h);
        ctx.fillStyle='#394b5d'; ctx.fillRect(contact,y,targetW,h);
        ctx.strokeStyle='#8295a8'; ctx.strokeRect(contact,y,targetW,h);
        text('FLYER',flyerX+flyerW/2,y+82,'#f4f7fa',15,'center');
        text('TARGET',contact+targetW/2,y+82,'#dfe8f0',15,'center');
        line(contact+targetW,y-10,contact+targetW,y+h+10,'#dce5ee',2);
        text('free surface',contact+targetW,y+h+30,'#9fb0c2',12,'right');

        if (t < 0.18) {{
          line(flyerX+35,y-25,flyerX+110,y-25,'#56b4e9',4);
          ctx.fillStyle='#56b4e9'; ctx.beginPath(); ctx.moveTo(flyerX+115,y-25);ctx.lineTo(flyerX+101,y-33);ctx.lineTo(flyerX+101,y-17);ctx.fill();
          phaseLabel.textContent='flyer approach';
        }} else {{
          const q=Math.min((t-0.18)/0.48,1);
          const shockX=contact+targetW*q;
          const precursorX=Math.min(contact+targetW, shockX+55*(1-q));
          ctx.fillStyle='rgba(196,61,75,0.34)';ctx.fillRect(contact,y,Math.max(shockX-contact,0),h);
          line(shockX,y,shockX,y+h,'#ff6877',5);
          if (precursorX < contact+targetW) line(precursorX,y,precursorX,y+h,'#f5b942',3,[7,5]);
          text('shock',shockX,y-13,'#ff8792',12,'center');
          if (q<1) text('elastic precursor',precursorX,y+h+20,'#f5c865',11,'center');
          if (t < 0.66) phaseLabel.textContent='compression and precursor';
          if (t >= 0.66) {{
            const r=Math.min((t-0.66)/0.28,1);
            const releaseX=contact+targetW*(1-r);
            ctx.fillStyle='rgba(0,166,166,0.23)';ctx.fillRect(releaseX,y,contact+targetW-releaseX,h);
            line(releaseX,y,releaseX,y+h,'#3dd6cf',5);
            text('release',releaseX,y-13,'#74e5df',12,'center');
            phaseLabel.textContent = r < 0.62 ? 'free-surface release' : 'release interaction and possible spall';
          }}
        }}

        const gx=755, gy=62, gw=360, gh=180;
        line(gx,gy+gh,gx+gw,gy+gh,'#718399',1);line(gx,gy,gx,gy+gh,'#718399',1);
        text('idealised free-surface velocity',gx,gy-18,'#dce5ee',14);
        text('time',gx+gw,gy+gh+23,'#9fb0c2',12,'right');
        text('velocity',gx-8,gy+4,'#9fb0c2',12,'right');
        const pts=[[0,gy+gh],[0.18,gy+gh],[0.24,gy+gh*0.74],[0.38,gy+gh*0.74],[0.43,gy+gh*0.31],[0.66,gy+gh*0.31],[0.78,gy+gh*0.62],[0.88,gy+gh*0.54],[1,gy+gh*0.54]];
        ctx.beginPath();ctx.strokeStyle='#56b4e9';ctx.lineWidth=3;
        pts.forEach((p,i)=>{{const x=gx+p[0]*gw;const yy=p[1];if(i===0)ctx.moveTo(x,yy);else ctx.lineTo(x,yy);}});ctx.stroke();
        const markerX=gx+Math.min(t,1)*gw;line(markerX,gy,markerX,gy+gh,'#ffffff',1,[3,4]);
        text(impactP.toFixed(2)+' GPa shocked state',gx,gy+gh+52,'#ff8792',13);
        text('precursor  |  plateau  |  pullback',gx,gy+gh+73,'#9fb0c2',12);

        const tx=250, ty=300, tw=420;
        line(tx,ty,tx+tw,ty,'#718399',1);line(tx,ty,tx,ty+65,'#718399',1);
        text('x-t wave paths',tx,ty-16,'#dce5ee',14);
        line(tx,ty+65,tx+tw,ty,'#ff6877',3);
        line(tx+tw,ty,tx+tw*0.45,ty+65,'#3dd6cf',3);
        line(tx,ty+65,tx+tw*0.30,ty,'#f5b942',2,[6,4]);
        text('position',tx+tw,ty+84,'#9fb0c2',12,'right');
        text('time',tx-8,ty+4,'#9fb0c2',12,'right');
      }}
      function frame(now) {{
        if (!last) last=now;
        if (running) {{ t += (now-last)/5200; if(t>=1){{t=1;running=false;play.innerHTML='&#9654;';}} }}
        last=now;draw();requestAnimationFrame(frame);
      }}
      play.onclick=()=>{{running=!running;play.innerHTML=running?'&#10074;&#10074;':'&#9654;';}};
      reset.onclick=()=>{{t=0;running=false;play.innerHTML='&#9654;';draw();}};
      draw();requestAnimationFrame(frame);
    }})();
    </script>
    """


def render_animation() -> None:
    st.subheader("Plate-impact wave sequence")
    st.caption(
        "The animation is schematic: its wave speeds and transit label use the current case, while amplitudes and trace shape remain idealised."
    )
    st.iframe(_animation_html(), height=570, width="stretch", tab_index=0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Interface velocity", f"{shock.particle_velocity_m_s:.1f} m/s")
    c2.metric("Shock velocity", f"{shock.shock_velocity_m_s:.1f} m/s")
    c3.metric("Target transit", f"{float(st.session_state.target_thickness) / shock.shock_velocity_m_s * 1e3:.2f} us")
    c4.metric("HEL check", "Exceeded" if shock.hel_exceeded else "Below HEL")
    st.markdown(
        r"""
        The front obeys mass and momentum conservation,
        $\rho_0 U_s=\rho(U_s-u_p)$ and $P-P_0=\rho_0U_su_p$.
        A free surface reflects compression as release; interacting releases can create the
        tensile pullback used for the first spall estimate.
        """
    )


def render_virtual_test() -> None:
    st.subheader("Virtual plate-impact test", anchor="virtual-plate-impact-test")
    st.caption(
        "Configure the flyer, target, thicknesses and impact velocity in the sidebar. "
        "The workspace evaluates impedance matching and constructs an idealised rear "
        "free-surface velocity trace for pre-test planning and data comparison."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Interface pressure", f"{shock.pressure_pa / 1e9:.3f} GPa")
    c2.metric("Shock arrival", f"{virtual_test['shock_arrival_s'] * 1e6:.3f} us")
    c3.metric("Nominal pulse", f"{virtual_test['pulse_duration_s'] * 1e6:.3f} us")
    c4.metric("Free-surface plateau", f"{virtual_test['plateau_velocity_m_s']:.1f} m/s")

    trace = pd.DataFrame(
        {
            "time_us": np.asarray(virtual_test["time_s"]) * 1.0e6,
            "free_surface_velocity_m_s": virtual_test["free_surface_velocity_m_s"],
        }
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trace["time_us"],
            y=trace["free_surface_velocity_m_s"],
            name="virtual free-surface velocity",
            line=dict(color=COLORS["blue"], width=3),
        )
    )
    for value, label, color, position in [
        (
            virtual_test["shock_arrival_s"] * 1.0e6,
            "shock arrival",
            COLORS["orange"],
            "top left",
        ),
        (
            virtual_test["release_time_s"] * 1.0e6,
            "release",
            COLORS["cyan"],
            "top left",
        ),
        (
            virtual_test["pullback_time_s"] * 1.0e6,
            "pull-back",
            COLORS["red"],
            "top right",
        ),
    ]:
        fig.add_vline(
            x=value,
            line_dash="dash",
            line_color=color,
            annotation_text=label,
            annotation_font_color=color,
            annotation_position=position,
        )
    fig.update_xaxes(title="Time after impact (microseconds)")
    fig.update_yaxes(title="Free-surface velocity (m/s)")
    fig.update_layout(title="Idealised virtual diagnostic history")
    _plot(fig, 500)

    inputs = pd.DataFrame(
        [
            ["Target", material.name, f"{float(st.session_state.target_thickness):.3f} mm"],
            ["Flyer", values["flyer"].name, f"{float(st.session_state.flyer_thickness):.3f} mm"],
            ["Impact", values["impact_mode"], f"{float(st.session_state.impact_velocity):.1f} m/s"],
            ["Pullback screen", "acoustic spall input", f"{float(st.session_state.pullback_velocity):.1f} m/s"],
        ],
        columns=["Object", "Definition", "Current value"],
    )
    st.dataframe(inputs, hide_index=True, width="stretch")

    st.download_button(
        "Download virtual velocity history",
        data=trace.to_csv(index=False),
        file_name="virtual_plate_impact_velocity.csv",
        mime="text/csv",
        width="stretch",
    )
    st.info(
        "The virtual trace uses a linear Us-up Hugoniot, one-dimensional impedance "
        "matching, a target shock-transit time, a flyer-duration estimate and the "
        "hydrodynamic free-surface approximation u_fs = 2 up. It does not resolve "
        "elastic-precursor evolution, a release isentrope, window corrections, "
        "strength-dependent wave interactions or two-dimensional edge release."
    )


def render_validation() -> None:
    st.subheader(
        "Experimental plate-impact comparison",
        anchor="experimental-plate-impact-comparison",
    )
    st.caption(
        "Upload a measured free-surface velocity record or measured Us-up points. "
        "The app compares them with the current established reduction; agreement is "
        "reported with metrics and does not automatically validate a complete material model."
    )

    trace_tab, hugoniot_tab = st.tabs(
        ["Free-surface velocity", "Hugoniot points"]
    )

    with trace_tab:
        trace_template = pd.DataFrame(
            {
                "time_us": [0.0, 0.5, 1.0],
                "free_surface_velocity_m_s": [0.0, 0.0, 0.0],
            }
        )
        st.download_button(
            "Download velocity template",
            data=trace_template.to_csv(index=False),
            file_name="plate_impact_velocity_template.csv",
            mime="text/csv",
            key="velocity_template",
        )
        uploaded = st.file_uploader(
            "Upload velocity history",
            type=["csv", "xlsx", "xls"],
            key="plate_velocity_upload",
        )
        if uploaded is not None:
            try:
                frame = _read_uploaded_table(uploaded)
                columns = list(frame.columns)
                if len(columns) < 2:
                    raise ValueError("the uploaded table needs at least two columns")
                c1, c2, c3, c4 = st.columns(4)
                time_column = c1.selectbox("Time column", columns, key="plate_time_column")
                velocity_column = c2.selectbox(
                    "Velocity column",
                    columns,
                    index=min(1, len(columns) - 1),
                    key="plate_velocity_column",
                )
                time_unit = c3.selectbox(
                    "Time unit", ["microseconds", "milliseconds", "seconds"],
                    key="plate_time_unit",
                )
                velocity_unit = c4.selectbox(
                    "Velocity unit", ["m/s", "mm/s", "km/s"],
                    key="plate_velocity_unit",
                )
                clean = _finite_numeric_frame(frame, [time_column, velocity_column])
                if len(clean) < 3:
                    raise ValueError("fewer than three finite rows remain after cleaning")
                time_factor = {
                    "microseconds": 1.0e-6,
                    "milliseconds": 1.0e-3,
                    "seconds": 1.0,
                }[time_unit]
                velocity_factor = {"m/s": 1.0, "mm/s": 1.0e-3, "km/s": 1.0e3}[velocity_unit]
                measured_time = clean[time_column].to_numpy(dtype=float) * time_factor
                measured_velocity = clean[velocity_column].to_numpy(dtype=float) * velocity_factor
                order = np.argsort(measured_time)
                measured_time = measured_time[order]
                measured_velocity = measured_velocity[order]
                unique_time, unique_indices = np.unique(measured_time, return_index=True)
                measured_time = unique_time
                measured_velocity = measured_velocity[unique_indices]

                baseline_correct = st.checkbox(
                    "Subtract the pre-arrival mean velocity",
                    value=True,
                    key="plate_baseline_correct",
                )
                align_arrival = st.checkbox(
                    "Align the first 10% rise with the predicted shock arrival",
                    value=False,
                    key="plate_align_arrival",
                )
                if baseline_correct:
                    baseline_count = max(3, min(len(measured_velocity) // 10, 50))
                    measured_velocity = measured_velocity - float(
                        np.mean(measured_velocity[:baseline_count])
                    )
                amplitude = float(np.max(measured_velocity) - np.min(measured_velocity))
                threshold = float(np.min(measured_velocity) + 0.10 * amplitude)
                crossings = (
                    np.flatnonzero(measured_velocity >= threshold)
                    if amplitude > 0.0
                    else np.asarray([], dtype=int)
                )
                measured_arrival_s = (
                    float(measured_time[crossings[0]])
                    if crossings.size
                    else float("nan")
                )
                timing_shift_s = 0.0
                if align_arrival and np.isfinite(measured_arrival_s):
                    timing_shift_s = (
                        float(virtual_test["shock_arrival_s"]) - measured_arrival_s
                    )
                    measured_time = measured_time + timing_shift_s

                metrics = compare_velocity_histories(
                    measured_time,
                    measured_velocity,
                    np.asarray(virtual_test["time_s"]),
                    np.asarray(virtual_test["free_surface_velocity_m_s"]),
                )
                predicted_at_measurement = np.interp(
                    measured_time,
                    np.asarray(virtual_test["time_s"]),
                    np.asarray(virtual_test["free_surface_velocity_m_s"]),
                    left=np.nan,
                    right=np.nan,
                )
                compare_frame = pd.DataFrame(
                    {
                        "time_us": measured_time * 1.0e6,
                        "measured_velocity_m_s": measured_velocity,
                        "virtual_velocity_m_s": predicted_at_measurement,
                    }
                )

                figure = go.Figure()
                figure.add_trace(
                    go.Scatter(
                        x=compare_frame["time_us"],
                        y=compare_frame["measured_velocity_m_s"],
                        name="measured",
                        line=dict(color=COLORS["ink"], width=2),
                    )
                )
                figure.add_trace(
                    go.Scatter(
                        x=np.asarray(virtual_test["time_s"]) * 1.0e6,
                        y=virtual_test["free_surface_velocity_m_s"],
                        name="virtual test",
                        line=dict(color=COLORS["blue"], width=3, dash="dash"),
                    )
                )
                figure.update_xaxes(title="Time after impact (microseconds)")
                figure.update_yaxes(title="Free-surface velocity (m/s)")
                figure.update_layout(title="Measured and virtual velocity histories")
                _plot(figure, 500)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("RMSE", f"{metrics['rmse_m_s']:.2f} m/s")
                m2.metric("NRMSE", f"{100.0 * metrics['nrmse']:.2f}%")
                m3.metric("Bias", f"{metrics['bias_m_s']:.2f} m/s")
                m4.metric("R-squared", f"{metrics['r_squared']:.3f}")

                plateau_window = (
                    (measured_time >= float(virtual_test["shock_arrival_s"] + virtual_test["rise_time_s"]))
                    & (measured_time <= float(virtual_test["release_time_s"]))
                )
                plateau = (
                    float(np.median(measured_velocity[plateau_window]))
                    if np.any(plateau_window)
                    else float(np.max(measured_velocity))
                )
                post_release = measured_velocity[
                    measured_time >= float(virtual_test["release_time_s"])
                ]
                pullback = (
                    max(plateau - float(np.min(post_release)), 0.0)
                    if post_release.size
                    else float("nan")
                )
                predicted_plateau = float(virtual_test["plateau_velocity_m_s"])
                predicted_pullback = float(virtual_test["pullback_velocity_m_s"])
                arrival_residual_us = (
                    (measured_arrival_s - float(virtual_test["shock_arrival_s"]))
                    * 1.0e6
                    if np.isfinite(measured_arrival_s)
                    else float("nan")
                )
                plateau_error_pct = (
                    100.0 * (plateau - predicted_plateau) / predicted_plateau
                    if abs(predicted_plateau) > 1.0e-12
                    else float("nan")
                )
                pullback_error = (
                    pullback - predicted_pullback
                    if np.isfinite(pullback)
                    else float("nan")
                )
                experimental_spall = (
                    spall_strength_pa(
                        material.density_kg_m3,
                        material.longitudinal_speed_m_s,
                        pullback,
                    )
                    if np.isfinite(pullback)
                    else float("nan")
                )
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Arrival residual", f"{arrival_residual_us:.3f} us")
                p2.metric("Applied time shift", f"{timing_shift_s * 1.0e6:.3f} us")
                p3.metric("Plateau error", f"{plateau_error_pct:.2f}%")
                p4.metric("Pullback error", f"{pullback_error:.2f} m/s")
                st.dataframe(
                    pd.DataFrame(
                        [
                            ["Measured plateau", plateau, "m/s"],
                            ["Measured pullback", pullback, "m/s"],
                            ["Acoustic spall estimate", experimental_spall / 1.0e6, "MPa"],
                            ["Compared samples", metrics["samples"], "-"],
                        ],
                        columns=["Reduced quantity", "Value", "Unit"],
                    ).style.format({"Value": "{:.5g}"}),
                    hide_index=True,
                    width="stretch",
                )
                st.download_button(
                    "Download processed comparison",
                    data=compare_frame.to_csv(index=False),
                    file_name="plate_impact_velocity_comparison.csv",
                    mime="text/csv",
                    width="stretch",
                )
                st.session_state["shock_velocity_validation"] = {
                    "kind": "free_surface_velocity",
                    "source_file": uploaded.name,
                    "metrics": metrics,
                    "baseline_corrected": baseline_correct,
                    "arrival_aligned": align_arrival,
                    "measured_arrival_s": measured_arrival_s,
                    "arrival_residual_s": arrival_residual_us * 1.0e-6,
                    "applied_timing_shift_s": timing_shift_s,
                    "measured_plateau_velocity_m_s": plateau,
                    "plateau_error_percent": plateau_error_pct,
                    "measured_pullback_velocity_m_s": pullback,
                    "pullback_error_m_s": pullback_error,
                    "acoustic_spall_strength_pa": experimental_spall,
                }
            except Exception as exc:  # noqa: BLE001 - user upload diagnostics
                st.error(f"Could not reduce the uploaded velocity history: {exc}")
        else:
            st.session_state.pop("shock_velocity_validation", None)

    with hugoniot_tab:
        hugoniot_template = pd.DataFrame(
            {
                "particle_velocity_m_s": [100.0, 300.0, 600.0],
                "shock_velocity_m_s": [4435.0, 4705.0, 5110.0],
                "pressure_GPa_optional": [1.20, 3.81, 8.28],
            }
        )
        st.download_button(
            "Download Hugoniot template",
            data=hugoniot_template.to_csv(index=False),
            file_name="hugoniot_points_template.csv",
            mime="text/csv",
            key="hugoniot_template",
        )
        uploaded_points = st.file_uploader(
            "Upload measured Hugoniot points",
            type=["csv", "xlsx", "xls"],
            key="hugoniot_points_upload",
        )
        if uploaded_points is not None:
            try:
                frame = _read_uploaded_table(uploaded_points)
                columns = list(frame.columns)
                if len(columns) < 2:
                    raise ValueError("the uploaded table needs at least two columns")
                c1, c2, c3, c4 = st.columns(4)
                up_column = c1.selectbox("Particle-velocity column", columns, key="hugoniot_up_column")
                us_column = c2.selectbox(
                    "Shock-velocity column",
                    columns,
                    index=min(1, len(columns) - 1),
                    key="hugoniot_us_column",
                )
                velocity_unit = c3.selectbox(
                    "Velocity units", ["m/s", "km/s"], key="hugoniot_velocity_unit"
                )
                pressure_options = ["No pressure column", *columns]
                pressure_index = next(
                    (
                        index
                        for index, column in enumerate(columns, start=1)
                        if "press" in str(column).lower()
                    ),
                    0,
                )
                pressure_column = c4.selectbox(
                    "Measured-pressure column",
                    pressure_options,
                    index=pressure_index,
                    key="hugoniot_pressure_column",
                )
                if up_column == us_column:
                    raise ValueError("particle and shock velocity must use different columns")
                selected_columns = [up_column, us_column]
                if pressure_column != "No pressure column":
                    if pressure_column in selected_columns:
                        raise ValueError("the measured-pressure column must be distinct")
                    selected_columns.append(pressure_column)
                    pressure_unit = st.selectbox(
                        "Measured-pressure units",
                        ["GPa", "MPa", "Pa"],
                        key="hugoniot_pressure_unit",
                    )
                else:
                    pressure_unit = None
                clean = _finite_numeric_frame(frame, selected_columns)
                factor = 1.0 if velocity_unit == "m/s" else 1.0e3
                up = clean[up_column].to_numpy(dtype=float) * factor
                us = clean[us_column].to_numpy(dtype=float) * factor
                measured_pressure = None
                if pressure_column != "No pressure column":
                    pressure_factor = {"GPa": 1.0e9, "MPa": 1.0e6, "Pa": 1.0}[pressure_unit]
                    measured_pressure = (
                        clean[pressure_column].to_numpy(dtype=float) * pressure_factor
                    )
                fit = fit_linear_hugoniot(up, us)
                declared = material.bulk_sound_speed_m_s + material.hugoniot_slope * up
                declared_residual = declared - us
                declared_rmse = float(np.sqrt(np.mean(declared_residual**2)))
                declared_denominator = float(np.sum((us - np.mean(us)) ** 2))
                declared_r2 = (
                    float(1.0 - np.sum(declared_residual**2) / declared_denominator)
                    if declared_denominator > 1.0e-20
                    else float("nan")
                )
                jump_pressure = material.density_kg_m3 * us * up
                declared_pressure = material.density_kg_m3 * declared * up
                pressure_metrics = None
                if measured_pressure is not None:
                    pressure_residual = jump_pressure - measured_pressure
                    pressure_rmse = float(np.sqrt(np.mean(pressure_residual**2)))
                    pressure_bias = float(np.mean(pressure_residual))
                    pressure_scale = float(
                        np.max(measured_pressure) - np.min(measured_pressure)
                    )
                    if pressure_scale <= 1.0e-12:
                        pressure_scale = max(float(np.max(np.abs(measured_pressure))), 1.0)
                    pressure_metrics = {
                        "jump_pressure_rmse_pa": pressure_rmse,
                        "jump_pressure_nrmse": pressure_rmse / pressure_scale,
                        "jump_pressure_bias_pa": pressure_bias,
                    }

                order = np.argsort(up)
                if measured_pressure is not None:
                    figure = make_subplots(
                        rows=1,
                        cols=2,
                        subplot_titles=("Linear Hugoniot", "Pressure consistency"),
                    )
                else:
                    figure = go.Figure()
                figure.add_trace(
                    go.Scatter(
                        x=up,
                        y=us,
                        mode="markers",
                        name="measured points",
                        marker=dict(size=10, color=COLORS["ink"]),
                    ),
                    row=1 if measured_pressure is not None else None,
                    col=1 if measured_pressure is not None else None,
                )
                figure.add_trace(
                    go.Scatter(
                        x=up[order],
                        y=declared[order],
                        name="current material record",
                        line=dict(color=COLORS["blue"], width=3),
                    ),
                    row=1 if measured_pressure is not None else None,
                    col=1 if measured_pressure is not None else None,
                )
                figure.add_trace(
                    go.Scatter(
                        x=up[order],
                        y=np.asarray(fit["predicted_shock_velocity_m_s"])[order],
                        name="least-squares fit",
                        line=dict(color=COLORS["orange"], width=2, dash="dash"),
                    ),
                    row=1 if measured_pressure is not None else None,
                    col=1 if measured_pressure is not None else None,
                )
                if measured_pressure is not None:
                    figure.add_trace(
                        go.Scatter(
                            x=up,
                            y=measured_pressure / 1.0e9,
                            mode="markers",
                            name="reported pressure",
                            marker=dict(size=10, color=COLORS["green"]),
                        ),
                        row=1,
                        col=2,
                    )
                    figure.add_trace(
                        go.Scatter(
                            x=up[order],
                            y=jump_pressure[order] / 1.0e9,
                            name="rho0 Us up",
                            line=dict(color=COLORS["red"], width=2),
                        ),
                        row=1,
                        col=2,
                    )
                    figure.update_xaxes(
                        title_text="Particle velocity up (m/s)", row=1, col=1
                    )
                    figure.update_yaxes(
                        title_text="Shock velocity Us (m/s)", row=1, col=1
                    )
                    figure.update_xaxes(
                        title_text="Particle velocity up (m/s)", row=1, col=2
                    )
                    figure.update_yaxes(title_text="Pressure (GPa)", row=1, col=2)
                else:
                    figure.update_xaxes(title="Particle velocity up (m/s)")
                    figure.update_yaxes(title="Shock velocity Us (m/s)")
                figure.update_layout(title="Measured, declared and fitted linear Hugoniot")
                _plot(figure, 490)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Fitted c0", f"{fit['intercept_m_s']:.1f} m/s")
                m2.metric("Fitted slope s", f"{fit['slope']:.4f}")
                m3.metric("Declared RMSE", f"{declared_rmse:.1f} m/s")
                m4.metric("Declared R-squared", f"{declared_r2:.3f}")
                if pressure_metrics is not None:
                    p1, p2, p3 = st.columns(3)
                    p1.metric(
                        "Jump-pressure RMSE",
                        f"{pressure_metrics['jump_pressure_rmse_pa'] / 1.0e9:.4f} GPa",
                    )
                    p2.metric(
                        "Jump-pressure NRMSE",
                        f"{100.0 * pressure_metrics['jump_pressure_nrmse']:.2f}%",
                    )
                    p3.metric(
                        "Jump-pressure bias",
                        f"{pressure_metrics['jump_pressure_bias_pa'] / 1.0e9:.4f} GPa",
                    )
                comparison = pd.DataFrame(
                    {
                        "particle_velocity_m_s": up,
                        "measured_shock_velocity_m_s": us,
                        "declared_shock_velocity_m_s": declared,
                        "fitted_shock_velocity_m_s": fit["predicted_shock_velocity_m_s"],
                        "jump_pressure_GPa": jump_pressure / 1.0e9,
                        "declared_pressure_GPa": declared_pressure / 1.0e9,
                    }
                )
                if measured_pressure is not None:
                    comparison["measured_pressure_GPa"] = measured_pressure / 1.0e9
                st.download_button(
                    "Download Hugoniot comparison",
                    data=comparison.to_csv(index=False),
                    file_name="hugoniot_model_comparison.csv",
                    mime="text/csv",
                    width="stretch",
                )
                validation_record = {
                    "kind": "hugoniot_points",
                    "source_file": uploaded_points.name,
                    "fitted_intercept_m_s": fit["intercept_m_s"],
                    "fitted_slope": fit["slope"],
                    "fitted_rmse_m_s": fit["rmse_m_s"],
                    "fitted_r_squared": fit["r_squared"],
                    "declared_rmse_m_s": declared_rmse,
                    "declared_r_squared": declared_r2,
                }
                if pressure_metrics is not None:
                    validation_record.update(pressure_metrics)
                    validation_record["measured_pressure_unit"] = pressure_unit
                st.session_state["shock_hugoniot_validation"] = validation_record
                st.warning(
                    "A good linear Us-up fit validates only the stated Hugoniot relation "
                    "over the uploaded interval. It does not validate release, strength, "
                    "damage, spall or a complete JH/HJC/RHT material card."
                )
            except Exception as exc:  # noqa: BLE001 - user upload diagnostics
                st.error(f"Could not reduce the uploaded Hugoniot points: {exc}")
        else:
            st.session_state.pop("shock_hugoniot_validation", None)


def render_shock() -> None:
    st.subheader("Velocity-to-shock reduction")
    primary_metrics = st.columns(3)
    primary_metrics[0].metric("up", f"{shock.particle_velocity_m_s:.1f} m/s")
    primary_metrics[1].metric("Us", f"{shock.shock_velocity_m_s:.1f} m/s")
    primary_metrics[2].metric("Pressure", f"{shock.pressure_pa / 1e9:.3f} GPa")
    secondary_metrics = st.columns(2)
    secondary_metrics[0].metric("rho/rho0", f"{shock.compression_ratio:.4f}")
    secondary_metrics[1].metric("Delta E", f"{shock.internal_energy_change_j_kg / 1e3:.2f} kJ/kg")

    maximum_up = max(1.25 * shock.particle_velocity_m_s, 1000.0)
    curve = hugoniot_curve(material, maximum_up)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Us-up Hugoniot", "Pressure-volume Hugoniot"))
    fig.add_trace(
        go.Scatter(x=curve["particle_velocity_m_s"], y=curve["shock_velocity_m_s"], name="linear Hugoniot", line=dict(color=COLORS["blue"], width=3)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=[shock.particle_velocity_m_s], y=[shock.shock_velocity_m_s], name="current state", mode="markers", marker=dict(size=11, color=COLORS["red"])),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=curve["specific_volume_ratio"], y=curve["pressure_pa"] / 1e9, name="Hugoniot locus", line=dict(color=COLORS["cyan"], width=3), showlegend=False),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=[shock.specific_volume_ratio], y=[shock.pressure_pa / 1e9], mode="markers", marker=dict(size=11, color=COLORS["red"]), showlegend=False),
        row=1, col=2,
    )
    fig.add_hline(y=material.hel_pa / 1e9, line_dash="dash", line_color=COLORS["orange"], annotation_text="HEL", row=1, col=2)
    fig.update_xaxes(title_text="Particle velocity up (m/s)", row=1, col=1)
    fig.update_yaxes(title_text="Shock velocity Us (m/s)", row=1, col=1)
    fig.update_xaxes(title_text="Specific-volume ratio V/V0", autorange="reversed", row=1, col=2)
    fig.update_yaxes(title_text="Pressure (GPa)", row=1, col=2)
    _plot(fig, 500)

    st.latex(r"U_s=c_0+s u_p,\qquad P-P_0=\rho_0 U_su_p,\qquad E-E_0=\tfrac12(P+P_0)(V_0-V)")
    st.info(
        "The curve is a Hugoniot locus of shocked end states, not an arbitrary loading or release path. "
        "The linear fit is only valid over its measured material and pressure interval."
    )


def render_cj() -> None:
    st.subheader("Chapman-Jouguet source screen")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("C-J pressure", f"{cj.pressure_pa / 1e9:.2f} GPa")
    c2.metric("Product particle speed", f"{cj.particle_velocity_m_s:.0f} m/s")
    c3.metric("Product density ratio", f"{cj.density_ratio:.3f}")
    c4.metric("Coupled wall screen", f"{values['coupled_wall_pressure_pa'] / 1e9:.2f} GPa")

    pressure_labels = ["C-J products", "User coupling", "Acoustic cross-check", "Plate-impact state"]
    pressure_values = [
        cj.pressure_pa / 1e9,
        values["coupled_wall_pressure_pa"] / 1e9,
        values["acoustic_wall_pressure_pa"] / 1e9,
        shock.pressure_pa / 1e9,
    ]
    fig = go.Figure(
        go.Bar(
            x=pressure_labels,
            y=pressure_values,
            marker_color=[COLORS["red"], COLORS["orange"], COLORS["cyan"], COLORS["blue"]],
            text=[f"{p:.2f}" for p in pressure_values],
            textposition="outside",
        )
    )
    fig.update_yaxes(title="Pressure (GPa)")
    fig.update_layout(title="Separate source, boundary and material pressure objects", showlegend=False)
    _plot(fig, 440)

    st.latex(r"u_{CJ}=\frac{D}{\gamma+1},\qquad P_{CJ}=\frac{\rho_eD^2}{\gamma+1},\qquad \frac{\rho_{CJ}}{\rho_e}=\frac{\gamma+1}{\gamma}")
    st.warning(
        "This is the ideal-gas strong-detonation C-J screen. Measured VOD plus a calibrated products EOS "
        "is required for a source history. The user coupling factor and the acoustic impedance result are "
        "two different estimates; neither is a measured borehole-wall pressure."
    )


def render_field() -> None:
    st.subheader("Scaled-distance pressure and vibration")
    distance = float(st.session_state.observation_distance)
    mass = float(st.session_state.charge_mass)
    d_min = max(0.05, min(distance / 20.0, 1.0))
    d_max = max(distance * 8.0, 100.0)
    distances = np.geomspace(d_min, d_max, 300)
    pressures = empirical_peak_pressure_pa(
        distances, mass,
        float(st.session_state.reference_pressure_mpa) * 1e6,
        float(st.session_state.reference_z),
        float(st.session_state.pressure_exponent),
        pressure_cap_pa=values["coupled_wall_pressure_pa"],
    )
    ppv = empirical_ppv_mm_s(
        distances, mass, float(st.session_state.ppv_k), float(st.session_state.ppv_n)
    )
    ppv_imp = impedance_ppv_mm_s(pressures, material.density_kg_m3, material.longitudinal_speed_m_s)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cubic-root Z", f"{values['z_cubic']:.3f} m/kg^(1/3)")
    c2.metric("Square-root SD", f"{values['sd_sqrt']:.3f} m/kg^(1/2)")
    c3.metric("Peak pressure", f"{values['field_pressure_pa'] / 1e6:.3f} MPa")
    c4.metric("PPV site law", f"{values['ppv_mm_s']:.2f} mm/s")

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Empirical peak pressure", "PPV estimates"))
    fig.add_trace(go.Scatter(x=distances, y=pressures / 1e6, name="scaled pressure", line=dict(color=COLORS["red"], width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[distance], y=[values["field_pressure_pa"] / 1e6], mode="markers", name="field point", marker=dict(size=11, color=COLORS["ink"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=distances, y=ppv, name="site PPV law", line=dict(color=COLORS["blue"], width=3)), row=1, col=2)
    fig.add_trace(go.Scatter(x=distances, y=ppv_imp, name="P/(rho c) cross-check", line=dict(color=COLORS["cyan"], width=2, dash="dash")), row=1, col=2)
    fig.add_trace(go.Scatter(x=[distance], y=[values["ppv_mm_s"]], mode="markers", name="selected PPV", marker=dict(size=11, color=COLORS["ink"])), row=1, col=2)
    fig.update_xaxes(type="log", title="Distance R (m)", row=1, col=1)
    fig.update_yaxes(type="log", title="Peak pressure (MPa)", row=1, col=1)
    fig.update_xaxes(type="log", title="Distance R (m)", row=1, col=2)
    fig.update_yaxes(type="log", title="PPV (mm/s)", row=1, col=2)
    _plot(fig, 500)

    st.latex(r"Z=\frac{R}{W^{1/3}},\quad P=P_{ref}(Z/Z_{ref})^{-m},\qquad SD=\frac{R}{\sqrt{W}},\quad PPV=K(SD)^{-n}")
    st.caption(
        "K and n are site-law coefficients tied to the stated unit convention. The impedance conversion is a plane-wave cross-check, not a second PPV prediction law."
    )


def render_models() -> None:
    st.subheader("Established constitutive-model scope")
    model_table = pd.DataFrame(
        [
            ["JH-2", "Competent-rock crushing and residual strength", "Us-up, HEL, pressure-dependent strength, rate and fragmentation", "Scalar damage; weak unloading, Lode and tensile-crack memory"],
            ["HJC", "Concrete-like compaction and penetration", "UCS, triaxial strength, crush/lock EOS and dynamic compression", "Single scalar surface; limited directional fracture"],
            ["TCK", "Tensile release, spall and microcrack growth", "KIc or Gc, flaw population, pullback and recovered cracks", "Needs a separate pressure-dependent compression/EOS block"],
            ["RHT", "Compression, cap and residual regimes", "Three meridians, rate, porous EOS, HEL, release and tensile data", "Large coupled card; local scalar damage and mesh sensitivity"],
        ],
        columns=["Model", "Strongest use", "Minimum evidence", "Main limitation"],
    )
    st.dataframe(model_table, hide_index=True, width="stretch")

    st.markdown("#### Classical normalized strength surfaces")
    p1, p2, p3 = st.columns(3)
    with p1:
        model_damage = st.slider(
            "Scalar damage D", 0.0, 1.0, 0.35, 0.05, key="classical_damage"
        )
    with p2:
        model_rate_ratio = st.number_input(
            "Normalized rate ratio", min_value=0.01, max_value=1.0e10,
            value=1.0e4, format="%.3g", key="classical_rate_ratio"
        )
    with p3:
        rht_lode_factor = st.slider(
            "RHT Lode factor R3", 0.5, 1.5, 1.0, 0.05, key="classical_lode"
        )

    pressure_star = np.linspace(0.0, 6.0, 360)
    jh2_curve = jh2_normalized_strength(
        pressure_star, float(model_damage), float(model_rate_ratio)
    )
    hjc_curve = hjc_normalized_strength(
        pressure_star, float(model_damage), float(model_rate_ratio)
    )
    rate_factor = max(0.05, 1.0 + 0.03 * np.log(float(model_rate_ratio)))
    rht_curve = rht_normalized_failure_strength(
        pressure_star, rate_factor, float(rht_lode_factor)
    )
    classical_fig = go.Figure()
    classical_fig.add_trace(
        go.Scatter(
            x=pressure_star, y=jh2_curve, name="JH-2",
            line=dict(color=COLORS["blue"], width=3)
        )
    )
    classical_fig.add_trace(
        go.Scatter(
            x=pressure_star, y=hjc_curve, name="HJC",
            line=dict(color=COLORS["orange"], width=3)
        )
    )
    classical_fig.add_trace(
        go.Scatter(
            x=pressure_star, y=rht_curve, name="RHT failure meridian",
            line=dict(color=COLORS["green"], width=3)
        )
    )
    classical_fig.update_xaxes(title="Normalized pressure p*")
    classical_fig.update_yaxes(title="Normalized differential strength")
    classical_fig.update_layout(
        title="Equation-family comparison with illustrative coefficients"
    )
    _plot(classical_fig, 450)
    st.latex(
        r"\sigma^*_{\mathrm{JH2}}=(1-D)\sigma_i^*+D\sigma_f^*,"
        r"\quad \sigma^*_{\mathrm{HJC}}=\min\{S_{\max},"
        r"[A(1-D)+B(p^*)^N][1+C\ln\dot\varepsilon^*]\}"
    )
    st.caption(
        "The curves expose equation structure only. TCK is not placed on this "
        "compressive pressure-envelope plot because its defining calculation is "
        "finite tensile crack nucleation and growth. Replace every illustrative "
        "coefficient with a version-specific, experimentally calibrated card."
    )

    st.info(
        "These curves expose established equation families with illustrative "
        "coefficients. Use Chapter 18 to identify the data sources, uncertainty, "
        "limitations and application scope required for a calibrated software card."
    )


def render_engineering() -> None:
    st.subheader("From source pressure to engineering evidence")
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Borehole radius (mm)", min_value=1.0, max_value=1000.0, step=1.0, key="borehole_radius_mm")
    with c2:
        st.number_input("Near-field radial exponent", min_value=0.2, max_value=5.0, step=0.05, key="radial_exponent")
    rb = float(st.session_state.borehole_radius_mm) / 1000.0
    exponent = float(st.session_state.radial_exponent)
    wall = values["coupled_wall_pressure_pa"]
    hel_radius = threshold_radius_m(rb, wall, material.hel_pa, exponent)
    crush_radius = threshold_radius_m(rb, wall, material.uniaxial_compressive_strength_pa, exponent)
    fracture_radius = threshold_radius_m(rb, wall, material.tensile_strength_pa, exponent)

    zone_table = pd.DataFrame(
        [
            ["Shock/compaction screen", "P >= HEL", hel_radius, hel_radius / rb, "EOS, HEL, compaction and release"],
            ["Crushing screen", "P >= UCS", crush_radius, crush_radius / rb, "Pressure-dependent strength and fragmentation"],
            ["Fracture screen", "P >= tensile strength", fracture_radius, fracture_radius / rb, "Directional damage, free surfaces and joints"],
        ],
        columns=["Screen", "Declared threshold", "Radius (m)", "Radius / rb", "Required model evidence"],
    )
    st.dataframe(zone_table.style.format({"Radius (m)": "{:.3f}", "Radius / rb": "{:.1f}"}), hide_index=True, width="stretch")

    theta = np.linspace(0, 2 * np.pi, 240)
    fig = go.Figure()
    for radius, label, color in [
        (fracture_radius, "fracture threshold", COLORS["orange"]),
        (crush_radius, "crushing threshold", COLORS["red"]),
        (hel_radius, "HEL threshold", COLORS["blue"]),
        (rb, "borehole", COLORS["ink"]),
    ]:
        fig.add_trace(go.Scatter(x=radius * np.cos(theta), y=radius * np.sin(theta), mode="lines", fill="toself" if label == "borehole" else None, name=label, line=dict(color=color, width=3)))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, title="y (m)")
    fig.update_xaxes(title="x (m)")
    fig.update_layout(title="Threshold radii from a declared radial pressure law")
    _plot(fig, 520)

    st.markdown(
        """
        <div class="callout"><b>These are pressure-threshold screens, not predicted excavation zones.</b><br>
        Burden, free surfaces, in-situ stress, joints, charge geometry, gas expansion and cumulative rounds
        can move or rotate the actual damage boundary. Chapter 20 supplies the excavation state and
        Chapter 22 supplies the field validation quantities.</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Measurement and validation handoff")
    st.dataframe(
        pd.DataFrame(
            [
                ["Near charge", "Pressure or stress gauges, VOD, recovered core/CT", "Arrival, wall history, crushing, compaction and radial cracks"],
                ["Excavation boundary", "High-speed imaging, DIC where feasible, profile and borehole imaging", "Release, overbreak, EDZ and directional fracture"],
                ["Far field", "Calibrated geophones or accelerometers", "Three-component PPV, frequency, arrival and attenuation"],
                ["Repeated rounds", "Survey, UPV, permeability, support instruments and event log", "Cumulative damage and operational consequence"],
            ],
            columns=["Region", "Evidence", "Comparison quantity"],
        ),
        hide_index=True,
        width="stretch",
    )


def _export_records() -> tuple[dict, pd.DataFrame]:
    record = {
        "schema": "shock-blast-workspace-v3",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "case_name": st.session_state.case_name,
        "material": asdict(material),
        "impact": {
            "mode": values["impact_mode"],
            "impact_velocity_m_s": float(st.session_state.impact_velocity),
            "flyer": asdict(values["flyer"]),
            "flyer_thickness_mm": float(st.session_state.flyer_thickness),
            "target_thickness_mm": float(st.session_state.target_thickness),
            "shock_state": shock.to_dict(),
            "pullback_velocity_m_s": float(st.session_state.pullback_velocity),
            "spall_strength_pa": values["spall_pa"],
        },
        "virtual_plate_impact": {
            "shock_arrival_s": float(virtual_test["shock_arrival_s"]),
            "pulse_duration_s": float(virtual_test["pulse_duration_s"]),
            "release_time_s": float(virtual_test["release_time_s"]),
            "plateau_velocity_m_s": float(virtual_test["plateau_velocity_m_s"]),
            "standing": "idealised one-dimensional planning and comparison trace",
        },
        "detonation_source": {
            "explosive_density_kg_m3": float(st.session_state.explosive_density),
            "product_gamma": float(st.session_state.product_gamma),
            "cj_state": cj.to_dict(),
            "coupling_factor": float(st.session_state.coupling_factor),
            "coupled_wall_pressure_pa": values["coupled_wall_pressure_pa"],
            "acoustic_transmission_crosscheck_pa": values["acoustic_wall_pressure_pa"],
        },
        "field_point": {
            "charge_mass_per_delay_kg": float(st.session_state.charge_mass),
            "distance_m": float(st.session_state.observation_distance),
            "cubic_root_scaled_distance": values["z_cubic"],
            "square_root_scaled_distance": values["sd_sqrt"],
            "pressure_pa": values["field_pressure_pa"],
            "ppv_site_law_mm_s": values["ppv_mm_s"],
            "ppv_impedance_crosscheck_mm_s": values["ppv_impedance_mm_s"],
            "pressure_regime": regime_label(values["field_pressure_pa"], material),
        },
        "assumptions": [
            "linear Us-up relation within a declared validity interval",
            "ideal-gas strong-detonation C-J source screen",
            "user-declared pressure coupling rather than measured wall pressure",
            "empirical cubic-root pressure and square-root PPV attenuation",
            "threshold radii are screening quantities rather than predicted damage boundaries",
        ],
    }
    validations = {}
    for name, state_key in [
        ("velocity_history", "shock_velocity_validation"),
        ("hugoniot_points", "shock_hugoniot_validation"),
    ]:
        validation = st.session_state.get(state_key)
        if isinstance(validation, dict):
            validations[name] = validation
    if validations:
        record["experimental_validation"] = validations
    rows = []
    for group in ["impact", "virtual_plate_impact", "detonation_source", "field_point"]:
        for key, value in record[group].items():
            if isinstance(value, (str, int, float, bool)):
                rows.append([group, key, value])
    return record, pd.DataFrame(rows, columns=["group", "quantity", "value"])


def render_summary() -> None:
    st.subheader("Auditable case summary")
    record, flat = _export_records()
    summary = pd.DataFrame(
        [
            ["Impact pressure", shock.pressure_pa / 1e9, "GPa", values["impact_mode"]],
            ["Shock arrival", virtual_test["shock_arrival_s"] * 1e6, "microseconds", "target thickness / Us"],
            ["Virtual free-surface plateau", virtual_test["plateau_velocity_m_s"], "m/s", "hydrodynamic 2 up screen"],
            ["HEL", material.hel_pa / 1e9, "GPa", "declared material input"],
            ["Spall strength", values["spall_pa"] / 1e6, "MPa", "acoustic pullback reduction"],
            ["C-J pressure", cj.pressure_pa / 1e9, "GPa", "ideal strong-detonation screen"],
            ["Coupled wall pressure", values["coupled_wall_pressure_pa"] / 1e9, "GPa", "user coupling factor"],
            ["Field pressure", values["field_pressure_pa"] / 1e6, "MPa", "cubic-root scaled-distance law"],
            ["Field PPV", values["ppv_mm_s"], "mm/s", "square-root site law"],
        ],
        columns=["Quantity", "Value", "Unit", "Reduction"],
    )
    st.dataframe(summary.style.format({"Value": "{:.4g}"}), hide_index=True, width="stretch")

    validations = record.get("experimental_validation", {})
    if isinstance(validations, dict) and validations:
        st.markdown("#### Experimental comparisons")
        validation_rows = []
        for comparison_name, validation in validations.items():
            if not isinstance(validation, dict):
                continue
            for key, value in validation.items():
                if key == "metrics" and isinstance(value, dict):
                    for metric, metric_value in value.items():
                        validation_rows.append(
                            [
                                comparison_name,
                                metric,
                                f"{metric_value:.6g}"
                                if isinstance(metric_value, (int, float))
                                else str(metric_value),
                            ]
                        )
                elif isinstance(value, (str, int, float, bool)):
                    validation_rows.append(
                        [
                            comparison_name,
                            key,
                            f"{value:.6g}"
                            if isinstance(value, (int, float)) and not isinstance(value, bool)
                            else str(value),
                        ]
                    )
        st.dataframe(
            pd.DataFrame(
                validation_rows,
                columns=["Comparison", "Quantity", "Value"],
            ),
            hide_index=True,
            width="stretch",
        )

    st.markdown("#### Validity statement")
    st.markdown(
        """
        - Use the plate-impact result only within the measured linear-Hugoniot interval and the declared impact geometry.
        - Replace the ideal C-J screen with measured VOD and a calibrated explosive-products EOS for simulation input.
        - Fit pressure and PPV laws to the site, charge convention, geology, sensor response and distance range actually used.
        - Calibrate EOS, strength, damage and fracture blocks with separate evidence; one pressure history does not identify the full material card.
        - Validate excavation outputs against withheld profile, EDZ, fragmentation, waveform and support-response observations.
        """
    )

    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download case JSON",
            data=json.dumps(record, indent=2),
            file_name="shock_blast_case.json",
            mime="application/json",
            width="stretch",
        )
    with right:
        st.download_button(
            "Download summary CSV",
            data=flat.to_csv(index=False),
            file_name="shock_blast_summary.csv",
            mime="text/csv",
            width="stretch",
        )


if page == "Case overview":
    render_overview()
elif page == "Plate-impact animation":
    render_animation()
elif page == "Virtual plate-impact test":
    render_virtual_test()
elif page == "Shock state and EOS":
    render_shock()
elif page == "Experimental validation":
    render_validation()
elif page == "C-J blast source":
    render_cj()
elif page == "Pressure and PPV":
    render_field()
elif page == "Established model scope":
    render_models()
elif page == "Engineering consequences":
    render_engineering()
else:
    render_summary()


st.divider()
st.caption(
    "Chapter links: Ch3 shock and detonation fundamentals | Ch7 shock/blast experiments | "
    "Ch13 fracture and cumulative damage | Ch14 constitutive models | Ch15 shock response | "
    "Ch20 excavation design | Ch22 blast damage and overbreak."
)
