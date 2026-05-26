"""
Integrated Tri-HB Streamlit app.

Run with:
    streamlit run tri_hb_integrated.py

This file keeps the existing specialist apps available while adding one
navigation shell and an experimental SHPB/Tri-HB data-reduction workspace.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent


st.set_page_config(
    page_title="Tri-HB Integrated",
    page_icon="Tri-HB",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


def run_legacy_app(filename: str) -> None:
    """Execute an existing app inside this integrated shell."""
    path = APP_DIR / filename
    source = strip_page_config(path.read_text(encoding="utf-8"))
    globals_dict = {
        "__file__": str(path),
        "__name__": "__main__",
    }
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

    latest_result = st.session_state.get("tri_hb_latest_result")
    use_simulator_result = False
    if latest_result is not None:
        use_simulator_result = st.checkbox(
            "Use latest Test Design and Simulator result",
            value=True,
            help="Run Step 1 first, then return here to analyse/export the simulated stress-strain history directly.",
        )

    uploaded = st.file_uploader(
        "Upload CSV or Excel data",
        type=["csv", "xlsx", "xls"],
        help="Expected columns include time and incident/reflected/transmitted strains, or direct stress and strain.",
        disabled=use_simulator_result,
    )

    with st.sidebar:
        st.header("Experimental analysis")
        analysis_mode = st.radio(
            "Input data type",
            ["Bar strain gauges", "Direct stress-strain"],
            help="Use bar strain gauges for SHPB/Tri-HB reduction; use direct stress-strain for cleaned data.",
        )
        st.divider()
        st.subheader("Specimen and bars")
        bar_E_GPa = st.number_input("Bar Young's modulus, Eb (GPa)", value=210.0, min_value=1.0, step=5.0)
        bar_C0 = st.number_input("Bar wave speed, C0 (m/s)", value=5172.0, min_value=1000.0, step=50.0)
        bar_d_mm = st.number_input("Round bar diameter (mm)", value=50.0, min_value=1.0, step=1.0)
        square_bar = st.checkbox("Use square 50 x 50 mm bar area", value=True)
        specimen_side_mm = st.number_input("Specimen side/diameter (mm)", value=50.0, min_value=1.0, step=1.0)
        specimen_length_mm = st.number_input("Specimen length (mm)", value=50.0, min_value=1.0, step=1.0)
        specimen_shape = st.radio("Specimen area", ["Square/cube", "Circular cylinder"], horizontal=True)

    if use_simulator_result and latest_result is not None:
        time_s = latest_result["time"]
        eps_i_pos = latest_result["epsI_x_pos"]
        eps_i_neg = latest_result["epsI_x_neg"]
        eps_r_pos = latest_result["epsR_x_pos"]
        eps_t = latest_result["epsT_x"]
        symmetric_factor = 2.0 if np.any(np.abs(eps_i_neg) > 0.0) else 1.0
        Ab = (0.050 * 0.050) if square_bar else np.pi * (bar_d_mm * 1e-3 / 2.0) ** 2
        Eb = bar_E_GPa * 1e9
        energy_i = Ab * Eb * bar_C0 * cumulative_trapezoid(eps_i_pos**2 + eps_i_neg**2, time_s)
        energy_r = Ab * Eb * bar_C0 * cumulative_trapezoid(symmetric_factor * eps_r_pos**2, time_s)
        energy_t = Ab * Eb * bar_C0 * cumulative_trapezoid(eps_t**2, time_s)
        sim_cfg = st.session_state.get("tri_hb_latest_config", {})
        sim_size = float(sim_cfg.get("specimen_size", specimen_length_mm * 1e-3))
        sim_volume = sim_size**3
        absorbed = cumulative_trapezoid(latest_result["sig_x"] * latest_result["rate_x"] * sim_volume, time_s)
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
        st.success("Using the latest result from Test Design and Simulator.")
        df_raw = out.copy()
        analysis_mode = "Simulator result"
    elif uploaded is None:
        st.info("Upload an experimental CSV or Excel file to begin reduction.")
        example = pd.DataFrame(
            {
                "time_us": np.linspace(0, 250, 6),
                "eps_I_ue": [0, 50, 250, 120, 20, 0],
                "eps_R_ue": [0, -10, -80, -40, -5, 0],
                "eps_T_ue": [0, 20, 160, 90, 10, 0],
            }
        )
        st.dataframe(example, use_container_width=True)
        return
    else:
        df_raw = read_uploaded_table(uploaded)
        df_raw = df_raw.dropna(how="all")
        st.dataframe(df_raw.head(20), use_container_width=True)

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
                name="stress-strain",
                line=dict(width=3),
            )
        )
        fig.update_layout(
            height=390,
            xaxis_title="Strain (%)",
            yaxis_title="Stress (MPa)",
            margin=dict(l=55, r=20, t=15, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

    tabs = st.tabs(["Time histories", "Energy", "Export"])
    with tabs[0]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=out["time_us"], y=out["stress_MPa"], name="stress (MPa)"))
        fig.add_trace(go.Scatter(x=out["time_us"], y=out["strain_rate_s-1"], name="strain rate (/s)", yaxis="y2"))
        fig.update_layout(
            height=430,
            xaxis_title="Time (us)",
            yaxis=dict(title="Stress (MPa)"),
            yaxis2=dict(title="Strain rate (/s)", overlaying="y", side="right"),
            margin=dict(l=55, r=55, t=20, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        if "energy_absorbed_J" not in out:
            st.info("Energy histories require incident, reflected and transmitted bar strain gauges.")
        else:
            fig = go.Figure()
            for col, label in [
                ("energy_incident_J", "Incident"),
                ("energy_reflected_J", "Reflected"),
                ("energy_transmitted_J", "Transmitted"),
                ("energy_absorbed_J", "Absorbed"),
            ]:
                fig.add_trace(go.Scatter(x=out["time_us"], y=out[col], name=label))
            fig.update_layout(
                height=430,
                xaxis_title="Time (us)",
                yaxis_title="Energy (J)",
                margin=dict(l=55, r=20, t=20, b=50),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.dataframe(out.head(50), use_container_width=True)
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


def overview_page() -> None:
    st.markdown("# Tri-HB Integrated Workspace")
    st.caption("Testing design, experimental data reduction, stress-wave interpretation, and DEM-oriented damage validation.")

    st.markdown(
        """
        Use the sidebar to move through the unified workflow:

        1. **Test design and simulator** keeps the developed Virtual Tri-HB app for loading-mode design and synthetic stress-strain curves.
        2. **Experimental data analysis** reduces bar-gauge or direct stress-strain files, or uses the latest Step 1 result directly.
        3. **Stress waves, stress path and energy** starts from the latest Step 1 geometry, prestress, pulse amplitude, duration and delays.
        4. **Damage evolution and DEM validation** starts from Step 1 and shows Step 2 reduced-data validation metrics when available.
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Design modes", "4")
    c2.metric("Integrated apps", "3")
    c3.metric("Experimental reducer", "CSV/XLSX")

    st.info(
        "The original files are retained for traceability. This integrated entry point is the recommended one to run."
    )


page = st.sidebar.radio(
    "Tri-HB workspace",
    [
        "Overview",
        "Test design and simulator",
        "Experimental data analysis",
        "Stress waves, stress path and energy",
        "Damage evolution and DEM validation",
    ],
)

if page == "Overview":
    overview_page()
elif page == "Test design and simulator":
    run_legacy_app("Tri-HB.py")
elif page == "Experimental data analysis":
    experimental_analysis_page()
elif page == "Stress waves, stress path and energy":
    run_legacy_app("wave_superposition.py")
else:
    run_legacy_app("wave_damage.py")
