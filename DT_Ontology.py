"""
Tunnel Digital Twin — Operator Dashboard
=========================================
Main Streamlit application entry point.

Run locally:
    streamlit run app.py

Deployed at:
    https://amandahuang-336-tunnel-dt2026-app-zmguxo.streamlit.app/

This is the human-facing interface for a serviceability-oriented
multimodal tunnel maintenance digital twin. It queries a populated
ontology (OWL/Turtle) through rdflib and presents defect records,
FMEA chains, and prescribed interventions.

Architecture:
    Protégé (authoring)
        │
        ▼  OWL/Turtle file
    ontology/tunnel_maintenance.ttl
        │
        ▼  rdflib loads into memory
    This Streamlit app
        │
        ├─ SPARQL queries on page load
        └─ Operator views defects, FMEA chains, interventions
"""

import sys
import streamlit as st

from utils.ontology_loader import load_ontology, load_defects
from utils.styling import apply_custom_css

# -----------------------------------------------------------------------------
# Page configuration — MUST be the first Streamlit call
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Tunnel DT — Transurban",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/amandahuang-336/tunnel-dt2026",
        "Report a bug": "https://github.com/amandahuang-336/tunnel-dt2026/issues",
        "About": (
            "Tunnel Digital Twin operator dashboard, accompanying the paper "
            "*Serviceability-oriented Multimodal Data Integration for Tunnel "
            "Maintenance Digital Twins in the Australian Context*."
        ),
    },
)

apply_custom_css()

# -----------------------------------------------------------------------------
# Session state initialisation
# -----------------------------------------------------------------------------
def init_session_state():
    """Initialise session state with sensible defaults."""
    defaults = {
        "ontology_loaded": False,
        "selected_defect_id": None,
        "current_tunnel": "Tunnel_A",
        "graph": None,
        "defects": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# -----------------------------------------------------------------------------
# Ontology loader — wrapped to handle errors gracefully
# -----------------------------------------------------------------------------
def ensure_ontology_loaded():
    """Load ontology and defect data into session state if not already done."""
    if st.session_state.ontology_loaded:
        return True

    try:
        with st.spinner("Loading ontology and defect data..."):
            graph = load_ontology()
            defects = load_defects(graph)
            st.session_state.graph = graph
            st.session_state.defects = defects
            st.session_state.ontology_loaded = True
        return True
    except Exception as e:
        st.error(f"Failed to load ontology: {e}")
        st.exception(e)
        return False


# -----------------------------------------------------------------------------
# Sidebar — global navigation, tunnel selector, ontology status
# -----------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🛣️ Tunnel DT")
        st.caption("Serviceability-Oriented Digital Twin")
        st.divider()

        tunnel_choice = st.selectbox(
            "Active tunnel",
            options=["Tunnel_A", "Tunnel_B", "Tunnel_C"],
            index=0,
            help=(
                "Select which tunnel's data to display. "
                "Tunnel A is fully populated; B and C are placeholders."
            ),
        )
        st.session_state.current_tunnel = tunnel_choice

        st.divider()

        # Ontology status indicator
        st.markdown("**Ontology status**")
        if st.session_state.ontology_loaded and st.session_state.graph is not None:
            n_triples = len(st.session_state.graph)
            n_defects = len(st.session_state.defects)
            st.success(f"✓ Loaded — {n_triples} triples")
            st.caption(f"{n_defects} defect instances")
        else:
            st.warning("Not loaded")

        st.divider()

        # Reload button (useful for testing)
        if st.button("Reload ontology", use_container_width=True):
            st.session_state.ontology_loaded = False
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

        st.divider()
        st.caption(
            "Built for the paper *Serviceability-oriented Multimodal "
            "Data Integration for Tunnel Maintenance Digital Twins in "
            "the Australian Context*."
        )

        # System info collapsed by default
        with st.expander("System info", expanded=False):
            st.caption(f"Python: `{sys.version.split()[0]}`")
            st.caption(f"Streamlit: `{st.__version__}`")


# -----------------------------------------------------------------------------
# Overview page content
# -----------------------------------------------------------------------------
def render_overview():
    st.title(f"{st.session_state.current_tunnel} — operational overview")
    st.caption(
        "Defects detected, prioritised, and traced to FMEA chains. "
        "Data as of inspection campaign 2024-03-15."
    )

    defects = st.session_state.defects
    if not defects:
        st.warning(
            "No defect data loaded. Check that "
            "`data/defects_tunnel_a.json` exists and is valid."
        )
        return

    # Filter by status
    active_defects = [d for d in defects if d.get("status") == "Active"]
    high_priority = [d for d in active_defects if d.get("priority") == "HIGH"]
    completeness_ok = [
        d for d in active_defects if d.get("completeness_score", 0) >= 0.75
    ]
    total_cost = sum(d.get("estimated_cost_aud", 0) for d in active_defects)

    # -----------------------------------------------------------------------------
    # Top metrics row
    # -----------------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Active defects",
            len(active_defects),
            help="Defects currently requiring tracking or intervention.",
        )
    with col2:
        st.metric(
            "High priority",
            len(high_priority),
            delta=f"{len(high_priority)} need attention" if high_priority else None,
            delta_color="inverse",
            help="Action required within 30 days.",
        )
    with col3:
        coverage_pct = (
            int(100 * len(completeness_ok) / len(active_defects))
            if active_defects
            else 0
        )
        st.metric(
            "FMEA coverage",
            f"{coverage_pct}%",
            help="Defects with diagnostic completeness ≥ 3/4.",
        )
    with col4:
        st.metric(
            "Est. cost",
            f"${total_cost / 1e6:.2f}M",
            help="Estimated intervention cost over the next 12 months.",
        )

    st.divider()

    # -----------------------------------------------------------------------------
    # Navigation hint
    # -----------------------------------------------------------------------------
    st.info(
        "**Navigate using the sidebar Pages menu:** "
        "**Defect Register** for a ranked list, "
        "**Defect Detail** for the full FMEA chain of a single defect, "
        "**SPARQL Console** for direct queries, "
        "**CV → COBie Bridge** for the defect semantic extraction pipeline, "
        "**Ontology Browser** to inspect the schema."
    )

    # -----------------------------------------------------------------------------
    # Section coverage
    # -----------------------------------------------------------------------------
    st.subheader("Multimodal coverage by tunnel section")

    sections = [
        ("Section 1 (K248+500 – K249+200)", 95, "🟢"),
        ("Section 2 (K249+200 – K249+900)", 78, "🟡"),
        ("Section 3 (K249+900 – K250+600)", 52, "🔴"),
        ("Section 4 (K250+600 – K251+300)", 67, "🟡"),
    ]

    for label, pct, icon in sections:
        col_a, col_b, col_c = st.columns([4, 8, 1])
        with col_a:
            st.write(f"{icon} {label}")
        with col_b:
            st.progress(pct / 100)
        with col_c:
            st.write(f"**{pct}%**")

    st.caption(
        "Coverage = percentage of ring assets with evidence from at least "
        "three of four modalities (RGB, RGBD, Thermal, GPR). Low-coverage "
        "sections are candidates for targeted follow-up survey."
    )

    st.divider()

    # -----------------------------------------------------------------------------
    # Top priority defects preview
    # -----------------------------------------------------------------------------
    st.subheader("Top priority defects")
    st.caption(
        "Ranked by condition state and diagnostic completeness. "
        "View the **Defect Register** for the full list, "
        "or the **Defect Detail** page for full FMEA chains."
    )

    top_defects = sorted(
        high_priority,
        key=lambda d: (
            -d.get("completeness_score", 0),
            d.get("discovered_on", ""),
        ),
    )[:5]

    if not top_defects:
        st.info("No high priority defects currently identified.")
        return

    for d in top_defects:
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([1.2, 4, 1, 1, 1])
            with col1:
                st.code(d["defect_id"], language=None)
            with col2:
                st.markdown(f"**{d['description']}**")
                st.caption(
                    f"Ring {d['ring_id']} · K{d['chainage_m']:.0f}m · {d['position']}"
                )
            with col3:
                score = d.get("completeness_score", 0)
                score_frac = f"{int(score * 4)}/4"
                st.metric(
                    "Completeness",
                    score_frac,
                    label_visibility="collapsed",
                )
            with col4:
                st.markdown(f":red[**{d.get('priority', '—')}**]")
            with col5:
                cost = d.get("estimated_cost_aud", 0)
                if cost:
                    st.write(f"${cost:,}")
                else:
                    st.write("—")


# -----------------------------------------------------------------------------
# Main flow — wrapped in try/except for graceful error handling
# -----------------------------------------------------------------------------
def main():
    render_sidebar()

    if not ensure_ontology_loaded():
        st.stop()

    render_overview()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        st.error("⚠️ The dashboard encountered an unexpected error.")
        st.exception(exc)
        st.markdown("---")
        st.caption(
            "If this persists, please report the issue at "
            "[GitHub Issues](https://github.com/amandahuang-336/tunnel-dt2026/issues)."
        )
else:
    # Streamlit imports the module; run main() directly
    try:
        main()
    except Exception as exc:
        st.error("⚠️ The dashboard encountered an unexpected error.")
        st.exception(exc)
