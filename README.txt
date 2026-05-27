Tri-HB symmetric multi-axis waveform patch

Files:
- Tri-HB.py: patched simulator page.
- tri_hb_integrated.py: unchanged integrated shell; it will load the patched Tri-HB.py when both files are in the same directory.

Run standalone:
    streamlit run Tri-HB.py

Run integrated workspace:
    streamlit run tri_hb_integrated.py

Main changes:
- Symmetric XYZ Bar Waveforms tab uses a compact 9-trace view:
  3 incident pair traces (X, Y, Z) + 6 reflected bar traces (+/- X, +/- Y, +/- Z).
- Simulator output and CSV export now include explicit +/- incident and reflected arrays for X, Y and Z.
- Equations tab now gives X/Y/Z one-sided and symmetric dynamic loading equations, static + dynamic stress decomposition, and individual-bar plasticity bound.
