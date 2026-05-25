# Tri-HB

Integrated Streamlit workspace for Triaxial Hopkinson Bar test design,
experimental data analysis, stress-wave interpretation, and DEM-oriented
damage validation.

## Recommended app

Run the combined application:

```powershell
streamlit run tri_hb_integrated.py
```

The integrated app includes:

- Test design and simulator from `Tri-HB.py`
- Experimental CSV/XLSX data reduction to stress, strain, strain-rate and energy
- Stress waves, stress path, failure index and energy from `wave_superposition.py`
- Damage evolution, transition regimes and DEM/experimental descriptors from `wave_damage.py`

The original specialist apps are retained for traceability and can still be run
directly if needed.
