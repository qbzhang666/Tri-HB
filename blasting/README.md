# Tri-HB

Interactive computation workspaces for the handbook *Dynamic Testing and
Engineering Applications of Brittle Geomaterials*.

## Quick start

```powershell
py -3.13 -m pip install -r requirements.txt
py -3.13 -m streamlit run tri_hb_integrated.py
py -3.13 -m streamlit run shock_blast_integrated.py --server.port 8502
```

The first command opens the Hopkinson-bar workspace. The second opens the
shock, plate-impact and blast workspace. Streamlit prints the local browser URL.
See [`docs/local_setup.md`](docs/local_setup.md) for detailed setup and
alternate-port commands.

## Hopkinson-bar workspace

`tri_hb_integrated.py` keeps one case definition across:

- application and test-family selection;
- specimen, bar and loading setup;
- simulated or uploaded signal reduction;
- wave timing, equilibrium and superposition checks;
- stress-path and invariant interpretation;
- damage, energy and experimental/DEM descriptors;
- summary figures, database records and report sources.

The loading families include uniaxial and confined SHPB, gas-gun Tri-HB and
programmable electromagnetic multiaxial loading. The app is an analytical and
reporting companion to the handbook; it does not provide a new constitutive
material implementation.

## Shock and blast workspace

`shock_blast_integrated.py` supports:

- flyer-target selection and impedance-matched plate-impact states;
- an idealised virtual free-surface velocity history;
- uploaded velocity-history comparison with alignment and error metrics;
- uploaded Hugoniot fitting and comparison;
- EOS, HEL, release, spall and fragmentation reductions;
- C-J source screening, scaled-distance pressure and site PPV;
- established-model scope guidance and engineering consequence screens;
- machine-readable case and comparison exports.

The solver-independent equations are in `models/shock_blast.py`. Calculations
are screening and reduction tools; the interface states where a hydrocode,
site-calibrated field model or additional experiment is required.

## Repository layout

```text
.
|-- tri_hb_integrated.py       # Hopkinson-bar Streamlit entry point
|-- shock_blast_integrated.py  # shock, plate-impact and blast entry point
|-- models/shock_blast.py      # solver-independent shock/blast reductions
|-- tests/test_shock_blast.py  # automated checks for the reduction core
|-- Tri-HB.py                  # apparatus simulator loaded by the first app
|-- wave_damage.py             # wave, stress-path and damage calculations
|-- Springer_Handbook/         # Springer handbook source and build script
|-- docs/                      # setup and manuscript planning documents
|-- scripts/                   # reproducible figure utilities
|-- paper/                     # paper sources and research-only archives
|-- requirements.txt
`-- launch_8502.ps1
```

## Dependencies

The active workspaces use `streamlit`, `numpy`, `pandas`, `plotly`,
`matplotlib` and `openpyxl`. Install the pinned project environment from
`requirements.txt`.
