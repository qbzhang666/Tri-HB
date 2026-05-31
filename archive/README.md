# archive/

Older standalone scripts and experimental snapshots that are **not on the
active code path** of the integrated app. They are kept here so prior versions
remain easy to recover (`git log -- <path>` shows the history before the
relocation), but they are not loaded by `tri_hb_integrated.py`, by
`Tri-HB.py`, or by `wave_damage.py`, and they are not imported by anything in
`scripts/` or `Handbook/`.

| File | What it was |
|---|---|
| `Cumulative Damage.py` | Standalone Streamlit prototype for cumulative damage analysis |
| `wave_superposition.py` | Standalone Streamlit prototype for wave superposition |
| `Tri-HB_step2_synced.py` | Earlier snapshot of `Tri-HB.py` |
| `Tri-HB_symmetric_9_waveforms.py` | Earlier snapshot of `Tri-HB.py` (9-waveform symmetric view) |
| `tri_hb_integrated_step2_synced.py` | Earlier snapshot of `tri_hb_integrated.py` |
| `wave_damage_numpy2_fixed.py` | Earlier snapshot of `wave_damage.py` (NumPy 2 compatibility pass) |

If one of these grows back into something you want to ship, move it back to
the repo root with `git mv archive/<file> .`
