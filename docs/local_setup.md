# Local setup — running the Tri-HB integrated app

The integrated app is the entry-point script `tri_hb_integrated.py`. It must be
run from a folder that also contains `Tri-HB.py` and `wave_damage.py`, because
those two files hold most of the app — the Step 1 simulator lives in
`Tri-HB.py`, and the Step 2/3/4 wave, stress-path, and damage pages live in
`wave_damage.py`. `tri_hb_integrated.py` loads them at runtime; on its own it
only provides the Overview page and the experimental data reducer. There are no
other runtime dependencies inside the repo.

## One-time install

Open PowerShell in the project folder and install the Python dependencies:

```powershell
cd "C:\Users\qianbinz\Monash Uni Enterprise Dropbox\Qianbing Zhang\00-GitHub\Tri-HB"
py -3.13 -m pip install -r requirements.txt
```

(Substitute `py -3.13` with whichever Python launcher you have. The app needs
Python ≥ 3.9.)

## Launch

Default port (8501):

```powershell
py -3.13 -m streamlit run tri_hb_integrated.py
```

A second instance on port 8502 (useful when port 8501 is busy) — either run
the included launcher,

```powershell
.\launch_8502.ps1
```

or invoke Streamlit directly:

```powershell
py -3.13 -m streamlit run tri_hb_integrated.py --server.port 8502
```

Then open the URL Streamlit prints (`http://localhost:8501` or `:8502`) in any
modern browser.

## Quick sanity checks

```powershell
py -3.13 --version
py -3.13 -m streamlit --version
```

## Files actually loaded at runtime

| File at repo root | Loaded by |
|---|---|
| `tri_hb_integrated.py` | the Streamlit entry point itself |
| `Tri-HB.py` | Step 1 → *Test design and simulator* |
| `wave_damage.py` | Step 2 → *Wave model* (and Steps 3, 4 with the same module re-rendered with a different tab order) |
| `requirements.txt` | the install step above |

Nothing else in the repo is touched by the running app. The `scripts/` folder
holds figure-export utilities for the `Handbook/` project; the `archive/`
folder holds older experimental copies that are kept for reference but are
not on any active code path.
