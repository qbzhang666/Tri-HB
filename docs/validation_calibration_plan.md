# Tri-HB Digital Twin — Validation & Calibration Plan

**Purpose.** Turn the Virtual Tri-HB integrated app from a phenomenological
teaching/design tool into a *validated* digital twin suitable for a Q1
publication, by (i) calibrating every adjustable model parameter against
measured data and (ii) validating the calibrated model against **independent**
data it never saw during fitting.

This document maps each app output to a measurable experimental quantity,
defines a staged calibration sequence that respects the parameter hierarchy,
and sets quantitative acceptance criteria.

---

## 0. Validation philosophy (read first)

1. **Split the data.** Never fit and validate on the same tests. Reserve a
   *calibration set* (used to fit parameters) and a disjoint *validation set*
   (used only for blind prediction + error reporting). A reviewer will reject
   any "validation" that is actually a refit.
2. **Calibrate hierarchically.** Parameters are coupled. You cannot fit the
   damage kinetics before the elastic and strength parameters are fixed. Follow
   the stages below in order; freeze each stage's parameters before the next.
3. **Be explicit about what the model predicts vs. prescribes.** In the current
   code the Step 2–4 stress history is the *imposed pulse* added to prestress
   (`sx = sx0 + A·g(t-Δt)`), not a solved wave equation. Quantities downstream
   of that (stress path, η_sup, regime, spatial damage profile) are therefore
   **kinematic/heuristic estimates**, not field predictions. They can be
   validated only at that level, or the core must be upgraded (see §7).
4. **Report uncertainty.** Rock scatters. Every calibrated strength/energy point
   needs ≥3 (preferably 5) nominally identical repeats with mean ± std.

---

## 1. Parameter inventory

### 1a. Fixed / independently measured (NOT fitted)

| Symbol (code) | Meaning | How fixed |
|---|---|---|
| `bar_E`, `bar_C0` | bar modulus, wave speed | bar time-of-flight + density; manufacturer cert |
| `bar_area` | bar cross-section | direct measurement |
| `sigma_prop`, `sigma_yield` | bar elastic limit | 42CrMo spec / tensile test |
| `rho` (specimen) | density | mass / volume |
| `epsdot_ref` = 1e-5 | DIF reference rate | convention, held fixed |
| `specimen_size`, `specimen_length` | geometry | direct measurement |

### 1b. Calibrated (fitted to the calibration set)

| Symbol (code) | Meaning | Stage |
|---|---|---|
| `E_s` | dynamic Young's modulus | 1 |
| `sigma_c0` | unconfined dynamic strength anchor (UCS) | 1 |
| `nu` | Poisson's ratio | 1 |
| `eps_peak` | strain at peak stress | 1 |
| `b_rate` | DIF rate-sensitivity coefficient | 2 |
| `soften` | post-peak softening rate | 2 |
| residual floor (currently 0.05·σ_peak) | residual strength fraction | 2 |
| `k_conf` | confinement strengthening slope | 3 |
| `A_fail`, `B_fail`, `n_fail`, `lode_amp` | Step-3 failure surface q_f(p,θ) | 4 |
| `tau_D`, `alpha_sat`, `m_over`, `beta_rate`, `F0`, `epsdot0` | Step-4 damage kinetics | 5 |
| cap factor (currently `p_cap = 4·σ_c0`) | hydrostatic cap | 6 (Mode 6) |

### 1c. Validated (predicted, then compared — never fitted)

Peak strength at held-out rates/confinements; strain-rate history; absorbed
energy; post-test stiffness/wave-speed loss; failure mode; (if core upgraded)
spatial damage localisation.

> **Consistency flag — RESOLVED in code.** The app previously contained
> **two independent failure descriptions**: the Step-1 strength
> `σ_peak = (σ_c0 + k_conf·σ_conf)·DIF` (drives the σ–ε curve) and a Step-3
> surface with hardcoded constants `A=15, B=1.3, n=0.75` (drives the damage
> index F = q/q_f). These disagreed badly — at the Step-1 failure point the old
> surface gave F ≈ 2.6–3.5 instead of 1.0.
>
> The Step-3 surface is now **derived** from the Step-1 strength model, so the
> two are one consistent set. For triaxial compression (θ=0, h=1), eliminating
> the confining stress from `σ_1 = (σ_c0 + k_conf σ_conf)·DIF` gives a linear
> Mohr–Coulomb surface that reproduces Step-1 exactly:
>
>   `q_f = A + B·p`,  with  `A = 3·UCS_eff/(k_conf+2)`,  `B = 3(k_conf−1)/(k_conf+2)`,  `n = 1`,
>
> where `UCS_eff = UCS·DIF`. Verified: F = q/q_f = 1.000 at the Step-1 failure
> strength for all presets and confinements. The integrated app exports
> `material_k_conf`, `material_b_rate`, `material_epsdot_ref` from Step 1, and
> `wave_damage.py` derives `A_fail, B_fail, n_fail` from them (default; a
> "Use Step-1-consistent surface" checkbox in Advanced mode lets a user enter
> independently calibrated A, B, n instead). For a publication, calibration
> Stage 4 should still fit A, B, n, a_θ to measured true-triaxial data and
> confirm they remain consistent with the Stage 1–3 strength fits.

---

## 2. Experiment → app-output mapping

Each row: an app output, the measurable that validates it, and the instrument.

| App output (where) | Measurable experimental quantity | Instrument / method |
|---|---|---|
| Incident/reflected/transmitted bar strains εI, εR, εT (Step 1) | raw bar-gauge strain histories | strain gauges + high-speed DAQ (≥1 MHz) |
| Bar wave speed C₀ | flight time over known bar length | pulse echo / two-gauge timing |
| Pulse shape (half-sine assumption) | measured incident waveform after pulse shaper | incident-bar gauge |
| Specimen σ–ε curve, peak strength (Step 1) | reduced three-wave σ–ε | gauge reduction (the app's own reducer) |
| Strain-rate history ε̇(t) (Step 1) | reduced strain rate | gauge reduction / DIC |
| DIF vs strain rate | strength at multiple striker velocities | series of Mode 1/4 tests |
| Confined strength vs p_c (k_conf) | peak strength at multiple chamber pressures | Mode 2 series |
| Failure surface q_f(p,θ) (Step 3) | peak (p, q, θ) at failure, true-triaxial | Mode 3 series, unequal σ₂,σ₃ |
| Absorbed/dissipated energy W_diss (Step 4) | incident−reflected−transmitted energy balance; fragment surface energy | gauge energy + sieve/BET on fragments |
| Stiffness loss E(D), wave-speed loss c_p(D) (Step 4) | pre/post ultrasonic modulus & velocity on recovered specimen | ultrasonic pulse velocity (UPV) |
| Damage variable D | inferred from measured E-loss: D = 1 − E_post/E₀ | UPV / unloading modulus |
| Spatial damage profile, D_c, S_x (Step 4) | surface strain localisation; internal crack map | DIC (surface), µCT (internal) |
| Equilibrium window t_eq, η_sup, Δt* (Step 2) | front/back face force balance vs time; arrival timing | two-face gauges / DIC timing |

---

## 3. Staged calibration sequence

Freeze each stage before starting the next.

**Stage 0 — System characterisation (no rock).**
Measure C₀, gauge factors, and the *post-shaper* incident waveform. Confirm it
is half-sine (this directly validates the `gas_gun_pulse` half-sine model and
the dispersion assumption). Check `a/(c_φ·τ_r) ≲ 0.1`.

**Stage 1 — Static / lowest-rate anchors (Mode 1, low V).**
Fit `E_s` (elastic slope), `sigma_c0` (strength at the lowest achievable rate,
or quasi-static UCS as the DIF=1 anchor), `nu` (lateral DIC), `eps_peak`
(strain at peak).

**Stage 2 — Rate dependence (Mode 1 and/or Mode 4, sweep V or τ).**
Fit `b_rate` from the slope of strength vs log₁₀(ε̇/ε̇_ref). Fit `soften` from
the post-peak unloading branch and check the residual-strength fraction against
the hard-coded 5% floor (adjust if data demand). ≥4 rate levels, ≥3 repeats.

**Stage 3 — Confinement (Mode 2, sweep p_c).**
Fit `k_conf` from peak strength vs confining pressure at fixed rate. ≥4
pressure levels.

**Stage 4 — Failure surface (Mode 3, unequal σ₂,σ₃).**
Fit `A_fail, B_fail, n_fail, lode_amp` to measured failure points in p–q–θ
space. Resolve the two-surface consistency flag (§1c) here.

**Stage 5 — Damage kinetics (Mode 4/5 + post-test).**
With strength/surface frozen, fit `tau_D, alpha_sat, m_over, beta_rate, F0`
so the modelled `D(t)` reproduces measured end-state damage `D = 1 − E_post/E₀`
(UPV) across rate and overstress levels. `epsdot0` set to the series mean rate.

**Stage 6 — Cap / hydrostatic (Mode 6).**
Fit the cap factor (currently 4·σ_c0) to measured p–ε_v compaction if doing
near-hydrostatic loading.

---

## 4. Validation protocol (independent data)

For every claimed-validated quantity:

1. Choose validation tests **not** in any calibration stage (held-out rates,
   confinements, a different specimen batch, or a mode not used for fitting —
   e.g. calibrate on Modes 1–3, validate the EM Modes 4–5 predictions).
2. Run the app forward with the frozen calibrated parameters (blind).
3. Overlay prediction vs measurement and report the error metrics in §5.
4. Tabulate per-quantity pass/fail against §6 acceptance criteria.

---

## 5. Error metrics (report all)

- **Peak strength / peak q:** relative error `|pred−meas|/meas` (%).
- **Curves (σ–ε, ε̇(t), energy(t)):** normalised RMSE (NRMSE) and R².
- **DIF trend:** slope and intercept of pred vs meas, with R².
- **D / E-loss:** absolute error in D and in % modulus loss.
- **Across a series:** mean ± std of the relative error; report n (repeats).

---

## 6. Acceptance criteria (typical Q1 bar)

| Quantity | Target |
|---|---|
| Dynamic peak strength | within ±10–15% of mean measured |
| Strain rate (reduced) | within ±10% |
| DIF trend (b_rate) | R² ≥ 0.9 on calibration; trend reproduced on validation |
| Confined strength trend | within ±15% across p_c range |
| Absorbed/dissipated energy | within ±20% |
| Stiffness / wave-speed loss E(D), c_p(D) | within ±15% |
| Failure mode / Lode dependence | qualitatively correct ordering across modes |

(Tighten/loosen per target journal norms; state the chosen thresholds in the
paper's Methods.)

---

## 7. Outputs that cannot be validated without a code change

Be transparent about these — either upgrade the code or scope them out of the
validation claims:

| Output | Why not validatable now | Fix |
|---|---|---|
| Spatial damage profile D(x), `D_c`, `S_x` | synthetic sum of Gaussians (code labels it "heuristic / planning indicator") | compute a real damage field from a 1-D damage solver, or import the DIC/µCT field directly and compute D_c, S_x from *that* |
| Centre superposition η_sup, regime map, Δt* | kinematic timing estimates, no spatial wave solution | upgrade the Step-2 core to a 1-D wave solver, or present these explicitly as timing diagnostics only |
| Stress path as a "prediction" | stress = imposed pulse + prestress | either present as an *applied loading-path design* (legitimate, just framed correctly) or solve the coupled bar–specimen problem |

---

## 8. Suggested test matrix (minimum for a credible paper)

One rock (e.g. the sandstone preset rock), cubic 50 mm specimens, ≥3 repeats per cell.

| Mode | Series (vary) | Levels | Use |
|---|---|---|---|
| 1 (gas-gun) | striker velocity | 4 | calib Stage 1–2 |
| 2 (chamber) | confining pressure p_c | 4 | calib Stage 3 |
| 3 (true triaxial) | σ₂≠σ₃ combinations | 4–6 | calib Stage 4 |
| 4 (EM uniaxial) | pulse duration τ at fixed peak | 3 | **validation** (rate, blind) |
| 5 (EM async) | inter-axis delay Δt | 3 | **validation** (path dependence) |
| post-test | UPV + µCT/DIC on recovered specimens | all | D, E(D), localisation |

This gives calibration on Modes 1–3 and **blind validation on Modes 4–5**, which
is exactly the independent-prediction argument a Q1 reviewer wants.

---

## 9. What "done" looks like

- A parameter table with calibrated values + confidence intervals.
- Calibration-fit figures (per stage) with R²/NRMSE.
- Blind-prediction figures (Modes 4–5) with error bars and the §6 table.
- A reconciled single failure surface (consistency flag closed).
- A clear Methods statement of the prescribed-pulse assumption and its scope.
- Either real damage-field validation, or explicit scoping-out of the synthetic
  descriptors.

Meeting these makes the digital-twin (Route A) or software/methods (Route B)
paper defensible at Q1.
