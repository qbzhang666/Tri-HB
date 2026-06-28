# Tri-HB Computational Workspace — Validation & Calibration Plan

**Purpose.** Turn the Virtual Tri-HB integrated app from a phenomenological
teaching/design tool into a *validated* computational workspace suitable for a Q1
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

> **Pre-stress dependence — do not treat the specimen elastic parameters as
> single constants.** The *bar* properties (`bar_E`, `bar_C0`, `bar_area`) are
> pre-stress-independent to within a sub-percent acoustoelastic shift
> (`ΔC0/C0 ≈ (β/E_b)·σ_s ≲ 0.3%`), so the bar-based three-wave reduction is robust
> at every confinement. The *specimen* parameters are not: confinement closes
> microcracks, so `E_s = E_s(p)` and `c_p = c_p(p)` can change by tens of percent
> over the first tens of MPa. They must therefore be **calibrated at each
> pre-stress level** (or fitted with a pressure-dependent modulus law), because
> every specimen-property-based output — the equilibrium window
> `t_eq ∝ L_s/c_p`, the Δt* regime map, and the energy partition — inherits this
> dependence. Under true-triaxial pre-stress the closure is directional, so the
> stiffness becomes an anisotropic, stress-dependent tensor with
> `c_p,x ≠ c_p,y ≠ c_p,z` (stress-induced anisotropy); the three orthogonal bars
> can measure the directional `c_p,i` and `E_i` to calibrate it. See the paper
> §"Validity of the three-wave method under static pre-stress".

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

Meeting these makes the workspace (Route A) or software/methods (Route B)
paper defensible at Q1.

---

## 10. Novelty & publication strategy

This section is the honest answer to "is the damage model innovative / publishable?"
It is deliberately candid: the model's *equations* are largely standard, the
*publishable contribution* lives in the true-triaxial loading-path physics plus
experimental validation — not in the damage law by itself.

### 10.1 Honest novelty audit (damage law as coded)

| Component (code) | Form | Status / origin |
|---|---|---|
| Overstress damage rate `Ḋ ∝ ⟨(F_eff−1)/F0⟩^m` (`wave_damage.py`) | Perzyna overstress | Standard (Perzyna 1966) |
| Load-shedding `F_eff=(1−D)F`, dissipation `W_diss=∫Y dD`, `Y=W_el` | continuum damage mechanics | Standard (Lemaitre, Krajcinovic) |
| Rate factor `(ε̇_eq/ε̇0)^β` | power-law DIF on kinetics | Standard; cf. TCK 1986, Grady–Kipp |
| Failure surface `q_f=(A+Bpⁿ)h(θ)` | pressure-dependent, Lode-modified | Standard family (Willam–Warnke, RHT, JH-2, HJC) |
| Stiffness/wave-speed loss `E(1−D)`, `c_p√(1−D)` | isotropic damage degradation | Standard |

**Verdict.** As a *standalone constitutive innovation* this would not pass a Q1
review — a reviewer recognises a competent recombination of JH-2 / RHT /
Perzyna-CDM ideas. The novelty must come from elsewhere (10.3) and from
validation (10.4).

### 10.2 Modelling capabilities added to support a stronger claim

Two specific reviewer-bait weaknesses were fixed in code (`wave_damage.py`):

1. **Selectable true-triaxial deviatoric criterion** (`lode_shape()`):
   `legacy` (historical cap, kept for comparison), `lode` (linear,
   extension-weakened), `willam` (Willam–Warnke convex). The `lode`/`willam`
   forms use the extension/compression meridian ratio `ρ_t/ρ_c` and make the
   **extension meridian correctly weaker than compression** (the legacy
   `1+a_θ(1−cos3θ)` did the opposite). **Default is now `lode`, ρ_t/ρ_c = 0.70.**
   The intermediate principal stress (σ₂) effect therefore enters through the
   Lode angle with the physically correct sign.
2. **Rate-dependent failure envelope** (`env_dif_mode`): optional
   `DIF = 1 + b_env·log₁₀(max(ε̇_eq/ε̇0, 1))` applied to the cohesion `A`
   (`Cohesion (A)`) or to both `A` and `B` (`Full (A and B)`); `None` keeps the
   quasi-static surface. This lets the **failure onset** shift with strain rate,
   not just the post-onset damage speed (previously rate entered only `Ḋ`).

These do not by themselves make the model novel, but they remove two obvious
objections and make the calibration targets (ρ_t/ρ_c, b_env) measurable.

### 10.3 Where the genuine, defensible novelty is

1. **True-triaxial dynamic loading-path & sequence effects.** Almost all SHPB
   damage work is 1-D or confined-axisymmetric (σ₂=σ₃). Independent X/Y/Z dynamic
   loading + a third-invariant (Lode) damage envelope + *asynchronous* multi-axis
   paths (the EM async modes, inter-axis delay Δt) is a real literature gap. The
   Monash Tri-HB is one of the few rigs that can *produce* σ₂≠σ₃ dynamic states —
   that capability is the contribution.
2. **Wave-interaction regime map.** `Δt* = delay/t_travel` classifying
   synchronous-superposition → reverberation → transitional → sequential/
   damage-memory (`wave_damage.py`). As a framework linking loading *timing* to
   damage *memory* in triaxial impact, this is moderately original if formalised
   and validated.
3. **Coupled Lode + rate-dependent envelope** calibrated and *blind-validated*
   against true-triaxial dynamic data — i.e. demonstrating that one parameter set
   predicts strength across (p, θ, ε̇) it was not fitted to.

### 10.4 The decisive requirement

None of 10.3 is publishable as *modelling* without experimental calibration and
**blind** prediction (§§3–4, 8). The single highest-leverage step remains:
calibrate on a subset of Monash Tri-HB tests, then blind-predict a held-out
loading path / rate. Until then the app is a phenomenological computational workspace and
design tool (legitimate, but a methods/instrument contribution — Route B below).

### 10.5 Calibrate-then-blind-predict protocol for the new envelope features

Extends Stages 4–5 (§3); freeze elastic/strength first.

| Parameter | Calibrate on | Blind-predict (held out) | Pass criterion |
|---|---|---|---|
| `ρ_t/ρ_c` (Lode) | Mode-3 points on compression & extension meridians | an intermediate σ₂ (true-triaxial) point not used in the fit | predicted `q_f` within ±10–15%; extension < compression ordering reproduced |
| `b_env` (envelope DIF) | peak strength vs log ε̇ at one confinement | peak strength at a held-out strain rate | DIF trend R²≥0.9 on calib; held-out onset within ±10–15% |
| joint (ρ, b_env, A, B) | Modes 1–3 | EM Modes 4–5 (rate + path) blind | §6 acceptance table |

Report whether the *rate-dependent envelope* (`Full`) outperforms the
quasi-static envelope (`None`) on the held-out set — that comparison is itself a
publishable result (does failure-onset rate-dependence matter beyond damage
kinetics?).

### 10.6 Target journals & framing

| Journal (Q1) | Framing that fits | Gating requirement |
|---|---|---|
| Int. J. Rock Mech. & Mining Sci. (IJRMMS) | true-triaxial dynamic strength/damage of rock | Mode-3 + blind validation |
| Rock Mechanics & Rock Engineering (RMRE) | loading-path/σ₂ effects on dynamic failure | experimental series + repeats |
| Int. J. Impact Engineering (IJIE) | rig + dynamic response under multiaxial impact | Stage-0 system characterisation + validation |
| Int. J. Mechanical Sciences / Eng. Fracture Mech. | the coupled Lode + rate envelope, validated | blind prediction across (p,θ,ε̇) |
| J. Mech. Phys. Solids (top) | only with a genuinely new theoretical claim | new mechanism + rigorous validation |

### 10.7 Recommended publication routes

| Route | What it is | Publishable now? | Gate |
|---|---|---|---|
| **A — Experimental** (strongest) | Monash Tri-HB tests on true-triaxial loading-path effects, this app as the reduction/interpretation framework | **Yes**, once tests exist | run §8 matrix |
| **B — Methods / instrument + computational workspace** | the rig + integrated digital workflow (reduction → wave → stress path → damage → report) | **Likely** | Stage-0 + a few validation shots |
| **C — Validated model** | the coupled Lode + rate-dependent damage model, blind-validated | **Eventually** | calibrate + blind-predict (§10.5) |

**Bottom line.** The standard pedigree of the damage law stops being a weakness
the moment predictive capability on a held-out triaxial loading path is shown —
at that point "physically-grounded and validated" is a strength, and Routes A/C
become defensible Q1 contributions.

---

## 11. Anisotropic (orthotropic) damage for blasting — Stage 1 and roadmap

Blasting-induced fracturing is **directional** (radial cracks + spall open
perpendicular to the local tension), so the stiffness and wave-speed loss differ
by direction. A scalar `D` cannot represent that. The app now offers an
**orthotropic damage tensor** as the constitutive basis for a blasting model, with
the Monash Tri-HB as the calibration platform.

### 11.1 What Stage 1 implements (in `wave_damage.py`, Advanced → "Damage representation")

A diagonal damage tensor `D = diag(D_x, D_y, D_z)` aligned with the bar axes (which
are the principal loading directions in the rig):

$$\dot D_i=\frac{(1-D_i)^{\alpha}}{\tau_D}\Big\langle\frac{(1-D_i)F_i-1}{F_0}\Big\rangle^{m}\Big(\frac{|\dot\varepsilon_{eq}|}{\dot\varepsilon_0}\Big)^{\beta},\qquad F_i=\frac{\langle-\varepsilon_i\rangle_+}{\varepsilon_{t0}},\quad \varepsilon_{t0}=\sigma_t/E_0$$

- **Tension-driven (unilateral in growth):** `D_i` grows only under directional
  extension `⟨−ε_i⟩₊` (none in pure compression), so cracks form normal to the
  axes that are in tension — e.g. axial X compression drives lateral `D_y, D_z`
  (axial splitting) with `D_x≈0`; confinement on an axis suppresses its damage.
- **Directional degradation (reported):** `E_i = E₀(1−D_i)`,
  `c_{p,i} = c_{p0}√(1−D_i)` — the per-axis quantities the three bars measure.
- **Reuses** the calibrated kinetics (`τ_D, α, m, β, F_0`); the only new parameter
  is the tensile strength `σ_t` (onset strain `ε_t0 = σ_t/E`).
- **Default stays isotropic** (scalar `D`) so existing results are unchanged;
  orthotropic is opt-in in Advanced.

**Verified:** isotropic mode reproduces the scalar model exactly (`D_x=D_y=D_z=D`),
and the energy balance closes to machine precision in both modes; a uniaxial
X-compression check gives `D_y,D_z ≈ 0.7–0.8` with `D_x ≈ 0` (correct axial
splitting) and the expected directional `E_i` loss.

### 11.2 Stage-1 scope / honest limitations

- The **energy partition** uses the scalar aggregate `D = max_i D_i` (keeps the
  first law thermodynamically consistent for any loading). A **full anisotropic
  energy partition** — each direction dissipating against its work-conjugate
  stress — is Stage 2.
- **Unilateral crack-closure** is implemented in *growth* (no compressive damage)
  but not yet as *stiffness recovery in compression* within the energy path
  (Stage 2; needs a tension/compression stress split to stay thermodynamically
  consistent).
- Bar-aligned (orthotropic) only; a **full 2nd-order damage tensor with
  principal-direction rotation** for general (rotating) blast stress paths is
  Stage 3.

### 11.3 How the Tri-HB calibrates the tensor (the publishable hook)

This is the decisive advantage: the three independent orthogonal bars measure
**directional** stress/strain, so each `D_i` is calibrated **component-by-component**
from the per-axis modulus / wave-speed loss — a uniaxial SHPB cannot do this.

| Tensor component | Calibrated from | Tri-HB measurement |
|---|---|---|
| `D_x` | axial modulus / wave-speed loss | X incident+transmission bars (E_x, c_{p,x}) |
| `D_y` | lateral-Y modulus / wave-speed loss | Y output bars (E_y, c_{p,y}) |
| `D_z` | lateral-Z modulus / wave-speed loss | Z output bars (E_z, c_{p,z}) |
| `σ_t`, onset | directional tensile/spall strength | low-confinement or reflected-tension shots |

Post-test UPV / unloading modulus along each axis on the recovered specimen gives
`D_i = 1 − E_{i,post}/E_0` directly.

### 11.4 Validation route for blasting

1. **Calibrate** `D_x, D_y, D_z` (and `σ_t`) on a Tri-HB series that produces
   different (σ₂, σ₃, ε̇) and hence different directional damage (§8 matrix).
2. **Blind-predict** the directional damage / fracture orientation on a held-out
   loading path, and check against the measured per-axis `E_i` loss + DIC/µCT
   crack maps.
3. **Apply** the calibrated tensor in a blast simulation (rotating, tensile-
   dominated stress paths) and validate the predicted fracture pattern / damage
   zone against field or plate-impact spall data.

Showing that one bar-aligned tensor, calibrated on Tri-HB, predicts blasting-style
directional fracture is the Route-A/C contribution — anisotropic dynamic damage
for rock blasting from true-triaxial Hopkinson-bar data.
