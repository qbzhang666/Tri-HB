"""Shock-to-field calculations for the Chapter 17 Streamlit workspace.

The functions in this module are deliberately solver independent. They cover
one-dimensional Rankine--Hugoniot reduction, a screening-level ideal C-J state,
and empirical blast attenuation. They do not replace an explosive-products EOS,
a hydrocode calculation, or a site-calibrated blast design.

All calculations use SI units. Pressure is positive in compression.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class HugoniotMaterial:
    """Minimum material record for a linear Us-up Hugoniot."""

    name: str = "Competent brittle geomaterial"
    density_kg_m3: float = 2700.0
    bulk_sound_speed_m_s: float = 4300.0
    hugoniot_slope: float = 1.35
    longitudinal_speed_m_s: float = 5200.0
    hel_pa: float = 3.0e9
    tensile_strength_pa: float = 12.0e6
    uniaxial_compressive_strength_pa: float = 150.0e6

    def __post_init__(self) -> None:
        positive = {
            "density_kg_m3": self.density_kg_m3,
            "bulk_sound_speed_m_s": self.bulk_sound_speed_m_s,
            "hugoniot_slope": self.hugoniot_slope,
            "longitudinal_speed_m_s": self.longitudinal_speed_m_s,
            "hel_pa": self.hel_pa,
            "tensile_strength_pa": self.tensile_strength_pa,
            "uniaxial_compressive_strength_pa": self.uniaxial_compressive_strength_pa,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class ShockState:
    """State behind a single planar shock from the declared initial state."""

    particle_velocity_m_s: float
    shock_velocity_m_s: float
    pressure_pa: float
    density_kg_m3: float
    compression_ratio: float
    specific_volume_ratio: float
    internal_energy_change_j_kg: float
    hel_exceeded: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class CJState:
    """Ideal-gas strong-detonation C-J screening state."""

    detonation_velocity_m_s: float
    particle_velocity_m_s: float
    pressure_pa: float
    product_density_kg_m3: float
    density_ratio: float
    specific_volume_ratio: float
    sonic_speed_m_s: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def hugoniot_state(
    material: HugoniotMaterial,
    particle_velocity_m_s: float,
    initial_pressure_pa: float = 0.0,
) -> ShockState:
    """Evaluate a linear-Hugoniot state using the jump conditions."""

    up = float(particle_velocity_m_s)
    if up < 0.0:
        raise ValueError("particle_velocity_m_s cannot be negative")
    if initial_pressure_pa < 0.0:
        raise ValueError("initial_pressure_pa cannot be negative")

    us = material.bulk_sound_speed_m_s + material.hugoniot_slope * up
    if up >= us:
        raise ValueError("particle velocity must remain below shock velocity")
    pressure = initial_pressure_pa + material.density_kg_m3 * us * up
    compression_ratio = us / (us - up)
    density = material.density_kg_m3 * compression_ratio
    volume_ratio = 1.0 / compression_ratio
    delta_specific_volume = 1.0 / material.density_kg_m3 - 1.0 / density
    energy = 0.5 * (pressure + initial_pressure_pa) * delta_specific_volume
    return ShockState(
        particle_velocity_m_s=up,
        shock_velocity_m_s=us,
        pressure_pa=pressure,
        density_kg_m3=density,
        compression_ratio=compression_ratio,
        specific_volume_ratio=volume_ratio,
        internal_energy_change_j_kg=energy,
        hel_exceeded=pressure >= material.hel_pa,
    )


def symmetric_impact_state(
    material: HugoniotMaterial,
    impact_velocity_m_s: float,
) -> ShockState:
    """Evaluate identical flyer/target impact, for which up = Vimpact / 2."""

    if impact_velocity_m_s < 0.0:
        raise ValueError("impact_velocity_m_s cannot be negative")
    return hugoniot_state(material, 0.5 * impact_velocity_m_s)


def impedance_match(
    flyer: HugoniotMaterial,
    target: HugoniotMaterial,
    impact_velocity_m_s: float,
    iterations: int = 100,
) -> tuple[ShockState, ShockState]:
    """Match two linear Hugoniots at a planar flyer-target interface.

    The returned states are ``(flyer_state, target_state)``. The flyer particle
    velocity is expressed in the flyer frame as ``Vimpact - u_interface``.
    """

    impact_velocity = float(impact_velocity_m_s)
    if impact_velocity <= 0.0:
        raise ValueError("impact_velocity_m_s must be positive")
    if iterations < 20:
        raise ValueError("iterations must be at least 20")

    def pressure_difference(interface_velocity: float) -> float:
        target_pressure = hugoniot_state(target, interface_velocity).pressure_pa
        flyer_relative_up = impact_velocity - interface_velocity
        flyer_pressure = hugoniot_state(flyer, flyer_relative_up).pressure_pa
        return target_pressure - flyer_pressure

    lo, hi = 0.0, impact_velocity
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if pressure_difference(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    interface_velocity = 0.5 * (lo + hi)
    target_state = hugoniot_state(target, interface_velocity)
    flyer_state = hugoniot_state(flyer, impact_velocity - interface_velocity)
    return flyer_state, target_state


def hugoniot_curve(
    material: HugoniotMaterial,
    maximum_particle_velocity_m_s: float,
    points: int = 240,
) -> dict[str, np.ndarray]:
    """Return a linear-Hugoniot curve for plotting and export."""

    if maximum_particle_velocity_m_s <= 0.0:
        raise ValueError("maximum_particle_velocity_m_s must be positive")
    if points < 2:
        raise ValueError("points must be at least two")
    up = np.linspace(0.0, maximum_particle_velocity_m_s, points)
    us = material.bulk_sound_speed_m_s + material.hugoniot_slope * up
    pressure = material.density_kg_m3 * us * up
    density_ratio = us / (us - up)
    volume_ratio = 1.0 / density_ratio
    energy = 0.5 * pressure * (
        1.0 / material.density_kg_m3
        - 1.0 / (material.density_kg_m3 * density_ratio)
    )
    return {
        "particle_velocity_m_s": up,
        "shock_velocity_m_s": us,
        "pressure_pa": pressure,
        "density_ratio": density_ratio,
        "specific_volume_ratio": volume_ratio,
        "internal_energy_change_j_kg": energy,
    }


def virtual_free_surface_velocity_history(
    target_state: ShockState,
    flyer_state: ShockState,
    target_thickness_m: float,
    flyer_thickness_m: float,
    pullback_velocity_m_s: float = 0.0,
    points: int = 900,
) -> dict[str, np.ndarray | float]:
    """Construct an idealised plate-impact free-surface velocity history.

    The arrival time follows the target shock transit, the nominal loading
    duration follows a flyer round trip, and the hydrodynamic free-surface
    plateau is twice the particle velocity. The trace is a transparent
    virtual-test screen rather than a characteristic or hydrocode solution.
    """

    if target_thickness_m <= 0.0 or flyer_thickness_m <= 0.0:
        raise ValueError("target and flyer thicknesses must be positive")
    if pullback_velocity_m_s < 0.0:
        raise ValueError("pullback_velocity_m_s cannot be negative")
    if points < 100:
        raise ValueError("points must be at least 100")

    shock_arrival_s = target_thickness_m / target_state.shock_velocity_m_s
    pulse_duration_s = 2.0 * flyer_thickness_m / flyer_state.shock_velocity_m_s
    pulse_duration_s = max(pulse_duration_s, shock_arrival_s * 0.08)
    rise_time_s = max(0.06 * pulse_duration_s, shock_arrival_s * 0.015)
    release_time_s = shock_arrival_s + pulse_duration_s
    pullback_time_s = release_time_s + 0.18 * pulse_duration_s
    rebound_time_s = pullback_time_s + 0.22 * pulse_duration_s
    end_time_s = rebound_time_s + 0.45 * pulse_duration_s

    time_s = np.linspace(0.0, end_time_s, points)
    plateau_velocity_m_s = 2.0 * target_state.particle_velocity_m_s
    pullback_level = max(plateau_velocity_m_s - pullback_velocity_m_s, 0.0)
    rebound_level = pullback_level + 0.45 * pullback_velocity_m_s

    def smoothstep(x: np.ndarray) -> np.ndarray:
        clipped = np.clip(x, 0.0, 1.0)
        return clipped * clipped * (3.0 - 2.0 * clipped)

    velocity = plateau_velocity_m_s * smoothstep(
        (time_s - shock_arrival_s) / rise_time_s
    )
    release_fraction = smoothstep(
        (time_s - release_time_s) / max(pullback_time_s - release_time_s, 1.0e-15)
    )
    velocity = velocity + (pullback_level - velocity) * release_fraction
    rebound_fraction = smoothstep(
        (time_s - pullback_time_s) / max(rebound_time_s - pullback_time_s, 1.0e-15)
    )
    velocity = velocity + (rebound_level - velocity) * rebound_fraction

    return {
        "time_s": time_s,
        "free_surface_velocity_m_s": velocity,
        "shock_arrival_s": shock_arrival_s,
        "pulse_duration_s": pulse_duration_s,
        "rise_time_s": rise_time_s,
        "release_time_s": release_time_s,
        "pullback_time_s": pullback_time_s,
        "plateau_velocity_m_s": plateau_velocity_m_s,
        "pullback_velocity_m_s": float(pullback_velocity_m_s),
    }


def compare_velocity_histories(
    measured_time_s: np.ndarray,
    measured_velocity_m_s: np.ndarray,
    predicted_time_s: np.ndarray,
    predicted_velocity_m_s: np.ndarray,
) -> dict[str, float | int]:
    """Compare a measured velocity history with a predicted history on overlap."""

    measured_time = np.asarray(measured_time_s, dtype=float)
    measured_velocity = np.asarray(measured_velocity_m_s, dtype=float)
    predicted_time = np.asarray(predicted_time_s, dtype=float)
    predicted_velocity = np.asarray(predicted_velocity_m_s, dtype=float)
    if measured_time.ndim != 1 or predicted_time.ndim != 1:
        raise ValueError("time arrays must be one-dimensional")
    if measured_time.size != measured_velocity.size:
        raise ValueError("measured time and velocity lengths must match")
    if predicted_time.size != predicted_velocity.size:
        raise ValueError("predicted time and velocity lengths must match")
    if measured_time.size < 3 or predicted_time.size < 3:
        raise ValueError("each history must contain at least three samples")
    if not (
        np.all(np.isfinite(measured_time))
        and np.all(np.isfinite(measured_velocity))
        and np.all(np.isfinite(predicted_time))
        and np.all(np.isfinite(predicted_velocity))
    ):
        raise ValueError("histories must contain only finite values")
    if np.any(np.diff(measured_time) <= 0.0) or np.any(np.diff(predicted_time) <= 0.0):
        raise ValueError("time arrays must be strictly increasing")

    start = max(float(measured_time[0]), float(predicted_time[0]))
    end = min(float(measured_time[-1]), float(predicted_time[-1]))
    mask = (measured_time >= start) & (measured_time <= end)
    if np.count_nonzero(mask) < 3:
        raise ValueError("histories do not have sufficient time overlap")
    measured = measured_velocity[mask]
    predicted = np.interp(measured_time[mask], predicted_time, predicted_velocity)
    residual = predicted - measured
    bias = float(np.mean(residual))
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    scale = float(np.max(measured) - np.min(measured))
    if scale <= 1.0e-12:
        scale = max(float(np.max(np.abs(measured))), 1.0e-12)
    nrmse = rmse / scale
    denominator = float(np.sum((measured - np.mean(measured)) ** 2))
    r_squared = (
        float(1.0 - np.sum(residual**2) / denominator)
        if denominator > 1.0e-20
        else float("nan")
    )
    return {
        "samples": int(measured.size),
        "overlap_start_s": start,
        "overlap_end_s": end,
        "bias_m_s": bias,
        "mae_m_s": mae,
        "rmse_m_s": rmse,
        "nrmse": nrmse,
        "r_squared": r_squared,
    }


def fit_linear_hugoniot(
    particle_velocity_m_s: np.ndarray,
    shock_velocity_m_s: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Fit a linear shock-velocity versus particle-velocity Hugoniot."""

    up = np.asarray(particle_velocity_m_s, dtype=float)
    us = np.asarray(shock_velocity_m_s, dtype=float)
    if up.ndim != 1 or us.ndim != 1 or up.size != us.size:
        raise ValueError("particle and shock velocity arrays must be equal-length vectors")
    if up.size < 2:
        raise ValueError("at least two Hugoniot points are required")
    if not np.all(np.isfinite(up)) or not np.all(np.isfinite(us)):
        raise ValueError("Hugoniot points must be finite")
    if np.any(up < 0.0) or np.any(us <= 0.0):
        raise ValueError("particle velocity cannot be negative and shock velocity must be positive")

    design = np.column_stack([np.ones_like(up), up])
    coefficients, *_ = np.linalg.lstsq(design, us, rcond=None)
    intercept, slope = map(float, coefficients)
    predicted = intercept + slope * up
    residual = predicted - us
    rmse = float(np.sqrt(np.mean(residual**2)))
    denominator = float(np.sum((us - np.mean(us)) ** 2))
    r_squared = (
        float(1.0 - np.sum(residual**2) / denominator)
        if denominator > 1.0e-20
        else float("nan")
    )
    return {
        "intercept_m_s": intercept,
        "slope": slope,
        "rmse_m_s": rmse,
        "r_squared": r_squared,
        "predicted_shock_velocity_m_s": predicted,
    }


def spall_strength_pa(
    density_kg_m3: float,
    longitudinal_speed_m_s: float,
    pullback_velocity_m_s: float,
) -> float:
    """First acoustic reduction of free-surface pullback to spall strength."""

    if density_kg_m3 <= 0.0 or longitudinal_speed_m_s <= 0.0:
        raise ValueError("density and longitudinal speed must be positive")
    if pullback_velocity_m_s < 0.0:
        raise ValueError("pullback velocity cannot be negative")
    return 0.5 * density_kg_m3 * longitudinal_speed_m_s * pullback_velocity_m_s


def cj_state(
    explosive_density_kg_m3: float,
    detonation_velocity_m_s: float,
    product_gamma: float,
) -> CJState:
    """Evaluate an ideal-gas, strong-detonation Chapman--Jouguet state.

    This closed form is intended for screening and teaching. Engineering source
    histories require measured VOD and a calibrated explosive-products EOS.
    """

    rho0 = float(explosive_density_kg_m3)
    velocity = float(detonation_velocity_m_s)
    gamma = float(product_gamma)
    if rho0 <= 0.0 or velocity <= 0.0:
        raise ValueError("explosive density and detonation velocity must be positive")
    if gamma <= 1.0:
        raise ValueError("product_gamma must be greater than one")

    particle_velocity = velocity / (gamma + 1.0)
    pressure = rho0 * velocity**2 / (gamma + 1.0)
    density_ratio = (gamma + 1.0) / gamma
    product_density = rho0 * density_ratio
    sonic_speed = velocity - particle_velocity
    return CJState(
        detonation_velocity_m_s=velocity,
        particle_velocity_m_s=particle_velocity,
        pressure_pa=pressure,
        product_density_kg_m3=product_density,
        density_ratio=density_ratio,
        specific_volume_ratio=1.0 / density_ratio,
        sonic_speed_m_s=sonic_speed,
    )


def cj_velocity_from_heat_release(heat_release_j_kg: float, product_gamma: float) -> float:
    """Strong-detonation estimate D_CJ = sqrt(2 (gamma^2 - 1) Q)."""

    if heat_release_j_kg <= 0.0:
        raise ValueError("heat_release_j_kg must be positive")
    if product_gamma <= 1.0:
        raise ValueError("product_gamma must be greater than one")
    return float(np.sqrt(2.0 * (product_gamma**2 - 1.0) * heat_release_j_kg))


def acoustic_pressure_transmission(
    incident_pressure_pa: float,
    source_impedance_pa_s_m: float,
    rock_impedance_pa_s_m: float,
) -> float:
    """Normal-incidence acoustic pressure transmission cross-check."""

    if incident_pressure_pa < 0.0:
        raise ValueError("incident pressure cannot be negative")
    if source_impedance_pa_s_m <= 0.0 or rock_impedance_pa_s_m <= 0.0:
        raise ValueError("impedances must be positive")
    coefficient = 2.0 * rock_impedance_pa_s_m / (
        source_impedance_pa_s_m + rock_impedance_pa_s_m
    )
    return incident_pressure_pa * coefficient


def cubic_root_scaled_distance(distance_m: np.ndarray | float, charge_mass_kg: float):
    """Hopkinson--Cranz scaled distance Z = R / W^(1/3)."""

    if charge_mass_kg <= 0.0:
        raise ValueError("charge_mass_kg must be positive")
    distance = np.asarray(distance_m, dtype=float)
    if np.any(distance <= 0.0):
        raise ValueError("distance_m must be positive")
    result = distance / charge_mass_kg ** (1.0 / 3.0)
    return float(result) if result.ndim == 0 else result


def square_root_scaled_distance(distance_m: np.ndarray | float, charge_mass_kg: float):
    """Square-root scaled distance SD = R / sqrt(W), commonly used for PPV."""

    if charge_mass_kg <= 0.0:
        raise ValueError("charge_mass_kg must be positive")
    distance = np.asarray(distance_m, dtype=float)
    if np.any(distance <= 0.0):
        raise ValueError("distance_m must be positive")
    result = distance / np.sqrt(charge_mass_kg)
    return float(result) if result.ndim == 0 else result


def empirical_peak_pressure_pa(
    distance_m: np.ndarray | float,
    charge_mass_kg: float,
    reference_pressure_pa: float,
    reference_scaled_distance: float,
    attenuation_exponent: float,
    pressure_cap_pa: float | None = None,
):
    """Power-law peak pressure on cubic-root scaled distance."""

    if reference_pressure_pa <= 0.0 or reference_scaled_distance <= 0.0:
        raise ValueError("reference pressure and scaled distance must be positive")
    if attenuation_exponent <= 0.0:
        raise ValueError("attenuation_exponent must be positive")
    z = cubic_root_scaled_distance(distance_m, charge_mass_kg)
    pressure = reference_pressure_pa * (
        np.asarray(z) / reference_scaled_distance
    ) ** (-attenuation_exponent)
    if pressure_cap_pa is not None:
        if pressure_cap_pa <= 0.0:
            raise ValueError("pressure_cap_pa must be positive")
        pressure = np.minimum(pressure, pressure_cap_pa)
    return float(pressure) if np.asarray(pressure).ndim == 0 else pressure


def empirical_ppv_mm_s(
    distance_m: np.ndarray | float,
    charge_mass_kg: float,
    site_constant: float,
    attenuation_exponent: float,
):
    """Site law PPV = K (R / sqrt(W))^(-n), returning millimetres/second."""

    if site_constant <= 0.0 or attenuation_exponent <= 0.0:
        raise ValueError("site constant and attenuation exponent must be positive")
    scaled = square_root_scaled_distance(distance_m, charge_mass_kg)
    ppv = site_constant * np.asarray(scaled) ** (-attenuation_exponent)
    return float(ppv) if np.asarray(ppv).ndim == 0 else ppv


def impedance_ppv_mm_s(
    pressure_pa: np.ndarray | float,
    density_kg_m3: float,
    wave_speed_m_s: float,
):
    """Plane-wave pressure/impedance cross-check, PPV = P/(rho c)."""

    if density_kg_m3 <= 0.0 or wave_speed_m_s <= 0.0:
        raise ValueError("density and wave speed must be positive")
    pressure = np.asarray(pressure_pa, dtype=float)
    if np.any(pressure < 0.0):
        raise ValueError("pressure cannot be negative")
    ppv = pressure / (density_kg_m3 * wave_speed_m_s) * 1000.0
    return float(ppv) if ppv.ndim == 0 else ppv


def threshold_radius_m(
    borehole_radius_m: float,
    wall_pressure_pa: float,
    threshold_pressure_pa: float,
    radial_attenuation_exponent: float,
) -> float:
    """Radius at which P = Pwall (rb / r)^m reaches a declared threshold."""

    if min(borehole_radius_m, wall_pressure_pa, threshold_pressure_pa) <= 0.0:
        raise ValueError("radius and pressures must be positive")
    if radial_attenuation_exponent <= 0.0:
        raise ValueError("radial_attenuation_exponent must be positive")
    ratio = max(wall_pressure_pa / threshold_pressure_pa, 1.0)
    return borehole_radius_m * ratio ** (1.0 / radial_attenuation_exponent)


def regime_label(pressure_pa: float, material: HugoniotMaterial) -> str:
    """Return a transparent pressure-based evidence regime label."""

    if pressure_pa < material.tensile_strength_pa:
        return "elastic or low-amplitude stress-wave range"
    if pressure_pa < material.uniaxial_compressive_strength_pa:
        return "fracture and directional-damage range"
    if pressure_pa < material.hel_pa:
        return "confined crushing below the HEL"
    return "shock compaction above the HEL"


def jh2_normalized_strength(
    pressure_star: np.ndarray | float,
    damage: float,
    rate_ratio: float,
    *,
    intact_coefficient: float = 0.8,
    fractured_coefficient: float = 0.45,
    rate_coefficient: float = 0.01,
    intact_exponent: float = 0.6,
    fractured_exponent: float = 0.8,
    tensile_pressure_star: float = 0.05,
):
    """JH-2 normalized pressure-rate-damage strength envelope."""

    if not 0.0 <= damage <= 1.0:
        raise ValueError("damage must lie in [0, 1]")
    if rate_ratio <= 0.0:
        raise ValueError("rate_ratio must be positive")
    pressure = np.asarray(pressure_star, dtype=float)
    if np.any(pressure < 0.0):
        raise ValueError("pressure_star cannot be negative")
    rate = max(0.0, 1.0 + rate_coefficient * np.log(rate_ratio))
    intact = intact_coefficient * np.maximum(
        pressure + tensile_pressure_star, 0.0
    ) ** intact_exponent * rate
    fractured = fractured_coefficient * pressure**fractured_exponent * rate
    strength = (1.0 - damage) * intact + damage * fractured
    return float(strength) if strength.ndim == 0 else strength


def hjc_normalized_strength(
    pressure_star: np.ndarray | float,
    damage: float,
    rate_ratio: float,
    *,
    cohesion_coefficient: float = 0.8,
    pressure_coefficient: float = 1.6,
    rate_coefficient: float = 0.007,
    pressure_exponent: float = 0.6,
    maximum_strength: float = 7.0,
):
    """HJC normalized strength with scalar damage and a strength cap."""

    if not 0.0 <= damage <= 1.0:
        raise ValueError("damage must lie in [0, 1]")
    if rate_ratio <= 0.0 or maximum_strength <= 0.0:
        raise ValueError("rate_ratio and maximum_strength must be positive")
    pressure = np.asarray(pressure_star, dtype=float)
    if np.any(pressure < 0.0):
        raise ValueError("pressure_star cannot be negative")
    rate = max(0.0, 1.0 + rate_coefficient * np.log(rate_ratio))
    strength = (
        cohesion_coefficient * (1.0 - damage)
        + pressure_coefficient * pressure**pressure_exponent
    ) * rate
    strength = np.minimum(strength, maximum_strength)
    return float(strength) if strength.ndim == 0 else strength


def rht_normalized_failure_strength(
    pressure_star: np.ndarray | float,
    rate_factor: float,
    lode_factor: float,
    *,
    failure_coefficient: float = 1.6,
    pressure_exponent: float = 0.65,
    tensile_intercept_star: float = -0.05,
):
    """RHT normalized failure meridian for a declared Lode and rate factor."""

    if rate_factor <= 0.0 or lode_factor <= 0.0:
        raise ValueError("rate_factor and lode_factor must be positive")
    pressure = np.asarray(pressure_star, dtype=float)
    if np.any(pressure < 0.0):
        raise ValueError("pressure_star cannot be negative")
    shifted_pressure = np.maximum(
        pressure - tensile_intercept_star * rate_factor, 0.0
    )
    strength = (
        failure_coefficient
        * shifted_pressure**pressure_exponent
        * lode_factor
        * rate_factor
    )
    return float(strength) if strength.ndim == 0 else strength
