"""Shared comparison utilities for local POP and Zemax POP ZBF fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pop.io.zbf import (
    ZbfField,
    zbf_physical_field_pop_convention_for_axis,
    zbf_reference_phase,
    zbf_reference_relative_field_pop_convention,
)


@dataclass
class ComparisonResult:
    """Comparison metrics plus retained diagnostic arrays."""

    summary: dict[str, Any]
    fields: dict[str, np.ndarray]
    residuals: dict[str, np.ndarray]


def compare_pop_state_to_zbf(
    *,
    state: Any,
    pop_reference_relative: np.ndarray,
    pop_reference_phase: np.ndarray,
    zbf: ZbfField,
    surface_name: str,
    mask_threshold: float = 0.1,
    zbf_axis_sign: float = 1.0,
) -> ComparisonResult:
    """Compare paired POP physical phase with a ZBF physical field.

    ``pop_reference_relative`` and ``pop_reference_phase`` are the paired
    native field and lift recorded by POP.  The raw ZBF samples are first
    converted to POP phasor convention and then lifted with the ZBF header
    reference using the supplied local-axis sign.  No unrelated PROPER and
    ZBF reference phases are combined.
    """

    pop_field = np.asarray(pop_reference_relative, dtype=np.complex128)
    pop_ref_phase = np.asarray(pop_reference_phase, dtype=np.float64)
    zbf_field = zbf_reference_relative_field_pop_convention(zbf)
    if pop_field.shape != zbf_field.shape:
        raise ValueError(
            f"POP field shape {pop_field.shape} does not match ZBF field shape {zbf_field.shape}"
        )
    if pop_ref_phase.shape != pop_field.shape:
        raise ValueError(
            f"POP reference phase shape {pop_ref_phase.shape} does not match POP field shape {pop_field.shape}"
        )

    pop_physical = pop_field * np.exp(1j * pop_ref_phase)
    axis_sign = 1.0 if float(zbf_axis_sign) >= 0.0 else -1.0
    zbf_ref_phase = zbf_reference_phase(zbf)
    zbf_physical = zbf_physical_field_pop_convention_for_axis(
        zbf,
        axis_sign=axis_sign,
    )

    pop_amplitude = np.abs(pop_field)
    zbf_amplitude = zbf.amplitude
    pop_intensity = pop_amplitude**2
    zbf_intensity = zbf_amplitude**2
    pop_peak = _safe_peak(pop_intensity)
    zbf_peak = _safe_peak(zbf_intensity)
    pop_relative_intensity = pop_intensity / pop_peak
    zbf_relative_intensity = zbf_intensity / zbf_peak
    relative_intensity_residual = pop_relative_intensity - zbf_relative_intensity

    threshold = float(mask_threshold)
    mask = (pop_relative_intensity > threshold) & (zbf_relative_intensity > threshold)

    phase_residual = np.angle(pop_field * np.conj(zbf_field))
    phase_residual_no_piston, phase_piston = _remove_piston(phase_residual, mask)

    sampling_mm = float(getattr(getattr(state, "grid_sampling", None), "sampling_mm", zbf.dx))
    wavelength_um = float(
        getattr(getattr(state, "pilot_beam_params", None), "wavelength_um", zbf.wavelength * 1e3)
    )
    pop_pixel_area_mm2 = sampling_mm**2
    zbf_pixel_area_mm2 = float(zbf.dx) * float(zbf.dy)
    pop_energy = float(np.sum(pop_intensity) * pop_pixel_area_mm2)
    zbf_energy = float(np.sum(zbf_intensity) * zbf_pixel_area_mm2)
    phase_rms_rad = _masked_rms(phase_residual_no_piston, mask)
    phase_pv_rad = _masked_peak_to_valley(phase_residual_no_piston, mask)

    summary: dict[str, Any] = {
        "surface_index": int(getattr(state, "surface_index", -1)),
        "surface_name": str(surface_name),
        "pop_position": str(getattr(state, "position", "")),
        "zbf_path": "" if zbf.path is None else str(zbf.path),
        "grid_shape": [int(pop_field.shape[0]), int(pop_field.shape[1])],
        "pop_sampling_mm": sampling_mm,
        "zbf_dx_mm": float(zbf.dx),
        "zbf_dy_mm": float(zbf.dy),
        "pop_wavelength_mm": wavelength_um * 1e-3,
        "zbf_wavelength_mm": float(zbf.wavelength),
        "mask_threshold": threshold,
        "mask_pixels": int(np.count_nonzero(mask)),
        "pop_energy": pop_energy,
        "zbf_energy": zbf_energy,
        "pop_intensity_peak": float(np.max(pop_intensity)) if pop_intensity.size else 0.0,
        "zbf_intensity_peak": float(np.max(zbf_intensity)) if zbf_intensity.size else 0.0,
        "relative_intensity_rms": _masked_rms(relative_intensity_residual, mask),
        "relative_intensity_pv": _masked_peak_to_valley(relative_intensity_residual, mask),
        "phase_piston_rad": phase_piston,
        "phase_rms_rad": phase_rms_rad,
        "phase_pv_rad": phase_pv_rad,
        "phase_rms_waves": phase_rms_rad / (2.0 * np.pi),
        "phase_pv_waves": phase_pv_rad / (2.0 * np.pi),
        "phase_comparison_reference": "paired_physical_pop_vs_zbf",
        "zbf_axis_sign": axis_sign,
    }

    fields = {
        "pop_reference_relative": pop_field,
        "pop_reference_phase": pop_ref_phase,
        "pop_zbf_reference_relative": pop_physical * np.exp(-1j * axis_sign * zbf_ref_phase),
        "pop_physical": pop_physical,
        "zbf_raw_ex": np.asarray(zbf.ex, dtype=np.complex128),
        "zbf_reference_relative": zbf_field,
        "zemax_reference_phase": zbf_ref_phase,
        "zbf_physical": zbf_physical,
    }
    residuals = {
        "mask": mask,
        "pop_amplitude": pop_amplitude,
        "zbf_amplitude": zbf_amplitude,
        "pop_intensity": pop_intensity,
        "zbf_intensity": zbf_intensity,
        "pop_relative_intensity": pop_relative_intensity,
        "zbf_relative_intensity": zbf_relative_intensity,
        "relative_intensity_residual": relative_intensity_residual,
        "phase_residual_rad": phase_residual,
        "phase_residual_no_piston_rad": phase_residual_no_piston,
    }
    return ComparisonResult(summary=summary, fields=fields, residuals=residuals)


def _remove_piston(phase: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Remove circular piston from a wrapped phase residual."""

    phase = np.asarray(phase, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return phase.copy(), 0.0
    phasor_mean = np.mean(np.exp(1j * phase[mask]))
    piston = float(np.angle(phasor_mean)) if abs(phasor_mean) > 0 else 0.0
    corrected = np.angle(np.exp(1j * (phase - piston)))
    return corrected, piston


def _masked_rms(values: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return float("nan")
    data = np.asarray(values, dtype=np.float64)[mask]
    return float(np.sqrt(np.mean(data**2)))


def _masked_peak_to_valley(values: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return float("nan")
    data = np.asarray(values, dtype=np.float64)[mask]
    return float(np.ptp(data))


def _safe_peak(values: np.ndarray) -> float:
    if values.size == 0:
        return 1.0
    peak = float(np.max(values))
    return peak if peak > 0.0 else 1.0
