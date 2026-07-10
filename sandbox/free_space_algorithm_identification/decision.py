"""Dimensioned numerical bounds and fail-closed field decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np

from .metrics import (
    FrozenRoi,
    UnwrapStatus,
    _phase_residual_on_roi,
    compare_physical_fields,
)
from .models import PointField2D


ComparisonKind = Literal["zemax_candidate", "candidate_pair"]


@dataclass(frozen=True)
class MetricUncertainty:
    complex_distance: float
    phase_waves: float
    intensity_relative: float
    power_relative: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.complex_distance,
                self.phase_waves,
                self.intensity_relative,
                self.power_relative,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("metric uncertainty components must be finite")
        if np.any(values < 0.0):
            raise ValueError("metric uncertainty components must be non-negative")
        object.__setattr__(self, "complex_distance", float(self.complex_distance))
        object.__setattr__(self, "phase_waves", float(self.phase_waves))
        object.__setattr__(self, "intensity_relative", float(self.intensity_relative))
        object.__setattr__(self, "power_relative", float(self.power_relative))


class PairDecision(str, Enum):
    CONSISTENT = "consistent"
    UNDECIDED = "undecided"
    UNDECIDED_ROI_SENSITIVITY = "undecided_roi_sensitivity"
    EXCLUDED = "excluded"


@dataclass(frozen=True)
class S13S14StructureGate:
    pointwise_intensity_relative_max: float
    power_relative_error: float
    phase_max_abs_waves: float
    piston_rad: float
    unwrap_status: UnwrapStatus
    intensity: PairDecision
    power: PairDecision
    phase: PairDecision
    overall: PairDecision

    def __post_init__(self) -> None:
        non_phase = np.asarray(
            [
                self.pointwise_intensity_relative_max,
                self.power_relative_error,
                self.piston_rad,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(non_phase)) or np.any(non_phase[:2] < 0.0):
            raise ValueError("structural-gate metrics must be finite and non-negative")
        if self.unwrap_status == "ambiguous":
            if not math.isnan(self.phase_max_abs_waves):
                raise ValueError("ambiguous structural phase must be NaN")
            if self.phase is PairDecision.CONSISTENT:
                raise ValueError("ambiguous structural phase cannot pass")
        elif not math.isfinite(self.phase_max_abs_waves) or self.phase_max_abs_waves < 0.0:
            raise ValueError("resolved structural phase must be finite and non-negative")


def _require_uncertainty(
    value: MetricUncertainty | None, *, label: str
) -> MetricUncertainty:
    if not isinstance(value, MetricUncertainty):
        raise ValueError(f"{label} must be a MetricUncertainty")
    return value


def _sum_uncertainties(components: tuple[MetricUncertainty, ...]) -> MetricUncertainty:
    values = (
        math.fsum(component.complex_distance for component in components),
        math.fsum(component.phase_waves for component in components),
        math.fsum(component.intensity_relative for component in components),
        math.fsum(component.power_relative for component in components),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("combined metric uncertainty must be finite")
    return MetricUncertainty(
        complex_distance=max(1e-12, values[0]),
        phase_waves=max(1e-10, values[1]),
        intensity_relative=values[2],
        power_relative=values[3],
    )


def combine_uncertainty(
    *,
    comparison: ComparisonKind,
    common: MetricUncertainty,
    zemax: MetricUncertainty | None = None,
    candidate: MetricUncertainty | None = None,
    candidate_a: MetricUncertainty | None = None,
    candidate_b: MetricUncertainty | None = None,
) -> MetricUncertainty:
    """Add like dimensions while preserving the two physical budget paths."""

    common_value = _require_uncertainty(common, label="common uncertainty")
    if comparison == "zemax_candidate":
        if candidate_a is not None or candidate_b is not None:
            raise ValueError(
                "Zemax-candidate uncertainty cannot contain candidate-pair slots"
            )
        return _sum_uncertainties(
            (
                _require_uncertainty(zemax, label="Zemax uncertainty"),
                _require_uncertainty(candidate, label="candidate uncertainty"),
                common_value,
            )
        )
    if comparison == "candidate_pair":
        if zemax is not None:
            raise ValueError(
                "candidate-pair uncertainty must exclude Zemax repeat uncertainty"
            )
        if candidate is not None:
            raise ValueError(
                "candidate-pair uncertainty requires separate candidate A and B slots"
            )
        return _sum_uncertainties(
            (
                _require_uncertainty(candidate_a, label="candidate A uncertainty"),
                _require_uncertainty(candidate_b, label="candidate B uncertainty"),
                common_value,
            )
        )
    raise ValueError("unknown uncertainty comparison kind")


def classify_pair(distance: float, uncertainty: float) -> PairDecision:
    """Apply the exact closed 3u and 5u boundaries to one metric dimension."""

    if isinstance(distance, (bool, np.bool_)) or not math.isfinite(float(distance)):
        raise ValueError("distance must be finite and non-negative")
    if distance < 0.0:
        raise ValueError("distance must be finite and non-negative")
    if isinstance(uncertainty, (bool, np.bool_)) or not math.isfinite(
        float(uncertainty)
    ):
        raise ValueError("uncertainty must be finite and positive")
    if uncertainty <= 0.0:
        raise ValueError("uncertainty must be finite and positive")
    three_u = 3.0 * float(uncertainty)
    five_u = 5.0 * float(uncertainty)
    if not math.isfinite(three_u) or not math.isfinite(five_u):
        raise ValueError("scaled uncertainty must remain finite")
    if distance <= three_u:
        return PairDecision.CONSISTENT
    if distance <= five_u:
        return PairDecision.UNDECIDED
    return PairDecision.EXCLUDED


def resolve_roi_decisions(
    decisions: tuple[PairDecision, PairDecision, PairDecision],
) -> PairDecision:
    """Reject a classification whose support/exclusion class changes by ROI."""

    values = tuple(decisions)
    if len(values) != 3 or not all(isinstance(value, PairDecision) for value in values):
        raise ValueError("exactly three ROI PairDecision values are required")
    if any(value is PairDecision.UNDECIDED_ROI_SENSITIVITY for value in values):
        raise ValueError("ROI sensitivity cannot be used as an input ROI decision")
    if values[0] is values[1] is values[2]:
        return values[0]
    return PairDecision.UNDECIDED_ROI_SENSITIVITY


def s13_s14_r_phi_given_q_gate(
    r_phi_given_q: PointField2D,
    f_q: PointField2D,
    phi_minus_q_rad: np.ndarray,
    roi: FrozenRoi,
    *,
    uncertainty: MetricUncertainty,
    candidate_model: str,
) -> S13S14StructureGate:
    """Evaluate the mandatory S13-to-S14 identity for ``R_Phi_given_Q``."""

    if candidate_model != "R_Phi_given_Q":
        raise ValueError(
            "the S13-to-S14 identity applies only to R_Phi_given_Q"
        )
    bounds = _require_uncertainty(uncertainty, label="structural uncertainty")
    phase_delta = np.asarray(phi_minus_q_rad, dtype=np.float64)
    if phase_delta.shape != f_q.values.shape or not np.all(np.isfinite(phase_delta)):
        raise ValueError("Phi14-Q14 must be a finite array matching the physical field")

    analytic_target = PointField2D(
        f_q.values * np.exp(1j * phase_delta), f_q.grid
    )
    metrics = compare_physical_fields(r_phi_given_q, analytic_target, roi)

    f_q_intensity = np.abs(f_q.values) ** 2
    r_intensity = np.abs(r_phi_given_q.values) ** 2
    peak_intensity = float(np.max(f_q_intensity))
    if peak_intensity <= 0.0 or not math.isfinite(peak_intensity):
        raise ValueError("F_Q must have finite nonzero intensity")
    pointwise_intensity_error = float(
        np.max(np.abs(r_intensity - f_q_intensity)) / peak_intensity
    )
    pixel_area = f_q.grid.pixel_area_mm2
    f_q_power = float(np.sum(f_q_intensity)) * pixel_area
    r_power = float(np.sum(r_intensity)) * pixel_area
    if f_q_power <= 0.0 or not math.isfinite(f_q_power):
        raise ValueError("F_Q must have finite nonzero power")
    power_error = abs(r_power - f_q_power) / f_q_power

    aligned = r_phi_given_q.values * np.exp(-1j * metrics.piston_rad)
    target_intensity = np.abs(analytic_target.values) ** 2
    unwrapped, unwrap_status = _phase_residual_on_roi(
        aligned, analytic_target.values, target_intensity, roi.mask
    )
    if unwrap_status == "ambiguous" or unwrapped is None:
        phase_error = float("nan")
        phase_decision = PairDecision.UNDECIDED
    else:
        phase_error = float(np.max(np.abs(unwrapped[roi.mask])) / (2.0 * np.pi))
        phase_decision = classify_pair(phase_error, bounds.phase_waves)

    intensity_decision = classify_pair(
        pointwise_intensity_error, bounds.intensity_relative
    )
    power_decision = classify_pair(power_error, bounds.power_relative)
    component_decisions = (intensity_decision, power_decision, phase_decision)
    if PairDecision.EXCLUDED in component_decisions:
        overall = PairDecision.EXCLUDED
    elif any(value is not PairDecision.CONSISTENT for value in component_decisions):
        overall = PairDecision.UNDECIDED
    else:
        overall = PairDecision.CONSISTENT

    return S13S14StructureGate(
        pointwise_intensity_relative_max=pointwise_intensity_error,
        power_relative_error=power_error,
        phase_max_abs_waves=phase_error,
        piston_rad=metrics.piston_rad,
        unwrap_status=unwrap_status,
        intensity=intensity_decision,
        power=power_decision,
        phase=phase_decision,
        overall=overall,
    )


__all__ = [
    "MetricUncertainty",
    "PairDecision",
    "S13S14StructureGate",
    "classify_pair",
    "combine_uncertainty",
    "resolve_roi_decisions",
    "s13_s14_r_phi_given_q_gate",
]
