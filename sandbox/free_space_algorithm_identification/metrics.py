"""Frozen-region metrics for already reconstructed physical total fields.

This module deliberately contains no reference-wave or propagation logic.  Its
inputs are point samples of physical total complex fields on one exact grid.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .models import PointField2D, UniformGrid2D


UnwrapStatus = Literal["not_needed", "unique", "ambiguous"]

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TWO_PI = 2.0 * np.pi
_NEIGHBOURS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
_NEIGHBOURS_4 = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _readonly_boolean_mask(mask: object) -> np.ndarray:
    source = np.asarray(mask)
    if source.ndim != 2 or source.dtype != np.bool_:
        raise ValueError("ROI mask must be a two-dimensional Boolean array")
    contiguous = np.array(source, dtype=np.bool_, order="C", copy=True)
    immutable = np.frombuffer(
        contiguous.tobytes(order="C"), dtype=np.bool_
    ).reshape(contiguous.shape)
    immutable.setflags(write=False)
    return immutable


def _mask_sha256(mask: np.ndarray) -> str:
    canonical = np.asarray(mask, dtype=np.bool_, order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _grid_is_identical(left: UniformGrid2D, right: UniformGrid2D) -> bool:
    return bool(
        isinstance(left, UniformGrid2D)
        and isinstance(right, UniformGrid2D)
        and np.array_equal(left.x_mm, right.x_mm)
        and np.array_equal(left.y_mm, right.y_mm)
    )


def _component_from_seed(support: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    selected = np.zeros_like(support, dtype=bool)
    if not support[seed]:
        return selected
    queue: deque[tuple[int, int]] = deque([seed])
    selected[seed] = True
    ny, nx = support.shape
    while queue:
        row, column = queue.popleft()
        for drow, dcolumn in _NEIGHBOURS_8:
            neighbour_row = row + drow
            neighbour_column = column + dcolumn
            if (
                0 <= neighbour_row < ny
                and 0 <= neighbour_column < nx
                and support[neighbour_row, neighbour_column]
                and not selected[neighbour_row, neighbour_column]
            ):
                selected[neighbour_row, neighbour_column] = True
                queue.append((neighbour_row, neighbour_column))
    return selected


def _mask_is_eight_connected(mask: np.ndarray) -> bool:
    indices = np.argwhere(mask)
    if indices.size == 0:
        return False
    seed = tuple(int(value) for value in indices[0])
    return bool(np.array_equal(_component_from_seed(mask, seed), mask))


@dataclass(frozen=True)
class FrozenRoi:
    threshold: float
    mask: np.ndarray
    mask_sha256: str
    reference_zbf_sha256: str
    grid: UniformGrid2D

    def __post_init__(self) -> None:
        if isinstance(self.threshold, (bool, np.bool_)) or not math.isfinite(
            float(self.threshold)
        ):
            raise ValueError("ROI threshold must be finite and positive")
        threshold = float(self.threshold)
        if threshold <= 0.0:
            raise ValueError("ROI threshold must be finite and positive")
        if not isinstance(self.grid, UniformGrid2D):
            raise ValueError("ROI grid must be a UniformGrid2D")
        mask = _readonly_boolean_mask(self.mask)
        if mask.shape != (self.grid.ny, self.grid.nx):
            raise ValueError("ROI mask shape does not match its grid")
        if not _mask_is_eight_connected(mask):
            raise ValueError("ROI mask must be nonempty and eight-connected")
        mask_sha256 = _require_sha256(self.mask_sha256, label="ROI mask SHA-256")
        if mask_sha256 != _mask_sha256(mask):
            raise ValueError("ROI mask SHA-256 does not match the canonical mask")
        reference_sha256 = _require_sha256(
            self.reference_zbf_sha256, label="reference ZBF SHA-256"
        )
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "mask_sha256", mask_sha256)
        object.__setattr__(self, "reference_zbf_sha256", reference_sha256)


@dataclass(frozen=True)
class FrozenRoiSet:
    primary: FrozenRoi
    checks: tuple[FrozenRoi, FrozenRoi]

    def __post_init__(self) -> None:
        checks = tuple(self.checks)
        if not isinstance(self.primary, FrozenRoi) or len(checks) != 2 or not all(
            isinstance(roi, FrozenRoi) for roi in checks
        ):
            raise ValueError("a frozen ROI set requires one primary and two checks")
        rois = (self.primary, *checks)
        if len({roi.threshold for roi in rois}) != 3:
            raise ValueError("frozen ROI thresholds must be distinct")
        if any(
            roi.reference_zbf_sha256 != self.primary.reference_zbf_sha256
            for roi in checks
        ):
            raise ValueError("frozen ROIs must come from one endpoint ZBF")
        if any(not _grid_is_identical(roi.grid, self.primary.grid) for roi in checks):
            raise ValueError("frozen ROIs must use one identical grid")
        object.__setattr__(self, "checks", checks)

    @property
    def ordered(self) -> tuple[FrozenRoi, FrozenRoi, FrozenRoi]:
        return (self.primary, *self.checks)


@dataclass(frozen=True)
class ComparisonMetrics:
    threshold: float
    piston_rad: float
    complex_relative_l2: float
    symmetric_distance: float
    phase_rms_waves: float
    phase_pv_waves: float
    intensity_relative_l2: float
    full_window_power_relative_error: float
    outside_roi_energy_fraction: float
    coherence_magnitude: float
    unwrap_status: UnwrapStatus

    def __post_init__(self) -> None:
        if self.unwrap_status not in {"not_needed", "unique", "ambiguous"}:
            raise ValueError("unknown phase unwrap status")
        finite_values = (
            self.threshold,
            self.piston_rad,
            self.complex_relative_l2,
            self.symmetric_distance,
            self.intensity_relative_l2,
            self.full_window_power_relative_error,
            self.outside_roi_energy_fraction,
            self.coherence_magnitude,
        )
        if not np.all(np.isfinite(np.asarray(finite_values, dtype=np.float64))):
            raise ValueError("non-phase comparison metrics must be finite")
        if self.threshold <= 0.0 or any(
            value < 0.0
            for value in (
                self.complex_relative_l2,
                self.symmetric_distance,
                self.intensity_relative_l2,
                self.full_window_power_relative_error,
                self.outside_roi_energy_fraction,
                self.coherence_magnitude,
            )
        ):
            raise ValueError("comparison metrics must be non-negative")
        if self.outside_roi_energy_fraction > 1.0 or self.coherence_magnitude > 1.0:
            raise ValueError("fractional comparison metrics cannot exceed one")
        phase = np.asarray(
            [self.phase_rms_waves, self.phase_pv_waves], dtype=np.float64
        )
        if self.unwrap_status == "ambiguous":
            if not np.all(np.isnan(phase)):
                raise ValueError("ambiguous phase metrics must be NaN")
        elif not np.all(np.isfinite(phase)) or np.any(phase < 0.0):
            raise ValueError("resolved phase metrics must be finite and non-negative")


def build_frozen_rois(
    reference: PointField2D,
    *,
    reference_zbf_sha256: str,
    thresholds: tuple[float, float, float] = (1e-3, 1e-2, 1e-6),
) -> FrozenRoiSet:
    """Freeze peak-connected regions from the Zemax endpoint alone."""

    if not isinstance(reference, PointField2D):
        raise ValueError("reference must be a physical PointField2D")
    reference_hash = _require_sha256(
        reference_zbf_sha256, label="reference ZBF SHA-256"
    )
    if not isinstance(thresholds, tuple) or len(thresholds) != 3:
        raise ValueError("exactly three ordered ROI thresholds are required")
    parsed: list[float] = []
    for threshold in thresholds:
        if isinstance(threshold, (bool, np.bool_)):
            raise ValueError("ROI thresholds must be finite and positive")
        value = float(threshold)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("ROI thresholds must be finite and positive")
        if value > 1.0:
            raise ValueError("relative ROI thresholds cannot exceed one")
        parsed.append(value)
    if len(set(parsed)) != 3:
        raise ValueError("ROI thresholds must be distinct")

    try:
        with np.errstate(over="raise", invalid="raise"):
            intensity = np.abs(reference.values) ** 2
            peak_flat = int(np.argmax(intensity))
            peak_value = float(intensity.flat[peak_flat])
            summed_intensity = float(np.sum(intensity, dtype=np.float64))
    except FloatingPointError as exc:
        raise ValueError("reference field power must remain finite") from exc
    pixel_area = reference.grid.pixel_area_mm2
    total_energy = summed_intensity * pixel_area
    if (
        not math.isfinite(pixel_area)
        or pixel_area <= 0.0
        or not math.isfinite(peak_value)
        or peak_value <= 0.0
        or not math.isfinite(total_energy)
        or total_energy <= 0.0
    ):
        raise ValueError("reference field must have finite nonzero power")
    peak = np.unravel_index(peak_flat, intensity.shape)

    frozen: list[FrozenRoi] = []
    for threshold in parsed:
        support = intensity >= threshold * peak_value
        mask = _component_from_seed(support, peak)
        mask_hash = _mask_sha256(mask)
        frozen.append(
            FrozenRoi(
                threshold=threshold,
                mask=mask,
                mask_sha256=mask_hash,
                reference_zbf_sha256=reference_hash,
                grid=reference.grid,
            )
        )
    return FrozenRoiSet(primary=frozen[0], checks=(frozen[1], frozen[2]))


def _validated_complex_array(values: object, *, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite two-dimensional complex array")
    return array


def _validated_mask(mask: object, *, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(mask)
    if array.dtype != np.bool_ or array.shape != shape or not np.any(array):
        raise ValueError("mask must be a nonempty Boolean array matching the fields")
    return array


def _overlap_piston(
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    pixel_area_mm2: float,
) -> float:
    if not math.isfinite(pixel_area_mm2) or pixel_area_mm2 <= 0.0:
        raise ValueError("pixel area must be finite and positive")
    try:
        with np.errstate(over="raise", invalid="raise"):
            overlap = (
                np.sum(candidate[mask] * np.conj(reference[mask]))
                * pixel_area_mm2
            )
    except FloatingPointError as exc:
        raise ValueError("complex overlap must remain finite") from exc
    if not np.isfinite(overlap):
        raise ValueError("complex overlap must remain finite")
    if overlap.real == 0.0 and overlap.imag == 0.0:
        return 0.0
    return float(np.angle(overlap))


def _symmetric_distance_with_piston(
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    pixel_area_mm2: float,
    piston_rad: float,
) -> float:
    if not math.isfinite(pixel_area_mm2) or pixel_area_mm2 <= 0.0:
        raise ValueError("pixel area must be finite and positive")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            aligned = candidate[mask] * np.exp(-1j * piston_rad)
            target = reference[mask]
            numerator = (
                2.0
                * float(np.sum(np.abs(aligned - target) ** 2, dtype=np.float64))
                * pixel_area_mm2
            )
            denominator = (
                float(
                    np.sum(
                        np.abs(candidate[mask]) ** 2 + np.abs(target) ** 2,
                        dtype=np.float64,
                    )
                )
                * pixel_area_mm2
            )
    except FloatingPointError as exc:
        raise ValueError("symmetric-distance norms must remain finite") from exc
    if (
        numerator < 0.0
        or not math.isfinite(numerator)
        or denominator <= 0.0
        or not math.isfinite(denominator)
    ):
        raise ValueError("symmetric distance requires nonzero finite field energy")
    return float(np.sqrt(max(0.0, numerator / denominator)))


def symmetric_complex_distance(
    a: np.ndarray,
    b: np.ndarray,
    *,
    mask: np.ndarray,
    pixel_area_mm2: float,
) -> float:
    """Return the symmetric field distance after one analytic piston only."""

    first = _validated_complex_array(a, label="first field")
    second = _validated_complex_array(b, label="second field")
    if first.shape != second.shape:
        raise ValueError("symmetric distance fields must have identical shapes")
    selected = _validated_mask(mask, shape=first.shape)
    if not math.isfinite(float(pixel_area_mm2)) or pixel_area_mm2 <= 0.0:
        raise ValueError("pixel area must be finite and positive")
    piston = _overlap_piston(first, second, selected, float(pixel_area_mm2))
    return _symmetric_distance_with_piston(
        first, second, selected, float(pixel_area_mm2), piston
    )


def _nearest_edge_cycles(delta_rad: float) -> tuple[int, bool]:
    quotient = delta_rad / _TWO_PI
    cycles = math.floor(quotient + 0.5)
    residual = delta_rad - cycles * _TWO_PI
    tolerance = 64.0 * np.finfo(np.float64).eps * np.pi
    return int(cycles), bool(abs(abs(residual) - np.pi) <= tolerance)


def _unwrap_phase_on_mask(
    wrapped_phase: np.ndarray,
    mask: np.ndarray,
    *,
    anchor: tuple[int, int],
) -> tuple[np.ndarray | None, UnwrapStatus]:
    """Unwrap by assigning integer cycles on the sampled four-neighbour graph."""

    phase = np.asarray(wrapped_phase, dtype=np.float64)
    if phase.shape != mask.shape or not np.all(np.isfinite(phase[mask])):
        return None, "ambiguous"
    cycles = np.zeros(mask.shape, dtype=np.int64)
    assigned = np.zeros(mask.shape, dtype=bool)
    assigned[anchor] = True
    queue: deque[tuple[int, int]] = deque([anchor])
    used_nonzero_cycle = False
    ny, nx = mask.shape

    while queue:
        row, column = queue.popleft()
        for drow, dcolumn in _NEIGHBOURS_4:
            neighbour_row = row + drow
            neighbour_column = column + dcolumn
            if not (
                0 <= neighbour_row < ny
                and 0 <= neighbour_column < nx
                and mask[neighbour_row, neighbour_column]
            ):
                continue
            edge_cycles, tie = _nearest_edge_cycles(
                float(phase[neighbour_row, neighbour_column] - phase[row, column])
            )
            if tie:
                return None, "ambiguous"
            proposed = int(cycles[row, column]) - edge_cycles
            if assigned[neighbour_row, neighbour_column]:
                if int(cycles[neighbour_row, neighbour_column]) != proposed:
                    return None, "ambiguous"
                continue
            cycles[neighbour_row, neighbour_column] = proposed
            assigned[neighbour_row, neighbour_column] = True
            used_nonzero_cycle = used_nonzero_cycle or proposed != 0
            queue.append((neighbour_row, neighbour_column))

    if not np.array_equal(assigned, mask):
        return None, "ambiguous"
    unwrapped = phase + _TWO_PI * cycles
    return unwrapped, "unique" if used_nonzero_cycle else "not_needed"


def _phase_residual_on_roi(
    candidate_aligned: np.ndarray,
    reference: np.ndarray,
    reference_intensity: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray | None, UnwrapStatus]:
    if np.any(np.abs(candidate_aligned[mask]) == 0.0) or np.any(
        np.abs(reference[mask]) == 0.0
    ):
        return None, "ambiguous"
    wrapped = np.angle(candidate_aligned * np.conj(reference))
    peak_flat = int(np.argmax(np.where(mask, reference_intensity, -np.inf)))
    anchor = np.unravel_index(peak_flat, mask.shape)
    return _unwrap_phase_on_mask(wrapped, mask, anchor=anchor)


def _validate_roi_against_grid(roi: FrozenRoi, grid: UniformGrid2D) -> None:
    if not isinstance(roi, FrozenRoi):
        raise ValueError("comparison requires a FrozenRoi")
    if not _grid_is_identical(roi.grid, grid):
        raise ValueError("frozen ROI and physical fields must use an identical grid")


def _metrics_for_roi_unchecked(
    candidate: PointField2D,
    reference: PointField2D,
    roi: FrozenRoi,
    *,
    piston_rad: float,
) -> ComparisonMetrics:
    mask = roi.mask
    pixel_area = reference.grid.pixel_area_mm2
    reference_intensity = np.abs(reference.values) ** 2
    candidate_intensity = np.abs(candidate.values) ** 2
    reference_roi_energy = float(np.sum(reference_intensity[mask])) * pixel_area
    if reference_roi_energy <= 0.0 or not math.isfinite(reference_roi_energy):
        raise ValueError("reference ROI must have finite nonzero power")
    aligned = candidate.values * np.exp(-1j * piston_rad)

    difference_energy = float(
        np.sum(np.abs(aligned[mask] - reference.values[mask]) ** 2)
    ) * pixel_area
    complex_relative_l2 = float(np.sqrt(difference_energy / reference_roi_energy))
    symmetric_distance = _symmetric_distance_with_piston(
        candidate.values,
        reference.values,
        mask,
        pixel_area,
        piston_rad,
    )

    unwrapped, unwrap_status = _phase_residual_on_roi(
        aligned, reference.values, reference_intensity, mask
    )
    if unwrap_status == "ambiguous" or unwrapped is None:
        phase_rms_waves = float("nan")
        phase_pv_waves = float("nan")
    else:
        residual_waves = unwrapped[mask] / _TWO_PI
        phase_rms_waves = float(
            np.sqrt(
                np.sum(reference_intensity[mask] * residual_waves**2)
                * pixel_area
                / reference_roi_energy
            )
        )
        phase_pv_waves = float(np.max(residual_waves) - np.min(residual_waves))

    intensity_denominator = float(np.sum(reference_intensity[mask] ** 2)) * pixel_area
    if intensity_denominator <= 0.0 or not math.isfinite(intensity_denominator):
        raise ValueError("reference ROI intensity norm must be finite and nonzero")
    intensity_relative_l2 = float(
        np.sqrt(
            float(
                np.sum(
                    (candidate_intensity[mask] - reference_intensity[mask]) ** 2
                )
            )
            * pixel_area
            / intensity_denominator
        )
    )

    reference_full_power = float(np.sum(reference_intensity)) * pixel_area
    candidate_full_power = float(np.sum(candidate_intensity)) * pixel_area
    if reference_full_power <= 0.0 or not math.isfinite(reference_full_power):
        raise ValueError("reference field must have finite nonzero full-window power")
    full_window_power_relative_error = abs(
        candidate_full_power - reference_full_power
    ) / reference_full_power
    outside_roi_energy_fraction = max(
        0.0,
        min(1.0, 1.0 - reference_roi_energy / reference_full_power),
    )

    candidate_roi_energy = float(np.sum(candidate_intensity[mask])) * pixel_area
    if candidate_roi_energy == 0.0:
        coherence = 0.0
    else:
        overlap = np.sum(
            candidate.values[mask] * np.conj(reference.values[mask])
        ) * pixel_area
        coherence = float(
            abs(overlap) / np.sqrt(candidate_roi_energy * reference_roi_energy)
        )
        coherence = max(0.0, min(1.0, coherence))

    return ComparisonMetrics(
        threshold=roi.threshold,
        piston_rad=piston_rad,
        complex_relative_l2=complex_relative_l2,
        symmetric_distance=symmetric_distance,
        phase_rms_waves=phase_rms_waves,
        phase_pv_waves=phase_pv_waves,
        intensity_relative_l2=intensity_relative_l2,
        full_window_power_relative_error=full_window_power_relative_error,
        outside_roi_energy_fraction=outside_roi_energy_fraction,
        coherence_magnitude=coherence,
        unwrap_status=unwrap_status,
    )


def _metrics_for_roi(
    candidate: PointField2D,
    reference: PointField2D,
    roi: FrozenRoi,
    *,
    piston_rad: float,
) -> ComparisonMetrics:
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _metrics_for_roi_unchecked(
                candidate, reference, roi, piston_rad=piston_rad
            )
    except FloatingPointError as exc:
        raise ValueError("comparison metric norms must remain finite") from exc


def compare_physical_fields(
    candidate: PointField2D,
    reference: PointField2D,
    rois: FrozenRoiSet,
) -> tuple[ComparisonMetrics, ComparisonMetrics, ComparisonMetrics]:
    """Compare physical total fields after one primary-ROI phase piston."""

    if not isinstance(candidate, PointField2D) or not isinstance(
        reference, PointField2D
    ):
        raise ValueError("comparison inputs must be physical PointField2D values")
    if not _grid_is_identical(candidate.grid, reference.grid):
        raise ValueError("candidate and reference must use one identical grid")
    if not isinstance(rois, FrozenRoiSet):
        raise ValueError("comparison requires a FrozenRoiSet with a primary ROI")
    ordered = rois.ordered
    for roi in ordered:
        _validate_roi_against_grid(roi, reference.grid)

    primary = ordered[0]
    piston = _overlap_piston(
        candidate.values,
        reference.values,
        primary.mask,
        reference.grid.pixel_area_mm2,
    )
    metrics = tuple(
        _metrics_for_roi(candidate, reference, roi, piston_rad=piston)
        for roi in ordered
    )
    return metrics


__all__ = [
    "ComparisonMetrics",
    "FrozenRoi",
    "FrozenRoiSet",
    "build_frozen_rois",
    "compare_physical_fields",
    "symmetric_complex_distance",
]
