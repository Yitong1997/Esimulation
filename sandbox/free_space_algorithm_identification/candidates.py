"""Fixed, non-tunable Fresnel candidates for the three biconic segments."""

from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import proper

from .field_contract import (
    PilotState,
    quadratic_reference_phase,
    reference_phases,
    spherical_reference_phase,
)
from .fourier import resample_bandlimited
from .fresnel import (
    _axial_phase,
    _medium_parameters,
    _ptp_carrier_removed_cell_samples,
    scaled_dft_cell_samples,
)
from .models import PointField2D, SegmentSpec, UniformGrid2D


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CROP_ENERGY_LIMIT = 1.0e-10
_EDGE_ENERGY_LIMIT = 1.0e-10


def _require_sha256(value: str, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _field_sha256(field: PointField2D) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(field.values, dtype="<c16").tobytes(order="C"))
    return digest.hexdigest()


def _grid_sha256(grid: UniformGrid2D) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(grid.x_mm, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(grid.y_mm, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _same_grid(left: UniformGrid2D, right: UniformGrid2D) -> bool:
    return bool(
        left.nx == right.nx
        and left.ny == right.ny
        and np.allclose(left.x_mm, right.x_mm, rtol=2e-13, atol=0.0)
        and np.allclose(left.y_mm, right.y_mm, rtol=2e-13, atol=0.0)
    )


@dataclass(frozen=True)
class CandidateResult:
    segment_key: str
    operator_id: Literal["H", "F_Q", "R_Phi_given_Q", "R_Phi_given_Phi"]
    input_sha256: str
    input_grid_sha256: str
    output: PointField2D
    predicted_target_zeta_mm: float
    diagnostics: Mapping[str, float | str | bool]

    def __post_init__(self) -> None:
        if not isinstance(self.segment_key, str) or not self.segment_key:
            raise ValueError("candidate segment key must be nonempty")
        if self.operator_id not in {
            "H",
            "F_Q",
            "R_Phi_given_Q",
            "R_Phi_given_Phi",
        }:
            raise ValueError("unknown candidate operator")
        _require_sha256(self.input_sha256, label="candidate input hash")
        _require_sha256(self.input_grid_sha256, label="candidate grid hash")
        if not isinstance(self.output, PointField2D):
            raise ValueError("candidate output must be a PointField2D")
        if not np.isfinite(self.predicted_target_zeta_mm):
            raise ValueError("predicted target pilot position must be finite")
        copied = dict(self.diagnostics)
        for key, value in copied.items():
            if not isinstance(key, str) or not isinstance(value, (float, str, bool)):
                raise ValueError("candidate diagnostics must use scalar audited values")
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError("candidate diagnostics must be finite")
        object.__setattr__(self, "diagnostics", MappingProxyType(copied))


@dataclass(frozen=True)
class FiniteSupportMap:
    field: PointField2D
    eta_crop: float
    square_axis: Literal["x", "y"]
    interpolation_id: Literal["zero_padded_fourier_5pct"] = (
        "zero_padded_fourier_5pct"
    )

    def __post_init__(self) -> None:
        if not isinstance(self.field, PointField2D):
            raise ValueError("finite-support result must be a PointField2D")
        if not np.isfinite(self.eta_crop) or self.eta_crop < 0.0:
            raise ValueError("finite-support crop fraction must be nonnegative")
        if self.square_axis not in ("x", "y"):
            raise ValueError("square axis must be x or y")
        if self.interpolation_id != "zero_padded_fourier_5pct":
            raise ValueError("unknown finite-support interpolation")


def _window_edges(grid: UniformGrid2D) -> tuple[float, float, float, float]:
    return (
        float(grid.x_mm[0]),
        float(grid.x_mm[0] + grid.nx * grid.dx_mm),
        float(grid.y_mm[0]),
        float(grid.y_mm[0] + grid.ny * grid.dy_mm),
    )


def _zero_padded_fourier_resample(
    field: PointField2D, target_grid: UniformGrid2D
) -> PointField2D:
    pad_x = int(np.ceil(0.05 * field.grid.nx))
    pad_y = int(np.ceil(0.05 * field.grid.ny))
    padded_grid = UniformGrid2D.centered(
        nx=field.grid.nx + 2 * pad_x,
        ny=field.grid.ny + 2 * pad_y,
        dx_mm=field.grid.dx_mm,
        dy_mm=field.grid.dy_mm,
    )
    padded_values = np.zeros(
        (padded_grid.ny, padded_grid.nx), dtype=np.complex128
    )
    padded_values[
        pad_y : pad_y + field.grid.ny,
        pad_x : pad_x + field.grid.nx,
    ] = field.values
    mapped = resample_bandlimited(
        PointField2D(padded_values, padded_grid), target_grid
    )
    values = np.array(mapped.values, dtype=np.complex128, copy=True)
    x, y = np.meshgrid(target_grid.x_mm, target_grid.y_mm)
    left, right, bottom, top = _window_edges(field.grid)
    outside = (x < left) | (x >= right) | (y < bottom) | (y >= top)
    values[outside] = 0.0j
    return PointField2D(values, target_grid)


def map_slow_field_to_square(
    slow_field: PointField2D, *, square_axis: Literal["x", "y"]
) -> FiniteSupportMap:
    """Map a rectangular slow field to a predeclared finite square support."""

    if not isinstance(slow_field, PointField2D):
        raise ValueError("slow_field must be a PointField2D")
    if square_axis not in ("x", "y"):
        raise ValueError("square_axis must be x or y")
    if slow_field.grid.nx != slow_field.grid.ny or slow_field.grid.nx % 2:
        raise ValueError("stock PROPER requires an even N by N source array")
    step = slow_field.grid.dx_mm if square_axis == "x" else slow_field.grid.dy_mm
    square = UniformGrid2D.centered(
        nx=slow_field.grid.nx,
        ny=slow_field.grid.ny,
        dx_mm=step,
        dy_mm=step,
    )
    square_left, square_right, square_bottom, square_top = _window_edges(square)
    sx, sy = np.meshgrid(slow_field.grid.x_mm, slow_field.grid.y_mm)
    cropped = (
        (sx < square_left)
        | (sx >= square_right)
        | (sy < square_bottom)
        | (sy >= square_top)
    )
    intensity = np.abs(slow_field.values) ** 2
    total = float(np.sum(intensity) * slow_field.grid.pixel_area_mm2)
    removed = float(
        np.sum(intensity[cropped]) * slow_field.grid.pixel_area_mm2
    )
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("finite-support crop denominator must be finite and positive")
    eta_crop = removed / total
    if not np.isfinite(eta_crop) or eta_crop > _CROP_ENERGY_LIMIT:
        raise ValueError("finite-support crop energy exceeds the 1e-10 hard gate")
    mapped = _zero_padded_fourier_resample(slow_field, square)
    source_left, source_right, source_bottom, source_top = _window_edges(
        slow_field.grid
    )
    tx, ty = np.meshgrid(square.x_mm, square.y_mm)
    added = (
        (tx < source_left)
        | (tx >= source_right)
        | (ty < source_bottom)
        | (ty >= source_top)
    )
    if np.any(mapped.values[added] != 0.0j):
        raise RuntimeError("finite-support extension did not create exact complex zeros")
    return FiniteSupportMap(
        field=mapped, eta_crop=float(eta_crop), square_axis=square_axis
    )


def _edge_energy_fraction(field: PointField2D) -> float:
    if field.grid.nx != field.grid.ny:
        raise ValueError("PROPER output edge gate requires a square grid")
    width = int(np.ceil(0.05 * field.grid.nx))
    mask = np.zeros(field.values.shape, dtype=bool)
    mask[:width, :] = True
    mask[-width:, :] = True
    mask[:, :width] = True
    mask[:, -width:] = True
    intensity = np.abs(field.values) ** 2
    total = float(np.sum(intensity) * field.grid.pixel_area_mm2)
    edge = float(np.sum(intensity[mask]) * field.grid.pixel_area_mm2)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("output edge-energy denominator must be finite and positive")
    fraction = edge / total
    if not np.isfinite(fraction):
        raise ValueError("output edge-energy fraction must be finite")
    return float(fraction)


def _require_target_inside(
    source_grid: UniformGrid2D, target_grid: UniformGrid2D
) -> None:
    left, right, bottom, top = _window_edges(source_grid)
    if (
        float(target_grid.x_mm[0]) < left
        or float(target_grid.x_mm[-1]) >= right
        or float(target_grid.y_mm[0]) < bottom
        or float(target_grid.y_mm[-1]) >= top
    ):
        raise ValueError("target comparison grid extends outside computed output support")


def _map_computed_slow_field(
    slow_field: PointField2D, target_grid: UniformGrid2D
) -> tuple[PointField2D, float, bool]:
    eta_edge = _edge_energy_fraction(slow_field)
    if eta_edge > _EDGE_ENERGY_LIMIT:
        raise ValueError("output edge energy exceeds the 1e-10 hard gate")
    if _same_grid(slow_field.grid, target_grid):
        return slow_field, eta_edge, False
    _require_target_inside(slow_field.grid, target_grid)
    return _zero_padded_fourier_resample(slow_field, target_grid), eta_edge, True


def lift_q_relative_slow_field(
    slow_q_field: PointField2D,
    *,
    target_grid: UniformGrid2D,
    predicted_target_pilot: PilotState,
    wavelength_vacuum_mm: float,
    refractive_index: float,
    model_distance_mm: float,
) -> PointField2D:
    """Map a computed Q-relative field before restoring Q and axial carrier."""

    if not isinstance(slow_q_field, PointField2D):
        raise ValueError("slow_q_field must be a PointField2D")
    if not isinstance(target_grid, UniformGrid2D):
        raise ValueError("target_grid must be a UniformGrid2D")
    mapped, _, _ = _map_computed_slow_field(slow_q_field, target_grid)
    phases = reference_phases(
        target_grid,
        predicted_target_pilot,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    _, k_per_mm = _medium_parameters(
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    reduced, carrier = _axial_phase(k_per_mm, model_distance_mm)
    if not np.isfinite(reduced):
        raise ValueError("axial carrier must be finite")
    return PointField2D(
        mapped.values * np.exp(1j * phases.q_rad) * carrier,
        target_grid,
    )


def _validate_branch(
    segment: SegmentSpec, start_pilot: PilotState
) -> tuple[float, float, complex, PilotState]:
    if not isinstance(segment, SegmentSpec):
        raise ValueError("segment must be a SegmentSpec")
    if not isinstance(start_pilot, PilotState):
        raise ValueError("start_pilot must be a PilotState")
    if not np.isfinite(segment.model_distance_mm) or segment.model_distance_mm <= 0.0:
        raise ValueError("segment model distance must be positive")
    predicted = PilotState(
        zeta_mm=start_pilot.zeta_mm + segment.model_distance_mm,
        rayleigh_mm=start_pilot.rayleigh_mm,
        waist_mm=start_pilot.waist_mm,
    )
    expected_start_inside = segment.branch[0] == "I"
    expected_end_inside = segment.branch[1] == "I"
    if start_pilot.inside != expected_start_inside or predicted.inside != expected_end_inside:
        raise ValueError("pilot classifications do not match the fixed segment branch")
    a = -start_pilot.zeta_mm
    b = predicted.zeta_mm
    if segment.branch == "OO":
        if not (a < 0.0 and b > 0.0):
            raise ValueError("current S7-to-S8 OO constant requires a<0 and b>0")
        constant = 1.0 + 0.0j
    elif segment.branch == "OI":
        if not (a > 0.0 and b < 0.0):
            raise ValueError("current S12-to-S13 OI constant requires a>0 and b<0")
        constant = -1.0j
    elif segment.branch == "IO":
        if not (a > 0.0 and b > 0.0):
            raise ValueError("current S13-to-S14 IO constant requires a>0 and b>0")
        constant = -1.0j
    else:
        raise ValueError("unsupported segment branch")
    return float(a), float(b), complex(constant), predicted


def _phase_for_kind(
    grid: UniformGrid2D,
    *,
    kind: Literal["q", "phi"],
    signed_distance_mm: float,
    wavelength_vacuum_mm: float,
    refractive_index: float,
) -> np.ndarray:
    if kind == "q":
        return quadratic_reference_phase(
            grid,
            wavelength_vacuum_mm=wavelength_vacuum_mm,
            refractive_index=refractive_index,
            signed_waist_distance_mm=signed_distance_mm,
        )
    return spherical_reference_phase(
        grid,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
        signed_waist_distance_mm=signed_distance_mm,
    )


def _stw(
    cell_samples: np.ndarray,
    grid: UniformGrid2D,
    *,
    kind: Literal["q", "phi"],
    signed_distance_mm: float,
    wavelength_vacuum_mm: float,
    refractive_index: float,
) -> tuple[np.ndarray, UniformGrid2D]:
    transformed = scaled_dft_cell_samples(
        cell_samples,
        grid,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
        signed_distance_mm=signed_distance_mm,
    )
    phase = _phase_for_kind(
        transformed.grid,
        kind=kind,
        signed_distance_mm=signed_distance_mm,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    return transformed.cell_samples * np.exp(1j * phase), transformed.grid


def _wts(
    cell_samples: np.ndarray,
    grid: UniformGrid2D,
    *,
    kind: Literal["q", "phi"],
    signed_distance_mm: float,
    wavelength_vacuum_mm: float,
    refractive_index: float,
) -> tuple[np.ndarray, UniformGrid2D]:
    phase = _phase_for_kind(
        grid,
        kind=kind,
        signed_distance_mm=signed_distance_mm,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    transformed = scaled_dft_cell_samples(
        cell_samples * np.exp(1j * phase),
        grid,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
        signed_distance_mm=signed_distance_mm,
    )
    return transformed.cell_samples, transformed.grid


def _ptp(
    cell_samples: np.ndarray,
    grid: UniformGrid2D,
    *,
    signed_distance_mm: float,
    wavelength_vacuum_mm: float,
    refractive_index: float,
) -> tuple[np.ndarray, UniformGrid2D]:
    wavelength_medium_mm, _ = _medium_parameters(
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    return (
        _ptp_carrier_removed_cell_samples(
            cell_samples,
            grid,
            wavelength_medium_mm=wavelength_medium_mm,
            signed_distance_mm=signed_distance_mm,
        ),
        grid,
    )


def _branch_cell_operator(
    cell_samples: np.ndarray,
    grid: UniformGrid2D,
    *,
    branch: Literal["OO", "OI", "IO"],
    internal_kind: Literal["q", "phi"],
    a_mm: float,
    b_mm: float,
    wavelength_vacuum_mm: float,
    refractive_index: float,
) -> tuple[np.ndarray, UniformGrid2D]:
    common = dict(
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    if branch == "OO":
        values, current = _stw(
            cell_samples,
            grid,
            kind=internal_kind,
            signed_distance_mm=a_mm,
            **common,
        )
        return _wts(
            values,
            current,
            kind=internal_kind,
            signed_distance_mm=b_mm,
            **common,
        )
    if branch == "OI":
        values, current = _stw(
            cell_samples,
            grid,
            kind=internal_kind,
            signed_distance_mm=a_mm,
            **common,
        )
        return _ptp(
            values,
            current,
            signed_distance_mm=b_mm,
            **common,
        )
    values, current = _ptp(
        cell_samples,
        grid,
        signed_distance_mm=a_mm,
        **common,
    )
    return _wts(
        values,
        current,
        kind=internal_kind,
        signed_distance_mm=b_mm,
        **common,
    )


def _validate_target_reference(
    target_grid: UniformGrid2D,
    target_phi_rad: np.ndarray,
    target_zbf_sha256: str,
) -> np.ndarray:
    _require_sha256(target_zbf_sha256, label="target ZBF hash")
    phi = np.asarray(target_phi_rad, dtype=np.float64)
    if phi.shape != (target_grid.ny, target_grid.nx) or not np.all(np.isfinite(phi)):
        raise ValueError("target reference phase must match the captured target grid")
    return phi


def _candidate(
    *,
    segment: SegmentSpec,
    physical_start: PointField2D,
    start_pilot: PilotState,
    wavelength_vacuum_mm: float,
    refractive_index: float,
    operator_id: Literal["F_Q", "R_Phi_given_Q", "R_Phi_given_Phi"],
    target_grid: UniformGrid2D | None,
    target_phi_rad: np.ndarray | None,
    target_zbf_sha256: str | None,
) -> CandidateResult:
    if not isinstance(physical_start, PointField2D):
        raise ValueError("physical_start must be a PointField2D")
    a_mm, b_mm, constant, predicted = _validate_branch(segment, start_pilot)
    start_refs = reference_phases(
        physical_start.grid,
        start_pilot,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    start_reference = (
        start_refs.q_rad if operator_id == "F_Q" else start_refs.phi_rad
    )
    residual = physical_start.values * np.exp(-1j * start_reference)
    cell_samples = np.sqrt(physical_start.grid.pixel_area_mm2) * residual
    internal_kind: Literal["q", "phi"] = (
        "phi" if operator_id == "R_Phi_given_Phi" else "q"
    )
    propagated_cell, natural_output_grid = _branch_cell_operator(
        cell_samples,
        physical_start.grid,
        branch=segment.branch,
        internal_kind=internal_kind,
        a_mm=a_mm,
        b_mm=b_mm,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    natural_slow = PointField2D(
        constant
        * propagated_cell
        / np.sqrt(natural_output_grid.pixel_area_mm2),
        natural_output_grid,
    )
    actual_output_grid = natural_output_grid if target_grid is None else target_grid
    mapped_slow, eta_edge, output_resampled = _map_computed_slow_field(
        natural_slow, actual_output_grid
    )
    if operator_id == "F_Q":
        boundary = reference_phases(
            actual_output_grid,
            predicted,
            wavelength_vacuum_mm=wavelength_vacuum_mm,
            refractive_index=refractive_index,
        ).q_rad
        target_hash_diagnostic = ""
    else:
        if target_grid is None or target_phi_rad is None or target_zbf_sha256 is None:
            raise ValueError("target reference and target ZBF hash are required")
        boundary = _validate_target_reference(
            target_grid, target_phi_rad, target_zbf_sha256
        )
        target_hash_diagnostic = target_zbf_sha256
    _, k_per_mm = _medium_parameters(
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    reduced, carrier = _axial_phase(k_per_mm, segment.model_distance_mm)
    output = PointField2D(
        mapped_slow.values * np.exp(1j * boundary) * carrier,
        actual_output_grid,
    )
    diagnostics: dict[str, float | str | bool] = {
        "branch": segment.branch,
        "branch_constant_real": float(constant.real),
        "branch_constant_imag": float(constant.imag),
        "axial_carrier_nominal_rad": float(k_per_mm * segment.model_distance_mm),
        "axial_carrier_reduced_rad": reduced,
        "target_zbf_sha256": target_hash_diagnostic,
        "uses_predicted_target_q": operator_id == "F_Q",
        "natural_output_grid_sha256": _grid_sha256(natural_output_grid),
        "target_output_grid_sha256": _grid_sha256(actual_output_grid),
        "output_resampled": output_resampled,
        "output_mapping": (
            "zero_padded_fourier_5pct" if output_resampled else "identity"
        ),
        "eta_edge": eta_edge,
    }
    return CandidateResult(
        segment_key=segment.key,
        operator_id=operator_id,
        input_sha256=_field_sha256(physical_start),
        input_grid_sha256=_grid_sha256(physical_start.grid),
        output=output,
        predicted_target_zeta_mm=predicted.zeta_mm,
        diagnostics=diagnostics,
    )


def candidate_f_q(
    *,
    segment: SegmentSpec,
    physical_start: PointField2D,
    start_pilot: PilotState,
    wavelength_vacuum_mm: float,
    refractive_index: float,
    target_grid: UniformGrid2D | None = None,
) -> CandidateResult:
    return _candidate(
        segment=segment,
        physical_start=physical_start,
        start_pilot=start_pilot,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
        operator_id="F_Q",
        target_grid=target_grid,
        target_phi_rad=None,
        target_zbf_sha256=None,
    )


def candidate_r_phi_given_q(
    *,
    segment: SegmentSpec,
    physical_start: PointField2D,
    start_pilot: PilotState,
    target_grid: UniformGrid2D,
    target_phi_rad: np.ndarray,
    target_zbf_sha256: str,
    wavelength_vacuum_mm: float,
    refractive_index: float,
) -> CandidateResult:
    _validate_target_reference(target_grid, target_phi_rad, target_zbf_sha256)
    return _candidate(
        segment=segment,
        physical_start=physical_start,
        start_pilot=start_pilot,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
        operator_id="R_Phi_given_Q",
        target_grid=target_grid,
        target_phi_rad=target_phi_rad,
        target_zbf_sha256=target_zbf_sha256,
    )


def candidate_r_phi_given_phi(
    *,
    segment: SegmentSpec,
    physical_start: PointField2D,
    start_pilot: PilotState,
    target_grid: UniformGrid2D,
    target_phi_rad: np.ndarray,
    target_zbf_sha256: str,
    wavelength_vacuum_mm: float,
    refractive_index: float,
) -> CandidateResult:
    _validate_target_reference(target_grid, target_phi_rad, target_zbf_sha256)
    return _candidate(
        segment=segment,
        physical_start=physical_start,
        start_pilot=start_pilot,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
        operator_id="R_Phi_given_Phi",
        target_grid=target_grid,
        target_phi_rad=target_phi_rad,
        target_zbf_sha256=target_zbf_sha256,
    )


def _implementation_metrics(
    actual: PointField2D, reference: PointField2D
) -> tuple[float, float, float]:
    if not _same_grid(actual.grid, reference.grid):
        raise ValueError("implementation fields must share the exact natural grid")
    reference_norm = float(np.linalg.norm(reference.values))
    if not np.isfinite(reference_norm) or reference_norm <= 0.0:
        raise ValueError("implementation reference norm must be positive")
    complex_l2 = float(np.linalg.norm(actual.values - reference.values) / reference_norm)
    power_reference = float(
        np.sum(np.abs(reference.values) ** 2) * reference.grid.pixel_area_mm2
    )
    power_actual = float(
        np.sum(np.abs(actual.values) ** 2) * actual.grid.pixel_area_mm2
    )
    power_error = abs(power_actual - power_reference) / power_reference
    intensity = np.abs(reference.values) ** 2
    maximum = float(np.max(intensity))
    support = intensity / maximum >= 1.0e-6
    if not np.any(support):
        raise ValueError("implementation phase support is empty")
    phase_waves = np.angle(actual.values[support] * np.conj(reference.values[support])) / (
        2.0 * np.pi
    )
    maximum_phase = float(np.max(np.abs(phase_waves)))
    if not np.all(np.isfinite([complex_l2, power_error, maximum_phase])):
        raise ValueError("implementation metrics must be finite")
    return complex_l2, float(power_error), maximum_phase


def _center_shift_with_receipt(
    values: np.ndarray, *, stage: Literal["input", "output"]
) -> tuple[np.ndarray, str]:
    source = np.asarray(values, dtype=np.complex128, order="C")
    if source.ndim != 2 or not np.all(np.isfinite(source)):
        raise ValueError("center-shift source must be a finite complex array")
    shifted = np.asarray(proper.prop_shift_center(source), dtype=np.complex128)
    if shifted.shape != source.shape or not np.all(np.isfinite(shifted)):
        raise ValueError("center-shift output must preserve finite array shape")
    digest = hashlib.sha256()
    digest.update(stage.encode("ascii"))
    digest.update(np.asarray(source.shape, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(source, dtype="<c16").tobytes(order="C"))
    digest.update(np.asarray(shifted, dtype="<c16").tobytes(order="C"))
    return shifted, digest.hexdigest()


def run_stock_proper_fq(
    *,
    segment: SegmentSpec,
    physical_start: PointField2D,
    start_pilot: PilotState,
    target_grid: UniformGrid2D,
    wavelength_vacuum_mm: float,
    refractive_index: float,
    square_axis: Literal["x", "y"],
) -> CandidateResult:
    """Run unmodified PROPER with explicit Q-relative cell-energy samples."""

    a_mm, b_mm, constant, predicted = _validate_branch(segment, start_pilot)
    del a_mm, b_mm
    start_refs = reference_phases(
        physical_start.grid,
        start_pilot,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    chi_phi = PointField2D(
        physical_start.values * np.exp(-1j * start_refs.phi_rad),
        physical_start.grid,
    )
    mapped = map_slow_field_to_square(chi_phi, square_axis=square_axis)
    square_refs = reference_phases(
        mapped.field.grid,
        start_pilot,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    psi_start = mapped.field.values * np.exp(
        1j * (square_refs.phi_rad - square_refs.q_rad)
    )
    input_cell = np.sqrt(mapped.field.grid.pixel_area_mm2) * psi_start

    wavelength_medium_mm, k_per_mm = _medium_parameters(
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    ngrid = mapped.field.grid.nx
    dx_m = mapped.field.grid.dx_mm * 1.0e-3
    wf = proper.WaveFront(
        ngrid * dx_m,
        ngrid,
        wavelength_medium_mm * 1.0e-3,
        ngrid,
        start_pilot.waist_mm * 1.0e-3,
        start_pilot.rayleigh_mm * 1.0e-3,
    )
    wf.dx = dx_m
    wf.z = 0.0
    wf.z_w0 = -start_pilot.zeta_mm * 1.0e-3
    wf.z_Rayleigh = start_pilot.rayleigh_mm * 1.0e-3
    wf.w0 = start_pilot.waist_mm * 1.0e-3
    wf.reference_surface = "PLANAR" if start_pilot.inside else "SPHERI"
    wf.beam_type_old = "INSIDE_" if start_pilot.inside else "OUTSIDE"
    shift_receipts: dict[str, str] = {}
    wf.wfarr, shift_receipts["input"] = _center_shift_with_receipt(
        input_cell, stage="input"
    )

    global_names = (
        "phase_offset",
        "print_it",
        "verbose",
        "print_total_intensity",
        "do_table",
        "use_fftw",
        "use_ffti",
    )
    saved = {name: getattr(proper, name) for name in global_names}
    try:
        proper.phase_offset = False
        proper.print_it = False
        proper.verbose = False
        proper.print_total_intensity = False
        proper.do_table = False
        proper.use_fftw = False
        proper.use_ffti = False
        proper.prop_propagate(wf, segment.model_distance_mm * 1.0e-3)
    finally:
        for name, value in saved.items():
            setattr(proper, name, value)

    native_grid = UniformGrid2D.centered(
        nx=ngrid,
        ny=ngrid,
        dx_mm=wf.dx * 1.0e3,
        dy_mm=wf.dx * 1.0e3,
    )
    output_cell, shift_receipts["output"] = _center_shift_with_receipt(
        wf.wfarr, stage="output"
    )
    slow_q_native = PointField2D(
        output_cell / np.sqrt(native_grid.pixel_area_mm2), native_grid
    )
    predicted_refs = reference_phases(
        native_grid,
        predicted,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    reduced, carrier = _axial_phase(k_per_mm, segment.model_distance_mm)
    stock_native = PointField2D(
        constant
        * slow_q_native.values
        * np.exp(1j * predicted_refs.q_rad)
        * carrier,
        native_grid,
    )

    square_physical = PointField2D(
        mapped.field.values * np.exp(1j * square_refs.phi_rad), mapped.field.grid
    )
    independent = candidate_f_q(
        segment=segment,
        physical_start=square_physical,
        start_pilot=start_pilot,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    complex_l2, power_error, maximum_phase = _implementation_metrics(
        stock_native, independent.output
    )
    if complex_l2 > 1.0e-10 or power_error > 1.0e-10 or maximum_phase > 1.0e-9:
        raise RuntimeError("stock PROPER failed the no-fit same-operator closure gate")

    mapped_output = lift_q_relative_slow_field(
        PointField2D(constant * slow_q_native.values, native_grid),
        target_grid=target_grid,
        predicted_target_pilot=predicted,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
        model_distance_mm=segment.model_distance_mm,
    )
    output_resampled = not _same_grid(native_grid, target_grid)
    diagnostics: dict[str, float | str | bool] = {
        "branch": segment.branch,
        "square_axis": square_axis,
        "eta_crop": float(mapped.eta_crop),
        "input_mapping": mapped.interpolation_id,
        "eta_edge": _edge_energy_fraction(slow_q_native),
        "natural_output_grid_sha256": _grid_sha256(native_grid),
        "target_output_grid_sha256": _grid_sha256(target_grid),
        "output_resampled": output_resampled,
        "output_mapping": (
            "zero_padded_fourier_5pct" if output_resampled else "identity"
        ),
        "implementation_complex_relative_l2": complex_l2,
        "implementation_power_relative_error": power_error,
        "implementation_max_phase_waves": maximum_phase,
        "center_shift_count": float(len(shift_receipts)),
        "input_center_shift_sha256": shift_receipts["input"],
        "output_center_shift_sha256": shift_receipts["output"],
        "branch_constant_real": float(constant.real),
        "branch_constant_imag": float(constant.imag),
        "axial_carrier_reduced_rad": reduced,
    }
    return CandidateResult(
        segment_key=segment.key,
        operator_id="F_Q",
        input_sha256=_field_sha256(physical_start),
        input_grid_sha256=_grid_sha256(physical_start.grid),
        output=mapped_output,
        predicted_target_zeta_mm=predicted.zeta_mm,
        diagnostics=diagnostics,
    )


__all__ = [
    "CandidateResult",
    "FiniteSupportMap",
    "candidate_f_q",
    "candidate_r_phi_given_phi",
    "candidate_r_phi_given_q",
    "lift_q_relative_slow_field",
    "map_slow_field_to_square",
    "run_stock_proper_fq",
]
