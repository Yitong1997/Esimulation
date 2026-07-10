"""Bit-faithful construction and validation of derived Zemax beam inputs."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .field_contract import (
    pilot_from_zbf,
    point_values_to_zbf_payload,
    reference_phases,
    zbf_payload_to_point_values,
)
from .fourier import resample_bandlimited
from .models import (
    DerivationStrategy,
    PointField2D,
    SampleValueConvention,
    SurfaceConvention,
    UniformGrid2D,
)
from .zbf_binary import (
    DX_OFFSET,
    DY_OFFSET,
    NX_OFFSET,
    NY_OFFSET,
    HeaderDifference,
    LosslessZbf,
    compare_headers,
    patch_sampling_header,
    read_lossless_zbf,
    sha256_file,
    write_lossless_zbf,
)


_ALLOWED_SAMPLING_HEADER_BYTES = frozenset(
    range(NX_OFFSET, NY_OFFSET + 4)
) | frozenset(range(DX_OFFSET, DY_OFFSET + 8))
_FIXED_WINDOW_FIELD_L2_LIMIT = 1e-10
_FIXED_WINDOW_PHASE_LIMIT_WAVES = 1e-8
_FIXED_WINDOW_INTENSITY_LIMIT_PERCENT = 1e-6
_RELATIVE_ENERGY_LIMIT = 1e-10
_EDGE_TARGET_FRACTION = 1e-12
_EDGE_HARD_GATE_FRACTION = 1e-10


@dataclass(frozen=True)
class ComponentDerivedInputValidation:
    component: str
    slow_field_common_node_relative_l2: float
    physical_phase_rms_waves: float
    normalized_intensity_rms_percent: float
    relative_energy_error: float
    edge_energy_fraction_x: float
    edge_energy_fraction_y: float
    fixed_step_overlap_bitwise: bool
    added_samples_exact_zero: bool


@dataclass(frozen=True)
class DerivedInputValidation:
    source_sha256: str
    output_sha256: str
    source_header_hex: str
    output_header_hex: str
    raw_header_diff: HeaderDifference
    changed_header_byte_indices: tuple[int, ...]
    unexpected_header_byte_changes: int
    trailing_bytes_preserved: bool
    components: tuple[ComponentDerivedInputValidation, ...]
    byte_exact_copy: bool
    edge_energy_target_fraction: float
    edge_energy_hard_gate_fraction: float
    applied_edge_energy_gate_fraction: float
    power_normalization_applied: bool = False

    @property
    def slow_field_common_node_relative_l2(self) -> float:
        return max(
            (item.slow_field_common_node_relative_l2 for item in self.components),
            default=0.0,
        )

    @property
    def physical_phase_rms_waves(self) -> float:
        return max(
            (item.physical_phase_rms_waves for item in self.components),
            default=0.0,
        )

    @property
    def normalized_intensity_rms_percent(self) -> float:
        return max(
            (item.normalized_intensity_rms_percent for item in self.components),
            default=0.0,
        )

    @property
    def relative_energy_error(self) -> float:
        return max(
            (item.relative_energy_error for item in self.components),
            default=0.0,
        )

    @property
    def edge_energy_fraction(self) -> float:
        return max(
            (
                max(item.edge_energy_fraction_x, item.edge_energy_fraction_y)
                for item in self.components
            ),
            default=0.0,
        )

    @property
    def edge_energy_target_met(self) -> bool:
        return self.edge_energy_fraction <= self.edge_energy_target_fraction

    @property
    def fixed_step_overlap_bitwise(self) -> bool:
        return all(item.fixed_step_overlap_bitwise for item in self.components)

    @property
    def added_samples_exact_zero(self) -> bool:
        return all(item.added_samples_exact_zero for item in self.components)


@dataclass(frozen=True)
class DerivedInputResult:
    source_path: Path
    output_path: Path
    strategy: DerivationStrategy
    sample_value_convention: SampleValueConvention
    source_sha256: str
    output_sha256: str
    validation: DerivedInputValidation
    upstream_run_case_sha256: str | None = None


def _same_float_bits(left: float, right: float) -> bool:
    return np.float64(left).tobytes() == np.float64(right).tobytes()


def _require_same_grid(source: UniformGrid2D, target: UniformGrid2D) -> None:
    if source.nx != target.nx or source.ny != target.ny:
        raise ValueError("copy strategy requires identical grid dimensions")
    if not _same_float_bits(source.dx_mm, target.dx_mm) or not _same_float_bits(
        source.dy_mm, target.dy_mm
    ):
        raise ValueError("copy strategy requires identical nominal dx/dy bits")


def _require_centered_zbf_grid(grid: UniformGrid2D) -> None:
    centered = UniformGrid2D.centered(
        nx=grid.nx,
        ny=grid.ny,
        dx_mm=grid.dx_mm,
        dy_mm=grid.dy_mm,
    )
    if not np.array_equal(grid.x_mm, centered.x_mm) or not np.array_equal(
        grid.y_mm, centered.y_mm
    ):
        raise ValueError("ZBF target grid must use the centered sample-at-zero convention")


def _require_fixed_window(source: UniformGrid2D, target: UniformGrid2D) -> None:
    for label, source_n, source_d, target_n, target_d in (
        ("x", source.nx, source.dx_mm, target.nx, target.dx_mm),
        ("y", source.ny, source.dy_mm, target.ny, target.dy_mm),
    ):
        if target_n < source_n or target_n % source_n:
            raise ValueError(f"fixed-window {label} refinement must be an integer factor")
        if not math.isclose(
            target_n * target_d,
            source_n * source_d,
            rel_tol=8 * np.finfo(np.float64).eps,
            abs_tol=0.0,
        ):
            raise ValueError(f"fixed-window {label} extent must remain unchanged")


def _require_fixed_step(source: UniformGrid2D, target: UniformGrid2D) -> None:
    if not _same_float_bits(source.dx_mm, target.dx_mm) or not _same_float_bits(
        source.dy_mm, target.dy_mm
    ):
        raise ValueError("fixed-step extension requires identical nominal dx/dy bits")
    if target.nx < source.nx or target.ny < source.ny:
        raise ValueError("zero extension cannot crop the source grid")
    if any(value % 2 for value in (source.nx, source.ny, target.nx, target.ny)):
        raise ValueError("centered fixed-step extension requires even grid dimensions")


def _component_arrays(beam: LosslessZbf) -> tuple[tuple[str, np.ndarray], ...]:
    arrays: list[tuple[str, np.ndarray]] = [("Ex", beam.ex)]
    if beam.ey is not None:
        arrays.append(("Ey", beam.ey))
    return tuple(arrays)


def _outer_edge_energy_fractions(values: np.ndarray) -> tuple[float, float]:
    intensity = np.abs(values) ** 2
    total = float(np.sum(intensity, dtype=np.float64))
    if total == 0.0:
        return 0.0, 0.0
    ny, nx = intensity.shape
    edge_nx = max(1, math.ceil(0.05 * nx))
    edge_ny = max(1, math.ceil(0.05 * ny))
    x_energy = float(
        np.sum(intensity[:, :edge_nx], dtype=np.float64)
        + np.sum(intensity[:, -edge_nx:], dtype=np.float64)
    )
    y_energy = float(
        np.sum(intensity[:edge_ny, :], dtype=np.float64)
        + np.sum(intensity[-edge_ny:, :], dtype=np.float64)
    )
    return x_energy / total, y_energy / total


def _effective_edge_gate(requested_maximum: float) -> float:
    if not np.isfinite(requested_maximum) or requested_maximum < 0.0:
        raise ValueError("max_edge_energy_fraction must be finite and nonnegative")
    return min(float(requested_maximum), _EDGE_HARD_GATE_FRACTION)


def _require_edge_gate(beam: LosslessZbf, *, maximum: float) -> None:
    for component, values in _component_arrays(beam):
        x_fraction, y_fraction = _outer_edge_energy_fractions(values)
        if max(x_fraction, y_fraction) > maximum:
            raise ValueError(
                f"{component} outer-five-percent edge-energy fraction exceeds hard gate"
            )


def _resample_payload(
    values: np.ndarray,
    *,
    source_grid: UniformGrid2D,
    target_grid: UniformGrid2D,
    sample_value_convention: SampleValueConvention,
) -> np.ndarray:
    point_values = zbf_payload_to_point_values(
        values,
        source_grid,
        sample_value_convention=sample_value_convention,
    )
    refined = resample_bandlimited(PointField2D(point_values, source_grid), target_grid)
    return point_values_to_zbf_payload(
        refined.values,
        target_grid,
        sample_value_convention=sample_value_convention,
    )


def _zero_extend_raw(
    values: np.ndarray, *, target_shape: tuple[int, int]
) -> np.ndarray:
    source_ny, source_nx = values.shape
    target_ny, target_nx = target_shape
    x0 = target_nx // 2 - source_nx // 2
    x1 = target_nx // 2 + source_nx // 2
    y0 = target_ny // 2 - source_ny // 2
    y1 = target_ny // 2 + source_ny // 2
    extended = np.zeros(target_shape, dtype=np.complex128)
    extended[y0:y1, x0:x1] = values
    return extended


def _intermediate_extension_grid(
    source: UniformGrid2D, target: UniformGrid2D
) -> UniformGrid2D:
    dimensions: list[int] = []
    for label, target_n, target_d, source_d, source_n in (
        ("x", target.nx, target.dx_mm, source.dx_mm, source.nx),
        ("y", target.ny, target.dy_mm, source.dy_mm, source.ny),
    ):
        nominal = target_n * target_d / source_d
        rounded = int(round(nominal))
        if rounded < source_n or rounded % 2 or not math.isclose(
            nominal,
            rounded,
            rel_tol=8 * np.finfo(np.float64).eps,
            abs_tol=8 * np.finfo(np.float64).eps,
        ):
            raise ValueError(
                f"zero-extend-then-refine {label} window is not an even integer grid"
            )
        dimensions.append(rounded)
    return UniformGrid2D.centered(
        nx=dimensions[0],
        ny=dimensions[1],
        dx_mm=source.dx_mm,
        dy_mm=source.dy_mm,
    )


def _write_derived(
    source: LosslessZbf,
    output_path: Path,
    *,
    target_grid: UniformGrid2D,
    ex: np.ndarray,
    ey: np.ndarray | None,
) -> None:
    header = patch_sampling_header(
        source.header,
        nx=target_grid.nx,
        ny=target_grid.ny,
        dx=target_grid.dx_mm,
        dy=target_grid.dy_mm,
    )
    derived = LosslessZbf(
        path=None,
        source_sha256="",
        header=header,
        ex=ex,
        ey=ey,
        trailing_bytes=source.trailing_bytes,
    )
    write_lossless_zbf(output_path, derived)


def _common_node_indices(
    source: UniformGrid2D, output: UniformGrid2D
) -> tuple[np.ndarray, np.ndarray] | None:
    if output.nx % source.nx or output.ny % source.ny:
        return None
    rx = output.nx // source.nx
    ry = output.ny // source.ny
    if not math.isclose(
        output.nx * output.dx_mm,
        source.nx * source.dx_mm,
        rel_tol=8 * np.finfo(np.float64).eps,
        abs_tol=0.0,
    ) or not math.isclose(
        output.ny * output.dy_mm,
        source.ny * source.dy_mm,
        rel_tol=8 * np.finfo(np.float64).eps,
        abs_tol=0.0,
    ):
        return None
    return np.arange(source.ny) * ry, np.arange(source.nx) * rx


def _center_slices(
    source_shape: tuple[int, int], output_shape: tuple[int, int]
) -> tuple[slice, slice] | None:
    source_ny, source_nx = source_shape
    output_ny, output_nx = output_shape
    if (
        output_nx < source_nx
        or output_ny < source_ny
        or any(value % 2 for value in (source_nx, source_ny, output_nx, output_ny))
    ):
        return None
    x0 = output_nx // 2 - source_nx // 2
    y0 = output_ny // 2 - source_ny // 2
    return slice(y0, y0 + source_ny), slice(x0, x0 + source_nx)


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left.reshape(-1)))
    numerator = float(np.linalg.norm((right - left).reshape(-1)))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def _phase_rms_waves(left: np.ndarray, right: np.ndarray) -> float:
    weights = np.abs(left) ** 2
    valid = (weights > 0.0) & (np.abs(right) > 0.0)
    if not np.any(valid):
        return 0.0
    phase_waves = np.angle(right[valid] * np.conj(left[valid])) / (2 * np.pi)
    return float(np.sqrt(np.sum(weights[valid] * phase_waves**2) / np.sum(weights[valid])))


def _normalized_intensity_rms_percent(
    left: np.ndarray, right: np.ndarray
) -> float:
    left_i = np.abs(left) ** 2
    right_i = np.abs(right) ** 2
    left_peak = float(np.max(left_i, initial=0.0))
    right_peak = float(np.max(right_i, initial=0.0))
    if left_peak == 0.0 and right_peak == 0.0:
        return 0.0
    if left_peak == 0.0 or right_peak == 0.0:
        return math.inf
    return 100.0 * float(
        np.sqrt(np.mean((right_i / right_peak - left_i / left_peak) ** 2))
    )


def _relative_energy_error(
    source: np.ndarray,
    output: np.ndarray,
    *,
    source_grid: UniformGrid2D,
    output_grid: UniformGrid2D,
) -> float:
    source_energy = float(
        np.sum(np.abs(source) ** 2, dtype=np.float64) * source_grid.pixel_area_mm2
    )
    output_energy = float(
        np.sum(np.abs(output) ** 2, dtype=np.float64) * output_grid.pixel_area_mm2
    )
    if source_energy == 0.0:
        return 0.0 if output_energy == 0.0 else math.inf
    return abs(output_energy - source_energy) / source_energy


def _physical_values(
    point_payload: np.ndarray,
    beam: LosslessZbf,
    *,
    convention: SurfaceConvention,
) -> np.ndarray:
    pilot = pilot_from_zbf(beam, convention)
    phases = reference_phases(
        beam.grid,
        pilot,
        wavelength_vacuum_mm=beam.header.wavelength_vacuum_mm,
        refractive_index=beam.header.refractive_index,
    )
    return np.conj(point_payload) * np.exp(1j * phases.phi_rad)


def validate_derived_input(
    source_path: str | Path,
    output_path: str | Path,
    *,
    strategy: DerivationStrategy,
    convention: SurfaceConvention,
    sample_value_convention: SampleValueConvention,
    max_edge_energy_fraction: float = 1e-10,
) -> DerivedInputValidation:
    """Validate a derived input and raise if any strategy-specific hard gate fails."""

    source_file = Path(source_path).resolve()
    output_file = Path(output_path).resolve()
    effective_edge_gate = _effective_edge_gate(max_edge_energy_fraction)
    source = read_lossless_zbf(source_file)
    output = read_lossless_zbf(output_file)
    intermediate_grid: UniformGrid2D | None = None
    if strategy in {"exact_copy", "chained_zemax_output"}:
        _require_same_grid(source.grid, output.grid)
    elif strategy == "fourier_refine_fixed_window":
        _require_fixed_window(source.grid, output.grid)
    elif strategy == "zero_extend_fixed_sampling":
        _require_fixed_step(source.grid, output.grid)
    elif strategy == "zero_extend_then_fourier_refine":
        intermediate_grid = _intermediate_extension_grid(source.grid, output.grid)
        _require_fixed_step(source.grid, intermediate_grid)
        _require_fixed_window(intermediate_grid, output.grid)
    else:
        raise ValueError(f"unknown derivation strategy: {strategy}")
    header_diff = compare_headers(source.header, output.header)
    changed_header_bytes = tuple(
        index
        for index, (left, right) in enumerate(
            zip(source.header.raw_bytes, output.header.raw_bytes, strict=True)
        )
        if left != right
    )
    unexpected = tuple(
        index for index in changed_header_bytes if index not in _ALLOWED_SAMPLING_HEADER_BYTES
    )
    if unexpected or header_diff.changed_reserved_int_indices or header_diff.changed_reserved_double_indices:
        raise ValueError("derived ZBF changed an unexpected header byte")
    if any(
        field not in {"nx", "ny", "dx", "dy"}
        for field in header_diff.changed_named_fields
    ):
        raise ValueError("derived ZBF changed a non-sampling named header field")
    if source.trailing_bytes != output.trailing_bytes:
        raise ValueError("derived ZBF changed uninterpreted trailing bytes")
    if (source.ey is None) != (output.ey is None):
        raise ValueError("derived ZBF changed polarization payload structure")

    source_components = dict(_component_arrays(source))
    output_components = dict(_component_arrays(output))
    common_indices = _common_node_indices(source.grid, output.grid)
    overlap_slices = _center_slices(source.ex.shape, output.ex.shape)
    component_validations: list[ComponentDerivedInputValidation] = []
    for component, source_raw in source_components.items():
        output_raw = output_components[component]
        source_point = zbf_payload_to_point_values(
            source_raw,
            source.grid,
            sample_value_convention=sample_value_convention,
        )
        output_point = zbf_payload_to_point_values(
            output_raw,
            output.grid,
            sample_value_convention=sample_value_convention,
        )
        edge_x, edge_y = _outer_edge_energy_fractions(source_point)
        common_l2 = 0.0
        phase_rms = 0.0
        intensity_rms = 0.0
        overlap_bitwise = strategy not in {
            "zero_extend_fixed_sampling",
            "zero_extend_then_fourier_refine",
        }
        added_zero = overlap_bitwise

        if strategy == "fourier_refine_fixed_window":
            if common_indices is None:
                raise ValueError("fixed-window output has no exact source common nodes")
            iy, ix = common_indices
            output_common = output_point[np.ix_(iy, ix)]
            common_l2 = _relative_l2(source_point, output_common)
            source_physical = _physical_values(
                source_point, source, convention=convention
            )
            output_physical = _physical_values(
                output_point, output, convention=convention
            )[np.ix_(iy, ix)]
            phase_rms = _phase_rms_waves(source_physical, output_physical)
            intensity_rms = _normalized_intensity_rms_percent(
                source_physical, output_physical
            )
        elif strategy == "zero_extend_fixed_sampling":
            if overlap_slices is None:
                raise ValueError("fixed-step output has no centered source overlap")
            ys, xs = overlap_slices
            overlap_bitwise = output_raw[ys, xs].tobytes() == source_raw.tobytes()
            mask = np.ones(output_raw.shape, dtype=bool)
            mask[ys, xs] = False
            added_zero = output_raw[mask].tobytes() == np.zeros(
                int(np.sum(mask)), dtype=np.complex128
            ).tobytes()
            if not overlap_bitwise:
                raise ValueError("fixed-step overlap is not bitwise identical")
            if not added_zero:
                raise ValueError("fixed-step added samples are not exact complex zero")
        elif strategy == "zero_extend_then_fourier_refine":
            if intermediate_grid is None:
                raise RuntimeError("combined derivation is missing its intermediate grid")
            intermediate_raw = _zero_extend_raw(
                source_raw,
                target_shape=(intermediate_grid.ny, intermediate_grid.nx),
            )
            intermediate_slices = _center_slices(
                source_raw.shape, intermediate_raw.shape
            )
            if intermediate_slices is None:
                raise ValueError("combined derivation has no centered source overlap")
            ys, xs = intermediate_slices
            overlap_bitwise = (
                intermediate_raw[ys, xs].tobytes() == source_raw.tobytes()
            )
            mask = np.ones(intermediate_raw.shape, dtype=bool)
            mask[ys, xs] = False
            added_zero = intermediate_raw[mask].tobytes() == np.zeros(
                int(np.sum(mask)), dtype=np.complex128
            ).tobytes()
            expected_raw = _resample_payload(
                intermediate_raw,
                source_grid=intermediate_grid,
                target_grid=output.grid,
                sample_value_convention=sample_value_convention,
            )
            expected_point = zbf_payload_to_point_values(
                expected_raw,
                output.grid,
                sample_value_convention=sample_value_convention,
            )
            common_l2 = _relative_l2(expected_point, output_point)
            if not overlap_bitwise or not added_zero:
                raise ValueError("combined derivation failed its exact fixed-step audit")

        energy_error = _relative_energy_error(
            source_point,
            output_point,
            source_grid=source.grid,
            output_grid=output.grid,
        )
        component_validations.append(
            ComponentDerivedInputValidation(
                component=component,
                slow_field_common_node_relative_l2=common_l2,
                physical_phase_rms_waves=phase_rms,
                normalized_intensity_rms_percent=intensity_rms,
                relative_energy_error=energy_error,
                edge_energy_fraction_x=edge_x,
                edge_energy_fraction_y=edge_y,
                fixed_step_overlap_bitwise=overlap_bitwise,
                added_samples_exact_zero=added_zero,
            )
        )

    validation = DerivedInputValidation(
        source_sha256=source.source_sha256,
        output_sha256=output.source_sha256,
        source_header_hex=source.header.raw_bytes.hex(),
        output_header_hex=output.header.raw_bytes.hex(),
        raw_header_diff=header_diff,
        changed_header_byte_indices=changed_header_bytes,
        unexpected_header_byte_changes=len(unexpected),
        trailing_bytes_preserved=True,
        components=tuple(component_validations),
        byte_exact_copy=source_file.read_bytes() == output_file.read_bytes(),
        edge_energy_target_fraction=_EDGE_TARGET_FRACTION,
        edge_energy_hard_gate_fraction=_EDGE_HARD_GATE_FRACTION,
        applied_edge_energy_gate_fraction=effective_edge_gate,
        power_normalization_applied=False,
    )
    if strategy in {"exact_copy", "chained_zemax_output"} and not validation.byte_exact_copy:
        raise ValueError("copy strategy output is not byte-identical")
    if strategy == "fourier_refine_fixed_window":
        if validation.slow_field_common_node_relative_l2 > _FIXED_WINDOW_FIELD_L2_LIMIT:
            raise ValueError("fixed-window common-node slow-field gate failed")
        if validation.physical_phase_rms_waves > _FIXED_WINDOW_PHASE_LIMIT_WAVES:
            raise ValueError("fixed-window physical-phase gate failed")
        if validation.normalized_intensity_rms_percent > _FIXED_WINDOW_INTENSITY_LIMIT_PERCENT:
            raise ValueError("fixed-window normalized-intensity gate failed")
    if (
        strategy == "zero_extend_then_fourier_refine"
        and validation.slow_field_common_node_relative_l2
        > _FIXED_WINDOW_FIELD_L2_LIMIT
    ):
        raise ValueError("combined derivation Fourier-refinement gate failed")
    if validation.relative_energy_error > _RELATIVE_ENERGY_LIMIT:
        raise ValueError("derived input pixel-area-weighted energy gate failed")
    if strategy in {"zero_extend_fixed_sampling", "zero_extend_then_fourier_refine"}:
        if validation.edge_energy_fraction > effective_edge_gate:
            raise ValueError("derived input edge-energy hard gate failed")
    return validation


def derive_zbf_input(
    source_path: str | Path,
    output_path: str | Path,
    *,
    target_grid: UniformGrid2D,
    strategy: DerivationStrategy,
    convention: SurfaceConvention,
    sample_value_convention: SampleValueConvention,
    max_edge_energy_fraction: float = 1e-10,
) -> DerivedInputResult:
    """Construct one permitted derived input without fitting or normalizing power."""

    source_file = Path(source_path).resolve()
    output_file = Path(output_path).resolve()
    if source_file == output_file:
        raise ValueError("derived input output path must differ from its source")
    if not isinstance(target_grid, UniformGrid2D):
        raise ValueError("target_grid must be a UniformGrid2D")
    _require_centered_zbf_grid(target_grid)
    if sample_value_convention not in {"point_value", "cell_energy"}:
        raise ValueError("sample_value_convention must be explicit")
    effective_edge_gate = _effective_edge_gate(max_edge_energy_fraction)
    source = read_lossless_zbf(source_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        if strategy in {"exact_copy", "chained_zemax_output"}:
            _require_same_grid(source.grid, target_grid)
            shutil.copy2(source_file, output_file)
        elif strategy == "fourier_refine_fixed_window":
            _require_fixed_window(source.grid, target_grid)
            ex = _resample_payload(
                source.ex,
                source_grid=source.grid,
                target_grid=target_grid,
                sample_value_convention=sample_value_convention,
            )
            ey = (
                None
                if source.ey is None
                else _resample_payload(
                    source.ey,
                    source_grid=source.grid,
                    target_grid=target_grid,
                    sample_value_convention=sample_value_convention,
                )
            )
            _write_derived(source, output_file, target_grid=target_grid, ex=ex, ey=ey)
        elif strategy == "zero_extend_fixed_sampling":
            _require_fixed_step(source.grid, target_grid)
            _require_edge_gate(source, maximum=effective_edge_gate)
            ex = _zero_extend_raw(
                source.ex, target_shape=(target_grid.ny, target_grid.nx)
            )
            ey = (
                None
                if source.ey is None
                else _zero_extend_raw(
                    source.ey, target_shape=(target_grid.ny, target_grid.nx)
                )
            )
            _write_derived(source, output_file, target_grid=target_grid, ex=ex, ey=ey)
        elif strategy == "zero_extend_then_fourier_refine":
            _require_edge_gate(source, maximum=effective_edge_gate)
            intermediate_grid = _intermediate_extension_grid(source.grid, target_grid)
            _require_fixed_step(source.grid, intermediate_grid)
            _require_fixed_window(intermediate_grid, target_grid)

            def extend_then_refine(values: np.ndarray) -> np.ndarray:
                extended_raw = _zero_extend_raw(
                    values,
                    target_shape=(intermediate_grid.ny, intermediate_grid.nx),
                )
                return _resample_payload(
                    extended_raw,
                    source_grid=intermediate_grid,
                    target_grid=target_grid,
                    sample_value_convention=sample_value_convention,
                )

            ex = extend_then_refine(source.ex)
            ey = None if source.ey is None else extend_then_refine(source.ey)
            _write_derived(source, output_file, target_grid=target_grid, ex=ex, ey=ey)
        else:
            raise ValueError(f"unknown derivation strategy: {strategy}")

        validation = validate_derived_input(
            source_file,
            output_file,
            strategy=strategy,
            convention=convention,
            sample_value_convention=sample_value_convention,
            max_edge_energy_fraction=max_edge_energy_fraction,
        )
    except Exception:
        if output_file.exists():
            output_file.unlink()
        raise

    output_sha256 = sha256_file(output_file)
    return DerivedInputResult(
        source_path=source_file,
        output_path=output_file,
        strategy=strategy,
        sample_value_convention=sample_value_convention,
        source_sha256=source.source_sha256,
        output_sha256=output_sha256,
        validation=validation,
        upstream_run_case_sha256=(
            source.source_sha256 if strategy == "chained_zemax_output" else None
        ),
    )


__all__ = [
    "ComponentDerivedInputValidation",
    "DerivedInputResult",
    "DerivedInputValidation",
    "derive_zbf_input",
    "validate_derived_input",
]
