"""Fixed, non-tunable Fresnel candidates for the three biconic segments."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import warnings
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import proper

from .field_contract import (
    MappedZbfField,
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
_PAIRED_PILOT_RELATIVE_TOLERANCE = 1.0e-8
_PAIRED_PILOT_ABSOLUTE_TOLERANCE_MM = 1.0e-12
_PAIRED_DISTANCE_ABSOLUTE_TOLERANCE_MM = 2.0e-6
HalfOpenRegion = tuple[float, float, float, float]
CandidateOperator = Literal["F_Q", "R_Phi_given_Q", "R_Phi_given_Phi"]
Branch = Literal["OO", "OI", "IO"]
ReferenceKind = Literal["q", "phi"]


def _require_sha256(value: str, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_sha256(
    schema: str, records: tuple[tuple[str, bytes], ...]
) -> str:
    digest = hashlib.sha256()

    def add(payload: bytes) -> None:
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)

    add(schema.encode("ascii"))
    for label, payload in records:
        add(label.encode("ascii"))
        add(bytes(payload))
    return digest.hexdigest()


def _shape_bytes(shape: tuple[int, ...]) -> bytes:
    return np.asarray(shape, dtype="<i8").tobytes(order="C")


def _grid_sha256(grid: UniformGrid2D) -> str:
    if not isinstance(grid, UniformGrid2D):
        raise ValueError("grid hash requires a UniformGrid2D")
    x_values = np.asarray(grid.x_mm, dtype="<f8", order="C")
    y_values = np.asarray(grid.y_mm, dtype="<f8", order="C")
    return _canonical_sha256(
        "bts.uniform_grid_2d/v2",
        (
            ("dtype", b"<f8"),
            ("shape", _shape_bytes((grid.ny, grid.nx))),
            ("ny", struct.pack("<Q", grid.ny)),
            ("nx", struct.pack("<Q", grid.nx)),
            ("x_axis_length", struct.pack("<Q", x_values.size)),
            ("y_axis_length", struct.pack("<Q", y_values.size)),
            ("x_axis", x_values.tobytes(order="C")),
            ("y_axis", y_values.tobytes(order="C")),
        ),
    )


def _field_sha256(
    field: PointField2D, *, sample_kind: str = "physical_point_field"
) -> str:
    if not isinstance(field, PointField2D):
        raise ValueError("field hash requires a PointField2D")
    if not isinstance(sample_kind, str) or not sample_kind:
        raise ValueError("field hash requires a nonempty sample kind")
    values = np.asarray(field.values, dtype="<c16", order="C")
    return _canonical_sha256(
        "bts.point_field_2d/v2",
        (
            ("sample_kind", sample_kind.encode("ascii")),
            ("dtype", b"<c16"),
            ("shape", _shape_bytes(values.shape)),
            ("ny", struct.pack("<Q", field.grid.ny)),
            ("nx", struct.pack("<Q", field.grid.nx)),
            ("grid_sha256", _grid_sha256(field.grid).encode("ascii")),
            ("values", values.tobytes(order="C")),
        ),
    )


def _array_hash_records(
    label: str, values: np.ndarray, *, dtype: str
) -> tuple[tuple[str, bytes], ...]:
    array = np.asarray(values, dtype=dtype, order="C")
    return (
        (f"{label}_dtype", dtype.encode("ascii")),
        (f"{label}_shape", _shape_bytes(array.shape)),
        (label, array.tobytes(order="C")),
    )


def _mapped_input_sha256(mapped: MappedZbfField) -> str:
    if not isinstance(mapped, MappedZbfField):
        raise ValueError("mapped-field hash requires MappedZbfField evidence")
    pilot = np.asarray(
        [mapped.pilot.zeta_mm, mapped.pilot.rayleigh_mm, mapped.pilot.waist_mm],
        dtype="<f8",
    )
    records: tuple[tuple[str, bytes], ...] = (
        ("source_sha256", mapped.source_sha256.encode("ascii")),
        (
            "convention_evidence_sha256",
            mapped.convention_evidence_sha256.encode("ascii"),
        ),
        (
            "sample_value_convention",
            mapped.sample_value_convention.encode("ascii"),
        ),
        ("grid_sha256", _grid_sha256(mapped.physical.grid).encode("ascii")),
        ("pilot_dtype", b"<f8"),
        ("pilot_shape", _shape_bytes(pilot.shape)),
        ("pilot", pilot.tobytes(order="C")),
        ("pilot_inside", b"\x01" if mapped.pilot.inside else b"\x00"),
    )
    records += _array_hash_records("q_rad", mapped.references.q_rad, dtype="<f8")
    records += _array_hash_records("phi_rad", mapped.references.phi_rad, dtype="<f8")
    records += _array_hash_records(
        "reference_relative", mapped.reference_relative, dtype="<c16"
    )
    records += _array_hash_records(
        "physical", mapped.physical.values, dtype="<c16"
    )
    records += (
        (
            "physical_field_sha256",
            _field_sha256(mapped.physical).encode("ascii"),
        ),
    )
    return _canonical_sha256("bts.mapped_zbf_candidate_input/v2", records)


def _same_grid(left: UniformGrid2D, right: UniformGrid2D) -> bool:
    return bool(
        left.nx == right.nx
        and left.ny == right.ny
        and np.allclose(left.x_mm, right.x_mm, rtol=2e-13, atol=0.0)
        and np.allclose(left.y_mm, right.y_mm, rtol=2e-13, atol=0.0)
    )


@dataclass(frozen=True)
class PathSpec:
    """One fixed residual-reference path; no phase kind is caller-selectable."""

    path_id: str
    operator_id: CandidateOperator
    branch: Branch
    input_reference: ReferenceKind
    internal_phase: ReferenceKind
    stage_order: tuple[str, str]
    output_reference: ReferenceKind
    branch_constant: complex

    def __post_init__(self) -> None:
        if not isinstance(self.path_id, str) or not self.path_id:
            raise ValueError("path id must be nonempty")
        if self.operator_id not in {
            "F_Q",
            "R_Phi_given_Q",
            "R_Phi_given_Phi",
        }:
            raise ValueError("unknown path operator")
        expected_input = "q" if self.operator_id == "F_Q" else "phi"
        expected_internal = (
            "phi" if self.operator_id == "R_Phi_given_Phi" else "q"
        )
        expected_output = "q" if self.operator_id == "F_Q" else "phi"
        if (
            self.input_reference != expected_input
            or self.internal_phase != expected_internal
            or self.output_reference != expected_output
        ):
            raise ValueError("path references do not match the fixed operator")
        expected_stages = {
            "OO": ("STW:a", "WTS:b"),
            "OI": ("STW:a", "PTP:b"),
            "IO": ("PTP:a", "WTS:b"),
        }
        if self.branch not in expected_stages or self.stage_order != expected_stages[
            self.branch
        ]:
            raise ValueError("path stage order does not match the fixed branch")
        expected_constant = 1.0 + 0.0j if self.branch == "OO" else -1.0j
        if complex(self.branch_constant) != expected_constant:
            raise ValueError("path branch constant does not match the fixed branch")

    @property
    def path_sha256(self) -> str:
        constant = np.asarray(
            [self.branch_constant.real, self.branch_constant.imag], dtype="<f8"
        )
        return _canonical_sha256(
            "bts.fresnel_path_spec/v1",
            (
                ("path_id", self.path_id.encode("ascii")),
                ("operator_id", self.operator_id.encode("ascii")),
                ("branch", self.branch.encode("ascii")),
                ("input_reference", self.input_reference.encode("ascii")),
                ("internal_phase", self.internal_phase.encode("ascii")),
                ("stage_0", self.stage_order[0].encode("ascii")),
                ("stage_1", self.stage_order[1].encode("ascii")),
                ("output_reference", self.output_reference.encode("ascii")),
                ("branch_constant_dtype", b"<f8"),
                ("branch_constant", constant.tobytes(order="C")),
            ),
        )


@dataclass(frozen=True)
class PairedTargetEvidence:
    """Immutable binding of one target ZBF to its segment and start evidence."""

    segment_key: str
    start_source_sha256: str
    start_evidence_sha256: str
    target_zbf_name: str
    target: MappedZbfField

    def __post_init__(self) -> None:
        if not isinstance(self.segment_key, str) or not self.segment_key:
            raise ValueError("paired target requires a segment key")
        _require_sha256(
            self.start_source_sha256, label="paired-target start source hash"
        )
        _require_sha256(
            self.start_evidence_sha256, label="paired-target start evidence hash"
        )
        if not isinstance(self.target_zbf_name, str) or not self.target_zbf_name:
            raise ValueError("paired target requires the expected target ZBF name")
        if not isinstance(self.target, MappedZbfField):
            raise ValueError("paired target requires MappedZbfField evidence")

    @classmethod
    def bind(
        cls,
        *,
        segment: SegmentSpec,
        start: MappedZbfField,
        target: MappedZbfField,
    ) -> "PairedTargetEvidence":
        if not isinstance(segment, SegmentSpec):
            raise ValueError("paired target requires a SegmentSpec")
        if not isinstance(start, MappedZbfField):
            raise ValueError("paired target requires MappedZbfField start evidence")
        return cls(
            segment_key=segment.key,
            start_source_sha256=start.source_sha256,
            start_evidence_sha256=_mapped_input_sha256(start),
            target_zbf_name=segment.target_zbf_name,
            target=target,
        )

    @property
    def pair_sha256(self) -> str:
        return _canonical_sha256(
            "bts.paired_target_evidence/v1",
            (
                ("segment_key", self.segment_key.encode("utf-8")),
                (
                    "start_source_sha256",
                    self.start_source_sha256.encode("ascii"),
                ),
                (
                    "start_evidence_sha256",
                    self.start_evidence_sha256.encode("ascii"),
                ),
                ("target_zbf_name", self.target_zbf_name.encode("utf-8")),
                (
                    "target_evidence_sha256",
                    _mapped_input_sha256(self.target).encode("ascii"),
                ),
            ),
        )


PATH_SPECS: tuple[PathSpec, ...] = (
    PathSpec(
        "F_Q:OO:v1",
        "F_Q",
        "OO",
        "q",
        "q",
        ("STW:a", "WTS:b"),
        "q",
        1.0 + 0.0j,
    ),
    PathSpec(
        "F_Q:OI:v1", "F_Q", "OI", "q", "q", ("STW:a", "PTP:b"), "q", -1.0j
    ),
    PathSpec(
        "F_Q:IO:v1", "F_Q", "IO", "q", "q", ("PTP:a", "WTS:b"), "q", -1.0j
    ),
    PathSpec(
        "R_Phi_given_Q:OO:v1",
        "R_Phi_given_Q",
        "OO",
        "phi",
        "q",
        ("STW:a", "WTS:b"),
        "phi",
        1.0 + 0.0j,
    ),
    PathSpec(
        "R_Phi_given_Q:OI:v1",
        "R_Phi_given_Q",
        "OI",
        "phi",
        "q",
        ("STW:a", "PTP:b"),
        "phi",
        -1.0j,
    ),
    PathSpec(
        "R_Phi_given_Q:IO:v1",
        "R_Phi_given_Q",
        "IO",
        "phi",
        "q",
        ("PTP:a", "WTS:b"),
        "phi",
        -1.0j,
    ),
    PathSpec(
        "R_Phi_given_Phi:OO:v1",
        "R_Phi_given_Phi",
        "OO",
        "phi",
        "phi",
        ("STW:a", "WTS:b"),
        "phi",
        1.0 + 0.0j,
    ),
    PathSpec(
        "R_Phi_given_Phi:OI:v1",
        "R_Phi_given_Phi",
        "OI",
        "phi",
        "phi",
        ("STW:a", "PTP:b"),
        "phi",
        -1.0j,
    ),
    PathSpec(
        "R_Phi_given_Phi:IO:v1",
        "R_Phi_given_Phi",
        "IO",
        "phi",
        "phi",
        ("PTP:a", "WTS:b"),
        "phi",
        -1.0j,
    ),
)


def _select_path(operator_id: CandidateOperator, branch: str) -> PathSpec:
    selected = tuple(
        spec
        for spec in PATH_SPECS
        if spec.operator_id == operator_id and spec.branch == branch
    )
    if len(selected) != 1:
        raise ValueError("operator and branch do not select exactly one fixed path")
    return selected[0]


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
    source_half_open_region: HalfOpenRegion
    target_half_open_region: HalfOpenRegion
    intersection_half_open_region: HalfOpenRegion
    cropped_half_open_regions: tuple[HalfOpenRegion, ...]
    added_half_open_regions: tuple[HalfOpenRegion, ...]
    source_sample_count: int
    target_sample_count: int
    intersection_source_sample_count: int
    intersection_target_sample_count: int
    cropped_sample_count: int
    added_sample_count: int
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
        region_names = (
            "source_half_open_region",
            "target_half_open_region",
            "intersection_half_open_region",
        )
        for name in region_names:
            region = _validated_half_open_region(getattr(self, name), label=name)
            object.__setattr__(self, name, region)
        for name in ("cropped_half_open_regions", "added_half_open_regions"):
            regions = tuple(
                _validated_half_open_region(region, label=name)
                for region in getattr(self, name)
            )
            object.__setattr__(self, name, regions)
        count_names = (
            "source_sample_count",
            "target_sample_count",
            "intersection_source_sample_count",
            "intersection_target_sample_count",
            "cropped_sample_count",
            "added_sample_count",
        )
        for name in count_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError("finite-support sample counts must be integers")
            if int(value) < 0:
                raise ValueError("finite-support sample counts must be nonnegative")
            object.__setattr__(self, name, int(value))
        if (
            self.intersection_source_sample_count + self.cropped_sample_count
            != self.source_sample_count
            or self.intersection_target_sample_count + self.added_sample_count
            != self.target_sample_count
        ):
            raise ValueError("finite-support sample counts do not close")


def _window_edges(grid: UniformGrid2D) -> tuple[float, float, float, float]:
    return (
        float(grid.x_mm[0]),
        float(grid.x_mm[0] + grid.nx * grid.dx_mm),
        float(grid.y_mm[0]),
        float(grid.y_mm[0] + grid.ny * grid.dy_mm),
    )


def _validated_half_open_region(
    region: tuple[float, float, float, float], *, label: str
) -> HalfOpenRegion:
    values = tuple(float(value) for value in region)
    if len(values) != 4 or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain four finite edges")
    left, right, bottom, top = values
    if not left < right or not bottom < top:
        raise ValueError(f"{label} must be a nonempty half-open rectangle")
    return left, right, bottom, top


def _region_intersection(left: HalfOpenRegion, right: HalfOpenRegion) -> HalfOpenRegion:
    intersection = (
        max(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        min(left[3], right[3]),
    )
    return _validated_half_open_region(intersection, label="support intersection")


def _region_difference(
    outer: HalfOpenRegion, intersection: HalfOpenRegion
) -> tuple[HalfOpenRegion, ...]:
    left, right, bottom, top = outer
    ix_left, ix_right, iy_bottom, iy_top = intersection
    candidates = (
        (left, ix_left, bottom, top),
        (ix_right, right, bottom, top),
        (ix_left, ix_right, bottom, iy_bottom),
        (ix_left, ix_right, iy_top, top),
    )
    return tuple(
        _validated_half_open_region(region, label="support difference")
        for region in candidates
        if region[0] < region[1] and region[2] < region[3]
    )


def _region_json(value: object) -> str:
    return json.dumps(
        value, separators=(",", ":"), ensure_ascii=True, allow_nan=False
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
    explicitly_cropped = np.array(slow_field.values, dtype=np.complex128, copy=True)
    explicitly_cropped[cropped] = 0.0j
    mapped = _zero_padded_fourier_resample(
        PointField2D(explicitly_cropped, slow_field.grid), square
    )
    source_region = _window_edges(slow_field.grid)
    target_region = _window_edges(square)
    intersection_region = _region_intersection(source_region, target_region)
    source_left, source_right, source_bottom, source_top = source_region
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
        field=mapped,
        eta_crop=float(eta_crop),
        square_axis=square_axis,
        source_half_open_region=source_region,
        target_half_open_region=target_region,
        intersection_half_open_region=intersection_region,
        cropped_half_open_regions=_region_difference(
            source_region, intersection_region
        ),
        added_half_open_regions=_region_difference(target_region, intersection_region),
        source_sample_count=int(slow_field.grid.nx * slow_field.grid.ny),
        target_sample_count=int(square.nx * square.ny),
        intersection_source_sample_count=int(np.count_nonzero(~cropped)),
        intersection_target_sample_count=int(np.count_nonzero(~added)),
        cropped_sample_count=int(np.count_nonzero(cropped)),
        added_sample_count=int(np.count_nonzero(added)),
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


def _validate_mapped_evidence(
    mapped: MappedZbfField,
    *,
    wavelength_vacuum_mm: float,
    refractive_index: float,
    label: str,
) -> None:
    if not isinstance(mapped, MappedZbfField):
        raise ValueError(f"{label} must be MappedZbfField evidence")
    _medium_parameters(
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    expected_rayleigh_mm = (
        np.pi
        * refractive_index
        * mapped.pilot.waist_mm**2
        / wavelength_vacuum_mm
    )
    if not np.isclose(
        mapped.pilot.rayleigh_mm,
        expected_rayleigh_mm,
        rtol=1.0e-8,
        atol=1.0e-12,
    ):
        raise ValueError(
            f"{label} pilot violates the Rayleigh relation for wavelength and index"
        )
    expected_references = reference_phases(
        mapped.physical.grid,
        mapped.pilot,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    for phase_label, actual, expected in (
        ("Q", mapped.references.q_rad, expected_references.q_rad),
        ("Phi", mapped.references.phi_rad, expected_references.phi_rad),
    ):
        if not np.allclose(actual, expected, rtol=2.0e-13, atol=2.0e-12):
            raise ValueError(
                f"{label} reference {phase_label} does not match pilot, wavelength, index, and grid"
            )
    reconstructed = mapped.reference_relative * np.exp(
        1j * mapped.references.phi_rad
    )
    scale = max(1.0, float(np.max(np.abs(reconstructed))))
    if not np.allclose(
        mapped.physical.values,
        reconstructed,
        rtol=2.0e-13,
        atol=2.0e-14 * scale,
    ):
        raise ValueError(
            f"{label} physical field is not chi times exp(+i Phi)"
        )
    _mapped_input_sha256(mapped)


def _validate_paired_target(
    *,
    evidence: PairedTargetEvidence,
    start: MappedZbfField,
    segment: SegmentSpec,
    wavelength_vacuum_mm: float,
    refractive_index: float,
) -> MappedZbfField:
    if not isinstance(evidence, PairedTargetEvidence):
        raise ValueError("target must be immutable paired-target evidence")
    if (
        evidence.segment_key != segment.key
        or evidence.target_zbf_name != segment.target_zbf_name
    ):
        raise ValueError("paired-target segment binding does not match the candidate")
    if evidence.start_source_sha256 != start.source_sha256:
        raise ValueError("paired-target start source binding does not match")
    if evidence.start_evidence_sha256 != _mapped_input_sha256(start):
        raise ValueError("paired-target start evidence binding does not match")
    target = evidence.target
    _validate_mapped_evidence(
        target,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
        label="target ZBF",
    )
    if target.source_sha256 == start.source_sha256:
        raise ValueError("paired target requires distinct start and target ZBF sources")
    if target.convention_evidence_sha256 != start.convention_evidence_sha256:
        raise ValueError("paired target does not share current-run convention evidence")
    if target.sample_value_convention != start.sample_value_convention:
        raise ValueError("paired target sample-value convention does not match start")
    expected_inside = segment.branch[1] == "I"
    if target.pilot.inside != expected_inside:
        raise ValueError("target pilot classification does not match the segment branch")
    if not np.allclose(
        [target.pilot.rayleigh_mm, target.pilot.waist_mm],
        [start.pilot.rayleigh_mm, start.pilot.waist_mm],
        rtol=_PAIRED_PILOT_RELATIVE_TOLERANCE,
        atol=_PAIRED_PILOT_ABSOLUTE_TOLERANCE_MM,
    ):
        raise ValueError("target pilot invariant does not match the paired start ZBF")
    actual_distance = target.pilot.zeta_mm - start.pilot.zeta_mm
    if not np.isclose(
        actual_distance,
        segment.model_distance_mm,
        rtol=0.0,
        atol=_PAIRED_DISTANCE_ABSOLUTE_TOLERANCE_MM,
    ):
        raise ValueError("paired-target pilot distance does not match the segment")
    return target


def _validate_branch(
    segment: SegmentSpec, start_pilot: PilotState
) -> tuple[float, float, PilotState]:
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
    elif segment.branch == "OI":
        if not (a > 0.0 and b < 0.0):
            raise ValueError("current S12-to-S13 OI constant requires a>0 and b<0")
    elif segment.branch == "IO":
        if not (a > 0.0 and b > 0.0):
            raise ValueError("current S13-to-S14 IO constant requires a>0 and b>0")
    else:
        raise ValueError("unsupported segment branch")
    return float(a), float(b), predicted


def _natural_output_grid(
    *,
    segment: SegmentSpec,
    start: MappedZbfField,
    wavelength_vacuum_mm: float,
    refractive_index: float,
) -> UniformGrid2D:
    """Return the fixed-path natural grid without evaluating a field."""

    _validate_mapped_evidence(
        start,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
        label="start ZBF",
    )
    a_mm, b_mm, _ = _validate_branch(segment, start.pilot)
    path = _select_path("F_Q", segment.branch)
    wavelength_medium_mm, _ = _medium_parameters(
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    distances = {"a": a_mm, "b": b_mm}
    current = start.physical.grid
    for stage in path.stage_order:
        operator, distance_id = stage.split(":", maxsplit=1)
        if operator == "PTP":
            continue
        distance = distances[distance_id]
        current = UniformGrid2D.centered(
            nx=current.nx,
            ny=current.ny,
            dx_mm=wavelength_medium_mm
            * abs(distance)
            / (current.nx * current.dx_mm),
            dy_mm=wavelength_medium_mm
            * abs(distance)
            / (current.ny * current.dy_mm),
        )
    return current


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
    path: PathSpec,
    a_mm: float,
    b_mm: float,
    wavelength_vacuum_mm: float,
    refractive_index: float,
) -> tuple[np.ndarray, UniformGrid2D]:
    common = dict(
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    values = np.asarray(cell_samples, dtype=np.complex128)
    current = grid
    distances = {"a": a_mm, "b": b_mm}
    for stage in path.stage_order:
        operator, distance_id = stage.split(":", maxsplit=1)
        distance = distances[distance_id]
        if operator == "STW":
            values, current = _stw(
                values,
                current,
                kind=path.internal_phase,
                signed_distance_mm=distance,
                **common,
            )
        elif operator == "WTS":
            values, current = _wts(
                values,
                current,
                kind=path.internal_phase,
                signed_distance_mm=distance,
                **common,
            )
        elif operator == "PTP":
            values, current = _ptp(
                values,
                current,
                signed_distance_mm=distance,
                **common,
            )
        else:
            raise RuntimeError("fixed path contains an unknown propagation stage")
    return values, current


def _candidate(
    *,
    segment: SegmentSpec,
    start: MappedZbfField,
    wavelength_vacuum_mm: float,
    refractive_index: float,
    operator_id: CandidateOperator,
    paired_target: PairedTargetEvidence | None,
) -> CandidateResult:
    _validate_mapped_evidence(
        start,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
        label="start ZBF",
    )
    a_mm, b_mm, predicted = _validate_branch(segment, start.pilot)
    path = _select_path(operator_id, segment.branch)
    if paired_target is not None:
        target = _validate_paired_target(
            evidence=paired_target,
            start=start,
            segment=segment,
            wavelength_vacuum_mm=wavelength_vacuum_mm,
            refractive_index=refractive_index,
        )
    elif path.output_reference == "phi":
        raise ValueError("a paired target ZBF is required for a Phi output reference")
    else:
        target = None

    start_reference = (
        start.references.q_rad
        if path.input_reference == "q"
        else start.references.phi_rad
    )
    residual = start.physical.values * np.exp(-1j * start_reference)
    cell_samples = np.sqrt(start.physical.grid.pixel_area_mm2) * residual
    propagated_cell, natural_output_grid = _branch_cell_operator(
        cell_samples,
        start.physical.grid,
        path=path,
        a_mm=a_mm,
        b_mm=b_mm,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    natural_slow = PointField2D(
        path.branch_constant
        * propagated_cell
        / np.sqrt(natural_output_grid.pixel_area_mm2),
        natural_output_grid,
    )
    actual_output_grid = (
        natural_output_grid if target is None else target.physical.grid
    )
    mapped_slow, eta_edge, output_resampled = _map_computed_slow_field(
        natural_slow, actual_output_grid
    )
    if path.output_reference == "q":
        boundary = reference_phases(
            actual_output_grid,
            predicted,
            wavelength_vacuum_mm=wavelength_vacuum_mm,
            refractive_index=refractive_index,
        ).q_rad
    else:
        if target is None:
            raise RuntimeError("fixed Phi path lost its paired target ZBF")
        boundary = target.references.phi_rad
    _, k_per_mm = _medium_parameters(
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    reduced, carrier = _axial_phase(k_per_mm, segment.model_distance_mm)
    output = PointField2D(
        mapped_slow.values * np.exp(1j * boundary) * carrier,
        actual_output_grid,
    )
    target_source_sha256 = "" if target is None else target.source_sha256
    target_evidence_sha256 = "" if target is None else _mapped_input_sha256(target)
    target_convention_sha256 = (
        "" if target is None else target.convention_evidence_sha256
    )
    target_sample_convention = (
        "" if target is None else target.sample_value_convention
    )
    target_pair_sha256 = (
        "" if paired_target is None else paired_target.pair_sha256
    )
    diagnostics: dict[str, float | str | bool] = {
        "branch": segment.branch,
        "path_id": path.path_id,
        "path_sha256": path.path_sha256,
        "input_reference": path.input_reference,
        "internal_phase": path.internal_phase,
        "stage_order": ",".join(path.stage_order),
        "output_reference": path.output_reference,
        "a_mm": a_mm,
        "b_mm": b_mm,
        "branch_constant_real": float(path.branch_constant.real),
        "branch_constant_imag": float(path.branch_constant.imag),
        "axial_carrier_nominal_rad": float(k_per_mm * segment.model_distance_mm),
        "axial_carrier_reduced_rad": reduced,
        "start_zbf_sha256": start.source_sha256,
        "start_convention_evidence_sha256": start.convention_evidence_sha256,
        "start_sample_value_convention": start.sample_value_convention,
        "target_zbf_sha256": target_source_sha256,
        "target_evidence_sha256": target_evidence_sha256,
        "target_convention_evidence_sha256": target_convention_sha256,
        "target_sample_value_convention": target_sample_convention,
        "target_pair_sha256": target_pair_sha256,
        "uses_predicted_target_q": path.output_reference == "q",
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
        input_sha256=_mapped_input_sha256(start),
        input_grid_sha256=_grid_sha256(start.physical.grid),
        output=output,
        predicted_target_zeta_mm=predicted.zeta_mm,
        diagnostics=diagnostics,
    )


def candidate_f_q(
    *,
    segment: SegmentSpec,
    start: MappedZbfField,
    target: PairedTargetEvidence,
    wavelength_vacuum_mm: float,
    refractive_index: float,
) -> CandidateResult:
    return _candidate(
        segment=segment,
        start=start,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
        operator_id="F_Q",
        paired_target=target,
    )


def candidate_r_phi_given_q(
    *,
    segment: SegmentSpec,
    start: MappedZbfField,
    target: PairedTargetEvidence,
    wavelength_vacuum_mm: float,
    refractive_index: float,
) -> CandidateResult:
    return _candidate(
        segment=segment,
        start=start,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
        operator_id="R_Phi_given_Q",
        paired_target=target,
    )


def candidate_r_phi_given_phi(
    *,
    segment: SegmentSpec,
    start: MappedZbfField,
    target: PairedTargetEvidence,
    wavelength_vacuum_mm: float,
    refractive_index: float,
) -> CandidateResult:
    return _candidate(
        segment=segment,
        start=start,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
        operator_id="R_Phi_given_Phi",
        paired_target=target,
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
    receipt = _canonical_sha256(
        "bts.proper_center_shift/v2",
        (
            ("stage", stage.encode("ascii")),
            ("dtype", b"<c16"),
            ("shape", _shape_bytes(source.shape)),
            ("source", np.asarray(source, dtype="<c16").tobytes(order="C")),
            ("shifted", np.asarray(shifted, dtype="<c16").tobytes(order="C")),
        ),
    )
    return shifted, receipt


def run_stock_proper_fq(
    *,
    segment: SegmentSpec,
    start: MappedZbfField,
    target: PairedTargetEvidence,
    wavelength_vacuum_mm: float,
    refractive_index: float,
    square_axis: Literal["x", "y"],
) -> CandidateResult:
    """Run unmodified PROPER with explicit Q-relative cell-energy samples."""

    _validate_mapped_evidence(
        start,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
        label="start ZBF",
    )
    target_field = _validate_paired_target(
        evidence=target,
        start=start,
        segment=segment,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    a_mm, b_mm, predicted = _validate_branch(segment, start.pilot)
    path = _select_path("F_Q", segment.branch)
    constant = path.branch_constant
    start_pilot = start.pilot
    target_grid = target_field.physical.grid
    chi_phi = PointField2D(start.reference_relative, start.physical.grid)
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
    square_evidence = MappedZbfField(
        physical=square_physical,
        reference_relative=mapped.field.values,
        references=square_refs,
        pilot=start_pilot,
        source_sha256=_field_sha256(
            square_physical, sample_kind="derived_square_physical_point_field"
        ),
        convention_evidence_sha256=start.convention_evidence_sha256,
        sample_value_convention="point_value",
    )
    closure_relative = np.ones(
        (native_grid.ny, native_grid.nx), dtype=np.complex128
    )
    closure_physical = PointField2D(
        closure_relative * np.exp(1j * predicted_refs.phi_rad), native_grid
    )
    closure_target = MappedZbfField(
        physical=closure_physical,
        reference_relative=closure_relative,
        references=predicted_refs,
        pilot=predicted,
        source_sha256=_field_sha256(
            closure_physical, sample_kind="internal_closure_target_point_field"
        ),
        convention_evidence_sha256=start.convention_evidence_sha256,
        sample_value_convention="point_value",
    )
    closure_pair = PairedTargetEvidence.bind(
        segment=segment, start=square_evidence, target=closure_target
    )
    independent = candidate_f_q(
        segment=segment,
        start=square_evidence,
        target=closure_pair,
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
        "path_id": path.path_id,
        "path_sha256": path.path_sha256,
        "input_reference": path.input_reference,
        "internal_phase": path.internal_phase,
        "stage_order": ",".join(path.stage_order),
        "output_reference": path.output_reference,
        "a_mm": a_mm,
        "b_mm": b_mm,
        "square_axis": square_axis,
        "eta_crop": float(mapped.eta_crop),
        "input_mapping": mapped.interpolation_id,
        "source_half_open_region": _region_json(mapped.source_half_open_region),
        "target_half_open_region": _region_json(mapped.target_half_open_region),
        "intersection_half_open_region": _region_json(
            mapped.intersection_half_open_region
        ),
        "cropped_half_open_regions": _region_json(
            mapped.cropped_half_open_regions
        ),
        "added_half_open_regions": _region_json(mapped.added_half_open_regions),
        "source_sample_count": float(mapped.source_sample_count),
        "target_sample_count": float(mapped.target_sample_count),
        "intersection_source_sample_count": float(
            mapped.intersection_source_sample_count
        ),
        "intersection_target_sample_count": float(
            mapped.intersection_target_sample_count
        ),
        "cropped_sample_count": float(mapped.cropped_sample_count),
        "added_sample_count": float(mapped.added_sample_count),
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
        "axial_carrier_nominal_rad": float(
            k_per_mm * segment.model_distance_mm
        ),
        "axial_carrier_reduced_rad": reduced,
        "start_zbf_sha256": start.source_sha256,
        "start_convention_evidence_sha256": start.convention_evidence_sha256,
        "start_sample_value_convention": start.sample_value_convention,
        "target_zbf_sha256": target_field.source_sha256,
        "target_evidence_sha256": _mapped_input_sha256(target_field),
        "target_convention_evidence_sha256": (
            target_field.convention_evidence_sha256
        ),
        "target_sample_value_convention": (
            target_field.sample_value_convention
        ),
        "target_pair_sha256": target.pair_sha256,
    }
    return CandidateResult(
        segment_key=segment.key,
        operator_id="F_Q",
        input_sha256=_mapped_input_sha256(start),
        input_grid_sha256=_grid_sha256(start.physical.grid),
        output=mapped_output,
        predicted_target_zeta_mm=predicted.zeta_mm,
        diagnostics=diagnostics,
    )


__all__ = [
    "CandidateResult",
    "FiniteSupportMap",
    "PATH_SPECS",
    "PairedTargetEvidence",
    "PathSpec",
    "candidate_f_q",
    "candidate_r_phi_given_phi",
    "candidate_r_phi_given_q",
    "lift_q_relative_slow_field",
    "map_slow_field_to_square",
    "run_stock_proper_fq",
]
