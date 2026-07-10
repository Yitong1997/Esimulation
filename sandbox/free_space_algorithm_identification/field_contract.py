"""Fail-closed coordinate, phasor, and ZBF physical-field contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .models import (
    PointField2D,
    SampleValueConvention,
    SurfaceConvention,
    UniformGrid2D,
)
from .zbf_binary import LosslessZbf


EvidenceOrigin = Literal["synthetic_test", "live_zosapi"]

_EXPECTED_SURFACE_SIDES = {
    7: "after",
    8: "after",
    12: "after",
    13: "after",
    14: "after",
}
_EXPECTED_AXIS_SIGNS = {7: -1, 8: -1, 12: 1, 13: 1, 14: 1}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _readonly_array(values: object, *, dtype: object) -> np.ndarray:
    array = np.array(values, dtype=dtype, order="C", copy=True)
    immutable = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )
    immutable.setflags(write=False)
    return immutable


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _float_matches(left: float, right: float) -> bool:
    scale = max(1.0, abs(float(left)), abs(float(right)))
    allowance = 64.0 * np.finfo(np.float64).eps * scale
    return bool(abs(float(left) - float(right)) <= allowance)


@dataclass(frozen=True)
class PilotState:
    zeta_mm: float
    rayleigh_mm: float
    waist_mm: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.zeta_mm, self.rayleigh_mm, self.waist_mm], dtype=np.float64
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("pilot values must be finite")
        if self.rayleigh_mm <= 0.0 or self.waist_mm <= 0.0:
            raise ValueError("pilot Rayleigh distance and waist must be positive")

    @property
    def inside(self) -> bool:
        return abs(self.zeta_mm) < self.rayleigh_mm


@dataclass(frozen=True)
class ReferencePhases:
    q_rad: np.ndarray
    phi_rad: np.ndarray

    def __post_init__(self) -> None:
        q_rad = _readonly_array(self.q_rad, dtype=np.float64)
        phi_rad = _readonly_array(self.phi_rad, dtype=np.float64)
        if q_rad.ndim != 2 or q_rad.shape != phi_rad.shape:
            raise ValueError("reference phase arrays must be matching two-dimensional grids")
        if not np.all(np.isfinite(q_rad)) or not np.all(np.isfinite(phi_rad)):
            raise ValueError("reference phase arrays must be finite")
        object.__setattr__(self, "q_rad", q_rad)
        object.__setattr__(self, "phi_rad", phi_rad)


@dataclass(frozen=True)
class MappedZbfField:
    physical: PointField2D
    reference_relative: np.ndarray
    references: ReferencePhases
    pilot: PilotState
    source_sha256: str
    convention_evidence_sha256: str
    sample_value_convention: SampleValueConvention

    def __post_init__(self) -> None:
        reference_relative = _readonly_array(
            self.reference_relative, dtype=np.complex128
        )
        if reference_relative.shape != self.physical.values.shape:
            raise ValueError("reference-relative field shape does not match physical field")
        if reference_relative.shape != self.references.phi_rad.shape:
            raise ValueError("reference phase shape does not match field")
        if not np.all(np.isfinite(reference_relative)):
            raise ValueError("reference-relative field must be finite")
        _require_sha256(self.source_sha256, label="ZBF source hash")
        _require_sha256(
            self.convention_evidence_sha256, label="convention evidence hash"
        )
        if self.sample_value_convention not in {"point_value", "cell_energy"}:
            raise ValueError("unknown ZBF sample-value convention")
        object.__setattr__(self, "reference_relative", reference_relative)


@dataclass(frozen=True)
class RawGridEvidence:
    nx: int
    ny: int
    min_x_mm: float
    min_y_mm: float
    dx_mm: float
    dy_mm: float
    x_checkpoints: tuple[tuple[int, float], ...]
    y_checkpoints: tuple[tuple[int, float], ...]
    z_checkpoints: tuple[tuple[int, int, float], ...]
    values_checkpoints: tuple[tuple[int, int, float], ...]
    raw_grid_array_sha256: str
    input_zbf_sha256: str
    output_zbf_sha256: str
    model_sha256: str
    cfg_sha256: str
    run_id: str
    origin: EvidenceOrigin
    api_array_order: str
    package_array_order: str
    api_to_package_transform: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "x_checkpoints",
            tuple((int(index), float(value)) for index, value in self.x_checkpoints),
        )
        object.__setattr__(
            self,
            "y_checkpoints",
            tuple((int(index), float(value)) for index, value in self.y_checkpoints),
        )
        object.__setattr__(
            self,
            "z_checkpoints",
            tuple(
                (int(ix), int(iy), float(value))
                for ix, iy, value in self.z_checkpoints
            ),
        )
        object.__setattr__(
            self,
            "values_checkpoints",
            tuple(
                (int(ix), int(iy), float(value))
                for ix, iy, value in self.values_checkpoints
            ),
        )

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "api_array_order": self.api_array_order,
            "api_to_package_transform": self.api_to_package_transform,
            "cfg_sha256": self.cfg_sha256,
            "dx_mm": self.dx_mm,
            "dy_mm": self.dy_mm,
            "input_zbf_sha256": self.input_zbf_sha256,
            "min_x_mm": self.min_x_mm,
            "min_y_mm": self.min_y_mm,
            "model_sha256": self.model_sha256,
            "nx": self.nx,
            "ny": self.ny,
            "origin": self.origin,
            "output_zbf_sha256": self.output_zbf_sha256,
            "package_array_order": self.package_array_order,
            "raw_grid_array_sha256": self.raw_grid_array_sha256,
            "run_id": self.run_id,
            "values_checkpoints": self.values_checkpoints,
            "x_checkpoints": self.x_checkpoints,
            "y_checkpoints": self.y_checkpoints,
            "z_checkpoints": self.z_checkpoints,
        }

    @property
    def evidence_sha256(self) -> str:
        return _canonical_sha256(self._canonical_payload())


@dataclass(frozen=True)
class ConventionValidation:
    surface_sides: tuple[tuple[int, str], ...]
    axis_signs: tuple[tuple[int, int], ...]
    raw_zbf_phasor: str
    raw_grid_evidence: tuple[RawGridEvidence, ...]
    raw_grid_evidence_sha256: tuple[str, ...]
    report_sha256: str
    model_sha256: str
    cfg_sha256: str
    phase_unit: str
    run_id: str
    origin: EvidenceOrigin
    authoritative: bool
    validation_status: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "surface_sides",
            tuple(sorted((int(surface), str(side)) for surface, side in self.surface_sides)),
        )
        object.__setattr__(
            self,
            "axis_signs",
            tuple(sorted((int(surface), int(sign)) for surface, sign in self.axis_signs)),
        )
        object.__setattr__(self, "raw_grid_evidence", tuple(self.raw_grid_evidence))
        object.__setattr__(
            self,
            "raw_grid_evidence_sha256",
            tuple(str(value) for value in self.raw_grid_evidence_sha256),
        )

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "authoritative": self.authoritative,
            "axis_signs": self.axis_signs,
            "cfg_sha256": self.cfg_sha256,
            "model_sha256": self.model_sha256,
            "origin": self.origin,
            "phase_unit": self.phase_unit,
            "raw_grid_evidence": tuple(
                evidence._canonical_payload() for evidence in self.raw_grid_evidence
            ),
            "raw_grid_evidence_sha256": self.raw_grid_evidence_sha256,
            "raw_zbf_phasor": self.raw_zbf_phasor,
            "report_sha256": self.report_sha256,
            "run_id": self.run_id,
            "surface_sides": self.surface_sides,
            "validation_status": self.validation_status,
        }

    @property
    def evidence_sha256(self) -> str:
        return _canonical_sha256(self._canonical_payload())


def validate_raw_grid_contract(
    evidence: RawGridEvidence,
    *,
    raw_values_api_xy: np.ndarray | None = None,
) -> np.ndarray | None:
    """Validate raw ``IAR_DataGrid`` evidence and optionally transpose Values once."""

    if not isinstance(evidence, RawGridEvidence):
        raise ValueError("raw-grid evidence is required")
    if (
        isinstance(evidence.nx, bool)
        or isinstance(evidence.ny, bool)
        or evidence.nx < 2
        or evidence.ny < 2
        or evidence.nx % 2
        or evidence.ny % 2
    ):
        raise ValueError("raw-grid sample-at-zero contract requires even Nx and Ny")
    scalars = np.asarray(
        [evidence.min_x_mm, evidence.min_y_mm, evidence.dx_mm, evidence.dy_mm],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(scalars)) or evidence.dx_mm <= 0.0 or evidence.dy_mm <= 0.0:
        raise ValueError("raw-grid coordinates and spacings must be finite and positive")

    expected_min_x = -(evidence.nx / 2) * evidence.dx_mm
    expected_min_y = -(evidence.ny / 2) * evidence.dy_mm
    if not _float_matches(evidence.min_x_mm, expected_min_x) or not _float_matches(
        evidence.min_y_mm, expected_min_y
    ):
        raise ValueError("raw grid violates the sample-at-zero MinX/MinY contract")

    def validate_axis(
        checkpoints: tuple[tuple[int, float], ...],
        *,
        count: int,
        minimum: float,
        spacing: float,
        label: str,
    ) -> None:
        by_index = {index: value for index, value in checkpoints}
        if len(by_index) != len(checkpoints):
            raise ValueError(f"raw-grid {label} checkpoints contain duplicate indices")
        required = {0, count // 2, count - 1}
        if not required.issubset(by_index):
            raise ValueError(
                f"raw-grid {label} checkpoints do not prove the sample-at-zero contract"
            )
        for index, value in checkpoints:
            if index < 0 or index >= count or not math.isfinite(value):
                raise ValueError(f"raw-grid {label} checkpoint is invalid")
            expected = minimum + index * spacing
            if not _float_matches(value, expected):
                raise ValueError(
                    f"raw-grid {label} labels violate the sample-at-zero coordinate contract"
                )
        if not _float_matches(by_index[count // 2], 0.0):
            raise ValueError(f"raw-grid {label} center is not the sample at zero")

    validate_axis(
        evidence.x_checkpoints,
        count=evidence.nx,
        minimum=evidence.min_x_mm,
        spacing=evidence.dx_mm,
        label="X",
    )
    validate_axis(
        evidence.y_checkpoints,
        count=evidence.ny,
        minimum=evidence.min_y_mm,
        spacing=evidence.dy_mm,
        label="Y",
    )

    z_values = {(ix, iy): value for ix, iy, value in evidence.z_checkpoints}
    api_values = {
        (ix, iy): value for ix, iy, value in evidence.values_checkpoints
    }
    if (
        not z_values
        or len(z_values) != len(evidence.z_checkpoints)
        or len(api_values) != len(evidence.values_checkpoints)
        or z_values.keys() != api_values.keys()
    ):
        raise ValueError("raw-grid Z and Values checkpoints are missing or mismatched")
    for (ix, iy), z_value in z_values.items():
        value = api_values[(ix, iy)]
        if ix < 0 or ix >= evidence.nx or iy < 0 or iy >= evidence.ny:
            raise ValueError("raw-grid Z/Values checkpoint index is outside the array")
        if not math.isfinite(z_value) or not math.isfinite(value):
            raise ValueError("raw-grid Z/Values checkpoints must be finite")
        if not _float_matches(z_value, value):
            raise ValueError("raw-grid Z(ix,iy) does not equal Values[ix,iy]")

    if evidence.api_array_order != "Values[x,y]":
        raise ValueError("raw ZOS-API array order must be Values[x,y]")
    if evidence.package_array_order != "field[y,x]":
        raise ValueError("package/ZBF array order must be field[y,x]")
    if evidence.api_to_package_transform != "transpose":
        raise ValueError("raw API values require exactly one explicit transpose")
    if evidence.origin not in {"synthetic_test", "live_zosapi"}:
        raise ValueError("unknown raw-grid evidence origin")
    if not isinstance(evidence.run_id, str) or not evidence.run_id:
        raise ValueError("raw-grid evidence requires a run id")
    for label, value in (
        ("raw-grid array hash", evidence.raw_grid_array_sha256),
        ("input ZBF hash", evidence.input_zbf_sha256),
        ("output ZBF hash", evidence.output_zbf_sha256),
        ("model hash", evidence.model_sha256),
        ("CFG hash", evidence.cfg_sha256),
    ):
        _require_sha256(value, label=label)

    if raw_values_api_xy is None:
        return None
    raw_values = np.asarray(raw_values_api_xy)
    if raw_values.shape != (evidence.nx, evidence.ny):
        raise ValueError("raw Values array shape does not match Nx/Ny in API [x,y] order")
    if not np.all(np.isfinite(raw_values)):
        raise ValueError("raw Values array must be finite")
    actual_hash = hashlib.sha256(raw_values.tobytes(order="C")).hexdigest()
    if actual_hash != evidence.raw_grid_array_sha256:
        raise ValueError("raw-grid array hash mismatch")
    package_values = _readonly_array(raw_values.T, dtype=raw_values.dtype)
    return package_values


def validate_convention_validation(
    validation: ConventionValidation,
    *,
    surface: int | None = None,
) -> None:
    """Reject incomplete or non-authoritative convention evidence."""

    if not isinstance(validation, ConventionValidation):
        raise ValueError("formal convention validation is required")
    if validation.authoritative is not True:
        raise ValueError("convention validation is not authoritative")
    if validation.validation_status != "passed":
        raise ValueError("convention validation status is not passed")
    if validation.origin not in {"synthetic_test", "live_zosapi"}:
        raise ValueError("unknown convention-validation origin")
    if not isinstance(validation.run_id, str) or not validation.run_id:
        raise ValueError("convention validation requires a run id")
    if validation.phase_unit not in {"radians", "waves"}:
        raise ValueError("unknown raw phase unit")
    if validation.raw_zbf_phasor != "conj(Ex)":
        raise ValueError("raw ZBF phasor conversion must be the uniform conj(Ex) contract")

    sides = dict(validation.surface_sides)
    signs = dict(validation.axis_signs)
    if len(sides) != len(validation.surface_sides) or sides != _EXPECTED_SURFACE_SIDES:
        raise ValueError("surface-side validation does not contain the five fixed after sides")
    if len(signs) != len(validation.axis_signs) or signs != _EXPECTED_AXIS_SIGNS:
        raise ValueError("branch/axis validation does not match the fixed surface registry")
    if surface is not None and surface not in sides:
        raise ValueError("requested surface is absent from convention validation")

    if not validation.raw_grid_evidence:
        raise ValueError("raw-grid evidence is missing")
    actual_evidence_hashes = tuple(
        evidence.evidence_sha256 for evidence in validation.raw_grid_evidence
    )
    if validation.raw_grid_evidence_sha256 != actual_evidence_hashes:
        raise ValueError("raw-grid evidence hash mismatch")
    for evidence in validation.raw_grid_evidence:
        validate_raw_grid_contract(evidence)
        if evidence.model_sha256 != validation.model_sha256:
            raise ValueError("raw-grid model hash does not match convention validation")
        if evidence.cfg_sha256 != validation.cfg_sha256:
            raise ValueError("raw-grid CFG hash does not match convention validation")
        if evidence.run_id != validation.run_id:
            raise ValueError("raw-grid run id does not match convention validation")
        if evidence.origin != validation.origin:
            raise ValueError("raw-grid origin does not match convention validation")

    for label, value in (
        ("report hash", validation.report_sha256),
        ("model hash", validation.model_sha256),
        ("CFG hash", validation.cfg_sha256),
    ):
        _require_sha256(value, label=label)
    _require_sha256(validation.evidence_sha256, label="convention evidence hash")


def pilot_from_zbf(beam: LosslessZbf, convention: SurfaceConvention) -> PilotState:
    h = beam.header
    if convention.side != "after" or convention.axis_sign not in {-1, 1}:
        raise ValueError("surface convention must identify an after side and a signed axis")
    if h.units != 0:
        raise ValueError("biconic physical contract requires ZBF millimetre units")
    if not np.allclose(
        [h.zx, h.rx, h.wx],
        [h.zy, h.ry, h.wy],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("axisymmetric reference contract is not satisfied")
    if not np.all(np.isfinite([h.zx, h.rx, h.wx])) or h.rx <= 0.0 or h.wx <= 0.0:
        raise ValueError("ZBF pilot values must be finite and positive")
    expected_rayleigh = (
        np.pi * h.refractive_index * h.wx**2 / h.wavelength_vacuum_mm
    )
    if not np.isclose(h.rx, expected_rayleigh, rtol=1e-8, atol=1e-12):
        raise ValueError("ZBF Rayleigh distance and waist are inconsistent")
    zeta = convention.axis_sign * h.zx
    return PilotState(zeta_mm=zeta, rayleigh_mm=h.rx, waist_mm=h.wx)


def _validate_reference_inputs(
    *, wavelength_vacuum_mm: float, refractive_index: float, signed_distance: float
) -> None:
    if not np.all(
        np.isfinite([wavelength_vacuum_mm, refractive_index, signed_distance])
    ):
        raise ValueError("reference-phase inputs must be finite")
    if wavelength_vacuum_mm <= 0.0 or refractive_index <= 0.0:
        raise ValueError("reference wavelength and refractive index must be positive")


def quadratic_reference_phase(
    grid: UniformGrid2D,
    *,
    wavelength_vacuum_mm: float,
    refractive_index: float,
    signed_waist_distance_mm: float,
) -> np.ndarray:
    _validate_reference_inputs(
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
        signed_distance=signed_waist_distance_mm,
    )
    if signed_waist_distance_mm == 0.0:
        return np.zeros((grid.ny, grid.nx), dtype=np.float64)
    x, y = np.meshgrid(grid.x_mm, grid.y_mm)
    k = 2 * np.pi * refractive_index / wavelength_vacuum_mm
    return k * (x * x + y * y) / (2 * signed_waist_distance_mm)


def spherical_reference_phase(
    grid: UniformGrid2D,
    *,
    wavelength_vacuum_mm: float,
    refractive_index: float,
    signed_waist_distance_mm: float,
) -> np.ndarray:
    _validate_reference_inputs(
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
        signed_distance=signed_waist_distance_mm,
    )
    if signed_waist_distance_mm == 0.0:
        return np.zeros((grid.ny, grid.nx), dtype=np.float64)
    x, y = np.meshgrid(grid.x_mm, grid.y_mm)
    z = signed_waist_distance_mm
    radius_squared = x * x + y * y
    sag = radius_squared / (np.sqrt(z * z + radius_squared) + abs(z))
    k = 2 * np.pi * refractive_index / wavelength_vacuum_mm
    return k * np.sign(z) * sag


def reference_phases(
    grid: UniformGrid2D,
    pilot: PilotState,
    *,
    wavelength_vacuum_mm: float,
    refractive_index: float,
) -> ReferencePhases:
    if pilot.inside:
        zeros = np.zeros((grid.ny, grid.nx), dtype=np.float64)
        return ReferencePhases(q_rad=zeros.copy(), phi_rad=zeros)
    return ReferencePhases(
        q_rad=quadratic_reference_phase(
            grid,
            wavelength_vacuum_mm=wavelength_vacuum_mm,
            refractive_index=refractive_index,
            signed_waist_distance_mm=pilot.zeta_mm,
        ),
        phi_rad=spherical_reference_phase(
            grid,
            wavelength_vacuum_mm=wavelength_vacuum_mm,
            refractive_index=refractive_index,
            signed_waist_distance_mm=pilot.zeta_mm,
        ),
    )


def zbf_payload_to_point_values(
    values: np.ndarray,
    grid: UniformGrid2D,
    *,
    sample_value_convention: SampleValueConvention,
) -> np.ndarray:
    point_values = np.array(values, dtype=np.complex128, order="C", copy=True)
    if sample_value_convention == "point_value":
        return point_values
    if sample_value_convention == "cell_energy":
        return point_values / np.sqrt(grid.pixel_area_mm2)
    raise ValueError("unknown ZBF sample-value convention")


def point_values_to_zbf_payload(
    values: np.ndarray,
    grid: UniformGrid2D,
    *,
    sample_value_convention: SampleValueConvention,
) -> np.ndarray:
    point_values = np.array(values, dtype=np.complex128, order="C", copy=True)
    if sample_value_convention == "point_value":
        return point_values
    if sample_value_convention == "cell_energy":
        return point_values * np.sqrt(grid.pixel_area_mm2)
    raise ValueError("unknown ZBF sample-value convention")


def _complex_payload_bytes(values: np.ndarray) -> bytes:
    native = np.asarray(values, dtype=np.complex128).reshape(-1)
    return native.astype("<c16", copy=False).tobytes(order="C")


def _copy_and_validate_zbf_payload(beam: LosslessZbf) -> tuple[np.ndarray, np.ndarray | None]:
    h = beam.header
    if h.is_polarized not in {0, 1}:
        raise ValueError("ZBF polarization flag must be zero or one")
    ex = np.array(beam.ex, dtype=np.complex128, order="C", copy=True)
    ey = (
        None
        if beam.ey is None
        else np.array(beam.ey, dtype=np.complex128, order="C", copy=True)
    )
    expected_shape = (h.ny, h.nx)
    if ex.shape != expected_shape:
        raise ValueError("ZBF Ex shape does not match its header")
    if h.is_polarized and ey is None:
        raise ValueError("polarized ZBF is missing Ey")
    if not h.is_polarized and ey is not None:
        raise ValueError("unpolarized ZBF unexpectedly contains Ey")
    if ey is not None and ey.shape != expected_shape:
        raise ValueError("ZBF Ey shape does not match its header")
    if not np.all(np.isfinite(ex)) or (ey is not None and not np.all(np.isfinite(ey))):
        raise ValueError("ZBF field payloads must be finite")

    digest = hashlib.sha256()
    digest.update(h.raw_bytes)
    digest.update(_complex_payload_bytes(ex))
    if ey is not None:
        digest.update(_complex_payload_bytes(ey))
    digest.update(beam.trailing_bytes)
    if digest.hexdigest() != beam.source_sha256:
        raise ValueError("LosslessZbf content hash does not match its field payload")
    return ex, ey


def physical_field_from_zbf(
    beam: LosslessZbf,
    *,
    convention: SurfaceConvention,
    convention_validation: ConventionValidation,
    sample_value_convention: SampleValueConvention,
) -> MappedZbfField:
    validate_convention_validation(
        convention_validation, surface=convention.surface
    )
    if (
        dict(convention_validation.surface_sides)[convention.surface]
        != convention.side
        or dict(convention_validation.axis_signs)[convention.surface]
        != convention.axis_sign
    ):
        raise ValueError("surface convention does not match authoritative validation")
    ex, ey = _copy_and_validate_zbf_payload(beam)
    if ey is not None:
        raise NotImplementedError(
            "biconic scalar ranking requires an approved Jones-field extension"
        )
    grid = UniformGrid2D.centered(
        nx=beam.header.nx,
        ny=beam.header.ny,
        dx_mm=beam.header.dx,
        dy_mm=beam.header.dy,
    )
    pilot = pilot_from_zbf(beam, convention)
    refs = reference_phases(
        grid,
        pilot,
        wavelength_vacuum_mm=beam.header.wavelength_vacuum_mm,
        refractive_index=beam.header.refractive_index,
    )
    point_payload = zbf_payload_to_point_values(
        ex, grid, sample_value_convention=sample_value_convention
    )
    reference_relative = np.conj(point_payload)
    physical = PointField2D(reference_relative * np.exp(1j * refs.phi_rad), grid)
    return MappedZbfField(
        physical=physical,
        reference_relative=reference_relative,
        references=refs,
        pilot=pilot,
        source_sha256=beam.source_sha256,
        convention_evidence_sha256=convention_validation.evidence_sha256,
        sample_value_convention=sample_value_convention,
    )


__all__ = [
    "ConventionValidation",
    "MappedZbfField",
    "PilotState",
    "RawGridEvidence",
    "ReferencePhases",
    "physical_field_from_zbf",
    "pilot_from_zbf",
    "point_values_to_zbf_payload",
    "quadratic_reference_phase",
    "reference_phases",
    "spherical_reference_phase",
    "validate_convention_validation",
    "validate_raw_grid_contract",
    "zbf_payload_to_point_values",
]
