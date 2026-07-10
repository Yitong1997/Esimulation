from __future__ import annotations

import hashlib
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sandbox.free_space_algorithm_identification.biconic_case import (
    S7,
    S8,
    S12,
    S13,
    S14,
)
from sandbox.free_space_algorithm_identification.field_contract import (
    ConventionValidation,
    PilotState,
    RawGridEvidence,
    physical_field_from_zbf,
    pilot_from_zbf,
    point_values_to_zbf_payload,
    quadratic_reference_phase,
    reference_phases,
    spherical_reference_phase,
    validate_convention_validation,
    validate_raw_grid_contract,
    zbf_payload_to_point_values,
)
from sandbox.free_space_algorithm_identification.models import (
    SurfaceConvention,
    UniformGrid2D,
)
from sandbox.free_space_algorithm_identification.zbf_binary import (
    LosslessZbf,
    RawZbfHeader,
)


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SHA_E = "e" * 64


def _raw_header(
    *,
    nx: int = 4,
    ny: int = 4,
    dx_mm: float = 0.1,
    dy_mm: float = 0.2,
    raw_z_mm: float = -2.0,
    rayleigh_mm: float = 1.0,
    wavelength_vacuum_mm: float = 0.02,
    refractive_index: float = 2.0,
    units: int = 0,
    is_polarized: int = 0,
    zy_mm: float | None = None,
    ry_mm: float | None = None,
    wy_mm: float | None = None,
) -> RawZbfHeader:
    waist_mm = np.sqrt(
        rayleigh_mm * wavelength_vacuum_mm / (np.pi * refractive_index)
    )
    ints = (1, nx, ny, is_polarized, units, 0, 0, 0, 0)
    doubles = (
        dx_mm,
        dy_mm,
        raw_z_mm,
        rayleigh_mm,
        waist_mm,
        raw_z_mm if zy_mm is None else zy_mm,
        rayleigh_mm if ry_mm is None else ry_mm,
        waist_mm if wy_mm is None else wy_mm,
        wavelength_vacuum_mm,
        refractive_index,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    return RawZbfHeader.from_bytes(struct.pack("<9i20d", *ints, *doubles))


def make_small_unpolarized_lossless_zbf(
    *,
    raw_z_mm: float = -2.0,
    rayleigh_mm: float = 1.0,
    ex: np.ndarray | None = None,
    **header_overrides: object,
) -> LosslessZbf:
    header = _raw_header(
        raw_z_mm=raw_z_mm,
        rayleigh_mm=rayleigh_mm,
        **header_overrides,
    )
    if ex is None:
        rows, columns = np.indices((header.ny, header.nx))
        ex = (
            0.25
            + 0.31 * columns
            - 0.17 * rows
            + 1j * (-0.4 + 0.23 * columns + 0.11 * rows)
        )
    return LosslessZbf(
        path=None,
        source_sha256="",
        header=header,
        ex=np.asarray(ex, dtype=np.complex128),
        ey=None,
        trailing_bytes=b"independent-test-oracle",
    )


def _api_values() -> np.ndarray:
    return np.arange(24, dtype="<f8").reshape(6, 4)


def make_raw_grid_evidence(
    *,
    values_api_xy: np.ndarray | None = None,
    x_offset_mm: float = 0.0,
    origin: str = "synthetic_test",
) -> RawGridEvidence:
    values = _api_values() if values_api_xy is None else values_api_xy
    nx, ny = values.shape
    dx_mm = 0.25
    dy_mm = 0.5
    min_x_mm = -(nx // 2) * dx_mm
    min_y_mm = -(ny // 2) * dy_mm
    x_indices = (0, nx // 2, nx - 1)
    y_indices = (0, ny // 2, ny - 1)
    sample_indices = ((0, 0), (nx // 2, ny // 2), (nx - 1, ny - 1))
    return RawGridEvidence(
        nx=nx,
        ny=ny,
        min_x_mm=min_x_mm,
        min_y_mm=min_y_mm,
        dx_mm=dx_mm,
        dy_mm=dy_mm,
        x_checkpoints=tuple(
            (index, min_x_mm + index * dx_mm + x_offset_mm)
            for index in x_indices
        ),
        y_checkpoints=tuple(
            (index, min_y_mm + index * dy_mm) for index in y_indices
        ),
        z_checkpoints=tuple(
            (ix, iy, float(values[ix, iy])) for ix, iy in sample_indices
        ),
        values_checkpoints=tuple(
            (ix, iy, float(values[ix, iy])) for ix, iy in sample_indices
        ),
        raw_grid_array_sha256=hashlib.sha256(
            values.tobytes(order="C")
        ).hexdigest(),
        input_zbf_sha256=_SHA_A,
        output_zbf_sha256=_SHA_B,
        model_sha256=_SHA_C,
        cfg_sha256=_SHA_D,
        run_id="in-memory-contract-test",
        origin=origin,
        api_array_order="Values[x,y]",
        package_array_order="field[y,x]",
        api_to_package_transform="transpose",
    )


def make_in_memory_authoritative_validation_fixture() -> ConventionValidation:
    evidence = make_raw_grid_evidence()
    return ConventionValidation(
        surface_sides=(
            (7, "after"),
            (8, "after"),
            (12, "after"),
            (13, "after"),
            (14, "after"),
        ),
        axis_signs=((7, -1), (8, -1), (12, 1), (13, 1), (14, 1)),
        raw_zbf_phasor="conj(Ex)",
        raw_grid_evidence=(evidence,),
        raw_grid_evidence_sha256=(evidence.evidence_sha256,),
        report_sha256=_SHA_E,
        model_sha256=_SHA_C,
        cfg_sha256=_SHA_D,
        phase_unit="radians",
        run_id="in-memory-contract-test",
        origin="synthetic_test",
        authoritative=True,
        validation_status="passed",
    )


def test_reference_uses_signed_waist_distance_not_gaussian_curvature() -> None:
    grid = UniformGrid2D.centered(nx=4, ny=4, dx_mm=0.1, dy_mm=0.1)
    q = quadratic_reference_phase(
        grid,
        wavelength_vacuum_mm=0.02,
        refractive_index=2.0,
        signed_waist_distance_mm=-2.0,
    )
    phi = spherical_reference_phase(
        grid,
        wavelength_vacuum_mm=0.02,
        refractive_index=2.0,
        signed_waist_distance_mm=-2.0,
    )
    x, y = np.meshgrid(grid.x_mm, grid.y_mm)
    k = 2 * np.pi * 2.0 / 0.02
    np.testing.assert_allclose(q, k * (x * x + y * y) / -4.0)
    np.testing.assert_allclose(
        phi, -k * (np.sqrt(4.0 + x * x + y * y) - 2.0)
    )


def test_pilot_inside_is_derived_with_a_strict_rayleigh_boundary() -> None:
    pilot = PilotState(zeta_mm=-0.015, rayleigh_mm=0.761, waist_mm=0.0508)
    grid = UniformGrid2D.centered(nx=8, ny=8, dx_mm=0.01, dy_mm=0.01)
    phases = reference_phases(
        grid,
        pilot,
        wavelength_vacuum_mm=0.01064,
        refractive_index=1.0,
    )
    np.testing.assert_array_equal(phases.phi_rad, 0.0)
    np.testing.assert_array_equal(phases.q_rad, 0.0)
    with pytest.raises(ValueError):
        phases.phi_rad.setflags(write=True)
    with pytest.raises(ValueError):
        phases.q_rad.setflags(write=True)
    assert PilotState(zeta_mm=0.999, rayleigh_mm=1.0, waist_mm=0.1).inside
    assert not PilotState(zeta_mm=1.0, rayleigh_mm=1.0, waist_mm=0.1).inside
    assert not PilotState(zeta_mm=-1.0, rayleigh_mm=1.0, waist_mm=0.1).inside
    for kwargs in (
        {"zeta_mm": np.nan, "rayleigh_mm": 1.0, "waist_mm": 0.1},
        {"zeta_mm": 0.0, "rayleigh_mm": 0.0, "waist_mm": 0.1},
    ):
        with pytest.raises(ValueError):
            PilotState(**kwargs)


def test_pilot_from_zbf_derives_axis_sign_and_validates_header_physics() -> None:
    beam = make_small_unpolarized_lossless_zbf(raw_z_mm=-2.5)
    assert pilot_from_zbf(beam, S7).zeta_mm == 2.5
    assert pilot_from_zbf(beam, S12).zeta_mm == -2.5
    template = make_small_unpolarized_lossless_zbf()
    bad_formula_bytes = bytearray(template.header.raw_bytes)
    struct.pack_into("<d", bad_formula_bytes, 36 + 3 * 8, 1.1)
    struct.pack_into("<d", bad_formula_bytes, 36 + 6 * 8, 1.1)
    bad_formula = replace(
        template, header=RawZbfHeader.from_bytes(bytes(bad_formula_bytes))
    )
    for beam in (
        make_small_unpolarized_lossless_zbf(units=1),
        make_small_unpolarized_lossless_zbf(zy_mm=-2.1),
        bad_formula,
    ):
        with pytest.raises(ValueError):
            pilot_from_zbf(beam, S7)


def test_point_and_cell_payload_conversions_have_the_expected_area_factor() -> None:
    grid = UniformGrid2D.centered(nx=4, ny=4, dx_mm=0.25, dy_mm=0.04)
    payload = np.array([[1.0 + 2.0j, -3.0 + 4.0j]], dtype=np.complex128)
    point = zbf_payload_to_point_values(
        payload, grid, sample_value_convention="cell_energy"
    )
    np.testing.assert_allclose(point, payload / np.sqrt(grid.pixel_area_mm2))
    np.testing.assert_allclose(
        point_values_to_zbf_payload(
            point, grid, sample_value_convention="cell_energy"
        ),
        payload,
    )
    np.testing.assert_array_equal(
        zbf_payload_to_point_values(
            payload, grid, sample_value_convention="point_value"
        ),
        payload,
    )


@pytest.mark.parametrize(
    ("convention", "raw_z_over_rayleigh"),
    [
        (S7, -2.0),
        (S8, 2.0),
        (S12, 0.5),
        (S13, -0.5),
        (S14, 1.0),
    ],
    ids=("S7-positive-outside", "S8-negative-outside", "S12-positive-inside", "S13-negative-inside", "S14-boundary"),
)
def test_all_registered_raw_zbf_surfaces_use_one_conjugated_phasor_oracle(
    convention: SurfaceConvention,
    raw_z_over_rayleigh: float,
) -> None:
    rayleigh = 1.25
    beam = make_small_unpolarized_lossless_zbf(
        raw_z_mm=raw_z_over_rayleigh * rayleigh,
        rayleigh_mm=rayleigh,
    )
    mapped = physical_field_from_zbf(
        beam,
        convention=convention,
        convention_validation=make_in_memory_authoritative_validation_fixture(),
        sample_value_convention="point_value",
    )
    zeta = convention.axis_sign * beam.header.zx
    x, y = np.meshgrid(mapped.physical.grid.x_mm, mapped.physical.grid.y_mm)
    if abs(zeta) < rayleigh:
        phi = np.zeros_like(x)
    else:
        k = 2 * np.pi * beam.header.refractive_index / beam.header.wavelength_vacuum_mm
        phi = k * np.sign(zeta) * (
            np.sqrt(zeta * zeta + x * x + y * y) - abs(zeta)
        )
    np.testing.assert_allclose(
        mapped.physical.values, np.conj(beam.ex) * np.exp(1j * phi)
    )
    assert mapped.pilot.inside is (abs(zeta) < rayleigh)
    assert mapped.convention_evidence_sha256 == (
        make_in_memory_authoritative_validation_fixture().evidence_sha256
    )
    with pytest.raises(ValueError):
        mapped.reference_relative.setflags(write=True)


def test_physical_mapping_copies_and_defensively_validates_lossless_payload() -> None:
    validation = make_in_memory_authoritative_validation_fixture()
    beam = make_small_unpolarized_lossless_zbf()
    mapped = physical_field_from_zbf(
        beam,
        convention=S7,
        convention_validation=validation,
        sample_value_convention="point_value",
    )
    frozen_reference = mapped.reference_relative.copy()
    beam.ex[...] = 99.0 - 101.0j
    np.testing.assert_array_equal(mapped.reference_relative, frozen_reference)
    assert not np.shares_memory(mapped.reference_relative, beam.ex)

    original = make_small_unpolarized_lossless_zbf()
    stale = LosslessZbf(
        path=Path("synthetic-stale-hash.ZBF"),
        source_sha256="0" * 64,
        header=original.header,
        ex=original.ex,
        ey=None,
        trailing_bytes=original.trailing_bytes,
    )
    invalid_beams = (
        replace(
            make_small_unpolarized_lossless_zbf(),
            ex=np.ones((2, 2), dtype=np.complex128),
        ),
        replace(
            make_small_unpolarized_lossless_zbf(),
            ey=np.ones((4, 4), dtype=np.complex128),
        ),
        replace(make_small_unpolarized_lossless_zbf(is_polarized=1), ey=None),
        replace(
            make_small_unpolarized_lossless_zbf(),
            ex=np.full((4, 4), np.nan + 1j, dtype=np.complex128),
        ),
        stale,
    )
    for invalid in invalid_beams:
        with pytest.raises(ValueError):
            physical_field_from_zbf(
                invalid,
                convention=S7,
                convention_validation=validation,
                sample_value_convention="point_value",
            )
    polarized = replace(
        make_small_unpolarized_lossless_zbf(is_polarized=1),
        ey=np.ones((4, 4), dtype=np.complex128),
    )
    with pytest.raises(NotImplementedError):
        physical_field_from_zbf(
            polarized,
            convention=S7,
            convention_validation=validation,
            sample_value_convention="point_value",
        )
    with pytest.raises(ValueError, match="surface convention"):
        physical_field_from_zbf(
            original,
            convention=SurfaceConvention(7, "after", 1),
            convention_validation=validation,
            sample_value_convention="point_value",
        )


def test_valid_raw_grid_contract_uses_sample_at_zero_and_one_transpose() -> None:
    values_api_xy = _api_values()
    evidence = make_raw_grid_evidence(values_api_xy=values_api_xy)
    package_values_yx = validate_raw_grid_contract(
        evidence, raw_values_api_xy=values_api_xy
    )
    assert evidence.min_x_mm == -(evidence.nx / 2) * evidence.dx_mm
    assert evidence.min_y_mm == -(evidence.ny / 2) * evidence.dy_mm
    assert dict(evidence.x_checkpoints)[evidence.nx // 2] == 0.0
    assert dict(evidence.y_checkpoints)[evidence.ny // 2] == 0.0
    np.testing.assert_array_equal(package_values_yx, values_api_xy.T)
    assert package_values_yx.shape == (evidence.ny, evidence.nx)
    with pytest.raises(ValueError):
        package_values_yx.setflags(write=True)


def test_raw_grid_contract_rejects_half_step_value_and_hash_mismatches() -> None:
    values = _api_values()
    evidence = make_raw_grid_evidence(values_api_xy=values)
    for invalid in (
        make_raw_grid_evidence(x_offset_mm=0.125),
        replace(
            evidence,
            values_checkpoints=((0, 0, 999.0), (3, 2, 14.0), (5, 3, 23.0)),
        ),
        replace(evidence, api_array_order="Values[y,x]"),
        replace(evidence, raw_grid_array_sha256="0" * 64),
    ):
        with pytest.raises(ValueError):
            validate_raw_grid_contract(invalid, raw_values_api_xy=values)


def test_authoritative_validation_has_canonical_provenance_hash() -> None:
    first = make_in_memory_authoritative_validation_fixture()
    second = make_in_memory_authoritative_validation_fixture()
    assert first.evidence_sha256 == second.evidence_sha256
    assert len(first.evidence_sha256) == 64
    assert replace(first, run_id="different").evidence_sha256 != first.evidence_sha256
    with pytest.raises(TypeError):
        ConventionValidation(
            **{
                **first.__dict__,
                "evidence_sha256": "caller-controlled",
            }
        )
    assert first.origin == "synthetic_test"
    assert first.authoritative is True
    assert not hasattr(first, "artifact_ref")
    validate_convention_validation(first, surface=14)


def test_convention_validation_fails_closed() -> None:
    valid = make_in_memory_authoritative_validation_fixture()
    invalid_cases = (
        replace(valid, authoritative=False),
        replace(valid, raw_grid_evidence=()),
        replace(valid, raw_grid_evidence_sha256=("0" * 64,)),
        replace(valid, raw_zbf_phasor=((7, "conj(Ex)"), (8, "Ex"))),
        replace(
            valid,
            axis_signs=((7, 1), (8, -1), (12, 1), (13, 1), (14, 1)),
        ),
        replace(valid, phase_unit="degrees"),
        replace(valid, model_sha256="f" * 64),
    )
    for validation in invalid_cases:
        with pytest.raises(ValueError):
            validate_convention_validation(validation, surface=7)
