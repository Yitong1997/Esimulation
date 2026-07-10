from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from sandbox.free_space_algorithm_identification.biconic_case import S7
from sandbox.free_space_algorithm_identification.derived_inputs import (
    derive_zbf_input,
    validate_derived_input,
)
from sandbox.free_space_algorithm_identification.field_contract import (
    pilot_from_zbf,
    reference_phases,
)
from sandbox.free_space_algorithm_identification.models import UniformGrid2D
from sandbox.free_space_algorithm_identification.zbf_binary import (
    HEADER_BYTES,
    LosslessZbf,
    RawZbfHeader,
    patch_sampling_header,
    read_lossless_zbf,
    write_lossless_zbf,
)


def _periodic_fields(*, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(nx, dtype=np.float64)[np.newaxis, :]
    y = np.arange(ny, dtype=np.float64)[:, np.newaxis]
    ex = (
        0.8 * np.exp(2j * np.pi * (x / nx + y / ny))
        + 0.23 * np.exp(2j * np.pi * (2 * x / nx - y / ny))
    )
    ey = (
        -0.31j * np.exp(2j * np.pi * (-x / nx + 2 * y / ny))
        + 0.17 * np.exp(2j * np.pi * (x / nx + 2 * y / ny))
    )
    return ex.astype(np.complex128), ey.astype(np.complex128)


def _compact_fields(*, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    ex = np.zeros((ny, nx), dtype=np.complex128)
    ey = np.zeros((ny, nx), dtype=np.complex128)
    ex[2:4, 3:5] = np.array(
        [[1.0 + 2.0j, -3.0 + 4.0j], [5.0 - 6.0j, -7.0 - 8.0j]]
    )
    ey[2:4, 3:5] = np.array(
        [[-0.25 + 0.5j, 0.75 - 1.0j], [1.25 + 1.5j, -1.75 + 2.0j]]
    )
    return ex, ey


def _write_source(
    path: Path,
    *,
    compact: bool = False,
    edge_loaded: bool = False,
) -> Path:
    nx, ny = 8, 6
    wavelength_vacuum_mm = 0.01064
    refractive_index = 1.37
    rayleigh_mm = 1.25
    waist_mm = np.sqrt(
        rayleigh_mm * wavelength_vacuum_mm / (np.pi * refractive_index)
    )
    ints = (17, nx, ny, 1, 0, 101, -202, 303, -404)
    doubles = (
        0.2,
        0.3,
        -3.0,
        rayleigh_mm,
        waist_mm,
        -3.0,
        rayleigh_mm,
        waist_mm,
        wavelength_vacuum_mm,
        refractive_index,
        0.875,
        0.625,
        11.25,
        -12.5,
        13.75,
        -14.0,
        15.5,
        -16.75,
        17.0,
        -18.25,
    )
    header = RawZbfHeader.from_bytes(struct.pack("<9i20d", *ints, *doubles))
    ex, ey = (
        _compact_fields(nx=nx, ny=ny)
        if compact
        else _periodic_fields(nx=nx, ny=ny)
    )
    if edge_loaded:
        ex[0, :] = 1.0 + 0.5j
        ey[:, -1] = -0.75 + 0.25j
    beam = LosslessZbf(
        path=None,
        source_sha256="",
        header=header,
        ex=ex,
        ey=ey,
        trailing_bytes=b"TASK5-UNINTERPRETED-TAIL",
    )
    write_lossless_zbf(path, beam)
    return path


def _refined_grid() -> UniformGrid2D:
    return UniformGrid2D.centered(
        nx=16,
        ny=12,
        dx_mm=0.1,
        dy_mm=0.15,
    )


def _extended_grid() -> UniformGrid2D:
    return UniformGrid2D.centered(
        nx=12,
        ny=10,
        dx_mm=0.2,
        dy_mm=0.3,
    )


def _center_overlap(
    source_shape: tuple[int, int], target_shape: tuple[int, int]
) -> tuple[slice, slice]:
    source_ny, source_nx = source_shape
    target_ny, target_nx = target_shape
    x0 = target_nx // 2 - source_nx // 2
    y0 = target_ny // 2 - source_ny // 2
    return slice(y0, y0 + source_ny), slice(x0, x0 + source_nx)


def test_native_case_is_a_byte_exact_copy(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "native.ZBF")
    output = tmp_path / "native-copy.ZBF"
    beam = read_lossless_zbf(source)

    result = derive_zbf_input(
        source,
        output,
        target_grid=beam.grid,
        strategy="exact_copy",
        convention=S7,
        sample_value_convention="point_value",
    )

    assert output.read_bytes() == source.read_bytes()
    assert result.source_sha256 == result.output_sha256
    assert result.validation.byte_exact_copy
    assert result.validation.power_normalization_applied is False

    shifted_grid = UniformGrid2D(
        x_mm=beam.grid.x_mm + beam.grid.dx_mm,
        y_mm=beam.grid.y_mm,
    )
    with pytest.raises(ValueError, match="centered"):
        derive_zbf_input(
            source,
            tmp_path / "shifted-copy.ZBF",
            target_grid=shifted_grid,
            strategy="exact_copy",
            convention=S7,
            sample_value_convention="point_value",
        )

    with pytest.raises(ValueError, match="chained.*provenance"):
        derive_zbf_input(
            source,
            tmp_path / "chained-missing-provenance.ZBF",
            target_grid=beam.grid,
            strategy="chained_zemax_output",
            convention=S7,
            sample_value_convention="point_value",
        )

    producer_case = "S12_S13:ZO1"
    upstream_run_case_sha256 = "f" * 64
    for index, (case_id, digest) in enumerate(
        (
            (None, upstream_run_case_sha256),
            (producer_case, None),
            (producer_case, "not-a-sha256"),
            (producer_case, beam.source_sha256),
        )
    ):
        with pytest.raises(ValueError, match="chained.*provenance"):
            derive_zbf_input(
                source,
                tmp_path / f"chained-invalid-provenance-{index}.ZBF",
                target_grid=beam.grid,
                strategy="chained_zemax_output",
                convention=S7,
                sample_value_convention="point_value",
                producer_case=case_id,
                upstream_run_case_sha256=digest,
            )

    chained = derive_zbf_input(
        source,
        tmp_path / "chained-valid.ZBF",
        target_grid=beam.grid,
        strategy="chained_zemax_output",
        convention=S7,
        sample_value_convention="point_value",
        producer_case=producer_case,
        upstream_run_case_sha256=upstream_run_case_sha256,
    )
    assert chained.producer_case == producer_case
    assert chained.upstream_run_case_sha256 == upstream_run_case_sha256
    assert chained.source_sha256 == beam.source_sha256
    assert chained.upstream_run_case_sha256 != chained.source_sha256
    assert chained.validation.producer_case == producer_case
    assert (
        chained.validation.upstream_run_case_sha256
        == upstream_run_case_sha256
    )


def test_fixed_window_refinement_recovers_original_complex_nodes(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "periodic.ZBF")
    source_beam = read_lossless_zbf(source)

    for sample_value_convention in ("point_value", "cell_energy"):
        output = tmp_path / f"refined-{sample_value_convention}.ZBF"
        result = derive_zbf_input(
            source,
            output,
            target_grid=_refined_grid(),
            strategy="fourier_refine_fixed_window",
            convention=S7,
            sample_value_convention=sample_value_convention,
        )
        refined = read_lossless_zbf(output)
        raw_factor = (
            1.0
            if sample_value_convention == "point_value"
            else np.sqrt(refined.grid.pixel_area_mm2 / source_beam.grid.pixel_area_mm2)
        )
        np.testing.assert_allclose(
            refined.ex[::2, ::2],
            source_beam.ex * raw_factor,
            rtol=1e-10,
            atol=1e-12,
        )
        assert refined.ey is not None and source_beam.ey is not None
        np.testing.assert_allclose(
            refined.ey[::2, ::2],
            source_beam.ey * raw_factor,
            rtol=1e-10,
            atol=1e-12,
        )
        assert result.validation.slow_field_common_node_relative_l2 <= 1e-10
        assert result.validation.physical_phase_rms_waves <= 1e-8


def test_fixed_step_extension_preserves_original_ex_and_ey_exactly(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "compact.ZBF", compact=True)
    output = tmp_path / "extended.ZBF"
    original = read_lossless_zbf(source)

    result = derive_zbf_input(
        source,
        output,
        target_grid=_extended_grid(),
        strategy="zero_extend_fixed_sampling",
        convention=S7,
        sample_value_convention="cell_energy",
    )

    extended = read_lossless_zbf(output)
    ys, xs = _center_overlap(original.ex.shape, extended.ex.shape)
    assert extended.ey is not None and original.ey is not None
    assert extended.ex[ys, xs].tobytes() == original.ex.tobytes()
    assert extended.ey[ys, xs].tobytes() == original.ey.tobytes()
    added = np.ones(extended.ex.shape, dtype=bool)
    added[ys, xs] = False
    assert extended.ex[added].tobytes() == np.zeros(added.sum(), np.complex128).tobytes()
    assert extended.ey[added].tobytes() == np.zeros(added.sum(), np.complex128).tobytes()
    assert result.validation.fixed_step_overlap_bitwise
    assert result.validation.added_samples_exact_zero
    assert result.validation.edge_energy_target_met
    assert result.validation.edge_energy_hard_gate_fraction == 1e-10

    combined = derive_zbf_input(
        source,
        tmp_path / "extended-and-refined.ZBF",
        target_grid=UniformGrid2D.centered(
            nx=24,
            ny=20,
            dx_mm=0.1,
            dy_mm=0.15,
        ),
        strategy="zero_extend_then_fourier_refine",
        convention=S7,
        sample_value_convention="point_value",
    )
    assert combined.validation.fixed_step_overlap_bitwise
    assert combined.validation.added_samples_exact_zero
    combined_beam = read_lossless_zbf(combined.output_path)
    intermediate_grid = UniformGrid2D.centered(
        nx=12,
        ny=10,
        dx_mm=0.2,
        dy_mm=0.3,
    )
    intermediate_ex = np.zeros((10, 12), dtype=np.complex128)
    intermediate_ys, intermediate_xs = _center_overlap(
        original.ex.shape, intermediate_ex.shape
    )
    intermediate_ex[intermediate_ys, intermediate_xs] = original.ex
    output_common = combined_beam.ex[::2, ::2]
    expected_l2 = np.linalg.norm(output_common - intermediate_ex) / np.linalg.norm(
        intermediate_ex
    )

    pilot = pilot_from_zbf(original, S7)
    intermediate_phi = reference_phases(
        intermediate_grid,
        pilot,
        wavelength_vacuum_mm=original.header.wavelength_vacuum_mm,
        refractive_index=original.header.refractive_index,
    ).phi_rad
    output_phi = reference_phases(
        combined_beam.grid,
        pilot,
        wavelength_vacuum_mm=original.header.wavelength_vacuum_mm,
        refractive_index=original.header.refractive_index,
    ).phi_rad[::2, ::2]
    intermediate_physical = np.conj(intermediate_ex) * np.exp(1j * intermediate_phi)
    output_physical = np.conj(output_common) * np.exp(1j * output_phi)
    weights = np.abs(intermediate_physical) ** 2
    valid = (weights > 0.0) & (np.abs(output_physical) > 0.0)
    phase_waves = np.angle(
        output_physical[valid] * np.conj(intermediate_physical[valid])
    ) / (2 * np.pi)
    expected_phase_rms = np.sqrt(
        np.sum(weights[valid] * phase_waves**2) / np.sum(weights[valid])
    )
    intermediate_i = np.abs(intermediate_physical) ** 2
    output_i = np.abs(output_physical) ** 2
    expected_intensity_rms = 100 * np.sqrt(
        np.mean(
            (
                output_i / np.max(output_i)
                - intermediate_i / np.max(intermediate_i)
            )
            ** 2
        )
    )
    ex_metrics = next(
        item for item in combined.validation.components if item.component == "Ex"
    )
    assert ex_metrics.slow_field_common_node_relative_l2 == pytest.approx(
        expected_l2, rel=1e-12, abs=0.0
    )
    assert ex_metrics.physical_phase_rms_waves == pytest.approx(
        expected_phase_rms, rel=1e-12, abs=0.0
    )
    assert ex_metrics.normalized_intensity_rms_percent == pytest.approx(
        expected_intensity_rms, rel=1e-12, abs=0.0
    )
    assert np.all(
        np.isfinite(
            [
                expected_l2,
                expected_phase_rms,
                expected_intensity_rms,
            ]
        )
    )


def test_zero_extension_rejects_excessive_edge_energy(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "edge-loaded.ZBF", compact=True, edge_loaded=True)
    for index, requested_maximum in enumerate((1e-10, 1.0)):
        output = tmp_path / f"must-not-exist-{index}.ZBF"
        with pytest.raises(ValueError, match="edge-energy"):
            derive_zbf_input(
                source,
                output,
                target_grid=_extended_grid(),
                strategy="zero_extend_fixed_sampling",
                convention=S7,
                sample_value_convention="point_value",
                max_edge_energy_fraction=requested_maximum,
            )
        assert not output.exists()

    template = read_lossless_zbf(
        _write_source(tmp_path / "finite-template.ZBF", compact=True)
    )
    for index, invalid_value in enumerate(
        (np.nan + 0j, np.inf + 0j, 1e308 + 0j)
    ):
        invalid_ex = template.ex.copy()
        invalid_ex[2, 3] = invalid_value
        invalid_source = tmp_path / f"nonfinite-or-overflow-{index}.ZBF"
        write_lossless_zbf(
            invalid_source,
            LosslessZbf(
                path=None,
                source_sha256="",
                header=template.header,
                ex=invalid_ex,
                ey=template.ey,
                trailing_bytes=template.trailing_bytes,
            ),
        )
        invalid_output = tmp_path / f"nonfinite-or-overflow-output-{index}.ZBF"
        with pytest.raises(ValueError, match="finite"):
            derive_zbf_input(
                invalid_source,
                invalid_output,
                target_grid=_extended_grid(),
                strategy="zero_extend_fixed_sampling",
                convention=S7,
                sample_value_convention="point_value",
            )
        assert not invalid_output.exists()


def test_derived_input_never_applies_power_normalization(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "cell-energy.ZBF")
    output = tmp_path / "cell-energy-refined.ZBF"
    original = read_lossless_zbf(source)

    result = derive_zbf_input(
        source,
        output,
        target_grid=_refined_grid(),
        strategy="fourier_refine_fixed_window",
        convention=S7,
        sample_value_convention="cell_energy",
    )

    refined = read_lossless_zbf(output)
    expected_factor = np.sqrt(
        refined.grid.pixel_area_mm2 / original.grid.pixel_area_mm2
    )
    np.testing.assert_allclose(
        refined.ex[::2, ::2] / original.ex,
        expected_factor,
        rtol=1e-10,
        atol=1e-12,
    )
    assert result.validation.relative_energy_error <= 1e-10
    assert result.validation.power_normalization_applied is False


def test_only_nx_ny_dx_dy_and_payload_may_change(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source.ZBF", compact=True)
    output = tmp_path / "derived.ZBF"
    before = read_lossless_zbf(source)
    derive_zbf_input(
        source,
        output,
        target_grid=_extended_grid(),
        strategy="zero_extend_fixed_sampling",
        convention=S7,
        sample_value_convention="point_value",
    )
    after = read_lossless_zbf(output)
    validation = validate_derived_input(
        source,
        output,
        strategy="zero_extend_fixed_sampling",
        convention=S7,
        sample_value_convention="point_value",
    )

    changed_header_bytes = {
        index
        for index, (left, right) in enumerate(
            zip(before.header.raw_bytes, after.header.raw_bytes, strict=True)
        )
        if left != right
    }
    allowed_header_bytes = set(range(4, 12)) | set(range(36, 52))
    assert changed_header_bytes <= allowed_header_bytes
    assert after.trailing_bytes == before.trailing_bytes
    assert after.header.raw_bytes[HEADER_BYTES - 64 :] == before.header.raw_bytes[
        HEADER_BYTES - 64 :
    ]
    assert validation.unexpected_header_byte_changes == 0

    tampered = tmp_path / "tampered-step.ZBF"
    tampered_beam = LosslessZbf(
        path=None,
        source_sha256="",
        header=patch_sampling_header(
            after.header,
            nx=after.header.nx,
            ny=after.header.ny,
            dx=2 * after.header.dx,
            dy=after.header.dy,
        ),
        ex=after.ex,
        ey=after.ey,
        trailing_bytes=after.trailing_bytes,
    )
    write_lossless_zbf(tampered, tampered_beam)
    with pytest.raises(ValueError, match="fixed-step"):
        validate_derived_input(
            source,
            tampered,
            strategy="zero_extend_fixed_sampling",
            convention=S7,
            sample_value_convention="cell_energy",
        )
