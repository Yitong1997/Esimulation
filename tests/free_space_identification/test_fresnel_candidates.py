from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from sandbox.free_space_algorithm_identification.biconic_case import BICONIC_SEGMENTS
from sandbox.free_space_algorithm_identification import candidates as candidate_module
from sandbox.free_space_algorithm_identification.candidates import (
    candidate_f_q,
    candidate_r_phi_given_phi,
    candidate_r_phi_given_q,
    lift_q_relative_slow_field,
    map_slow_field_to_square,
    run_stock_proper_fq,
)
from sandbox.free_space_algorithm_identification.field_contract import (
    PilotState,
    reference_phases,
)
from sandbox.free_space_algorithm_identification.fresnel import (
    propagate_ptp_fresnel,
    propagate_scaled_fresnel,
    scaled_dft_analytic_factor,
    scaled_dft_cell_samples,
)
from sandbox.free_space_algorithm_identification.models import (
    PointField2D,
    SegmentSpec,
    UniformGrid2D,
)


proper = candidate_module.proper


_WAVELENGTH_MM = 0.05
_REFRACTIVE_INDEX = 1.0


def _segment(key: str, distance_mm: float) -> SegmentSpec:
    original = next(segment for segment in BICONIC_SEGMENTS if segment.key == key)
    return replace(original, model_distance_mm=distance_mm)


def _grid_hash_for_test(grid: UniformGrid2D) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(grid.x_mm, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(grid.y_mm, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _gaussian_start(
    *, grid: UniformGrid2D, pilot: PilotState
) -> PointField2D:
    references = reference_phases(
        grid,
        pilot,
        wavelength_vacuum_mm=_WAVELENGTH_MM,
        refractive_index=_REFRACTIVE_INDEX,
    )
    x, y = np.meshgrid(grid.x_mm, grid.y_mm)
    width = pilot.waist_mm * np.sqrt(
        1.0 + (pilot.zeta_mm / pilot.rayleigh_mm) ** 2
    )
    chi_phi = (pilot.waist_mm / width) * np.exp(-(x * x + y * y) / width**2)
    return PointField2D(chi_phi * np.exp(1j * references.phi_rad), grid)


def _branch_cases() -> tuple[tuple[SegmentSpec, PilotState], ...]:
    rayleigh = 0.4
    waist = np.sqrt(rayleigh * _WAVELENGTH_MM / np.pi)
    return (
        (_segment("S07_S08", 1.6), PilotState(0.8, rayleigh, waist)),
        (_segment("S12_S13", 0.75), PilotState(-0.8, rayleigh, waist)),
        (_segment("S13_S14", 0.85), PilotState(-0.05, rayleigh, waist)),
    )


def _dense_scaled_fresnel(
    source: PointField2D,
    target: UniformGrid2D,
    *,
    wavelength_vacuum_mm: float,
    refractive_index: float,
    distance_mm: float,
) -> np.ndarray:
    wavelength_medium = wavelength_vacuum_mm / refractive_index
    k = 2.0 * np.pi / wavelength_medium
    xs, ys = np.meshgrid(source.grid.x_mm, source.grid.y_mm)
    xt, yt = np.meshgrid(target.x_mm, target.y_mm)
    chirped = source.values * np.exp(1j * k * (xs * xs + ys * ys) / (2.0 * distance_mm))
    x_kernel = np.exp(
        -2j
        * np.pi
        * np.outer(source.grid.x_mm, target.x_mm)
        / (wavelength_medium * distance_mm)
    )
    y_kernel = np.exp(
        -2j
        * np.pi
        * np.outer(target.y_mm, source.grid.y_mm)
        / (wavelength_medium * distance_mm)
    )
    envelope = (
        np.exp(1j * k * (xt * xt + yt * yt) / (2.0 * distance_mm))
        * (y_kernel @ chirped @ x_kernel)
        * source.grid.pixel_area_mm2
        / (1j * wavelength_medium * distance_mm)
    )
    return envelope * np.exp(
        1j * np.remainder(k * distance_mm, 2.0 * np.pi)
    )


def _dense_zero_padded_resample(
    source: PointField2D, target: UniformGrid2D
) -> np.ndarray:
    pad_x = int(np.ceil(0.05 * source.grid.nx))
    pad_y = int(np.ceil(0.05 * source.grid.ny))
    padded_grid = UniformGrid2D.centered(
        nx=source.grid.nx + 2 * pad_x,
        ny=source.grid.ny + 2 * pad_y,
        dx_mm=source.grid.dx_mm,
        dy_mm=source.grid.dy_mm,
    )
    padded = np.zeros((padded_grid.ny, padded_grid.nx), dtype=np.complex128)
    padded[pad_y : pad_y + source.grid.ny, pad_x : pad_x + source.grid.nx] = (
        source.values
    )
    fx = np.fft.fftshift(np.fft.fftfreq(padded_grid.nx, padded_grid.dx_mm))
    fy = np.fft.fftshift(np.fft.fftfreq(padded_grid.ny, padded_grid.dy_mm))
    x_forward = np.exp(-2j * np.pi * np.outer(padded_grid.x_mm, fx))
    y_forward = np.exp(-2j * np.pi * np.outer(fy, padded_grid.y_mm))
    spectrum = (y_forward @ padded @ x_forward) * padded_grid.pixel_area_mm2
    x_inverse = np.exp(2j * np.pi * np.outer(fx, target.x_mm))
    y_inverse = np.exp(2j * np.pi * np.outer(target.y_mm, fy))
    return (y_inverse @ spectrum @ x_inverse) * (fx[1] - fx[0]) * (
        fy[1] - fy[0]
    )


def test_scaled_fresnel_matches_dense_integral_and_analytic_gaussian() -> None:
    source_grid = UniformGrid2D.centered(
        nx=6, ny=5, dx_mm=0.13, dy_mm=0.17
    )
    x, y = np.meshgrid(source_grid.x_mm, source_grid.y_mm)
    source = PointField2D(
        (1.0 + 0.2 * x - 0.1j * y) * np.exp(-0.7 * (x * x + y * y)),
        source_grid,
    )
    target = UniformGrid2D.centered(nx=4, ny=3, dx_mm=0.11, dy_mm=0.16)
    result = propagate_scaled_fresnel(
        source,
        target,
        wavelength_vacuum_mm=0.4,
        refractive_index=1.25,
        distance_mm=1.2,
        batch_size=2,
    )
    expected = _dense_scaled_fresnel(
        source,
        target,
        wavelength_vacuum_mm=0.4,
        refractive_index=1.25,
        distance_mm=1.2,
    )
    np.testing.assert_allclose(result.values, expected, rtol=2e-12, atol=2e-12)

    gaussian_grid = UniformGrid2D.centered(
        nx=128, ny=128, dx_mm=0.05, dy_mm=0.05
    )
    gx, gy = np.meshgrid(gaussian_grid.x_mm, gaussian_grid.y_mm)
    waist_mm = 0.5
    distance_mm = 6.4
    gaussian = PointField2D(
        np.exp(-(gx * gx + gy * gy) / waist_mm**2), gaussian_grid
    )
    propagated = propagate_scaled_fresnel(
        gaussian,
        gaussian_grid,
        wavelength_vacuum_mm=_WAVELENGTH_MM,
        refractive_index=1.0,
        distance_mm=distance_mm,
    )
    rayleigh_mm = np.pi * waist_mm**2 / _WAVELENGTH_MM
    normalized_distance = distance_mm / rayleigh_mm
    analytic = np.exp(
        -(gx * gx + gy * gy)
        / (waist_mm**2 * (1.0 + 1j * normalized_distance))
    ) / (1.0 + 1j * normalized_distance)
    analytic *= np.exp(
        1j
        * np.remainder(
            2.0 * np.pi * distance_mm / _WAVELENGTH_MM, 2.0 * np.pi
        )
    )
    relative_l2 = np.linalg.norm(propagated.values - analytic) / np.linalg.norm(
        analytic
    )
    assert relative_l2 < 2e-10


def test_cell_energy_dft_power_sampling_and_signed_constant() -> None:
    grid = UniformGrid2D.centered(nx=6, ny=4, dx_mm=0.2, dy_mm=0.3)
    x, y = np.meshgrid(grid.x_mm, grid.y_mm)
    point_values = np.exp(-0.4 * (x * x + y * y)) * (
        1.0 + 0.07 * x + 0.11j * y
    )
    cell_values = np.sqrt(grid.pixel_area_mm2) * point_values
    wavelength = 0.5
    k = 2.0 * np.pi / wavelength

    for distance in (1.2, -1.2):
        q_in = k * (x * x + y * y) / (2.0 * distance)
        transformed = scaled_dft_cell_samples(
            cell_values * np.exp(1j * q_in),
            grid,
            wavelength_vacuum_mm=wavelength,
            refractive_index=1.0,
            signed_distance_mm=distance,
        )
        assert transformed.grid.dx_mm == pytest.approx(
            wavelength * abs(distance) / (grid.nx * grid.dx_mm)
        )
        assert transformed.grid.dy_mm == pytest.approx(
            wavelength * abs(distance) / (grid.ny * grid.dy_mm)
        )
        assert np.sum(np.abs(transformed.cell_samples) ** 2) == pytest.approx(
            np.sum(np.abs(cell_values) ** 2), rel=2e-15
        )
        assert scaled_dft_analytic_factor(distance) == (
            -1j if distance > 0.0 else 1j
        )

        xo, yo = np.meshgrid(transformed.grid.x_mm, transformed.grid.y_mm)
        via_dft = (
            scaled_dft_analytic_factor(distance)
            * transformed.cell_samples
            / np.sqrt(transformed.grid.pixel_area_mm2)
            * np.exp(1j * k * (xo * xo + yo * yo) / (2.0 * distance))
        )
        carrier_removed = _dense_scaled_fresnel(
            PointField2D(point_values, grid),
            transformed.grid,
            wavelength_vacuum_mm=wavelength,
            refractive_index=1.0,
            distance_mm=distance,
        ) / np.exp(1j * np.remainder(k * distance, 2.0 * np.pi))
        np.testing.assert_allclose(via_dft, carrier_removed, rtol=3e-14, atol=3e-14)

    mode_grid = UniformGrid2D.centered(nx=12, ny=10, dx_mm=0.2, dy_mm=0.25)
    mx, my = np.meshgrid(mode_grid.x_mm, mode_grid.y_mm)
    fx = 2.0 / (mode_grid.nx * mode_grid.dx_mm)
    fy = -1.0 / (mode_grid.ny * mode_grid.dy_mm)
    mode = PointField2D(np.exp(2j * np.pi * (fx * mx + fy * my)), mode_grid)
    distance = 0.7
    ptp = propagate_ptp_fresnel(
        mode,
        wavelength_vacuum_mm=wavelength,
        refractive_index=1.0,
        distance_mm=distance,
    )
    expected_mode = mode.values * np.exp(
        -1j * np.pi * wavelength * distance * (fx * fx + fy * fy)
    ) * np.exp(1j * np.remainder(k * distance, 2.0 * np.pi))
    np.testing.assert_allclose(ptp.values, expected_mode, rtol=2e-14, atol=2e-14)


def test_all_branches_stock_proper_match_square_fresnel_without_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = UniformGrid2D.centered(nx=64, ny=64, dx_mm=0.02, dy_mm=0.02)
    shift_calls = 0
    real_shift = proper.prop_shift_center

    def counted_shift(values: np.ndarray) -> np.ndarray:
        nonlocal shift_calls
        shift_calls += 1
        return real_shift(values)

    def forbidden_entrance(*_args, **_kwargs):
        raise AssertionError("prop_define_entrance must not be called")

    monkeypatch.setattr(proper, "prop_shift_center", counted_shift)
    monkeypatch.setattr(proper, "prop_define_entrance", forbidden_entrance)
    monkeypatch.setattr(proper, "phase_offset", True)
    monkeypatch.setattr(proper, "print_it", True)
    monkeypatch.setattr(proper, "verbose", True)

    for segment, pilot in _branch_cases():
        physical = _gaussian_start(grid=grid, pilot=pilot)
        independent = candidate_f_q(
            segment=segment,
            physical_start=physical,
            start_pilot=pilot,
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=_REFRACTIVE_INDEX,
        )
        stock = run_stock_proper_fq(
            segment=segment,
            physical_start=physical,
            start_pilot=pilot,
            target_grid=independent.output.grid,
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=_REFRACTIVE_INDEX,
            square_axis="x",
        )
        np.testing.assert_allclose(
            stock.output.values,
            independent.output.values,
            rtol=1e-10,
            atol=1e-12,
        )
        assert stock.diagnostics["implementation_complex_relative_l2"] <= 1e-10
        assert stock.diagnostics["implementation_power_relative_error"] <= 1e-10
        assert stock.diagnostics["implementation_max_phase_waves"] <= 1e-9
        assert stock.diagnostics["center_shift_count"] == 2.0
        assert len(stock.diagnostics["input_center_shift_sha256"]) == 64
        assert len(stock.diagnostics["output_center_shift_sha256"]) == 64
        assert stock.diagnostics["output_mapping"] == "identity"

    assert shift_calls == 2 * len(_branch_cases())
    assert proper.phase_offset is True
    assert proper.print_it is True
    assert proper.verbose is True


def test_square_variants_use_finite_support_and_resample_q_slow_field() -> None:
    source_grid = UniformGrid2D.centered(
        nx=10, ny=10, dx_mm=0.2, dy_mm=0.1
    )
    slow_values = np.zeros((10, 10), dtype=np.complex128)
    slow_values[3:7, 3:7] = 1.0 + 0.25j
    slow = PointField2D(slow_values, source_grid)

    expanded = map_slow_field_to_square(slow, square_axis="x")
    assert expanded.field.grid.dx_mm == pytest.approx(0.2)
    assert expanded.field.grid.dy_mm == pytest.approx(0.2)
    outside_source = (
        (expanded.field.grid.y_mm < source_grid.y_mm[0])
        | (expanded.field.grid.y_mm >= source_grid.y_mm[0] + source_grid.ny * source_grid.dy_mm)
    )
    np.testing.assert_array_equal(expanded.field.values[outside_source, :], 0.0j)
    assert expanded.eta_crop == 0.0
    assert expanded.interpolation_id == "zero_padded_fourier_5pct"

    cropped = map_slow_field_to_square(slow, square_axis="y")
    assert cropped.field.grid.dx_mm == pytest.approx(0.1)
    assert cropped.field.grid.dy_mm == pytest.approx(0.1)
    assert cropped.eta_crop == 0.0
    edge_values = slow_values.copy()
    edge_values[:, 0] = 1.0
    with pytest.raises(ValueError, match="crop"):
        map_slow_field_to_square(
            PointField2D(edge_values, source_grid), square_axis="y"
        )

    output_grid = UniformGrid2D.centered(
        nx=20, ny=20, dx_mm=0.1, dy_mm=0.1
    )
    ox, oy = np.meshgrid(output_grid.x_mm, output_grid.y_mm)
    q_slow_values = (1.0 + 0.2 * ox + 0.3j * oy).astype(np.complex128)
    q_slow_values[[0, -1], :] = 0.0
    q_slow_values[:, [0, -1]] = 0.0
    q_slow = PointField2D(q_slow_values, output_grid)
    target = UniformGrid2D.centered(nx=10, ny=10, dx_mm=0.099, dy_mm=0.098)
    predicted = PilotState(zeta_mm=1.0, rayleigh_mm=0.4, waist_mm=0.08)
    lifted = lift_q_relative_slow_field(
        q_slow,
        target_grid=target,
        predicted_target_pilot=predicted,
        wavelength_vacuum_mm=_WAVELENGTH_MM,
        refractive_index=1.0,
        model_distance_mm=0.7,
    )
    q_target = reference_phases(
        target,
        predicted,
        wavelength_vacuum_mm=_WAVELENGTH_MM,
        refractive_index=1.0,
    ).q_rad
    k = 2.0 * np.pi / _WAVELENGTH_MM
    expected_slow = _dense_zero_padded_resample(q_slow, target)
    expected = expected_slow * np.exp(1j * q_target) * np.exp(
        1j * np.remainder(k * 0.7, 2.0 * np.pi)
    )
    np.testing.assert_allclose(lifted.values, expected, rtol=2e-14, atol=2e-14)


def test_fixed_candidates_bind_input_q_phi_and_structural_identities() -> None:
    grid = UniformGrid2D.centered(nx=128, ny=128, dx_mm=0.02, dy_mm=0.02)
    target_hash = hashlib.sha256(b"current-run-target-zbf").hexdigest()

    for segment, pilot in _branch_cases():
        physical = _gaussian_start(grid=grid, pilot=pilot)
        natural_fq = candidate_f_q(
            segment=segment,
            physical_start=physical,
            start_pilot=pilot,
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=1.0,
        )
        target_grid = UniformGrid2D.centered(
            nx=natural_fq.output.grid.nx,
            ny=natural_fq.output.grid.ny,
            dx_mm=natural_fq.output.grid.dx_mm * (1.0 - 1.0e-4),
            dy_mm=natural_fq.output.grid.dy_mm * (1.0 - 2.0e-4),
        )
        fq = candidate_f_q(
            segment=segment,
            physical_start=physical,
            start_pilot=pilot,
            target_grid=target_grid,
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=1.0,
        )
        predicted = PilotState(
            zeta_mm=pilot.zeta_mm + segment.model_distance_mm,
            rayleigh_mm=pilot.rayleigh_mm,
            waist_mm=pilot.waist_mm,
        )
        observed_target = PilotState(
            zeta_mm=predicted.zeta_mm + (0.02 if not predicted.inside else 0.0),
            rayleigh_mm=predicted.rayleigh_mm,
            waist_mm=predicted.waist_mm,
        )
        target_phi = reference_phases(
            target_grid,
            observed_target,
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=1.0,
        ).phi_rad
        r_phi_q = candidate_r_phi_given_q(
            segment=segment,
            physical_start=physical,
            start_pilot=pilot,
            target_grid=target_grid,
            target_phi_rad=target_phi,
            target_zbf_sha256=target_hash,
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=1.0,
        )
        r_phi_phi = candidate_r_phi_given_phi(
            segment=segment,
            physical_start=physical,
            start_pilot=pilot,
            target_grid=target_grid,
            target_phi_rad=target_phi,
            target_zbf_sha256=target_hash,
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=1.0,
        )
        assert {
            fq.input_sha256,
            r_phi_q.input_sha256,
            r_phi_phi.input_sha256,
        } == {fq.input_sha256}
        assert {
            fq.input_grid_sha256,
            r_phi_q.input_grid_sha256,
            r_phi_phi.input_grid_sha256,
        } == {fq.input_grid_sha256}
        assert fq.predicted_target_zeta_mm == pytest.approx(predicted.zeta_mm)
        assert r_phi_q.predicted_target_zeta_mm == pytest.approx(predicted.zeta_mm)
        with pytest.raises(TypeError):
            fq.diagnostics["tampered"] = True
        assert fq.diagnostics["output_resampled"] is True
        assert r_phi_q.diagnostics["output_resampled"] is True
        assert fq.diagnostics["output_mapping"] == "zero_padded_fourier_5pct"
        assert fq.diagnostics["eta_edge"] <= 1e-10
        assert fq.diagnostics["natural_output_grid_sha256"] == _grid_hash_for_test(
            natural_fq.output.grid
        )
        assert r_phi_q.diagnostics["target_zbf_sha256"] == target_hash
        assert r_phi_phi.diagnostics["target_zbf_sha256"] == target_hash

        if segment.branch == "IO":
            np.testing.assert_allclose(
                np.abs(r_phi_q.output.values),
                np.abs(fq.output.values),
                rtol=2e-14,
                atol=2e-14,
            )
            q_target = reference_phases(
                fq.output.grid,
                predicted,
                wavelength_vacuum_mm=_WAVELENGTH_MM,
                refractive_index=1.0,
            ).q_rad
            np.testing.assert_allclose(
                r_phi_q.output.values,
                fq.output.values * np.exp(1j * (target_phi - q_target)),
                rtol=3e-14,
                atol=3e-14,
            )

    with pytest.raises(ValueError, match="target reference"):
        candidate_r_phi_given_q(
            segment=_branch_cases()[0][0],
            physical_start=_gaussian_start(grid=grid, pilot=_branch_cases()[0][1]),
            start_pilot=_branch_cases()[0][1],
            target_grid=grid,
            target_phi_rad=np.zeros((3, 3)),
            target_zbf_sha256=target_hash,
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=1.0,
        )
