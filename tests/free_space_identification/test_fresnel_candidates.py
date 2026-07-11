from __future__ import annotations

import hashlib
import json
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
    MappedZbfField,
    PilotState,
    ReferencePhases,
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


def _mapped_gaussian(
    *, grid: UniformGrid2D, pilot: PilotState, evidence_label: str
) -> MappedZbfField:
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
    return MappedZbfField(
        physical=PointField2D(
            chi_phi * np.exp(1j * references.phi_rad), grid
        ),
        reference_relative=chi_phi.astype(np.complex128),
        references=references,
        pilot=pilot,
        source_sha256=hashlib.sha256(
            ("source:" + evidence_label).encode("utf-8")
        ).hexdigest(),
        convention_evidence_sha256=hashlib.sha256(
            b"shared-current-run-convention"
        ).hexdigest(),
        sample_value_convention="point_value",
    )


def _paired_target(
    *, segment: SegmentSpec, start: MappedZbfField, target: MappedZbfField
) -> candidate_module.PairedTargetEvidence:
    return candidate_module.PairedTargetEvidence.bind(
        segment=segment, start=start, target=target
    )


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
    np.testing.assert_allclose(result.field.values, expected, rtol=2e-12, atol=2e-12)
    expected_nominal = 2.0 * np.pi * 1.25 * 1.2 / 0.4
    assert result.axial_carrier_nominal_rad == pytest.approx(expected_nominal)
    assert result.axial_carrier_reduced_rad == pytest.approx(
        np.remainder(expected_nominal, 2.0 * np.pi)
    )
    with pytest.raises((AttributeError, TypeError)):
        result.axial_carrier_reduced_rad = 0.0

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
    relative_l2 = np.linalg.norm(propagated.field.values - analytic) / np.linalg.norm(
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
    np.testing.assert_allclose(ptp.field.values, expected_mode, rtol=2e-14, atol=2e-14)
    expected_nominal = k * distance
    assert ptp.axial_carrier_nominal_rad == pytest.approx(expected_nominal)
    assert ptp.axial_carrier_reduced_rad == pytest.approx(
        np.remainder(expected_nominal, 2.0 * np.pi)
    )


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
        start = _mapped_gaussian(
            grid=grid, pilot=pilot, evidence_label=f"{segment.key}:start"
        )
        natural_grid = candidate_module._natural_output_grid(
            segment=segment,
            start=start,
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=_REFRACTIVE_INDEX,
        )
        predicted = PilotState(
            zeta_mm=pilot.zeta_mm + segment.model_distance_mm,
            rayleigh_mm=pilot.rayleigh_mm,
            waist_mm=pilot.waist_mm,
        )
        target_field = _mapped_gaussian(
            grid=natural_grid,
            pilot=predicted,
            evidence_label=f"{segment.key}:target",
        )
        target = _paired_target(
            segment=segment, start=start, target=target_field
        )
        independent = candidate_f_q(
            segment=segment,
            start=start,
            target=target,
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=_REFRACTIVE_INDEX,
        )
        stock = run_stock_proper_fq(
            segment=segment,
            start=start,
            target=target,
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
        assert stock.diagnostics["a_mm"] == pytest.approx(-pilot.zeta_mm)
        assert stock.diagnostics["b_mm"] == pytest.approx(predicted.zeta_mm)
        nominal = (
            2.0
            * np.pi
            * _REFRACTIVE_INDEX
            * segment.model_distance_mm
            / _WAVELENGTH_MM
        )
        assert stock.diagnostics["axial_carrier_nominal_rad"] == pytest.approx(
            nominal
        )
        assert stock.diagnostics["axial_carrier_reduced_rad"] == pytest.approx(
            np.remainder(nominal, 2.0 * np.pi)
        )
        assert len(stock.diagnostics["path_sha256"]) == 64
        assert stock.diagnostics["path_id"].startswith("F_Q:")
        for name in (
            "source_half_open_region",
            "target_half_open_region",
            "intersection_half_open_region",
            "cropped_half_open_regions",
            "added_half_open_regions",
        ):
            assert isinstance(stock.diagnostics[name], str)
        for name in (
            "source_sample_count",
            "target_sample_count",
            "intersection_source_sample_count",
            "intersection_target_sample_count",
            "cropped_sample_count",
            "added_sample_count",
        ):
            assert stock.diagnostics[name] >= 0.0
        stock_input_map = map_slow_field_to_square(
            PointField2D(start.reference_relative, start.physical.grid),
            square_axis="x",
        )
        assert json.loads(stock.diagnostics["source_half_open_region"]) == list(
            stock_input_map.source_half_open_region
        )
        assert json.loads(stock.diagnostics["target_half_open_region"]) == list(
            stock_input_map.target_half_open_region
        )
        assert json.loads(
            stock.diagnostics["intersection_half_open_region"]
        ) == list(stock_input_map.intersection_half_open_region)
        assert json.loads(stock.diagnostics["cropped_half_open_regions"]) == [
            list(region) for region in stock_input_map.cropped_half_open_regions
        ]
        assert json.loads(stock.diagnostics["added_half_open_regions"]) == [
            list(region) for region in stock_input_map.added_half_open_regions
        ]
        assert stock.diagnostics["cropped_sample_count"] == float(
            stock_input_map.cropped_sample_count
        )
        assert stock.diagnostics["added_sample_count"] == float(
            stock_input_map.added_sample_count
        )

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
    assert expanded.cropped_half_open_regions == ()
    assert len(expanded.added_half_open_regions) == 2
    assert expanded.intersection_source_sample_count == 100
    assert expanded.intersection_target_sample_count == 50
    assert expanded.cropped_sample_count == 0
    assert expanded.added_sample_count == 50

    cropped = map_slow_field_to_square(slow, square_axis="y")
    assert cropped.field.grid.dx_mm == pytest.approx(0.1)
    assert cropped.field.grid.dy_mm == pytest.approx(0.1)
    assert cropped.eta_crop == 0.0
    assert cropped.source_half_open_region == pytest.approx((-1.0, 1.0, -0.5, 0.5))
    assert cropped.target_half_open_region == pytest.approx((-0.5, 0.5, -0.5, 0.5))
    assert cropped.intersection_half_open_region == pytest.approx(
        (-0.5, 0.5, -0.5, 0.5)
    )
    assert len(cropped.cropped_half_open_regions) == 2
    assert cropped.added_half_open_regions == ()
    assert cropped.source_sample_count == 100
    assert cropped.target_sample_count == 100
    assert cropped.intersection_source_sample_count == 50
    assert cropped.intersection_target_sample_count == 100
    assert cropped.cropped_sample_count == 50
    assert cropped.added_sample_count == 0

    tiny_crop_values = slow_values.copy()
    source_x, source_y = np.meshgrid(source_grid.x_mm, source_grid.y_mm)
    target_left, target_right, target_bottom, target_top = (
        cropped.target_half_open_region
    )
    crop_mask = (
        (source_x < target_left)
        | (source_x >= target_right)
        | (source_y < target_bottom)
        | (source_y >= target_top)
    )
    tiny_crop_values[crop_mask] = 1.0e-7 - 2.0e-7j
    tiny_crop = map_slow_field_to_square(
        PointField2D(tiny_crop_values, source_grid), square_axis="y"
    )
    assert 0.0 < tiny_crop.eta_crop < 1.0e-10
    explicitly_precropped = tiny_crop_values.copy()
    explicitly_precropped[crop_mask] = 0.0j
    precrop_oracle = _dense_zero_padded_resample(
        PointField2D(explicitly_precropped, source_grid), tiny_crop.field.grid
    )
    untrimmed_oracle = _dense_zero_padded_resample(
        PointField2D(tiny_crop_values, source_grid), tiny_crop.field.grid
    )
    np.testing.assert_allclose(
        tiny_crop.field.values, precrop_oracle, rtol=3e-14, atol=3e-14
    )
    assert np.linalg.norm(tiny_crop.field.values - untrimmed_oracle) > 1.0e-8
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

    assert len(candidate_module.PATH_SPECS) == 9
    assert len({spec.path_id for spec in candidate_module.PATH_SPECS}) == 9
    assert len({spec.path_sha256 for spec in candidate_module.PATH_SPECS}) == 9
    expected_stages = {
        "OO": ("STW:a", "WTS:b"),
        "OI": ("STW:a", "PTP:b"),
        "IO": ("PTP:a", "WTS:b"),
    }
    for spec in candidate_module.PATH_SPECS:
        assert spec.stage_order == expected_stages[spec.branch]
        assert spec.input_reference == ("q" if spec.operator_id == "F_Q" else "phi")
        assert spec.internal_phase == (
            "phi" if spec.operator_id == "R_Phi_given_Phi" else "q"
        )
        assert spec.output_reference == (
            "q" if spec.operator_id == "F_Q" else "phi"
        )
        assert spec.branch_constant == (
            1.0 + 0.0j if spec.branch == "OO" else -1.0j
        )
    with pytest.raises((AttributeError, TypeError)):
        candidate_module.PATH_SPECS[0].path_id = "tampered"

    collision_grid_2x4 = UniformGrid2D(
        x_mm=np.array([0.0, 1.0, 2.0, 3.0]),
        y_mm=np.array([4.0, 5.0]),
    )
    collision_grid_4x2 = UniformGrid2D(
        x_mm=np.array([0.0, 1.0]),
        y_mm=np.array([2.0, 3.0, 4.0, 5.0]),
    )
    concatenated_2x4 = np.concatenate(
        (collision_grid_2x4.x_mm, collision_grid_2x4.y_mm)
    ).astype("<f8").tobytes()
    concatenated_4x2 = np.concatenate(
        (collision_grid_4x2.x_mm, collision_grid_4x2.y_mm)
    ).astype("<f8").tobytes()
    assert concatenated_2x4 == concatenated_4x2
    assert candidate_module._grid_sha256(
        collision_grid_2x4
    ) != candidate_module._grid_sha256(collision_grid_4x2)
    flat_values = np.arange(8, dtype=np.float64).astype(np.complex128)
    field_2x4 = PointField2D(flat_values.reshape(2, 4), collision_grid_2x4)
    field_4x2 = PointField2D(flat_values.reshape(4, 2), collision_grid_4x2)
    assert field_2x4.values.tobytes(order="C") == field_4x2.values.tobytes(order="C")
    assert candidate_module._field_sha256(
        field_2x4
    ) != candidate_module._field_sha256(field_4x2)

    for segment, pilot in _branch_cases():
        start = _mapped_gaussian(
            grid=grid, pilot=pilot, evidence_label=f"{segment.key}:start"
        )
        natural_output_grid = candidate_module._natural_output_grid(
            segment=segment,
            start=start,
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=1.0,
        )
        target_grid = UniformGrid2D.centered(
            nx=natural_output_grid.nx,
            ny=natural_output_grid.ny,
            dx_mm=natural_output_grid.dx_mm * (1.0 - 1.0e-4),
            dy_mm=natural_output_grid.dy_mm * (1.0 - 2.0e-4),
        )
        predicted = PilotState(
            zeta_mm=pilot.zeta_mm + segment.model_distance_mm,
            rayleigh_mm=pilot.rayleigh_mm,
            waist_mm=pilot.waist_mm,
        )
        observed_target = PilotState(
            zeta_mm=predicted.zeta_mm,
            rayleigh_mm=predicted.rayleigh_mm,
            waist_mm=predicted.waist_mm,
        )
        target_field = _mapped_gaussian(
            grid=target_grid,
            pilot=observed_target,
            evidence_label=f"{segment.key}:target",
        )
        target = _paired_target(
            segment=segment, start=start, target=target_field
        )
        fq = candidate_f_q(
            segment=segment,
            start=start,
            target=target,
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=1.0,
        )
        r_phi_q = candidate_r_phi_given_q(
            segment=segment,
            start=start,
            target=target,
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=1.0,
        )
        r_phi_phi = candidate_r_phi_given_phi(
            segment=segment,
            start=start,
            target=target,
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
        assert fq.diagnostics["natural_output_grid_sha256"] == candidate_module._grid_sha256(
            natural_output_grid
        )
        for result in (fq, r_phi_q, r_phi_phi):
            spec = next(
                spec
                for spec in candidate_module.PATH_SPECS
                if spec.operator_id == result.operator_id
                and spec.branch == segment.branch
            )
            assert result.diagnostics["path_id"] == spec.path_id
            assert result.diagnostics["path_sha256"] == spec.path_sha256
            assert result.diagnostics["input_reference"] == spec.input_reference
            assert result.diagnostics["internal_phase"] == spec.internal_phase
            assert result.diagnostics["stage_order"] == ",".join(spec.stage_order)
            assert result.diagnostics["output_reference"] == spec.output_reference
            assert result.diagnostics["a_mm"] == pytest.approx(-pilot.zeta_mm)
            assert result.diagnostics["b_mm"] == pytest.approx(predicted.zeta_mm)
            assert result.diagnostics["target_zbf_sha256"] == (
                target.target.source_sha256
            )
            assert result.diagnostics["target_evidence_sha256"] == (
                candidate_module._mapped_input_sha256(target.target)
            )
            assert result.diagnostics["target_pair_sha256"] == target.pair_sha256
        assert fq.diagnostics["uses_predicted_target_q"] is True
        assert r_phi_q.diagnostics["uses_predicted_target_q"] is False

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
                fq.output.values
                * np.exp(1j * (target.target.references.phi_rad - q_target)),
                rtol=3e-14,
                atol=3e-14,
            )

        relabelled_source = replace(
            start,
            source_sha256=hashlib.sha256(
                f"{segment.key}:other-source".encode("utf-8")
            ).hexdigest(),
        )
        relabelled_convention = replace(
            start,
            convention_evidence_sha256=hashlib.sha256(
                f"{segment.key}:other-convention".encode("utf-8")
            ).hexdigest(),
        )
        alternate_sample_kind = replace(
            start, sample_value_convention="cell_energy"
        )
        assert len(
            {
                candidate_module._mapped_input_sha256(start),
                candidate_module._mapped_input_sha256(relabelled_source),
                candidate_module._mapped_input_sha256(relabelled_convention),
                candidate_module._mapped_input_sha256(alternate_sample_kind),
            }
        ) == 4

    segment, pilot = _branch_cases()[0]
    start = _mapped_gaussian(grid=grid, pilot=pilot, evidence_label="bad:start")
    natural_grid = candidate_module._natural_output_grid(
        segment=segment,
        start=start,
        wavelength_vacuum_mm=_WAVELENGTH_MM,
        refractive_index=1.0,
    )
    target_pilot = PilotState(
        zeta_mm=pilot.zeta_mm + segment.model_distance_mm,
        rayleigh_mm=pilot.rayleigh_mm,
        waist_mm=pilot.waist_mm,
    )
    target_field = _mapped_gaussian(
        grid=natural_grid, pilot=target_pilot, evidence_label="bad:target"
    )
    target = _paired_target(
        segment=segment, start=start, target=target_field
    )
    wrong_physical = PointField2D(
        start.physical.values * np.exp(0.01j), start.physical.grid
    )
    with pytest.raises(ValueError, match="physical"):
        candidate_f_q(
            segment=segment,
            start=replace(start, physical=wrong_physical),
            target=target,
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=1.0,
        )
    with pytest.raises(ValueError, match="pilot|reference"):
        candidate_f_q(
            segment=segment,
            start=replace(
                start,
                pilot=PilotState(
                    pilot.zeta_mm + 0.01, pilot.rayleigh_mm, pilot.waist_mm
                ),
            ),
            target=target,
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=1.0,
        )
    wrong_refs = ReferencePhases(
        q_rad=target_field.references.q_rad + 0.01,
        phi_rad=target_field.references.phi_rad,
    )
    with pytest.raises(ValueError, match="reference"):
        candidate_r_phi_given_q(
            segment=segment,
            start=start,
            target=replace(
                target, target=replace(target_field, references=wrong_refs)
            ),
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=1.0,
        )
    wrong_branch_target = _mapped_gaussian(
        grid=target_field.physical.grid,
        pilot=PilotState(
            0.0, target_field.pilot.rayleigh_mm, target_field.pilot.waist_mm
        ),
        evidence_label="wrong-branch-target",
    )
    with pytest.raises(ValueError, match="target pilot classification"):
        candidate_r_phi_given_q(
            segment=segment,
            start=start,
            target=replace(target, target=wrong_branch_target),
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=1.0,
        )
    wrong_target_physical = PointField2D(
        target_field.physical.values * np.exp(-0.02j), target_field.physical.grid
    )
    with pytest.raises(ValueError, match="physical"):
        candidate_r_phi_given_q(
            segment=segment,
            start=start,
            target=replace(
                target,
                target=replace(target_field, physical=wrong_target_physical),
            ),
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=1.0,
        )
    with pytest.raises(ValueError, match="pilot|reference"):
        candidate_r_phi_given_phi(
            segment=segment,
            start=start,
            target=replace(
                target,
                target=replace(
                    target_field,
                    pilot=PilotState(
                        target_field.pilot.zeta_mm + 0.01,
                        target_field.pilot.rayleigh_mm,
                        target_field.pilot.waist_mm,
                    ),
                ),
            ),
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=1.0,
        )

    same_endpoint = _paired_target(segment=segment, start=start, target=start)
    with pytest.raises(ValueError, match="distinct start and target"):
        candidate_r_phi_given_q(
            segment=segment,
            start=start,
            target=same_endpoint,
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=1.0,
        )
    with pytest.raises(ValueError, match="segment binding"):
        candidate_f_q(
            segment=segment,
            start=start,
            target=replace(target, segment_key="S13_S14"),
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=1.0,
        )

    inside_segment, inside_pilot = _branch_cases()[2]
    inside_start = _mapped_gaussian(
        grid=grid, pilot=inside_pilot, evidence_label="inside:start"
    )
    inside_natural_grid = candidate_module._natural_output_grid(
        segment=inside_segment,
        start=inside_start,
        wavelength_vacuum_mm=_WAVELENGTH_MM,
        refractive_index=1.0,
    )
    inside_target_pilot = PilotState(
        inside_pilot.zeta_mm + inside_segment.model_distance_mm,
        inside_pilot.rayleigh_mm,
        inside_pilot.waist_mm,
    )
    inside_target_field = _mapped_gaussian(
        grid=inside_natural_grid,
        pilot=inside_target_pilot,
        evidence_label="inside:target",
    )
    inside_target = _paired_target(
        segment=inside_segment,
        start=inside_start,
        target=inside_target_field,
    )
    bad_waist_start = replace(
        inside_start,
        pilot=PilotState(
            inside_pilot.zeta_mm,
            inside_pilot.rayleigh_mm,
            1.1 * inside_pilot.waist_mm,
        ),
    )
    with pytest.raises(ValueError, match="Rayleigh"):
        candidate_f_q(
            segment=inside_segment,
            start=bad_waist_start,
            target=inside_target,
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=1.0,
        )
    bad_inside_zeta = replace(
        inside_start,
        pilot=PilotState(
            -0.1, inside_pilot.rayleigh_mm, inside_pilot.waist_mm
        ),
    )
    np.testing.assert_array_equal(
        reference_phases(
            grid,
            bad_inside_zeta.pilot,
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=1.0,
        ).phi_rad,
        inside_start.references.phi_rad,
    )
    with pytest.raises(ValueError, match="start evidence binding"):
        candidate_f_q(
            segment=inside_segment,
            start=bad_inside_zeta,
            target=inside_target,
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=1.0,
        )
    rebound_bad_inside_target = _paired_target(
        segment=inside_segment,
        start=bad_inside_zeta,
        target=inside_target_field,
    )
    with pytest.raises(ValueError, match="pilot distance"):
        candidate_f_q(
            segment=inside_segment,
            start=bad_inside_zeta,
            target=rebound_bad_inside_target,
            wavelength_vacuum_mm=_WAVELENGTH_MM,
            refractive_index=1.0,
        )
