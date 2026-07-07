from __future__ import annotations

import numpy as np
import pytest

from pop.core import GridSampling, OpticalAxisState, PilotBeamParams
from pop.options import PropagationOptions
from pop.propagation.free_space import propagate_free_space
from pop.wavefront.sampler import sample_rays_from_wavefront


def _axis() -> OpticalAxisState:
    return OpticalAxisState(
        position=np.zeros(3),
        direction=np.array([0.0, 0.0, 1.0]),
        frame=np.eye(3),
        coord_sys=None,
        path_length=0.0,
    )


def test_propagation_options_default_to_native_proper_and_pilot_only() -> None:
    options = PropagationOptions()

    assert options.free_space_mode == "native_proper"
    assert options.element_phase_mode == "pilot_only"


def test_pilot_phase_uses_gaussian_curvature_not_spherical_sag() -> None:
    pilot = PilotBeamParams.from_gaussian_source(
        wavelength_um=10.0,
        w0_mm=0.1,
        z0_mm=-1.0,
    )
    grid_size = 9
    physical_size_mm = 8.0
    phase = pilot.compute_phase_grid(grid_size, physical_size_mm)

    coords = (np.arange(grid_size) - grid_size // 2) * (physical_size_mm / grid_size)
    x_grid, y_grid = np.meshgrid(coords, coords)
    r_sq = x_grid**2 + y_grid**2
    wavelength_mm = pilot.wavelength_um * 1e-3
    k = 2.0 * np.pi * pilot.current_refractive_index / wavelength_mm
    gaussian_expected = k * r_sq / (2.0 * pilot.curvature_radius_mm)

    np.testing.assert_allclose(phase, gaussian_expected, rtol=1e-12, atol=1e-12)

    r = pilot.curvature_radius_mm
    spherical_sag = r_sq / (r * (1.0 + np.sqrt(1.0 - r_sq / (r * r))))
    spherical_phase = k * spherical_sag
    assert np.max(np.abs(phase - spherical_phase)) > 1e-3


def test_pilot_only_sampling_ignores_non_pilot_residual_phase() -> None:
    grid = GridSampling.create(grid_size=33, physical_size_mm=8.0)
    pilot = PilotBeamParams.from_gaussian_source(
        wavelength_um=10.0,
        w0_mm=1.0,
        z0_mm=-5.0,
    )
    x_grid, _ = grid.get_coordinate_arrays()
    amplitude = pilot.compute_amplitude_grid(grid.grid_size, grid.physical_size_mm)
    pilot_phase = pilot.compute_phase_grid(grid.grid_size, grid.physical_size_mm)
    residual_phase = 0.05 * x_grid

    base_local, base_global = sample_rays_from_wavefront(
        amplitude=amplitude,
        phase=pilot_phase,
        grid_sampling=grid,
        entrance_axis=_axis(),
        pilot_beam_params=pilot,
        num_rays=49,
        sampling_sigma=3.0,
        element_phase_mode="pilot_only",
    )
    residual_local, residual_global = sample_rays_from_wavefront(
        amplitude=amplitude,
        phase=pilot_phase + residual_phase,
        grid_sampling=grid,
        entrance_axis=_axis(),
        pilot_beam_params=pilot,
        num_rays=49,
        sampling_sigma=3.0,
        element_phase_mode="pilot_only",
    )

    np.testing.assert_allclose(residual_local.L, base_local.L)
    np.testing.assert_allclose(residual_local.M, base_local.M)
    np.testing.assert_allclose(residual_global.opd, base_global.opd)


def test_full_wavefront_sampling_preserves_legacy_residual_phase_response() -> None:
    grid = GridSampling.create(grid_size=33, physical_size_mm=8.0)
    pilot = PilotBeamParams.from_gaussian_source(
        wavelength_um=10.0,
        w0_mm=1.0,
        z0_mm=-5.0,
    )
    x_grid, _ = grid.get_coordinate_arrays()
    amplitude = pilot.compute_amplitude_grid(grid.grid_size, grid.physical_size_mm)
    pilot_phase = pilot.compute_phase_grid(grid.grid_size, grid.physical_size_mm)
    residual_phase = 0.05 * x_grid

    base_local, base_global = sample_rays_from_wavefront(
        amplitude=amplitude,
        phase=pilot_phase,
        grid_sampling=grid,
        entrance_axis=_axis(),
        pilot_beam_params=pilot,
        num_rays=49,
        sampling_sigma=3.0,
        element_phase_mode="full_wavefront",
    )
    residual_local, residual_global = sample_rays_from_wavefront(
        amplitude=amplitude,
        phase=pilot_phase + residual_phase,
        grid_sampling=grid,
        entrance_axis=_axis(),
        pilot_beam_params=pilot,
        num_rays=49,
        sampling_sigma=3.0,
        element_phase_mode="full_wavefront",
    )

    assert np.max(np.abs(residual_local.L - base_local.L)) > 1e-6
    assert np.max(np.abs(residual_global.opd - base_global.opd)) > 1e-6


def test_native_free_space_delegates_to_proper_without_forced_asm(monkeypatch) -> None:
    import proper

    pilot = PilotBeamParams.from_gaussian_source(
        wavelength_um=10.0,
        w0_mm=1.0,
        z0_mm=0.0,
    )
    wfo = proper.prop_begin(2.0e-3, 10.0e-6, 32, 0.5)
    wfo.w0 = pilot.waist_radius_mm * 1e-3
    wfo.z_Rayleigh = pilot.rayleigh_length_mm * 1e-3
    wfo.z_w0 = 0.0
    original_z_rayleigh = wfo.z_Rayleigh
    calls: list[float] = []

    def fake_prop_propagate(wf, dz):
        calls.append(dz)
        wf.z = wf.z + dz

    def fail_prop_qphase(_wf, _c):
        raise AssertionError("native mode must not call hand-managed prop_qphase")

    monkeypatch.setattr(proper, "prop_propagate", fake_prop_propagate)
    monkeypatch.setattr(proper, "prop_qphase", fail_prop_qphase)

    _grid, algorithm = propagate_free_space(
        wfo,
        distance_mm=2.5,
        n=1.0,
        force_asm=True,
        auto_asm=True,
        pilot_beam=pilot,
        free_space_mode="native_proper",
    )

    assert calls == pytest.approx([0.0025])
    assert algorithm == "PROPER native"
    assert wfo.z_Rayleigh == pytest.approx(original_z_rayleigh)
