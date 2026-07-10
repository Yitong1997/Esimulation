import numpy as np
import pytest
import scipy.fft

import sandbox.free_space_algorithm_identification.fourier as fourier_module
from sandbox.free_space_algorithm_identification.fourier import (
    Spectrum2D,
    cell_energy_to_point,
    evaluate_field_fourier_czt,
    evaluate_spectrum_czt,
    forward_continuous_spectrum,
    point_to_cell_energy,
    resample_bandlimited,
    resample_cubic_complex,
    resample_lanczos_complex,
)
from sandbox.free_space_algorithm_identification.models import PointField2D, UniformGrid2D


def smooth_test_field(*, nx: int, ny: int, dx: float, dy: float) -> PointField2D:
    grid = UniformGrid2D.centered(nx=nx, ny=ny, dx_mm=dx, dy_mm=dy)
    x, y = np.meshgrid(grid.x_mm, grid.y_mm)
    amplitude = np.exp(-0.37 * x**2 - 0.23 * y**2) * (1.0 + 0.07 * x * y)
    phase = 0.31 * x - 0.27 * y + 0.09 * x * y
    return PointField2D(amplitude * np.exp(1j * phase), grid)


def direct_inverse_spectrum_sum(
    spectrum: Spectrum2D, output: UniformGrid2D
) -> PointField2D:
    x_kernel = np.exp(2j * np.pi * np.outer(spectrum.fx_cpm, output.x_mm))
    y_kernel = np.exp(2j * np.pi * np.outer(output.y_mm, spectrum.fy_cpm))
    dfx = spectrum.fx_cpm[1] - spectrum.fx_cpm[0]
    dfy = spectrum.fy_cpm[1] - spectrum.fy_cpm[0]
    values = (y_kernel @ spectrum.values @ x_kernel) * dfx * dfy
    return PointField2D(values, output)


def direct_forward_field_sum(
    field: PointField2D, fx_cpm: np.ndarray, fy_cpm: np.ndarray
) -> np.ndarray:
    x_kernel = np.exp(-2j * np.pi * np.outer(field.grid.x_mm, fx_cpm))
    y_kernel = np.exp(-2j * np.pi * np.outer(fy_cpm, field.grid.y_mm))
    return (y_kernel @ field.values @ x_kernel) * field.grid.pixel_area_mm2


def evaluate_analytic_bandlimited_field(grid: UniformGrid2D) -> np.ndarray:
    x, y = np.meshgrid(grid.x_mm, grid.y_mm)
    return (
        np.exp(2j * np.pi * (0.21 * x - 0.18 * y))
        + 0.27 * np.exp(2j * np.pi * (-0.43 * x + 0.31 * y))
    )


def analytic_bandlimited_complex_field() -> PointField2D:
    grid = UniformGrid2D.centered(nx=65, ny=67, dx_mm=0.04, dy_mm=0.04)
    return PointField2D(evaluate_analytic_bandlimited_field(grid), grid)


def relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def _compensated_complex_sum(values: np.ndarray) -> complex:
    total = 0.0j
    correction = 0.0j
    for value in values:
        adjusted = value - correction
        updated = total + adjusted
        correction = (updated - total) - adjusted
        total = updated
    return total


def test_czt_matches_direct_inverse_sum_on_arbitrary_uniform_grid() -> None:
    field = smooth_test_field(nx=8, ny=10, dx=0.2, dy=0.15)
    spectrum = forward_continuous_spectrum(field)
    output = UniformGrid2D.centered(nx=7, ny=9, dx_mm=0.07, dy_mm=0.08)
    actual = evaluate_spectrum_czt(spectrum, output, batch_size=3)
    expected = direct_inverse_spectrum_sum(spectrum, actual.grid)
    np.testing.assert_allclose(actual.values, expected.values, rtol=1e-11, atol=1e-12)


def test_point_cell_energy_roundtrip_preserves_physical_power() -> None:
    field = smooth_test_field(nx=8, ny=8, dx=0.2, dy=0.3)
    samples = point_to_cell_energy(field)
    restored = cell_energy_to_point(samples, field.grid)
    np.testing.assert_allclose(restored.values, field.values, rtol=0, atol=1e-14)
    assert np.isclose(
        np.sum(abs(samples) ** 2),
        np.sum(abs(field.values) ** 2) * field.grid.pixel_area_mm2,
    )


def test_forward_czt_matches_direct_continuous_fourier_sum() -> None:
    field = smooth_test_field(nx=8, ny=10, dx=0.2, dy=0.15)
    fx = -0.73 + np.arange(7) * 0.11
    fy = -0.51 + np.arange(9) * 0.09
    actual = evaluate_field_fourier_czt(field, fx, fy, batch_size=3)
    expected = direct_forward_field_sum(field, actual.fx_cpm, actual.fy_cpm)
    np.testing.assert_allclose(actual.values, expected, rtol=1e-11, atol=1e-12)


def test_forward_fft_matches_an_independent_natural_grid_sum() -> None:
    field = smooth_test_field(nx=7, ny=6, dx=0.2, dy=0.15)
    actual = forward_continuous_spectrum(field)
    expected = direct_forward_field_sum(field, actual.fx_cpm, actual.fy_cpm)
    np.testing.assert_allclose(actual.values, expected, rtol=1e-11, atol=1e-12)


def test_non_square_impulse_keeps_y_x_and_fy_fx_axis_order() -> None:
    grid = UniformGrid2D.centered(nx=5, ny=7, dx_mm=0.2, dy_mm=0.3)
    values = np.zeros((grid.ny, grid.nx), dtype=np.complex128)
    values[1, 3] = 0.4 - 0.9j
    field = PointField2D(values, grid)
    fx = -0.61 + np.arange(4) * 0.17
    fy = -0.47 + np.arange(6) * 0.13

    actual = evaluate_field_fourier_czt(field, fx, fy, batch_size=2)
    expected = direct_forward_field_sum(field, actual.fx_cpm, actual.fy_cpm)

    assert actual.values.shape == (fy.size, fx.size)
    np.testing.assert_allclose(actual.values, expected, rtol=1e-12, atol=1e-13)


def test_axes_batch_and_spectrum_contracts_reject_bad_inputs_and_freeze_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field = smooth_test_field(nx=6, ny=5, dx=0.2, dy=0.15)
    spectrum = forward_continuous_spectrum(field)
    output = UniformGrid2D.centered(nx=5, ny=4, dx_mm=0.1, dy_mm=0.12)

    for bad_fx in (
        np.array([0.0, 0.2, 0.41]),
        np.array([0.4, 0.2, 0.0]),
        np.array([0.0, np.nan, 0.2]),
        np.array([0.0]),
    ):
        with pytest.raises(ValueError):
            evaluate_field_fourier_czt(field, bad_fx, np.array([-0.2, 0.0, 0.2]))
    for bad_batch_size in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            evaluate_spectrum_czt(spectrum, output, batch_size=bad_batch_size)
    with pytest.raises(ValueError):
        Spectrum2D(
            values=np.ones((3, 3)),
            fx_cpm=np.array([0.0, 0.1, 0.25]),
            fy_cpm=np.array([-0.1, 0.0, 0.1]),
            source_grid=field.grid,
        )
    with pytest.raises(ValueError):
        Spectrum2D(
            values=np.full((3, 3), np.nan),
            fx_cpm=np.array([-0.1, 0.0, 0.1]),
            fy_cpm=np.array([-0.1, 0.0, 0.1]),
            source_grid=field.grid,
        )
    for array in (spectrum.values, spectrum.fx_cpm, spectrum.fy_cpm):
        with pytest.raises(ValueError):
            array.setflags(write=True)

    captured_axes: list[np.ndarray] = []
    real_builder = fourier_module._build_forward_zoom_normalized

    def capture_builder(coordinates, frequencies):
        captured_axes.append(frequencies.copy())
        return real_builder(coordinates, frequencies)

    monkeypatch.setattr(
        fourier_module, "_build_forward_zoom_normalized", capture_builder
    )
    awkward_fx = -15.740993235220827 + np.arange(186) * 2.405745391668958
    actual = evaluate_field_fourier_czt(
        field,
        awkward_fx,
        -0.7 + np.arange(5) * 0.13,
        batch_size=2,
    )
    np.testing.assert_array_equal(actual.fx_cpm, captured_axes[0])
    np.testing.assert_array_equal(actual.fy_cpm, captured_axes[1])

    captured_inverse_axes: list[np.ndarray] = []
    real_inverse_builder = fourier_module._build_inverse_zoom_normalized

    def capture_inverse_builder(frequencies, coordinates):
        captured_inverse_axes.append(frequencies.copy())
        return real_inverse_builder(frequencies, coordinates)

    monkeypatch.setattr(
        fourier_module, "_build_inverse_zoom_normalized", capture_inverse_builder
    )
    evaluate_spectrum_czt(actual, output, batch_size=2)
    np.testing.assert_array_equal(actual.fx_cpm, captured_inverse_axes[0])
    np.testing.assert_array_equal(actual.fy_cpm, captured_inverse_axes[1])


def test_inverse_and_forward_czt_never_transform_more_than_one_batch_slab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_size = 3
    field = smooth_test_field(nx=8, ny=10, dx=0.2, dy=0.15)
    output = UniformGrid2D.centered(nx=7, ny=9, dx_mm=0.07, dy_mm=0.08)
    spectrum = forward_continuous_spectrum(field)
    inverse_calls: list[tuple[int, tuple[int, ...]]] = []
    forward_calls: list[tuple[int, tuple[int, ...]]] = []
    real_inverse = fourier_module._apply_inverse_zoom
    real_forward = fourier_module._apply_forward_zoom

    def inverse_probe(values, zoom, phase, step, *, axis):
        inverse_calls.append((axis, values.shape))
        slab = values.shape[0] if axis == 1 else values.shape[1]
        assert slab <= batch_size
        return real_inverse(values, zoom, phase, step, axis=axis)

    def forward_probe(values, zoom, phase, step, *, axis):
        forward_calls.append((axis, values.shape))
        slab = values.shape[0] if axis == 1 else values.shape[1]
        assert slab <= batch_size
        return real_forward(values, zoom, phase, step, axis=axis)

    monkeypatch.setattr(fourier_module, "_apply_inverse_zoom", inverse_probe)
    monkeypatch.setattr(fourier_module, "_apply_forward_zoom", forward_probe)
    evaluate_spectrum_czt(spectrum, output, batch_size=batch_size)
    evaluate_field_fourier_czt(
        field,
        -0.7 + np.arange(7) * 0.1,
        -0.5 + np.arange(9) * 0.08,
        batch_size=batch_size,
    )

    assert {axis for axis, _ in inverse_calls} == {0, 1}
    assert {axis for axis, _ in forward_calls} == {0, 1}


def test_lanczos_and_cubic_checks_interpolate_complex_field_not_wrapped_phase() -> None:
    field = analytic_bandlimited_complex_field()
    target = UniformGrid2D.centered(nx=15, ny=17, dx_mm=0.08, dy_mm=0.07)
    lanczos8 = resample_lanczos_complex(field, target, lobes=8)
    lanczos12 = resample_lanczos_complex(field, target, lobes=12)
    cubic = resample_cubic_complex(field, target)
    expected = evaluate_analytic_bandlimited_field(lanczos8.grid)
    lanczos8_error = relative_l2(lanczos8.values, expected)
    lanczos12_error = relative_l2(lanczos12.values, expected)
    assert lanczos8_error < 2.2e-4
    assert lanczos12_error < lanczos8_error
    assert relative_l2(cubic.values, expected) < 2e-3


def test_bandlimited_resampling_is_periodic_fixed_window_interpolation() -> None:
    source = UniformGrid2D.centered(nx=8, ny=6, dx_mm=0.2, dy_mm=0.3)
    lx = source.nx * source.dx_mm
    ly = source.ny * source.dy_mm
    x, y = np.meshgrid(source.x_mm, source.y_mm)
    values = (
        np.exp(2j * np.pi * (2 * x / lx - y / ly))
        + 0.3 * np.exp(2j * np.pi * (-3 * x / lx + 2 * y / ly))
    )
    field = PointField2D(values, source)
    target = UniformGrid2D.centered(nx=16, ny=12, dx_mm=0.1, dy_mm=0.15)

    actual = resample_bandlimited(field, target, batch_size=3)
    tx, ty = np.meshgrid(actual.grid.x_mm, actual.grid.y_mm)
    expected = (
        np.exp(2j * np.pi * (2 * tx / lx - ty / ly))
        + 0.3 * np.exp(2j * np.pi * (-3 * tx / lx + 2 * ty / ly))
    )

    assert source.nx * source.dx_mm == target.nx * target.dx_mm
    assert source.ny * source.dy_mm == target.ny * target.dy_mm
    assert relative_l2(actual.values, expected) < 1e-11


@pytest.mark.slow
def test_zoomfft_is_stable_at_n12288_on_native_and_quarter_sample_coordinates() -> None:
    n = 12_288
    dx = 0.03125
    coordinates = (np.arange(n, dtype=np.float64) - n // 2) * dx
    frequencies = scipy.fft.fftshift(scipy.fft.fftfreq(n, dx))
    spectral_values = (
        np.exp(-0.5 * (frequencies / 2.7) ** 2)
        * np.exp(1j * (0.013 * frequencies**2 - 0.21 * frequencies))
    )
    expected_df = 1.0 / (n * dx)

    zoom, phase, df = fourier_module._build_inverse_zoom(frequencies, coordinates)
    actual_native = fourier_module._apply_inverse_zoom(
        spectral_values, zoom, phase, df, axis=0
    )
    expected_native = (
        scipy.fft.fftshift(scipy.fft.ifft(scipy.fft.ifftshift(spectral_values)))
        * n
        * expected_df
    )
    assert relative_l2(actual_native, expected_native) <= 1e-10

    quarter_offsets = np.array(
        [
            -2.0,
            -1.75,
            -1.5,
            -1.25,
            -1.0,
            -0.75,
            -0.5,
            -0.25,
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
            2.0,
        ]
    )
    probe_coordinates = quarter_offsets * dx
    probe_zoom, probe_phase, probe_df = fourier_module._build_inverse_zoom(
        frequencies, probe_coordinates
    )
    actual_probe = fourier_module._apply_inverse_zoom(
        spectral_values, probe_zoom, probe_phase, probe_df, axis=0
    )
    expected_probe = np.array(
        [
            _compensated_complex_sum(
                spectral_values * np.exp(2j * np.pi * frequencies * coordinate)
            )
            * expected_df
            for coordinate in probe_coordinates
        ]
    )
    assert relative_l2(actual_probe, expected_probe) <= 1e-10
