"""Continuous-Fourier sampling and independent complex-field interpolators.

Arrays use ``[y, x]`` order and spectra use ``[fy, fx]`` order throughout.
The transform pair is

``F(fx, fy) = sum(E(x, y) exp(-i 2 pi (fx x + fy y))) dx dy``

and its inverse uses the positive exponential and ``dfx dfy`` measure.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import scipy.fft
import scipy.ndimage
import scipy.signal

from .models import PointField2D, UniformGrid2D


_AXIS_ULPS = 32


def _normalize_uniform_axis(axis: np.ndarray, *, name: str) -> np.ndarray:
    """Validate and canonicalize an axis using a small ULP-scaled tolerance.

    The returned nodes are generated once as ``start + arange(size) * step``.
    This avoids independently rounded endpoint, phase, and result axes in a
    ZoomFFT evaluation.  The tolerance only permits floating-point accumulation
    around that canonical sequence; it does not admit intentionally nonuniform
    sampling.
    """

    values = np.array(axis, dtype=np.float64, copy=True)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must be one-dimensional with at least two samples")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.all(np.diff(values) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")

    step = float((values[-1] - values[0]) / (values.size - 1))
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError(f"{name} must have a finite positive step")
    canonical = values[0] + np.arange(values.size, dtype=np.float64) * step
    scale = max(
        float(np.max(np.abs(values))),
        abs(float(values[0])),
        abs(step) * (values.size - 1),
        abs(step),
        np.finfo(np.float64).tiny,
    )
    tolerance = _AXIS_ULPS * float(np.spacing(scale))
    if float(np.max(np.abs(values - canonical))) > tolerance:
        raise ValueError(f"{name} must be uniformly spaced")
    return canonical


def _immutable_copy(values: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    contiguous = np.asarray(values, dtype=dtype, order="C")
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    immutable = immutable.reshape(contiguous.shape)
    immutable.setflags(write=False)
    return immutable


def _canonical_grid(grid: UniformGrid2D) -> UniformGrid2D:
    if not isinstance(grid, UniformGrid2D):
        raise ValueError("grid must be a UniformGrid2D")
    x = _normalize_uniform_axis(grid.x_mm, name="x_mm")
    y = _normalize_uniform_axis(grid.y_mm, name="y_mm")
    return UniformGrid2D(x_mm=x, y_mm=y)


def _require_batch_size(batch_size: int) -> int:
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, Integral):
        raise ValueError("batch_size must be a positive integer")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be a positive integer")
    return int(batch_size)


@dataclass(frozen=True)
class Spectrum2D:
    """Immutable samples of a two-dimensional continuous Fourier spectrum."""

    values: np.ndarray
    fx_cpm: np.ndarray
    fy_cpm: np.ndarray
    source_grid: UniformGrid2D

    def __post_init__(self) -> None:
        if not isinstance(self.source_grid, UniformGrid2D):
            raise ValueError("source_grid must be a UniformGrid2D")
        fx = _normalize_uniform_axis(self.fx_cpm, name="fx_cpm")
        fy = _normalize_uniform_axis(self.fy_cpm, name="fy_cpm")
        values = np.asarray(self.values, dtype=np.complex128)
        if values.shape != (fy.size, fx.size):
            raise ValueError("spectrum shape does not match frequency axes")
        if not np.all(np.isfinite(values)):
            raise ValueError("spectrum values must be finite")

        object.__setattr__(
            self, "values", _immutable_copy(values, dtype=np.dtype(np.complex128))
        )
        object.__setattr__(self, "fx_cpm", _immutable_copy(fx, dtype=np.dtype(np.float64)))
        object.__setattr__(self, "fy_cpm", _immutable_copy(fy, dtype=np.dtype(np.float64)))

    @classmethod
    def _from_normalized(
        cls,
        *,
        values: np.ndarray,
        fx_cpm: np.ndarray,
        fy_cpm: np.ndarray,
        source_grid: UniformGrid2D,
    ) -> "Spectrum2D":
        """Build from the exact axes already used by a transform evaluation."""

        spectrum = object.__new__(cls)
        if not isinstance(source_grid, UniformGrid2D):
            raise ValueError("source_grid must be a UniformGrid2D")
        complex_values = np.asarray(values, dtype=np.complex128)
        if complex_values.shape != (fy_cpm.size, fx_cpm.size):
            raise ValueError("spectrum shape does not match frequency axes")
        if not np.all(np.isfinite(complex_values)):
            raise ValueError("spectrum values must be finite")
        object.__setattr__(
            spectrum,
            "values",
            _immutable_copy(complex_values, dtype=np.dtype(np.complex128)),
        )
        object.__setattr__(
            spectrum,
            "fx_cpm",
            _immutable_copy(fx_cpm, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(
            spectrum,
            "fy_cpm",
            _immutable_copy(fy_cpm, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(spectrum, "source_grid", source_grid)
        return spectrum


def forward_continuous_spectrum(
    field: PointField2D, *, workers: int = -1
) -> Spectrum2D:
    """Evaluate the continuous forward convention on the natural FFT grid."""

    if not isinstance(field, PointField2D):
        raise ValueError("field must be a PointField2D")
    values = scipy.fft.fftshift(
        scipy.fft.fft2(scipy.fft.ifftshift(field.values), workers=workers)
    ) * field.grid.pixel_area_mm2
    fx = scipy.fft.fftshift(
        scipy.fft.fftfreq(field.grid.nx, field.grid.dx_mm)
    )
    fy = scipy.fft.fftshift(
        scipy.fft.fftfreq(field.grid.ny, field.grid.dy_mm)
    )
    return Spectrum2D(
        values=values,
        fx_cpm=fx,
        fy_cpm=fy,
        source_grid=field.grid,
    )


def _build_inverse_zoom_normalized(
    frequencies: np.ndarray, coordinates: np.ndarray
) -> tuple[scipy.signal.ZoomFFT, np.ndarray, float]:
    df = float(frequencies[1] - frequencies[0])
    zoom = scipy.signal.ZoomFFT(
        frequencies.size,
        [float(coordinates[0]), float(coordinates[-1])],
        m=coordinates.size,
        fs=1.0 / df,
        endpoint=True,
    )
    phase = np.exp(2j * np.pi * frequencies[0] * coordinates)
    return zoom, phase, df


def _build_inverse_zoom(
    frequencies: np.ndarray, coordinates: np.ndarray
) -> tuple[scipy.signal.ZoomFFT, np.ndarray, float]:
    normalized_frequencies = _normalize_uniform_axis(frequencies, name="frequencies")
    normalized_coordinates = _normalize_uniform_axis(coordinates, name="coordinates")
    return _build_inverse_zoom_normalized(
        normalized_frequencies, normalized_coordinates
    )


def _apply_inverse_zoom(
    values: np.ndarray,
    zoom: scipy.signal.ZoomFFT,
    phase: np.ndarray,
    df: float,
    *,
    axis: int,
) -> np.ndarray:
    transformed = np.conj(zoom(np.conj(values), axis=axis))
    shape = [1] * transformed.ndim
    shape[axis] = phase.size
    return transformed * phase.reshape(shape) * df


def _build_forward_zoom_normalized(
    coordinates: np.ndarray, frequencies: np.ndarray
) -> tuple[scipy.signal.ZoomFFT, np.ndarray, float]:
    dx = float(coordinates[1] - coordinates[0])
    zoom = scipy.signal.ZoomFFT(
        coordinates.size,
        [float(frequencies[0]), float(frequencies[-1])],
        m=frequencies.size,
        fs=1.0 / dx,
        endpoint=True,
    )
    phase = np.exp(-2j * np.pi * frequencies * coordinates[0])
    return zoom, phase, dx


def _build_forward_zoom(
    coordinates: np.ndarray, frequencies: np.ndarray
) -> tuple[scipy.signal.ZoomFFT, np.ndarray, float]:
    normalized_coordinates = _normalize_uniform_axis(coordinates, name="coordinates")
    normalized_frequencies = _normalize_uniform_axis(frequencies, name="frequencies")
    return _build_forward_zoom_normalized(
        normalized_coordinates, normalized_frequencies
    )


def _apply_forward_zoom(
    values: np.ndarray,
    zoom: scipy.signal.ZoomFFT,
    phase: np.ndarray,
    dx: float,
    *,
    axis: int,
) -> np.ndarray:
    transformed = zoom(values, axis=axis)
    shape = [1] * transformed.ndim
    shape[axis] = phase.size
    return transformed * phase.reshape(shape) * dx


def evaluate_spectrum_czt(
    spectrum: Spectrum2D,
    output_grid: UniformGrid2D,
    *,
    batch_size: int = 128,
) -> PointField2D:
    """Evaluate an inverse continuous spectrum sum on a uniform output grid."""

    if not isinstance(spectrum, Spectrum2D):
        raise ValueError("spectrum must be a Spectrum2D")
    batch_size = _require_batch_size(batch_size)
    output = _canonical_grid(output_grid)
    # Spectrum2D has already validated and canonicalized these immutable axes.
    # Re-normalizing can move legal large-offset nodes by an ULP and would make
    # the returned spectrum coordinates differ from the nodes evaluated here.
    fx = spectrum.fx_cpm
    fy = spectrum.fy_cpm

    xzoom, xphase, dfx = _build_inverse_zoom_normalized(fx, output.x_mm)
    after_x = np.empty((fy.size, output.nx), dtype=np.complex128)
    for y0 in range(0, fy.size, batch_size):
        ys = slice(y0, min(y0 + batch_size, fy.size))
        after_x[ys, :] = _apply_inverse_zoom(
            spectrum.values[ys, :], xzoom, xphase, dfx, axis=1
        )

    yzoom, yphase, dfy = _build_inverse_zoom_normalized(fy, output.y_mm)
    result = np.empty((output.ny, output.nx), dtype=np.complex128)
    for x0 in range(0, output.nx, batch_size):
        xs = slice(x0, min(x0 + batch_size, output.nx))
        result[:, xs] = _apply_inverse_zoom(
            after_x[:, xs], yzoom, yphase, dfy, axis=0
        )
    return PointField2D(result, output)


def evaluate_field_fourier_czt(
    field: PointField2D,
    fx_cpm: np.ndarray,
    fy_cpm: np.ndarray,
    *,
    batch_size: int = 128,
) -> Spectrum2D:
    """Evaluate the forward continuous convention on uniform frequency axes."""

    if not isinstance(field, PointField2D):
        raise ValueError("field must be a PointField2D")
    batch_size = _require_batch_size(batch_size)
    fx = _normalize_uniform_axis(fx_cpm, name="fx_cpm")
    fy = _normalize_uniform_axis(fy_cpm, name="fy_cpm")
    x = _normalize_uniform_axis(field.grid.x_mm, name="x_mm")
    y = _normalize_uniform_axis(field.grid.y_mm, name="y_mm")

    xzoom, xphase, dx = _build_forward_zoom_normalized(x, fx)
    after_x = np.empty((field.grid.ny, fx.size), dtype=np.complex128)
    for y0 in range(0, field.grid.ny, batch_size):
        ys = slice(y0, min(y0 + batch_size, field.grid.ny))
        after_x[ys, :] = _apply_forward_zoom(
            field.values[ys, :], xzoom, xphase, dx, axis=1
        )

    yzoom, yphase, dy = _build_forward_zoom_normalized(y, fy)
    result = np.empty((fy.size, fx.size), dtype=np.complex128)
    for x0 in range(0, fx.size, batch_size):
        xs = slice(x0, min(x0 + batch_size, fx.size))
        result[:, xs] = _apply_forward_zoom(
            after_x[:, xs], yzoom, yphase, dy, axis=0
        )
    return Spectrum2D._from_normalized(
        values=result,
        fx_cpm=fx,
        fy_cpm=fy,
        source_grid=field.grid,
    )


def resample_bandlimited(
    field: PointField2D,
    output_grid: UniformGrid2D,
    *,
    batch_size: int = 128,
    workers: int = -1,
) -> PointField2D:
    """Periodically interpolate a finite grid as a trigonometric polynomial.

    This operation is not zero extension.  A fixed-window refinement is valid
    when each axis separately satisfies ``N_target * delta_target =
    N_source * delta_source``.  Expanding a finite window requires the explicit
    zero-extension operation defined by the caller's sampling workflow.
    """

    spectrum = forward_continuous_spectrum(field, workers=workers)
    return evaluate_spectrum_czt(spectrum, output_grid, batch_size=batch_size)


def _lanczos_weights(
    source: np.ndarray, target: np.ndarray, *, lobes: int
) -> np.ndarray:
    step = float(source[1] - source[0])
    sample_positions = (target[:, np.newaxis] - source[0]) / step
    distances = sample_positions - np.arange(source.size, dtype=np.float64)
    weights = np.sinc(distances) * np.sinc(distances / lobes)
    weights[np.abs(distances) >= lobes] = 0.0
    normalization = np.sum(weights, axis=1, keepdims=True)
    normalized = np.zeros_like(weights)
    np.divide(
        weights,
        normalization,
        out=normalized,
        where=np.abs(normalization) > 32 * np.finfo(np.float64).eps,
    )
    return normalized


def resample_lanczos_complex(
    field: PointField2D,
    output_grid: UniformGrid2D,
    *,
    lobes: int = 8,
) -> PointField2D:
    """Apply separable normalized Lanczos-8 or Lanczos-12 to a complex field."""

    if not isinstance(field, PointField2D):
        raise ValueError("field must be a PointField2D")
    if isinstance(lobes, (bool, np.bool_)) or not isinstance(lobes, Integral):
        raise ValueError("lobes must be 8 or 12")
    if int(lobes) not in (8, 12):
        raise ValueError("lobes must be 8 or 12")
    output = _canonical_grid(output_grid)
    source_x = _normalize_uniform_axis(field.grid.x_mm, name="x_mm")
    source_y = _normalize_uniform_axis(field.grid.y_mm, name="y_mm")
    wx = _lanczos_weights(source_x, output.x_mm, lobes=int(lobes))
    wy = _lanczos_weights(source_y, output.y_mm, lobes=int(lobes))

    real = wy @ field.values.real @ wx.T
    imaginary = wy @ field.values.imag @ wx.T
    return PointField2D(real + 1j * imaginary, output)


def resample_cubic_complex(
    field: PointField2D, output_grid: UniformGrid2D
) -> PointField2D:
    """Cubic sensitivity check applied independently to real and imaginary parts."""

    if not isinstance(field, PointField2D):
        raise ValueError("field must be a PointField2D")
    output = _canonical_grid(output_grid)
    source_x = _normalize_uniform_axis(field.grid.x_mm, name="x_mm")
    source_y = _normalize_uniform_axis(field.grid.y_mm, name="y_mm")
    x_indices = (output.x_mm - source_x[0]) / (source_x[1] - source_x[0])
    y_indices = (output.y_mm - source_y[0]) / (source_y[1] - source_y[0])
    yi, xi = np.meshgrid(y_indices, x_indices, indexing="ij")
    coordinates = np.stack((yi, xi))

    real = scipy.ndimage.map_coordinates(
        field.values.real,
        coordinates,
        order=3,
        mode="constant",
        cval=0.0,
    )
    imaginary = scipy.ndimage.map_coordinates(
        field.values.imag,
        coordinates,
        order=3,
        mode="constant",
        cval=0.0,
    )
    return PointField2D(real + 1j * imaginary, output)


def point_to_cell_energy(field: PointField2D) -> np.ndarray:
    """Convert point-value field samples to square-root cell-energy samples."""

    if not isinstance(field, PointField2D):
        raise ValueError("field must be a PointField2D")
    return np.array(
        field.values * np.sqrt(field.grid.pixel_area_mm2),
        dtype=np.complex128,
        copy=True,
    )


def cell_energy_to_point(
    samples: np.ndarray, grid: UniformGrid2D
) -> PointField2D:
    """Convert square-root cell-energy samples back to point values."""

    values = np.array(samples, dtype=np.complex128, copy=True)
    if values.shape != (grid.ny, grid.nx):
        raise ValueError("cell-energy sample shape does not match grid")
    if not np.all(np.isfinite(values)):
        raise ValueError("cell-energy samples must be finite")
    return PointField2D(values / np.sqrt(grid.pixel_area_mm2), grid)


__all__ = [
    "Spectrum2D",
    "cell_energy_to_point",
    "evaluate_field_fourier_czt",
    "evaluate_spectrum_czt",
    "forward_continuous_spectrum",
    "point_to_cell_energy",
    "resample_bandlimited",
    "resample_cubic_complex",
    "resample_lanczos_complex",
]
