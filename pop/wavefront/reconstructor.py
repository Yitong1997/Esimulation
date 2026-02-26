"""Rays to wavefront reconstruction."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from pop.core import GridSampling
from pop.utils import format_trace_context


def _griddata_interpolate(
    points: NDArray[np.floating],
    values: NDArray[np.floating],
    x_grid: NDArray[np.floating],
    y_grid: NDArray[np.floating],
    label: str,
) -> NDArray[np.floating]:
    from scipy.interpolate import griddata
    import warnings

    try:
        return griddata(points, values, (x_grid, y_grid), method="cubic", fill_value=0.0)
    except Exception:
        try:
            return griddata(
                points, values, (x_grid, y_grid), method="linear", fill_value=0.0
            )
        except Exception as exc:
            warnings.warn(
                f"griddata failed for {label} ({exc}); falling back to nearest interpolation.",
                RuntimeWarning,
            )
            try:
                return griddata(
                    points, values, (x_grid, y_grid), method="nearest", fill_value=0.0
                )
            except Exception as exc2:
                warnings.warn(
                    f"griddata failed with nearest interpolation for {label} ({exc2}); returning zeros.",
                    RuntimeWarning,
                )
                return np.zeros_like(x_grid, dtype=float)


def _compute_jacobian_amplitude(
    x_in: NDArray[np.floating],
    y_in: NDArray[np.floating],
    x_out: NDArray[np.floating],
    y_out: NDArray[np.floating],
) -> NDArray[np.floating]:
    from scipy.interpolate import RBFInterpolator

    n_rays = len(x_in)
    if n_rays < 4:
        return np.ones(n_rays)

    points_in = np.column_stack([x_in, y_in])
    try:
        interp_x = RBFInterpolator(points_in, x_out, kernel="thin_plate_spline")
        interp_y = RBFInterpolator(points_in, y_out, kernel="thin_plate_spline")
    except Exception:
        return np.ones(n_rays)

    eps = 1e-6
    jacobian_det = np.zeros(n_rays)
    for i in range(n_rays):
        x0, y0 = x_in[i], y_in[i]
        dx_out_dx_in = (
            interp_x([[x0 + eps, y0]])[0] - interp_x([[x0 - eps, y0]])[0]
        ) / (2 * eps)
        dx_out_dy_in = (
            interp_x([[x0, y0 + eps]])[0] - interp_x([[x0, y0 - eps]])[0]
        ) / (2 * eps)
        dy_out_dx_in = (
            interp_y([[x0 + eps, y0]])[0] - interp_y([[x0 - eps, y0]])[0]
        ) / (2 * eps)
        dy_out_dy_in = (
            interp_y([[x0, y0 + eps]])[0] - interp_y([[x0, y0 - eps]])[0]
        ) / (2 * eps)
        jacobian_det[i] = abs(
            dx_out_dx_in * dy_out_dy_in - dx_out_dy_in * dy_out_dx_in
        )

    jacobian_det = np.maximum(jacobian_det, 1e-10)
    amplitude = 1.0 / np.sqrt(jacobian_det)
    mean_amp = float(np.mean(amplitude))
    if mean_amp > 0:
        amplitude = amplitude / mean_amp
    return amplitude


def reconstruct_wavefront(
    ray_x_in: NDArray[np.floating],
    ray_y_in: NDArray[np.floating],
    ray_x_out: NDArray[np.floating],
    ray_y_out: NDArray[np.floating],
    residual_opd_waves: NDArray[np.floating],
    grid_sampling: GridSampling,
    input_amplitude: Optional[NDArray[np.floating]] = None,
    trace_context: Optional[dict[str, object]] = None,
    phase_method: str = "griddata",
    zernike_terms: Optional[int] = None,
    zernike_normalize: bool = True,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    import warnings

    ray_x_in = np.asarray(ray_x_in, dtype=float).ravel()
    ray_y_in = np.asarray(ray_y_in, dtype=float).ravel()
    ray_x_out = np.asarray(ray_x_out, dtype=float).ravel()
    ray_y_out = np.asarray(ray_y_out, dtype=float).ravel()
    residual_opd_waves = np.asarray(residual_opd_waves, dtype=float).ravel()
    if input_amplitude is not None:
        input_amplitude = np.asarray(input_amplitude, dtype=float).ravel()

    expected_len = ray_x_in.size
    if not (
        ray_y_in.size == expected_len
        and ray_x_out.size == expected_len
        and ray_y_out.size == expected_len
        and residual_opd_waves.size == expected_len
        and (input_amplitude is None or input_amplitude.size == expected_len)
    ):
        raise ValueError("Ray arrays must have the same length for wavefront reconstruction.")

    finite_mask = (
        np.isfinite(ray_x_in)
        & np.isfinite(ray_y_in)
        & np.isfinite(ray_x_out)
        & np.isfinite(ray_y_out)
        & np.isfinite(residual_opd_waves)
    )
    if input_amplitude is not None:
        finite_mask &= np.isfinite(input_amplitude)

    if not np.any(finite_mask):
        warnings.warn(
            "No finite rays for wavefront reconstruction; returning zero grids.",
            RuntimeWarning,
        )
        grid_shape = (grid_sampling.grid_size, grid_sampling.grid_size)
        return np.zeros(grid_shape), np.zeros(grid_shape)

    if not np.all(finite_mask):
        dropped = int(finite_mask.size - np.count_nonzero(finite_mask))
        warnings.warn(
            f"Dropping {dropped} invalid rays in wavefront reconstruction.",
            RuntimeWarning,
        )

    ray_x_in = ray_x_in[finite_mask]
    ray_y_in = ray_y_in[finite_mask]
    ray_x_out = ray_x_out[finite_mask]
    ray_y_out = ray_y_out[finite_mask]
    residual_opd_waves = residual_opd_waves[finite_mask]
    if input_amplitude is not None:
        input_amplitude = input_amplitude[finite_mask]
    dropped_invalid = int(expected_len - ray_x_in.size)

    jac_amp = _compute_jacobian_amplitude(ray_x_in, ray_y_in, ray_x_out, ray_y_out)
    if input_amplitude is not None:
        amp_at_rays = jac_amp * input_amplitude
    else:
        amp_at_rays = jac_amp

    coords = (np.arange(grid_sampling.grid_size) - grid_sampling.grid_size // 2) * grid_sampling.sampling_mm
    x_grid, y_grid = np.meshgrid(coords, coords)

    points_out = np.column_stack([ray_x_out, ray_y_out])

    phase_at_rays = residual_opd_waves * 2.0 * np.pi

    amplitude_grid = _griddata_interpolate(
        points_out, amp_at_rays, x_grid, y_grid, label="amplitude"
    )

    if phase_method == "griddata":
        phase_grid = _griddata_interpolate(
            points_out, phase_at_rays, x_grid, y_grid, label="phase"
        )
    elif phase_method == "zernike":
        if zernike_terms is None or zernike_terms < 1:
            raise ValueError("zernike_terms must be a positive integer when using phase_method='zernike'")
        try:
            from pop.wavefront.zernike import fit_zernike_phase, evaluate_zernike_grid

            grid_radius = grid_sampling.physical_size_mm / 2.0
            coeffs, _ = fit_zernike_phase(
                phase_at_rays,
                ray_x_out,
                ray_y_out,
                grid_radius,
                zernike_terms,
                normalize=zernike_normalize,
            )
            phase_grid = evaluate_zernike_grid(
                x_grid, y_grid, grid_radius, coeffs, normalize=zernike_normalize
            )
        except Exception as exc:
            warnings.warn(
                f"Zernike phase fit failed ({exc}); falling back to griddata.",
                RuntimeWarning,
            )
            phase_grid = _griddata_interpolate(
                points_out, phase_at_rays, x_grid, y_grid, label="phase"
            )
    else:
        raise ValueError(f"Unknown phase_method: {phase_method!r}")

    amplitude_grid = np.nan_to_num(amplitude_grid, nan=0.0)
    phase_grid = np.nan_to_num(phase_grid, nan=0.0)

    max_jump = max(
        float(np.max(np.abs(np.diff(phase_grid, axis=0)))),
        float(np.max(np.abs(np.diff(phase_grid, axis=1)))),
    )
    if max_jump > np.pi:
        max_jump_waves = max_jump / (2.0 * np.pi)
        threshold = np.pi
        threshold_waves = threshold / (2.0 * np.pi)
        grid_line = (
            f"网格: size={grid_sampling.grid_size}, "
            f"sampling={grid_sampling.sampling_mm:.6f} mm, "
            f"physical={grid_sampling.physical_size_mm:.6f} mm"
        )
        print("[POP][相位重建报警] 检测到相位不连续，可能采样频率不足。")
        print(
            f"[POP][相位重建报警] max_jump={max_jump:.4f} rad "
            f"({max_jump_waves:.4f} waves) > 阈值={threshold:.4f} rad "
            f"({threshold_waves:.4f} waves)"
        )
        print(
            f"[POP][相位重建报警] {grid_line}; "
            f"有效射线={ray_x_in.size}/{expected_len}, 无效丢弃={dropped_invalid}"
        )
        context_line = format_trace_context(trace_context)
        if context_line:
            print(f"[POP][相位重建报警] 步骤: {context_line}")
        warnings.warn("Phase discontinuity detected in reconstructed wavefront", RuntimeWarning)

    return amplitude_grid, phase_grid
