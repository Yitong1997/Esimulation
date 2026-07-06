"""Free-space propagation interface (PROPER)."""

from __future__ import annotations

import contextlib
import copy
import io
from typing import Optional, Tuple, Any
import warnings
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pop.core import GridSampling, OpticalAxisState, PilotBeamParams, PropagationState
from pop.utils import format_trace_context


def _unwrap_phase_2d(phase: NDArray[np.floating]) -> NDArray[np.floating]:
    """Robust 2D unwrapping with best-available backend."""
    try:
        from skimage.restoration import unwrap_phase

        return np.asarray(unwrap_phase(phase), dtype=float)
    except Exception:
        pass
    try:
        from scipy.signal import unwrap as scipy_unwrap

        return np.asarray(scipy_unwrap(scipy_unwrap(phase, axis=0), axis=1), dtype=float)
    except Exception:
        return np.asarray(np.unwrap(np.unwrap(phase, axis=0), axis=1), dtype=float)


def _compute_intensity_mask(
    amplitude: NDArray[np.floating],
    threshold_ratio: float = 0.10,
) -> NDArray[np.bool_]:
    amp = np.asarray(amplitude, dtype=float)
    if amp.size == 0:
        return np.zeros_like(amp, dtype=bool)
    max_amp = float(np.max(amp))
    if max_amp <= 0.0:
        return np.zeros_like(amp, dtype=bool)
    return amp > (threshold_ratio * max_amp)


def _max_phase_jump(phase_grid: NDArray[np.floating]) -> float:
    if phase_grid.size == 0:
        return 0.0
    jump_y = (
        float(np.max(np.abs(np.diff(phase_grid, axis=0))))
        if phase_grid.shape[0] > 1
        else 0.0
    )
    jump_x = (
        float(np.max(np.abs(np.diff(phase_grid, axis=1))))
        if phase_grid.shape[1] > 1
        else 0.0
    )
    return max(jump_x, jump_y)


def _phase_jump_stats(
    phase_grid: NDArray[np.floating],
    mask: Optional[NDArray[np.bool_]] = None,
) -> dict[str, float]:
    """Summarize nearest-neighbor phase jumps, optionally only inside a valid mask."""
    if phase_grid.size == 0:
        return {"max": 0.0, "p999": 0.0, "p99": 0.0, "rms": 0.0, "count": 0.0}

    jumps: list[NDArray[np.floating]] = []
    if phase_grid.shape[0] > 1:
        jump_y = np.abs(np.diff(phase_grid, axis=0))
        if mask is not None and mask.shape == phase_grid.shape:
            jump_y = jump_y[mask[:-1, :] & mask[1:, :]]
        else:
            jump_y = jump_y.ravel()
        jumps.append(np.asarray(jump_y, dtype=float))
    if phase_grid.shape[1] > 1:
        jump_x = np.abs(np.diff(phase_grid, axis=1))
        if mask is not None and mask.shape == phase_grid.shape:
            jump_x = jump_x[mask[:, :-1] & mask[:, 1:]]
        else:
            jump_x = jump_x.ravel()
        jumps.append(np.asarray(jump_x, dtype=float))

    if not jumps:
        return {"max": 0.0, "p999": 0.0, "p99": 0.0, "rms": 0.0, "count": 0.0}
    values = np.concatenate(jumps)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"max": 0.0, "p999": 0.0, "p99": 0.0, "rms": 0.0, "count": 0.0}
    return {
        "max": float(np.max(values)),
        "p999": float(np.percentile(values, 99.9)),
        "p99": float(np.percentile(values, 99.0)),
        "rms": float(np.sqrt(np.mean(values ** 2))),
        "count": float(values.size),
    }


def _auto_unwarp_debug_path(
    trace_context: Optional[dict[str, Any]],
    output_dir: Optional[str | Path],
    accepted: bool,
) -> Optional[Path]:
    if output_dir is None:
        return None
    idx = "unknown"
    pos = "incident"
    if trace_context:
        idx = str(trace_context.get("to_surface_index", idx))
        pos = str(trace_context.get("to_position", pos)).lower()
    try:
        surface_idx = int(idx)
        prefix = f"surface_{surface_idx:02d}_{pos}"
    except (TypeError, ValueError):
        prefix = f"surface_{idx}_{pos}"
    status = "ACCEPTED" if accepted else "REJECTED"
    return Path(output_dir) / f"{prefix}_auto_unwarp_{status}.png"


def _plot_auto_unwarp_debug(
    *,
    amplitude: NDArray[np.floating],
    residual_folded: NDArray[np.floating],
    residual_unwrapped: NDArray[np.floating],
    mask: NDArray[np.bool_],
    stats_before: dict[str, float],
    stats_after: dict[str, float],
    accepted: bool,
    save_path: Optional[str | Path],
    trace_context: Optional[dict[str, Any]] = None,
) -> None:
    if save_path is None:
        return
    try:
        import matplotlib.pyplot as plt

        folded_waves = residual_folded / (2.0 * np.pi)
        unwrapped_waves = residual_unwrapped / (2.0 * np.pi)
        delta_waves = unwrapped_waves - folded_waves
        amp = np.asarray(amplitude, dtype=float)
        amp_norm = amp / float(np.max(amp)) if amp.size and np.max(amp) > 0 else amp

        before_jumps = []
        after_jumps = []
        if residual_folded.shape[0] > 1:
            pair = mask[:-1, :] & mask[1:, :]
            before_jumps.append(np.abs(np.diff(residual_folded, axis=0))[pair])
            after_jumps.append(np.abs(np.diff(residual_unwrapped, axis=0))[pair])
        if residual_folded.shape[1] > 1:
            pair = mask[:, :-1] & mask[:, 1:]
            before_jumps.append(np.abs(np.diff(residual_folded, axis=1))[pair])
            after_jumps.append(np.abs(np.diff(residual_unwrapped, axis=1))[pair])
        before_vals = np.concatenate(before_jumps) if before_jumps else np.array([])
        after_vals = np.concatenate(after_jumps) if after_jumps else np.array([])
        before_vals = before_vals[np.isfinite(before_vals)] / (2.0 * np.pi)
        after_vals = after_vals[np.isfinite(after_vals)] / (2.0 * np.pi)

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        vlim = max(
            0.5,
            float(np.nanpercentile(np.abs(folded_waves[mask]), 99.0)) if np.any(mask) else 0.5,
        )

        def _imshow(ax, data, title, cmap="RdBu_r", vmin=None, vmax=None):
            im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        _imshow(axes[0, 0], folded_waves, "Folded residual (waves)", vmin=-vlim, vmax=vlim)
        _imshow(axes[0, 1], unwrapped_waves, "Unwrapped residual (waves)")
        _imshow(axes[0, 2], delta_waves, "Unwrap delta (waves)", cmap="viridis")
        _imshow(axes[1, 0], amp_norm, "Normalized amplitude", cmap="magma", vmin=0.0, vmax=1.0)
        _imshow(axes[1, 1], mask.astype(float), "Acceptance mask", cmap="gray", vmin=0.0, vmax=1.0)

        ax = axes[1, 2]
        if before_vals.size:
            ax.hist(before_vals, bins=80, alpha=0.6, label="folded", color="tab:red")
        if after_vals.size:
            ax.hist(after_vals, bins=80, alpha=0.6, label="unwrapped", color="tab:green")
        ax.set_title("Masked neighbor jumps (waves)")
        ax.set_xlabel("absolute jump")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

        context_line = format_trace_context(trace_context)
        status = "ACCEPTED" if accepted else "REJECTED"
        fig.suptitle(
            f"AutoUnwarp@Incident {status}\n"
            f"masked max: {stats_before['max']:.3f} -> {stats_after['max']:.3f} rad, "
            f"p99.9: {stats_before['p999']:.3f} -> {stats_after['p999']:.3f} rad\n"
            f"{context_line}",
            fontsize=11,
        )
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[POP][AutoUnwarp@Incident] debug plot saved: {save_path}")
    except Exception as exc:
        print(f"[POP][AutoUnwarp@Incident] debug plot failed: {exc}")


def _detect_wrapped_residual_near_one_wave(
    residual_folded: NDArray[np.floating],
    amplitude: NDArray[np.floating],
    threshold_ratio: float = 0.10,
    pv_target_waves: float = 1.0,
    pv_tol_waves: float = 0.10,
) -> tuple[bool, float, float, float]:
    """Detect '-0.5~0.5 wave oscillation' folding pattern in valid intensity region."""
    mask = _compute_intensity_mask(amplitude, threshold_ratio=threshold_ratio)
    if not np.any(mask):
        return False, float("nan"), float("nan"), float("nan")
    residual_waves = residual_folded / (2.0 * np.pi)
    valid = residual_waves[mask]
    pv_waves = float(np.ptp(valid))
    min_waves = float(np.min(valid))
    max_waves = float(np.max(valid))
    is_near_target = abs(pv_waves - pv_target_waves) <= pv_tol_waves
    touches_pm_half = (min_waves <= -0.45) and (max_waves >= 0.45)
    return bool(is_near_target and touches_pm_half), pv_waves, min_waves, max_waves


def _should_auto_unwarp_at_incident(trace_context: Optional[dict[str, Any]]) -> bool:
    if not trace_context:
        return False
    return str(trace_context.get("to_position", "")).lower() == "entrance"


def _is_surface_in_refit_scope(
    target_surface_index: int,
    pilot_refit_surface_indices: Optional[list[int]],
) -> bool:
    if pilot_refit_surface_indices is None:
        return False
    return target_surface_index in pilot_refit_surface_indices


def _enable_auto_unwarp_for_current_step(
    auto_unwarp_at_incident: bool,
    target_surface_index: int,
    target_position: str,
    auto_unwarp_surface_indices: Optional[Sequence[int] | str] = None,
) -> bool:
    """Enable incident auto-unwarp on specified surfaces (or all if indices == 'all')."""
    if not auto_unwarp_at_incident:
        return False
    if str(target_position).lower() != "entrance":
        return False
    
    if isinstance(auto_unwarp_surface_indices, str) and auto_unwarp_surface_indices.lower() == "all":
        return True
    
    if auto_unwarp_surface_indices is None:
        return False

    return target_surface_index in auto_unwarp_surface_indices


def _compute_signed_distance(
    current_axis: OpticalAxisState,
    target_axis: OpticalAxisState,
) -> float:
    # Prefer traced optical-axis path length when available; this is the most
    # physically consistent quantity for q-parameter propagation.
    current_path = getattr(current_axis, "path_length", None)
    target_path = getattr(target_axis, "path_length", None)
    if current_path is not None and target_path is not None:
        try:
            delta_path = float(target_path) - float(current_path)
            if np.isfinite(delta_path):
                return delta_path
        except Exception:
            pass

    displacement = target_axis.position - current_axis.position
    if float(np.linalg.norm(displacement)) < 1e-12:
        return 0.0
    direction = np.asarray(current_axis.direction, dtype=float)
    projected = float(np.dot(displacement, direction))
    if abs(projected) < 1e-12:
        return 0.0
    return projected


def _propagate_pilot_with_compensated_q(
    pilot: PilotBeamParams,
    distance_mm: float,
) -> PilotBeamParams:
    """Propagate pilot q with extended precision on the real branch.

    Near focus, q_real is obtained by subtracting two large values accumulated
    over long segments. Use longdouble to reduce cancellation drift before
    converting back to the regular PilotBeamParams representation.
    """
    q = complex(pilot.q_parameter)
    q_real = np.longdouble(np.real(q))
    q_imag = np.longdouble(np.imag(q))
    distance_ld = np.longdouble(distance_mm)
    q_new = complex(float(q_real + distance_ld), float(q_imag))
    return PilotBeamParams.from_q_parameter(
        q_new,
        pilot.wavelength_um,
        pilot.current_refractive_index,
    )


def _check_residual_phase_range(
    residual_phase: NDArray[np.floating],
    amplitude: NDArray[np.floating],
    threshold: float = np.pi / 2.0,
    grid_sampling: GridSampling | None = None,
    trace_context: Optional[dict[str, Any]] = None,
) -> None:
    if amplitude.size == 0:
        return
    max_amp = float(np.max(amplitude))
    if max_amp <= 0:
        return
    valid_mask = amplitude > 0.01 * max_amp
    if not np.any(valid_mask):
        return
    max_residual = float(np.max(np.abs(residual_phase[valid_mask])))
    if max_residual > threshold:
        max_residual_waves = max_residual / (2.0 * np.pi)
        threshold_waves = threshold / (2.0 * np.pi)
        rms_residual = float(np.sqrt(np.mean(residual_phase[valid_mask] ** 2)))
        rms_residual_waves = rms_residual / (2.0 * np.pi)
        valid_count = int(np.count_nonzero(valid_mask))
        total_count = int(valid_mask.size)
        grid_line = None
        if grid_sampling is not None:
            grid_line = (
                f"网格: size={grid_sampling.grid_size}, "
                f"sampling={grid_sampling.sampling_mm:.6f} mm, "
                f"physical={grid_sampling.physical_size_mm:.6f} mm"
            )
        else:
            grid_line = f"网格: shape={residual_phase.shape}"
        print(
            "[POP][残差报警] 残差相位过大，可能导致采样/重建问题："
            f"max={max_residual:.4f} rad ({max_residual_waves:.4f} waves) "
            f"> 阈值={threshold:.4f} rad ({threshold_waves:.4f} waves)"
        )
        print(
            f"[POP][残差报警] {grid_line}; "
            f"RMS={rms_residual:.4f} rad ({rms_residual_waves:.4f} waves); "
            f"有效点={valid_count}/{total_count} (amp>1%峰值)"
        )
        context_line = format_trace_context(trace_context)
        if context_line:
            print(f"[POP][残差报警] 步骤: {context_line}")
        warnings.warn(
            f"Residual phase exceeds recommended range: {max_residual:.2f} rad "
            f"> {threshold:.2f} rad. This may cause sampling issues.",
            UserWarning,
        )


def _sync_proper_gaussian_params(
    wfo: Any,
    pilot_beam_params: PilotBeamParams | None,
    update_reference_surface: bool = False,
    skip_wfarr_transition: bool = False,
) -> None:
    """同步 PROPER wfo 对象的高斯光束参数与 PilotBeamParams。

    参数:
        wfo: PROPER WaveFront 对象
        pilot_beam_params: 新的 pilot beam 参数
        update_reference_surface: 是否更新 reference_surface 和 beam_type_old
        skip_wfarr_transition: 为 True 时只更新元数据（reference_surface,
            beam_type_old），跳过 prop_qphase 对 wfarr 的修改。
            适用于调用方会自行重建 wfarr 的场景（如 Grid Refit）。
    """
    if wfo is None or pilot_beam_params is None:
        return
    import proper

    current_n = pilot_beam_params.current_refractive_index or 1.0
    wfo.w0 = pilot_beam_params.waist_radius_mm * 1e-3
    wfo.z_Rayleigh = (pilot_beam_params.rayleigh_length_mm / current_n) * 1e-3
    z_m = float(getattr(wfo, "z", 0.0))
    wfo.z_w0 = z_m + (pilot_beam_params.waist_position_mm / current_n) * 1e-3

    if update_reference_surface:
        import proper

        rayleigh_factor = proper.rayleigh_factor
        current_z = float(getattr(wfo, "z", 0.0))
        dist_from_waist = abs(wfo.z_w0 - current_z)
        
        # Determine target physical state
        if dist_from_waist < rayleigh_factor * wfo.z_Rayleigh:
            target_ref = "PLANAR"
            target_beam_type = "INSIDE_"
        else:
            target_ref = "SPHERI"
            target_beam_type = "OUTSIDE"
            
        current_ref = getattr(wfo, "reference_surface", "PLANAR")
        
        if not skip_wfarr_transition:
            # State transition: Synchronize wfarr content via prop_qphase
            if current_ref == "PLANAR" and target_ref == "SPHERI":
                # PLANAR -> SPHERI: Remove curvature (physically moving to far field representation)
                # R is positive for diverging beam (z > z_w0), negative for converging (z < z_w0)
                # prop_qphase(wfo, -R) removes spherical phase of radius R.
                # R_ref = z - z_w0 matches PROPER's convention.
                R = current_z - wfo.z_w0
                if abs(R) > 1e-12:
                    proper.prop_qphase(wfo, -R)
                    
            elif current_ref == "SPHERI" and target_ref == "PLANAR":
                # SPHERI -> PLANAR: Add curvature (physically moving to near field representation)
                R = current_z - wfo.z_w0
                if abs(R) > 1e-12:
                    proper.prop_qphase(wfo, R)

        wfo.reference_surface = target_ref
        wfo.beam_type_old = target_beam_type



def propagate_free_space(
    wfo: Any,
    distance_mm: float,
    n: float = 1.0,
    force_asm: Optional[bool] = None,
    auto_asm: bool = True,
    pilot_beam: Optional[PilotBeamParams] = None,
    free_space_mode: str = "native_proper",
) -> Tuple[GridSampling, str]:
    import proper

    distance_m = distance_mm * 1e-3

    if free_space_mode == "native_proper":
        if pilot_beam is not None:
            _sync_proper_gaussian_params(wfo, pilot_beam, update_reference_surface=False)
        proper.prop_propagate(wfo, distance_m / n)
        sampling_m = proper.prop_get_sampling(wfo)
        if hasattr(wfo, "dx"):
            try:
                wfo.dx = sampling_m
            except AttributeError:
                pass
        return GridSampling.from_proper(wfo), "PROPER native"

    if free_space_mode != "legacy_auto_asm":
        raise ValueError("free_space_mode must be 'native_proper' or 'legacy_auto_asm'")
    
    # 1. Determine Strategy
    use_asm = False
    algo_name = "FFT (Standard)"
    reason = "Default"

    # Strategy A: Manual Override
    if force_asm is not None:
        use_asm = force_asm
        algo_name = "ASM (Forced)" if use_asm else "FFT (Standard)"
        reason = "Manual Override"

        # Fix 2: Sync state even in forced mode to ensure consistency
        if use_asm and pilot_beam is not None:
             _sync_proper_gaussian_params(wfo, pilot_beam, update_reference_surface=True)
    
    # Strategy B: Auto-ASM (Sampling Competition)
    elif auto_asm and pilot_beam is not None:
        # 首先判断是否涉及远场传输
        # 使用Rayleigh距离判断：|z - z_w0| < z_R 为近场
        z_R_mm = pilot_beam.rayleigh_length_mm
        current_n = pilot_beam.current_refractive_index or 1.0
        # Fix 3: Consistent Rayleigh definition
        rayleigh_factor = proper.rayleigh_factor
        z_R_effective_mm = (z_R_mm / current_n) * rayleigh_factor
        # Ensure PROPER state is consistent with the pilot beam before selecting propagator.
        _sync_proper_gaussian_params(wfo, pilot_beam, update_reference_surface=True)
        
        # 当前位置相对束腰的距离
        dist_from_waist_start_mm = abs(pilot_beam.waist_position_mm / current_n)
        # 传输后相对束腰的距离
        target_pilot = pilot_beam.propagate(distance_mm)
        dist_from_waist_end_mm = abs(target_pilot.waist_position_mm / current_n)
        
        # 判断是否为近场-近场传输（起点和终点都在近场）
        is_start_near_field = dist_from_waist_start_mm < z_R_effective_mm
        is_end_near_field = dist_from_waist_end_mm < z_R_effective_mm
        is_near_near_propagation = is_start_near_field and is_end_near_field
        is_far_near_propagation = (not is_start_near_field) and is_end_near_field
        
        if is_near_near_propagation:
            # 近场-近场传输：直接使用ASM，网格尺寸保持不变
            use_asm = True
            algo_name = "ASM (Near-Near)"
            reason = f"Near-field to Near-field (|z_start|={dist_from_waist_start_mm:.2f}mm, |z_end|={dist_from_waist_end_mm:.2f}mm < z_R={z_R_effective_mm:.2f}mm)"
        elif is_far_near_propagation:
            # 远场-近场传输：直接使用PROPER默认选项（聚焦）
            use_asm = False
            algo_name = "FFT (Far-Near)"
            reason = (
                "Far-field to Near-field: "
                f"|z_start|={dist_from_waist_start_mm:.2f}mm >= z_R={z_R_effective_mm:.2f}mm, "
                f"|z_end|={dist_from_waist_end_mm:.2f}mm < z_R"
            )
        else:
            # 涉及远场传输：进行采样竞争检测
            # Get Current Grid Info
            grid_size = proper.prop_get_gridsize(wfo)
            sampling_m = proper.prop_get_sampling(wfo)
            L_current_mm = grid_size * sampling_m * 1e3
            
            try:
                w_target_mm = float(target_pilot.spot_size_mm)
                if not np.isfinite(w_target_mm) or w_target_mm <= 0:
                    raise ValueError(f"Invalid target beam size: {w_target_mm}")
                
                # 1. Calc ASM Score (Constant Grid)
                # Fix 1: Remove artificial restriction on ASM
                if L_current_mm > 0:
                    dx_asm_mm = L_current_mm / grid_size
                    score_asm = (2 * w_target_mm) / dx_asm_mm
                else:
                    score_asm = 0.0
                
                # 2. Calc FFT Score (PROPER regime-aware scaling)
                # Use effective distances to waist (dz_in/dz_out) to match PROPER's
                # STW/WTS/PTP grid scaling.
                if L_current_mm > 0:
                    wavelength_mm = wfo.lamda * 1e3
                    dz_in_eff_mm = max(dist_from_waist_start_mm, 1e-12)
                    dz_out_eff_mm = max(dist_from_waist_end_mm, 1e-12)
                    if is_start_near_field and is_end_near_field:
                        L_fft_mm = L_current_mm
                    elif is_start_near_field and (not is_end_near_field):
                        # INSIDE -> OUTSIDE: PTP + WTS
                        L_fft_mm = wavelength_mm * dz_out_eff_mm * grid_size / L_current_mm
                    elif (not is_start_near_field) and is_end_near_field:
                        # OUTSIDE -> INSIDE: STW + PTP
                        L_fft_mm = wavelength_mm * dz_in_eff_mm * grid_size / L_current_mm
                    else:
                        # OUTSIDE -> OUTSIDE: STW + WTS
                        L_fft_mm = L_current_mm * (dz_out_eff_mm / dz_in_eff_mm)
                    dx_fft_mm = L_fft_mm / grid_size
                    score_fft = (2 * w_target_mm) / dx_fft_mm if dx_fft_mm > 0 else 0.0
                else:
                    score_fft = 0.0

                # 3. Decision
                # Adjusted Priority: Prefer FFT by default.
                # Only force ASM if FFT grid size explodes (approx > 10x ASM size).
                # Since Score ~ 1/GridSize, this implies Score_ASM > 10.0 * Score_FFT.
                if score_asm >= 10.0 * score_fft:
                    use_asm = True
                    algo_name = "ASM (Auto)"
                    reason = f"Grid Explosion Avoidance: ASM Score({score_asm:.1f}) >> FFT({score_fft:.1f})"
                else:
                    use_asm = False
                    algo_name = "FFT (Auto)"
                    reason = f"Prefer FFT: Score FFT({score_fft:.1f}) vs ASM({score_asm:.1f})"

                # Fix 4: Undersampling Warning
                selected_score = score_asm if use_asm else score_fft
                if selected_score < 2.0:
                    warnings.warn(
                        f"[POP][Undersampling] Selected {algo_name} has low sampling score ({selected_score:.2f} < 2.0). "
                        f"Target Beam = {w_target_mm:.3f} mm. Results may be inaccurate.",
                        UserWarning
                    )
                    
            except Exception as e:
                # Fallback if calculation fails
                use_asm = False
                algo_name = "FFT (Fallback)"
                reason = f"Error: {e}"

    # Logging — Zemax POP 风格
    if abs(distance_mm) > 0.1 and pilot_beam is not None:
        import proper as _proper_log
        _grid_size = _proper_log.prop_get_gridsize(wfo)
        _sampling_m = _proper_log.prop_get_sampling(wfo)
        _dx_mm = _sampling_m * 1e3
        _L_mm = _grid_size * _dx_mm

        _w0 = pilot_beam.waist_radius_mm
        _z_w0 = pilot_beam.waist_position_mm
        _z_R = pilot_beam.rayleigh_length_mm
        _w_spot = pilot_beam.spot_size_mm
        _beam_px = (2.0 * _w_spot) / _dx_mm if _dx_mm > 0 else 0.0
        _current_n = pilot_beam.current_refractive_index or 1.0

        # 判断近场/远场
        _rayleigh_factor = _proper_log.rayleigh_factor
        _z_R_eff = (_z_R / _current_n) * _rayleigh_factor
        _dist_start = abs(_z_w0 / _current_n)
        _target_pilot = pilot_beam.propagate(distance_mm)
        _dist_end = abs(_target_pilot.waist_position_mm / _current_n)
        _start_inside = _dist_start < _z_R_eff
        _end_inside = _dist_end < _z_R_eff

        _target_w = _target_pilot.spot_size_mm
        _target_w0 = _target_pilot.waist_radius_mm
        # 束腰处的阵列尺寸（近场保持不变，远场按 FFT 缩放）
        _waist_L_mm = _L_mm  # 近场默认保持
        _target_beam_px = (2.0 * _target_w) / _dx_mm if _dx_mm > 0 else 0.0

        # 传播类型
        if _start_inside and _end_inside:
            _prop_type = "内到内的传播"
        elif _start_inside and not _end_inside:
            _prop_type = "内到外的传播"
        elif not _start_inside and _end_inside:
            _prop_type = "外到内的传播"
        else:
            _prop_type = "外到外的传播"

        _fmt = lambda v: f"{v:.4E}"
        _fmt5 = lambda v: f"{v:.5E}"

        print(f"\n光束传输的距离: {_fmt5(abs(distance_mm))}")
        print()
        print(f"初始增量X, Y的大小: {_fmt5(_dx_mm)} {_fmt5(_dx_mm)}")
        print()
        print(f"初始阵列X, Y的大小: {_fmt5(_L_mm)} {_fmt5(_L_mm)}")
        print()
        print(f"初始Pilot光束束腰     x, y: {_fmt5(_w0)} {_fmt5(_w0)}")
        print()
        print(f"初始Pilot光束位置x, y: {_fmt5(abs(_z_w0))} {_fmt5(abs(_z_w0))}")
        print()
        print(f"初始Pilot光束瑞利范围x, y: {_fmt5(_z_R)} {_fmt5(_z_R)}")
        print()
        print(f"X-瑞利范围{'以内' if _start_inside else '以外'}.")
        print()
        print(f"Y-瑞利范围{'以内' if _start_inside else '以外'}.")
        print()
        print(f"初始Pilot光束的尺寸x, y: {_fmt5(_w_spot)} {_fmt5(_w_spot)}")
        print()
        print(f"初始Pilot光束X,Y像素: {_beam_px:.2f} {_beam_px:.2f}")
        print()
        print(f"内部透射度    : {1.0:.6f}")
        print()
        print(f"使用{_prop_type}.")
        print()
        print(f"束腰阵列X, Y的尺寸: {_fmt5(_waist_L_mm)} {_fmt5(_waist_L_mm)}")
        print()
        print(f"Pilot光束大小x, y: {_fmt5(_target_w)} {_fmt5(_target_w)}")
        print()
        print(f"PilotX, Y 的像素: {_target_beam_px:.2f} {_target_beam_px:.2f}")
        print()
    
    # Execution
    orig_z_ray = getattr(wfo, "z_Rayleigh", None)
    orig_z_w0 = getattr(wfo, "z_w0", None)

    if use_asm and orig_z_ray is not None:
        # CRITICAL FIX: Handle Entry Transition (SPHERI -> PLANAR)
        # If we are effectively in Far Field (SPHERI), we must add curvature back
        # to the array because PTP (Near Field Algo) expects Total Phase.
        current_ref = getattr(wfo, "reference_surface", "PLANAR")
        if current_ref == "SPHERI":
            current_z = float(getattr(wfo, "z", 0.0))
            # Note: wfo.z_w0 is still the original valid one here
            R = current_z - wfo.z_w0
            if abs(R) > 1e-12:
                proper.prop_qphase(wfo, R) # Add curvature (Convert Residue -> Total)

        # Force PROPER to use PTP (ASM-like) by making the beam "near-field".
        # This keeps the reference surface as PLANAR during the propagation
        wfo.z_Rayleigh = 1e20
        wfo.z_w0 = wfo.z
        if hasattr(wfo, "beam_type_old"):
            wfo.beam_type_old = "INSIDE_"
        if hasattr(wfo, "reference_surface"):
            wfo.reference_surface = "PLANAR"

    try:
        proper.prop_propagate(wfo, distance_m / n)
    finally:
        if use_asm and orig_z_ray is not None:
            # Restore true beam parameters
            wfo.z_Rayleigh = orig_z_ray
            wfo.z_w0 = orig_z_w0
            
            # CRITICAL FIX: Ensure physical state consistency after forced ASM (PTP).
            # PTP leaves wfo in "PLANAR" state with "INSIDE_" beam type.
            # If we are actually in the far field, we MUST synchronize this to "SPHERI"/"OUTSIDE"
            # so the NEXT propagation step starts with the correct assumptions.
            
            # Reuse logic from _sync_proper_gaussian_params to handle
            # potential PLANAR -> SPHERI transition (removing curvature from wfarr)
            # We don't need to pass pilot_beam here as we have wfo's own parameters restored.
            
            import proper
            rayleigh_factor = proper.rayleigh_factor
            current_z = float(getattr(wfo, "z", 0.0))
            dist_from_waist = abs(wfo.z_w0 - current_z)
            
            if dist_from_waist < rayleigh_factor * wfo.z_Rayleigh:
                target_ref = "PLANAR"
                target_beam_type = "INSIDE_"
            else:
                target_ref = "SPHERI"
                target_beam_type = "OUTSIDE"
            

            
            current_ref = getattr(wfo, "reference_surface", "PLANAR")
            
            if current_ref == "PLANAR" and target_ref == "SPHERI":
                # Forced ASM (PTP) finished, result is PLANAR.
                # But we are in Far Field. Must convert to SPHERI.
                R = current_z - wfo.z_w0
                if abs(R) > 1e-12:
                    proper.prop_qphase(wfo, -R)
            
            # Update markers
            wfo.reference_surface = target_ref
            wfo.beam_type_old = target_beam_type


    sampling_m = proper.prop_get_sampling(wfo)
    if hasattr(wfo, "dx"):
        try:
            wfo.dx = sampling_m
        except AttributeError:
            pass
            
    return GridSampling.from_proper(wfo), algo_name



def _compute_proper_reference_phase(
    wfo: Any,
    grid_sampling: GridSampling,
) -> NDArray[np.floating]:
    """Return PROPER's reference phase by applying PROPER's own q-phase routine."""
    import proper

    if getattr(wfo, "reference_surface", "PLANAR") == "PLANAR":
        return np.zeros((grid_sampling.grid_size, grid_sampling.grid_size))
    r_ref_m = wfo.z - wfo.z_w0
    if abs(r_ref_m) < 1e-12:
        return np.zeros((grid_sampling.grid_size, grid_sampling.grid_size))
    ref_wfo = copy.copy(wfo)
    ref_wfo.wfarr = np.ones_like(wfo.wfarr, dtype=np.complex128)
    with contextlib.redirect_stdout(io.StringIO()):
        proper.prop_qphase(ref_wfo, r_ref_m)
    return proper.prop_shift_center(np.angle(ref_wfo.wfarr))



def proper_to_amplitude_phase(
    wfo: Any,
    grid_sampling: GridSampling,
    pilot_beam_params: Optional[PilotBeamParams] = None,
    auto_unwarp_at_incident: bool = False,
    trace_context: Optional[dict[str, Any]] = None,
    auto_unwarp_debug_dir: Optional[str | Path] = None,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    import proper

    amplitude = proper.prop_get_amplitude(wfo)
    residual_phase = proper.prop_get_phase(wfo)
    _check_residual_phase_range(
        residual_phase,
        amplitude,
        grid_sampling=grid_sampling,
        trace_context=trace_context,
    )
    ref_phase = _compute_proper_reference_phase(wfo, grid_sampling)
    wrapped_phase = residual_phase + ref_phase
    if pilot_beam_params is None:
        return amplitude, wrapped_phase
    pilot_phase = pilot_beam_params.compute_phase_grid(
        grid_sampling.grid_size, grid_sampling.physical_size_mm
    )
    residual_wrapped = wrapped_phase - pilot_phase
    residual_folded = np.angle(np.exp(1j * residual_wrapped))

    if auto_unwarp_at_incident and _should_auto_unwarp_at_incident(trace_context):
        is_folded, pv_before, min_before, max_before = _detect_wrapped_residual_near_one_wave(
            residual_folded, amplitude, threshold_ratio=0.10
        )
        if is_folded:
            mask = _compute_intensity_mask(amplitude, threshold_ratio=0.10)
            jump_before = _max_phase_jump(residual_folded)
            jump_before_masked = _phase_jump_stats(residual_folded, mask)
            residual_unwrapped = _unwrap_phase_2d(residual_wrapped)
            # Remove global 2π offset ambiguity to stay close to wrapped residual.
            if np.any(mask):
                delta_cycles = np.median((residual_unwrapped[mask] - residual_folded[mask]) / (2.0 * np.pi))
            else:
                delta_cycles = np.median((residual_unwrapped - residual_folded) / (2.0 * np.pi))
            residual_unwrapped = residual_unwrapped - (2.0 * np.pi * np.round(delta_cycles))
            jump_after = _max_phase_jump(residual_unwrapped)
            jump_after_masked = _phase_jump_stats(residual_unwrapped, mask)
            phase_unwrapped = pilot_phase + residual_unwrapped
            residual_after = phase_unwrapped - pilot_phase
            if np.any(mask):
                pv_after = float(np.ptp((residual_after[mask]) / (2.0 * np.pi)))
                rms_after = float(np.sqrt(np.mean(((residual_after[mask]) / (2.0 * np.pi)) ** 2)))
            else:
                pv_after = float("nan")
                rms_after = float("nan")
            context_line = format_trace_context(trace_context)
            unwrap_accepted = (
                jump_after_masked["count"] > 0
                and jump_after_masked["p999"] <= max(jump_before_masked["p999"], np.pi)
                and jump_after_masked["max"] <= max(jump_before_masked["max"], 2.0 * np.pi)
            )
            debug_path = _auto_unwarp_debug_path(
                trace_context,
                auto_unwarp_debug_dir,
                accepted=unwrap_accepted,
            )
            _plot_auto_unwarp_debug(
                amplitude=amplitude,
                residual_folded=residual_folded,
                residual_unwrapped=residual_unwrapped,
                mask=mask,
                stats_before=jump_before_masked,
                stats_after=jump_after_masked,
                accepted=unwrap_accepted,
                save_path=debug_path,
                trace_context=trace_context,
            )
            if unwrap_accepted:
                print(
                    "[POP][AutoUnwarp@Incident] 检测到疑似折叠残差，接受 residual unwrap: "
                    f"PV_before={pv_before:.4f} waves (min={min_before:.4f}, max={max_before:.4f}) -> "
                    f"PV_after={pv_after:.4f} waves, RMS_after={rms_after:.4f} waves; "
                    f"jump_before={jump_before:.4f} rad, jump_after={jump_after:.4f} rad; "
                    f"masked_p999={jump_before_masked['p999']:.4f}->{jump_after_masked['p999']:.4f} rad, "
                    f"masked_max={jump_before_masked['max']:.4f}->{jump_after_masked['max']:.4f} rad"
                )
                if context_line:
                    print(f"[POP][AutoUnwarp@Incident] 步骤: {context_line}")
                return amplitude, phase_unwrapped
            print(
                "[POP][AutoUnwarp@Incident] 检测到疑似折叠残差，但 unwrap 未改善连续性，回退原策略: "
                f"jump_before={jump_before:.4f} rad, jump_after={jump_after:.4f} rad; "
                f"masked_p999={jump_before_masked['p999']:.4f}->{jump_after_masked['p999']:.4f} rad, "
                f"masked_max={jump_before_masked['max']:.4f}->{jump_after_masked['max']:.4f} rad"
            )
            if context_line:
                print(f"[POP][AutoUnwarp@Incident] 步骤: {context_line}")

    # 默认保持历史行为（残差限制到 [-pi, pi]）
    return amplitude, pilot_phase + residual_folded


def _refit_pilot_from_grid(
    amplitude: NDArray[np.floating],
    phase: NDArray[np.floating],
    grid_sampling: GridSampling,
    pilot: PilotBeamParams,
    pv_threshold_waves: float,
    target_surface_index: Optional[int] = None,
    debug_resample_dir: Optional[str | Path] = None,
) -> Optional[PilotBeamParams]:
    """Refit pilot beam from grid-based amplitude/phase when residual PV is too large.

    Returns (new_pilot, is_accepted).
    If refit fails or is rejected, is_accepted is False.
    new_pilot may be None if calculation completely fails.
    """
    from pop.analysis import _fit_phase_curvature, compute_intensity_moments

    amp = np.asarray(amplitude, dtype=float)
    max_amp = float(np.max(amp)) if amp.size else 0.0
    if max_amp <= 0:
        return None, False
    mask = amp > 0.01 * max_amp
    if np.count_nonzero(mask) < 10:
        return None, False

    grid_size = grid_sampling.grid_size
    physical_mm = grid_sampling.physical_size_mm
    pilot_phase = pilot.compute_phase_grid(grid_size, physical_mm)
    residual = np.angle(np.exp(1j * (phase - pilot_phase)))
    residual_waves = residual / (2.0 * np.pi)
    pv = float(np.ptp(residual_waves[mask]))
    rms = float(np.std(residual_waves[mask]))
    print(
        f"[POP][Grid Refit 检查] Surface {target_surface_index} (exit): "
        f"残差 PV={pv:.4f} waves, RMS={rms:.4f} waves "
        f"(阈值={pv_threshold_waves:.4f} waves)"
    )
    
    if pv <= pv_threshold_waves:
        print(
            f"[POP][Grid Refit 跳过] Surface {target_surface_index}: "
            f"PV {pv:.4f} ≤ 阈值 {pv_threshold_waves:.4f} waves"
        )
        return None, False

    # Fit curvature - Method 1: Gradient Local
    n = pilot.current_refractive_index
    wavelength_mm = pilot.wavelength_um * 1e-3
    coords = (np.arange(grid_size) - grid_size // 2) * grid_sampling.sampling_mm
    x_grid, y_grid = np.meshgrid(coords, coords)
    intensity = amp ** 2
    total_i = float(np.sum(intensity[mask]))
    if total_i <= 0:
        return None, False
    cx = float(np.sum(x_grid[mask] * intensity[mask]) / total_i)
    cy = float(np.sum(y_grid[mask] * intensity[mask]) / total_i)

    r_fit_grad = _fit_phase_curvature(
        phase, x_grid, y_grid, intensity, cx, cy, wavelength_mm, n,
        enable_global_scan=True
    )
    
    # Fit curvature - Method 2: Unwrapped Phase Global (New)
    from pop.analysis import _fit_phase_curvature_unwrapped
    r_fit_unwrap = _fit_phase_curvature_unwrapped(
        phase, x_grid, y_grid, intensity, wavelength_mm, n
    )
    
    # Decide which R to use?
    # Let's compare them by generated residual
    candidates = []
    if np.isfinite(r_fit_grad):
        candidates.append(("Gradient", r_fit_grad))
    if np.isfinite(r_fit_unwrap):
        candidates.append(("Unwrap", r_fit_unwrap))
        
    if not candidates:
        return None, False

    # Helper to evaluate candidate
    def evaluate_candidate_R(r_test, w_fit_val_ignored):
        # NOTE: w_fit_val_ignored is the intensity-fitted size, which we use for logging only.
        # For the physics of propagation consistency, we MUST preserve the pilot's own w.
        w_used = pilot.spot_size_mm
        
        if np.isinf(r_test):
            inv_q_real = 0.0
        else:
            inv_q_real = 1.0 / r_test
        inv_q_imag = -wavelength_mm / (np.pi * n * w_used ** 2)
        inv_q = inv_q_real + 1j * inv_q_imag
        if abs(inv_q) < 1e-30: return None, np.inf
        q_new = 1.0 / inv_q
        test_pilot = PilotBeamParams.from_q_parameter(q_new, pilot.wavelength_um, n)
        test_phase = test_pilot.compute_phase_grid(grid_size, physical_mm)
        # Residual RMS
        test_res = np.angle(np.exp(1j * (phase - test_phase)))
        test_rms = np.std(test_res[mask])
        return test_pilot, test_rms

    # Fit spot size from intensity moments (Common)
    var_x = float(np.sum((x_grid[mask] - cx) ** 2 * intensity[mask]) / total_i)
    var_y = float(np.sum((y_grid[mask] - cy) ** 2 * intensity[mask]) / total_i)
    sigma = np.sqrt(max(0.5 * (var_x + var_y), 0.0))
    w_fit = 2.0 * sigma
    if w_fit < 1e-12:
        return None, False

    best_pilot = None
    best_rms = np.inf
    best_method = ""
    
    log_msg = f"[POP][Refit Comparison] Surface {target_surface_index}: "
    
    for name, r_val in candidates:
        cand_pilot, cand_rms = evaluate_candidate_R(r_val, w_fit)
        if cand_pilot is None: continue
        log_msg += f"[{name}: R={r_val:.2f}mm, RMS={cand_rms:.4f} rad] "
        if cand_rms < best_rms:
            best_rms = cand_rms
            best_pilot = cand_pilot
            best_method = name
            
    print(log_msg)
    
    if best_pilot is None:
        return None, False
        
    new_pilot = best_pilot

    # Verify the refit actually reduced the PV
    # Use best_pilot
    new_pilot_phase = new_pilot.compute_phase_grid(grid_size, physical_mm)
    new_residual = np.angle(np.exp(1j * (phase - new_pilot_phase)))
    new_pv = float(np.ptp(new_residual[mask] / (2.0 * np.pi)))
    
    # [DEBUG] 保存残差数据用于 Zernike 分析（通过 debug_resample_dir 控制输出路径）
    if debug_resample_dir is not None:
        import os
        try:
            out_dir = str(debug_resample_dir)
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, f"s{target_surface_index}_zernike_data.npz")
            np.savez(save_path, 
                     residual_before=residual, 
                     residual_after=new_residual, 
                     mask=mask, 
                     beam_radius=w_fit/2.0,
                     grid_size=grid_size,
                     physical_mm=physical_mm)
            print(f"[POP][DEBUG] Saved S{target_surface_index} residual data for Zernike analysis to {save_path}")
        except Exception as e:
            print(f"[POP][DEBUG] Failed to save Zernike data: {e}")

    # 始终接受 refit 结果（不再根据 PV 改善与否拒绝）
    old_R = pilot.curvature_radius_mm
    old_w = pilot.spot_size_mm
    new_rms = float(np.std(new_residual[mask] / (2.0 * np.pi)))
    pv_improved = "改善" if new_pv < pv else "未改善"
    print(
        f"[POP][Grid Refit 接受] Surface {target_surface_index} (方法: {best_method}): "
        f"PV={pv:.4f} → {new_pv:.4f} waves ({pv_improved}), "
        f"RMS={rms:.4f} → {new_rms:.4f} waves | "
        f"R: {old_R:.2f} → {new_pilot.curvature_radius_mm:.2f} mm, "
        f"w: {old_w:.4f} → {new_pilot.spot_size_mm:.4f} mm (kept), "
        f"w0: {pilot.waist_radius_mm:.4f} → {new_pilot.waist_radius_mm:.4f} mm"
    )

    return new_pilot, True


def _resample_wavefront(
    wfo: Any,
    grid_sampling: GridSampling,
    new_pilot: PilotBeamParams,
    amplitude: NDArray[np.floating],
    phase: NDArray[np.floating],
    resample_min_beam_pixels: int,
    resample_beam_pixels_target: Optional[int] = None,
    auto_unwarp_at_incident: bool = False,
    trace_context: Optional[dict[str, Any]] = None,
    auto_unwarp_debug_dir: Optional[str | Path] = None,
) -> Tuple[GridSampling, NDArray[np.floating], NDArray[np.floating], bool, Optional[dict[str, Any]]]:
    """Check if beam is undersampled and resample if needed.

    Returns (grid_sampling, amplitude, phase, did_resample, resample_debug_info).
    resample_debug_info 包含 resample 前后的原始数据，用于 debug 绘图。
    """
    import proper

    beam_diameter_mm = 2.0 * float(new_pilot.spot_size_mm)
    if not np.isfinite(beam_diameter_mm) or beam_diameter_mm <= 0:
        return grid_sampling, amplitude, phase, False, None

    pixels_across = beam_diameter_mm / grid_sampling.sampling_mm
    if pixels_across >= resample_min_beam_pixels:
        return grid_sampling, amplitude, phase, False, None

    # Resolve target: use beam_ratio-based default if not specified
    effective_target = resample_beam_pixels_target
    if effective_target is None:
        effective_target = int(grid_sampling.grid_size * grid_sampling.beam_ratio)

    # Compute magnification needed
    mag = float(effective_target) / max(pixels_across, 1e-6)
    if mag <= 1.0:
        return grid_sampling, amplitude, phase, False, None

    grid_size = grid_sampling.grid_size
    old_physical_mm = grid_sampling.physical_size_mm

    # 保存 resample 前的数据用于 debug
    amplitude_before = amplitude.copy()
    phase_before = phase.copy()
    grid_sampling_before = GridSampling(
        grid_size=grid_sampling.grid_size,
        physical_size_mm=grid_sampling.physical_size_mm,
        sampling_mm=grid_sampling.sampling_mm,
        beam_ratio=grid_sampling.beam_ratio,
    )

    # Resample the complex wavefront array in PROPER
    centered = proper.prop_shift_center(wfo.wfarr)
    new_centered = proper.prop_magnify(centered, mag, grid_size, CONSERVE=True)
    wfo.wfarr = proper.prop_shift_center(new_centered)

    # Update sampling
    new_dx_m = wfo.dx / mag
    wfo.dx = new_dx_m

    # Create new GridSampling from updated PROPER state
    new_grid_sampling = GridSampling.from_proper(wfo)

    # Re-extract amplitude and phase with the new grid
    new_amplitude, new_phase = proper_to_amplitude_phase(
        wfo,
        new_grid_sampling,
        new_pilot,
        auto_unwarp_at_incident=auto_unwarp_at_incident,
        trace_context=trace_context,
        auto_unwarp_debug_dir=auto_unwarp_debug_dir,
    )

    new_pixels = beam_diameter_mm / new_grid_sampling.sampling_mm
    print(
        f"[POP][Resample] beam_pixels: {pixels_across:.1f} -> {new_pixels:.1f}, "
        f"physical_size: {old_physical_mm:.4f} -> {new_grid_sampling.physical_size_mm:.4f} mm, "
        f"sampling: {grid_sampling.sampling_mm:.6f} -> {new_grid_sampling.sampling_mm:.6f} mm/px, "
        f"mag={mag:.2f}"
    )

    # 构建 debug 信息
    resample_debug_info = {
        "amplitude_before": amplitude_before,
        "phase_before": phase_before,
        "grid_sampling_before": grid_sampling_before,
        "amplitude_after": new_amplitude,
        "phase_after": new_phase,
        "grid_sampling_after": new_grid_sampling,
        "mag": mag,
        "beam_pixels_before": pixels_across,
        "beam_pixels_after": new_pixels,
    }

    return new_grid_sampling, new_amplitude, new_phase, True, resample_debug_info


def propagate_state(
    state: PropagationState,
    target_axis_state: OpticalAxisState,
    target_surface_index: int,
    target_position: str = "entrance",
    trace_context: Optional[dict[str, Any]] = None,
    force_asm: Optional[bool] = None,
    auto_asm: bool = True,
    free_space_mode: str = "native_proper",
    auto_resample: bool = False,
    resample_min_beam_pixels: int = 10,
    resample_beam_pixels_target: Optional[int] = None,
    pilot_refit_surface_indices: Optional[list[int]] = None,
    pilot_refit_pv_threshold_waves: float = 0.5,
    auto_unwarp_at_incident: bool = False,
    auto_unwarp_surface_indices: Optional[list[int]] = None,
    debug_resample_dir: Optional[str | Path] = None,
) -> PropagationState:
    if state.proper_wfo is None:
        raise ValueError("PropagationState.proper_wfo is required for free-space propagation")

    distance_mm = _compute_signed_distance(state.optical_axis_state, target_axis_state)
    enable_auto_unwarp_here = _enable_auto_unwarp_for_current_step(
        auto_unwarp_at_incident=auto_unwarp_at_incident,
        target_surface_index=target_surface_index,
        target_position=target_position,
        auto_unwarp_surface_indices=auto_unwarp_surface_indices,
    )

    # If distance is negligible, return state as is (with update metadata)
    # BUT we must still check for Refit/Resample if it's an Exit surface
    if abs(distance_mm) < 1e-12:
        intermediate_state = PropagationState(
            surface_index=target_surface_index,
            position=target_position,
            amplitude=state.amplitude.copy(),
            phase=state.phase.copy(),
            pilot_beam_params=state.pilot_beam_params,
            optical_axis_state=target_axis_state,
            grid_sampling=state.grid_sampling,
            proper_wfo=state.proper_wfo,
            force_asm=state.force_asm, # Preserve
            propagation_algorithm="N/A (Zero Dist)",
        )
        return _apply_refit_and_resample(
            intermediate_state,
            target_surface_index,
            target_position,
            pilot_refit_surface_indices,
            pilot_refit_pv_threshold_waves,
            enable_auto_unwarp_here,
            auto_resample,
            resample_min_beam_pixels,
            resample_beam_pixels_target,
            trace_context=trace_context,
            debug_resample_dir=debug_resample_dir,
        )

    # Use state.force_asm if not overridden by argument
    effective_force_asm = force_asm if force_asm is not None else state.force_asm

    grid_sampling, algo_used = propagate_free_space(
        state.proper_wfo,
        distance_mm,
        state.pilot_beam_params.current_refractive_index,
        force_asm=effective_force_asm,
        auto_asm=auto_asm,
        pilot_beam=state.pilot_beam_params,
        free_space_mode=free_space_mode,
    )

    new_pilot = _propagate_pilot_with_compensated_q(state.pilot_beam_params, distance_mm)
    _sync_proper_gaussian_params(state.proper_wfo, new_pilot)
    amplitude, phase = proper_to_amplitude_phase(
        state.proper_wfo,
        grid_sampling,
        new_pilot,
        auto_unwarp_at_incident=enable_auto_unwarp_here,
        trace_context=trace_context,
        auto_unwarp_debug_dir=debug_resample_dir,
    )

    new_force_asm = effective_force_asm if effective_force_asm is not None else state.force_asm

    final_state = PropagationState(
        surface_index=target_surface_index,
        position=target_position,
        amplitude=amplitude,
        phase=phase,
        pilot_beam_params=new_pilot,
        optical_axis_state=target_axis_state,
        grid_sampling=grid_sampling,
        proper_wfo=state.proper_wfo,
        force_asm=new_force_asm,
        propagation_algorithm=algo_used,
    )

    return _apply_refit_and_resample(
        final_state,
        target_surface_index,
        target_position,
        pilot_refit_surface_indices,
        pilot_refit_pv_threshold_waves,
        enable_auto_unwarp_here,
        auto_resample,
        resample_min_beam_pixels,
        resample_beam_pixels_target,
        trace_context=trace_context,
        debug_resample_dir=debug_resample_dir,
    )


def _apply_refit_and_resample(
    state: PropagationState,
    target_surface_index: int,
    target_position: str,
    pilot_refit_surface_indices: Optional[list[int]],
    pilot_refit_pv_threshold_waves: float,
    auto_unwarp_at_incident: bool,
    auto_resample: bool,
    resample_min_beam_pixels: int,
    resample_beam_pixels_target: Optional[int] = None,
    trace_context: Optional[dict[str, Any]] = None,
    debug_resample_dir: Optional[str | Path] = None,
) -> PropagationState:
    """Helper to apply Pilot Refit and Auto-Resample on a PropagationState."""

    amplitude = state.amplitude
    phase = state.phase
    grid_sampling = state.grid_sampling
    new_pilot = state.pilot_beam_params
    _refit_messages: list[str] = []

    # --- Pilot Refit (grid-based, for non-element surfaces at exit position only) ---
    if (
        pilot_refit_surface_indices is not None
        and target_surface_index in pilot_refit_surface_indices
        and target_position == "exit"
    ):
        refitted, is_accepted = _refit_pilot_from_grid(
            amplitude, phase, grid_sampling, new_pilot, pilot_refit_pv_threshold_waves,
            target_surface_index=target_surface_index,
            debug_resample_dir=debug_resample_dir,
        )

        # Plotting Logic: Plot if ANY refit result was calculated (Accepted or Rejected)
        if refitted is not None:
            old_R = new_pilot.curvature_radius_mm
            old_w = new_pilot.spot_size_mm
            old_w0 = new_pilot.waist_radius_mm
            old_z_waist = new_pilot.waist_position_mm

            # 1. Capture state BEFORE update
            pilot_before = new_pilot
            pilot_phase_before = pilot_before.compute_phase_grid(
                grid_sampling.grid_size, grid_sampling.physical_size_mm
            )
            phase_before = phase # Total physical phase

            # 2. Determine "After" state for plotting
            # Even if rejected, we want to show what the refit WOULD have looked like.
            pilot_after_sim = refitted
            pilot_phase_after = pilot_after_sim.compute_phase_grid(
                grid_sampling.grid_size, grid_sampling.physical_size_mm
            )

            # For the plot's "After" panel, we need the residual phase *relative to the new pilot*.
            # The plotting function expects 'amplitude' and 'phase' (total).
            # It computes residual = phase - pilot_phase.
            # So passing standard 'phase' and 'pilot_after_sim' works for visualization.

            # 3. Plotting
            from pop.visualization import plot_refit_diagnostics
            status_tag = "ACCEPTED" if is_accepted else "REJECTED"
            save_path = (
                (Path(debug_resample_dir) if debug_resample_dir is not None else Path("tests/debug_output_refit"))
                / f"surface_{target_surface_index:02d}_grid_refit_{status_tag}.png"
            )

            print(f"[POP][Grid Refit] 生成诊断图 ({status_tag}): {save_path}")
            plot_refit_diagnostics(
                phase_before=phase_before,
                phase_after=phase, # Logic in plotter uses this against new_pilot
                pilot_phase_before=pilot_phase_before,
                pilot_phase_after=pilot_phase_after,
                amplitude=amplitude,
                grid_sampling=grid_sampling,
                old_pilot=pilot_before,
                new_pilot=pilot_after_sim,
                surface_index=target_surface_index,
                save_path=save_path,
                show=False
            )

            # 4. Apply Update ONLY IF ACCEPTED
            if is_accepted:
                # --- Update PROPER's wfarr ---
                # The total physical wavefront (amplitude * exp(i*phase)) must be invariant.
                # PROPER stores: wfarr = amplitude * exp(i * (Total_Phase - Reference_Phase))
                # Refit changes pilot parameters -> Reference_Phase changes.
                # So we must update wfarr:
                # wfarr_new = wfarr_old * exp(i * (Reference_Phase_Old - Reference_Phase_New))
                #           = amplitude * exp(i * (Total_Phase - Reference_Phase_New))

                new_pilot = refitted
                # 更新 PROPER 高斯参数和 reference_surface/beam_type_old 元数据，
                # 但跳过 prop_qphase 对 wfarr 的修改——因为下面会从物理真值重建 wfarr。
                _sync_proper_gaussian_params(
                    state.proper_wfo, new_pilot,
                    update_reference_surface=True,
                    skip_wfarr_transition=True,
                )

                # Calculate NEW Reference Phase (based on updated w0, z_R, R)
                # 此时 reference_surface 已正确反映新 pilot 的近场/远场状态，
                # 所以 _compute_proper_reference_phase 会返回正确的参考相位。
                ref_phase_new = _compute_proper_reference_phase(state.proper_wfo, grid_sampling)

                # Calculate New Residual: Total_Phase - New_Reference
                # Note: 'phase' variable holds Total Unwrapped Physical Phase
                new_residual_phase = phase - ref_phase_new

                # Update wfarr
                # PROPER expects complex field.
                cwf_new = amplitude * np.exp(1j * new_residual_phase)

                # Shift if necessary (assuming PROPER uses FFT layout internally but our arrays are centered)
                # But proper_to_amplitude_phase returns centered arrays.
                import proper
                wfarr_shifted = proper.prop_shift_center(cwf_new)
                state.proper_wfo.wfarr[:, :] = wfarr_shifted[:, :]

                # Re-extract with updated pilot (just to verify)
                amplitude, phase = proper_to_amplitude_phase(
                    state.proper_wfo,
                    grid_sampling,
                    new_pilot,
                    auto_unwarp_at_incident=auto_unwarp_at_incident,
                    trace_context=trace_context,
                    auto_unwarp_debug_dir=debug_resample_dir,
                )

                _refit_messages.append(
                    f"[Grid Refit] R: {old_R:.2f} → {new_pilot.curvature_radius_mm:.2f} mm, "
                    f"w: {old_w:.4f} → {new_pilot.spot_size_mm:.4f} mm"
                )
                _refit_messages.append(
                    f"[Grid Refit] w0: {old_w0:.4f} → {new_pilot.waist_radius_mm:.4f} mm, "
                    f"z_waist: {old_z_waist:.4f} → {new_pilot.waist_position_mm:.4f} mm"
                )

    # --- Auto Resample ---
    _resample_messages: list[str] = []
    if auto_resample:
        old_sampling_mm = grid_sampling.sampling_mm
        old_physical_mm = grid_sampling.physical_size_mm
        grid_sampling, amplitude, phase, did_resample, resample_debug_info = _resample_wavefront(
            wfo=state.proper_wfo,
            grid_sampling=grid_sampling,
            new_pilot=new_pilot,
            amplitude=amplitude,
            phase=phase,
            resample_min_beam_pixels=resample_min_beam_pixels,
            resample_beam_pixels_target=resample_beam_pixels_target,
            auto_unwarp_at_incident=auto_unwarp_at_incident,
            trace_context=trace_context,
            auto_unwarp_debug_dir=debug_resample_dir,
        )
        if did_resample:
            beam_diam_mm = 2.0 * float(new_pilot.spot_size_mm)
            old_px = beam_diam_mm / old_sampling_mm if old_sampling_mm > 0 else 0
            new_px = beam_diam_mm / grid_sampling.sampling_mm if grid_sampling.sampling_mm > 0 else 0
            _resample_messages.append(
                f"[Resample] beam_pixels: {old_px:.1f} -> {new_px:.1f}"
            )
            _resample_messages.append(
                f"[Resample] physical_size: {old_physical_mm:.4f} -> {grid_sampling.physical_size_mm:.4f} mm, "
                f"sampling: {old_sampling_mm:.6f} -> {grid_sampling.sampling_mm:.6f} mm/px"
            )

            # --- Resample Debug 绘图 ---
            if resample_debug_info is not None and debug_resample_dir is not None:
                try:
                    from pop.visualization import plot_resample_debug

                    save_dir = Path(debug_resample_dir)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / f"surface_{target_surface_index:02d}_{target_position}_resample_debug.png"

                    plot_resample_debug(
                        phase_before=resample_debug_info["phase_before"],
                        phase_after=resample_debug_info["phase_after"],
                        amplitude_before=resample_debug_info["amplitude_before"],
                        amplitude_after=resample_debug_info["amplitude_after"],
                        grid_sampling_before=resample_debug_info["grid_sampling_before"],
                        grid_sampling_after=resample_debug_info["grid_sampling_after"],
                        pilot_beam=new_pilot,
                        surface_index=target_surface_index,
                        position=target_position,
                        mag=resample_debug_info["mag"],
                        beam_pixels_before=resample_debug_info["beam_pixels_before"],
                        beam_pixels_after=resample_debug_info["beam_pixels_after"],
                        save_path=save_path,
                        show=False,
                    )
                except Exception as exc:
                    print(f"[POP][Resample Debug] 绘图失败: {exc}")

    # Return updated state
    return PropagationState(
        surface_index=state.surface_index,
        position=state.position,
        amplitude=amplitude,
        phase=phase,
        pilot_beam_params=new_pilot,
        optical_axis_state=state.optical_axis_state,
        grid_sampling=grid_sampling,
        proper_wfo=state.proper_wfo,
        force_asm=state.force_asm,
        propagation_algorithm=state.propagation_algorithm,
        messages=(state.messages or []) + _refit_messages + _resample_messages,
    )
