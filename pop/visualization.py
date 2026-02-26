"""Visualization helpers for POP results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


from pop.analysis import (
    compute_gaussian_fit_and_m2,
    compute_log_intensity,
    compute_metrics,
    compute_state_moments,
    compute_valid_mask,
)
from pop.core import OpticalAxisState, PropagationState, GridSampling, PilotBeamParams
from pop.coordinates.transforms import rotation_matrix_to_euler
from pop.result import SurfaceDebugInfo, SurfaceRecord

from matplotlib import pyplot as plt
import matplotlib.cm
from numpy.typing import NDArray




def _configure_matplotlib_fonts(plt) -> None:
    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
    ]
    current = plt.rcParams.get("font.sans-serif", [])
    if isinstance(current, str):
        current = [current]
    plt.rcParams["font.sans-serif"] = [f for f in preferred if f not in current] + list(current)
    plt.rcParams["axes.unicode_minus"] = False


def _format_value(value: float, precision: int = 4) -> str:
    if np.isinf(value):
        return "inf"
    return f"{value:.{precision}f}"


def _format_surface_info(surface: Any, position: str) -> list[str]:
    name = getattr(surface, "comment", "") or ""
    surface_type = getattr(surface, "surface_type", "") or "standard"
    material = getattr(surface, "material", "") or "air"
    is_mirror = bool(getattr(surface, "is_mirror", False))
    semi_aperture = float(getattr(surface, "semi_aperture", 0.0) or 0.0)
    radius = float(getattr(surface, "radius", np.inf))
    conic = float(getattr(surface, "conic", 0.0))
    radius_x = float(getattr(surface, "radius_x", np.inf))
    conic_x = float(getattr(surface, "conic_x", 0.0))
    vertex = np.asarray(getattr(surface, "vertex_position", [0.0, 0.0, 0.0]), dtype=float)
    orientation = np.asarray(getattr(surface, "orientation", np.eye(3)), dtype=float)
    rx, ry, rz = rotation_matrix_to_euler(orientation)
    rx_deg, ry_deg, rz_deg = np.rad2deg([rx, ry, rz])

    header = f"Surface {getattr(surface, 'index', '?')} ({position})"
    if name.strip():
        header += f" - {name.strip()}"

    return [
        header,
        f"Type={surface_type}, Material={material}, Mirror={is_mirror}",
        f"Radius={_format_value(radius)} mm, Conic={conic:.4f}, "
        f"RadiusX={_format_value(radius_x)} mm, ConicX={conic_x:.4f}",
        f"Semi-aperture={semi_aperture:.4f} mm",
        f"Vertex=(x,y,z)=({vertex[0]:.4f}, {vertex[1]:.4f}, {vertex[2]:.4f}) mm",
        f"Orientation Euler(deg)=(rx={rx_deg:.2f}, ry={ry_deg:.2f}, rz={rz_deg:.2f})",
    ]


def _format_axis_info(axis_state: Optional[OpticalAxisState]) -> list[str]:
    if axis_state is None:
        return ["Optical axis: unavailable"]
    pos = np.asarray(axis_state.position, dtype=float)
    direction = np.asarray(axis_state.direction, dtype=float)
    return [
        f"Optical axis pos=(x,y,z)=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) mm",
        f"Optical axis dir=(L,M,N)=({direction[0]:.6f}, {direction[1]:.6f}, {direction[2]:.6f})",
    ]


def _format_grid_info(state: PropagationState, mask_threshold: float = 0.01) -> list[str]:
    grid = state.grid_sampling
    pilot = state.pilot_beam_params
    lines = [
        f"Grid: size={grid.grid_size}, physical_size={grid.physical_size_mm:.4f} mm, "
        f"sampling={grid.sampling_mm:.4f} mm",
        f"Wavelength={pilot.wavelength_um:.4f} um, n={pilot.current_refractive_index:.4f}",
        f"Pilot beam: w={pilot.spot_size_mm:.4f} mm, R={_format_value(pilot.curvature_radius_mm)} mm",
        f"Pilot Params: w0={pilot.waist_radius_mm:.4f} mm, zR={pilot.rayleigh_length_mm:.4f} mm, z_waist={pilot.waist_position_mm:.4f} mm",
    ]

    z_waist = float(pilot.waist_position_mm)
    z_r = float(pilot.rayleigh_length_mm)
    if np.isfinite(z_waist) and np.isfinite(z_r) and z_r > 0:
        region = "near-field" if abs(z_waist) <= z_r else "far-field"
        lines.append(
            f"Pilot regime: {region} (|z_waist|={abs(z_waist):.4f} mm, z_R={z_r:.4f} mm)"
        )
    else:
        lines.append("Pilot regime: unavailable")

    algo = getattr(state, "propagation_algorithm", "N/A")
    lines.append(f"Propagation Algo: {algo}")

    fit_info = compute_gaussian_fit_and_m2(state, mask_threshold=mask_threshold)
    w_avg = fit_info.get("w_avg_mm", np.nan)
    r_fit = fit_info.get("curvature_radius_mm", np.nan)
    w0_fit = fit_info.get("waist_radius_mm", np.nan)
    z0_fit = fit_info.get("waist_position_mm", np.nan)
    if np.isfinite(w_avg):
        lines.append(
            "Sim fit: "
            f"w={_format_value(w_avg)} mm, R={_format_value(r_fit)} mm, "
            f"z_waist={_format_value(z0_fit)} mm, w0={_format_value(w0_fit)} mm"
        )
    else:
        lines.append("Sim fit: unavailable")

    m2_x = fit_info.get("m2_x", np.nan)
    m2_y = fit_info.get("m2_y", np.nan)
    m2_avg = fit_info.get("m2_avg", np.nan)
    pilot_m2 = fit_info.get("m2_pilot", np.nan)
    lines.append(
        "Beam quality: "
        f"M2_sim(x,y,avg)=({_format_value(m2_x, 3)}, "
        f"{_format_value(m2_y, 3)}, {_format_value(m2_avg, 3)}), "
        f"M2_pilot={_format_value(pilot_m2, 3)}"
    )

    lines.append("Coord: local wavefront grid, X/Y in mm, origin at chief ray intersection")

    # Append any pilot-refit or resample messages stored on the state
    state_messages = getattr(state, "messages", None)
    if state_messages:
        for msg in state_messages:
            lines.append(msg)

    return lines


def _add_footer(fig, lines: list[str]) -> None:
    footer = "\n".join(lines)
    n_lines = max(1, footer.count("\n") + 1)
    bottom = min(0.35, 0.08 + 0.018 * (n_lines - 1))
    try:
        fig.subplots_adjust(bottom=bottom)
    except Exception:
        pass
    fig.text(
        0.5,
        0.005,
        footer,
        ha="center",
        va="bottom",
        fontsize=9,
        bbox=dict(facecolor="#f2f2f2", alpha=0.9, pad=5),
    )


def _plot_im(ax, data, title, extent, cmap, vmin=None, vmax=None, cbar_label=None):
    import matplotlib.pyplot as plt

    _configure_matplotlib_fonts(plt)
    im = ax.imshow(
        data,
        extent=extent,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    return im


def plot_wavefront_analysis(
    state: PropagationState,
    surface: Any,
    axis_state: Optional[OpticalAxisState],
    position: str,
    save_path: Optional[str | Path] = None,
    show: bool = True,
    dpi: int = 200,
    plot_set: str = "basic",
    panels: Optional[list[str]] = None,
    max_cols: int = 3,
    mask_threshold: float = 0.01,
    slice_axis: str = "y",
    slice_index: Optional[int] = None,
    cmap_amp: str = "viridis",
    cmap_intensity: str = "hot",
    cmap_phase: str = "RdBu",
    cmap_residual: str = "RdBu_r",
    log_intensity_floor_db: float = -60.0,
    centroid_radius_scale: float = 2.0,
    centroid_marker_size: int = 40,
):
    import matplotlib.pyplot as plt

    _configure_matplotlib_fonts(plt)
    grid = state.grid_sampling
    pilot = state.pilot_beam_params
    sim_amp = state.amplitude
    sim_phase = state.phase

    pilot_amp = pilot.compute_amplitude_grid(grid.grid_size, grid.physical_size_mm)
    pilot_phase = pilot.compute_phase_grid(grid.grid_size, grid.physical_size_mm)

    amp_residual = sim_amp - pilot_amp
    phase_residual = sim_phase - pilot_phase

    metrics = compute_metrics(
        sim_amp,
        amp_residual,
        phase_residual,
        grid,
        mask_threshold=mask_threshold,
    )
    valid_mask = compute_valid_mask(sim_amp, mask_threshold)
    moments = compute_state_moments(state, mask_threshold=mask_threshold)

    half_size = grid.physical_size_mm / 2.0
    extent = [-half_size, half_size, -half_size, half_size]

    amp_residual_masked = amp_residual
    phase_residual_masked = phase_residual
    if np.any(valid_mask):
        amp_residual_masked = np.where(valid_mask, amp_residual, np.nan)
        # Phase residual: Do NOT mask, show full grid as requested for Zernike analysis
        # user wants to see the behavior of the fitted wavefront outside the beam too.
        # phase_residual_masked = np.where(valid_mask, phase_residual, np.nan)
        pass

    vmax_amp_res = np.nanmax(np.abs(amp_residual_masked)) if np.any(np.isfinite(amp_residual_masked)) else None
    
    # Calculate vmin/vmax for phase from the FULL grid to show everything
    vmax_phase_res = np.nanmax(np.abs(phase_residual)) if np.any(np.isfinite(phase_residual)) else None

    sim_phase_waves = sim_phase / (2.0 * np.pi)
    pilot_phase_waves = pilot_phase / (2.0 * np.pi)
    
    # Use unmasked residual for plotting phase
    phase_residual_waves = phase_residual / (2.0 * np.pi)

    if panels is None:
        plot_set = plot_set.lower().strip()
        if plot_set == "basic":
            panels = [
                "amplitude",
                "pilot_amplitude",
                "amplitude_residual",
                "phase",
                "pilot_phase",
                "phase_residual",
            ]
        elif plot_set == "extended":
            panels = [
                "amplitude",
                "intensity",
                "pilot_amplitude",
                "amplitude_residual",
                "amplitude_slice",
                "phase",
                "pilot_phase",
                "phase_residual",
                "phase_slice",
            ]
        elif plot_set == "full":
            panels = [
                "amplitude",
                "intensity",
                "pilot_amplitude",
                "amplitude_residual",
                "amplitude_slice",
                "amplitude_residual_hist",
                "phase",
                "pilot_phase",
                "phase_residual",
                "phase_slice",
                "phase_residual_hist",
            ]
        elif plot_set == "report":
            panels = [
                "intensity",
                "intensity_log",
                "pilot_amplitude",
                "amplitude_residual",
                "phase",
                "pilot_phase",
                "phase_residual",
                "intensity_centroid",
            ]
        else:
            raise ValueError("plot_set must be 'basic', 'extended', 'full', or 'report'")

    panels = list(panels)
    n_panels = len(panels)
    cols = max(1, min(max_cols, n_panels))
    rows = int(np.ceil(n_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.2 * rows))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    coords = (np.arange(grid.grid_size) - grid.grid_size // 2) * grid.sampling_mm
    center_idx = grid.grid_size // 2
    if slice_index is not None:
        center_idx = int(np.clip(slice_index, 0, grid.grid_size - 1))
    slice_axis = slice_axis.lower().strip()
    if slice_axis not in ("x", "y"):
        slice_axis = "y"
    if slice_axis == "y":
        slice_coord = coords
        sim_amp_slice = sim_amp[center_idx, :]
        pilot_amp_slice = pilot_amp[center_idx, :]
        amp_res_slice = amp_residual[center_idx, :]
        sim_phase_slice = sim_phase_waves[center_idx, :]
        pilot_phase_slice = pilot_phase_waves[center_idx, :]
        phase_res_slice = phase_residual_waves[center_idx, :]
        slice_label = f"y={coords[center_idx]:.3f} mm"
    else:
        slice_coord = coords
        sim_amp_slice = sim_amp[:, center_idx]
        pilot_amp_slice = pilot_amp[:, center_idx]
        amp_res_slice = amp_residual[:, center_idx]
        sim_phase_slice = sim_phase_waves[:, center_idx]
        pilot_phase_slice = pilot_phase_waves[:, center_idx]
        phase_res_slice = phase_residual_waves[:, center_idx]
        slice_label = f"x={coords[center_idx]:.3f} mm"

    panel_axes = list(axes.flat)
    for ax, panel in zip(panel_axes, panels):
        if panel == "amplitude":
            _plot_im(ax, sim_amp, "Simulation Amplitude", extent, cmap=cmap_amp)
        elif panel == "intensity":
            _plot_im(ax, sim_amp**2, "Simulation Intensity", extent, cmap=cmap_intensity)
        elif panel == "intensity_log":
            log_intensity = compute_log_intensity(
                sim_amp**2, floor_db=log_intensity_floor_db
            )
            _plot_im(
                ax,
                log_intensity,
                f"Log Intensity (dB, floor={log_intensity_floor_db:.0f})",
                extent,
                cmap="magma",
            )
        elif panel == "intensity_centroid":
            _plot_im(
                ax,
                sim_amp**2,
                "Intensity + Centroid",
                extent,
                cmap=cmap_intensity,
            )
            if moments:
                cx = moments.get("centroid_x", 0.0)
                cy = moments.get("centroid_y", 0.0)
                ax.scatter(
                    [cx],
                    [cy],
                    s=centroid_marker_size,
                    color="cyan",
                    marker="+",
                )
                sigma_x = moments.get("sigma_x", np.nan)
                sigma_y = moments.get("sigma_y", np.nan)
                sigma = max(float(sigma_x), float(sigma_y))
                if np.isfinite(sigma) and sigma > 0:
                    radius = sigma * centroid_radius_scale
                    circle = plt.Circle(
                        (cx, cy),
                        radius,
                        fill=False,
                        edgecolor="cyan",
                        linewidth=1.2,
                        alpha=0.8,
                    )
                    ax.add_patch(circle)
        elif panel == "pilot_amplitude":
            _plot_im(ax, pilot_amp, "Pilot Beam Amplitude", extent, cmap=cmap_amp)
        elif panel == "amplitude_residual":
            _plot_im(
                ax,
                amp_residual_masked,
                f"Amplitude Residual (Sim-Pilot)\nRMS={metrics['amp_rms_percent']:.3f}%, PV={metrics['amp_pv_percent']:.3f}%",
                extent,
                cmap=cmap_residual,
                vmin=-vmax_amp_res if vmax_amp_res else None,
                vmax=vmax_amp_res if vmax_amp_res else None,
            )
        elif panel == "phase":
            _plot_im(ax, sim_phase_waves, "Simulation Phase (waves)", extent, cmap=cmap_phase)
            
            # Add contours for pilot phase range
            # Use valid_mask to determine relevant region for contours range calculation
            # This ensures we pick levels relevant to the actual beam area
            pilot_for_levels = np.where(valid_mask, pilot_phase_waves, np.nan) if np.any(valid_mask) else pilot_phase_waves
            
            p_min = np.nanmin(pilot_for_levels)
            p_max = np.nanmax(pilot_for_levels)
            
            if np.isfinite(p_min) and np.isfinite(p_max) and p_max > p_min:
                ptp = p_max - p_min
                # Levels at 50% and 95% of the P-V range relative to min
                # Using 0.95 to ensure the top contour is visible (1.0 might be a single point or excluded)
                levels = sorted(list(set([p_min + 0.5 * ptp, p_min + 0.95 * ptp])))
                if len(levels) >= 1:
                    try:
                        c = ax.contour(
                            pilot_phase_waves,
                            extent=extent,
                            levels=levels,
                            origin="lower",
                            colors="k", # Black for high contrast
                            linewidths=1.0,
                            linestyles="dashed",
                            alpha=0.8,
                        )
                        ax.clabel(c, inline=True, fontsize=8, fmt='%.2f')
                    except Exception:
                        pass
        elif panel == "pilot_phase":
            _plot_im(ax, pilot_phase_waves, "Pilot Beam Phase (waves)", extent, cmap=cmap_phase)
        elif panel == "phase_residual":
            _plot_im(
                ax,
                phase_residual_waves,
                f"Phase Residual (Sim-Pilot, waves)\nRMS={metrics['phase_rms_waves']:.4f}, PV={metrics['phase_pv_waves']:.4f}",
                extent,
                cmap=cmap_residual,
                vmin=-vmax_phase_res / (2.0 * np.pi) if vmax_phase_res else None,
                vmax=vmax_phase_res / (2.0 * np.pi) if vmax_phase_res else None,
            )
        elif panel == "amplitude_slice":
            ax.plot(slice_coord, sim_amp_slice, label="Sim", color="tab:blue")
            ax.plot(slice_coord, pilot_amp_slice, label="Pilot", color="tab:orange")
            ax.plot(slice_coord, amp_res_slice, label="Residual", color="tab:green", linestyle="--")
            ax.set_title(f"Amplitude Slice ({slice_label})")
            ax.set_xlabel("Coord (mm)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
            ax.legend()
        elif panel == "phase_slice":
            ax.plot(slice_coord, sim_phase_slice, label="Sim", color="tab:blue")
            ax.plot(slice_coord, pilot_phase_slice, label="Pilot", color="tab:orange")
            ax.plot(slice_coord, phase_res_slice, label="Residual", color="tab:green", linestyle="--")
            ax.set_title(f"Phase Slice ({slice_label})")
            ax.set_xlabel("Coord (mm)")
            ax.set_ylabel("Phase (waves)")
            ax.grid(True, alpha=0.3)
            ax.legend()
        elif panel == "amplitude_residual_hist":
            data = amp_residual[valid_mask] if np.any(valid_mask) else amp_residual.flatten()
            ax.hist(data, bins=50, color="steelblue", alpha=0.8)
            ax.axvline(0.0, color="red", linestyle="--", linewidth=1.5)
            ax.set_title("Amplitude Residual Histogram")
            ax.set_xlabel("Residual")
            ax.set_ylabel("Count")
        elif panel == "phase_residual_hist":
            data = phase_residual_waves[valid_mask] if np.any(valid_mask) else phase_residual_waves.flatten()
            ax.hist(data, bins=50, color="steelblue", alpha=0.8)
            ax.axvline(0.0, color="red", linestyle="--", linewidth=1.5)
            ax.set_title("Phase Residual Histogram (Sim-Pilot, waves)")
            ax.set_xlabel("Residual (waves)")
            ax.set_ylabel("Count")
        else:
            ax.text(0.5, 0.5, f"Unknown panel: {panel}", ha="center", va="center")
            ax.set_axis_off()

    for ax in panel_axes[len(panels) :]:
        ax.set_axis_off()

    title_lines = _format_surface_info(surface, position)
    fig.suptitle("\n".join(title_lines[:2]), fontsize=12)

    footer_lines = title_lines[2:]
    footer_lines.extend(_format_axis_info(axis_state))
    footer_lines.extend(_format_grid_info(state, mask_threshold=mask_threshold))
    footer_lines.append(
        "Metrics: "
        f"max_amp={metrics['max_amp']:.4f}, energy={metrics['energy']:.4f}, "
        f"centroid=({metrics['centroid_x']:.4f}, {metrics['centroid_y']:.4f}) mm"
    )
    footer_lines.append(
        "Residuals: "
        f"amp_RMS={metrics['amp_rms_percent']:.3f}%, amp_PV={metrics['amp_pv_percent']:.3f}%, "
        f"phase_RMS={metrics['phase_rms_waves']:.4f} waves, phase_PV={metrics['phase_pv_waves']:.4f} waves"
    )
    footer_lines.append(f"Mask threshold: {mask_threshold:.3f} * max_amp")
    _add_footer(fig, footer_lines)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    plt.close(fig)
    return fig


def _scatter(ax, x, y, title, xlabel="X (mm)", ylabel="Y (mm)", size=8, color=None):
    if x is None or y is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return
    ax.scatter(x, y, s=size, c=color, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)


def _scatter_colored(ax, x, y, c, title, cmap="viridis", xlabel="X (mm)", ylabel="Y (mm)"):
    import matplotlib.pyplot as plt

    if x is None or y is None or c is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return
    sc = ax.scatter(x, y, c=c, s=10, cmap=cmap, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)


def _plot_debug_rays(
    debug: SurfaceDebugInfo,
    surface: Any,
    axis_state: Optional[OpticalAxisState],
    save_path: Optional[str | Path],
    show: bool,
    dpi: int,
):
    import matplotlib.pyplot as plt

    _configure_matplotlib_fonts(plt)
    in_rays = debug.entrance_rays_local or {}
    out_rays = debug.exit_rays_local or {}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    reconstruction_mask = getattr(debug, "reconstruction_mask", None)
    
    # helper for masked plotting
    def _plot_pts(ax, x, y, c, label, cmap="viridis", mask=None):
        if x is None or y is None or len(x) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            return
            
        # Plot rejected rays first (background)
        if mask is not None:
             rej_indices = np.where(~mask)[0] 
             if len(rej_indices) > 0:
                 ax.scatter(x[rej_indices], y[rej_indices], c='k', marker='x', s=15, alpha=0.3, label="Filtered")
             
             valid_indices = np.where(mask)[0]
             x = x[valid_indices]
             y = y[valid_indices]
             if c is not None and len(c) == len(mask):
                 c = c[valid_indices]

        if len(x) > 0:
            if c is not None:
                sc = ax.scatter(x, y, c=c, s=10, cmap=cmap, alpha=0.8, label="Valid")
                plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.scatter(x, y, s=8, alpha=0.7)
        
        ax.set_title(label)
        ax.grid(True, alpha=0.2)
        # ax.legend() # Optional, might clutter

    _scatter(
        axes[0, 0],
        in_rays.get("x"),
        in_rays.get("y"),
        "Entrance Rays (Local)",
    )
    _scatter(
        axes[0, 1],
        out_rays.get("x"),
        out_rays.get("y"),
        "Exit Rays (Local)",
    )
    _scatter(
        axes[0, 2],
        out_rays.get("L"),
        out_rays.get("M"),
        "Exit Ray Directions (L, M)",
        xlabel="L",
        ylabel="M",
        size=10,
    )

    _plot_pts(
        axes[1, 0],
        out_rays.get("x"),
        out_rays.get("y"),
        debug.absolute_opd_waves,
        "Absolute OPD (waves)",
        cmap="viridis",
        mask=reconstruction_mask
    )
    _plot_pts(
        axes[1, 1],
        out_rays.get("x"),
        out_rays.get("y"),
        debug.pilot_opd_waves,
        "Pilot OPD (waves)",
        cmap="viridis",
        mask=reconstruction_mask
    )
    _plot_pts(
        axes[1, 2],
        out_rays.get("x"),
        out_rays.get("y"),
        debug.residual_opd_waves,
        "Residual OPD (waves)",
        cmap="RdBu_r",
        mask=reconstruction_mask
    )

    title_lines = _format_surface_info(surface, "debug")
    fig.suptitle("\n".join(title_lines[:2]), fontsize=12)
    footer_lines = title_lines[2:]
    footer_lines.extend(_format_axis_info(axis_state))
    _add_footer(fig, footer_lines)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    plt.close(fig)
    return fig


def _plot_debug_rays_global(
    debug: SurfaceDebugInfo,
    surface: Any,
    axis_state: Optional[OpticalAxisState],
    save_path: Optional[str | Path],
    show: bool,
    dpi: int,
):
    import matplotlib.pyplot as plt

    _configure_matplotlib_fonts(plt)
    in_rays = debug.entrance_rays_global or {}
    out_rays = debug.exit_rays_global or {}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    _scatter(
        axes[0, 0],
        in_rays.get("x"),
        in_rays.get("y"),
        "Entrance Rays (Global XY)",
    )
    _scatter(
        axes[0, 1],
        out_rays.get("x"),
        out_rays.get("y"),
        "Exit Rays (Global XY)",
    )
    _scatter(
        axes[0, 2],
        out_rays.get("L"),
        out_rays.get("M"),
        "Exit Directions (Global L, M)",
        xlabel="L",
        ylabel="M",
        size=10,
    )
    _scatter(
        axes[1, 0],
        in_rays.get("x"),
        in_rays.get("z"),
        "Entrance Rays (Global XZ)",
        xlabel="X (mm)",
        ylabel="Z (mm)",
    )
    _scatter(
        axes[1, 1],
        out_rays.get("x"),
        out_rays.get("z"),
        "Exit Rays (Global XZ)",
        xlabel="X (mm)",
        ylabel="Z (mm)",
    )
    if out_rays.get("z") is None:
        axes[1, 2].text(0.5, 0.5, "No data", ha="center", va="center")
        axes[1, 2].set_axis_off()
    else:
        axes[1, 2].hist(out_rays.get("z"), bins=50, color="steelblue", alpha=0.8)
        axes[1, 2].set_title("Exit Z Distribution (Global)")
        axes[1, 2].set_xlabel("Z (mm)")
        axes[1, 2].set_ylabel("Count")

    title_lines = _format_surface_info(surface, "debug-global")
    fig.suptitle("\n".join(title_lines[:2]), fontsize=12)
    footer_lines = title_lines[2:]
    footer_lines.extend(_format_axis_info(axis_state))
    _add_footer(fig, footer_lines)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    plt.close(fig)
    return fig


def _plot_debug_opd_hist(
    debug: SurfaceDebugInfo,
    surface: Any,
    axis_state: Optional[OpticalAxisState],
    save_path: Optional[str | Path],
    show: bool,
    dpi: int,
):
    import matplotlib.pyplot as plt

    _configure_matplotlib_fonts(plt)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    plt.subplots_adjust(wspace=0.3)

    datasets = [
        ("Absolute OPD (waves)", debug.absolute_opd_waves, "tab:blue"),
        ("Pilot OPD (waves)", debug.pilot_opd_waves, "tab:orange"),
        ("Residual OPD (waves)", debug.residual_opd_waves, "tab:green"),
    ]
    for ax, (title, data, color) in zip(axes, datasets):
        if data is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue
        ax.hist(np.asarray(data).flatten(), bins=60, color=color, alpha=0.8)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("OPD (waves)")
        ax.set_ylabel("Count")

    title_lines = _format_surface_info(surface, "debug-opd-hist")
    fig.suptitle("\n".join(title_lines[:2]), fontsize=12)
    footer_lines = title_lines[2:]
    footer_lines.extend(_format_axis_info(axis_state))
    _add_footer(fig, footer_lines)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    plt.close(fig)
    return fig


def _plot_debug_direction_hist(
    debug: SurfaceDebugInfo,
    surface: Any,
    axis_state: Optional[OpticalAxisState],
    save_path: Optional[str | Path],
    show: bool,
    dpi: int,
):
    import matplotlib.pyplot as plt

    _configure_matplotlib_fonts(plt)
    in_rays = debug.entrance_rays_local or {}
    out_rays = debug.exit_rays_local or {}

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    datasets = [
        ("Entrance L", in_rays.get("L"), axes[0, 0]),
        ("Entrance M", in_rays.get("M"), axes[0, 1]),
        ("Exit L", out_rays.get("L"), axes[1, 0]),
        ("Exit M", out_rays.get("M"), axes[1, 1]),
    ]
    for title, data, ax in datasets:
        if data is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue
        ax.hist(np.asarray(data).flatten(), bins=60, color="slateblue", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Direction Cosine")
        ax.set_ylabel("Count")

    title_lines = _format_surface_info(surface, "debug-direction-hist")
    fig.suptitle("\n".join(title_lines[:2]), fontsize=12)
    footer_lines = title_lines[2:]
    footer_lines.extend(_format_axis_info(axis_state))
    _add_footer(fig, footer_lines)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    plt.close(fig)
    return fig


def _plot_debug_reconstruction(
    debug: SurfaceDebugInfo,
    surface: Any,
    state: Optional[PropagationState],
    axis_state: Optional[OpticalAxisState],
    save_path: Optional[str | Path],
    show: bool,
    dpi: int,
):
    import matplotlib.pyplot as plt

    _configure_matplotlib_fonts(plt)
    if state is None:
        return None

    grid = state.grid_sampling
    half_size = grid.physical_size_mm / 2.0
    extent = [-half_size, half_size, -half_size, half_size]

    residual_phase = debug.residual_phase_grid
    if residual_phase is None:
        residual_phase = state.phase - state.pilot_beam_params.compute_phase_grid(
            grid.grid_size, grid.physical_size_mm
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    _plot_im(axes[0, 0], state.amplitude, "Reconstructed Amplitude", extent, cmap="viridis")
    _plot_im(
        axes[0, 1],
        residual_phase / (2.0 * np.pi),
        "Residual Phase (Sim-Pilot, waves)",
        extent,
        cmap="RdBu_r",
    )

    pilot_phase = state.pilot_beam_params.compute_phase_grid(
        grid.grid_size, grid.physical_size_mm
    )
    _plot_im(
        axes[1, 0],
        pilot_phase / (2.0 * np.pi),
        "Pilot Phase (waves)",
        extent,
        cmap="RdBu",
    )
    _plot_im(
        axes[1, 1],
        state.phase / (2.0 * np.pi),
        "Full Phase (waves)",
        extent,
        cmap="RdBu",
    )

    title_lines = _format_surface_info(surface, "debug-recon")
    fig.suptitle("\n".join(title_lines[:2]), fontsize=12)
    footer_lines = title_lines[2:]
    footer_lines.extend(_format_axis_info(axis_state))
    footer_lines.extend(_format_grid_info(state, mask_threshold=0.01))

    # Add pilot refit summary if available in debug info
    refit_info = getattr(debug, "pilot_refit_info", None)
    if refit_info is not None:
        footer_lines.append(
            f"** Pilot Refit Applied ** PV: {refit_info['pv_before_waves']:.4f} -> {refit_info['pv_after_waves']:.4f} waves"
        )
        footer_lines.append(
            f"   R: {refit_info['R_before_mm']:.2f} -> {refit_info['R_after_mm']:.2f} mm, "
            f"w: {refit_info.get('w_before_mm', 0.0):.4f} -> {refit_info.get('w_after_mm', 0.0):.4f} mm, "
            f"w0: {refit_info.get('w0_before_mm', 0.0):.4f} -> {refit_info.get('w0_after_mm', 0.0):.4f} mm"
        )

    _add_footer(fig, footer_lines)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    plt.close(fig)
    return fig


def plot_surface_debug(
    surface_record: SurfaceRecord,
    save_dir: Optional[str | Path] = None,
    show: bool = True,
    dpi: int = 200,
    plots: Optional[list[str]] = None,
    filename_prefix: Optional[str] = None,
):
    debug = surface_record.debug
    if debug is None:
        return []

    if plots is None:
        plots = ["rays_local", "reconstruction"]
    plots = [p.lower().strip() for p in plots]
    if "all" in plots:
        plots = [
            "rays_local",
            "rays_global",
            "opd_hist",
            "direction_hist",
            "reconstruction",
        ]

    save_dir_path = Path(save_dir) if save_dir is not None else None
    saved = []

    rays_path = None
    recon_path = None
    prefix = f"{filename_prefix}_" if filename_prefix else ""
    if save_dir_path is not None:
        save_dir_path.mkdir(parents=True, exist_ok=True)
        rays_path = save_dir_path / f"{prefix}debug_rays_opd.png"
        recon_path = save_dir_path / f"{prefix}debug_reconstruction.png"

    if "rays" in plots or "rays_local" in plots:
        _plot_debug_rays(
            debug=debug,
            surface=surface_record.surface,
            axis_state=surface_record.exit_axis,
            save_path=rays_path,
            show=show,
            dpi=dpi,
        )
        if rays_path is not None:
            saved.append(str(rays_path))

    if "rays_global" in plots:
        global_path = None if save_dir_path is None else save_dir_path / f"{prefix}debug_rays_global.png"
        _plot_debug_rays_global(
            debug=debug,
            surface=surface_record.surface,
            axis_state=surface_record.exit_axis,
            save_path=global_path,
            show=show,
            dpi=dpi,
        )
        if global_path is not None:
            saved.append(str(global_path))

    if "opd_hist" in plots:
        opd_path = None if save_dir_path is None else save_dir_path / f"{prefix}debug_opd_hist.png"
        _plot_debug_opd_hist(
            debug=debug,
            surface=surface_record.surface,
            axis_state=surface_record.exit_axis,
            save_path=opd_path,
            show=show,
            dpi=dpi,
        )
        if opd_path is not None:
            saved.append(str(opd_path))

    if "direction_hist" in plots:
        dir_path = None if save_dir_path is None else save_dir_path / f"{prefix}debug_direction_hist.png"
        _plot_debug_direction_hist(
            debug=debug,
            surface=surface_record.surface,
            axis_state=surface_record.exit_axis,
            save_path=dir_path,
            show=show,
            dpi=dpi,
        )
        if dir_path is not None:
            saved.append(str(dir_path))

    if "reconstruction" in plots:
        _plot_debug_reconstruction(
            debug=debug,
            surface=surface_record.surface,
            state=surface_record.exit,
            axis_state=surface_record.exit_axis,
            save_path=recon_path,
            show=show,
            dpi=dpi,
        )
        if recon_path is not None:
            saved.append(str(recon_path))

    # Auto-detect if refit happened and plot comparison if not explicitly requested but available
    # Or if "refit" or "refit_comparison" is in plots
    if (
        debug.refit_occurred
        and debug.pilot_opd_waves_pre_refit is not None
        and debug.residual_opd_waves_pre_refit is not None
    ) and ("refit" in plots or "refit_comparison" in plots or "all" in plots or True):
        # Default to plotting if refit happened, to ensure visibility
        refit_path = (
            None
            if save_dir_path is None
            else save_dir_path / f"{prefix}debug_refit_comparison.png"
        )
        _plot_debug_refit_comparison(
            debug=debug,
            surface=surface_record.surface,
            axis_state=surface_record.exit_axis,
            save_path=refit_path,
            show=show,
            dpi=dpi,
        )
        if refit_path is not None:
            saved.append(str(refit_path))

    return saved


def _plot_debug_refit_comparison(
    debug: SurfaceDebugInfo,
    surface: Any,
    axis_state: Optional[OpticalAxisState],
    save_path: Optional[str | Path],
    show: bool,
    dpi: int,
):
    import matplotlib.pyplot as plt

    _configure_matplotlib_fonts(plt)
    
    # Pre-refit data
    pilot_pre = debug.pilot_opd_waves_pre_refit
    resid_pre = debug.residual_opd_waves_pre_refit
    
    # Post-refit data
    pilot_post = debug.pilot_opd_waves
    resid_post = debug.residual_opd_waves
    
    # Coordinates (using local rays from debug info)
    rays = debug.exit_rays_local
    if rays is None or pilot_pre is None or resid_pre is None:
        return

    x = rays.get("x")
    y = rays.get("y")
    
    # Mask for plotting (same as other debug plots)
    reconstruction_mask = getattr(debug, "reconstruction_mask", None)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # Helper to plot masked scatter
    def _plot_pts(ax, data, title, cmap):
        if x is None or y is None or data is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            return
            
        # Plot filtered (background)
        if reconstruction_mask is not None:
             rej_indices = np.where(~reconstruction_mask)[0] 
             if len(rej_indices) > 0:
                 ax.scatter(x[rej_indices], y[rej_indices], c='k', marker='x', s=15, alpha=0.1, label="Filtered")
             
             valid_indices = np.where(reconstruction_mask)[0]
             plot_x = x[valid_indices]
             plot_y = y[valid_indices]
             plot_data = data[valid_indices]
        else:
             plot_x = x
             plot_y = y
             plot_data = data

        if len(plot_x) > 0:
            sc = ax.scatter(plot_x, plot_y, c=plot_data, s=10, cmap=cmap, alpha=0.8)
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        
        ax.set_title(title)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.grid(True, alpha=0.2)

    # Row 1: Pre-refit
    _plot_pts(axes[0, 0], pilot_pre, "PRE-Refit: Pilot OPD (waves)", "viridis")
    
    # Check PV of pre-refit residual for title
    pv_pre = np.ptp(resid_pre[reconstruction_mask]) if reconstruction_mask is not None and np.any(reconstruction_mask) else np.ptp(resid_pre)
    rms_pre = np.std(resid_pre[reconstruction_mask]) if reconstruction_mask is not None and np.any(reconstruction_mask) else np.std(resid_pre)
    
    _plot_pts(axes[0, 1], resid_pre, f"PRE-Refit: Residual (PV={pv_pre:.4f}, RMS={rms_pre:.4f})", "RdBu_r")

    # Row 2: Post-refit
    _plot_pts(axes[1, 0], pilot_post, "POST-Refit: Pilot OPD (waves)", "viridis")
    
    pv_post = np.ptp(resid_post[reconstruction_mask]) if reconstruction_mask is not None and np.any(reconstruction_mask) else np.ptp(resid_post)
    rms_post = np.std(resid_post[reconstruction_mask]) if reconstruction_mask is not None and np.any(reconstruction_mask) else np.std(resid_post)

    _plot_pts(axes[1, 1], resid_post, f"POST-Refit: Residual (PV={pv_post:.4f}, RMS={rms_post:.4f})", "RdBu_r")

    # Title & Footer
    refit_info = getattr(debug, "pilot_refit_info", {}) or {}
    
    title_lines = _format_surface_info(surface, "refit-comparison")
    fig.suptitle("\n".join(title_lines[:2]), fontsize=12)
    
    footer_lines = title_lines[2:]
    footer_lines.extend(_format_axis_info(axis_state))
    
    if refit_info:
        footer_lines.append("-" * 30)
        footer_lines.append("** REFIT DETAILS **")
        footer_lines.append(f"PV Improvement: {refit_info.get('pv_before_waves', 0):.4f} -> {refit_info.get('pv_after_waves', 0):.4f} waves")
        footer_lines.append(f"Curvature R: {refit_info.get('R_before_mm', 0):.2f} -> {refit_info.get('R_after_mm', 0):.2f} mm")
        footer_lines.append(f"Beam Size w: {refit_info.get('w_before_mm', 0):.4f} -> {refit_info.get('w_after_mm', 0):.4f} mm")

    _add_footer(fig, footer_lines)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    plt.close(fig)
    return fig



def _resolve_surface_geometry(
    surface: Any,
    surface_overrides: Optional[dict[str, object]],
) -> tuple[np.ndarray, float, float, list[float]]:
    orientation = np.asarray(getattr(surface, "orientation", np.eye(3)), dtype=float)
    radius = float(getattr(surface, "radius", np.inf))
    radius_x = float(getattr(surface, "radius_x", np.inf))
    asphere_coeffs = list(getattr(surface, "asphere_coeffs", []) or [])
    if surface_overrides:
        if surface_overrides.get("orientation") is not None:
            orientation = np.asarray(surface_overrides["orientation"], dtype=float)
        if surface_overrides.get("radius") is not None:
            radius = float(surface_overrides["radius"])
        if surface_overrides.get("radius_x") is not None:
            radius_x = float(surface_overrides["radius_x"])
        if surface_overrides.get("asphere_coeffs") is not None:
            asphere_coeffs = list(surface_overrides["asphere_coeffs"] or [])
    return orientation, radius, radius_x, asphere_coeffs


def _compute_conic_sag(r_sq: np.ndarray, radius: float, conic: float) -> np.ndarray:
    if not np.isfinite(radius):
        return np.zeros_like(r_sq, dtype=float)
    c = 1.0 / radius
    disc = 1.0 - (1.0 + conic) * (c**2) * r_sq
    sag = np.full_like(r_sq, np.nan, dtype=float)
    valid = disc >= 0.0
    if np.any(valid):
        sag[valid] = c * r_sq[valid] / (1.0 + np.sqrt(disc[valid]))
    return sag


def _compute_even_asphere_sag(
    r_sq: np.ndarray,
    radius: float,
    conic: float,
    coeffs: Sequence[float],
) -> np.ndarray:
    sag = _compute_conic_sag(r_sq, radius, conic)
    if coeffs:
        r_pow = r_sq * r_sq
        for idx, coeff in enumerate(coeffs):
            if idx > 0:
                r_pow = r_pow * r_sq
            sag = sag + float(coeff) * r_pow
    return sag


def _compute_biconic_sag(
    x: np.ndarray,
    y: np.ndarray,
    radius_x: float,
    radius_y: float,
    conic_x: float,
    conic_y: float,
) -> np.ndarray:
    sag_x = _compute_conic_sag(x**2, radius_x, conic_x)
    sag_y = _compute_conic_sag(y**2, radius_y, conic_y)
    sag = sag_x + sag_y
    invalid = np.isnan(sag_x) | np.isnan(sag_y)
    sag[invalid] = np.nan
    return sag


def _build_plane_mesh(
    origin: np.ndarray,
    frame: np.ndarray,
    size_x: float,
    size_y: float,
    samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(-size_x / 2.0, size_x / 2.0, samples)
    ys = np.linspace(-size_y / 2.0, size_y / 2.0, samples)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    local = np.stack([X, Y, Z], axis=-1)
    global_pts = local @ frame.T + origin
    return global_pts[..., 0], global_pts[..., 1], global_pts[..., 2]


def _build_surface_mesh(
    surface: Any,
    orientation: np.ndarray,
    radius: float,
    radius_x: float,
    asphere_coeffs: Sequence[float],
    aperture: float,
    samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.linspace(-aperture, aperture, samples)
    X, Y = np.meshgrid(coords, coords)
    r_sq = X**2 + Y**2

    surface_type = (getattr(surface, "surface_type", "") or "").lower()
    conic = float(getattr(surface, "conic", 0.0))
    conic_x = float(getattr(surface, "conic_x", 0.0))

    if surface_type == "biconic":
        sag = _compute_biconic_sag(X, Y, radius_x, radius, conic_x, conic)
    elif surface_type == "even_asphere":
        sag = _compute_even_asphere_sag(r_sq, radius, conic, asphere_coeffs)
    else:
        sag = _compute_conic_sag(r_sq, radius, conic)

    if aperture > 0:
        sag = np.where(r_sq <= aperture**2, sag, np.nan)

    local = np.stack([X, Y, sag], axis=-1)
    vertex = np.asarray(getattr(surface, "vertex_position", [0.0, 0.0, 0.0]), dtype=float)
    global_pts = local @ orientation.T + vertex
    return global_pts[..., 0], global_pts[..., 1], global_pts[..., 2]


def _rays_dict_to_points(rays: Optional[dict[str, np.ndarray]]) -> Optional[np.ndarray]:
    if not rays:
        return None
    x = rays.get("x")
    y = rays.get("y")
    z = rays.get("z")
    if x is None or y is None or z is None:
        return None
    return np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)])


def _rays_dict_to_dirs(rays: Optional[dict[str, np.ndarray]]) -> Optional[np.ndarray]:
    if not rays:
        return None
    l = rays.get("L")
    m = rays.get("M")
    n = rays.get("N")
    if l is None or m is None or n is None:
        return None
    return np.column_stack([np.asarray(l), np.asarray(m), np.asarray(n)])


def _rays_dict_to_values(
    rays: Optional[dict[str, np.ndarray]], key: str
) -> Optional[np.ndarray]:
    if not rays:
        return None
    value = rays.get(key)
    if value is None:
        return None
    return np.asarray(value)


def _sample_indices(count: int, max_count: int) -> np.ndarray:
    if count <= max_count:
        return np.arange(count, dtype=int)
    return np.linspace(0, count - 1, max_count, dtype=int)


def _estimate_aperture_from_points(
    points: Optional[np.ndarray],
    origin: np.ndarray,
    frame: np.ndarray,
    fallback: float,
) -> float:
    if points is None or points.size == 0:
        return fallback
    local = (points - origin) @ frame
    r = np.sqrt(local[:, 0] ** 2 + local[:, 1] ** 2)
    if r.size == 0:
        return fallback
    max_r = float(np.max(r))
    if max_r <= 0:
        return fallback
    return max(max_r * 1.1, fallback)


def _positive_quadrant_mask(
    points: Optional[np.ndarray],
    origin: np.ndarray,
    frame: np.ndarray,
) -> Optional[np.ndarray]:
    if points is None or points.size == 0:
        return None
    local = (points - origin) @ frame
    return (local[:, 0] >= 0.0) & (local[:, 1] >= 0.0)


def _select_beam_points(
    surface_pts: Optional[np.ndarray],
    entrance_pts: Optional[np.ndarray],
    exit_pts: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if surface_pts is not None and surface_pts.size:
        return surface_pts
    if entrance_pts is not None and entrance_pts.size:
        return entrance_pts
    if exit_pts is not None and exit_pts.size:
        return exit_pts
    return None


def _draw_xy_axes_3d(
    ax,
    origin: np.ndarray,
    frame: np.ndarray,
    scale: float,
    label_prefix: str,
    color_x: str,
    color_y: str,
    linestyle: str,
    alpha: float,
    axis_order: str,
) -> None:
    origin = np.asarray(origin, dtype=float)
    frame = np.asarray(frame, dtype=float)
    x_axis = frame[:, 0] * scale
    y_axis = frame[:, 1] * scale

    x_end = origin + x_axis
    y_end = origin + y_axis

    axis_order = axis_order.lower().strip()
    if len(axis_order) != 3 or set(axis_order) != {"x", "y", "z"}:
        raise ValueError("axis_order must be a permutation of 'xyz'")
    axis_map = {"x": 0, "y": 1, "z": 2}
    idx = [axis_map[name] for name in axis_order]
    origin = origin[idx]
    x_end = x_end[idx]
    y_end = y_end[idx]

    ax.plot(
        [origin[0], x_end[0]],
        [origin[1], x_end[1]],
        [origin[2], x_end[2]],
        color=color_x,
        linestyle=linestyle,
        linewidth=1.4,
        alpha=alpha,
    )
    ax.plot(
        [origin[0], y_end[0]],
        [origin[1], y_end[1]],
        [origin[2], y_end[2]],
        color=color_y,
        linestyle=linestyle,
        linewidth=1.4,
        alpha=alpha,
    )
    ax.text(x_end[0], x_end[1], x_end[2], f"{label_prefix}-X", fontsize=8, color=color_x)
    ax.text(y_end[0], y_end[1], y_end[2], f"{label_prefix}-Y", fontsize=8, color=color_y)


def plot_surface_raytrace_3d(
    surface: Any,
    entrance_axis: Optional[OpticalAxisState],
    exit_axis: Optional[OpticalAxisState],
    debug: Optional[SurfaceDebugInfo] = None,
    entrance_rays: Optional[dict[str, np.ndarray]] = None,
    surface_rays: Optional[dict[str, np.ndarray]] = None,
    exit_rays: Optional[dict[str, np.ndarray]] = None,
    surface_overrides: Optional[dict[str, object]] = None,
    save_path: Optional[str | Path] = None,
    show: bool = True,
    dpi: int = 200,
    max_rays: int = 5,
    plane_samples: int = 18,
    surface_samples: int = 60,
    elevation: float = 30.0,
    azimuth: float = -60.0,
    axis_order: str = "zyx",
    positive_quadrant: bool = True,
    ray_color_by: Optional[str] = None,
    ray_cmap: str = "viridis",
    ray_colorbar: bool = True,
    ray_color_percentile: Optional[tuple[float, float]] = (2.0, 98.0),
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    _configure_matplotlib_fonts(plt)

    max_rays = max(1, int(max_rays))
    axis_order = axis_order.lower().strip()
    if len(axis_order) != 3 or set(axis_order) != {"x", "y", "z"}:
        raise ValueError("axis_order must be a permutation of 'xyz'")
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_idx = [axis_map[name] for name in axis_order]

    if debug is not None:
        if entrance_rays is None:
            entrance_rays = debug.entrance_rays_global
        if surface_rays is None:
            surface_rays = getattr(debug, "surface_rays_global", None)
        if exit_rays is None:
            exit_rays = debug.exit_rays_global

    reconstruction_mask = getattr(debug, "reconstruction_mask", None) if debug is not None else None

    entrance_pts_full = _rays_dict_to_points(entrance_rays)
    surface_pts_full = _rays_dict_to_points(surface_rays)
    exit_pts_full = _rays_dict_to_points(exit_rays)
    entrance_dirs_full = _rays_dict_to_dirs(entrance_rays)
    surface_dirs_full = _rays_dict_to_dirs(surface_rays)
    exit_dirs_full = _rays_dict_to_dirs(exit_rays)

    color_by = ray_color_by
    color_key = color_by if color_by else None

    entrance_color_full = _rays_dict_to_values(entrance_rays, color_key) if color_by else None
    surface_color_full = _rays_dict_to_values(surface_rays, color_key) if color_by else None
    exit_color_full = _rays_dict_to_values(exit_rays, color_key) if color_by else None

    orientation, radius, radius_x, asphere_coeffs = _resolve_surface_geometry(
        surface, surface_overrides
    )
    vertex = np.asarray(getattr(surface, "vertex_position", [0.0, 0.0, 0.0]), dtype=float)

    base_aperture = float(getattr(surface, "semi_aperture", 0.0) or 0.0)
    fallback_aperture = max(5.0, base_aperture * 1.1) if base_aperture > 0 else 5.0
    beam_pts = _select_beam_points(surface_pts_full, entrance_pts_full, exit_pts_full)
    if beam_pts is not None:
        surface_aperture = _estimate_aperture_from_points(
            beam_pts, vertex, orientation, fallback_aperture
        )
    else:
        surface_aperture = base_aperture if base_aperture > 0 else fallback_aperture

    entrance_plane = None
    if entrance_axis is not None:
        entrance_ap = _estimate_aperture_from_points(
            entrance_pts_full,
            np.asarray(entrance_axis.position, dtype=float),
            np.asarray(entrance_axis.frame, dtype=float),
            surface_aperture,
        )
        entrance_plane = _build_plane_mesh(
            np.asarray(entrance_axis.position, dtype=float),
            np.asarray(entrance_axis.frame, dtype=float),
            entrance_ap * 2.0,
            entrance_ap * 2.0,
            plane_samples,
        )

    exit_plane = None
    if exit_axis is not None:
        exit_ap = _estimate_aperture_from_points(
            exit_pts_full,
            np.asarray(exit_axis.position, dtype=float),
            np.asarray(exit_axis.frame, dtype=float),
            surface_aperture,
        )
        exit_plane = _build_plane_mesh(
            np.asarray(exit_axis.position, dtype=float),
            np.asarray(exit_axis.frame, dtype=float),
            exit_ap * 2.0,
            exit_ap * 2.0,
            plane_samples,
        )

    surface_mesh = _build_surface_mesh(
        surface=surface,
        orientation=orientation,
        radius=radius,
        radius_x=radius_x,
        asphere_coeffs=asphere_coeffs,
        aperture=surface_aperture,
        samples=surface_samples,
    )

    # 只筛选光线（不裁剪入/出射面与镜面）
    entrance_pts = entrance_pts_full
    surface_pts = surface_pts_full
    exit_pts = exit_pts_full
    entrance_dirs = entrance_dirs_full
    surface_dirs = surface_dirs_full
    exit_dirs = exit_dirs_full
    entrance_color = entrance_color_full
    surface_color = surface_color_full
    exit_color = exit_color_full
    
    # Store compatible mask for consistent filtering
    quadrant_mask = None
    if positive_quadrant and entrance_axis is not None and entrance_pts_full is not None:
        quadrant_mask = _positive_quadrant_mask(
            entrance_pts_full,
            np.asarray(entrance_axis.position, dtype=float),
            np.asarray(entrance_axis.frame, dtype=float),
        )

    # Apply quadrant mask if needed (update reconstruction mask too)
    if quadrant_mask is not None:
        def _apply_mask_to_all(mask_to_apply):
            nonlocal entrance_pts, surface_pts, exit_pts
            nonlocal entrance_dirs, surface_dirs, exit_dirs
            nonlocal entrance_color, surface_color, exit_color, reconstruction_mask

            if entrance_pts is not None and entrance_pts.shape[0] == mask_to_apply.shape[0]:
                entrance_pts = entrance_pts[mask_to_apply]
            if surface_pts is not None and surface_pts.shape[0] == mask_to_apply.shape[0]:
                surface_pts = surface_pts[mask_to_apply]
            if exit_pts is not None and exit_pts.shape[0] == mask_to_apply.shape[0]:
                exit_pts = exit_pts[mask_to_apply]
            
            if entrance_dirs is not None and entrance_dirs.shape[0] == mask_to_apply.shape[0]:
                entrance_dirs = entrance_dirs[mask_to_apply]
            if surface_dirs is not None and surface_dirs.shape[0] == mask_to_apply.shape[0]:
                surface_dirs = surface_dirs[mask_to_apply]
            if exit_dirs is not None and exit_dirs.shape[0] == mask_to_apply.shape[0]:
                exit_dirs = exit_dirs[mask_to_apply]

            if entrance_color is not None and entrance_color.shape[0] == mask_to_apply.shape[0]:
                entrance_color = entrance_color[mask_to_apply]
            if surface_color is not None and surface_color.shape[0] == mask_to_apply.shape[0]:
                surface_color = surface_color[mask_to_apply]
            if exit_color is not None and exit_color.shape[0] == mask_to_apply.shape[0]:
                exit_color = exit_color[mask_to_apply]
            
            if reconstruction_mask is not None and reconstruction_mask.shape[0] == mask_to_apply.shape[0]:
                reconstruction_mask = reconstruction_mask[mask_to_apply]

        _apply_mask_to_all(quadrant_mask)

    def _reorder_mesh(mesh: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]):
        if mesh is None:
            return None
        xg, yg, zg = mesh
        stacked = np.stack([xg, yg, zg], axis=-1)[..., axis_idx]
        return stacked[..., 0], stacked[..., 1], stacked[..., 2]

    entrance_plane_plot = _reorder_mesh(entrance_plane)
    exit_plane_plot = _reorder_mesh(exit_plane)
    surface_mesh_plot = _reorder_mesh(surface_mesh)

    def _reorder_points(points: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if points is None:
            return None
        return points[:, axis_idx]

    entrance_pts_plot = _reorder_points(entrance_pts)
    surface_pts_plot = _reorder_points(surface_pts)
    exit_pts_plot = _reorder_points(exit_pts)

    entrance_dirs_plot = _reorder_points(entrance_dirs)
    surface_dirs_plot = _reorder_points(surface_dirs)
    exit_dirs_plot = _reorder_points(exit_dirs)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    plane_color_in = "#f4c27a"
    plane_color_out = "#cbb5e8"
    surf_color = "#d9d9d9"

    color_label = None
    vmin = None
    vmax = None
    if color_by:
        color_label = "OPD" if color_by == "opd" else "Intensity"
        values_pool = [
            arr for arr in (entrance_color, surface_color, exit_color) if arr is not None and arr.size
        ]
        if values_pool:
            all_vals = np.concatenate([np.asarray(v).ravel() for v in values_pool])
            finite_vals = all_vals[np.isfinite(all_vals)]
            if finite_vals.size:
                if ray_color_percentile is not None:
                    vmin, vmax = np.nanpercentile(finite_vals, ray_color_percentile)
                else:
                    vmin = float(np.nanmin(finite_vals))
                    vmax = float(np.nanmax(finite_vals))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    vmin = None
                    vmax = None

    if entrance_plane_plot is not None:
        ax.plot_surface(
            *entrance_plane_plot,
            color=plane_color_in,
            alpha=0.25,
            linewidth=0.0,
            edgecolor="none",
        )
    ax.plot_surface(
        *surface_mesh_plot,
        color=surf_color,
        alpha=0.55,
        linewidth=0.0,
        edgecolor="none",
    )
    if exit_plane_plot is not None:
        ax.plot_surface(
            *exit_plane_plot,
            color=plane_color_out,
            alpha=0.25,
            linewidth=0.0,
            edgecolor="none",
        )

    def _scatter_subset(
        points: Optional[np.ndarray],
        color: str,
        size: int,
        alpha: float,
        values: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        invert_mask: bool = False,
    ):
        if points is None or points.size == 0:
            return None
        
        # Determine subset indices based on mask
        if mask is not None:
            if invert_mask:
                subset_indices = np.where(~mask)[0]
            else:
                subset_indices = np.where(mask)[0]
        else:
            subset_indices = np.arange(len(points))
            
        if subset_indices.size == 0:
            return None

        # Downsample if needed
        if len(subset_indices) > max_rays:
             idx_in_subset = _sample_indices(len(subset_indices), max_rays)
             final_indices = subset_indices[idx_in_subset]
        else:
             final_indices = subset_indices

        pts = points[final_indices]
        if values is None:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=size, color=color, alpha=alpha)
            return None
        vals = values[final_indices]
        return ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=size,
            c=vals,
            cmap=ray_cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
        )

    color_mappable = None
    
    # Plot rejected rays first (if any)
    if reconstruction_mask is not None:
        _scatter_subset(entrance_pts_plot, "gray", 3, 0.1, mask=reconstruction_mask, invert_mask=True)
        # We don't plot values for rejected rays to keep it clean
        
    # Plot valid rays
    color_mappable = _scatter_subset(
        entrance_pts_plot, "tab:blue", 7, 0.5, values=entrance_color, mask=reconstruction_mask
    ) or color_mappable
    color_mappable = _scatter_subset(
        surface_pts_plot, "tab:red", 12, 0.85, values=surface_color, mask=reconstruction_mask
    ) or color_mappable
    color_mappable = _scatter_subset(
        exit_pts_plot, "tab:green", 7, 0.5, values=exit_color, mask=reconstruction_mask
    ) or color_mappable

    ray_count = 0
    plotted_rays = 0
    if entrance_pts_plot is not None and surface_pts_plot is not None and exit_pts_plot is not None:
        ray_count = min(len(entrance_pts_plot), len(surface_pts_plot), len(exit_pts_plot))
        if ray_count > 0:
            # Prepare full arrays truncated to min length
            p0_full = entrance_pts_plot[:ray_count]
            p1_full = surface_pts_plot[:ray_count]
            p2_full = exit_pts_plot[:ray_count]
            mask_full = reconstruction_mask[:ray_count] if reconstruction_mask is not None else np.ones(ray_count, dtype=bool)
            
            # Select indices for valid lines
            valid_indices = np.where(mask_full)[0]
            if valid_indices.size > 0:
                indices = _sample_indices(len(valid_indices), max_rays)
                plotted_rays = len(indices)
                final_indices = valid_indices[indices]
                
                for idx in final_indices:
                    p0 = p0_full[idx]
                    p1 = p1_full[idx]
                    p2 = p2_full[idx]
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color="tab:orange", linewidth=1.0, alpha=0.8)
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="tab:purple", linewidth=1.0, alpha=0.8)

            # Select indices for rejected lines (fewer)
            rejected_indices = np.where(~mask_full)[0]
            if rejected_indices.size > 0:
                 rej_indices = _sample_indices(len(rejected_indices), max(1, max_rays // 2))
                 final_rej_indices = rejected_indices[rej_indices]
                 for idx in final_rej_indices:
                    p0 = p0_full[idx]
                    p1 = p1_full[idx]
                    p2 = p2_full[idx]
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color="gray", linewidth=0.5, alpha=0.1)

            arrow_idx = indices[:: max(1, len(indices) // 8)]
            arrow_length = max(0.5, surface_aperture * 0.25)
            if arrow_idx.size > 0:
                p0 = entrance_pts_plot[arrow_idx]
                p1 = surface_pts_plot[arrow_idx]
                p2 = exit_pts_plot[arrow_idx]
                if entrance_dirs_plot is not None:
                    dir_in = entrance_dirs_plot[arrow_idx]
                else:
                    dir_in = p1 - p0
                if surface_dirs_plot is not None:
                    dir_out = surface_dirs_plot[arrow_idx]
                elif exit_dirs_plot is not None:
                    dir_out = exit_dirs_plot[arrow_idx]
                else:
                    dir_out = p2 - p1
                norm_in = np.linalg.norm(dir_in, axis=1)
                norm_out = np.linalg.norm(dir_out, axis=1)
                valid_in = norm_in > 1e-12
                valid_out = norm_out > 1e-12
                if np.any(valid_in):
                    ax.quiver(
                        p0[valid_in, 0],
                        p0[valid_in, 1],
                        p0[valid_in, 2],
                        dir_in[valid_in, 0],
                        dir_in[valid_in, 1],
                        dir_in[valid_in, 2],
                        length=arrow_length,
                        normalize=True,
                        color="tab:orange",
                        alpha=0.95,
                        linewidth=1.0,
                    )
                if np.any(valid_out):
                    ax.quiver(
                        p1[valid_out, 0],
                        p1[valid_out, 1],
                        p1[valid_out, 2],
                        dir_out[valid_out, 0],
                        dir_out[valid_out, 1],
                        dir_out[valid_out, 2],
                        length=arrow_length,
                        normalize=True,
                        color="tab:purple",
                        alpha=0.95,
                        linewidth=1.0,
                    )

    axis_scale = max(1.0, surface_aperture * 0.6)
    if entrance_axis is not None:
        _draw_xy_axes_3d(
            ax=ax,
            origin=np.asarray(entrance_axis.position, dtype=float),
            frame=np.asarray(entrance_axis.frame, dtype=float),
            scale=axis_scale,
            label_prefix="In",
            color_x="#1f77b4",
            color_y="#2ca02c",
            linestyle="-",
            alpha=0.9,
            axis_order=axis_order,
        )
    if exit_axis is not None:
        _draw_xy_axes_3d(
            ax=ax,
            origin=np.asarray(exit_axis.position, dtype=float),
            frame=np.asarray(exit_axis.frame, dtype=float),
            scale=axis_scale,
            label_prefix="Out",
            color_x="#6baed6",
            color_y="#98df8a",
            linestyle="--",
            alpha=0.9,
            axis_order=axis_order,
        )

    if color_mappable is not None and ray_colorbar and color_label:
        fig.colorbar(
            color_mappable,
            ax=ax,
            shrink=0.6,
            pad=0.02,
            label=color_label,
        )

    legend_handles = [
        Patch(facecolor=plane_color_in, edgecolor="gray", label="Entrance plane", alpha=0.25),
        Patch(facecolor=surf_color, edgecolor="gray", label="Surface", alpha=0.55),
        Patch(facecolor=plane_color_out, edgecolor="gray", label="Exit plane", alpha=0.25),
        Line2D([0], [0], color="tab:orange", label="Incoming rays"),
        Line2D([0], [0], color="tab:purple", label="Outgoing rays"),
        Line2D([0], [0], marker="o", color="w", label="Surface hits", markerfacecolor="tab:red", markersize=6),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")

    axis_labels = {"x": "X", "y": "Y", "z": "Z"}
    ax.set_xlabel(f"{axis_labels[axis_order[0]]} (mm)")
    ax.set_ylabel(f"{axis_labels[axis_order[1]]} (mm)")
    ax.set_zlabel(f"{axis_labels[axis_order[2]]} (mm)")
    ax.grid(False)
    ax.view_init(elev=elevation, azim=azimuth)

    points_for_scale = []
    for mesh in (entrance_plane_plot, surface_mesh_plot, exit_plane_plot):
        if mesh is None:
            continue
        xg, yg, zg = mesh
        flat = np.column_stack([xg.ravel(), yg.ravel(), zg.ravel()])
        finite = np.isfinite(flat).all(axis=1)
        if np.any(finite):
            points_for_scale.append(flat[finite])
    for pts in (entrance_pts_plot, surface_pts_plot, exit_pts_plot):
        if pts is not None and pts.size > 0:
            points_for_scale.append(pts)
    if points_for_scale:
        _set_equal_3d_axes(ax, np.vstack(points_for_scale))

    title_lines = _format_surface_info(surface, "raytrace-3d")
    fig.suptitle("\n".join(title_lines[:2]), fontsize=12)

    footer_lines = title_lines[2:]
    if entrance_axis is not None:
        pos = np.asarray(entrance_axis.position, dtype=float)
        direction = np.asarray(entrance_axis.direction, dtype=float)
        footer_lines.append(
            f"Entrance axis pos=({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}) mm, "
            f"dir=({direction[0]:.6f},{direction[1]:.6f},{direction[2]:.6f})"
        )
    if exit_axis is not None:
        pos = np.asarray(exit_axis.position, dtype=float)
        direction = np.asarray(exit_axis.direction, dtype=float)
        footer_lines.append(
            f"Exit axis pos=({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}) mm, "
            f"dir=({direction[0]:.6f},{direction[1]:.6f},{direction[2]:.6f})"
        )
    if surface_overrides:
        override_keys = sorted(str(k) for k, v in surface_overrides.items() if v is not None)
        if override_keys:
            footer_lines.append(f"Overrides: {', '.join(override_keys)}")
    if ray_count:
        footer_lines.append(f"Rays: total={ray_count}, plotted={plotted_rays}")
    _add_footer(fig, footer_lines)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    plt.close(fig)
    return fig


def _get_wavefront_3d_field(
    state: PropagationState,
    plot_type: str,
    mask_threshold: float,
) -> tuple[np.ndarray, str, str]:
    plot_type = plot_type.lower().strip()
    grid = state.grid_sampling
    pilot = state.pilot_beam_params
    sim_amp = state.amplitude
    sim_phase = state.phase
    pilot_amp = pilot.compute_amplitude_grid(grid.grid_size, grid.physical_size_mm)
    pilot_phase = pilot.compute_phase_grid(grid.grid_size, grid.physical_size_mm)
    residual_phase = sim_phase - pilot_phase
    residual_amp = sim_amp - pilot_amp

    valid_mask = compute_valid_mask(sim_amp, mask_threshold)
    if plot_type == "amplitude":
        return sim_amp, "Amplitude", "hot"
    if plot_type == "phase":
        return sim_phase / (2.0 * np.pi), "Phase (waves)", "twilight"
    if plot_type == "pilot_phase":
        return pilot_phase / (2.0 * np.pi), "Pilot Phase (waves)", "twilight"
    if plot_type == "residual_phase":
        data = np.where(valid_mask, residual_phase / (2.0 * np.pi), np.nan)
        return data, "Residual Phase (Sim-Pilot, waves)", "RdBu_r"
    if plot_type == "pilot_amplitude":
        return pilot_amp, "Pilot Amplitude", "hot"
    if plot_type == "residual_amplitude":
        data = np.where(valid_mask, residual_amp, np.nan)
        return data, "Residual Amplitude", "RdBu_r"
    if plot_type == "intensity":
        return sim_amp**2, "Intensity", "hot"
    raise ValueError(f"Unknown plot_type: {plot_type}")


def plot_wavefront_3d(
    state: PropagationState,
    surface: Any,
    axis_state: Optional[OpticalAxisState],
    position: str,
    plot_type: str = "residual_phase",
    save_path: Optional[str | Path] = None,
    show: bool = True,
    dpi: int = 200,
    elevation: float = 30.0,
    azimuth: float = -60.0,
    stride: Optional[int] = None,
    mask_threshold: float = 0.01,
):
    import matplotlib.pyplot as plt

    _configure_matplotlib_fonts(plt)
    grid = state.grid_sampling
    coords = (np.arange(grid.grid_size) - grid.grid_size // 2) * grid.sampling_mm
    X, Y = np.meshgrid(coords, coords)

    data, title, cmap = _get_wavefront_3d_field(state, plot_type, mask_threshold)
    n = grid.grid_size
    stride = stride or max(1, n // 120)
    X_ds = X[::stride, ::stride]
    Y_ds = Y[::stride, ::stride]
    Z_ds = data[::stride, ::stride]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X_ds,
        Y_ds,
        Z_ds,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.9,
    )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel(title)
    ax.set_title(f"Surface {getattr(surface, 'index', '?')} ({position}) - {title}")
    ax.view_init(elev=elevation, azim=azimuth)
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, pad=0.08, label=title)

    footer_lines = _format_surface_info(surface, position)[2:]
    footer_lines.extend(_format_axis_info(axis_state))
    footer_lines.extend(_format_grid_info(state))
    _add_footer(fig, footer_lines)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    plt.close(fig)
    return fig


def plot_surface_detail_3d(
    state: PropagationState,
    surface: Any,
    axis_state: Optional[OpticalAxisState],
    position: str,
    save_path: Optional[str | Path] = None,
    show: bool = True,
    dpi: int = 200,
    elevation: float = 30.0,
    azimuth: float = -60.0,
    stride: Optional[int] = None,
    mask_threshold: float = 0.01,
):
    import matplotlib.pyplot as plt

    _configure_matplotlib_fonts(plt)
    plot_types = [
        "amplitude",
        "phase",
        "pilot_phase",
        "residual_phase",
        "pilot_amplitude",
        "residual_amplitude",
    ]

    grid = state.grid_sampling
    coords = (np.arange(grid.grid_size) - grid.grid_size // 2) * grid.sampling_mm
    X, Y = np.meshgrid(coords, coords)

    n = grid.grid_size
    stride = stride or max(1, n // 120)
    X_ds = X[::stride, ::stride]
    Y_ds = Y[::stride, ::stride]

    fig = plt.figure(figsize=(18, 10))
    for i, plot_type in enumerate(plot_types):
        data, title, cmap = _get_wavefront_3d_field(state, plot_type, mask_threshold)
        Z_ds = data[::stride, ::stride]
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        surf = ax.plot_surface(
            X_ds,
            Y_ds,
            Z_ds,
            cmap=cmap,
            linewidth=0,
            antialiased=True,
            alpha=0.9,
        )
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel(title)
        ax.set_title(title, fontsize=10)
        ax.view_init(elev=elevation, azim=azimuth)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.08)

    title_lines = _format_surface_info(surface, position)
    fig.suptitle("\n".join(title_lines[:2]), fontsize=12)
    footer_lines = title_lines[2:]
    footer_lines.extend(_format_axis_info(axis_state))
    footer_lines.extend(_format_grid_info(state, mask_threshold=mask_threshold))
    plt.tight_layout()
    _add_footer(fig, footer_lines)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    plt.close(fig)
    return fig


def _set_equal_3d_axes(ax, points: np.ndarray) -> None:
    if points.size == 0:
        return
    finite_mask = np.isfinite(points).all(axis=1)
    if not np.any(finite_mask):
        return
    points = points[finite_mask]
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    ranges = maxs - mins
    max_range = float(np.max(ranges)) if np.any(ranges) else 1.0
    mid = (mins + maxs) / 2.0
    ax.set_xlim(mid[0] - max_range / 2.0, mid[0] + max_range / 2.0)
    ax.set_ylim(mid[1] - max_range / 2.0, mid[1] + max_range / 2.0)
    ax.set_zlim(mid[2] - max_range / 2.0, mid[2] + max_range / 2.0)


def plot_optical_axis_3d(
    surface_records: Sequence[SurfaceRecord],
    source_position: Optional[Sequence[float]] = (0.0, 0.0, 0.0),
    save_path: Optional[str | Path] = None,
    show: bool = True,
    dpi: int = 200,
    show_frames: bool = True,
    frame_scale: Optional[float] = None,
    frame_alpha: float = 0.7,
    annotate: bool = True,
    annotate_mode: str = "index",
    path_color: str = "tab:orange",
    point_color: str = "tab:blue",
    title: Optional[str] = None,
    axis_order: str = "zyx",
    show_optiland: bool = False,
    optiland_wavelength_um: Optional[float] = None,
    optiland_entrance_pupil_diameter: float = 10.0,
):
    import matplotlib.pyplot as plt

    _configure_matplotlib_fonts(plt)
    positions: list[np.ndarray] = []
    labels: list[str] = []
    axis_states: list[Optional[OpticalAxisState]] = []

    if source_position is not None:
        source_pos = np.asarray(source_position, dtype=float)
        positions.append(source_pos)
        labels.append("Source")
        axis_states.append(None)

    for record in surface_records:
        axis_state = record.exit_axis or record.entrance_axis
        if axis_state is None:
            continue
        positions.append(np.asarray(axis_state.position, dtype=float))
        axis_states.append(axis_state)
        if annotate:
            mode = annotate_mode.lower().strip()
            if mode == "name":
                label = record.name
            elif mode == "both":
                label = f"S{record.index}: {record.name}"
            else:
                label = f"S{record.index}"
            labels.append(label)

    if not positions:
        return None

    axis_order = axis_order.lower().strip()
    if len(axis_order) != 3 or set(axis_order) != {"x", "y", "z"}:
        raise ValueError("axis_order must be a permutation of 'xyz'")
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_idx = [axis_map[axis_name] for axis_name in axis_order]

    points = np.vstack(positions)
    points_plot = points[:, axis_idx]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(points_plot[:, 0], points_plot[:, 1], points_plot[:, 2], color=path_color, linewidth=1.5)
    ax.scatter(points_plot[:, 0], points_plot[:, 1], points_plot[:, 2], color=point_color, s=28)

    if annotate and labels:
        for point, label in zip(points_plot, labels):
            ax.text(point[0], point[1], point[2], label, fontsize=8, color="black")

    if show_frames:
        if frame_scale is None:
            if points.shape[0] >= 2:
                diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
                diffs = diffs[diffs > 1e-9]
                base = float(np.median(diffs)) if diffs.size else 1.0
                frame_scale = max(1.0, base * 0.15)
            else:
                frame_scale = 1.0
        for axis_state in axis_states:
            if axis_state is None:
                continue
            origin = np.asarray(axis_state.position, dtype=float)
            frame = np.asarray(axis_state.frame, dtype=float)
            origin_plot = origin[axis_idx]
            x_axis = frame[:, 0] * frame_scale
            y_axis = frame[:, 1] * frame_scale
            z_axis = frame[:, 2] * frame_scale
            x_axis_plot = x_axis[axis_idx]
            y_axis_plot = y_axis[axis_idx]
            z_axis_plot = z_axis[axis_idx]
            ax.quiver(
                origin_plot[0],
                origin_plot[1],
                origin_plot[2],
                x_axis_plot[0],
                x_axis_plot[1],
                x_axis_plot[2],
                color="tab:red",
                alpha=frame_alpha,
                arrow_length_ratio=0.18,
                linewidth=1.0,
            )
            ax.quiver(
                origin_plot[0],
                origin_plot[1],
                origin_plot[2],
                y_axis_plot[0],
                y_axis_plot[1],
                y_axis_plot[2],
                color="tab:green",
                alpha=frame_alpha,
                arrow_length_ratio=0.18,
                linewidth=1.0,
            )
            ax.quiver(
                origin_plot[0],
                origin_plot[1],
                origin_plot[2],
                z_axis_plot[0],
                z_axis_plot[1],
                z_axis_plot[2],
                color="tab:blue",
                alpha=frame_alpha,
                arrow_length_ratio=0.18,
                linewidth=1.0,
            )

    axis_labels = {"x": "X", "y": "Y", "z": "Z"}
    ax.set_xlabel(f"{axis_labels[axis_order[0]]} (mm)")
    ax.set_ylabel(f"{axis_labels[axis_order[1]]} (mm)")
    ax.set_zlabel(f"{axis_labels[axis_order[2]]} (mm)")
    ax.set_title(title or "Optical Axis (Recentered, 3D)")
    _set_equal_3d_axes(ax, points_plot)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show_optiland:
        try:
            from pop.io.zmx import to_optiland
            from sequential_system.zmx_visualization import view_3d

            surfaces = [record.surface for record in surface_records]
            if surfaces:
                wavelength_um = optiland_wavelength_um
                if wavelength_um is None:
                    for record in surface_records:
                        state = record.exit or record.entrance
                        if state is None:
                            continue
                        pilot = getattr(state, "pilot_beam_params", None)
                        wavelength_um = getattr(pilot, "wavelength_um", None) if pilot else None
                        if wavelength_um is not None:
                            wavelength_um = float(wavelength_um)
                            break
                if wavelength_um is None:
                    wavelength_um = 0.633
                optic = to_optiland(
                    surfaces,
                    wavelength_um=wavelength_um,
                    entrance_pupil_diameter=optiland_entrance_pupil_diameter,
                )
                view_3d(optic)
        except Exception as exc:
            print(f"错误: 无法使用 optiland 3D 可视化 ({exc})")
    if show:
        plt.show()
        return None
    plt.close(fig)
    return fig


def plot_refit_diagnostics(
    phase_before: NDArray[np.floating],
    phase_after: NDArray[np.floating],
    pilot_phase_before: NDArray[np.floating],
    pilot_phase_after: NDArray[np.floating],
    amplitude: NDArray[np.floating],
    grid_sampling: GridSampling,
    old_pilot: PilotBeamParams,
    new_pilot: PilotBeamParams,
    surface_index: int,
    save_path: Optional[str | Path] = None,
    show: bool = True,
    cutoff_percent: float = 1.0,
):
    """Plot diagnostics for Pilot Beam Refit.
    
    Shows:
    1. Phase Residual Comparison (Before vs After)
    2. Pilot Phase Comparison (Before vs After)
    3. Amplitude/Intensity Profile
    4. Text Info: R, w, PV change
    """
    print(f"!!! EXECUTING PLOT_REFIT_DIAGNOSTICS for Surface {surface_index} !!!")
    _configure_matplotlib_fonts(plt)
    
    grid_size = grid_sampling.grid_size
    physical_mm = grid_sampling.physical_size_mm
    coords = (np.arange(grid_size) - grid_size // 2) * grid_sampling.sampling_mm
    extent = [-physical_mm/2.0, physical_mm/2.0, -physical_mm/2.0, physical_mm/2.0]
    
    # Calculate Residuals
    # Residual = Phase_Total - Pilot_Phase
    
    # Phase before is TOTAL physical phase
    res_before = np.angle(np.exp(1j * (phase_before - pilot_phase_before)))
    # Phase after is TOTAL physical phase (should be same as before physically, but numerically maybe slightly different due to reconstruction if we re-propagated? No, here we just compare against new pilot)
    res_after = np.angle(np.exp(1j * (phase_after - pilot_phase_after)))
    
    res_before_waves = res_before / (2.0*np.pi)
    res_after_waves = res_after / (2.0*np.pi)
    
    # Masking for stats
    amp_max = np.max(amplitude)
    mask = amplitude > (cutoff_percent / 100.0) * amp_max
    
    pv_before = np.ptp(res_before_waves[mask]) if np.any(mask) else 0.0
    rms_before = np.std(res_before_waves[mask]) if np.any(mask) else 0.0
    pv_after = np.ptp(res_after_waves[mask]) if np.any(mask) else 0.0
    rms_after = np.std(res_after_waves[mask]) if np.any(mask) else 0.0
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    
    # Row 1: Before
    _plot_im(axes[0,0], pilot_phase_before / (2.0*np.pi), f"Old Pilot Phase\nR={old_pilot.curvature_radius_mm:.1f}mm", extent, "RdBu")
    im_res_pre = _plot_im(axes[0,1], res_before_waves, f"Residual BEFORE\nPV={pv_before:.3f}w, RMS={rms_before:.3f}w", extent, "RdBu_r")
    
    # Hist Before
    if np.any(mask):
        axes[0,2].hist(res_before_waves[mask], bins=50, alpha=0.7, color='red', label='Before')
        axes[0,2].grid(True, alpha=0.3)
    axes[0,2].set_title(f"Residual Hist Before")
    axes[0,2].set_xlabel("Waves")
    
    # Intensity
    _plot_im(axes[0,3], amplitude**2, "Intensity Profile", extent, "hot")

    # Row 2: After
    _plot_im(axes[1,0], pilot_phase_after / (2.0*np.pi), f"New Pilot Phase\nR={new_pilot.curvature_radius_mm:.1f}mm", extent, "RdBu")
    
    # Use same vmin/vmax for residual comparison context if possible
    try:
        clim = im_res_pre.get_clim()
        if np.any(mask):
            vmin = min(clim[0], np.min(res_after_waves[mask]))
            vmax = max(clim[1], np.max(res_after_waves[mask]))
        else:
            vmin, vmax = clim
    except:
        vmin, vmax = None, None
    
    _plot_im(axes[1,1], res_after_waves, f"Residual AFTER\nPV={pv_after:.3f}w, RMS={rms_after:.3f}w", extent, "RdBu_r", vmin=vmin, vmax=vmax)
    
    # Hist After
    if np.any(mask):
        axes[1,2].hist(res_after_waves[mask], bins=50, alpha=0.7, color='green', label='After')
        axes[1,2].grid(True, alpha=0.3)
    axes[1,2].set_title(f"Residual Hist After")
    axes[1,2].set_xlabel("Waves")
    
    # Difference Map (Old Pilot - New Pilot)
    diff_pilot = (pilot_phase_after - pilot_phase_before) / (2.0*np.pi)
    _plot_im(axes[1,3], diff_pilot, "Pilot Correction\n(New - Old)", extent, "viridis")

    
    # Overall Title
    fig.suptitle(f"Pilot Beam Refit Diagnostics - Surface {surface_index}\n"
                 f"R: {old_pilot.curvature_radius_mm:.2f} -> {new_pilot.curvature_radius_mm:.2f} mm | "
                 f"w: {old_pilot.spot_size_mm:.3f} -> {new_pilot.spot_size_mm:.3f} mm", fontsize=14)
                 
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig

def plot_resample_debug(
    phase_before: NDArray[np.floating],
    phase_after: NDArray[np.floating],
    amplitude_before: NDArray[np.floating],
    amplitude_after: NDArray[np.floating],
    grid_sampling_before: GridSampling,
    grid_sampling_after: GridSampling,
    pilot_beam: PilotBeamParams,
    surface_index: int,
    position: str = "exit",
    mag: float = 1.0,
    beam_pixels_before: float = 0.0,
    beam_pixels_after: float = 0.0,
    save_path: Optional[str | Path] = None,
    show: bool = False,
    dpi: int = 200,
):
    """绘制 resample 前后的完整 debug 对比图。

    包含：
    - resample 前后的原始相位（total phase）2D 分布
    - resample 前后的残差相位（total - pilot）2D 分布
    - resample 前后的振幅 2D 分布
    - 中心截面对比曲线（相位 + 振幅）
    - 网格参数标注（physical_size, sampling, beam_pixels, mag）
    """
    _configure_matplotlib_fonts(plt)

    gs_b = grid_sampling_before
    gs_a = grid_sampling_after
    n_b = gs_b.grid_size
    n_a = gs_a.grid_size
    half_b = gs_b.physical_size_mm / 2.0
    half_a = gs_a.physical_size_mm / 2.0
    extent_b = [-half_b, half_b, -half_b, half_b]
    extent_a = [-half_a, half_a, -half_a, half_a]

    # pilot 相位（分别在两个网格上计算）
    pilot_phase_b = pilot_beam.compute_phase_grid(n_b, gs_b.physical_size_mm)
    pilot_phase_a = pilot_beam.compute_phase_grid(n_a, gs_a.physical_size_mm)

    # 残差相位 = total - pilot（单位：waves）
    res_b = (phase_before - pilot_phase_b) / (2.0 * np.pi)
    res_a = (phase_after - pilot_phase_a) / (2.0 * np.pi)

    # total 相位（单位：waves）
    total_b_waves = phase_before / (2.0 * np.pi)
    total_a_waves = phase_after / (2.0 * np.pi)

    # 有效区域掩膜
    amp_max_b = np.max(amplitude_before) if amplitude_before.size > 0 else 1.0
    amp_max_a = np.max(amplitude_after) if amplitude_after.size > 0 else 1.0
    mask_b = amplitude_before > 0.01 * amp_max_b
    mask_a = amplitude_after > 0.01 * amp_max_a

    # 统计量
    def _stats(arr, mask):
        if np.any(mask):
            v = arr[mask]
            return float(np.ptp(v)), float(np.std(v))
        return 0.0, 0.0

    pv_res_b, rms_res_b = _stats(res_b, mask_b)
    pv_res_a, rms_res_a = _stats(res_a, mask_a)
    pv_tot_b, rms_tot_b = _stats(total_b_waves, mask_b)
    pv_tot_a, rms_tot_a = _stats(total_a_waves, mask_a)

    # ---- 布局：4 行 × 3 列 ----
    # Row 0: 振幅 before / 振幅 after / 振幅中心截面对比
    # Row 1: total phase before / total phase after / total phase 中心截面对比
    # Row 2: residual phase before / residual phase after / residual 中心截面对比
    # Row 3: pilot phase before / pilot phase after / 参数信息文本
    fig, axes = plt.subplots(4, 3, figsize=(18, 22))
    plt.subplots_adjust(hspace=0.38, wspace=0.32)

    # --- Row 0: 振幅 ---
    _plot_im(axes[0, 0], amplitude_before, "Resample 前 振幅", extent_b, "viridis")
    _plot_im(axes[0, 1], amplitude_after, "Resample 后 振幅", extent_a, "viridis")

    coords_b = (np.arange(n_b) - n_b // 2) * gs_b.sampling_mm
    coords_a = (np.arange(n_a) - n_a // 2) * gs_a.sampling_mm
    center_b = n_b // 2
    center_a = n_a // 2
    ax_sl = axes[0, 2]
    ax_sl.plot(coords_b, amplitude_before[center_b, :], label="Before", color="tab:blue")
    ax_sl.plot(coords_a, amplitude_after[center_a, :], label="After", color="tab:orange", linestyle="--")
    ax_sl.set_title("振幅中心截面对比 (y=0)")
    ax_sl.set_xlabel("X (mm)")
    ax_sl.set_ylabel("Amplitude")
    ax_sl.legend()
    ax_sl.grid(True, alpha=0.3)

    # --- Row 1: Total Phase ---
    _plot_im(
        axes[1, 0], total_b_waves,
        f"Resample 前 Total Phase (waves)\nPV={pv_tot_b:.4f}, RMS={rms_tot_b:.4f}",
        extent_b, "RdBu",
    )
    _plot_im(
        axes[1, 1], total_a_waves,
        f"Resample 后 Total Phase (waves)\nPV={pv_tot_a:.4f}, RMS={rms_tot_a:.4f}",
        extent_a, "RdBu",
    )
    ax_sl2 = axes[1, 2]
    ax_sl2.plot(coords_b, total_b_waves[center_b, :], label="Before", color="tab:blue")
    ax_sl2.plot(coords_a, total_a_waves[center_a, :], label="After", color="tab:orange", linestyle="--")
    ax_sl2.set_title("Total Phase 中心截面对比 (y=0, waves)")
    ax_sl2.set_xlabel("X (mm)")
    ax_sl2.set_ylabel("Phase (waves)")
    ax_sl2.legend()
    ax_sl2.grid(True, alpha=0.3)

    # --- Row 2: Residual Phase ---
    _plot_im(
        axes[2, 0], res_b,
        f"Resample 前 Residual Phase (waves)\nPV={pv_res_b:.4f}, RMS={rms_res_b:.4f}",
        extent_b, "RdBu_r",
    )
    _plot_im(
        axes[2, 1], res_a,
        f"Resample 后 Residual Phase (waves)\nPV={pv_res_a:.4f}, RMS={rms_res_a:.4f}",
        extent_a, "RdBu_r",
    )
    ax_sl3 = axes[2, 2]
    ax_sl3.plot(coords_b, res_b[center_b, :], label="Before", color="tab:blue")
    ax_sl3.plot(coords_a, res_a[center_a, :], label="After", color="tab:orange", linestyle="--")
    ax_sl3.set_title("Residual Phase 中心截面对比 (y=0, waves)")
    ax_sl3.set_xlabel("X (mm)")
    ax_sl3.set_ylabel("Residual (waves)")
    ax_sl3.legend()
    ax_sl3.grid(True, alpha=0.3)

    # --- Row 3: Pilot Phase + 参数信息 ---
    pilot_b_waves = pilot_phase_b / (2.0 * np.pi)
    pilot_a_waves = pilot_phase_a / (2.0 * np.pi)
    _plot_im(axes[3, 0], pilot_b_waves, "Resample 前 Pilot Phase (waves)", extent_b, "RdBu")
    _plot_im(axes[3, 1], pilot_a_waves, "Resample 后 Pilot Phase (waves)", extent_a, "RdBu")

    # 参数信息文本面板
    ax_info = axes[3, 2]
    ax_info.set_axis_off()
    info_lines = [
        f"Surface {surface_index} ({position})",
        "",
        "── Resample 前 ──",
        f"  grid_size = {n_b}",
        f"  physical_size = {gs_b.physical_size_mm:.4f} mm",
        f"  sampling = {gs_b.sampling_mm:.6f} mm/px",
        f"  beam_pixels ≈ {beam_pixels_before:.1f}",
        "",
        "── Resample 后 ──",
        f"  grid_size = {n_a}",
        f"  physical_size = {gs_a.physical_size_mm:.4f} mm",
        f"  sampling = {gs_a.sampling_mm:.6f} mm/px",
        f"  beam_pixels ≈ {beam_pixels_after:.1f}",
        "",
        f"magnification = {mag:.3f}",
        "",
        "── Pilot Beam ──",
        f"  w = {pilot_beam.spot_size_mm:.4f} mm",
        f"  R = {_format_value(pilot_beam.curvature_radius_mm)} mm",
        f"  w0 = {pilot_beam.waist_radius_mm:.4f} mm",
        f"  λ = {pilot_beam.wavelength_um:.4f} μm",
    ]
    ax_info.text(
        0.05, 0.95, "\n".join(info_lines),
        transform=ax_info.transAxes,
        fontsize=10, verticalalignment="top", fontfamily="monospace",
        bbox=dict(facecolor="#f8f8f8", alpha=0.9, pad=8),
    )

    fig.suptitle(
        f"Resample Debug — Surface {surface_index} ({position})\n"
        f"beam_pixels: {beam_pixels_before:.1f} → {beam_pixels_after:.1f}  |  "
        f"mag = {mag:.3f}  |  "
        f"sampling: {gs_b.sampling_mm:.6f} → {gs_a.sampling_mm:.6f} mm/px",
        fontsize=13,
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[POP][Resample Debug] 已保存: {save_path}")
    if show:
        plt.show()
        return None
    plt.close(fig)
    return fig



def plot_element_refit_diagnostics(
    ray_x: NDArray[np.floating],
    ray_y: NDArray[np.floating],
    ray_intensity: NDArray[np.floating],
    absolute_opd_waves: NDArray[np.floating],
    pilot_opd_before: NDArray[np.floating],
    residual_opd_before: NDArray[np.floating],
    pilot_opd_after: NDArray[np.floating],
    residual_opd_after: NDArray[np.floating],
    old_pilot: "PilotBeamParams",
    new_pilot: "PilotBeamParams",
    surface_index: int,
    save_path: Optional[str | Path] = None,
    show: bool = False,
    dpi: int = 150,
):
    """元件路径 Pilot Beam Refit 的独立诊断绘图。

    不依赖 debug 模式，只要 refit 发生就可以调用。
    显示：refit 前后的 pilot OPD、残差 OPD 散点图 + 直方图 + 参数信息。
    """
    _configure_matplotlib_fonts(plt)

    finite = (
        np.isfinite(ray_x) & np.isfinite(ray_y)
        & np.isfinite(absolute_opd_waves)
        & np.isfinite(ray_intensity)
    )
    amp = np.sqrt(np.maximum(ray_intensity, 0.0))
    amp_max = np.max(amp[finite]) if np.any(finite) else 1.0
    mask = finite & (amp > 0.01 * amp_max)

    def _pv_rms(arr):
        v = arr[mask] if np.any(mask) else arr
        return float(np.ptp(v)), float(np.std(v))

    pv_b, rms_b = _pv_rms(residual_opd_before)
    pv_a, rms_a = _pv_rms(residual_opd_after)

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.35)

    def _scat(ax, data, title, cmap="viridis"):
        if not np.any(mask):
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            return
        sc = ax.scatter(
            ray_x[mask], ray_y[mask], c=data[mask],
            s=6, cmap=cmap, alpha=0.8,
        )
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    # Row 0: refit 前
    _scat(axes[0, 0], pilot_opd_before, "Refit 前: Pilot OPD (waves)")
    _scat(axes[0, 1], residual_opd_before,
          f"Refit 前: 残差 OPD\nPV={pv_b:.4f}w  RMS={rms_b:.4f}w", "RdBu_r")
    if np.any(mask):
        axes[0, 2].hist(residual_opd_before[mask], bins=50,
                        color="tab:red", alpha=0.7)
    axes[0, 2].set_title("Refit 前 残差直方图")
    axes[0, 2].set_xlabel("Residual OPD (waves)")
    axes[0, 2].grid(True, alpha=0.3)
    _scat(axes[0, 3], absolute_opd_waves, "绝对 OPD (waves)")

    # Row 1: refit 后
    _scat(axes[1, 0], pilot_opd_after, "Refit 后: Pilot OPD (waves)")
    _scat(axes[1, 1], residual_opd_after,
          f"Refit 后: 残差 OPD\nPV={pv_a:.4f}w  RMS={rms_a:.4f}w", "RdBu_r")
    if np.any(mask):
        axes[1, 2].hist(residual_opd_after[mask], bins=50,
                        color="tab:green", alpha=0.7)
    axes[1, 2].set_title("Refit 后 残差直方图")
    axes[1, 2].set_xlabel("Residual OPD (waves)")
    axes[1, 2].grid(True, alpha=0.3)

    # 参数信息面板
    ax_info = axes[1, 3]
    ax_info.set_axis_off()
    info_lines = [
        f"Surface {surface_index}  (Element Refit)",
        "",
        "── Refit 前 ──",
        f"  R  = {_format_value(old_pilot.curvature_radius_mm)} mm",
        f"  w  = {old_pilot.spot_size_mm:.4f} mm",
        f"  w0 = {old_pilot.waist_radius_mm:.4f} mm",
        f"  PV = {pv_b:.4f} waves",
        f"  RMS= {rms_b:.4f} waves",
        "",
        "── Refit 后 ──",
        f"  R  = {_format_value(new_pilot.curvature_radius_mm)} mm",
        f"  w  = {new_pilot.spot_size_mm:.4f} mm",
        f"  w0 = {new_pilot.waist_radius_mm:.4f} mm",
        f"  PV = {pv_a:.4f} waves",
        f"  RMS= {rms_a:.4f} waves",
    ]
    ax_info.text(
        0.05, 0.95, "\n".join(info_lines),
        transform=ax_info.transAxes, fontsize=9,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(facecolor="#f8f8f8", alpha=0.9, pad=8),
    )

    fig.suptitle(
        f"Element Refit Diagnostics — Surface {surface_index}\n"
        f"R: {_format_value(old_pilot.curvature_radius_mm)} → "
        f"{_format_value(new_pilot.curvature_radius_mm)} mm  |  "
        f"w: {old_pilot.spot_size_mm:.4f} → {new_pilot.spot_size_mm:.4f} mm  |  "
        f"PV: {pv_b:.4f} → {pv_a:.4f} waves",
        fontsize=12,
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[POP][Element Refit] 诊断图已保存: {save_path}")
    if show:
        plt.show()
        return None
    plt.close(fig)
    return fig


__all__ = [
    "plot_wavefront_analysis",
    "plot_surface_debug",
    "plot_surface_raytrace_3d",
    "plot_wavefront_3d",
    "plot_surface_detail_3d",
    "plot_optical_axis_3d",
    "plot_refit_diagnostics",
    "plot_resample_debug",
    "plot_element_refit_diagnostics",
]
