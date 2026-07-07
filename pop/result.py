"""Propagation result utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

from pop.analysis import compute_state_moments
from pop.core import OpticalAxisState, PropagationState, SurfaceInteractionInfo
from pop.utils import resolve_output_dir


@dataclass
class SurfaceDebugInfo:
    """Debug data captured for one surface."""

    surface_index: int
    entrance_rays_local: Optional[dict[str, np.ndarray]] = None
    entrance_rays_global: Optional[dict[str, np.ndarray]] = None
    surface_rays_global: Optional[dict[str, np.ndarray]] = None
    exit_rays_local: Optional[dict[str, np.ndarray]] = None
    exit_rays_global: Optional[dict[str, np.ndarray]] = None
    input_amplitude: Optional[np.ndarray] = None
    absolute_opd_waves: Optional[np.ndarray] = None
    pilot_opd_waves: Optional[np.ndarray] = None
    residual_opd_waves: Optional[np.ndarray] = None
    residual_phase_grid: Optional[np.ndarray] = None
    reconstruction_mask: Optional[np.ndarray] = None
    pilot_refit_info: Optional[dict[str, object]] = None  # before/after pilot params when refit occurred
    pilot_opd_waves_pre_refit: Optional[np.ndarray] = None
    residual_opd_waves_pre_refit: Optional[np.ndarray] = None
    refit_occurred: bool = False


@dataclass
class SurfaceRecord:
    """Per-surface container for entrance/exit states and debug info."""

    index: int
    surface: Any
    entrance: Optional[PropagationState]
    exit: Optional[PropagationState]
    entrance_axis: Optional[OpticalAxisState]
    exit_axis: Optional[OpticalAxisState]
    interaction_info: Optional[SurfaceInteractionInfo] = None
    surface_overrides: Optional[dict[str, object]] = None
    debug: Optional[SurfaceDebugInfo] = None

    @property
    def name(self) -> str:
        comment = getattr(self.surface, "comment", "") or ""
        if comment.strip():
            return comment.strip()
        surface_type = getattr(self.surface, "surface_type", "") or ""
        if surface_type:
            return surface_type
        return f"Surface {self.index}"

    def get_state(self, position: str) -> Optional[PropagationState]:
        position = position.lower().strip()
        if position == "entrance":
            return self.entrance
        if position == "exit":
            return self.exit
        raise ValueError("position must be 'entrance' or 'exit'")


@dataclass
class PropagationResult:
    """Propagation result container."""

    final_state: PropagationState
    surface_states: List[PropagationState]
    total_path_length: float
    success: bool = True
    error_message: str = ""
    surfaces: List[SurfaceRecord] = field(default_factory=list)

    def get_final_wavefront(self) -> np.ndarray:
        return self.final_state.get_complex_amplitude()

    def get_final_amplitude(self) -> np.ndarray:
        return self.final_state.amplitude

    def get_final_intensity(self) -> np.ndarray:
        return self.final_state.get_intensity()

    def get_final_phase(self) -> np.ndarray:
        return self.final_state.get_phase()

    def plot(self) -> None:
        import matplotlib.pyplot as plt

        amplitude = self.get_final_amplitude()
        phase = self.get_final_phase()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(amplitude, cmap="viridis")
        axes[0].set_title("Amplitude")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(phase, cmap="twilight")
        axes[1].set_title("Phase")
        plt.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        plt.show()

    def summary(self) -> str:
        return (
            f"PropagationResult(success={self.success}, "
            f"surfaces={len(self.surface_states)}, "
            f"total_path_length_mm={self.total_path_length:.6f})"
        )

    def summarize_surfaces(
        self,
        positions: Optional[Sequence[str] | str] = None,
        mask_threshold: float = 0.01,
    ) -> list[dict[str, object]]:
        if positions is None:
            position_list = ["exit"]
        elif isinstance(positions, str):
            pos = positions.lower().strip()
            if pos in ("both", "all"):
                position_list = ["entrance", "exit"]
            else:
                position_list = [pos]
        else:
            position_list = [p.lower().strip() for p in positions]

        summaries: list[dict[str, object]] = []
        for record in self.surfaces:
            for position in position_list:
                try:
                    state = record.get_state(position)
                except ValueError:
                    continue
                if state is None:
                    continue
                axis_state = record.entrance_axis if position == "entrance" else record.exit_axis
                moments = compute_state_moments(state, mask_threshold=mask_threshold)
                grid = state.grid_sampling
                pilot = state.pilot_beam_params
                intensity = np.asarray(state.amplitude, dtype=float) ** 2
                pixel_area = float(grid.sampling_mm) ** 2
                energy = float(np.sum(intensity) * pixel_area) if intensity.size else 0.0
                summaries.append(
                    {
                        "surface_index": record.index,
                        "surface_name": record.name,
                        "position": position,
                        "surface_type": getattr(record.surface, "surface_type", None),
                        "material": getattr(record.surface, "material", None),
                        "is_mirror": bool(getattr(record.surface, "is_mirror", False)),
                        "algorithm": state.propagation_algorithm,
                        "grid_size": grid.grid_size,
                        "sampling_mm": grid.sampling_mm,
                        "physical_size_mm": grid.physical_size_mm,
                        "pilot_w_mm": pilot.spot_size_mm,
                        "pilot_r_mm": pilot.curvature_radius_mm,
                        "pilot_z_waist_mm": pilot.waist_position_mm,
                        "pilot_w0_mm": pilot.waist_radius_mm,  # w0 (Waist Radius)
                        "pilot_zr_mm": pilot.rayleigh_length_mm, # z_R (Rayleigh Length)
                        "centroid_x": moments.get("centroid_x") if moments else np.nan,
                        "centroid_y": moments.get("centroid_y") if moments else np.nan,
                        "sigma_x": moments.get("sigma_x") if moments else np.nan,
                        "sigma_y": moments.get("sigma_y") if moments else np.nan,
                        "energy": energy,
                        "path_length_mm": getattr(axis_state, "path_length", None) if axis_state else None,
                    }
                )
        return summaries

    def save_report(
        self,
        output_dir: Optional[str | Path] = None,
        title: Optional[str] = None,
        positions: Optional[Sequence[str] | str] = None,
        surface_indices: Optional[Sequence[int]] = None,
        plot_set: str = "basic",
        include_visualizations: bool = True,
        include_layout: bool = True,
        include_axis_3d: bool = False,
        show: bool = False,
        dpi: int = 200,
        mask_threshold: float = 0.01,
    ) -> Path:
        base_dir = resolve_output_dir(output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        if include_visualizations:
            self.save_visualizations(
                mode="normal",
                output_dir=base_dir,
                show=show,
                dpi=dpi,
                surface_indices=surface_indices,
                positions=positions,
                plot_set=plot_set,
                mask_threshold=mask_threshold,
            )

        layout_paths: list[Path] = []
        if include_layout and self.surfaces:
            try:
                from pop.system import System

                sys_obj = System(name=title or "POP Report", surfaces=[s.surface for s in self.surfaces])
                layout_path = base_dir / "layout_2d.png"
                sys_obj.plot_layout(
                    mode="2d",
                    projection="YZ",
                    save_path=str(layout_path),
                    show=show,
                    annotate=True,
                    show_optical_axis=True,
                )
                layout_paths.append(layout_path)
            except Exception:
                pass

        if include_axis_3d and self.surfaces:
            axis_path = base_dir / "optical_axis_3d.png"
            self.plot_optical_axis_3d(save_path=axis_path, show=show, dpi=dpi)
            layout_paths.append(axis_path)

        summaries = self.summarize_surfaces(
            positions=positions,
            mask_threshold=mask_threshold,
        )
        if surface_indices is not None:
            surface_set = set(surface_indices)
            summaries = [s for s in summaries if s["surface_index"] in surface_set]

        def _fmt(value: object, precision: int = 4) -> str:
            if value is None:
                return ""
            try:
                val = float(value)
            except (TypeError, ValueError):
                return str(value)
            if np.isnan(val):
                return ""
            if np.isinf(val):
                return "inf"
            return f"{val:.{precision}f}"

        header = [
            "Surface",
            "Pos",
            "Name",
            "Type",
            "Material",
            "Alg",
            "Grid",
            "dx(mm)",
            "w_pilot(mm)",
            "R_pilot(mm)",
            "w0(mm)",
            "zR(mm)",
            "Centroid(mm)",
            "Sigma(mm)",
        ]
        table_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
        for row in summaries:
            table_lines.append(
                "| "
                + " | ".join(
                    [
                        f"S{row['surface_index']}",
                        str(row["position"]),
                        str(row["surface_name"]),
                        str(row.get("surface_type", "")),
                        str(row.get("material", "")),
                        str(row.get("algorithm", "")),
                        str(row.get("grid_size", "")),
                        _fmt(row.get("sampling_mm")),
                        _fmt(row.get("pilot_w_mm")),
                        _fmt(row.get("pilot_r_mm")),
                        _fmt(row.get("pilot_w0_mm")), # Added
                        _fmt(row.get("pilot_zr_mm")), # Added
                        f"({_fmt(row.get('centroid_x'))},{_fmt(row.get('centroid_y'))})",
                        f"({_fmt(row.get('sigma_x'))},{_fmt(row.get('sigma_y'))})",
                    ]
                )
                + " |"
            )

        image_files = sorted(base_dir.glob("*.png"))
        md_lines = [
            f"# {title or 'POP Propagation Report'}",
            "",
            "## Summary",
            "",
            *table_lines,
            "",
        ]
        if layout_paths:
            md_lines.append("## Layout")
            md_lines.append("")
            for path in layout_paths:
                md_lines.append(f"![{path.name}]({path.name})")
            md_lines.append("")
        if image_files:
            md_lines.append("## Figures")
            md_lines.append("")
            for path in image_files:
                if path in layout_paths:
                    continue
                md_lines.append(f"![{path.name}]({path.name})")
            md_lines.append("")

        md_path = base_dir / "index.md"
        md_path.write_text("\n".join(md_lines), encoding="utf-8")

        html_lines = [
            "<html><head><meta charset='utf-8'>",
            f"<title>{title or 'POP Propagation Report'}</title>",
            "<style>body{font-family:Arial,Helvetica,sans-serif;margin:20px;} table{border-collapse:collapse;} th,td{border:1px solid #ccc;padding:6px 8px;font-size:13px;} img{max-width:100%;}</style>",
            "</head><body>",
            f"<h1>{title or 'POP Propagation Report'}</h1>",
            "<h2>Summary</h2>",
            "<table>",
        ]
        html_lines.append("<tr>" + "".join(f"<th>{h}</th>" for h in header) + "</tr>")
        for row in summaries:
            html_lines.append(
                "<tr>"
                + "".join(
                    f"<td>{cell}</td>"
                    for cell in [
                        f"S{row['surface_index']}",
                        row["position"],
                        row["surface_name"],
                        row.get("surface_type", ""),
                        row.get("material", ""),
                        row.get("algorithm", ""),
                        row.get("grid_size", ""),
                        _fmt(row.get("sampling_mm")),
                        _fmt(row.get("pilot_w_mm")),
                        _fmt(row.get("pilot_r_mm")),
                        _fmt(row.get("pilot_w0_mm")), # Added
                        _fmt(row.get("pilot_zr_mm")), # Added
                        f"({_fmt(row.get('centroid_x'))},{_fmt(row.get('centroid_y'))})",
                        f"({_fmt(row.get('sigma_x'))},{_fmt(row.get('sigma_y'))})",
                    ]
                )
                + "</tr>"
            )
        html_lines.append("</table>")

        if layout_paths:
            html_lines.append("<h2>Layout</h2>")
            for path in layout_paths:
                html_lines.append(f"<img src='{path.name}' alt='{path.name}'/>")

        if image_files:
            html_lines.append("<h2>Figures</h2>")
            for path in image_files:
                if path in layout_paths:
                    continue
                html_lines.append(f"<img src='{path.name}' alt='{path.name}'/>")

        html_lines.append("</body></html>")
        html_path = base_dir / "index.html"
        html_path.write_text("\n".join(html_lines), encoding="utf-8")

        return md_path

    def get_surface(self, index: int) -> SurfaceRecord:
        for record in self.surfaces:
            if record.index == index:
                return record
        raise ValueError(f"Surface {index} not found in PropagationResult.surfaces")

    def plot_surface(
        self,
        index: int,
        position: str = "exit",
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
    ):
        surface = self.get_surface(index)
        state = surface.get_state(position)
        if state is None:
            raise ValueError(f"Surface {index} has no {position} state to plot")
        axis_state = surface.entrance_axis if position == "entrance" else surface.exit_axis
        from pop.visualization import plot_wavefront_analysis

        return plot_wavefront_analysis(
            state=state,
            surface=surface.surface,
            axis_state=axis_state,
            position=position,
            save_path=save_path,
            show=show,
            dpi=dpi,
            plot_set=plot_set,
            panels=panels,
            max_cols=max_cols,
            mask_threshold=mask_threshold,
            slice_axis=slice_axis,
            slice_index=slice_index,
            cmap_amp=cmap_amp,
            cmap_intensity=cmap_intensity,
            cmap_phase=cmap_phase,
            cmap_residual=cmap_residual,
        )

    def plot_surface_debug(
        self,
        index: int,
        save_dir: Optional[str | Path] = None,
        show: bool = True,
        dpi: int = 200,
        plots: Optional[list[str]] = None,
        filename_prefix: Optional[str] = None,
    ):
        surface = self.get_surface(index)
        from pop.visualization import plot_surface_debug

        return plot_surface_debug(
            surface_record=surface,
            save_dir=save_dir,
            show=show,
            dpi=dpi,
            plots=plots,
            filename_prefix=filename_prefix,
        )

    def plot_surface_raytrace_3d(
        self,
        index: int,
        save_path: Optional[str | Path] = None,
        show: bool = True,
        dpi: int = 200,
        max_rays: int = 5,
        elevation: float = 30.0,
        azimuth: float = -60.0,
    ):
        surface = self.get_surface(index)
        if surface.debug is None:
            raise ValueError(
                f"Surface {index} has no debug rays. Run propagate(..., debug=True) "
                f"or enable debug_plot_3d first."
            )
        from pop.visualization import plot_surface_raytrace_3d

        return plot_surface_raytrace_3d(
            surface=surface.surface,
            entrance_axis=surface.entrance_axis,
            exit_axis=surface.exit_axis,
            debug=surface.debug,
            surface_overrides=surface.surface_overrides,
            save_path=save_path,
            show=show,
            dpi=dpi,
            max_rays=max_rays,
            elevation=elevation,
            azimuth=azimuth,
        )

    def plot_surface_3d(
        self,
        index: int,
        position: str = "exit",
        plot_type: str = "residual_phase",
        save_path: Optional[str | Path] = None,
        show: bool = True,
        dpi: int = 200,
        elevation: float = 30.0,
        azimuth: float = -60.0,
        stride: Optional[int] = None,
        mask_threshold: float = 0.01,
    ):
        surface = self.get_surface(index)
        state = surface.get_state(position)
        if state is None:
            raise ValueError(f"Surface {index} has no {position} state to plot")
        axis_state = surface.entrance_axis if position == "entrance" else surface.exit_axis
        from pop.visualization import plot_wavefront_3d

        return plot_wavefront_3d(
            state=state,
            surface=surface.surface,
            axis_state=axis_state,
            position=position,
            plot_type=plot_type,
            save_path=save_path,
            show=show,
            dpi=dpi,
            elevation=elevation,
            azimuth=azimuth,
            stride=stride,
            mask_threshold=mask_threshold,
        )

    def plot_surface_detail_3d(
        self,
        index: int,
        position: str = "exit",
        save_path: Optional[str | Path] = None,
        show: bool = True,
        dpi: int = 200,
        elevation: float = 30.0,
        azimuth: float = -60.0,
        stride: Optional[int] = None,
        mask_threshold: float = 0.01,
    ):
        surface = self.get_surface(index)
        state = surface.get_state(position)
        if state is None:
            raise ValueError(f"Surface {index} has no {position} state to plot")
        axis_state = surface.entrance_axis if position == "entrance" else surface.exit_axis
        from pop.visualization import plot_surface_detail_3d

        return plot_surface_detail_3d(
            state=state,
            surface=surface.surface,
            axis_state=axis_state,
            position=position,
            save_path=save_path,
            show=show,
            dpi=dpi,
            elevation=elevation,
            azimuth=azimuth,
            stride=stride,
            mask_threshold=mask_threshold,
        )

    def plot_optical_axis_3d(
        self,
        save_path: Optional[str | Path] = None,
        show: bool = True,
        dpi: int = 200,
        show_frames: bool = True,
        frame_scale: Optional[float] = None,
        frame_alpha: float = 0.7,
        annotate: bool = True,
        annotate_mode: str = "index",
        source_position: Optional[Sequence[float]] = (0.0, 0.0, 0.0),
        path_color: str = "tab:orange",
        point_color: str = "tab:blue",
        title: Optional[str] = None,
        axis_order: str = "zyx",
        show_optiland: bool = False,
        optiland_wavelength_um: Optional[float] = None,
        optiland_entrance_pupil_diameter: float = 10.0,
    ):
        from pop.visualization import plot_optical_axis_3d

        return plot_optical_axis_3d(
            surface_records=self.surfaces,
            source_position=source_position,
            save_path=save_path,
            show=show,
            dpi=dpi,
            show_frames=show_frames,
            frame_scale=frame_scale,
            frame_alpha=frame_alpha,
            annotate=annotate,
            annotate_mode=annotate_mode,
            path_color=path_color,
            point_color=point_color,
            title=title,
            axis_order=axis_order,
            show_optiland=show_optiland,
            optiland_wavelength_um=optiland_wavelength_um,
            optiland_entrance_pupil_diameter=optiland_entrance_pupil_diameter,
        )

    def save_visualizations(
        self,
        mode: str = "all",
        output_dir: Optional[str | Path] = None,
        show: bool = False,
        dpi: int = 200,
        surface_indices: Optional[Sequence[int]] = None,
        positions: Optional[Sequence[str] | str] = None,
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
        debug_plots: Optional[list[str]] = None,
        include_3d: bool = False,
        plot_3d_set: str = "detail",
        plot_3d_types: Optional[Sequence[str]] = None,
        plot_3d_elevation: float = 30.0,
        plot_3d_azimuth: float = -60.0,
        plot_3d_stride: Optional[int] = None,
        plot_3d_mask_threshold: float = 0.01,
    ) -> Path:
        from pop.utils import resolve_output_dir

        base_dir = resolve_output_dir(output_dir)
        mode = mode.lower().strip()
        if mode not in ("normal", "debug", "all"):
            raise ValueError("mode must be 'normal', 'debug', or 'all'")

        if surface_indices is None:
            selected_surfaces = list(self.surfaces)
        else:
            index_set = set(surface_indices)
            selected_surfaces = [s for s in self.surfaces if s.index in index_set]

        if positions is None:
            position_list = ["entrance", "exit"]
        elif isinstance(positions, str):
            pos = positions.lower().strip()
            if pos in ("both", "all"):
                position_list = ["entrance", "exit"]
            else:
                position_list = [pos]
        else:
            position_list = [p.lower().strip() for p in positions]
        for pos in position_list:
            if pos not in ("entrance", "exit"):
                raise ValueError("positions must include only 'entrance' and/or 'exit'")

        if mode in ("normal", "all"):
            for surface in selected_surfaces:
                for position in position_list:
                    if position == "entrance" and surface.entrance is None:
                        continue
                    if position == "exit" and surface.exit is None:
                        continue
                    filename = f"surface_{surface.index:02d}_{position}_{plot_set}.png"
                    self.plot_surface(
                        surface.index,
                        position=position,
                        save_path=base_dir / filename,
                        show=show,
                        dpi=dpi,
                        plot_set=plot_set,
                        panels=panels,
                        max_cols=max_cols,
                        mask_threshold=mask_threshold,
                        slice_axis=slice_axis,
                        slice_index=slice_index,
                        cmap_amp=cmap_amp,
                        cmap_intensity=cmap_intensity,
                        cmap_phase=cmap_phase,
                        cmap_residual=cmap_residual,
                    )

        if mode in ("debug", "all"):
            for surface in selected_surfaces:
                if surface.debug is None:
                    continue
                self.plot_surface_debug(
                    surface.index,
                    save_dir=base_dir,
                    show=show,
                    dpi=dpi,
                    plots=debug_plots,
                    filename_prefix=f"surface_{surface.index:02d}",
                )

        if include_3d:
            plot_3d_set = plot_3d_set.lower().strip()
            if plot_3d_types is None:
                plot_3d_types_list = ["residual_phase"]
            else:
                plot_3d_types_list = [p.lower().strip() for p in plot_3d_types]
            for surface in selected_surfaces:
                for position in position_list:
                    if position == "entrance" and surface.entrance is None:
                        continue
                    if position == "exit" and surface.exit is None:
                        continue
                    if plot_3d_set in ("detail", "all"):
                        self.plot_surface_detail_3d(
                            surface.index,
                            position=position,
                            save_path=base_dir
                            / f"surface_{surface.index:02d}_{position}_detail_3d.png",
                            show=show,
                            dpi=dpi,
                            elevation=plot_3d_elevation,
                            azimuth=plot_3d_azimuth,
                            stride=plot_3d_stride,
                            mask_threshold=plot_3d_mask_threshold,
                        )
                    if plot_3d_set in ("single", "types", "all"):
                        for plot_type in plot_3d_types_list:
                            self.plot_surface_3d(
                                surface.index,
                                position=position,
                                plot_type=plot_type,
                                save_path=base_dir
                                / f"surface_{surface.index:02d}_{position}_{plot_type}_3d.png",
                                show=show,
                                dpi=dpi,
                                elevation=plot_3d_elevation,
                                azimuth=plot_3d_azimuth,
                                stride=plot_3d_stride,
                                mask_threshold=plot_3d_mask_threshold,
                            )

        return base_dir
