"""System definition for POP."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from pop.coordinates.transforms import rotation_matrix_to_euler
from pop.io.zmx import GlobalSurfaceDefinition, load_zmx, to_optiland


def _rotation_matrix_from_tilts(
    tilt_x_deg: float = 0.0,
    tilt_y_deg: float = 0.0,
    tilt_z_deg: float = 0.0,
) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Rot

    angles_rad = np.deg2rad([tilt_x_deg, tilt_y_deg, tilt_z_deg])
    return Rot.from_euler("xyz", angles_rad).as_matrix()


@dataclass
class System:
    """Optical system definition."""

    name: str = ""
    surfaces: List[GlobalSurfaceDefinition] = field(default_factory=list)

    @classmethod
    def from_zmx(cls, path: str, name: str = "") -> "System":
        surfaces = load_zmx(path)
        return cls(name=name, surfaces=surfaces)

    def to_script(self, filename: str) -> None:
        """Export system to Python script."""
        from pop.io.script import save_system_as_script
        save_system_as_script(self, filename)

    def add_mirror(
        self,
        z: float,
        radius: float = np.inf,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
        tilt_z: float = 0.0,
        semi_aperture: float = 0.0,
        conic: float = 0.0,
    ) -> "System":
        orientation = _rotation_matrix_from_tilts(tilt_x, tilt_y, tilt_z)
        vertex_position = np.array([0.0, 0.0, float(z)])
        surface = GlobalSurfaceDefinition(
            index=len(self.surfaces),
            surface_type="standard" if np.isfinite(radius) else "flat",
            vertex_position=vertex_position,
            orientation=orientation,
            radius=radius,
            conic=conic,
            is_mirror=True,
            semi_aperture=semi_aperture,
            material="mirror",
        )
        self.surfaces.append(surface)
        return self

    def add_lens(
        self,
        z: float,
        focal_length: float,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
        tilt_z: float = 0.0,
    ) -> "System":
        orientation = _rotation_matrix_from_tilts(tilt_x, tilt_y, tilt_z)
        vertex_position = np.array([0.0, 0.0, float(z)])
        surface = GlobalSurfaceDefinition(
            index=len(self.surfaces),
            surface_type="paraxial",
            vertex_position=vertex_position,
            orientation=orientation,
            radius=np.inf,
            conic=0.0,
            is_mirror=False,
            semi_aperture=0.0,
            material="air",
            focal_length=float(focal_length),
        )
        self.surfaces.append(surface)
        return self

    def plot_layout(
        self,
        mode: str = "2d",
        projection: str = "YZ",
        num_rays: int = 5,
        save_path: Optional[str] = None,
        show: bool = True,
        wavelength_um: float = 0.633,
        entrance_pupil_diameter: float = 10.0,
        annotate: bool = True,
        label_mode: str = "index",
        label_fields: Optional[list[str]] = None,
        font_size: int = 8,
        surface_indices: Optional[Sequence[int]] = None,
        show_normals: bool = False,
        show_normals_values: bool = False,
        show_incident_dirs: bool = False,
        normal_scale: Optional[float] = None,
        normal_color: str = "tab:purple",
        normal_alpha: float = 0.8,
        show_axes: bool = False,
        axis_labels: bool = False,
        axis_scale: Optional[float] = None,
        axis_alpha: float = 0.7,
        show_optical_axis: bool = False,
        optical_axis_color: str = "tab:red",
        incident_scale: Optional[float] = None,
        incident_color: str = "tab:orange",
        incident_alpha: float = 0.8,
        label_offset: Optional[float] = None,
        label_rotation: float = 0.0,
        label_box: bool = False,
        label_alpha: float = 0.9,
        label_avoid_overlap: bool = True,
        label_min_dist: Optional[float] = None,
        label_step: Optional[float] = None,
        annotate_3d: bool = False,
        annotate_3d_surface_ids: bool = True,
        annotate_3d_intersections: bool = True,
        annotate_3d_font_size: int = 12,
        backend: str = "auto",
    ) -> Optional[Tuple[Any, Any]]:
        """Plot optical layout (2D/3D), similar to BTS API.

        backend (3D only): "auto" (default), "optiland", or "mpl".
        """
        import matplotlib.pyplot as plt

        surfaces = self._select_surfaces(surface_indices)
        if len(surfaces) == 0:
            if mode.lower() == "2d":
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "系统为空", ha="center", va="center", fontsize=14)
                ax.set_title(f"光学系统: {self.name}")
                return fig, ax
            return None

        if mode.lower() == "3d":
            backend_norm = backend.lower().strip()
            if backend_norm not in ("auto", "optiland", "mpl"):
                raise ValueError("backend must be 'auto', 'optiland', or 'mpl'")
            mpl_flags = (
                annotate_3d
                or save_path is not None
                or show_normals
                or show_axes
                or show_optical_axis
                or annotate_3d_surface_ids
                or annotate_3d_intersections
            )
            use_mpl = backend_norm == "mpl" or (backend_norm == "auto" and mpl_flags)
            if use_mpl:
                fig, ax = self._plot_layout_3d_mpl(
                    surfaces=surfaces,
                    show_normals=show_normals,
                    normal_scale=normal_scale,
                    normal_color=normal_color,
                    normal_alpha=normal_alpha,
                    show_axes=show_axes,
                    axis_scale=axis_scale,
                    axis_alpha=axis_alpha,
                    axis_labels=axis_labels,
                    show_optical_axis=show_optical_axis,
                    optical_axis_color=optical_axis_color,
                    annotate_surface_ids=annotate_3d_surface_ids,
                    annotate_intersections=annotate_3d_intersections,
                    annotate_font_size=annotate_3d_font_size,
                )
                if save_path:
                    save_dir = Path(save_path).parent
                    if save_dir and not save_dir.exists():
                        save_dir.mkdir(parents=True, exist_ok=True)
                    fig.savefig(save_path, dpi=150, bbox_inches="tight")
                    print(f"光路图已保存到: {save_path}")
                if show:
                    plt.show()
                return fig, ax
            try:
                optic = to_optiland(
                    surfaces,
                    wavelength_um=wavelength_um,
                    entrance_pupil_diameter=entrance_pupil_diameter,
                )
                from sequential_system.zmx_visualization import view_3d

                if show:
                    overlay_surface_ids = annotate_3d or annotate_3d_surface_ids
                    overlay_intersections = annotate_3d or annotate_3d_intersections
                    view_3d(
                        optic,
                        overlay_surfaces=surfaces,
                        overlay_show_normals=show_normals,
                        overlay_normal_scale=normal_scale,
                        overlay_normal_color=normal_color,
                        overlay_show_axes=show_axes,
                        overlay_axis_scale=axis_scale,
                        overlay_axis_labels=axis_labels,
                        overlay_show_optical_axis=show_optical_axis,
                        overlay_optical_axis_color=optical_axis_color,
                        overlay_annotate_surface_ids=overlay_surface_ids,
                        overlay_annotate_intersections=overlay_intersections,
                        overlay_text_size=annotate_3d_font_size,
                        overlay_label_offset=label_offset,
                    )
                return None
            except Exception as exc:
                print(f"错误: 无法使用 3D 可视化 ({exc})")
                return None

        try:
            optic = to_optiland(
                surfaces,
                wavelength_um=wavelength_um,
                entrance_pupil_diameter=entrance_pupil_diameter,
            )
            from sequential_system.zmx_visualization import view_2d

            fig, ax, _ = view_2d(
                optic,
                projection=projection,
                num_rays=num_rays,
            )
            ax.set_title(f"光学系统: {self.name} ({projection} 投影)")
        except Exception as exc:
            print(f"警告: 无法使用 optiland 可视化 ({exc})，使用简化视图")
            fig, ax = self._plot_simple_layout(
                projection=projection,
                annotate=False,
                surfaces=surfaces,
            )

        if annotate:
            self._annotate_layout(
                ax=ax,
                projection=projection,
                label_mode=label_mode,
                label_fields=label_fields,
                font_size=font_size,
                surfaces=surfaces,
                full_surfaces=self.surfaces,
                show_normals=show_normals,
                show_normals_values=show_normals_values,
                show_incident_dirs=show_incident_dirs,
                normal_scale=normal_scale,
                normal_color=normal_color,
                normal_alpha=normal_alpha,
                show_axes=show_axes,
                axis_labels=axis_labels,
                axis_scale=axis_scale,
                axis_alpha=axis_alpha,
                show_optical_axis=show_optical_axis,
                optical_axis_color=optical_axis_color,
                incident_scale=incident_scale,
                incident_color=incident_color,
                incident_alpha=incident_alpha,
                label_offset=label_offset,
                label_rotation=label_rotation,
                label_box=label_box,
                label_alpha=label_alpha,
                label_avoid_overlap=label_avoid_overlap,
                label_min_dist=label_min_dist,
                label_step=label_step,
            )

        if save_path:
            save_dir = Path(save_path).parent
            if save_dir and not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"光路图已保存到: {save_path}")

        if show:
            plt.show()

        return fig, ax

    def _plot_simple_layout(
        self,
        projection: str = "YZ",
        annotate: bool = True,
        surfaces: Optional[Sequence[GlobalSurfaceDefinition]] = None,
    ) -> Tuple[Any, Any]:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        surfaces = list(surfaces) if surfaces is not None else self.surfaces
        projection = projection.upper()
        for surface in surfaces:
            vertex = np.asarray(surface.vertex_position, dtype=float)
            semi_ap = float(getattr(surface, "semi_aperture", 0.0) or 0.0)
            if projection == "YZ":
                x_coord = vertex[2]
                y_min, y_max = vertex[1] - semi_ap, vertex[1] + semi_ap
            elif projection == "XZ":
                x_coord = vertex[2]
                y_min, y_max = vertex[0] - semi_ap, vertex[0] + semi_ap
            else:
                x_coord = vertex[0]
                y_min, y_max = vertex[1] - semi_ap, vertex[1] + semi_ap

            color = "blue" if surface.is_mirror else "gray"
            linestyle = "-" if surface.is_mirror else "--"
            ax.plot(
                [x_coord, x_coord],
                [y_min, y_max],
                color=color,
                linestyle=linestyle,
                linewidth=2,
            )
            if annotate:
                ax.annotate(f"{surface.index}", (x_coord, y_max), ha="center", fontsize=8)

        xlabel = "Z (mm)" if projection in ("YZ", "XZ") else "X (mm)"
        ylabel = "Y (mm)" if projection in ("YZ", "XY") else "X (mm)"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"光学系统: {self.name} ({projection} 投影) - 简化视图")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        return fig, ax

    def _build_surface_label(
        self,
        surface: GlobalSurfaceDefinition,
        label_mode: str,
        label_fields: Optional[list[str]],
    ) -> str:
        def _format_field(field: str) -> Optional[str]:
            field = field.lower().strip()
            if field == "index":
                return f"S{surface.index}"
            if field == "name":
                return surface.comment.strip() if surface.comment else None
            if field == "type":
                return f"type={surface.surface_type}"
            if field == "radius":
                return f"R={surface.radius:.3f}"
            if field == "conic":
                return f"K={surface.conic:.4f}"
            if field == "material":
                return f"mat={surface.material}"
            if field == "vertex":
                v = surface.vertex_position
                return f"v=({v[0]:.2f},{v[1]:.2f},{v[2]:.2f})"
            if field == "aperture":
                return f"ap={surface.semi_aperture:.2f}"
            if field == "tilt":
                rx, ry, rz = rotation_matrix_to_euler(surface.orientation)
                rx_deg, ry_deg, rz_deg = np.rad2deg([rx, ry, rz])
                return f"tilt=({rx_deg:.1f},{ry_deg:.1f},{rz_deg:.1f})deg"
            if field == "normal":
                n = surface.orientation[:, 2]
                return f"n=({n[0]:.2f},{n[1]:.2f},{n[2]:.2f})"
            if field == "thickness":
                return f"t={surface.thickness:.2f}"
            return None

        if label_fields:
            parts = [p for p in (_format_field(f) for f in label_fields) if p]
            return "\n".join(parts) if parts else f"S{surface.index}"

        label_mode = label_mode.lower().strip()
        if label_mode == "index":
            return f"S{surface.index}"
        if label_mode == "name":
            return surface.comment.strip() if surface.comment else f"S{surface.index}"
        if label_mode == "params":
            parts = [
                f"S{surface.index}",
                f"type={surface.surface_type}",
                f"R={surface.radius:.3f}",
                f"K={surface.conic:.4f}",
            ]
            if surface.material:
                parts.append(f"mat={surface.material}")
            return "\n".join(parts)
        if label_mode == "full":
            parts = [
                f"S{surface.index}",
                f"type={surface.surface_type}",
                f"R={surface.radius:.3f}",
                f"K={surface.conic:.4f}",
                f"ap={surface.semi_aperture:.2f}",
                f"mat={surface.material}",
            ]
            return "\n".join(parts)
        return f"S{surface.index}"

    def _annotate_layout(
        self,
        ax: Any,
        projection: str,
        label_mode: str = "index",
        label_fields: Optional[list[str]] = None,
        font_size: int = 8,
        surfaces: Optional[Sequence[GlobalSurfaceDefinition]] = None,
        full_surfaces: Optional[Sequence[GlobalSurfaceDefinition]] = None,
        show_normals: bool = False,
        show_normals_values: bool = False,
        show_incident_dirs: bool = False,
        normal_scale: Optional[float] = None,
        normal_color: str = "tab:purple",
        normal_alpha: float = 0.8,
        show_axes: bool = False,
        axis_labels: bool = False,
        axis_scale: Optional[float] = None,
        axis_alpha: float = 0.7,
        show_optical_axis: bool = False,
        optical_axis_color: str = "tab:red",
        incident_scale: Optional[float] = None,
        incident_color: str = "tab:orange",
        incident_alpha: float = 0.8,
        label_offset: Optional[float] = None,
        label_rotation: float = 0.0,
        label_box: bool = False,
        label_alpha: float = 0.9,
        label_avoid_overlap: bool = True,
        label_min_dist: Optional[float] = None,
        label_step: Optional[float] = None,
    ) -> None:
        projection = projection.upper()
        y_min, y_max = ax.get_ylim()
        dy = label_offset if label_offset is not None else (0.02 * (y_max - y_min) if y_max > y_min else 1.0)

        surfaces = list(surfaces) if surfaces is not None else self.surfaces
        full_surfaces = list(full_surfaces) if full_surfaces is not None else self.surfaces
        index_to_position = {surface.index: i for i, surface in enumerate(full_surfaces)}

        if show_optical_axis and len(surfaces) > 1:
            points = [self._project_point(s.vertex_position, projection) for s in surfaces]
            xs, ys = zip(*points)
            ax.plot(xs, ys, color=optical_axis_color, linestyle="-", linewidth=1.0, alpha=0.7)

        x_min, x_max = ax.get_xlim()
        x_range = x_max - x_min
        y_range = y_max - y_min
        min_dist = label_min_dist if label_min_dist is not None else 0.04 * max(x_range, y_range)
        step = label_step if label_step is not None else 0.03 * max(x_range, y_range)
        used_positions: list[tuple[float, float]] = []

        for surface in surfaces:
            vertex = np.asarray(surface.vertex_position, dtype=float)
            semi_ap = float(getattr(surface, "semi_aperture", 0.0) or 0.0)
            if projection == "YZ":
                x_coord = vertex[2]
                y_coord = vertex[1] + semi_ap + dy
            elif projection == "XZ":
                x_coord = vertex[2]
                y_coord = vertex[0] + semi_ap + dy
            else:
                x_coord = vertex[0]
                y_coord = vertex[1] + semi_ap + dy

            label = self._build_surface_label(surface, label_mode, label_fields)
            text_kwargs = {}
            if label_box:
                text_kwargs["bbox"] = dict(facecolor="white", alpha=label_alpha, pad=2)
            if label_avoid_overlap:
                x_coord, y_coord = self._resolve_label_position(
                    x_coord,
                    y_coord,
                    used_positions,
                    min_dist=min_dist,
                    step=step,
                )
            ax.text(
                x_coord,
                y_coord,
                label,
                fontsize=font_size,
                ha="center",
                va="bottom",
                rotation=label_rotation,
                **text_kwargs,
            )

            if show_incident_dirs:
                direction = self._compute_incident_direction(
                    full_surfaces,
                    index_to_position,
                    surface.index,
                )
                scale = incident_scale if incident_scale is not None else max(5.0, semi_ap * 0.6)
                self._draw_vector(
                    ax=ax,
                    origin=vertex,
                    vector=direction * scale,
                    projection=projection,
                    color=incident_color,
                    alpha=incident_alpha,
                )

            if show_normals:
                normal = np.asarray(surface.orientation[:, 2], dtype=float)
                scale = normal_scale if normal_scale is not None else max(5.0, semi_ap * 0.6)
                self._draw_vector(
                    ax=ax,
                    origin=vertex,
                    vector=normal * scale,
                    projection=projection,
                    color=normal_color,
                    alpha=normal_alpha,
                )
                if show_normals_values:
                    n = normal / (np.linalg.norm(normal) + 1e-12)
                    nx, ny, nz = n.tolist()
                    x_end, y_end = self._project_point(vertex + normal * scale, projection)
                    ax.text(
                        x_end,
                        y_end,
                        f"n=({nx:.2f},{ny:.2f},{nz:.2f})",
                        fontsize=max(6, font_size - 1),
                        ha="left",
                        va="bottom",
                        color=normal_color,
                    )

            if show_axes:
                scale = axis_scale if axis_scale is not None else max(4.0, semi_ap * 0.4)
                x_axis = np.asarray(surface.orientation[:, 0], dtype=float) * scale
                y_axis = np.asarray(surface.orientation[:, 1], dtype=float) * scale
                z_axis = np.asarray(surface.orientation[:, 2], dtype=float) * scale
                self._draw_vector(ax, vertex, x_axis, projection, color="tab:red", alpha=axis_alpha)
                self._draw_vector(ax, vertex, y_axis, projection, color="tab:green", alpha=axis_alpha)
                self._draw_vector(ax, vertex, z_axis, projection, color="tab:blue", alpha=axis_alpha)
                if axis_labels:
                    self._label_axis(ax, vertex, x_axis, projection, "X", "tab:red")
                    self._label_axis(ax, vertex, y_axis, projection, "Y", "tab:green")
                    self._label_axis(ax, vertex, z_axis, projection, "Z", "tab:blue")

    def _select_surfaces(
        self,
        surface_indices: Optional[Sequence[int]],
    ) -> List[GlobalSurfaceDefinition]:
        if surface_indices is None:
            return list(self.surfaces)
        index_set = set(surface_indices)
        return [surface for surface in self.surfaces if surface.index in index_set]

    def _resolve_label_position(
        self,
        x: float,
        y: float,
        used_positions: list[tuple[float, float]],
        min_dist: float,
        step: float,
    ) -> tuple[float, float]:
        new_x, new_y = x, y
        for _ in range(30):
            if all(
                (new_x - ux) ** 2 + (new_y - uy) ** 2 >= min_dist**2
                for ux, uy in used_positions
            ):
                break
            new_y += step
        used_positions.append((new_x, new_y))
        return new_x, new_y

    def _compute_incident_direction(
        self,
        surfaces: Sequence[GlobalSurfaceDefinition],
        index_to_position: dict[int, int],
        surface_index: int,
    ) -> np.ndarray:
        if surface_index not in index_to_position:
            return np.array([0.0, 0.0, 1.0])
        idx = index_to_position[surface_index]
        current = np.asarray(surfaces[idx].vertex_position, dtype=float)
        if idx > 0:
            prev = np.asarray(surfaces[idx - 1].vertex_position, dtype=float)
            direction = current - prev
        elif idx + 1 < len(surfaces):
            nxt = np.asarray(surfaces[idx + 1].vertex_position, dtype=float)
            direction = nxt - current
        else:
            direction = np.asarray(surfaces[idx].orientation[:, 2], dtype=float)
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            direction = np.asarray(surfaces[idx].orientation[:, 2], dtype=float)
            norm = np.linalg.norm(direction)
        return direction / (norm + 1e-12)

    def _project_point(
        self,
        point: np.ndarray,
        projection: str,
    ) -> tuple[float, float]:
        point = np.asarray(point, dtype=float)
        projection = projection.upper()
        if projection == "YZ":
            return float(point[2]), float(point[1])
        if projection == "XZ":
            return float(point[2]), float(point[0])
        return float(point[0]), float(point[1])

    def _draw_vector(
        self,
        ax: Any,
        origin: np.ndarray,
        vector: np.ndarray,
        projection: str,
        color: str,
        alpha: float,
    ) -> None:
        start = np.asarray(origin, dtype=float)
        end = start + np.asarray(vector, dtype=float)
        x0, y0 = self._project_point(start, projection)
        x1, y1 = self._project_point(end, projection)
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color=color, alpha=alpha, linewidth=1.2),
        )

    def _label_axis(
        self,
        ax: Any,
        origin: np.ndarray,
        vector: np.ndarray,
        projection: str,
        label: str,
        color: str,
    ) -> None:
        end = np.asarray(origin, dtype=float) + np.asarray(vector, dtype=float)
        x1, y1 = self._project_point(end, projection)
        ax.text(x1, y1, label, color=color, fontsize=7, ha="left", va="bottom")

    def _plot_layout_3d_mpl(
        self,
        surfaces: Sequence[GlobalSurfaceDefinition],
        show_normals: bool,
        normal_scale: Optional[float],
        normal_color: str,
        normal_alpha: float,
        show_axes: bool,
        axis_scale: Optional[float],
        axis_alpha: float,
        axis_labels: bool,
        show_optical_axis: bool,
        optical_axis_color: str,
        annotate_surface_ids: bool,
        annotate_intersections: bool,
        annotate_font_size: Optional[int] = None,
    ) -> Tuple[Any, Any]:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        vertices = np.array([s.vertex_position for s in surfaces], dtype=float)
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color="tab:blue", s=25)

        if show_optical_axis and len(vertices) > 1:
            ax.plot(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                color=optical_axis_color,
                linewidth=1.0,
                alpha=0.7,
            )

        text_size = int(annotate_font_size) if annotate_font_size is not None else 8
        coord_size = max(6, text_size - 2)
        axis_label_size = max(7, text_size - 3)

        for surface in surfaces:
            v = np.asarray(surface.vertex_position, dtype=float)
            if annotate_surface_ids:
                ax.text(v[0], v[1], v[2], f"S{surface.index}", fontsize=text_size, color="black")
            if annotate_intersections:
                ax.text(
                    v[0],
                    v[1],
                    v[2],
                    f"({v[0]:.2f},{v[1]:.2f},{v[2]:.2f})",
                    fontsize=coord_size,
                    color="gray",
                )
            semi_ap = float(getattr(surface, "semi_aperture", 0.0) or 0.0)
            if show_normals:
                scale = normal_scale if normal_scale is not None else max(5.0, semi_ap * 0.6)
                normal = np.asarray(surface.orientation[:, 2], dtype=float) * scale
                self._draw_vector_3d(ax, v, normal, normal_color, normal_alpha)
            if show_axes:
                scale = axis_scale if axis_scale is not None else max(4.0, semi_ap * 0.4)
                x_axis = np.asarray(surface.orientation[:, 0], dtype=float) * scale
                y_axis = np.asarray(surface.orientation[:, 1], dtype=float) * scale
                z_axis = np.asarray(surface.orientation[:, 2], dtype=float) * scale
                self._draw_vector_3d(ax, v, x_axis, "tab:red", axis_alpha)
                self._draw_vector_3d(ax, v, y_axis, "tab:green", axis_alpha)
                self._draw_vector_3d(ax, v, z_axis, "tab:blue", axis_alpha)
                if axis_labels:
                    ax.text(*(v + x_axis), "X", color="tab:red", fontsize=axis_label_size)
                    ax.text(*(v + y_axis), "Y", color="tab:green", fontsize=axis_label_size)
                    ax.text(*(v + z_axis), "Z", color="tab:blue", fontsize=axis_label_size)

        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_title(f"光学系统: {self.name} (3D)")

        ranges = np.ptp(vertices, axis=0)
        max_range = np.max(ranges) if np.any(ranges) else 1.0
        mid = np.mean(vertices, axis=0)
        ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
        ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
        ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)

        return fig, ax

    def _draw_vector_3d(
        self,
        ax: Any,
        origin: np.ndarray,
        vector: np.ndarray,
        color: str,
        alpha: float,
    ) -> None:
        origin = np.asarray(origin, dtype=float)
        vector = np.asarray(vector, dtype=float)
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            vector[0],
            vector[1],
            vector[2],
            arrow_length_ratio=0.15,
            color=color,
            alpha=alpha,
            linewidth=1.0,
        )
