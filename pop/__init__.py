"""
POP package entrypoint.

This module wires vendor dependencies and local src modules into sys.path and exposes the package
namespace for downstream modules.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from . import coordinates, core, io, propagation, source, utils, visualization, wavefront
from .options import DebugOptions, PlotOptions, PropagationOptions

_VENDOR_DIRS = ("src", "optiland-master", "proper_v3.3.4_python")


def _add_vendor_path(relative_path: str) -> Optional[str]:
    root = Path(__file__).resolve().parents[1]
    vendor_path = (root / relative_path).resolve()
    if not vendor_path.exists():
        return None
    vendor_path_str = str(vendor_path)
    if vendor_path_str not in sys.path:
        sys.path.insert(0, vendor_path_str)
        return vendor_path_str
    return None


def configure_vendor_paths() -> tuple[str, ...]:
    """Ensure vendor packages are on sys.path, returning newly added paths."""
    added: list[str] = []
    for relative_path in _VENDOR_DIRS:
        added_path = _add_vendor_path(relative_path)
        if added_path:
            added.append(added_path)
    return tuple(added)


def check_vendor_imports() -> dict[str, bool]:
    """Lightweight availability check for optiland/proper imports."""
    results: dict[str, bool] = {}
    try:
        from optiland import Optic  # noqa: F401
        from optiland.surfaces import Surface  # noqa: F401

        results["optiland"] = hasattr(Surface, "trace")
    except Exception:
        results["optiland"] = False
    try:
        import proper  # noqa: F401

        results["proper"] = hasattr(proper, "prop_run")
    except Exception:
        results["proper"] = False
    return results


configure_vendor_paths()

from .result import PropagationResult, SurfaceRecord
from .source import CustomSource, GaussianSource
from .system import System


def load_zmx(path: str) -> System:
    return System.from_zmx(path)


def _normalize_material_name(material: str | None) -> str:
    if material is None:
        return "air"
    material = material.strip()
    return material if material else "air"


def _should_trace_element(surface, incident_material: str) -> bool:
    surface_type = getattr(surface, "surface_type", "") or ""
    if surface_type.lower() in ("coordinate_break", "coord_break"):
        return False
    if surface_type.lower() == "paraxial":
        return True
    if getattr(surface, "is_mirror", False):
        return True
    surface_material = _normalize_material_name(getattr(surface, "material", None))
    incident_material = _normalize_material_name(incident_material)
    return surface_material != incident_material


def _clone_state_with_wfo(state: core.PropagationState) -> core.PropagationState:
    if state.proper_wfo is None:
        return state
    return replace(state, proper_wfo=copy.deepcopy(state.proper_wfo))


def _intersect_axis_with_plane(
    position: np.ndarray,
    direction: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    import numpy as np

    plane_normal = np.asarray(plane_normal, dtype=float)
    denom = float(np.dot(direction, plane_normal))
    if abs(denom) < 1e-12:
        return position.copy()
    t = float(np.dot(plane_point - position, plane_normal) / denom)
    return position + t * direction


def _build_surface_overrides(surface, interaction: core.SurfaceInteractionInfo) -> dict[str, object]:
    import numpy as np

    overrides: dict[str, object] = {}
    if not interaction.normal_flipped:
        return overrides

    sign_factor = interaction.sign_factor
    if interaction.canonical_orientation is not None:
        overrides["orientation"] = interaction.canonical_orientation

    radius = float(surface.radius)
    if np.isfinite(radius):
        overrides["radius"] = radius * sign_factor

    radius_x = float(getattr(surface, "radius_x", np.inf))
    if np.isfinite(radius_x):
        overrides["radius_x"] = radius_x * sign_factor

    surface_type = (getattr(surface, "surface_type", "") or "").lower()
    if surface_type == "even_asphere":
        coeffs = list(surface.asphere_coeffs or [])
        if coeffs:
            overrides["asphere_coeffs"] = [-float(c) for c in coeffs]

    return overrides


def _print_surface_list(surfaces, print_status: bool = True):
    """打印所有 GlobalSurfaceDefinition 表面的信息，包括列表索引和 ZMX 原始索引。
    
    参数:
        surfaces: GlobalSurfaceDefinition 列表
        print_status: 是否打印（如果为 False，则跳过打印）
    """
    if not print_status:
        return
    
    print("\n" + "=" * 100)
    print("GlobalSurfaceDefinition 表面列表 (在主光路追迹前)")
    print("=" * 100)
    print(f"{'列表索引':<8} {'ZMX索引':<8} {'类型':<15} {'名称/注释':<30} {'R(mm)':<12} {'材料':<12} {'反射镜':<8}")
    print("-" * 100)
    
    for list_idx, surface in enumerate(surfaces):
        zmx_index = surface.index
        surface_type = getattr(surface, "surface_type", "unknown")
        comment = getattr(surface, "comment", "") or f"Surface {zmx_index}"
        radius = getattr(surface, "radius", np.inf)
        material = getattr(surface, "material", "air")
        is_mirror = getattr(surface, "is_mirror", False)
        
        # 格式化曲率半径
        if np.isinf(radius):
            radius_str = "inf"
        else:
            radius_str = f"{radius:.2f}"
        
        # 格式化双锥面信息
        if surface_type == "biconic":
            radius_x = getattr(surface, "radius_x", np.inf)
            if not np.isinf(radius_x):
                radius_str = f"Rx={radius_x:.2f}, Ry={radius_str}"
        
        mirror_str = "是" if is_mirror else "否"
        
        print(f"{list_idx:<8} {zmx_index:<8} {surface_type:<15} {comment[:28]:<30} {radius_str:<12} {material:<12} {mirror_str:<8}")
    
    print("-" * 100)
    print(f"总计: {len(surfaces)} 个光学表面")
    print("注意: '列表索引' 是 GlobalSurfaceDefinition 列表中的位置 (从 0 开始)")
    print("      'ZMX索引' 是 ZMX 文件中的原始表面索引 (包含坐标断点等)")
    print("      pilot_refit_surface_indices 应使用 'ZMX索引'")
    print("=" * 100 + "\n")


def _print_chief_ray_table(surfaces, chief_ray_data, print_status: bool = True):
    """打印主光线数据表格。

    参数:
        surfaces: GlobalSurfaceDefinition 列表
        chief_ray_data: _trace_chief_ray 的返回结果
        print_status: 是否打印
    """
    if not print_status:
        return

    print("\n" + "=" * 100)
    print("主光线数据 (Chief Ray Data) — 全局坐标")
    print("=" * 100)
    header = (
        f"{'Surf':<6} {'X (mm)':<14} {'Y (mm)':<14} {'Z (mm)':<14} "
        f"{'L':<14} {'M':<14} {'N':<14}"
    )
    print(header)
    print("-" * 100)

    for i, surface in enumerate(surfaces):
        data = chief_ray_data[i]
        pt = data["intersection"]
        d = data["exit_direction"]
        zmx_idx = surface.index
        print(
            f"S{zmx_idx:<5} {pt[0]:< 14.6f}{pt[1]:< 14.6f}{pt[2]:< 14.6f}"
            f"{d[0]:< 14.6f}{d[1]:< 14.6f}{d[2]:< 14.6f}"
        )

    print("=" * 100 + "\n")


def _trace_chief_ray(surfaces, wavelength_um: float, coordinate_priority: str = "x"):
    import numpy as np
    from optiland.rays import RealRays

    from pop.propagation.element import _build_optiland_surface_group, _trace_with_signed_opd

    rays = RealRays(
        x=[0.0],
        y=[0.0],
        z=[0.0],
        L=[0.0],
        M=[0.0],
        N=[1.0],
        intensity=[1.0],
        wavelength=wavelength_um,
    )
    rays.opd = np.zeros(1)

    chief_ray_data = []
    current_material = "air"
    current_position = np.array([0.0, 0.0, 0.0])
    current_direction = np.array([0.0, 0.0, 1.0])
    prev_frame = None

    for surface in surfaces:
        entrance_axis = core.OpticalAxisState(
            position=current_position,
            direction=current_direction,
            frame=coordinates.transforms.build_min_rotation_frame(
                current_direction, prev_frame, priority_axis=coordinate_priority
            ),
            coord_sys=None,
            path_length=0.0,
        )

        interaction = coordinates.transforms.build_surface_interaction_info(
            surface_index=surface.index,
            orientation=surface.orientation,
            entrance_dir=current_direction,
            is_mirror=surface.is_mirror,
        )

        surface_overrides = _build_surface_overrides(surface, interaction)
        orientation_override = surface_overrides.get("orientation")
        radius_override = surface_overrides.get("radius")
        radius_x_override = surface_overrides.get("radius_x")
        asphere_coeffs_override = surface_overrides.get("asphere_coeffs")
        if _should_trace_element(surface, current_material):
            optiland_group = _build_optiland_surface_group(
                surface=surface,
                entrance_axis=entrance_axis,
                exit_axis=None,
                incident_material_name=current_material,
                pilot_params=None,
                interaction_info=interaction,
                orientation_override=orientation_override,
                radius_override=radius_override,
                radius_x_override=radius_x_override,
                asphere_coeffs_override=asphere_coeffs_override,
                include_exit_plane=False,
            )
            # CRITICAL FIX: Ensure surface 1 links to surface 0
            if len(optiland_group.surfaces) > 1:
                 optiland_group.surfaces[1].previous_surface = optiland_group.surfaces[0]
            
            traced_surface = optiland_group.surfaces[1]
            _trace_with_signed_opd(traced_surface, rays)

            if len(traced_surface.x) == 0:
                raise RuntimeError(
                    f"Chief ray trace failed at surface {surface.index}: no intersection."
                )

            intersection = np.array(
                [traced_surface.x[0], traced_surface.y[0], traced_surface.z[0]]
            )
            exit_direction = np.array(
                [traced_surface.L[0], traced_surface.M[0], traced_surface.N[0]]
            )
            trace_branch = "raytrace"
        else:
            orientation_for_plane = (
                orientation_override
                if orientation_override is not None
                else np.asarray(surface.orientation, dtype=float)
            )
            plane_normal = orientation_for_plane[:, 2]
            intersection = _intersect_axis_with_plane(
                current_position,
                current_direction,
                np.asarray(surface.vertex_position, dtype=float),
                plane_normal,
            )
            exit_direction = current_direction.copy()
            rays.x = np.array([intersection[0]])
            rays.y = np.array([intersection[1]])
            rays.z = np.array([intersection[2]])
            rays.L = np.array([exit_direction[0]])
            rays.M = np.array([exit_direction[1]])
            rays.N = np.array([exit_direction[2]])
            trace_branch = "plane"

        chief_ray_data.append(
            {
                "intersection": intersection,
                "exit_direction": exit_direction,
                "interaction_info": interaction,
                "surface_overrides": surface_overrides,
            }
        )

        current_position = intersection
        current_direction = exit_direction
        prev_frame = entrance_axis.frame

        if not surface.is_mirror and _should_trace_element(surface, current_material):
            current_material = _normalize_material_name(surface.material)

    return chief_ray_data


def _build_optical_axis_states(surfaces, chief_ray_data, coordinate_priority: str = "x"):
    import numpy as np

    axis_states = []
    current_direction = np.array([0.0, 0.0, 1.0])
    previous_position = np.array([0.0, 0.0, 0.0])
    path_length = 0.0
    prev_frame = None
    prev_euler = None

    for i, surface in enumerate(surfaces):
        if i < len(chief_ray_data):
            intersection = chief_ray_data[i]["intersection"]
            exit_direction = chief_ray_data[i]["exit_direction"]
        else:
            intersection = np.asarray(surface.vertex_position, dtype=float)
            exit_direction = current_direction.copy()

        distance = float(np.linalg.norm(intersection - previous_position))
        path_length += distance

        entrance_frame = coordinates.transforms.build_min_rotation_frame(
            current_direction, prev_frame, priority_axis=coordinate_priority
        )
        entrance_euler = coordinates.transforms.rotation_matrix_to_euler(
            entrance_frame, prev_euler
        )
        entrance_axis = core.OpticalAxisState(
            position=intersection,
            direction=current_direction,
            frame=entrance_frame,
            coord_sys=None,
            path_length=path_length,
            euler=entrance_euler,
        )

        exit_frame = coordinates.transforms.build_min_rotation_frame(
            exit_direction, entrance_frame, priority_axis=coordinate_priority
        )
        exit_euler = coordinates.transforms.rotation_matrix_to_euler(
            exit_frame, entrance_euler
        )
        exit_axis = core.OpticalAxisState(
            position=intersection,
            direction=exit_direction,
            frame=exit_frame,
            coord_sys=None,
            path_length=path_length,
            euler=exit_euler,
        )
        axis_states.append({"entrance": entrance_axis, "exit": exit_axis})

        previous_position = intersection
        current_direction = exit_direction
        prev_frame = exit_frame
        prev_euler = exit_euler

    return axis_states


def _propagate_paraxial(
    state: core.PropagationState,
    surface,
    exit_axis: core.OpticalAxisState,
    auto_unwarp_at_incident: bool = False,
    trace_context: Optional[dict[str, Any]] = None,
):
    import numpy as np
    import proper

    focal_length_mm = surface.focal_length
    if np.isinf(focal_length_mm):
        return state
    proper.prop_lens(state.proper_wfo, focal_length_mm * 1e-3)
    new_pilot = state.pilot_beam_params.apply_lens(focal_length_mm)
    propagation.free_space._sync_proper_gaussian_params(state.proper_wfo, new_pilot)
    grid_sampling = core.GridSampling.from_proper(state.proper_wfo)
    amplitude, phase = propagation.free_space.proper_to_amplitude_phase(
        state.proper_wfo,
        grid_sampling,
        new_pilot,
        auto_unwarp_at_incident=auto_unwarp_at_incident,
        trace_context=trace_context,
    )
    return core.PropagationState(
        surface_index=surface.index,
        position="exit",
        amplitude=amplitude,
        phase=phase,
        pilot_beam_params=new_pilot,
        optical_axis_state=exit_axis,
        grid_sampling=grid_sampling,
        proper_wfo=state.proper_wfo,
        force_asm=state.force_asm,
        propagation_algorithm="paraxial",
    )


def _format_position_label(position: Optional[str]) -> str:
    mapping = {
        "source": "光源",
        "entrance": "入口",
        "exit": "出射",
    }
    if not position:
        return ""
    return mapping.get(position, position)


def _format_surface_label(surface_index: Optional[int], position: Optional[str]) -> str:
    if surface_index is None or surface_index < 0:
        return "光源"
    pos_label = _format_position_label(position)
    if pos_label:
        return f"S{surface_index}/{pos_label}"
    return f"S{surface_index}"


def _format_value(value: Any, precision: int = 6) -> str:
    if value is None:
        return "None"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isinf(val):
        return "inf"
    if np.isnan(val):
        return "nan"
    return f"{val:.{precision}g}"


def _format_coeffs(coeffs: Optional[Sequence[float]], max_terms: int = 6) -> str:
    if not coeffs:
        return "[]"
    coeff_list = list(coeffs)
    head = ", ".join(_format_value(c) for c in coeff_list[:max_terms])
    if len(coeff_list) > max_terms:
        return f"[{head}, ...] (len={len(coeff_list)})"
    return f"[{head}]"


def _get_override_value(
    surface_overrides: Optional[dict[str, object]],
    key: str,
    fallback: Any,
) -> Any:
    if surface_overrides and surface_overrides.get(key) is not None:
        return surface_overrides.get(key)
    return fallback


def _format_surface_params(
    surface: Any,
    surface_overrides: Optional[dict[str, object]],
) -> str:
    surface_type = (getattr(surface, "surface_type", "") or "").lower()
    parts: list[str] = [f"类型={surface_type or 'unknown'}"]

    comment = getattr(surface, "comment", "")
    if comment:
        parts.append(f"名称={comment}")

    if surface_type == "paraxial":
        focal_length = _get_override_value(
            surface_overrides,
            "focal_length",
            getattr(surface, "focal_length", None),
        )
        parts.append(f"焦距={_format_value(focal_length)} mm")
    elif surface_type == "biconic":
        radius_y = _get_override_value(
            surface_overrides,
            "radius",
            getattr(surface, "radius", np.inf),
        )
        radius_x = _get_override_value(
            surface_overrides,
            "radius_x",
            getattr(surface, "radius_x", np.inf),
        )
        conic_y = getattr(surface, "conic", None)
        conic_x = getattr(surface, "conic_x", None)
        parts.append(f"Rx={_format_value(radius_x)} mm")
        parts.append(f"Ry={_format_value(radius_y)} mm")
        if conic_x is not None:
            parts.append(f"Kx={_format_value(conic_x)}")
        if conic_y is not None:
            parts.append(f"Ky={_format_value(conic_y)}")
    elif surface_type == "even_asphere":
        radius = _get_override_value(
            surface_overrides,
            "radius",
            getattr(surface, "radius", np.inf),
        )
        conic = getattr(surface, "conic", None)
        coeffs = _get_override_value(
            surface_overrides,
            "asphere_coeffs",
            getattr(surface, "asphere_coeffs", []),
        )
        parts.append(f"R={_format_value(radius)} mm")
        if conic is not None:
            parts.append(f"K={_format_value(conic)}")
        parts.append(f"系数={_format_coeffs(coeffs)}")
    else:
        radius = _get_override_value(
            surface_overrides,
            "radius",
            getattr(surface, "radius", np.inf),
        )
        conic = getattr(surface, "conic", None)
        parts.append(f"R={_format_value(radius)} mm")
        if conic is not None:
            parts.append(f"K={_format_value(conic)}")

    semi_aperture = getattr(surface, "semi_aperture", None)
    if semi_aperture is not None and abs(float(semi_aperture)) > 0:
        parts.append(f"半口径={_format_value(semi_aperture)} mm")

    thickness = getattr(surface, "thickness", None)
    if thickness is not None and abs(float(thickness)) > 0:
        parts.append(f"厚度={_format_value(thickness)} mm")

    material = getattr(surface, "material", None)
    if material:
        parts.append(f"材质={material}")

    parts.append(f"镜面={bool(getattr(surface, 'is_mirror', False))}")

    if surface_overrides:
        override_keys = sorted(k for k, v in surface_overrides.items() if v is not None)
        if override_keys:
            parts.append(f"覆盖={','.join(override_keys)}")

    return ", ".join(parts)


def _format_propagation_label(propagation_type: str) -> str:
    mapping = {
        "free_space": "自由空间",
        "free_space_entrance": "自由空间(到入口)",
        "free_space_exit": "自由空间(到出射)",
        "element": "元件/光线追迹",
        "paraxial": "近轴透镜",
    }
    return mapping.get(propagation_type, propagation_type)


def _build_trace_context(
    *,
    from_state: core.PropagationState,
    to_surface: Any,
    to_position: str,
    propagation_type: str,
    distance_mm: Optional[float],
    surface_overrides: Optional[dict[str, object]],
) -> dict[str, Any]:
    from_label = _format_surface_label(
        getattr(from_state, "surface_index", None), getattr(from_state, "position", None)
    )
    to_label = _format_surface_label(getattr(to_surface, "index", None), to_position)
    prop_label = _format_propagation_label(propagation_type)
    if distance_mm is None:
        distance_text = "NA"
    else:
        try:
            distance_text = (
                f"{distance_mm:.6f}" if np.isfinite(distance_mm) else str(distance_mm)
            )
        except TypeError:
            distance_text = str(distance_mm)
    surface_params = _format_surface_params(to_surface, surface_overrides)
    status_line = (
        f"{prop_label}: {from_label} -> {to_label} | "
        f"距离={distance_text} mm | {surface_params}"
    )
    return {
        "status_line": status_line,
        "propagation_type": propagation_type,
        "from_surface_index": getattr(from_state, "surface_index", None),
        "from_position": getattr(from_state, "position", None),
        "to_surface_index": getattr(to_surface, "index", None),
        "to_position": to_position,
        "distance_mm": distance_mm,
        "surface_type": getattr(to_surface, "surface_type", None),
        "surface_params": surface_params,
    }


def propagate(
    system: System,
    source: "GaussianSource | CustomSource",
    num_rays: Optional[int] = None,
    coordinate_priority: Optional[str] = None,
    debug: Optional[bool] = None,
    debug_plot_3d: Optional[bool] = None,
    debug_plot_3d_dir: Optional[str | Path] = None,
    debug_plot_3d_show: Optional[bool] = None,
    debug_plot_3d_ray_count: Optional[int] = None,
    plot_mode: Optional[str] = "all",
    plot_output_dir: Optional[str | Path] = None,
    plot_show: Optional[bool] = None,
    plot_positions: Optional[Sequence[str] | str] = None,
    plot_surface_indices: Optional[Sequence[int]] = None,
    plot_kwargs: Optional[dict[str, Any]] = None,
    plot_axis_3d: Optional[bool] = None,
    plot_axis_3d_dir: Optional[str | Path] = None,
    plot_axis_3d_show: Optional[bool] = None,
    plot_axis_3d_kwargs: Optional[dict[str, Any]] = None,
    print_status: Optional[bool] = None,
    options: Optional[PropagationOptions] = None,
    plot_options: Optional[PlotOptions] = None,
    debug_options: Optional[DebugOptions] = None,
    logger: Optional[Any] = None,
    auto_unwarp_at_incident: Optional[bool] = None,
) -> PropagationResult:
    options = options or PropagationOptions()
    plot_options = plot_options or PlotOptions()
    debug_options = debug_options or DebugOptions()

    if num_rays is None:
        num_rays = options.num_rays
    if coordinate_priority is None:
        coordinate_priority = options.coordinate_priority
    if print_status is None:
        print_status = options.print_status

    force_asm = options.force_asm
    auto_asm = options.auto_asm
    reconstruction_mask_ratio = options.reconstruction_mask_ratio
    phase_method = options.phase_method
    zernike_terms = options.zernike_terms
    zernike_normalize = options.zernike_normalize
    sampling_sigma = options.sampling_sigma
    pilot_refit_surface_indices = options.pilot_refit_surface_indices
    pilot_refit_pv_threshold_waves = options.pilot_refit_pv_threshold_waves
    if auto_unwarp_at_incident is None:
        auto_unwarp_at_incident = options.auto_unwarp_at_incident
    auto_unwarp_surface_indices = options.auto_unwarp_surface_indices
    auto_resample = options.auto_resample
    resample_min_beam_pixels = options.resample_min_beam_pixels
    resample_beam_pixels_target = options.resample_beam_pixels_target
    merge_free_space_surfaces = options.merge_free_space_surfaces

    if debug is None:
        debug = debug_options.enabled
    if debug_plot_3d is None:
        debug_plot_3d = debug_options.plot_3d
    if debug_plot_3d_dir is None:
        debug_plot_3d_dir = debug_options.plot_3d_dir
    if debug_plot_3d_show is None:
        debug_plot_3d_show = debug_options.plot_3d_show
    if debug_plot_3d_ray_count is None:
        debug_plot_3d_ray_count = debug_options.plot_3d_ray_count

    if plot_mode == "all":
        plot_mode = plot_options.mode
    if plot_output_dir is None:
        plot_output_dir = plot_options.output_dir
    if plot_show is None:
        plot_show = plot_options.show
    if plot_positions is None:
        plot_positions = plot_options.positions
    if plot_surface_indices is None:
        plot_surface_indices = plot_options.surface_indices
    if plot_kwargs is None:
        plot_kwargs = plot_options.kwargs
    if plot_axis_3d is None:
        plot_axis_3d = plot_options.axis_3d
    if plot_axis_3d_dir is None:
        plot_axis_3d_dir = plot_options.axis_3d_dir
    if plot_axis_3d_show is None:
        plot_axis_3d_show = plot_options.axis_3d_show
    if plot_axis_3d_kwargs is None:
        plot_axis_3d_kwargs = plot_options.axis_3d_kwargs

    if logger is None:
        log_fn = print
    elif callable(logger):
        log_fn = logger
    elif hasattr(logger, "info"):
        log_fn = logger.info
    else:
        raise TypeError("logger must be callable or logging.Logger")

    surfaces = system.surfaces
    _print_surface_list(surfaces, print_status=print_status)
    chief_ray_data = _trace_chief_ray(surfaces, source.wavelength_um, coordinate_priority=coordinate_priority)
    _print_chief_ray_table(surfaces, chief_ray_data, print_status=print_status)
    axis_states = _build_optical_axis_states(surfaces, chief_ray_data, coordinate_priority=coordinate_priority)

    axis_plot_config = dict(plot_axis_3d_kwargs) if plot_axis_3d_kwargs else {}
    for key in ("save_path", "show"):
        axis_plot_config.pop(key, None)
    if plot_axis_3d:
        axis_output_dir = utils.resolve_output_dir(plot_axis_3d_dir or plot_output_dir)
        axis_output_dir.mkdir(parents=True, exist_ok=True)
        axis_surface_records: list[SurfaceRecord] = []
        for i, surface in enumerate(surfaces):
            axis_state = axis_states[i]
            axis_surface_records.append(
                SurfaceRecord(
                    index=surface.index,
                    surface=surface,
                    entrance=None,
                    exit=None,
                    entrance_axis=axis_state["entrance"],
                    exit_axis=axis_state["exit"],
                    interaction_info=chief_ray_data[i].get("interaction_info"),
                    surface_overrides=chief_ray_data[i].get("surface_overrides"),
                    debug=None,
                )
            )
        visualization.plot_optical_axis_3d(
            surface_records=axis_surface_records,
            save_path=axis_output_dir / "optical_axis_3d.png",
            show=plot_axis_3d_show,
            **axis_plot_config,
        )

    amplitude, phase, pilot_beam, proper_wfo = source.create_initial_wavefront()
    grid_sampling = core.GridSampling.from_proper(proper_wfo)
    source_axis = core.OpticalAxisState(
        position=np.array([0.0, 0.0, 0.0]),
        direction=np.array([0.0, 0.0, 1.0]),
        frame=coordinates.transforms.build_min_rotation_frame(
            np.array([0.0, 0.0, 1.0]), priority_axis=coordinate_priority
        ),
        coord_sys=None,
        path_length=0.0,
    )

    current_state = core.PropagationState(
        surface_index=-1,
        position="source",
        amplitude=amplitude,
        phase=phase,
        pilot_beam_params=pilot_beam,
        optical_axis_state=source_axis,
        grid_sampling=grid_sampling,
        proper_wfo=proper_wfo,
    )

    surface_states = []
    surface_records: list[SurfaceRecord] = []
    current_material = "air"
    plot_mode_normalized = plot_mode.lower().strip() if plot_mode else None
    if plot_mode_normalized is not None and plot_mode_normalized not in ("normal", "debug", "all"):
        raise ValueError("plot_mode must be 'normal', 'debug', or 'all'")
    if plot_mode_normalized in ("debug", "all") and not debug:
        debug = True
    debug_enabled = debug or debug_plot_3d
    debug_store: dict[int, "SurfaceDebugInfo"] | None = {} if debug_enabled else None
    plot_surface_index_set = set(plot_surface_indices) if plot_surface_indices is not None else None

    # 解析 resample debug 绘图输出目录（与出入射面信息使用同一文件夹）
    resample_debug_dir: str | Path | None = None
    if auto_resample or pilot_refit_surface_indices is not None:
        resample_debug_dir = utils.resolve_output_dir(plot_output_dir)

    plot_config = {}
    if plot_mode_normalized is not None:
        plot_config = dict(plot_kwargs) if plot_kwargs else {}
        for key in ("mode", "surface_indices"):
            plot_config.pop(key, None)
        if plot_output_dir is not None:
            plot_config["output_dir"] = plot_output_dir
        if plot_positions is not None:
            plot_config["positions"] = plot_positions
        if plot_show is not None:
            plot_config["show"] = plot_show
    prev_was_free_space = False
    for i, surface in enumerate(surfaces):
        axis_state = axis_states[i]
        entrance_axis = axis_state["entrance"]
        exit_axis = axis_state["exit"]
        interaction_info = chief_ray_data[i].get("interaction_info")
        surface_overrides = chief_ray_data[i].get("surface_overrides")

        # [FEATURE] Direct free-space propagation from last effective surface
        is_free_space = not _should_trace_element(surface, current_material)
        next_is_free_space = False
        if merge_free_space_surfaces and is_free_space and i < len(surfaces) - 1:
            next_surface = surfaces[i + 1]
            next_is_free_space = not _should_trace_element(next_surface, current_material)
        in_free_space_run = merge_free_space_surfaces and is_free_space and (
            prev_was_free_space or next_is_free_space
        )

        propagation_base = current_state
        if in_free_space_run:
            propagation_base = _clone_state_with_wfo(current_state)

        distance_to_entrance = propagation.free_space._compute_signed_distance(
            propagation_base.optical_axis_state, entrance_axis
        )
        entrance_trace = _build_trace_context(
            from_state=current_state,
            to_surface=surface,
            to_position="entrance",
            propagation_type="free_space_entrance",
            distance_mm=distance_to_entrance,
            surface_overrides=surface_overrides,
        )
        if print_status:
            log_fn(f"[POP] {entrance_trace['status_line']}")
        entrance_state = propagation.free_space.propagate_state(
            propagation_base,
            entrance_axis,
            target_surface_index=surface.index,
            target_position="entrance",
            trace_context=entrance_trace,
            force_asm=force_asm,
            auto_asm=auto_asm,
            auto_resample=auto_resample,
            resample_min_beam_pixels=resample_min_beam_pixels,
            resample_beam_pixels_target=resample_beam_pixels_target,
            pilot_refit_surface_indices=pilot_refit_surface_indices,
            pilot_refit_pv_threshold_waves=pilot_refit_pv_threshold_waves,
            auto_unwarp_at_incident=auto_unwarp_at_incident,
            auto_unwarp_surface_indices=auto_unwarp_surface_indices,
            debug_resample_dir=resample_debug_dir,
        )
        working_state = entrance_state

        if getattr(surface, "surface_type", "") == "paraxial":
            distance_mm = float(exit_axis.path_length - entrance_axis.path_length)
            paraxial_trace = _build_trace_context(
                from_state=working_state,
                to_surface=surface,
                to_position="exit",
                propagation_type="paraxial",
                distance_mm=distance_mm,
                surface_overrides=surface_overrides,
            )
            if print_status:
                log_fn(f"[POP] {paraxial_trace['status_line']}")
            working_state = _propagate_paraxial(
                working_state,
                surface,
                exit_axis,
                auto_unwarp_at_incident=auto_unwarp_at_incident,
                trace_context=paraxial_trace,
            )
        elif _should_trace_element(surface, current_material):
            distance_mm = float(exit_axis.path_length - entrance_axis.path_length)
            element_trace = _build_trace_context(
                from_state=working_state,
                to_surface=surface,
                to_position="exit",
                propagation_type="element",
                distance_mm=distance_mm,
                surface_overrides=surface_overrides,
            )
            if print_status:
                log_fn(f"[POP] {element_trace['status_line']}")
            working_state = propagation.element.propagate_element(
                working_state,
                surface,
                entrance_axis,
                exit_axis,
                target_surface_index=surface.index,
                num_rays=num_rays,
                incident_material_name=current_material,
                interaction_info=interaction_info,
                surface_overrides=surface_overrides,
                reconstruction_mask_ratio=reconstruction_mask_ratio,
                phase_method=phase_method,
                zernike_terms=zernike_terms,
                zernike_normalize=zernike_normalize,
                debug=debug_enabled,
                debug_store=debug_store,
                debug_plot_3d=debug_plot_3d,
                debug_plot_3d_dir=debug_plot_3d_dir,
                debug_plot_3d_show=debug_plot_3d_show,
                debug_plot_3d_ray_count=debug_plot_3d_ray_count,
                trace_context=element_trace,
                sampling_sigma=sampling_sigma,
                pilot_refit_surface_indices=pilot_refit_surface_indices,
                pilot_refit_pv_threshold_waves=pilot_refit_pv_threshold_waves,
                refit_debug_dir=resample_debug_dir,
                enable_ideal_planar_mirror=options.enable_ideal_planar_mirror,
            )
        else:
            distance_to_exit = propagation.free_space._compute_signed_distance(
                working_state.optical_axis_state, exit_axis
            )
            exit_trace = _build_trace_context(
                from_state=working_state,
                to_surface=surface,
                to_position="exit",
                propagation_type="free_space_exit",
                distance_mm=distance_to_exit,
                surface_overrides=surface_overrides,
            )
            if print_status:
                log_fn(f"[POP] {exit_trace['status_line']}")
            working_state = propagation.free_space.propagate_state(
                working_state,
                exit_axis,
                target_surface_index=surface.index,
                target_position="exit",
                trace_context=exit_trace,
                force_asm=force_asm,
                auto_asm=auto_asm,
                auto_resample=auto_resample,
                resample_min_beam_pixels=resample_min_beam_pixels,
                resample_beam_pixels_target=resample_beam_pixels_target,
                pilot_refit_surface_indices=pilot_refit_surface_indices,
                pilot_refit_pv_threshold_waves=pilot_refit_pv_threshold_waves,
                auto_unwarp_at_incident=auto_unwarp_at_incident,
                auto_unwarp_surface_indices=auto_unwarp_surface_indices,
                debug_resample_dir=resample_debug_dir,
            )

        surface_states.append(working_state)
        surface_records.append(
            SurfaceRecord(
                index=surface.index,
                surface=surface,
                entrance=entrance_state,
                exit=working_state,
                entrance_axis=entrance_axis,
                exit_axis=exit_axis,
                interaction_info=interaction_info,
                surface_overrides=surface_overrides,
                debug=None if debug_store is None else debug_store.get(surface.index),
            )
        )
        if plot_mode_normalized is not None:
            if plot_surface_index_set is None or surface.index in plot_surface_index_set:
                progressive_result = PropagationResult(
                    final_state=working_state,
                    surface_states=surface_states,
                    total_path_length=exit_axis.path_length,
                    surfaces=surface_records,
                )
                progressive_result.save_visualizations(
                    mode=plot_mode_normalized,
                    surface_indices=[surface.index],
                    **plot_config,
                )

        if not in_free_space_run or i == len(surfaces) - 1:
            current_state = working_state
        if not surface.is_mirror and _should_trace_element(surface, current_material):
            current_material = _normalize_material_name(surface.material)
        prev_was_free_space = is_free_space

    total_path_length = axis_states[-1]["exit"].path_length if axis_states else 0.0
    return PropagationResult(
        final_state=current_state,
        surface_states=surface_states,
        total_path_length=total_path_length,
        surfaces=surface_records,
    )


__all__ = [
    "GaussianSource",
    "System",
    "PropagationResult",
    "PropagationOptions",
    "PlotOptions",
    "DebugOptions",
    "CustomSource",
    "check_vendor_imports",
    "configure_vendor_paths",
    "coordinates",
    "core",
    "io",
    "load_zmx",
    "propagation",
    "propagate",
    "source",
    "utils",
    "visualization",
    "wavefront",
]
