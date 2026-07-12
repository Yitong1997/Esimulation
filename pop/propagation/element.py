"""Element propagation interface (hybrid ray tracing)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, MutableMapping, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from pop.core import GridSampling, OpticalAxisState, PilotBeamParams, PropagationState
from pop.coordinates.transforms import (
    build_surface_interaction_info,
    rotation_matrix_to_euler,
    transform_rays_to_local,
)
from pop.propagation.free_space import (
    _check_residual_phase_range,
    _compute_proper_reference_phase,
    _sync_proper_gaussian_params,
)
from pop.propagation.pilot_beam import apply_mirror, apply_refraction
from pop.propagation.ideal import propagate_ideal_mirror
from pop.reference_frames import snapshot_reference_frame
from pop.wavefront.reconstructor import reconstruct_wavefront
from pop.wavefront.sampler import sample_rays_from_wavefront
from pop.result import SurfaceDebugInfo


def _pilot_debug_enabled(surface_index: int) -> bool:
    flag = os.getenv("POP_PILOT_DEBUG", "")
    if flag.strip().lower() not in ("1", "true", "yes", "on"):
        return False
    raw_surfaces = os.getenv("POP_PILOT_DEBUG_SURFACES", "1,2").strip()
    if not raw_surfaces:
        return True
    allowed: set[int] = set()
    for token in raw_surfaces.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            allowed.add(int(token))
        except ValueError:
            continue
    if not allowed:
        return True
    return surface_index in allowed


def _format_pilot_radius(r: float) -> str:
    """格式化 Pilot 光束半径为科学计数法。R=inf 显示为 0。"""
    if np.isinf(r) or abs(r) > 1e15:
        return "0.00000E+00"
    return f"{r:.5E}"


def _log_pilot_update_debug(
    *,
    surface: Any,
    surface_index: int,
    entrance_axis: OpticalAxisState,
    exit_axis: Optional[OpticalAxisState],
    interaction_point: NDArray[np.floating],
    orientation: NDArray[np.floating],
    radius_input_mm: float,
    sign_factor: float,
    alignment: float,
    local_point_mm: NDArray[np.floating],
    effective_surface_radius_mm: float,
    effective_abcd_radius_mm: float,
    pilot_before: PilotBeamParams,
    pilot_after: PilotBeamParams,
    transform_type: str,
    orientation_override_applied: bool,
    radius_override_applied: bool,
    radius_x_input_mm: float | None = None,
    interaction_info_sign_factor: float | None = None,
    interaction_info_normal_flipped: bool | None = None,
) -> None:
    # === Zemax 风格输出 ===
    before_frame = np.asarray(entrance_axis.frame, dtype=float)
    after_frame = np.asarray(exit_axis.frame, dtype=float) if exit_axis is not None else before_frame

    R_before = pilot_before.curvature_radius_mm
    R_after = pilot_after.curvature_radius_mm

    print(f"\n从在 {surface_index}之前 到在 {surface_index}之后 的表面传输")
    print()
    print("之前方向矩阵:")
    print()
    for row in range(3):
        print(f"{before_frame[row, 0]:.9f} {before_frame[row, 1]:.9f} {before_frame[row, 2]:.9f}")
    print()
    print(f"入射Pilot光束半径x, y: {_format_pilot_radius(R_before)} {_format_pilot_radius(R_before)}")
    print()
    print(f"出射Pilot光束半径x, y: {_format_pilot_radius(R_after)} {_format_pilot_radius(R_after)}")
    print()
    print("之后方向矩阵:")
    print()
    for row in range(3):
        print(f"{after_frame[row, 0]:.9f} {after_frame[row, 1]:.9f} {after_frame[row, 2]:.9f}")
    print()
    print(
        f"[POP][PilotDebug] transform={transform_type}, alignment={alignment:+.9f}, "
        f"sign_factor={sign_factor:+.1f}, info_sign={interaction_info_sign_factor}"
    )
    print(
        f"[POP][PilotDebug] orientation_override={orientation_override_applied}, "
        f"radius_override={radius_override_applied}, normal_flipped={interaction_info_normal_flipped}"
    )
    print(
        f"[POP][PilotDebug] interaction_point_mm=({interaction_point[0]:.6f}, {interaction_point[1]:.6f}, {interaction_point[2]:.6f}), "
        f"local_point_mm=({local_point_mm[0]:.6f}, {local_point_mm[1]:.6f}, {local_point_mm[2]:.6f})"
    )
    print(
        f"[POP][PilotDebug] radius_in_mm={radius_input_mm:.6f}, "
        f"effective_surface_R_mm={effective_surface_radius_mm:.6f}, "
        f"effective_abcd_R_mm={effective_abcd_radius_mm:.6f}"
    )
    if radius_x_input_mm is not None:
        print(f"[POP][PilotDebug] radius_x_in_mm={radius_x_input_mm:.6f}")
    print()


def _get_refit_core_thresholds() -> tuple[float, ...]:
    """Return candidate core-intensity thresholds for pilot refit.

    Environment override:
    - POP_PILOT_REFIT_CORE_THRESHOLDS="0.8,0.7,0.6,0.5"
    """
    default = (0.80, 0.70, 0.60, 0.50)
    raw = os.getenv("POP_PILOT_REFIT_CORE_THRESHOLDS", "").strip()
    if not raw:
        return default

    parsed: list[float] = []
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError:
            continue
        if 0.0 < value <= 1.0:
            parsed.append(value)
    return tuple(parsed) if parsed else default


def _get_refractive_index(material: str, wavelength_um: float) -> float:
    material_lower = material.lower()
    if material_lower in ("air", "", "mirror"):
        return 1.0
    try:
        from optiland.materials.material import Material

        mat = Material(material)
        return float(mat.n(wavelength_um))
    except Exception:
        fallback_indices = {
            "n-bk7": 1.5168,
            "bk7": 1.5168,
            "fused_silica": 1.4585,
            "fs": 1.4585,
            "sf11": 1.7847,
            "laf2": 1.7440,
        }
        return fallback_indices.get(material_lower, 1.5)


def _apply_ray_mask(rays, mask: NDArray[np.bool_]) -> None:
    mask = np.asarray(mask, dtype=bool).ravel()
    attrs = ("x", "y", "z", "L", "M", "N", "opd", "i", "intensity", "w", "L0", "M0", "N0")
    for attr in attrs:
        if not hasattr(rays, attr):
            continue
        value = getattr(rays, attr)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim == 0:
            continue
        if arr.shape[0] != mask.shape[0]:
            continue
        setattr(rays, attr, arr[mask])


def _trace_to_exit_plane(rays, exit_axis: OpticalAxisState, n_out: float) -> None:
    p0 = np.asarray(exit_axis.position, dtype=float)
    n = np.asarray(exit_axis.direction, dtype=float)

    x = np.asarray(rays.x)
    y = np.asarray(rays.y)
    z = np.asarray(rays.z)
    l = np.asarray(rays.L)
    m = np.asarray(rays.M)
    n_dir = np.asarray(rays.N)

    origins = np.column_stack([x, y, z])
    directions = np.column_stack([l, m, n_dir])

    denom = np.dot(directions, n)
    plane_offset = np.dot(p0 - origins, n)
    parallel_mask = np.abs(denom) < 1e-12
    on_plane_mask = np.isclose(plane_offset, 0.0, atol=1e-9)
    invalid_mask = parallel_mask & ~on_plane_mask
    if np.any(invalid_mask):
        valid_mask = ~invalid_mask
        if not np.any(valid_mask):
            raise ValueError("All rays are parallel to the exit plane and do not intersect it.")
    else:
        valid_mask = None

    denom_safe = np.where(parallel_mask, np.nan, denom)
    t = plane_offset / denom_safe
    t = np.where(np.isnan(t), 0.0, t)

    new_positions = origins + directions * t[:, None]
    if valid_mask is not None:
        _apply_ray_mask(rays, valid_mask)
        new_positions = new_positions[valid_mask]
        t = t[valid_mask]

    rays.x = new_positions[:, 0]
    rays.y = new_positions[:, 1]
    rays.z = new_positions[:, 2]

    opd_increment = n_out * t
    if hasattr(rays, "opd"):
        rays.opd = np.asarray(rays.opd) + opd_increment
    else:
        rays.opd = opd_increment


def _trace_with_signed_opd(surface, rays) -> None:
    if not hasattr(rays, "opd") or rays.opd is None:
        rays.opd = np.zeros(len(rays.x))

    opd_before = np.asarray(rays.opd).copy()
    x_before = np.asarray(rays.x).copy()
    y_before = np.asarray(rays.y).copy()
    z_before = np.asarray(rays.z).copy()
    L_before = np.asarray(rays.L).copy()
    M_before = np.asarray(rays.M).copy()
    N_before = np.asarray(rays.N).copy()

    surface.trace(rays)

    x_after = np.asarray(rays.x)
    y_after = np.asarray(rays.y)
    z_after = np.asarray(rays.z)
    opd_after = np.asarray(rays.opd)

    opd_increment_abs = opd_after - opd_before
    dx = x_after - x_before
    dy = y_after - y_before
    dz = z_after - z_before
    dot_product = dx * L_before + dy * M_before + dz * N_before
    sign_t = np.sign(dot_product)
    sign_t[sign_t == 0] = 1.0

    rays.opd = opd_before + sign_t * opd_increment_abs


def _rays_to_dict(rays) -> dict[str, np.ndarray]:
    def _arr(value):
        if value is None:
            return None
        return np.asarray(value).copy()

    intensity = getattr(rays, "i", None)
    if intensity is None:
        intensity = getattr(rays, "intensity", None)

    return {
        "x": _arr(getattr(rays, "x", None)),
        "y": _arr(getattr(rays, "y", None)),
        "z": _arr(getattr(rays, "z", None)),
        "L": _arr(getattr(rays, "L", None)),
        "M": _arr(getattr(rays, "M", None)),
        "N": _arr(getattr(rays, "N", None)),
        "intensity": _arr(intensity),
        "opd": _arr(getattr(rays, "opd", None)),
    }


def _build_optiland_surface_group(
    surface,
    entrance_axis: OpticalAxisState,
    exit_axis: OpticalAxisState | None,
    incident_material_name: str | None,
    pilot_params: PilotBeamParams | None,
    interaction_info: SurfaceInteractionInfo | None = None,
    orientation_override: NDArray[np.floating] | None = None,
    radius_override: float | None = None,
    radius_x_override: float | None = None,
    asphere_coeffs_override: list[float] | None = None,
    focal_length_override: float | None = None,
    include_exit_plane: bool = True,
):
    from optiland.materials import IdealMaterial
    from optiland.surfaces.surface_group import SurfaceGroup

    group = SurfaceGroup()

    if incident_material_name and incident_material_name.strip():
        if (
            pilot_params is not None
            and incident_material_name.strip().lower() == "air"
            and abs(pilot_params.current_refractive_index - 1.0) > 1e-6
        ):
            incident_material = IdealMaterial(
                n=float(pilot_params.current_refractive_index)
            )
        else:
            incident_material = incident_material_name
    else:
        if pilot_params is None:
            incident_material = "air"
        else:
            incident_material = IdealMaterial(
                n=float(pilot_params.current_refractive_index)
            )

    entrance_euler = getattr(entrance_axis, "euler", None)
    if entrance_euler is not None:
        entrance_rx, entrance_ry, entrance_rz = entrance_euler
    else:
        entrance_rx, entrance_ry, entrance_rz = rotation_matrix_to_euler(
            entrance_axis.frame
        )
    group.add_surface(
        index=0,
        surface_type="standard",
        radius=np.inf,
        material=incident_material,
        x=float(entrance_axis.position[0]),
        y=float(entrance_axis.position[1]),
        z=float(entrance_axis.position[2]),
        rx=float(entrance_rx),
        ry=float(entrance_ry),
        rz=float(entrance_rz),
    )

    if surface.is_mirror:
        material = "mirror"
    elif not surface.material or surface.material.lower() == "air":
        material = "air"
    else:
        material = surface.material

    orientation = np.asarray(
        orientation_override if orientation_override is not None else surface.orientation,
        dtype=float,
    )
    radius = float(radius_override if radius_override is not None else surface.radius)
    radius_x = (
        float(radius_x_override)
        if radius_x_override is not None
        else float(getattr(surface, "radius_x", np.inf))
    )
    if surface.is_mirror and interaction_info is None:
        interaction_info = build_surface_interaction_info(
            surface_index=surface.index,
            orientation=orientation,
            entrance_dir=entrance_axis.direction,
            is_mirror=True,
        )
        if interaction_info.normal_flipped and interaction_info.canonical_orientation is not None:
            orientation = interaction_info.canonical_orientation
            if np.isfinite(radius):
                radius = -radius
            if np.isfinite(radius_x):
                radius_x = -radius_x

    rx, ry, rz = rotation_matrix_to_euler(orientation)
    params: dict[str, object] = {
        "index": 1,
        "material": material,
        "x": float(surface.vertex_position[0]),
        "y": float(surface.vertex_position[1]),
        "z": float(surface.vertex_position[2]),
        "rx": float(rx),
        "ry": float(ry),
        "rz": float(rz),
    }

    surface_type = surface.surface_type
    if surface_type == "paraxial":
        params["surface_type"] = "paraxial"
        f_val = surface.focal_length
        if focal_length_override is not None:
            f_val = focal_length_override
        params["f"] = float(f_val)
    elif surface_type == "biconic":
        params["surface_type"] = "biconic"
        params["radius_x"] = float(radius_x)
        params["radius_y"] = float(radius)
        params["conic_x"] = float(surface.conic_x)
        params["conic_y"] = float(surface.conic)
    elif surface_type == "even_asphere":
        params["surface_type"] = "even_asphere"
        params["radius"] = float(radius)
        params["conic"] = float(surface.conic)
        if asphere_coeffs_override is not None:
            params["coefficients"] = list(asphere_coeffs_override)
        else:
            params["coefficients"] = list(surface.asphere_coeffs or [])
    else:
        params["surface_type"] = "standard"
        params["radius"] = float(radius)
        params["conic"] = float(surface.conic)

    # NOTE: semi_aperture is intentionally ignored for now in POP; avoid clipping rays in optiland.

    group.add_surface(**params)

    if include_exit_plane:
        if exit_axis is None:
            raise ValueError("exit_axis is required when include_exit_plane is True")
        exit_euler = getattr(exit_axis, "euler", None)
        if exit_euler is not None:
            exit_rx, exit_ry, exit_rz = exit_euler
        else:
            exit_rx, exit_ry, exit_rz = rotation_matrix_to_euler(exit_axis.frame)
        exit_material = group.surfaces[1].material_post
        group.add_surface(
            index=2,
            surface_type="standard",
            radius=np.inf,
            material=exit_material,
            x=float(exit_axis.position[0]),
            y=float(exit_axis.position[1]),
            z=float(exit_axis.position[2]),
            rx=float(exit_rx),
            ry=float(exit_ry),
            rz=float(exit_rz),
        )
    return group


def _compute_absolute_opd_waves(rays_local, wavelength_mm: float) -> NDArray[np.floating]:
    opd_mm = np.asarray(rays_local.opd)
    r_sq = np.asarray(rays_local.x) ** 2 + np.asarray(rays_local.y) ** 2
    chief_idx = int(np.argmin(r_sq))
    relative_opd_mm = opd_mm - opd_mm[chief_idx]
    return relative_opd_mm / wavelength_mm


def _compute_pilot_opd_waves(
    rays_local,
    pilot_params: PilotBeamParams,
    wavelength_mm: float,
) -> NDArray[np.floating]:
    del wavelength_mm  # Kept for the existing call signature.
    r_sq = np.asarray(rays_local.x) ** 2 + np.asarray(rays_local.y) ** 2
    pilot_opd_waves = pilot_params.compute_phase_from_radius_squared(r_sq) / (2.0 * np.pi)
    chief_idx = int(np.argmin(r_sq))
    return pilot_opd_waves - pilot_opd_waves[chief_idx]


def _amplitude_phase_to_proper(
    wfo: Any,
    amplitude: NDArray[np.floating],
    phase: NDArray[np.floating],
    grid_sampling: GridSampling,
    trace_context: Optional[dict[str, Any]] = None,
) -> None:
    import proper

    ref_phase = _compute_proper_reference_phase(wfo, grid_sampling)
    residual_phase = phase - ref_phase
    _check_residual_phase_range(
        residual_phase,
        amplitude,
        grid_sampling=grid_sampling,
        trace_context=trace_context,
    )
    complex_amplitude = amplitude * np.exp(1j * residual_phase)
    wfo.wfarr = proper.prop_shift_center(complex_amplitude)


def _ensure_rays_opd(rays) -> None:
    if not hasattr(rays, "opd") or rays.opd is None:
        rays.opd = np.zeros(len(rays.x), dtype=float)


def _global_points_to_local(
    points_global: NDArray[np.floating],
    origin: NDArray[np.floating],
    orientation: NDArray[np.floating],
) -> NDArray[np.floating]:
    origin = np.asarray(origin, dtype=float)
    orientation = np.asarray(orientation, dtype=float)
    return (orientation.T @ (points_global - origin).T).T


def _get_sag_orientation(
    surface: Any,
    interaction_info: SurfaceInteractionInfo | None,
    orientation_override: NDArray[np.floating] | None,
) -> NDArray[np.floating]:
    if orientation_override is not None:
        return np.asarray(orientation_override, dtype=float)
    if interaction_info is not None and interaction_info.canonical_orientation is not None:
        return np.asarray(interaction_info.canonical_orientation, dtype=float)
    return np.asarray(surface.orientation, dtype=float)


def _apply_surface_sag_to_opd(
    rays,
    surface: Any,
    interaction_info: SurfaceInteractionInfo | None,
    pilot_params: PilotBeamParams,
    surface_points_global: NDArray[np.floating],
    orientation_override: NDArray[np.floating] | None = None,
) -> None:
    sag_fn = getattr(surface, "sag_fn", None)
    if sag_fn is None:
        return
    if not callable(sag_fn):
        raise TypeError("surface.sag_fn must be callable if provided")
    if surface_points_global.size == 0:
        return

    orientation = _get_sag_orientation(surface, interaction_info, orientation_override)
    local = _global_points_to_local(
        surface_points_global,
        np.asarray(surface.vertex_position, dtype=float),
        orientation,
    )
    x_local = local[:, 0]
    y_local = local[:, 1]

    sag_mm = np.asarray(sag_fn(x_local, y_local), dtype=float)
    if sag_mm.ndim == 0:
        sag_mm = np.full_like(x_local, float(sag_mm))
    if sag_mm.shape != x_local.shape:
        raise ValueError("surface.sag_fn must return an array matching x/y shape")

    sag_reference = getattr(surface, "sag_reference", "incident")
    sag_reference = str(sag_reference).strip().lower()
    if sag_reference in ("incident", "incoming"):
        # User-defined sag is positive toward the incident side (opposite to entrance_dir).
        sag_mm = -sag_mm
    elif sag_reference in ("normal", "surface"):
        # Positive sag follows the local +Z axis used for ray tracing.
        pass
    elif sag_reference in ("opd", "opd_mm"):
        opd_mm = sag_mm
        _ensure_rays_opd(rays)
        rays.opd = np.asarray(rays.opd) + opd_mm
        return
    else:
        raise ValueError(f"Unknown surface.sag_reference value: {sag_reference!r}")

    n1 = float(pilot_params.current_refractive_index or 1.0)
    if surface.is_mirror:
        opd_mm = 2.0 * n1 * sag_mm
    else:
        n2 = _get_refractive_index(surface.material, pilot_params.wavelength_um)
        opd_mm = (n2 - n1) * sag_mm
    _ensure_rays_opd(rays)
    rays.opd = np.asarray(rays.opd) + opd_mm


def _axis_tangential_radius(radius: float, conic: float, d_axis_abs: float) -> float:
    """Return meridional radius for a conic section along one principal axis."""
    if not np.isfinite(radius):
        return np.inf
    if abs(radius) < 1e-12:
        return np.inf
    ratio = d_axis_abs / radius
    factor = 1.0 - conic * ratio * ratio
    if factor <= 0.0:
        return np.inf
    return radius * (factor ** 1.5)


def _incident_tangential_biconic_radius(
    *,
    radius_x: float,
    conic_x: float,
    radius_y: float,
    conic_y: float,
    local_point_mm: NDArray[np.floating],
    entrance_dir_local: NDArray[np.floating],
) -> float:
    """Project biconic principal curvatures onto the incident tangential direction."""
    dx_abs = abs(float(local_point_mm[0]))
    dy_abs = abs(float(local_point_mm[1]))
    rx_t = _axis_tangential_radius(radius_x, conic_x, dx_abs)
    ry_t = _axis_tangential_radius(radius_y, conic_y, dy_abs)

    tangent_xy = np.asarray([entrance_dir_local[0], entrance_dir_local[1]], dtype=float)
    tangent_norm = float(np.linalg.norm(tangent_xy))
    if tangent_norm < 1e-12:
        # Near-normal incidence: keep compatibility with historical Y-only behavior.
        u = np.array([0.0, 1.0], dtype=float)
    else:
        u = tangent_xy / tangent_norm

    kx = 0.0 if np.isinf(rx_t) else 1.0 / rx_t
    ky = 0.0 if np.isinf(ry_t) else 1.0 / ry_t
    k_tangent = kx * (u[0] ** 2) + ky * (u[1] ** 2)
    if abs(k_tangent) < 1e-15:
        return np.inf
    return 1.0 / k_tangent


def _effective_biconic_mirror_radius_for_abcd(
    *,
    radius_x: float,
    conic_x: float,
    radius_y: float,
    conic_y: float,
    local_point_mm: NDArray[np.floating],
    entrance_dir_local: NDArray[np.floating],
) -> float:
    """Return effective mirror radius including oblique-incidence tangential scaling."""
    tangential_surface_radius = _incident_tangential_biconic_radius(
        radius_x=radius_x,
        conic_x=conic_x,
        radius_y=radius_y,
        conic_y=conic_y,
        local_point_mm=local_point_mm,
        entrance_dir_local=entrance_dir_local,
    )
    if np.isinf(tangential_surface_radius):
        return np.inf

    # Tangential mirror power scales with 1/cos(i), so ABCD-equivalent radius is R*cos(i).
    cos_incidence = abs(float(entrance_dir_local[2]))
    if cos_incidence < 1e-12:
        return np.inf
    return tangential_surface_radius * cos_incidence


def _update_pilot_beam(
    pilot_params: PilotBeamParams,
    surface: Any,
    entrance_axis: OpticalAxisState,
    surface_index: int,
    interaction_point: NDArray[np.floating],
    interaction_info: SurfaceInteractionInfo | None = None,
    orientation_override: NDArray[np.floating] | None = None,
    radius_override: float | None = None,
    radius_x_override: float | None = None,
    exit_axis: Optional[OpticalAxisState] = None,
) -> PilotBeamParams:
    pilot_debug = _pilot_debug_enabled(surface_index)
    orientation = np.asarray(
        orientation_override if orientation_override is not None else surface.orientation,
        dtype=float,
    )
    radius = float(radius_override if radius_override is not None else surface.radius)
    radius_x = float(
        radius_x_override
        if radius_x_override is not None
        else getattr(surface, "radius_x", np.inf)
    )
    surface_type = str(getattr(surface, "surface_type", "") or "").strip().lower()

    if interaction_info is None:
        interaction_info = build_surface_interaction_info(
            surface_index=surface_index,
            orientation=orientation,
            entrance_dir=entrance_axis.direction,
            is_mirror=surface.is_mirror,
        )

    entrance_dir = np.asarray(entrance_axis.direction, dtype=float)
    surface_normal = orientation[:, 2]
    alignment = float(np.dot(entrance_dir, surface_normal))
    raw_sign_factor = 1.0 if alignment >= 0 else -1.0
    info_sign_factor = float(getattr(interaction_info, "sign_factor", raw_sign_factor))
    sign_factor = raw_sign_factor
    if np.isfinite(info_sign_factor) and abs(abs(info_sign_factor) - 1.0) < 1e-6:
        sign_factor = 1.0 if info_sign_factor >= 0.0 else -1.0
    info_normal_flipped = bool(getattr(interaction_info, "normal_flipped", False))
    interaction_point = np.asarray(interaction_point, dtype=float)
    vertex = np.asarray(surface.vertex_position, dtype=float)
    delta = interaction_point - vertex
    local_point = orientation.T @ delta
    d_off = float(np.sqrt(local_point[0] ** 2 + local_point[1] ** 2))

    # Surface overrides generated from interaction normalization may already
    # include a sign flip on radius terms. Recover geometric radii for the
    # local curvature model, then apply sign via sign_factor only once.
    radius_for_model = radius
    radius_x_for_model = radius_x
    if radius_override is not None and np.isfinite(radius_for_model) and abs(sign_factor) > 0.0:
        radius_for_model = radius_for_model / sign_factor
    if (
        radius_x_override is not None
        and np.isfinite(radius_x_for_model)
        and abs(sign_factor) > 0.0
    ):
        radius_x_for_model = radius_x_for_model / sign_factor

    if surface.is_mirror:
        r = radius_for_model
        effective_surface_radius = r
        if surface_type == "biconic":
            entrance_dir_local = orientation.T @ entrance_dir
            effective_surface_radius = _effective_biconic_mirror_radius_for_abcd(
                radius_x=radius_x_for_model,
                conic_x=float(getattr(surface, "conic_x", 0.0)),
                radius_y=r,
                conic_y=float(getattr(surface, "conic", 0.0)),
                local_point_mm=local_point,
                entrance_dir_local=entrance_dir_local,
            )
            updated = apply_mirror(pilot_params, effective_surface_radius, sign_factor=sign_factor)
        elif abs(surface.conic + 1.0) < 1e-10 and not np.isinf(r):
            effective_surface_radius = r + (d_off ** 2) / r
            updated = apply_mirror(
                pilot_params,
                r,
                d_off_axis_mm=d_off,
                sign_factor=sign_factor,
            )
        else:
            updated = apply_mirror(pilot_params, r, sign_factor=sign_factor)

        if pilot_debug:
            _log_pilot_update_debug(
                surface=surface,
                surface_index=surface_index,
                entrance_axis=entrance_axis,
                exit_axis=exit_axis,
                interaction_point=interaction_point,
                orientation=orientation,
                radius_input_mm=r,
                sign_factor=sign_factor,
                alignment=alignment,
                local_point_mm=local_point,
                effective_surface_radius_mm=effective_surface_radius,
                effective_abcd_radius_mm=effective_surface_radius * sign_factor,
                pilot_before=pilot_params,
                pilot_after=updated,
                transform_type="mirror",
                orientation_override_applied=orientation_override is not None,
                radius_override_applied=radius_override is not None,
                radius_x_input_mm=radius_x_for_model,
                interaction_info_sign_factor=info_sign_factor,
                interaction_info_normal_flipped=info_normal_flipped,
            )
        return updated

    n1 = pilot_params.current_refractive_index
    n2 = _get_refractive_index(surface.material, pilot_params.wavelength_um)
    updated = apply_refraction(pilot_params, radius_for_model, n1, n2, sign_factor=sign_factor)
    if pilot_debug:
        _log_pilot_update_debug(
            surface=surface,
            surface_index=surface_index,
            entrance_axis=entrance_axis,
            exit_axis=exit_axis,
            interaction_point=interaction_point,
            orientation=orientation,
            radius_input_mm=radius_for_model,
            sign_factor=sign_factor,
            alignment=alignment,
            local_point_mm=local_point,
            effective_surface_radius_mm=radius_for_model,
            effective_abcd_radius_mm=radius_for_model * sign_factor,
            pilot_before=pilot_params,
            pilot_after=updated,
            transform_type="refraction",
            orientation_override_applied=orientation_override is not None,
            radius_override_applied=radius_override is not None,
            radius_x_input_mm=radius_x_for_model,
            interaction_info_sign_factor=info_sign_factor,
            interaction_info_normal_flipped=info_normal_flipped,
        )
    return updated


def _refit_pilot_beam_from_rays(
    exit_local_rays,
    absolute_opd_waves: NDArray[np.floating],
    wavelength_mm: float,
    current_refractive_index: float,
    wavelength_um: float,
    target_w_mm: Optional[float] = None,
) -> Optional[PilotBeamParams]:
    """Fit a new PilotBeamParams from exit-plane ray data (curvature + spot size).

    Returns None if the fit cannot be performed (too few rays, degenerate data).
    """
    x = np.asarray(exit_local_rays.x, dtype=float)
    y = np.asarray(exit_local_rays.y, dtype=float)
    raw_intensity = getattr(exit_local_rays, "i", None)
    if raw_intensity is None:
        raw_intensity = getattr(exit_local_rays, "intensity", None)
    if raw_intensity is None:
        return None
    intensity = np.asarray(raw_intensity, dtype=float)
    r_sq = x ** 2 + y ** 2

    finite_mask = np.isfinite(r_sq) & np.isfinite(absolute_opd_waves) & np.isfinite(intensity)
    if np.count_nonzero(finite_mask) < 10:
        return None

    x_v = x[finite_mask]
    y_v = y[finite_mask]
    r_sq_v = r_sq[finite_mask]
    opd_v = absolute_opd_waves[finite_mask]
    intensity_v = intensity[finite_mask]

    # Fit curvature on the high-intensity core first.
    # Edge rays carry stronger high-order aberration terms and can bias a
    # global r^2 fit, which shows up as a systematic defocus offset in pilot R.
    fit_mask = np.ones_like(intensity_v, dtype=bool)
    peak_i = float(np.max(intensity_v))
    if np.isfinite(peak_i) and peak_i > 0.0:
        min_fit_points = max(12, int(0.01 * intensity_v.size))
        for frac in _get_refit_core_thresholds():
            candidate = intensity_v >= (frac * peak_i)
            if np.count_nonzero(candidate) >= min_fit_points:
                fit_mask = candidate
                break

    x_fit = x_v[fit_mask]
    y_fit = y_v[fit_mask]
    r_sq_fit = r_sq_v[fit_mask]
    opd_fit = opd_v[fit_mask]
    intensity_fit = intensity_v[fit_mask]

    # --- Fit curvature R from OPD vs r^2 ---
    # OPD_waves = n * r^2 / (2 * R * wavelength_mm)  + tilt + piston
    # Phase_rad = 2*pi * OPD_waves = k * r^2 / (2R) + ...
    # We fit: OPD_waves = a * r^2 + b * x + c * y + d
    A = np.column_stack([r_sq_fit, x_fit, y_fit, np.ones_like(r_sq_fit)])
    weights = np.sqrt(np.maximum(intensity_fit, 0.0))
    w_sum = np.sum(weights)
    if w_sum < 1e-12:
        return None
    Aw = A * weights[:, None]
    bw = opd_fit * weights
    try:
        coeffs, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
    except np.linalg.LinAlgError:
        return None

    a_coeff = coeffs[0]  # OPD_waves = a * r^2 + ...
    # a = n / (2 * R * wavelength_mm)  =>  R = n / (2 * a * wavelength_mm)
    n = current_refractive_index
    if abs(a_coeff) < 1e-15:
        R_fit = np.inf
    else:
        R_fit = n / (2.0 * a_coeff * wavelength_mm)

    # --- Fit spot size w from intensity-weighted second moments ---
    total_i = np.sum(intensity_v)
    if total_i < 1e-20:
        return None
    cx = np.sum(x_v * intensity_v) / total_i
    cy = np.sum(y_v * intensity_v) / total_i
    var_x = np.sum((x_v - cx) ** 2 * intensity_v) / total_i
    var_y = np.sum((y_v - cy) ** 2 * intensity_v) / total_i
    sigma = np.sqrt(max(0.5 * (var_x + var_y), 0.0))
    # For a Gaussian beam: I ~ exp(-2*r^2/w^2), sigma = w / 2
    w_fit = 2.0 * sigma
    if w_fit < 1e-12:
        return None

    # --- Build q-parameter and new PilotBeamParams ---
    lambda_mm = wavelength_um * 1e-3
    if np.isinf(R_fit):
        inv_q_real = 0.0
    else:
        inv_q_real = 1.0 / R_fit
    
    # Use target_w_mm if provided (Force w conservation), else use fitted w
    w_used = target_w_mm if target_w_mm is not None else w_fit
    
    inv_q_imag = -lambda_mm / (np.pi * n * w_used ** 2)
    inv_q = inv_q_real + 1j * inv_q_imag
    if abs(inv_q) < 1e-30:
        return None
    q_new = 1.0 / inv_q

    return PilotBeamParams.from_q_parameter(q_new, wavelength_um, n)


def propagate_element(
    state: PropagationState,
    surface: Any,
    entrance_axis: OpticalAxisState,
    exit_axis: OpticalAxisState,
    target_surface_index: int,
    num_rays: int = 1000,
    incident_material_name: str = "air",
    interaction_info: SurfaceInteractionInfo | None = None,
    surface_overrides: dict[str, object] | None = None,
    reconstruction_mask_ratio: Optional[float] = None,
    phase_method: str = "griddata",
    zernike_terms: Optional[int] = None,
    zernike_normalize: bool = True,
    debug: bool = False,
    debug_store: Optional[MutableMapping[int, SurfaceDebugInfo]] = None,
    debug_plot_3d: bool = False,
    debug_plot_3d_dir: Optional[str | Path] = None,
    debug_plot_3d_show: bool = True,
    debug_plot_3d_ray_count: int = 5,
    trace_context: Optional[dict[str, Any]] = None,
    sampling_sigma: float = 6.0,
    pilot_refit_surface_indices: Optional[Sequence[int]] = None,
    pilot_refit_pv_threshold_waves: float = 0.5,
    refit_debug_dir: Optional[str | Path] = None,
    enable_ideal_planar_mirror: bool = True,
    element_phase_mode: str = "pilot_only",
) -> PropagationState:
    if state.proper_wfo is None:
        raise ValueError("PropagationState.proper_wfo is required for element propagation")

    # --- Ideal Mirror Propagation Logic ---
    # Simplified handling for planar mirrors (tilted or not)
    # Checks coordinate space + optional phase, skipping ray tracing/sampling.
    eff_radius = float(surface.radius)
    if surface_overrides and "radius" in surface_overrides:
        eff_radius = float(surface_overrides["radius"])
        
    eff_focal_length = None
    if surface_overrides and "focal_length" in surface_overrides:
        eff_focal_length = float(surface_overrides["focal_length"])
        
    is_planar = np.isinf(eff_radius) or abs(eff_radius) > 1e10
    
    # Check if we should use ideal logic:
    # 1. Enabled flag
    # 2. Is a mirror
    # 3. Is planar (infinite radius)
    # 4. Not a specialized surface type that requires complex handling (checked generically)
    if enable_ideal_planar_mirror and surface.is_mirror and is_planar:
         # Note: We trust exit_axis has the correct reflected coordinate frame.
         # This is usually computed by the system trace before calling propagate_element.
         return propagate_ideal_mirror(
             state=state,
             entrance_axis=entrance_axis,
             exit_axis=exit_axis,
             target_surface_index=target_surface_index,
             focal_length_mm=eff_focal_length,
             debug=debug,
         )

    local_rays, global_rays = sample_rays_from_wavefront(
        amplitude=state.amplitude,
        phase=state.phase,
        grid_sampling=state.grid_sampling,
        entrance_axis=entrance_axis,
        pilot_beam_params=state.pilot_beam_params,
        num_rays=num_rays,
        sampling_sigma=sampling_sigma,
        element_phase_mode=element_phase_mode,
    )
    capture_debug = debug or debug_plot_3d
    entrance_local_rays = _rays_to_dict(local_rays) if debug else None
    entrance_global_rays = _rays_to_dict(global_rays) if capture_debug else None

    orientation_override = None
    radius_override = None
    radius_x_override = None
    asphere_coeffs_override = None
    focal_length_override = None
    if surface_overrides:
        orientation_override = surface_overrides.get("orientation")
        radius_override = surface_overrides.get("radius")
        radius_x_override = surface_overrides.get("radius_x")
        asphere_coeffs_override = surface_overrides.get("asphere_coeffs")
        focal_length_override = surface_overrides.get("focal_length")

    optiland_group = _build_optiland_surface_group(
        surface=surface,
        entrance_axis=entrance_axis,
        exit_axis=exit_axis,
        incident_material_name=incident_material_name,
        pilot_params=state.pilot_beam_params,
        interaction_info=interaction_info,
        orientation_override=orientation_override,
        radius_override=radius_override,
        radius_x_override=radius_x_override,
        asphere_coeffs_override=asphere_coeffs_override,
        focal_length_override=focal_length_override,
    )
    surface_rays_global = None
    for idx, traced_surface in enumerate(optiland_group.surfaces[1:], start=1):
        _trace_with_signed_opd(traced_surface, global_rays)
        if capture_debug and idx == 1:
            surface_rays_global = _rays_to_dict(global_rays)

    surface_points = optiland_group.surfaces[1]
    surface_points_global = np.column_stack(
        [
            np.asarray(getattr(surface_points, "x", []), dtype=float),
            np.asarray(getattr(surface_points, "y", []), dtype=float),
            np.asarray(getattr(surface_points, "z", []), dtype=float),
        ]
    )
    _apply_surface_sag_to_opd(
        rays=global_rays,
        surface=surface,
        interaction_info=interaction_info,
        pilot_params=state.pilot_beam_params,
        surface_points_global=surface_points_global,
        orientation_override=orientation_override,
    )

    new_pilot = _update_pilot_beam(
        pilot_params=state.pilot_beam_params,
        surface=surface,
        entrance_axis=entrance_axis,
        surface_index=target_surface_index,
        interaction_point=np.asarray(exit_axis.position, dtype=float),
        interaction_info=interaction_info,
        orientation_override=orientation_override,
        radius_override=radius_override,
        radius_x_override=radius_x_override,
        exit_axis=exit_axis,
    )
    _sync_proper_gaussian_params(state.proper_wfo, new_pilot, update_reference_surface=True)

    exit_global_rays = _rays_to_dict(global_rays) if capture_debug else None
    exit_local_rays = transform_rays_to_local(global_rays, exit_axis)
    exit_local_rays_dict = _rays_to_dict(exit_local_rays) if debug else None

    wavelength_mm = state.pilot_beam_params.wavelength_um * 1e-3
    absolute_opd_waves = _compute_absolute_opd_waves(exit_local_rays, wavelength_mm)
    pilot_opd_waves = _compute_pilot_opd_waves(exit_local_rays, new_pilot, wavelength_mm)
    residual_opd_waves = absolute_opd_waves - pilot_opd_waves
    if np.any(np.isfinite(residual_opd_waves)):
        # 以主光线（出射面局部坐标中心）为参考，避免均值去除导致中心偏移
        r_sq = np.asarray(exit_local_rays.x) ** 2 + np.asarray(exit_local_rays.y) ** 2
        finite_mask = np.isfinite(residual_opd_waves) & np.isfinite(r_sq)
        if np.any(finite_mask):
            chief_idx = int(np.argmin(r_sq[finite_mask]))
            chief_value = residual_opd_waves[finite_mask][chief_idx]
            residual_opd_waves = residual_opd_waves - chief_value

    ray_x_in = np.asarray(local_rays.x)
    ray_y_in = np.asarray(local_rays.y)
    ray_x_out = np.asarray(exit_local_rays.x)
    ray_y_out = np.asarray(exit_local_rays.y)
    input_amplitude = np.sqrt(np.asarray(local_rays.i))

    # --- Pilot beam refitting (if enabled) ---
    pilot_opd_waves_pre_refit: Optional[NDArray[np.floating]] = None
    residual_opd_waves_pre_refit: Optional[NDArray[np.floating]] = None
    refit_occurred: bool = False
    refit_info: Optional[dict[str, object]] = None

    if pilot_refit_surface_indices is not None and target_surface_index in pilot_refit_surface_indices:
        # 计算有效区域内残差 OPD 的 PV
        valid_mask_for_pv = np.isfinite(residual_opd_waves) & (input_amplitude > 0.01 * np.max(input_amplitude))
        if np.any(valid_mask_for_pv):
            pv_waves = float(np.ptp(residual_opd_waves[valid_mask_for_pv]))
            rms_waves = float(np.std(residual_opd_waves[valid_mask_for_pv]))
            print(
                f"[POP][Element Refit 检查] Surface {target_surface_index} (exit): "
                f"残差 PV={pv_waves:.4f} waves, RMS={rms_waves:.4f} waves "
                f"(阈值={pilot_refit_pv_threshold_waves:.4f} waves)"
            )
            if pv_waves > pilot_refit_pv_threshold_waves:
                # 保存 refit 前的状态（始终保存，用于绘图）
                pilot_opd_waves_pre_refit = pilot_opd_waves.copy()
                residual_opd_waves_pre_refit = residual_opd_waves.copy()
                old_pilot_for_plot = new_pilot  # refit 前的 pilot

                refitted = _refit_pilot_beam_from_rays(
                    exit_local_rays,
                    absolute_opd_waves,
                    wavelength_mm=wavelength_mm,
                    current_refractive_index=new_pilot.current_refractive_index,
                    wavelength_um=new_pilot.wavelength_um,
                    target_w_mm=new_pilot.spot_size_mm, # Force keep w
                )
                if refitted is not None:
                    old_r = new_pilot.curvature_radius_mm
                    old_w = new_pilot.spot_size_mm
                    old_w0 = new_pilot.waist_radius_mm
                    old_pv = pv_waves
                    new_pilot = refitted
                    _sync_proper_gaussian_params(state.proper_wfo, new_pilot, update_reference_surface=True)
                    # 用新 pilot 重新计算 pilot_opd 和 residual_opd
                    pilot_opd_waves = _compute_pilot_opd_waves(exit_local_rays, new_pilot, wavelength_mm)
                    residual_opd_waves = absolute_opd_waves - pilot_opd_waves
                    # 以主光线为参考重新居中残差
                    if np.any(np.isfinite(residual_opd_waves)):
                        r_sq = np.asarray(exit_local_rays.x) ** 2 + np.asarray(exit_local_rays.y) ** 2
                        finite_mask_resync = np.isfinite(residual_opd_waves) & np.isfinite(r_sq)
                        if np.any(finite_mask_resync):
                            chief_idx = int(np.argmin(r_sq[finite_mask_resync]))
                            chief_value = residual_opd_waves[finite_mask_resync][chief_idx]
                            residual_opd_waves = residual_opd_waves - chief_value
                    new_pv = float(np.ptp(residual_opd_waves[valid_mask_for_pv])) if np.any(valid_mask_for_pv) else old_pv
                    new_rms = float(np.std(residual_opd_waves[valid_mask_for_pv])) if np.any(valid_mask_for_pv) else 0.0
                    print(
                        f"[POP][Element Refit 接受] Surface {target_surface_index}: "
                        f"PV={old_pv:.4f} → {new_pv:.4f} waves, "
                        f"RMS={rms_waves:.4f} → {new_rms:.4f} waves | "
                        f"R: {old_r:.2f} → {new_pilot.curvature_radius_mm:.2f} mm, "
                        f"w: {old_w:.4f} → {new_pilot.spot_size_mm:.4f} mm, "
                        f"w0: {old_w0:.4f} → {new_pilot.waist_radius_mm:.4f} mm"
                    )
                    refit_occurred = True
                    refit_info = {
                        "pv_before_waves": old_pv,
                        "pv_after_waves": new_pv,
                        "R_before_mm": old_r,
                        "R_after_mm": new_pilot.curvature_radius_mm,
                        "w_before_mm": old_w,
                        "w_after_mm": new_pilot.spot_size_mm,
                        "w0_before_mm": old_w0,
                        "w0_after_mm": new_pilot.waist_radius_mm,
                        "old_pilot": old_pilot_for_plot,
                        "new_pilot": new_pilot,
                    }

                    # 独立绘图（不依赖 debug 模式）
                    try:
                        from pop.visualization import plot_element_refit_diagnostics
                        if refit_debug_dir is not None:
                            refit_save_dir = Path(refit_debug_dir)
                        elif debug_plot_3d_dir is not None:
                            refit_save_dir = Path(debug_plot_3d_dir)
                        else:
                            refit_save_dir = Path("tests/debug_output_refit")
                        refit_save_path = refit_save_dir / f"surface_{target_surface_index:02d}_element_refit.png"
                        raw_intensity = getattr(exit_local_rays, "i", getattr(exit_local_rays, "intensity", None))
                        if raw_intensity is not None:
                            plot_element_refit_diagnostics(
                                ray_x=np.asarray(exit_local_rays.x, dtype=float),
                                ray_y=np.asarray(exit_local_rays.y, dtype=float),
                                ray_intensity=np.asarray(raw_intensity, dtype=float),
                                absolute_opd_waves=absolute_opd_waves,
                                pilot_opd_before=pilot_opd_waves_pre_refit,
                                residual_opd_before=residual_opd_waves_pre_refit,
                                pilot_opd_after=pilot_opd_waves,
                                residual_opd_after=residual_opd_waves,
                                old_pilot=old_pilot_for_plot,
                                new_pilot=new_pilot,
                                surface_index=target_surface_index,
                                save_path=refit_save_path,
                                show=False,
                            )
                    except Exception as exc:
                        print(f"[POP][Element Refit] 绘图失败: {exc}")
                else:
                    print(
                        f"[POP][Element Refit 失败] Surface {target_surface_index}: "
                        f"拟合返回 None（数据不足或退化）"
                    )
            else:
                print(
                    f"[POP][Element Refit 跳过] Surface {target_surface_index}: "
                    f"PV {pv_waves:.4f} ≤ 阈值 {pilot_refit_pv_threshold_waves:.4f} waves"
                )

    
    residual_opd_waves_recon = residual_opd_waves
    input_amplitude_recon = input_amplitude

    # --- Reconstruction ---
    # Use ALL rays that arrived at the surface for reconstruction.
    # The sampler (6-sigma) already ensured we have a robust set of rays representing the beam.
    # No further masking is applied here to allow the Zernike fit to use all available data.
    
    reconstruction_mask = np.ones(len(ray_x_in), dtype=bool)
    # residual_opd_waves_recon & input_amplitude_recon are already set to full arrays above

    amplitude_grid, residual_phase_grid = reconstruct_wavefront(
        ray_x_in=ray_x_in,
        ray_y_in=ray_y_in,
        ray_x_out=ray_x_out,
        ray_y_out=ray_y_out,
        residual_opd_waves=residual_opd_waves_recon,
        grid_sampling=state.grid_sampling,
        input_amplitude=input_amplitude_recon,
        trace_context=trace_context,
        phase_method=phase_method,
        zernike_terms=zernike_terms,
        zernike_normalize=zernike_normalize,
    )


    pilot_phase = new_pilot.compute_phase_grid(
        state.grid_sampling.grid_size, state.grid_sampling.physical_size_mm
    )
    full_phase = residual_phase_grid + pilot_phase

    _amplitude_phase_to_proper(
        state.proper_wfo,
        amplitude_grid,
        full_phase,
        state.grid_sampling,
        trace_context=trace_context,
    )

    reference_relative_field, reference_phase = snapshot_reference_frame(
        state.proper_wfo,
        full_phase,
    )

    new_state = PropagationState(
        surface_index=target_surface_index,
        position="exit",
        amplitude=amplitude_grid,
        phase=full_phase,
        pilot_beam_params=new_pilot,
        optical_axis_state=exit_axis,
        grid_sampling=state.grid_sampling,
        proper_wfo=state.proper_wfo,
        force_asm=state.force_asm,
        propagation_algorithm="element",
        reference_relative_field=reference_relative_field,
        reference_phase=reference_phase,
    )

    if debug:
        debug_info = SurfaceDebugInfo(
            surface_index=target_surface_index,
            entrance_rays_local=entrance_local_rays,
            entrance_rays_global=entrance_global_rays,
            surface_rays_global=surface_rays_global,
            exit_rays_local=exit_local_rays_dict,
            exit_rays_global=exit_global_rays,
            input_amplitude=input_amplitude,
            absolute_opd_waves=absolute_opd_waves,
            pilot_opd_waves=pilot_opd_waves,
            residual_opd_waves=residual_opd_waves,
            pilot_opd_waves_pre_refit=pilot_opd_waves_pre_refit,
            residual_opd_waves_pre_refit=residual_opd_waves_pre_refit,
            refit_occurred=refit_occurred,
            pilot_refit_info=refit_info,
            residual_phase_grid=residual_phase_grid,
            reconstruction_mask=reconstruction_mask,
        )
        if debug_store is not None:
            debug_store[target_surface_index] = debug_info

    if debug_plot_3d:
        try:
            from pop.visualization import plot_surface_raytrace_3d

            save_path = None
            if debug_plot_3d_dir is not None:
                save_dir = Path(debug_plot_3d_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"surface_{target_surface_index:02d}_raytrace_3d.png"

            plot_surface_raytrace_3d(
                surface=surface,
                entrance_axis=entrance_axis,
                exit_axis=exit_axis,
                entrance_rays=entrance_global_rays,
                surface_rays=surface_rays_global,
                exit_rays=exit_global_rays,
                surface_overrides=surface_overrides,
                save_path=save_path,
                show=debug_plot_3d_show,
                max_rays=debug_plot_3d_ray_count,
            )
        except Exception as exc:
            print(f"警告: Surface {target_surface_index} 的 3D debug 绘图失败: {exc}")

    return new_state
