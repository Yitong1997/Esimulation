"""Coordinate transforms and frame construction.

Conventions:
- Right-handed frames.
- Frame columns are the local X, Y, Z axes expressed in global coordinates.
- Local Z is aligned with the optical axis (chief ray direction).
- Euler angles use the 'xyz' order (Rx -> Ry -> Rz), matching optiland.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from pop.core import OpticalAxisState, SurfaceInteractionInfo


def _normalize(vec: NDArray[np.floating]) -> NDArray[np.floating]:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("Cannot normalize zero-length vector")
    return vec / norm


def _interaction_debug_enabled(surface_index: int) -> bool:
    flag = os.getenv("POP_INTERACTION_DEBUG", "")
    if flag.strip().lower() not in ("1", "true", "yes", "on"):
        return False
    raw_surfaces = os.getenv("POP_INTERACTION_DEBUG_SURFACES", "2").strip()
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


def direction_to_euler(L: float, M: float, N: float) -> Tuple[float, float, float]:
    if abs(M) > 0.9999:
        rx = -np.sign(M) * np.pi / 2.0
        ry = 0.0
        rz = 0.0
        return rx, ry, rz
    rx = -np.arcsin(M)
    ry = np.arctan2(L, N)
    rz = 0.0
    return rx, ry, rz


def build_min_rotation_frame(
    z_new: NDArray[np.floating],
    prev_frame: Optional[NDArray[np.floating]] = None,
    ref_axis: Optional[NDArray[np.floating]] = None,
    priority_axis: str = "x",
) -> NDArray[np.floating]:
    z_axis = _normalize(np.asarray(z_new, dtype=float))
    if prev_frame is None:
        if priority_axis.lower() == "y":
            ref = np.asarray(ref_axis, dtype=float) if ref_axis is not None else np.array([0.0, 1.0, 0.0])
            ref = _normalize(ref)
            # If Z is parallel to Y, fundamental ambiguity. Fallback to X reference.
            if abs(np.dot(ref, z_axis)) > 0.95:
                ref = np.array([1.0, 0.0, 0.0])
                # X = Ref_X x Z? Standard X construction
                x_axis = _normalize(np.cross(ref, z_axis))
                y_axis = np.cross(z_axis, x_axis)
                # But we wanted Y priority. Since Y failed, we used X.
                # Result is standard.
                return np.column_stack([x_axis, y_axis, z_axis])
            else:
                # Use Y as primary
                # x = y x z
                x_axis = np.cross(ref, z_axis)
                x_axis = _normalize(x_axis)
                y_axis = np.cross(z_axis, x_axis)
                return np.column_stack([x_axis, y_axis, z_axis])
        else:
            # Default X priority
            ref = np.asarray(ref_axis, dtype=float) if ref_axis is not None else np.array([0.0, 1.0, 0.0])
            ref = _normalize(ref)
            if abs(np.dot(ref, z_axis)) > 0.95:
                ref = np.array([1.0, 0.0, 0.0])
            x_axis = _normalize(np.cross(ref, z_axis))
            y_axis = np.cross(z_axis, x_axis)
            return np.column_stack([x_axis, y_axis, z_axis])

    z_prev = _normalize(prev_frame[:, 2])
    v = np.cross(z_prev, z_axis)
    c = float(np.dot(z_prev, z_axis))
    s = float(np.linalg.norm(v))

    if s < 1e-8 and c > 0:
        return prev_frame.copy()

    # Determine reference axes from previous frame
    x_ref = np.asarray(prev_frame[:, 0], dtype=float)
    y_ref = np.asarray(prev_frame[:, 1], dtype=float)
    
    use_y_priority = (priority_axis.lower() == "y")
    
    if use_y_priority:
        # Try to project Y first
        y_proj = y_ref - np.dot(y_ref, z_axis) * z_axis
        
        if np.linalg.norm(y_proj) > 1e-4:
            y_axis = _normalize(y_proj)
            x_axis = np.cross(y_axis, z_axis)
        else:
            # Y singularity. Fallback to X projection.
            x_proj = x_ref - np.dot(x_ref, z_axis) * z_axis
            if np.linalg.norm(x_proj) > 1e-4:
                x_axis = _normalize(x_proj)
                y_axis = np.cross(z_axis, x_axis)
            else:
                # Global fallback
                ref = np.array([0.0, 1.0, 0.0])
                if abs(np.dot(ref, z_axis)) > 0.95:
                    ref = np.array([1.0, 0.0, 0.0])
                x_axis = _normalize(np.cross(ref, z_axis))
                y_axis = np.cross(z_axis, x_axis)
                
    else:
        # X Priority
        x_proj = x_ref - np.dot(x_ref, z_axis) * z_axis
        
        if np.linalg.norm(x_proj) > 1e-4:
            x_axis = _normalize(x_proj)
            y_axis = np.cross(z_axis, x_axis)
        else:
            # X singularity. Fallback to Y projection.
            y_proj = y_ref - np.dot(y_ref, z_axis) * z_axis
            if np.linalg.norm(y_proj) > 1e-4:
                y_axis = _normalize(y_proj)
                x_axis = np.cross(y_axis, z_axis)
            else:
                # Global fallback
                ref = np.array([0.0, 1.0, 0.0])
                if abs(np.dot(ref, z_axis)) > 0.95:
                    ref = np.array([1.0, 0.0, 0.0])
                x_axis = _normalize(np.cross(ref, z_axis))
                y_axis = np.cross(z_axis, x_axis)

    # Ensure alignment with previous frame to prevent flip
    if use_y_priority:
        if np.dot(y_axis, y_ref) < -1e-6:
             y_axis *= -1.0
             x_axis *= -1.0
    else:
        if np.dot(x_axis, x_ref) < -1e-6:
            x_axis *= -1.0
            y_axis *= -1.0

    return np.column_stack([x_axis, y_axis, z_axis])


def rotation_matrix_to_euler(
    rotation_matrix: NDArray[np.floating],
    prev_euler: Optional[Tuple[float, float, float]] = None,
) -> Tuple[float, float, float]:
    import warnings
    from scipy.spatial.transform import Rotation as Rot

    r = np.asarray(rotation_matrix, dtype=float)
    gimbal_lock = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rx, ry, rz = Rot.from_matrix(r).as_euler("xyz", degrees=False)
        gimbal_lock = any("Gimbal lock" in str(w.message) for w in caught)

    if gimbal_lock and prev_euler is not None:
        rz = float(prev_euler[2])
        if r[2, 0] < 0:
            ry = np.pi / 2.0
            delta = np.arctan2(r[0, 1], r[1, 1])
            rx = delta + rz
        else:
            ry = -np.pi / 2.0
            delta = np.arctan2(-r[0, 1], r[1, 1])
            rx = delta - rz

    if prev_euler is not None:
        def _wrap_near(angle: float, ref: float) -> float:
            two_pi = 2.0 * np.pi
            return angle + two_pi * round((ref - angle) / two_pi)

        rx = _wrap_near(rx, prev_euler[0])
        ry = _wrap_near(ry, prev_euler[1])
        rz = _wrap_near(rz, prev_euler[2])

    return float(rx), float(ry), float(rz)


def create_coordinate_system(
    origin: NDArray[np.floating],
    frame: NDArray[np.floating],
    prev_euler: Optional[Tuple[float, float, float]] = None,
    euler_override: Optional[Tuple[float, float, float]] = None,
):
    from optiland.coordinate_system import CoordinateSystem

    if euler_override is not None:
        rx, ry, rz = euler_override
    else:
        rx, ry, rz = rotation_matrix_to_euler(frame, prev_euler)
    origin = np.asarray(origin, dtype=float)
    return CoordinateSystem(
        x=float(origin[0]),
        y=float(origin[1]),
        z=float(origin[2]),
        rx=rx,
        ry=ry,
        rz=rz,
    )


def transform_rays_to_global(rays, entrance_axis: OpticalAxisState):
    if entrance_axis.coord_sys is None:
        entrance_axis.coord_sys = create_coordinate_system(
            entrance_axis.position,
            entrance_axis.frame,
            euler_override=entrance_axis.euler,
        )
    entrance_axis.coord_sys.globalize(rays)
    return rays


def transform_rays_to_local(rays, exit_axis: OpticalAxisState):
    if exit_axis.coord_sys is None:
        exit_axis.coord_sys = create_coordinate_system(
            exit_axis.position,
            exit_axis.frame,
            euler_override=exit_axis.euler,
        )
    exit_axis.coord_sys.localize(rays)
    return rays


def build_surface_interaction_info(
    surface_index: int,
    orientation: NDArray[np.floating],
    entrance_dir: NDArray[np.floating],
    is_mirror: bool,
) -> SurfaceInteractionInfo:
    orientation = np.asarray(orientation, dtype=float)
    entrance_dir = _normalize(np.asarray(entrance_dir, dtype=float))
    normal = orientation[:, 2]
    alignment = float(np.dot(entrance_dir, normal))
    normal_flipped = alignment < 0
    sign_factor = 1.0 if alignment >= 0 else -1.0

    canonical_orientation = orientation
    if normal_flipped:
        canonical_orientation = orientation @ np.diag([1.0, -1.0, -1.0])

    if _interaction_debug_enabled(surface_index):
        print(
            f"[POP][InteractionDebug] S{surface_index}: is_mirror={bool(is_mirror)}, "
            f"alignment={alignment:+.9f}, sign_factor={sign_factor:+.1f}, normal_flipped={normal_flipped}"
        )
        print(
            f"[POP][InteractionDebug] entrance_dir=({entrance_dir[0]:.6f}, {entrance_dir[1]:.6f}, {entrance_dir[2]:.6f}), "
            f"normal=({normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f})"
        )

    return SurfaceInteractionInfo(
        surface_index=surface_index,
        normal_global=-normal if normal_flipped else normal,
        normal_flipped=normal_flipped,
        sign_factor=sign_factor,
        canonical_orientation=canonical_orientation,
    )
