"""Core data structures for POP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class GridSampling:
    """Grid sampling parameters (mm)."""

    grid_size: int
    physical_size_mm: float
    sampling_mm: float
    beam_ratio: float = 0.5

    @classmethod
    def create(
        cls,
        grid_size: int,
        physical_size_mm: float,
        beam_ratio: float = 0.5,
    ) -> "GridSampling":
        sampling_mm = physical_size_mm / grid_size
        return cls(
            grid_size=grid_size,
            physical_size_mm=physical_size_mm,
            sampling_mm=sampling_mm,
            beam_ratio=beam_ratio,
        )

    @classmethod
    def from_proper(cls, wfo: Any) -> "GridSampling":
        import proper

        grid_size = proper.prop_get_gridsize(wfo)
        sampling_m = proper.prop_get_sampling(wfo)
        sampling_mm = sampling_m * 1e3
        physical_size_mm = sampling_mm * grid_size
        beam_ratio = getattr(wfo, "beam_ratio", 0.5)
        return cls(
            grid_size=grid_size,
            physical_size_mm=physical_size_mm,
            sampling_mm=sampling_mm,
            beam_ratio=beam_ratio,
        )

    def get_coordinate_arrays(self) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        coords = (np.arange(self.grid_size) - self.grid_size // 2) * self.sampling_mm
        return np.meshgrid(coords, coords)


@dataclass
class PilotBeamParams:
    """Pilot beam parameters tracked by ABCD law."""

    wavelength_um: float
    waist_radius_mm: float
    waist_position_mm: float
    curvature_radius_mm: float
    spot_size_mm: float
    q_parameter: complex
    current_refractive_index: float = 1.0

    @classmethod
    def from_gaussian_source(
        cls,
        wavelength_um: float,
        w0_mm: float,
        z0_mm: float,
        current_refractive_index: float = 1.0,
    ) -> "PilotBeamParams":
        wavelength_mm = wavelength_um * 1e-3
        z_r = np.pi * w0_mm**2 / wavelength_mm
        z = -z0_mm
        q = z + 1j * z_r
        if abs(z) < 1e-15:
            r = np.inf
            w = w0_mm
        else:
            r = z * (1 + (z_r / z) ** 2)
            w = w0_mm * np.sqrt(1 + (z / z_r) ** 2)
        return cls(
            wavelength_um=wavelength_um,
            waist_radius_mm=w0_mm,
            waist_position_mm=z0_mm,
            curvature_radius_mm=r,
            spot_size_mm=w,
            q_parameter=q,
            current_refractive_index=current_refractive_index,
        )

    @classmethod
    def from_q_parameter(
        cls,
        q: complex,
        wavelength_um: float,
        current_refractive_index: float = 1.0,
    ) -> "PilotBeamParams":
        wavelength_mm = wavelength_um * 1e-3
        inv_q = 1.0 / q
        real_part = np.real(inv_q)
        imag_part = np.imag(inv_q)
        # Near-waist numeric regime: tiny Re(1/q) should be treated as planar
        # to avoid unstable curvature blow-up from floating-point noise.
        real_eps = 1e-15 * max(1.0, abs(imag_part))
        if abs(real_part) < real_eps:
            r = np.inf
        else:
            r = 1.0 / real_part
        if abs(imag_part) < 1e-30:
            w_sq = np.inf
        else:
            w_sq = -wavelength_mm / (np.pi * current_refractive_index * imag_part)
        w = np.sqrt(w_sq) if w_sq > 0 else 0.0
        z_r = np.imag(q)
        w0 = (
            np.sqrt(wavelength_mm * z_r / (np.pi * current_refractive_index))
            if z_r > 0
            else w
        )
        z = np.real(q)
        z0 = -z
        return cls(
            wavelength_um=wavelength_um,
            waist_radius_mm=w0,
            waist_position_mm=z0,
            curvature_radius_mm=r,
            spot_size_mm=w,
            q_parameter=q,
            current_refractive_index=current_refractive_index,
        )

    def propagate(self, distance_mm: float) -> "PilotBeamParams":
        q_new = self.q_parameter + distance_mm
        return PilotBeamParams.from_q_parameter(
            q_new, self.wavelength_um, self.current_refractive_index
        )

    def apply_lens(self, focal_length_mm: float) -> "PilotBeamParams":
        if np.isinf(focal_length_mm):
            return self
        a, b, c, d = 1.0, 0.0, -1.0 / focal_length_mm, 1.0
        q_new = (a * self.q_parameter + b) / (c * self.q_parameter + d)
        return PilotBeamParams.from_q_parameter(
            q_new, self.wavelength_um, self.current_refractive_index
        )

    def apply_mirror(self, radius_mm: float) -> "PilotBeamParams":
        if np.isinf(radius_mm):
            return self
        a, b, c, d = 1.0, 0.0, 2.0 / radius_mm, 1.0
        q_new = (a * self.q_parameter + b) / (c * self.q_parameter + d)
        return PilotBeamParams.from_q_parameter(
            q_new, self.wavelength_um, self.current_refractive_index
        )

    def apply_refraction(self, radius_mm: float, n1: float, n2: float) -> "PilotBeamParams":
        if np.isinf(radius_mm):
            a, b, c, d = 1.0, 0.0, 0.0, n1 / n2
        else:
            a, b = 1.0, 0.0
            c = (n1 - n2) / (n2 * radius_mm)
            d = n1 / n2
        q_new = (a * self.q_parameter + b) / (c * self.q_parameter + d)
        return PilotBeamParams.from_q_parameter(q_new, self.wavelength_um, n2)

    @property
    def rayleigh_length_mm(self) -> float:
        return float(abs(np.imag(self.q_parameter)))

    def compute_phase_from_radius_squared(
        self,
        r_sq_mm: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        r_sq = np.asarray(r_sq_mm, dtype=float)
        if np.isinf(self.curvature_radius_mm):
            return np.zeros_like(r_sq)
        wavelength_mm = self.wavelength_um * 1e-3
        k = 2 * np.pi * self.current_refractive_index / wavelength_mm
        return k * r_sq / (2.0 * self.curvature_radius_mm)

    def compute_phase_grid(
        self,
        grid_size: int,
        physical_size_mm: float,
    ) -> NDArray[np.floating]:
        dx = physical_size_mm / grid_size
        coords = (np.arange(grid_size) - grid_size // 2) * dx
        x_grid, y_grid = np.meshgrid(coords, coords)
        r_sq = x_grid**2 + y_grid**2
        return self.compute_phase_from_radius_squared(r_sq)

    def compute_amplitude_grid(
        self,
        grid_size: int,
        physical_size_mm: float,
    ) -> NDArray[np.floating]:
        dx = physical_size_mm / grid_size
        coords = (np.arange(grid_size) - grid_size // 2) * dx
        x_grid, y_grid = np.meshgrid(coords, coords)
        r_sq = x_grid**2 + y_grid**2
        w = float(self.spot_size_mm)
        if not np.isfinite(w) or w <= 0:
            if np.isinf(w):
                return np.ones((grid_size, grid_size))
            return np.zeros((grid_size, grid_size))
        return np.exp(-r_sq / (w**2))


@dataclass
class OpticalAxisState:
    """Optical axis state in global coordinates."""

    position: NDArray[np.floating]
    direction: NDArray[np.floating]
    frame: NDArray[np.floating]
    coord_sys: Optional[Any]
    path_length: float
    euler: Optional[Tuple[float, float, float]] = None

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        self.direction = np.asarray(self.direction, dtype=float)
        self.frame = np.asarray(self.frame, dtype=float)
        if self.position.shape != (3,):
            raise ValueError("position must be shape (3,)")
        if self.direction.shape != (3,):
            raise ValueError("direction must be shape (3,)")
        if self.frame.shape != (3, 3):
            raise ValueError("frame must be shape (3, 3)")
        norm = np.linalg.norm(self.direction)
        if norm < 1e-12:
            raise ValueError("direction vector cannot be zero")
        self.direction = self.direction / norm
        if self.euler is not None:
            if len(self.euler) != 3:
                raise ValueError("euler must be a 3-tuple when provided")


@dataclass
class SurfaceInteractionInfo:
    """Normalized surface interaction metadata."""

    surface_index: int
    normal_global: NDArray[np.floating]
    normal_flipped: bool
    sign_factor: float
    canonical_orientation: Optional[NDArray[np.floating]] = None

    def __post_init__(self) -> None:
        self.normal_global = np.asarray(self.normal_global, dtype=float)
        if self.normal_global.shape != (3,):
            raise ValueError("normal_global must be shape (3,)")
        if self.canonical_orientation is not None:
            self.canonical_orientation = np.asarray(self.canonical_orientation, dtype=float)


@dataclass
class PropagationState:
    """Propagation state for one surface position."""

    surface_index: int
    position: str
    amplitude: NDArray[np.floating]
    phase: NDArray[np.floating]
    pilot_beam_params: PilotBeamParams
    optical_axis_state: OpticalAxisState
    grid_sampling: GridSampling
    proper_wfo: Optional[Any] = None
    force_asm: Optional[bool] = None
    propagation_algorithm: str = "N/A"
    messages: list[str] = field(default_factory=list)

    def get_complex_amplitude(self) -> NDArray[np.complexfloating]:
        return self.amplitude * np.exp(1j * self.phase)

    def get_intensity(self) -> NDArray[np.floating]:
        return self.amplitude**2

    def get_phase(self) -> NDArray[np.floating]:
        return self.phase
