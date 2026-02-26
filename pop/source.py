"""Source definitions for POP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any

import numpy as np

from .core import GridSampling, PilotBeamParams


@dataclass
class GaussianSource:
    """Gaussian source definition."""

    wavelength_um: float
    w0_mm: float
    grid_size: int = 256
    physical_size_mm: Optional[float] = None
    z0_mm: float = 0.0
    beam_diam_fraction: Optional[float] = None

    def __post_init__(self) -> None:
        if self.wavelength_um <= 0:
            raise ValueError("wavelength_um must be positive")
        if self.w0_mm <= 0:
            raise ValueError("w0_mm must be positive")
        if not isinstance(self.grid_size, int) or self.grid_size <= 0:
            raise ValueError("grid_size must be a positive integer")
        if self.physical_size_mm is not None and self.physical_size_mm <= 0:
            raise ValueError("physical_size_mm must be positive")
        if self.beam_diam_fraction is not None and self.beam_diam_fraction <= 0:
            raise ValueError("beam_diam_fraction must be positive")

    @property
    def wavelength_mm(self) -> float:
        return self.wavelength_um * 1e-3

    def _resolve_beam_ratio(self) -> Tuple[float, float]:
        if self.physical_size_mm is None:
            beam_ratio = self.beam_diam_fraction if self.beam_diam_fraction is not None else 0.5
            physical_size_mm = 2.0 * self.w0_mm / beam_ratio
            return beam_ratio, physical_size_mm
        beam_ratio = 2.0 * self.w0_mm / self.physical_size_mm
        return beam_ratio, self.physical_size_mm

    def get_grid_sampling(self) -> GridSampling:
        beam_ratio, physical_size_mm = self._resolve_beam_ratio()
        return GridSampling.create(
            grid_size=self.grid_size,
            physical_size_mm=physical_size_mm,
            beam_ratio=beam_ratio,
        )

    def create_initial_wavefront(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, PilotBeamParams, Any]:
        import proper

        beam_ratio, physical_size_mm = self._resolve_beam_ratio()
        wavelength_m = self.wavelength_um * 1e-6
        beam_diameter_m = 2.0 * self.w0_mm * 1e-3

        wfo = proper.prop_begin(
            beam_diameter_m,
            wavelength_m,
            self.grid_size,
            beam_ratio,
        )

        wavelength_mm = self.wavelength_um * 1e-3
        z_r_mm = np.pi * self.w0_mm**2 / wavelength_mm
        z_mm = -self.z0_mm

        wfo.w0 = self.w0_mm * 1e-3
        wfo.z_Rayleigh = z_r_mm * 1e-3
        wfo.z = z_mm * 1e-3
        wfo.z_w0 = 0.0

        rayleigh_factor = proper.rayleigh_factor
        if abs(wfo.z - wfo.z_w0) < rayleigh_factor * wfo.z_Rayleigh:
            wfo.beam_type_old = "INSIDE_"
            wfo.reference_surface = "PLANAR"
        else:
            wfo.beam_type_old = "OUTSIDE"
            wfo.reference_surface = "SPHERI"

        sampling_mm = proper.prop_get_sampling(wfo) * 1e3
        coords = (np.arange(self.grid_size) - self.grid_size // 2) * sampling_mm
        x_grid, y_grid = np.meshgrid(coords, coords)
        r_sq = x_grid**2 + y_grid**2

        if z_r_mm > 0:
            w = self.w0_mm * np.sqrt(1.0 + (z_mm / z_r_mm) ** 2)
        else:
            w = self.w0_mm

        if abs(z_mm) < 1e-15:
            r_curv = np.inf
        else:
            r_curv = z_mm * (1.0 + (z_r_mm / z_mm) ** 2)

        amplitude = np.exp(-r_sq / w**2)

        if np.isinf(r_curv):
            phase = np.zeros_like(r_sq)
        else:
            k = 2.0 * np.pi / wavelength_mm
            phase = k * r_sq / (2.0 * r_curv)

        if wfo.reference_surface == "SPHERI":
            r_sq_m = (x_grid * 1e-3) ** 2 + (y_grid * 1e-3) ** 2
            r_ref_m = wfo.z - wfo.z_w0
            if abs(r_ref_m) > 1e-12:
                k_m = 2.0 * np.pi / wavelength_m
                ref_phase = k_m * r_sq_m / (2.0 * r_ref_m)
                residual_phase = phase - ref_phase
                complex_amplitude = amplitude * np.exp(1j * residual_phase)
            else:
                complex_amplitude = amplitude * np.exp(1j * phase)
        else:
            complex_amplitude = amplitude * np.exp(1j * phase)

        wfo.wfarr = proper.prop_shift_center(complex_amplitude)

        pilot_beam = PilotBeamParams.from_gaussian_source(
            wavelength_um=self.wavelength_um,
            w0_mm=self.w0_mm,
            z0_mm=self.z0_mm,
        )

        return amplitude, phase, pilot_beam, wfo
