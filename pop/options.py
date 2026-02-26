"""Option groups for POP public APIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


@dataclass
class PropagationOptions:
    """Controls propagation-level behavior."""

    num_rays: int = 1000
    coordinate_priority: str = "x"
    print_status: bool = True
    force_asm: Optional[bool] = None
    auto_asm: bool = True
    reconstruction_mask_ratio: Optional[float] = None
    phase_method: str = "griddata"
    zernike_terms: Optional[int] = None
    zernike_normalize: bool = True
    sampling_sigma: float = 6.0
    # Pilot beam refit options
    pilot_refit_surface_indices: Optional[Sequence[int]] = None  # None = disabled; list of surface indices to enable refit
    pilot_refit_pv_threshold_waves: float = 0.5  # PV threshold in waves to trigger refit
    # Auto residual unwarp at incident planes:
    # detect near-1-wave folded residual in amp>10% region and reprocess via 2D unwrap.
    auto_unwarp_at_incident: bool = False
    auto_unwarp_surface_indices: Optional[Sequence[int] | str] = None  # None = disabled; 'all' = global enable; list = specific surfaces
    # Auto grid resample options
    auto_resample: bool = False  # enable automatic grid resampling when beam is undersampled
    resample_min_beam_pixels: int = 10  # trigger resample when beam diameter < N pixels
    resample_beam_pixels_target: Optional[int] = None  # target beam diameter in pixels after resample; None = use grid_size * beam_ratio
    
    # Free-space surface handling
    merge_free_space_surfaces: bool = True  # compute consecutive free-space surfaces from last effective surface

    # Mirror options
    enable_ideal_planar_mirror: bool = False # Use ideal propagation for planar mirrors


@dataclass
class PlotOptions:
    """Controls visualization output during propagation."""

    mode: Optional[str] = "all"
    output_dir: Optional[str | Path] = None
    show: Optional[bool] = False
    positions: Optional[Sequence[str] | str] = None
    surface_indices: Optional[Sequence[int]] = None
    kwargs: Optional[dict[str, Any]] = None
    axis_3d: bool = False
    axis_3d_dir: Optional[str | Path] = None
    axis_3d_show: bool = True
    axis_3d_kwargs: Optional[dict[str, Any]] = None


@dataclass
class DebugOptions:
    """Controls debug capture and 3D ray plotting."""

    enabled: bool = False
    plot_3d: bool = False
    plot_3d_dir: Optional[str | Path] = None
    plot_3d_show: bool = True
    plot_3d_ray_count: int = 5


__all__ = ["PropagationOptions", "PlotOptions", "DebugOptions"]
