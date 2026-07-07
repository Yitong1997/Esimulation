"""Local POP API runner for benchmark cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pop
from pop import DebugOptions, GaussianSource, PlotOptions, PropagationOptions, ZbfSource

from .config import BenchmarkConfig


@dataclass
class PopApiRun:
    """Result of one local POP API benchmark path."""

    mode: str
    result: Any
    source: Any


class PopApiRunner:
    """Run the local POP implementation using public package APIs only."""

    def run_gaussian_direct(self, config: BenchmarkConfig) -> PopApiRun:
        source = GaussianSource(
            wavelength_um=float(config.gaussian.wavelength_um),
            w0_mm=float(config.gaussian.w0_mm),
            z0_mm=float(config.gaussian.z0_mm),
            grid_size=int(config.sampling.grid_size),
            physical_size_mm=float(config.sampling.physical_size_mm),
            beam_diam_fraction=config.sampling.beam_diam_fraction,
        )
        return self._run(config, source, mode="gaussian_direct")

    def run_zbf_input(self, config: BenchmarkConfig) -> PopApiRun:
        if config.zbf_input is None:
            raise ValueError("config.zbf_input is required for ZBF input benchmark")
        source = ZbfSource(
            config.zbf_input.path,
            reference_mode=config.zbf_input.reference_mode,
        )
        return self._run(config, source, mode="zbf_input")

    def _run(self, config: BenchmarkConfig, source: Any, *, mode: str) -> PopApiRun:
        system = pop.load_zmx(str(config.zmx_path))
        result = pop.propagate(
            system,
            source,
            options=_build_propagation_options(config),
            plot_options=PlotOptions(mode=None, output_dir=config.output_dir, show=False),
            debug_options=DebugOptions(enabled=False, plot_3d=False, plot_3d_show=False),
        )
        return PopApiRun(mode=mode, result=result, source=source)


def _build_propagation_options(config: BenchmarkConfig) -> PropagationOptions:
    return PropagationOptions(
        num_rays=int(config.num_rays),
        coordinate_priority=str(config.coordinate_priority),
        print_status=bool(config.print_status),
        auto_asm=bool(config.auto_asm),
        reconstruction_mask_ratio=config.reconstruction_mask_ratio,
        phase_method=str(config.phase_method),
        zernike_terms=config.zernike_terms,
        sampling_sigma=float(config.sampling_sigma),
        auto_resample=bool(config.auto_resample),
        auto_unwarp_at_incident=bool(config.auto_unwarp_at_incident),
        auto_unwarp_surface_indices=config.auto_unwarp_surface_indices,
    )
