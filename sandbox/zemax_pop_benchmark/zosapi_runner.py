"""ZOSAPI/ZOSPy runner for Zemax POP benchmark cases."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig


@dataclass
class ZosPopRun:
    """Result metadata for one Zemax POP benchmark run."""

    mode: str
    output_dir: Path
    output_stem: str
    output_beam_dir: Path | None
    expected_output_zbf: Path | None
    analysis_result: Any


def build_gaussian_pop_kwargs(
    config: BenchmarkConfig,
    *,
    output_stem: str,
    data_type: str = "Irradiance",
) -> dict[str, Any]:
    """Build ZOSPy POP kwargs for direct Gaussian-waist input."""

    common = _common_pop_kwargs(config, output_stem=output_stem, data_type=data_type)
    common.update(
        {
            "beam_type": "GaussianWaist",
            "beam_parameters": {
                "Waist X": float(config.gaussian.w0_mm),
                "Waist Y": float(config.gaussian.w0_mm),
                "Decenter X": 0.0,
                "Decenter Y": 0.0,
                "Aperture X": 0.0,
                "Aperture Y": 0.0,
            },
            "auto_calculate_beam_sampling": False,
        }
    )
    return common


def build_zbf_pop_kwargs(
    config: BenchmarkConfig,
    *,
    beam_file: str,
    output_stem: str,
    data_type: str = "Irradiance",
) -> dict[str, Any]:
    """Build ZOSPy POP kwargs for File/ZBF input."""

    common = _common_pop_kwargs(config, output_stem=output_stem, data_type=data_type)
    common.update(
        {
            "beam_type": "File",
            "beam_file": str(beam_file),
            "auto_calculate_beam_sampling": False,
        }
    )
    return common


class ZosPopRunner:
    """Run Zemax Physical Optics Propagation through ZOSPy."""

    def __init__(
        self,
        *,
        connection_mode: str = "standalone",
        zospy_path: str | Path | None = None,
        beam_file_dir: str | Path | None = None,
    ) -> None:
        self.connection_mode = connection_mode
        self.zospy_path = Path(zospy_path) if zospy_path is not None else None
        self.beam_file_dir = Path(beam_file_dir) if beam_file_dir is not None else None

    def run_gaussian_direct(
        self,
        config: BenchmarkConfig,
        *,
        output_stem: str = "zemax_gaussian_direct",
    ) -> ZosPopRun:
        return self._run_pop(
            config,
            mode="gaussian_direct",
            kwargs=build_gaussian_pop_kwargs(config, output_stem=output_stem),
            output_stem=output_stem,
        )

    def run_zbf_input(
        self,
        config: BenchmarkConfig,
        *,
        output_stem: str = "zemax_zbf_input",
    ) -> ZosPopRun:
        if config.zbf_input is None:
            raise ValueError("config.zbf_input is required for ZBF input benchmark")
        zospy, _pop_cls = _import_zospy(self.zospy_path)
        zos, oss = _connect_and_load(zospy, config, self.connection_mode)
        try:
            beam_dir = _resolve_beam_file_dir(oss, self.beam_file_dir)
            copied = _copy_zbf_to_beam_dir(config.zbf_input.path, beam_dir)
            kwargs = build_zbf_pop_kwargs(
                config,
                beam_file=copied.name,
                output_stem=output_stem,
            )
            return self._run_pop_with_open_system(
                config,
                zos=zos,
                oss=oss,
                mode="zbf_input",
                kwargs=kwargs,
                output_stem=output_stem,
                output_beam_dir=beam_dir,
            )
        except Exception:
            zos.disconnect()
            raise

    def _run_pop(
        self,
        config: BenchmarkConfig,
        *,
        mode: str,
        kwargs: dict[str, Any],
        output_stem: str,
    ) -> ZosPopRun:
        zospy, _pop_cls = _import_zospy(self.zospy_path)
        zos, oss = _connect_and_load(zospy, config, self.connection_mode)
        try:
            beam_dir = _resolve_beam_file_dir(oss, self.beam_file_dir)
            return self._run_pop_with_open_system(
                config,
                zos=zos,
                oss=oss,
                mode=mode,
                kwargs=kwargs,
                output_stem=output_stem,
                output_beam_dir=beam_dir,
            )
        except Exception:
            zos.disconnect()
            raise

    def _run_pop_with_open_system(
        self,
        config: BenchmarkConfig,
        *,
        zos: Any,
        oss: Any,
        mode: str,
        kwargs: dict[str, Any],
        output_stem: str,
        output_beam_dir: Path | None,
    ) -> ZosPopRun:
        _zospy, pop_cls = _import_zospy(self.zospy_path)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            analysis = pop_cls(**kwargs)
            result = analysis.run(oss)
            expected = (
                output_beam_dir / f"{output_stem}.ZBF"
                if output_beam_dir is not None
                else None
            )
            return ZosPopRun(
                mode=mode,
                output_dir=config.output_dir,
                output_stem=output_stem,
                output_beam_dir=output_beam_dir,
                expected_output_zbf=expected,
                analysis_result=result,
            )
        finally:
            zos.disconnect()


def _common_pop_kwargs(
    config: BenchmarkConfig,
    *,
    output_stem: str,
    data_type: str,
) -> dict[str, Any]:
    return {
        "start_surface": 1,
        "end_surface": "Image",
        "x_sampling": int(config.sampling.grid_size),
        "y_sampling": int(config.sampling.grid_size),
        "x_width": float(config.sampling.physical_size_mm),
        "y_width": float(config.sampling.physical_size_mm),
        "separate_xy": False,
        "data_type": str(data_type),
        "show_as": "FalseColor",
        "use_total_power": False,
        "use_peak_irradiance": True,
        "peak_irradiance": 1.0,
        "save_output_beam": True,
        "output_beam_file": str(output_stem),
        "save_beam_at_all_surfaces": True,
    }


def _import_zospy(zospy_path: str | Path | None = None) -> tuple[Any, Any]:
    if zospy_path is not None:
        path = str(Path(zospy_path))
        if path not in sys.path:
            sys.path.insert(0, path)
    else:
        default_path = Path(r"D:\BTS_ZosApi\ZOSPy-main")
        if default_path.exists() and str(default_path) not in sys.path:
            sys.path.insert(0, str(default_path))

    import zospy as zp
    from zospy.analyses.physicaloptics import PhysicalOpticsPropagation

    return zp, PhysicalOpticsPropagation


def _connect_and_load(
    zospy: Any,
    config: BenchmarkConfig,
    connection_mode: str,
) -> tuple[Any, Any]:
    zmx_path = Path(config.zmx_path).resolve()
    if not zmx_path.is_file():
        raise FileNotFoundError(f"Zemax file not found: {zmx_path}")
    zos = zospy.ZOS()
    oss = zos.connect(mode=connection_mode)
    oss.load(str(zmx_path))
    return zos, oss


def _resolve_beam_file_dir(oss: Any, override: Path | None) -> Path | None:
    if override is not None:
        override.mkdir(parents=True, exist_ok=True)
        return override
    app = getattr(oss, "TheApplication", None)
    pop_dir = getattr(app, "POPDir", None)
    if not pop_dir:
        return None
    path = Path(str(pop_dir))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _copy_zbf_to_beam_dir(path: str | Path, beam_dir: Path | None) -> Path:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"ZBF input file not found: {source}")
    if beam_dir is None:
        return source
    target = beam_dir / source.name
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target
