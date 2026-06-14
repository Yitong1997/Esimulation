from __future__ import annotations

from pathlib import Path

from sandbox.zemax_pop_benchmark.config import (
    BenchmarkConfig,
    GaussianInputConfig,
    PopSamplingConfig,
)
from sandbox.zemax_pop_benchmark.zosapi_runner import (
    build_gaussian_pop_kwargs,
    build_zbf_pop_kwargs,
)


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        zmx_path=Path("system.zmx"),
        output_dir=Path("output/run"),
        sampling=PopSamplingConfig(grid_size=128, physical_size_mm=64.0),
        gaussian=GaussianInputConfig(wavelength_um=10.64, w0_mm=29.0),
    )


def test_build_gaussian_pop_kwargs_sets_gaussian_waist() -> None:
    kwargs = build_gaussian_pop_kwargs(_config(), output_stem="gaussian_out")

    assert kwargs["beam_type"] == "GaussianWaist"
    assert kwargs["beam_parameters"]["Waist X"] == 29.0
    assert kwargs["beam_parameters"]["Waist Y"] == 29.0
    assert kwargs["x_sampling"] == 128
    assert kwargs["x_width"] == 64.0
    assert kwargs["save_output_beam"] is True
    assert kwargs["output_beam_file"] == "gaussian_out"


def test_build_zbf_pop_kwargs_sets_file_beam() -> None:
    kwargs = build_zbf_pop_kwargs(_config(), beam_file="input.ZBF", output_stem="zbf_out")

    assert kwargs["beam_type"] == "File"
    assert kwargs["beam_file"] == "input.ZBF"
    assert kwargs["save_output_beam"] is True
