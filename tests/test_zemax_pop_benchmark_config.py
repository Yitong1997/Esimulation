from __future__ import annotations

from pathlib import Path

from sandbox.zemax_pop_benchmark.config import BenchmarkConfig, PopSamplingConfig


def test_benchmark_config_serializes_paths() -> None:
    config = BenchmarkConfig(
        zmx_path=Path("system.zmx"),
        output_dir=Path("output/run"),
        sampling=PopSamplingConfig(grid_size=128, physical_size_mm=64.0),
    )

    payload = config.to_jsonable()

    assert payload["zmx_path"] == "system.zmx"
    assert payload["sampling"]["grid_size"] == 128
