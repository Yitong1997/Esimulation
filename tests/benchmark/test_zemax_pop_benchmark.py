from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("BTS_RUN_ZEMAX_BENCHMARK") != "1",
    reason="Set BTS_RUN_ZEMAX_BENCHMARK=1 to run Zemax-dependent benchmark tests",
)


def test_gaussian_direct_benchmark_smoke(tmp_path: Path) -> None:
    from sandbox.zemax_pop_benchmark.run_biconic_zemax_pop_benchmark import run_benchmark

    result = run_benchmark(mode="gaussian", output_dir=tmp_path, grid_size=128)

    assert result["modes"]["gaussian_direct"]["surface_count"] >= 1


def test_zbf_input_nonideal_benchmark_smoke_requires_input(tmp_path: Path) -> None:
    from sandbox.zemax_pop_benchmark.run_biconic_zemax_pop_benchmark import run_benchmark

    with pytest.raises(ValueError, match="input_zbf"):
        run_benchmark(mode="zbf", output_dir=tmp_path, grid_size=128)
