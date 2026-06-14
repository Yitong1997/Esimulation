from __future__ import annotations

from pathlib import Path

import pytest

from sandbox.zemax_pop_benchmark.run_biconic_zemax_pop_benchmark import run_benchmark


def test_zbf_mode_requires_input_zbf_before_running_zemax(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="input_zbf"):
        run_benchmark(mode="zbf", output_dir=tmp_path, grid_size=128)
