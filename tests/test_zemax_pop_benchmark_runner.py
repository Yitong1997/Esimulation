from __future__ import annotations

from pathlib import Path

import pytest

from sandbox.zemax_pop_benchmark.run_biconic_zemax_pop_benchmark import (
    _resolve_default_zmx_path,
    run_benchmark,
)


def test_zbf_mode_requires_input_zbf_before_running_zemax(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="input_zbf"):
        run_benchmark(mode="zbf", output_dir=tmp_path, grid_size=128)


def test_default_zmx_resolver_searches_parent_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "BTS"
    worktree = workspace / ".worktrees" / "feature"
    baseline = workspace / "sandbox" / "Zemax_baseline"
    baseline.mkdir(parents=True)
    expected = baseline / "biconic_focus_test_expand_validation.zmx"
    expected.write_text("placeholder", encoding="utf-8")

    module_file = worktree / "sandbox" / "zemax_pop_benchmark" / "runner.py"
    module_file.parent.mkdir(parents=True)
    module_file.write_text("", encoding="utf-8")

    assert _resolve_default_zmx_path(module_file) == expected
