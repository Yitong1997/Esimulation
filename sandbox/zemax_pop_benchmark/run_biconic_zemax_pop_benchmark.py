"""Manual biconic benchmark runner for local POP API versus Zemax POP."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pop.io.zbf import read_zbf

from .comparison import compare_pop_state_to_zbf
from .config import BenchmarkConfig, PopSamplingConfig, ZbfInputConfig
from .popapi_runner import PopApiRunner, PopApiRun
from .zosapi_runner import ZosPopRunner, ZosPopRun

_BICONIC_ZMX_NAME = "biconic_focus_test_expand_validation.zmx"


def _resolve_default_zmx_path(module_file: str | Path) -> Path:
    """Resolve the biconic baseline ZMX across main workspaces and worktrees."""

    module_path = Path(module_file).resolve()
    for parent in module_path.parents:
        candidates = [
            parent / "sandbox" / "Zemax_baseline" / _BICONIC_ZMX_NAME,
        ]
        if parent.name == ".worktrees":
            candidates.append(
                parent.parent / "sandbox" / "Zemax_baseline" / _BICONIC_ZMX_NAME
            )
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return module_path.parents[3] / "sandbox" / "Zemax_baseline" / _BICONIC_ZMX_NAME


DEFAULT_ZMX_PATH = _resolve_default_zmx_path(Path(__file__))
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "output" / "biconic_zemax_pop_benchmark"
)


def run_benchmark(
    *,
    mode: str = "both",
    input_zbf: str | Path | None = None,
    output_dir: str | Path | None = None,
    zmx_path: str | Path | None = None,
    grid_size: int = 1024,
    physical_size_mm: float = 348.0,
) -> dict[str, Any]:
    """Run the requested benchmark mode and write config/summary JSON files."""

    mode = mode.lower().strip()
    if mode not in {"gaussian", "zbf", "both"}:
        raise ValueError("mode must be 'gaussian', 'zbf', or 'both'")
    if mode in {"zbf", "both"} and input_zbf is None:
        raise ValueError("input_zbf is required for ZBF input benchmark mode")

    out_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    config = BenchmarkConfig(
        zmx_path=Path(zmx_path) if zmx_path is not None else DEFAULT_ZMX_PATH,
        output_dir=out_dir,
        sampling=PopSamplingConfig(
            grid_size=int(grid_size),
            physical_size_mm=float(physical_size_mm),
            beam_diam_fraction=2.0 / 12.0,
        ),
        zbf_input=(
            ZbfInputConfig(path=Path(input_zbf))
            if input_zbf is not None
            else None
        ),
    )
    (out_dir / "config.json").write_text(
        json.dumps(config.to_jsonable(), indent=2),
        encoding="utf-8",
    )

    pop_runner = PopApiRunner()
    zos_runner = ZosPopRunner()
    results: dict[str, Any] = {
        "config": config.to_jsonable(),
        "modes": {},
    }

    if mode in {"gaussian", "both"}:
        pop_run = pop_runner.run_gaussian_direct(config)
        zemax_run = zos_runner.run_gaussian_direct(
            config,
            output_stem="biconic_gaussian_direct",
        )
        results["modes"]["gaussian_direct"] = _summarize_mode(
            config=config,
            pop_run=pop_run,
            zemax_run=zemax_run,
            zbf_glob="biconic_gaussian_direct*.ZBF",
        )

    if mode in {"zbf", "both"}:
        pop_run = pop_runner.run_zbf_input(config)
        zemax_run = zos_runner.run_zbf_input(
            config,
            output_stem="biconic_zbf_input",
        )
        results["modes"]["zbf_input"] = _summarize_mode(
            config=config,
            pop_run=pop_run,
            zemax_run=zemax_run,
            zbf_glob="biconic_zbf_input*.ZBF",
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    results["summary_json_path"] = str(summary_path)
    return results


def _summarize_mode(
    *,
    config: BenchmarkConfig,
    pop_run: PopApiRun,
    zemax_run: ZosPopRun,
    zbf_glob: str,
) -> dict[str, Any]:
    result = pop_run.result
    comparison = _compare_pop_result_to_zemax_outputs(
        result=result,
        zbf_dir=zemax_run.output_beam_dir,
        zbf_glob=zbf_glob,
        pop_position=config.comparison.pop_position,
        mask_threshold=config.comparison.mask_threshold,
        surface_indices=config.comparison.surface_indices,
    )
    return {
        "surface_count": len(getattr(result, "surfaces", [])),
        "zemax_output_dir": (
            None if zemax_run.output_beam_dir is None else str(zemax_run.output_beam_dir)
        ),
        "zemax_expected_output_zbf": (
            None
            if zemax_run.expected_output_zbf is None
            else str(zemax_run.expected_output_zbf)
        ),
        "comparison": comparison,
    }


def _compare_pop_result_to_zemax_outputs(
    *,
    result: Any,
    zbf_dir: Path | None,
    zbf_glob: str,
    pop_position: str,
    mask_threshold: float,
    surface_indices: Iterable[int] | None,
) -> dict[str, Any]:
    if zbf_dir is None:
        return {
            "matched_surface_count": 0,
            "skipped": [{"reason": "missing_zemax_output_beam_dir"}],
            "surfaces": [],
        }
    zbf_by_surface = _zbf_files_by_surface(zbf_dir, zbf_glob)
    surface_filter = set(surface_indices) if surface_indices is not None else None
    records = {int(record.index): record for record in getattr(result, "surfaces", [])}
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for surface_index, zbf_path in sorted(zbf_by_surface.items()):
        if surface_filter is not None and surface_index not in surface_filter:
            skipped.append(
                {
                    "surface_index": surface_index,
                    "zbf_path": str(zbf_path),
                    "reason": "not_in_surface_indices",
                }
            )
            continue
        record = records.get(surface_index)
        if record is None:
            skipped.append(
                {
                    "surface_index": surface_index,
                    "zbf_path": str(zbf_path),
                    "reason": "missing_pop_surface",
                }
            )
            continue
        state = record.get_state(pop_position)
        if state is None:
            skipped.append(
                {
                    "surface_index": surface_index,
                    "zbf_path": str(zbf_path),
                    "reason": f"missing_pop_{pop_position}_state",
                }
            )
            continue
        pop_field, pop_reference_phase = _state_reference_relative_field(state)
        comparison = compare_pop_state_to_zbf(
            state=state,
            pop_reference_relative=pop_field,
            pop_reference_phase=pop_reference_phase,
            zbf=read_zbf(zbf_path),
            surface_name=record.name,
            mask_threshold=mask_threshold,
        )
        rows.append(_jsonable(comparison.summary))

    for surface_index in sorted(records):
        if surface_index not in zbf_by_surface and (
            surface_filter is None or surface_index in surface_filter
        ):
            skipped.append({"surface_index": surface_index, "reason": "missing_zbf_file"})

    return {
        "zbf_dir": str(zbf_dir),
        "zbf_glob": zbf_glob,
        "pop_position": pop_position,
        "matched_surface_count": len(rows),
        "skipped_count": len(skipped),
        "surfaces": rows,
        "skipped": skipped,
    }


def _state_reference_relative_field(state: Any) -> tuple[np.ndarray, np.ndarray]:
    if getattr(state, "proper_wfo", None) is None:
        return np.asarray(state.get_complex_amplitude(), dtype=np.complex128), np.zeros_like(
            state.phase,
            dtype=np.float64,
        )
    import proper

    from pop.propagation.free_space import _compute_proper_reference_phase

    field = np.asarray(proper.prop_get_wavefront(state.proper_wfo), dtype=np.complex128)
    reference_phase = _compute_proper_reference_phase(
        state.proper_wfo,
        state.grid_sampling,
    )
    return field, reference_phase


def _zbf_files_by_surface(zbf_dir: Path, zbf_glob: str) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in sorted(Path(zbf_dir).glob(zbf_glob)):
        surface_index = _surface_index_from_zbf_name(path)
        if surface_index is None:
            continue
        files[surface_index] = path
    return files


def _surface_index_from_zbf_name(path: Path) -> int | None:
    match = re.search(r"_(\d{4})(?=\.zbf$)", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match is not None else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["gaussian", "zbf", "both"], default="both")
    parser.add_argument("--input-zbf", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zmx-path", type=Path, default=DEFAULT_ZMX_PATH)
    parser.add_argument("--grid-size", type=int, default=1024)
    parser.add_argument("--physical-size-mm", type=float, default=348.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_benchmark(
        mode=args.mode,
        input_zbf=args.input_zbf,
        output_dir=args.output_dir,
        zmx_path=args.zmx_path,
        grid_size=args.grid_size,
        physical_size_mm=args.physical_size_mm,
    )
    print(f"Summary written to {result['summary_json_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
