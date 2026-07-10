"""Predeclared branch-specific Zemax sampling matrices."""

from __future__ import annotations

import json
import math
from numbers import Integral
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .models import (
    CaseOutputSource,
    DerivationStrategy,
    NativeSurfaceSource,
    SamplingCase,
    SamplingPurpose,
    SamplingSource,
    SourceKind,
    UniformGrid2D,
)


S7_WAIST_DISTANCE_MM = 239.982966226840
S8_WAIST_DISTANCE_MM = 608.582967581705
S12_STW_WAIST_DISTANCE_MM = 608.615263635412
S14_WTS_WAIST_DISTANCE_MM = 1.984736370234


def _require_positive_finite(**values: float) -> None:
    if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
        labels = ", ".join(values)
        raise ValueError(f"sampling-law inputs must be positive and finite: {labels}")


def _require_sample_count(n: int) -> int:
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or int(n) < 2:
        raise ValueError("n must be an integer with at least two samples")
    return int(n)


def _medium_wavelength_mm(
    *, wavelength_vacuum_mm: float, refractive_index: float
) -> float:
    _require_positive_finite(
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    return wavelength_vacuum_mm / refractive_index


def outside_to_outside_output_sampling_mm(
    *,
    wavelength_vacuum_mm: float,
    refractive_index: float,
    input_dx_mm: float,
    start_waist_distance_mm: float,
    end_waist_distance_mm: float,
) -> float:
    """Return the Outside-to-Outside magnified interval for one axis."""

    _medium_wavelength_mm(
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    _require_positive_finite(
        input_dx_mm=input_dx_mm,
        start_waist_distance_mm=start_waist_distance_mm,
        end_waist_distance_mm=end_waist_distance_mm,
    )
    return input_dx_mm * end_waist_distance_mm / start_waist_distance_mm


def stw_output_sampling_mm(
    *,
    wavelength_vacuum_mm: float,
    refractive_index: float,
    waist_distance_mm: float,
    n: int,
    input_dx_mm: float,
) -> float:
    """Return the Outside-to-Inside STW natural output interval for one axis."""

    lambda_medium_mm = _medium_wavelength_mm(
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    _require_positive_finite(
        waist_distance_mm=waist_distance_mm, input_dx_mm=input_dx_mm
    )
    samples = _require_sample_count(n)
    return lambda_medium_mm * waist_distance_mm / (samples * input_dx_mm)


def wts_output_sampling_mm(
    *,
    wavelength_vacuum_mm: float,
    refractive_index: float,
    waist_distance_mm: float,
    n: int,
    input_dx_mm: float,
) -> float:
    """Return the Inside-to-Outside WTS natural output interval for one axis."""

    lambda_medium_mm = _medium_wavelength_mm(
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    _require_positive_finite(
        waist_distance_mm=waist_distance_mm, input_dx_mm=input_dx_mm
    )
    samples = _require_sample_count(n)
    return lambda_medium_mm * waist_distance_mm / (samples * input_dx_mm)


def _grid_for_surface(
    native_grids: Mapping[int, UniformGrid2D], surface: int
) -> UniformGrid2D:
    try:
        grid = native_grids[surface]
    except KeyError as error:
        raise ValueError(f"native grid for surface {surface} is required") from error
    if not isinstance(grid, UniformGrid2D):
        raise ValueError(f"native grid for surface {surface} must be a UniformGrid2D")
    if grid.nx != 1024 or grid.ny != 1024:
        raise ValueError("the fixed biconic sampling matrix requires native 1024 grids")
    return grid


def _native_source(surface: int) -> NativeSurfaceSource:
    return NativeSurfaceSource(surface=surface, role="fresh_continuous")


def _case_output_source(sequence: str) -> CaseOutputSource:
    return CaseOutputSource(producer_case=f"S12_S13:{sequence}", surface=13)


def _case(
    *,
    segment_key: str,
    sequence: str,
    source_kind: SourceKind,
    strategy: DerivationStrategy,
    purpose: SamplingPurpose,
    nx: int,
    ny: int,
    dx_mm: float,
    dy_mm: float,
    source: SamplingSource,
    expected_output_dx_mm: float,
    expected_output_dy_mm: float,
    depends_on_case: str | None = None,
    repeat_count: int = 1,
    establishes_physical_convergence: bool = True,
) -> SamplingCase:
    return SamplingCase(
        case_id=f"{segment_key}:{sequence}",
        segment_key=segment_key,
        source_kind=source_kind,
        strategy=strategy,
        purpose=purpose,
        nx=nx,
        ny=ny,
        dx_mm=dx_mm,
        dy_mm=dy_mm,
        source=source,
        expected_output_dx_mm=expected_output_dx_mm,
        expected_output_dy_mm=expected_output_dy_mm,
        depends_on_case=depends_on_case,
        repeat_count=repeat_count,
        establishes_physical_convergence=establishes_physical_convergence,
    )


def build_segment_sampling_cases(
    native_grids: Mapping[int, UniformGrid2D],
    *,
    wavelength_vacuum_mm: float,
    refractive_index: float,
) -> tuple[SamplingCase, ...]:
    """Build the complete fixed three-branch matrix using logical sources only."""

    _medium_wavelength_mm(
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    s7 = _grid_for_surface(native_grids, 7)
    s12 = _grid_for_surface(native_grids, 12)
    s13 = _grid_for_surface(native_grids, 13)
    cases: list[SamplingCase] = []

    def oo_output(dx_mm: float) -> float:
        return outside_to_outside_output_sampling_mm(
            wavelength_vacuum_mm=wavelength_vacuum_mm,
            refractive_index=refractive_index,
            input_dx_mm=dx_mm,
            start_waist_distance_mm=S7_WAIST_DISTANCE_MM,
            end_waist_distance_mm=S8_WAIST_DISTANCE_MM,
        )

    for sequence, factor, strategy, purpose, repeat_count in (
        ("native", 1, "exact_copy", "native", 2),
        ("R2", 2, "fourier_refine_fixed_window", "combined_resolution", 1),
        ("R4", 4, "fourier_refine_fixed_window", "combined_resolution", 2),
    ):
        dx_mm = s7.dx_mm / factor
        dy_mm = s7.dy_mm / factor
        cases.append(
            _case(
                segment_key="S07_S08",
                sequence=sequence,
                source_kind="native_zbf" if factor == 1 else "derived_zbf",
                strategy=strategy,
                purpose=purpose,
                nx=1024 * factor,
                ny=1024 * factor,
                dx_mm=dx_mm,
                dy_mm=dy_mm,
                source=_native_source(7),
                expected_output_dx_mm=oo_output(dx_mm),
                expected_output_dy_mm=oo_output(dy_mm),
                repeat_count=repeat_count,
            )
        )
    cases.append(
        _case(
            segment_key="S07_S08",
            sequence="W2",
            source_kind="derived_zbf",
            strategy="zero_extend_fixed_sampling",
            purpose="window_control",
            nx=2048,
            ny=2048,
            dx_mm=s7.dx_mm,
            dy_mm=s7.dy_mm,
            source=_native_source(7),
            expected_output_dx_mm=oo_output(s7.dx_mm),
            expected_output_dy_mm=oo_output(s7.dy_mm),
            repeat_count=1,
        )
    )

    def stw_output(n: int, dx_mm: float) -> float:
        return stw_output_sampling_mm(
            wavelength_vacuum_mm=wavelength_vacuum_mm,
            refractive_index=refractive_index,
            waist_distance_mm=S12_STW_WAIST_DISTANCE_MM,
            n=n,
            input_dx_mm=dx_mm,
        )

    for axis in ("x", "y"):
        s12_step = getattr(s12, f"d{axis}_mm")
        s13_step = getattr(s13, f"d{axis}_mm")
        expected_s13_step = stw_output(1024, s12_step)
        if not math.isclose(
            s13_step,
            expected_s13_step,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError(
                f"native S13 d{axis} does not match the S12-to-S13 STW law"
            )

    for sequence, factor, repeat_count in (
        ("ZI0", 1, 2),
        ("ZI1", 2, 1),
        ("ZI2", 4, 2),
    ):
        n = 1024 * factor
        dx_mm = s12.dx_mm / factor
        dy_mm = s12.dy_mm / factor
        cases.append(
            _case(
                segment_key="S12_S13",
                sequence=sequence,
                source_kind="native_zbf" if factor == 1 else "derived_zbf",
                strategy="exact_copy" if factor == 1 else "fourier_refine_fixed_window",
                purpose="input_resolution",
                nx=n,
                ny=n,
                dx_mm=dx_mm,
                dy_mm=dy_mm,
                source=_native_source(12),
                expected_output_dx_mm=stw_output(n, dx_mm),
                expected_output_dy_mm=stw_output(n, dy_mm),
                repeat_count=repeat_count,
            )
        )

    zo_cases: dict[str, SamplingCase] = {}
    for sequence, factor, repeat_count in (
        ("ZO0", 1, 2),
        ("ZO1", 2, 1),
        ("ZO2", 4, 2),
    ):
        n = 1024 * factor
        case = _case(
            segment_key="S12_S13",
            sequence=sequence,
            source_kind="native_zbf" if factor == 1 else "derived_zbf",
            strategy="exact_copy" if factor == 1 else "zero_extend_fixed_sampling",
            purpose="output_resolution",
            nx=n,
            ny=n,
            dx_mm=s12.dx_mm,
            dy_mm=s12.dy_mm,
            source=_native_source(12),
            expected_output_dx_mm=stw_output(n, s12.dx_mm),
            expected_output_dy_mm=stw_output(n, s12.dy_mm),
            repeat_count=repeat_count,
        )
        cases.append(case)
        zo_cases[sequence] = case
    cases.append(
        _case(
            segment_key="S12_S13",
            sequence="ZJ2",
            source_kind="derived_zbf",
            strategy="zero_extend_then_fourier_refine",
            purpose="combined_resolution",
            nx=4096,
            ny=4096,
            dx_mm=s12.dx_mm / 2,
            dy_mm=s12.dy_mm / 2,
            source=_native_source(12),
            expected_output_dx_mm=stw_output(4096, s12.dx_mm / 2),
            expected_output_dy_mm=stw_output(4096, s12.dy_mm / 2),
        )
    )

    def wts_output(n: int, dx_mm: float) -> float:
        return wts_output_sampling_mm(
            wavelength_vacuum_mm=wavelength_vacuum_mm,
            refractive_index=refractive_index,
            waist_distance_mm=S14_WTS_WAIST_DISTANCE_MM,
            n=n,
            input_dx_mm=dx_mm,
        )

    cases.append(
        _case(
            segment_key="S13_S14",
            sequence="native",
            source_kind="native_zbf",
            strategy="exact_copy",
            purpose="native",
            nx=1024,
            ny=1024,
            dx_mm=s13.dx_mm,
            dy_mm=s13.dy_mm,
            source=_native_source(13),
            expected_output_dx_mm=wts_output(1024, s13.dx_mm),
            expected_output_dy_mm=wts_output(1024, s13.dy_mm),
            repeat_count=2,
        )
    )
    for sequence, upstream, repeat_count in (
        ("input_R2", "ZO1", 1),
        ("input_R4", "ZO2", 2),
    ):
        upstream_case = zo_cases[upstream]
        source = _case_output_source(upstream)
        cases.append(
            _case(
                segment_key="S13_S14",
                sequence=sequence,
                source_kind="chained_zemax_output",
                strategy="chained_zemax_output",
                purpose="input_resolution",
                nx=upstream_case.nx,
                ny=upstream_case.ny,
                dx_mm=upstream_case.expected_output_dx_mm,
                dy_mm=upstream_case.expected_output_dy_mm,
                source=source,
                expected_output_dx_mm=wts_output(
                    upstream_case.nx, upstream_case.expected_output_dx_mm
                ),
                expected_output_dy_mm=wts_output(
                    upstream_case.ny, upstream_case.expected_output_dy_mm
                ),
                depends_on_case=source.producer_case,
                repeat_count=repeat_count,
            )
        )
    for sequence, factor in (
        ("interp_sensitivity_R2", 2),
        ("interp_sensitivity_R4", 4),
    ):
        dx_mm = s13.dx_mm / factor
        dy_mm = s13.dy_mm / factor
        cases.append(
            _case(
                segment_key="S13_S14",
                sequence=sequence,
                source_kind="derived_zbf",
                strategy="fourier_refine_fixed_window",
                purpose="interpolation_sensitivity",
                nx=1024 * factor,
                ny=1024 * factor,
                dx_mm=dx_mm,
                dy_mm=dy_mm,
                source=_native_source(13),
                expected_output_dx_mm=wts_output(1024 * factor, dx_mm),
                expected_output_dy_mm=wts_output(1024 * factor, dy_mm),
                establishes_physical_convergence=False,
            )
        )
    cases.append(
        _case(
            segment_key="S13_S14",
            sequence="output_O2",
            source_kind="derived_zbf",
            strategy="zero_extend_fixed_sampling",
            purpose="output_resolution",
            nx=2048,
            ny=2048,
            dx_mm=s13.dx_mm,
            dy_mm=s13.dy_mm,
            source=_native_source(13),
            expected_output_dx_mm=wts_output(2048, s13.dx_mm),
            expected_output_dy_mm=wts_output(2048, s13.dy_mm),
            repeat_count=2,
        )
    )
    input_r2 = zo_cases["ZO1"]
    combined_source = _case_output_source("ZO1")
    cases.append(
        _case(
            segment_key="S13_S14",
            sequence="combined",
            source_kind="derived_zbf",
            strategy="zero_extend_fixed_sampling",
            purpose="combined_resolution",
            nx=4096,
            ny=4096,
            dx_mm=input_r2.expected_output_dx_mm,
            dy_mm=input_r2.expected_output_dy_mm,
            source=combined_source,
            expected_output_dx_mm=wts_output(
                4096, input_r2.expected_output_dx_mm
            ),
            expected_output_dy_mm=wts_output(
                4096, input_r2.expected_output_dy_mm
            ),
            depends_on_case=combined_source.producer_case,
        )
    )
    return tuple(cases)


def _source_manifest_payload(source: SamplingSource) -> dict[str, object]:
    if isinstance(source, NativeSurfaceSource):
        return {
            "kind": "native_surface",
            "surface": source.surface,
            "role": source.role,
        }
    if isinstance(source, CaseOutputSource):
        return {
            "kind": "case_output",
            "producer_case": source.producer_case,
            "surface": source.surface,
        }
    raise TypeError("sampling source must remain a logical source")


def write_sampling_manifest(
    path: str | Path, cases: Iterable[SamplingCase]
) -> Path:
    """Write a deterministic frozen plan containing no guessed artifact paths."""

    output = Path(path)
    case_tuple = tuple(cases)
    if len({case.case_id for case in case_tuple}) != len(case_tuple):
        raise ValueError("sampling manifest case ids must be unique")
    payload = {
        "format_version": 1,
        "cases": [
            {
                "case_id": case.case_id,
                "segment_key": case.segment_key,
                "source_kind": case.source_kind,
                "strategy": case.strategy,
                "purpose": case.purpose,
                "nx": case.nx,
                "ny": case.ny,
                "dx_mm": case.dx_mm,
                "dy_mm": case.dy_mm,
                "source": _source_manifest_payload(case.source),
                "expected_output_dx_mm": case.expected_output_dx_mm,
                "expected_output_dy_mm": case.expected_output_dy_mm,
                "depends_on_case": case.depends_on_case,
                "repeat_count": case.repeat_count,
                "establishes_physical_convergence": (
                    case.establishes_physical_convergence
                ),
            }
            for case in case_tuple
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "S12_STW_WAIST_DISTANCE_MM",
    "S14_WTS_WAIST_DISTANCE_MM",
    "S7_WAIST_DISTANCE_MM",
    "S8_WAIST_DISTANCE_MM",
    "build_segment_sampling_cases",
    "outside_to_outside_output_sampling_mm",
    "stw_output_sampling_mm",
    "write_sampling_manifest",
    "wts_output_sampling_mm",
]
