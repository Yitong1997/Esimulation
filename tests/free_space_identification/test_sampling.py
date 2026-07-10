from __future__ import annotations

import json
from pathlib import Path

import pytest

from sandbox.free_space_algorithm_identification.models import (
    CaseOutputSource,
    NativeSurfaceSource,
    UniformGrid2D,
)
from sandbox.free_space_algorithm_identification.sampling import (
    build_segment_sampling_cases,
    outside_to_outside_output_sampling_mm,
    stw_output_sampling_mm,
    write_sampling_manifest,
    wts_output_sampling_mm,
)


def _native_grids() -> dict[int, UniformGrid2D]:
    s12_dx_mm = 0.6339798584
    s12_dy_mm = 0.52734375
    wavelength_vacuum_mm = 0.01064
    stw_waist_distance_mm = 608.615263635412
    return {
        7: UniformGrid2D.centered(
            nx=1024, ny=1024, dx_mm=0.08125, dy_mm=0.09375
        ),
        12: UniformGrid2D.centered(
            nx=1024, ny=1024, dx_mm=s12_dx_mm, dy_mm=s12_dy_mm
        ),
        13: UniformGrid2D.centered(
            nx=1024,
            ny=1024,
            dx_mm=(
                wavelength_vacuum_mm
                * stw_waist_distance_mm
                / (1024 * s12_dx_mm)
            ),
            dy_mm=(
                wavelength_vacuum_mm
                * stw_waist_distance_mm
                / (1024 * s12_dy_mm)
            ),
        ),
    }


def _cases_by_segment_and_sequence():
    cases = build_segment_sampling_cases(
        _native_grids(),
        wavelength_vacuum_mm=0.01064,
        refractive_index=1.0,
    )
    return {(case.segment_key, case.sequence): case for case in cases}


def test_s7_s8_fixed_window_refines_input_and_output_sampling() -> None:
    cases = _cases_by_segment_and_sequence()
    native = cases["S07_S08", "native"]
    r2 = cases["S07_S08", "R2"]
    r4 = cases["S07_S08", "R4"]

    assert (r2.nx, r2.ny, r2.dx_mm, r2.dy_mm) == (
        2048,
        2048,
        native.dx_mm / 2,
        native.dy_mm / 2,
    )
    assert (r4.nx, r4.ny, r4.dx_mm, r4.dy_mm) == (
        4096,
        4096,
        native.dx_mm / 4,
        native.dy_mm / 4,
    )
    for axis in ("x", "y"):
        input_dx = getattr(native, f"d{axis}_mm")
        expected = outside_to_outside_output_sampling_mm(
            wavelength_vacuum_mm=0.01064,
            refractive_index=1.0,
            input_dx_mm=input_dx,
            start_waist_distance_mm=239.982966226840,
            end_waist_distance_mm=608.582967581705,
        )
        assert getattr(native, f"expected_output_d{axis}_mm") == pytest.approx(expected)
        assert getattr(r2, f"expected_output_d{axis}_mm") == pytest.approx(expected / 2)
        assert getattr(r4, f"expected_output_d{axis}_mm") == pytest.approx(expected / 4)
    assert native.repeat_count == r4.repeat_count == 2
    assert cases["S07_S08", "W2"].repeat_count == 1


def test_s12_s13_fixed_window_n_does_not_refine_stw_output_sampling() -> None:
    d0 = stw_output_sampling_mm(
        wavelength_vacuum_mm=0.01064,
        refractive_index=1.0,
        waist_distance_mm=608.615263635412,
        n=1024,
        input_dx_mm=0.6339798584,
    )
    d1 = stw_output_sampling_mm(
        wavelength_vacuum_mm=0.01064,
        refractive_index=1.0,
        waist_distance_mm=608.615263635412,
        n=2048,
        input_dx_mm=0.6339798584 / 2,
    )
    assert d1 == pytest.approx(d0, rel=1e-14)
    assert stw_output_sampling_mm(
        wavelength_vacuum_mm=0.01064,
        refractive_index=1.6,
        waist_distance_mm=608.615263635412,
        n=1024,
        input_dx_mm=0.6339798584,
    ) == pytest.approx(d0 / 1.6)

    cases = _cases_by_segment_and_sequence()
    zi = [cases["S12_S13", name] for name in ("ZI0", "ZI1", "ZI2")]
    assert [case.nx for case in zi] == [1024, 2048, 4096]
    assert [case.dx_mm for case in zi] == [
        zi[0].dx_mm,
        zi[0].dx_mm / 2,
        zi[0].dx_mm / 4,
    ]
    assert [case.expected_output_dx_mm for case in zi] == pytest.approx(
        [zi[0].expected_output_dx_mm] * 3
    )
    assert zi[0].repeat_count == zi[2].repeat_count == 2


def test_s12_s13_window_expansion_refines_stw_output_sampling() -> None:
    cases = _cases_by_segment_and_sequence()
    zo = [cases["S12_S13", name] for name in ("ZO0", "ZO1", "ZO2")]

    assert [case.nx for case in zo] == [1024, 2048, 4096]
    assert [case.dx_mm for case in zo] == [zo[0].dx_mm] * 3
    assert [case.expected_output_dx_mm for case in zo] == pytest.approx(
        [
            zo[0].expected_output_dx_mm,
            zo[0].expected_output_dx_mm / 2,
            zo[0].expected_output_dx_mm / 4,
        ]
    )
    assert zo[1].strategy == zo[2].strategy == "zero_extend_fixed_sampling"
    assert zo[0].repeat_count == zo[2].repeat_count == 2


def test_s13_s14_fixed_window_does_not_refine_wts_output_sampling() -> None:
    d0 = wts_output_sampling_mm(
        wavelength_vacuum_mm=0.01064,
        refractive_index=1.25,
        waist_distance_mm=1.984736370234,
        n=1024,
        input_dx_mm=0.009975,
    )
    d1 = wts_output_sampling_mm(
        wavelength_vacuum_mm=0.01064,
        refractive_index=1.25,
        waist_distance_mm=1.984736370234,
        n=2048,
        input_dx_mm=0.009975 / 2,
    )
    assert d1 == pytest.approx(d0, rel=1e-14)

    cases = _cases_by_segment_and_sequence()
    native = cases["S13_S14", "native"]
    r2 = cases["S13_S14", "input_R2"]
    r4 = cases["S13_S14", "input_R4"]
    assert [case.expected_output_dx_mm for case in (native, r2, r4)] == pytest.approx(
        [native.expected_output_dx_mm] * 3
    )
    output_o2 = cases["S13_S14", "output_O2"]
    assert output_o2.expected_output_dx_mm == pytest.approx(
        native.expected_output_dx_mm / 2
    )


def test_s13_s14_high_resolution_input_depends_on_chained_s12_s13_output() -> None:
    cases = _cases_by_segment_and_sequence()
    for sequence, upstream in (("input_R2", "ZO1"), ("input_R4", "ZO2")):
        case = cases["S13_S14", sequence]
        assert case.source_kind == "chained_zemax_output"
        assert case.strategy == "chained_zemax_output"
        assert isinstance(case.source, CaseOutputSource)
        assert case.source.producer_case == f"S12_S13:{upstream}"
        assert case.source.surface == 13
        assert case.depends_on_case == case.source.producer_case
    combined = cases["S13_S14", "combined"]
    assert isinstance(combined.source, CaseOutputSource)
    assert combined.source.producer_case == "S12_S13:ZO1"
    assert combined.strategy == "zero_extend_fixed_sampling"

    inconsistent = _native_grids()
    native_s13 = inconsistent[13]
    inconsistent[13] = UniformGrid2D.centered(
        nx=1024,
        ny=1024,
        dx_mm=native_s13.dx_mm * 1.01,
        dy_mm=native_s13.dy_mm,
    )
    with pytest.raises(ValueError, match="native S13.*STW"):
        build_segment_sampling_cases(
            inconsistent,
            wavelength_vacuum_mm=0.01064,
            refractive_index=1.0,
        )


def test_native_s13_interpolation_is_not_labeled_physical_convergence(
    tmp_path: Path,
) -> None:
    all_cases = build_segment_sampling_cases(
        _native_grids(),
        wavelength_vacuum_mm=0.01064,
        refractive_index=1.0,
    )
    cases = {(case.segment_key, case.sequence): case for case in all_cases}
    for sequence in ("interp_sensitivity_R2", "interp_sensitivity_R4"):
        case = cases["S13_S14", sequence]
        assert case.purpose == "interpolation_sensitivity"
        assert case.establishes_physical_convergence is False
        assert isinstance(case.source, NativeSurfaceSource)
    manifest = write_sampling_manifest(tmp_path / "sampling.json", all_cases)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert all("path" not in item["source"] for item in payload["cases"])
    assert {
        item["source"]["kind"]
        for item in payload["cases"]
    } == {"native_surface", "case_output"}
    assert next(
        item
        for item in payload["cases"]
        if item["case_id"] == "S13_S14:interp_sensitivity_R4"
    )["establishes_physical_convergence"] is False
