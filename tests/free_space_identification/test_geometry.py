from __future__ import annotations

import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sandbox.free_space_algorithm_identification.biconic_case import BICONIC_SEGMENTS
from sandbox.free_space_algorithm_identification.geometry import (
    ReportNumber,
    load_segment_geometry,
    parse_native_intermediate_trace,
)
from sandbox.free_space_algorithm_identification.zbf_binary import (
    LosslessZbf,
    RawZbfHeader,
)


_RAW_Z_BY_SURFACE = {
    7: -239.982966226840,
    8: -608.582967581705,
    12: -608.615263635412,
    13: -0.015263632634,
    14: 1.984736370234,
}


def _beam(surface: int) -> LosslessZbf:
    wavelength_mm = 0.01064
    refractive_index = 1.0
    rayleigh_mm = 0.761678804035
    waist_mm = np.sqrt(
        rayleigh_mm * wavelength_mm / (np.pi * refractive_index)
    )
    ints = (1, 2, 2, 0, 0, 0, 0, 0, 0)
    z = _RAW_Z_BY_SURFACE[surface]
    doubles = (
        0.1,
        0.1,
        z,
        rayleigh_mm,
        waist_mm,
        z,
        rayleigh_mm,
        waist_mm,
        wavelength_mm,
        refractive_index,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    header = RawZbfHeader.from_bytes(struct.pack("<9i20d", *ints, *doubles))
    return LosslessZbf(
        path=None,
        source_sha256="",
        header=header,
        ex=np.ones((2, 2), dtype=np.complex128),
        ey=None,
        trailing_bytes=b"",
    )


def _orientation_block(matrix: np.ndarray) -> str:
    return "\n".join("\t" + "\t".join(f"{value:.9f}" for value in row) for row in matrix)


def _surface_block(
    surface: int,
    *,
    signed_distance_token: str,
    state: str,
    branch_text: str,
    orientation_after: np.ndarray | None = None,
) -> str:
    before = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.000003021],
            [0.0, -0.000003021, 1.0],
        ]
    )
    after = before if orientation_after is None else orientation_after
    return f"""Surface transfer from before {surface} to after {surface}
Orientation matrix before:
{_orientation_block(before)}
Incoming pilot radius x, y:\t-2.39985E+02\t-2.39985E+02
Outgoing pilot radius x, y:\t-2.39985E+02\t-2.39985E+02
Orientation matrix after:
{_orientation_block(after)}

Propagating beam distance:\t{signed_distance_token}
Starting pilot beam waist    x, y:\t5.0790E-02\t5.0790E-02
Starting pilot beam position x, y:\t{_RAW_Z_BY_SURFACE[surface]:.14E}\t{_RAW_Z_BY_SURFACE[surface]:.14E}
Starting pilot beam Rayleigh x, y:\t7.6168E-01\t7.6168E-01
X {state} Rayleigh range.
Y {state} Rayleigh range.
Internal transmittance    :\t1.000000
Using {branch_text} propagator.
"""


def _native_report(
    *,
    s7_distance: str = "-3.6860E+02",
    s7_branch: str = "Outside to Outside",
    s13_state: str = "Inside",
    s8_orientation_after: np.ndarray | None = None,
) -> str:
    chief_rows = "\n".join(
        f"{surface:4d}\t  0.000000E+00\t  {surface / 1000:.6E}\t  0.000000E+00\t  0.000000E+00\t  3.021480E-06\t  1.000000E+00"
        for surface in (7, 8, 12, 13, 14)
    )
    blocks = (
        _surface_block(
            7,
            signed_distance_token=s7_distance,
            state="Outside",
            branch_text=s7_branch,
        ),
        _surface_block(
            8,
            signed_distance_token="0.0000E+00",
            state="Outside",
            branch_text="Outside to Outside",
            orientation_after=s8_orientation_after,
        ),
        _surface_block(
            12,
            signed_distance_token="6.0860E+02",
            state="Outside",
            branch_text="Outside to Inside",
        ),
        _surface_block(
            13,
            signed_distance_token="2.0000E+00",
            state=s13_state,
            branch_text="Inside to Outside",
        ),
        _surface_block(
            14,
            signed_distance_token="0.0000E+00",
            state="Outside",
            branch_text="Outside to Outside",
        ),
    )
    joined_blocks = "\n".join(blocks)
    return f"""POP Propagation Report

Starting Propagation

Chief ray data:
Surf         X              Y              Z              L              M              N
{chief_rows}

{joined_blocks}
Final Beam Data:
"""


def _write_report(tmp_path: Path, text: str) -> Path:
    report = tmp_path / "native-pop-report.txt"
    report.write_text(text, encoding="utf-16")
    return report


def test_load_segment_geometry_records_model_report_and_raw_pilot_distances(
    tmp_path: Path,
) -> None:
    report_text = _native_report()
    trace = parse_native_intermediate_trace(report_text)
    assert [sample.surface for sample in trace.chief_rays] == [7, 8, 12, 13, 14]
    assert trace.chief_rays[0].y_mm.text == "7.000000E-03"
    assert trace.transfer_for_surface(7).signed_distance_mm == ReportNumber(
        text="-3.6860E+02", value=-368.6, last_digit_resolution=0.01
    )
    assert trace.transfer_for_surface(12).signed_distance_mm.last_digit_resolution == 0.01
    assert trace.transfer_for_surface(13).signed_distance_mm.last_digit_resolution == 0.0001
    report = _write_report(tmp_path, report_text)
    by_key = {}
    for segment in BICONIC_SEGMENTS:
        geometry = load_segment_geometry(
            report,
            segment=segment,
            start_beam=_beam(segment.start_surface),
            end_beam=_beam(segment.end_surface),
        )
        by_key[segment.key] = geometry
        assert geometry.propagation_distance_mm == segment.model_distance_mm
        assert geometry.model_distance_mm == segment.model_distance_mm
        assert geometry.raw_pilot_delta_mm == pytest.approx(
            _RAW_Z_BY_SURFACE[segment.end_surface]
            - _RAW_Z_BY_SURFACE[segment.start_surface],
            rel=0.0,
            abs=1e-15,
        )
        np.testing.assert_allclose(
            geometry.transverse_basis_change,
            np.eye(2),
            rtol=0.0,
            atol=1e-9,
        )
        with pytest.raises(ValueError):
            geometry.transverse_basis_change.setflags(write=True)

    assert by_key["S07_S08"].report_signed_distance_mm.value == -368.6
    assert by_key["S07_S08"].raw_pilot_delta_mm == pytest.approx(
        -368.600001354865, rel=0.0, abs=1e-12
    )
    assert by_key["S12_S13"].report_signed_distance_mm.value == 608.6
    assert by_key["S12_S13"].raw_pilot_delta_mm == pytest.approx(
        608.600000002778, rel=0.0, abs=1e-12
    )
    assert by_key["S13_S14"].report_signed_distance_mm.value == 2.0
    assert by_key["S13_S14"].raw_pilot_delta_mm == pytest.approx(
        2.000000002868, rel=0.0, abs=1e-12
    )


def test_segment_geometry_fails_closed_on_distance_branch_delta_or_basis(
    tmp_path: Path,
) -> None:
    segment = BICONIC_SEGMENTS[0]
    wrong_end = replace(_beam(8), header=_beam(7).header)
    tilted = np.array(
        [
            [0.999999, -0.001, 0.0],
            [0.001, 0.999999, 0.000003021],
            [0.0, -0.000003021, 1.0],
        ]
    )
    cases = (
        (_native_report(s7_distance="-3.6859E+02"), _beam(8)),
        (_native_report(s7_branch="Outside to Inside"), _beam(8)),
        (_native_report(), wrong_end),
        (_native_report(s8_orientation_after=tilted), _beam(8)),
    )
    for index, (report_text, end_beam) in enumerate(cases):
        report = tmp_path / f"invalid-{index}.txt"
        report.write_text(report_text, encoding="utf-8")
        with pytest.raises(ValueError):
            load_segment_geometry(
                report,
                segment=segment,
                start_beam=_beam(7),
                end_beam=end_beam,
            )
