from __future__ import annotations

import struct
from dataclasses import asdict, fields, replace
from pathlib import Path

import pytest

from sandbox.free_space_algorithm_identification.biconic_case import (
    BICONIC_SEGMENTS,
    S7,
)
from sandbox.free_space_algorithm_identification.geometry import ReportNumber
from sandbox.free_space_algorithm_identification.native_report import (
    NativePopRequest,
    NativeSettingsReadback,
    parse_native_pop_report,
    validate_native_transfer,
    validate_output_sampling,
    validate_settings_readback,
)
from sandbox.free_space_algorithm_identification.zbf_binary import RawZbfHeader


_FIXTURES = Path(__file__).with_name("fixtures")


def _fixture(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


def _request() -> NativePopRequest:
    return NativePopRequest(
        start_surface=7,
        end_surface=8,
        nx=1024,
        ny=1024,
        sample_size_enum="S_1024x1024",
        x_width_mm=256.01,
        y_width_mm=256.01,
        wavelength_number=1,
        wavelength_vacuum_mm=0.01064,
        refractive_index=1.0,
        field_number=1,
        use_polarization=False,
        normalization_mode="total_power",
        normalization_value=1.0,
        input_beam_file="biconic_focus_test_0007.ZBF",
        output_beam_file="biconic_focus_test_0008.ZBF",
        save_output_beam=True,
        save_beam_at_all_surfaces=False,
    )


def _readback(request: NativePopRequest | None = None) -> NativeSettingsReadback:
    actual = _request() if request is None else request
    return NativeSettingsReadback(**asdict(actual))


def _header(
    *,
    nx: int = 1024,
    ny: int = 1024,
    dx_mm: float,
    dy_mm: float,
    zx_mm: float,
    zy_mm: float | None = None,
    rayleigh_mm: float = 0.76159,
    waist_mm: float = 0.050788,
    is_polarized: int = 0,
    wavelength_vacuum_mm: float = 0.01064,
    refractive_index: float = 1.0,
) -> RawZbfHeader:
    ints = (1, nx, ny, is_polarized, 0, 0, 0, 0, 0)
    doubles = (
        dx_mm,
        dy_mm,
        zx_mm,
        rayleigh_mm,
        waist_mm,
        zx_mm if zy_mm is None else zy_mm,
        rayleigh_mm,
        waist_mm,
        wavelength_vacuum_mm,
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
    return RawZbfHeader.from_bytes(struct.pack("<9i20d", *ints, *doubles))


def _compact_parts(name: str) -> tuple[str, str]:
    text = _fixture(name)
    marker = "Starting delta X, Y size:"
    first = text.index(marker)
    second = text.index(marker, first + len(marker))
    return text[:second], text[second:]


def _synthetic_full_report() -> str:
    oo_start, oo_end = _compact_parts("native_oo_report.txt")
    oi_start, _ = _compact_parts("native_oi_report.txt")
    io_start, io_end = _compact_parts("native_io_report.txt")
    return "\n".join(
        (
            "Surface transfer from before 7 to after 7",
            oo_start,
            "Surface transfer from before 8 to after 8",
            oo_end,
            "Surface transfer from before 12 to after 12",
            oi_start,
            "Surface transfer from before 13 to after 13",
            io_start,
            "Surface transfer from before 14 to after 14",
            io_end,
        )
    )


def _segment_headers(index: int) -> tuple[RawZbfHeader, RawZbfHeader]:
    values = (
        (
            dict(
                dx_mm=0.25000748338153306,
                dy_mm=0.25001199074957775,
                zx_mm=-239.98296622683966,
                rayleigh_mm=0.7615935452618818,
                waist_mm=0.05078757830533947,
            ),
            dict(
                dx_mm=0.6340045637732635,
                dy_mm=0.6340159941987548,
                zx_mm=-608.5829675817045,
                rayleigh_mm=0.7615935486486892,
                waist_mm=0.05078757841826568,
            ),
        ),
        (
            dict(
                dx_mm=0.6339798581276602,
                dy_mm=0.634037020947161,
                zx_mm=-608.6152636354115,
                rayleigh_mm=0.7616788040351931,
                waist_mm=0.05079042100632709,
            ),
            dict(
                dx_mm=0.00997491149385745,
                dy_mm=0.00997401218664598,
                zx_mm=-0.015263632633932584,
                rayleigh_mm=0.7616788040351954,
                waist_mm=0.05079042100632716,
            ),
        ),
        (
            dict(
                dx_mm=0.00997491149385745,
                dy_mm=0.00997401218664598,
                zx_mm=-0.015263632633932584,
                rayleigh_mm=0.7616788040351954,
                waist_mm=0.05079042100632716,
            ),
            dict(
                dx_mm=0.002067452060096705,
                dy_mm=0.002067638471995169,
                zx_mm=1.9847363702338578,
                rayleigh_mm=0.7616788066082827,
                waist_mm=0.050790421092116726,
            ),
        ),
    )
    source_values, output_values = values[index]
    return _header(**source_values), _header(**output_values)


def _header_like(header: RawZbfHeader, **changes) -> RawZbfHeader:
    values = dict(
        nx=header.nx,
        ny=header.ny,
        dx_mm=header.dx,
        dy_mm=header.dy,
        zx_mm=header.zx,
        zy_mm=header.zy,
        rayleigh_mm=header.rx,
        waist_mm=header.wx,
        is_polarized=header.is_polarized,
        wavelength_vacuum_mm=header.wavelength_vacuum_mm,
        refractive_index=header.refractive_index,
    )
    values.update(changes)
    return _header(**values)


@pytest.mark.parametrize(
    (
        "fixture_name",
        "branch",
        "label",
        "signed_distance_mm",
        "input_grid",
        "output_grid",
        "warnings",
        "segment_index",
    ),
    (
        (
            "native_oo_report.txt",
            "OO",
            "Using Outside to Outside propagator.",
            -368.6,
            ((0.25001, 0.25001), (256.01, 256.01)),
            ((0.63400, 0.63402), (649.22, 649.23)),
            (),
            0,
        ),
        (
            "native_oi_report.txt",
            "OI",
            "Using Outside to Inside propagator.",
            608.6,
            ((0.63398, 0.63404), (649.20, 649.25)),
            ((0.0099749, 0.0099740), (10.214, 10.213)),
            ("**** WARNING: Low sampling of pilot beam detected.",),
            1,
        ),
        (
            "native_io_report.txt",
            "IO",
            "Using Inside to Outside propagator.",
            2.0,
            ((0.0099749, 0.0099740), (10.214, 10.213)),
            ((0.0020675, 0.0020676), (2.1171, 2.1173)),
            (
                "**** WARNING: Low sampling of pilot beam detected.",
                "**** WARNING: Low sampling of pilot beam detected.",
            ),
            2,
        ),
    ),
)
def test_parses_literal_oo_oi_io_distance_grids_and_all_warnings(
    fixture_name: str,
    branch: str,
    label: str,
    signed_distance_mm: float,
    input_grid: tuple[tuple[float, float], tuple[float, float]],
    output_grid: tuple[tuple[float, float], tuple[float, float]],
    warnings: tuple[str, ...],
    segment_index: int,
) -> None:
    report = parse_native_pop_report(_fixture(fixture_name))
    segment = BICONIC_SEGMENTS[segment_index]
    selected = parse_native_pop_report(
        _synthetic_full_report(), segment=segment
    )
    selected_by_start = parse_native_pop_report(
        _synthetic_full_report(), start_surface=segment.start_surface
    )

    assert report.branch == branch
    assert report.propagator_label == label
    assert report.signed_distance_mm.value == signed_distance_mm
    assert tuple(
        value.value for value in report.input_sampling.interval_mm
    ) == input_grid[0]
    assert tuple(
        value.value for value in report.input_sampling.width_mm
    ) == input_grid[1]
    assert tuple(
        value.value for value in report.output_sampling.interval_mm
    ) == output_grid[0]
    assert tuple(
        value.value for value in report.output_sampling.width_mm
    ) == output_grid[1]
    assert report.warnings == warnings
    assert selected == report
    assert selected_by_start == report

    if segment_index == 0:
        full = _synthetic_full_report()
        with pytest.raises(ValueError, match="selector"):
            parse_native_pop_report(full)
        with pytest.raises(ValueError, match="unique"):
            parse_native_pop_report(
                full + "\n" + full.split("Surface transfer from before 8", 1)[0],
                segment=segment,
            )
        with pytest.raises(ValueError, match="adjacent output"):
            parse_native_pop_report(
                full.replace(
                    "Surface transfer from before 8 to after 8",
                    "Surface transfer from before 9 to after 9",
                    1,
                ),
                segment=segment,
            )


@pytest.mark.parametrize("end_surface", (8, 7), ids=("transfer", "identity"))
def test_api_readback_alone_proves_start_end_n_enum_and_width(
    end_surface: int,
) -> None:
    report = parse_native_pop_report(_fixture("native_oo_report.txt"))
    request = replace(_request(), end_surface=end_surface)
    readback = _readback(request)

    validate_settings_readback(request, readback)
    report_fields = {field.name for field in fields(report)}
    assert report_fields.isdisjoint(
        {
            "start_surface",
            "end_surface",
            "nx",
            "ny",
            "sample_size_enum",
            "wavelength_number",
            "wavelength_vacuum_mm",
            "refractive_index",
            "field_number",
            "normalization_mode",
        }
    )

    mismatches = {
        "start_surface": 6,
        "end_surface": 9,
        "nx": 512,
        "ny": 512,
        "sample_size_enum": "S_512x512",
        "x_width_mm": 256.02,
        "y_width_mm": 256.02,
        "wavelength_number": 2,
        "wavelength_vacuum_mm": 0.01065,
        "refractive_index": 1.1,
        "field_number": 2,
        "use_polarization": True,
        "normalization_mode": "peak_irradiance",
        "normalization_value": 2.0,
        "input_beam_file": "wrong-input.ZBF",
        "output_beam_file": "wrong-output.ZBF",
        "save_output_beam": False,
        "save_beam_at_all_surfaces": True,
    }
    for field_name, wrong_value in mismatches.items():
        with pytest.raises(ValueError, match=field_name):
            validate_settings_readback(
                request,
                replace(readback, **{field_name: wrong_value}),
            )

    for invalid_enum in ("S_512x512", "1024x1024", "S_auto"):
        with pytest.raises(ValueError, match="sample_size_enum"):
            replace(request, sample_size_enum=invalid_enum)
    for numeric_field in (
        "x_width_mm",
        "y_width_mm",
        "wavelength_vacuum_mm",
        "refractive_index",
        "normalization_value",
    ):
        for invalid_value in (True, 1, "1.0"):
            with pytest.raises(ValueError, match=numeric_field):
                replace(request, **{numeric_field: invalid_value})


@pytest.mark.parametrize(
    ("fixture_name", "segment_index"),
    (
        ("native_oo_report.txt", 0),
        ("native_oi_report.txt", 1),
        ("native_io_report.txt", 2),
    ),
)
def test_request_readback_report_and_zbf_consistency_fails_closed(
    fixture_name: str,
    segment_index: int,
) -> None:
    segment = BICONIC_SEGMENTS[segment_index]
    source, target = _segment_headers(segment_index)
    request = replace(
        _request(),
        start_surface=segment.start_surface,
        end_surface=segment.end_surface,
        x_width_mm=source.nx * source.dx,
        y_width_mm=source.ny * source.dy,
        input_beam_file=segment.source_zbf_name,
        output_beam_file=segment.target_zbf_name,
    )
    readback = _readback(request)
    report = parse_native_pop_report(_fixture(fixture_name))

    validate_settings_readback(request, readback)
    validate_output_sampling(report, readback, source, target)
    validate_native_transfer(report, segment, source, target)
    assert S7.axis_sign == -1
    assert report.signed_distance_mm.contains(
        segment.start_convention.axis_sign * segment.model_distance_mm
    )
    assert report.signed_distance_mm.contains(target.zx - source.zx)
    assert report.signed_distance_mm.contains(target.zy - source.zy)

    polarized_readback = replace(readback, use_polarization=True)
    failures = (
        (
            "settings readback",
            lambda: validate_settings_readback(
                request, replace(readback, output_beam_file="wrong.ZBF")
            ),
        ),
        (
            "sampling",
            lambda: validate_output_sampling(
                report,
                readback,
                source,
                _header_like(target, nx=512, ny=512),
            ),
        ),
        (
            "sampling",
            lambda: validate_output_sampling(
                report,
                readback,
                _header_like(source, dx_mm=source.dx * 1.01),
                target,
            ),
        ),
        (
            "sampling",
            lambda: validate_output_sampling(
                report,
                readback,
                source,
                _header_like(target, dx_mm=target.dx * 1.01),
            ),
        ),
        (
            "wavelength",
            lambda: validate_output_sampling(
                report,
                readback,
                source,
                _header_like(
                    target,
                    wavelength_vacuum_mm=target.wavelength_vacuum_mm * 1.01,
                ),
            ),
        ),
        (
            "refractive index",
            lambda: validate_output_sampling(
                report,
                readback,
                source,
                _header_like(target, refractive_index=1.01),
            ),
        ),
        (
            "polarization",
            lambda: validate_output_sampling(
                report,
                polarized_readback,
                _header_like(source, is_polarized=2),
                _header_like(target, is_polarized=2),
            ),
        ),
        (
            "axis-sign distance",
            lambda: validate_native_transfer(
                replace(
                    report,
                    signed_distance_mm=ReportNumber.from_token("0.0000E+00"),
                ),
                segment,
                source,
                target,
            ),
        ),
        (
            "axis-sign distance",
            lambda: validate_native_transfer(
                report,
                segment,
                source,
                _header_like(target, zx_mm=target.zx + 0.01),
            ),
        ),
        (
            "axis-sign distance",
            lambda: validate_native_transfer(
                report,
                segment,
                source,
                _header_like(target, zy_mm=target.zy + 0.01),
            ),
        ),
        (
            "pilot",
            lambda: validate_native_transfer(
                report,
                segment,
                source,
                _header_like(target, waist_mm=target.wx * 1.1),
            ),
        ),
        (
            "branch",
            lambda: validate_native_transfer(
                report,
                BICONIC_SEGMENTS[(segment_index + 1) % 3],
                source,
                target,
            ),
        ),
    )
    for expected_reason, validation in failures:
        with pytest.raises(ValueError, match=expected_reason):
            validation()


def test_propagator_label_identifies_branch_but_never_kernel() -> None:
    report = parse_native_pop_report(_fixture("native_oi_report.txt"))

    assert report.branch == "OI"
    assert report.propagator_label == "Using Outside to Inside propagator."
    assert report.kernel_identity is None
    assert "kernel" not in {field.name for field in fields(report)}
