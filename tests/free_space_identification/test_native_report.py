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
        0.0006328,
        1.0,
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


def _oo_headers() -> tuple[RawZbfHeader, RawZbfHeader]:
    source = _header(
        dx_mm=256.01 / 1024,
        dy_mm=256.01 / 1024,
        zx_mm=-239.98,
    )
    target = _header(
        dx_mm=0.634,
        dy_mm=0.6340155,
        zx_mm=-608.58,
    )
    return source, target


@pytest.mark.parametrize(
    (
        "fixture_name",
        "branch",
        "label",
        "signed_distance_mm",
        "input_grid",
        "output_grid",
        "warnings",
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
        ),
        (
            "native_oi_report.txt",
            "OI",
            "Using Outside to Inside propagator.",
            608.6,
            ((0.63398, 0.63404), (649.20, 649.25)),
            ((0.0099749, 0.0099740), (10.214, 10.213)),
            ("**** WARNING: Low sampling of pilot beam detected.",),
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
) -> None:
    report = parse_native_pop_report(_fixture(fixture_name))

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


def test_request_readback_report_and_zbf_consistency_fails_closed() -> None:
    request = _request()
    readback = _readback(request)
    report = parse_native_pop_report(_fixture("native_oo_report.txt"))
    segment = BICONIC_SEGMENTS[0]
    source, target = _oo_headers()

    validate_settings_readback(request, readback)
    validate_output_sampling(report, readback, source, target)
    validate_native_transfer(report, segment, source, target)
    assert S7.axis_sign == -1
    assert report.signed_distance_mm.contains(
        S7.axis_sign * segment.model_distance_mm
    )

    inside_rayleigh = (
        ReportNumber.from_token("7.0000E+02"),
        ReportNumber.from_token("7.0000E+02"),
    )
    oi_report = replace(
        report,
        branch="OI",
        propagator_label="Using Outside to Inside propagator.",
        end_pilot=replace(
            report.end_pilot,
            rayleigh_mm=inside_rayleigh,
            inside=(True, True),
        ),
    )
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
                _header(
                    nx=512,
                    ny=512,
                    dx_mm=0.634,
                    dy_mm=0.6340155,
                    zx_mm=-608.58,
                ),
            ),
        ),
        (
            "sampling",
            lambda: validate_output_sampling(
                report,
                readback,
                _header(
                    dx_mm=0.25002,
                    dy_mm=256.01 / 1024,
                    zx_mm=-239.98,
                ),
                target,
            ),
        ),
        (
            "sampling",
            lambda: validate_output_sampling(
                report,
                readback,
                source,
                _header(
                    dx_mm=0.635,
                    dy_mm=0.6340155,
                    zx_mm=-608.58,
                ),
            ),
        ),
        (
            "axis-sign distance",
            lambda: validate_native_transfer(
                replace(
                    report,
                    signed_distance_mm=ReportNumber.from_token("3.6860E+02"),
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
                _header(
                    dx_mm=0.634,
                    dy_mm=0.6340155,
                    zx_mm=-608.57,
                ),
            ),
        ),
        (
            "pilot",
            lambda: validate_native_transfer(
                report,
                segment,
                source,
                _header(
                    dx_mm=0.634,
                    dy_mm=0.6340155,
                    zx_mm=-608.58,
                    waist_mm=0.06,
                ),
            ),
        ),
        (
            "branch",
            lambda: validate_native_transfer(
                oi_report,
                segment,
                source,
                _header(
                    dx_mm=0.634,
                    dy_mm=0.6340155,
                    zx_mm=-608.58,
                    rayleigh_mm=700.0,
                ),
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
