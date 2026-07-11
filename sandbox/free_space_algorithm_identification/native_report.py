"""Compact, fail-closed parsing of one native POP transfer report."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, fields

from .geometry import ReportNumber
from .models import Branch, SegmentSpec
from .zbf_binary import RawZbfHeader


_FLOAT_TOKEN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
_LABEL_TO_BRANCH: dict[str, Branch] = {
    "Using Outside to Outside propagator.": "OO",
    "Using Outside to Inside propagator.": "OI",
    "Using Inside to Outside propagator.": "IO",
}
_BRANCH_STATES: dict[Branch, tuple[bool, bool]] = {
    "OO": (False, False),
    "OI": (False, True),
    "IO": (True, False),
}
_SAMPLE_SIZE_ENUM = re.compile(r"S_(\d+)x(\d+)")
_SURFACE_TRANSFER_MARKER = re.compile(
    r"^[ \t]*Surface transfer from (before|after)[ \t]+(\d+)"
    r"[ \t]+to[ \t]+(before|after)[ \t]+(\d+)[ \t]*$",
    flags=re.MULTILINE,
)


NumberPair = tuple[ReportNumber, ReportNumber]


def _require_positive_pair(values: NumberPair, *, label: str) -> None:
    if len(values) != 2 or any(value.value <= 0.0 for value in values):
        raise ValueError(f"{label} must contain two positive printed values")


@dataclass(frozen=True)
class PrintedSampling:
    interval_mm: NumberPair
    width_mm: NumberPair

    def __post_init__(self) -> None:
        _require_positive_pair(self.interval_mm, label="sampling interval")
        _require_positive_pair(self.width_mm, label="sampling width")


@dataclass(frozen=True)
class PrintedPilot:
    waist_mm: NumberPair
    position_mm: NumberPair
    rayleigh_mm: NumberPair
    inside: tuple[bool, bool]

    def __post_init__(self) -> None:
        _require_positive_pair(self.waist_mm, label="pilot waist")
        _require_positive_pair(self.rayleigh_mm, label="pilot Rayleigh distance")
        if len(self.position_mm) != 2:
            raise ValueError("pilot position must contain two printed values")
        if len(self.inside) != 2 or any(
            type(value) is not bool for value in self.inside
        ):
            raise ValueError("pilot state must contain two inside/outside flags")


@dataclass(frozen=True)
class NativePopReport:
    branch: Branch
    propagator_label: str
    signed_distance_mm: ReportNumber
    input_sampling: PrintedSampling
    output_sampling: PrintedSampling
    start_pilot: PrintedPilot
    end_pilot: PrintedPilot
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        expected_branch = _LABEL_TO_BRANCH.get(self.propagator_label)
        if expected_branch != self.branch:
            raise ValueError(
                "propagator branch does not match its literal native label"
            )
        if self.start_pilot.inside[0] != self.start_pilot.inside[1]:
            raise ValueError("propagator branch is ambiguous across start pilot axes")
        if self.end_pilot.inside[0] != self.end_pilot.inside[1]:
            raise ValueError("propagator branch is ambiguous across end pilot axes")
        states = (self.start_pilot.inside[0], self.end_pilot.inside[0])
        if states != _BRANCH_STATES[self.branch]:
            raise ValueError("propagator branch contradicts printed pilot states")
        warnings = tuple(str(warning).strip() for warning in self.warnings)
        if any(not warning.startswith("**** WARNING:") for warning in warnings):
            raise ValueError("native warnings must retain their literal warning lines")
        object.__setattr__(self, "warnings", warnings)

    @property
    def kernel_identity(self) -> None:
        """A native branch label carries no Fresnel/ASM/RS kernel evidence."""

        return None


@dataclass(frozen=True)
class _NativePopSettings:
    start_surface: int
    end_surface: int
    nx: int
    ny: int
    sample_size_enum: str
    x_width_mm: float
    y_width_mm: float
    wavelength_number: int
    wavelength_vacuum_mm: float
    refractive_index: float
    field_number: int
    use_polarization: bool
    normalization_mode: str
    normalization_value: float
    input_beam_file: str
    output_beam_file: str
    save_output_beam: bool
    save_beam_at_all_surfaces: bool

    def __post_init__(self) -> None:
        integer_fields = (
            ("start_surface", self.start_surface, 0),
            ("end_surface", self.end_surface, 0),
            ("nx", self.nx, 1),
            ("ny", self.ny, 1),
            ("wavelength_number", self.wavelength_number, 1),
            ("field_number", self.field_number, 1),
        )
        for name, value, minimum in integer_fields:
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.end_surface < self.start_surface:
            raise ValueError("end_surface must not precede start_surface")
        for name, value in (
            ("x_width_mm", self.x_width_mm),
            ("y_width_mm", self.y_width_mm),
            ("wavelength_vacuum_mm", self.wavelength_vacuum_mm),
            ("refractive_index", self.refractive_index),
            ("normalization_value", self.normalization_value),
        ):
            if (
                type(value) is not float
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be a positive finite float")
        for name, value in (
            ("sample_size_enum", self.sample_size_enum),
            ("normalization_mode", self.normalization_mode),
            ("input_beam_file", self.input_beam_file),
            ("output_beam_file", self.output_beam_file),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        sample_size_match = _SAMPLE_SIZE_ENUM.fullmatch(self.sample_size_enum)
        if sample_size_match is None:
            raise ValueError(
                "sample_size_enum must have the literal form S_<nx>x<ny>"
            )
        enum_grid = tuple(int(value) for value in sample_size_match.groups())
        if enum_grid != (self.nx, self.ny):
            raise ValueError(
                "sample_size_enum must match nx and ny exactly: "
                f"enum={enum_grid}, grid={(self.nx, self.ny)}"
            )
        for name, value in (
            ("use_polarization", self.use_polarization),
            ("save_output_beam", self.save_output_beam),
            ("save_beam_at_all_surfaces", self.save_beam_at_all_surfaces),
        ):
            if type(value) is not bool:
                raise ValueError(f"{name} must be boolean")


@dataclass(frozen=True)
class NativePopRequest(_NativePopSettings):
    """POP settings requested by the runner before opening the analysis."""


@dataclass(frozen=True)
class NativeSettingsReadback(_NativePopSettings):
    """Actual POP settings read back from the connected OpticStudio API."""


def _unique_match(text: str, pattern: str, *, label: str) -> re.Match[str]:
    matches = tuple(re.finditer(pattern, text, flags=re.MULTILINE))
    if len(matches) != 1:
        raise ValueError(f"native report must contain exactly one {label}")
    return matches[0]


def _two_pairs(
    text: str,
    label_pattern: str,
    *,
    label: str,
) -> tuple[tuple[NumberPair, int], tuple[NumberPair, int]]:
    pattern = (
        rf"^[ \t]*{label_pattern}:[ \t]*"
        rf"({_FLOAT_TOKEN})[ \t]+({_FLOAT_TOKEN})[ \t]*$"
    )
    matches = tuple(re.finditer(pattern, text, flags=re.MULTILINE))
    if len(matches) != 2:
        raise ValueError(f"native report must contain exactly two {label} lines")
    parsed = tuple(
        (
            (
                ReportNumber.from_token(match.group(1)),
                ReportNumber.from_token(match.group(2)),
            ),
            match.start(),
        )
        for match in matches
    )
    return parsed[0], parsed[1]


def _two_axis_states(
    text: str, axis: str
) -> tuple[tuple[bool, int], tuple[bool, int]]:
    pattern = rf"^[ \t]*{axis} (Inside|Outside) Rayleigh range\.[ \t]*$"
    matches = tuple(re.finditer(pattern, text, flags=re.MULTILINE))
    if len(matches) != 2:
        raise ValueError(
            f"native report must contain exactly two {axis}-axis pilot-state lines"
        )
    parsed = tuple((match.group(1) == "Inside", match.start()) for match in matches)
    return parsed[0], parsed[1]


def _parse_compact_transfer(text: str) -> NativePopReport:
    """Parse a compact text containing exactly one native transfer."""

    distance_match = _unique_match(
        text,
        rf"^[ \t]*Propagating beam distance:[ \t]*({_FLOAT_TOKEN})[ \t]*$",
        label="propagating-beam distance",
    )
    label_match = _unique_match(
        text,
        r"^[ \t]*(Using (?:Outside|Inside) to (?:Outside|Inside) propagator\.)[ \t]*$",
        label="propagator label",
    )
    propagator_label = label_match.group(1)
    try:
        branch = _LABEL_TO_BRANCH[propagator_label]
    except KeyError as error:
        raise ValueError(
            f"unsupported native propagator label: {propagator_label}"
        ) from error

    interval = _two_pairs(
        text,
        r"Starting delta X, Y size",
        label="starting sampling-interval",
    )
    width = _two_pairs(
        text,
        r"Starting array X, Y size",
        label="starting array-width",
    )
    waist = _two_pairs(
        text,
        r"Starting pilot beam waist[ \t]+x, y",
        label="starting pilot-waist",
    )
    position = _two_pairs(
        text,
        r"Starting pilot beam position[ \t]+x, y",
        label="starting pilot-position",
    )
    rayleigh = _two_pairs(
        text,
        r"Starting pilot beam Rayleigh[ \t]+x, y",
        label="starting pilot-Rayleigh",
    )
    x_state = _two_axis_states(text, "X")
    y_state = _two_axis_states(text, "Y")

    paired_groups = (interval, width, waist, position, rayleigh, x_state, y_state)
    if any(group[0][1] >= label_match.start() for group in paired_groups):
        raise ValueError("native report input block must precede the propagator label")
    if any(group[1][1] <= label_match.end() for group in paired_groups):
        raise ValueError("native report output block must follow the propagator label")
    if distance_match.start() >= label_match.start():
        raise ValueError("native report distance must precede the propagator label")

    warnings = tuple(
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("**** WARNING:")
    )
    return NativePopReport(
        branch=branch,
        propagator_label=propagator_label,
        signed_distance_mm=ReportNumber.from_token(distance_match.group(1)),
        input_sampling=PrintedSampling(interval[0][0], width[0][0]),
        output_sampling=PrintedSampling(interval[1][0], width[1][0]),
        start_pilot=PrintedPilot(
            waist[0][0],
            position[0][0],
            rayleigh[0][0],
            (x_state[0][0], y_state[0][0]),
        ),
        end_pilot=PrintedPilot(
            waist[1][0],
            position[1][0],
            rayleigh[1][0],
            (x_state[1][0], y_state[1][0]),
        ),
        warnings=warnings,
    )


def _selected_transfer_text(
    text: str,
    markers: tuple[re.Match[str], ...],
    *,
    start_surface: int | None,
    segment: SegmentSpec | None,
) -> str:
    if segment is not None and not isinstance(segment, SegmentSpec):
        raise ValueError("segment selector must be a SegmentSpec")
    if start_surface is not None and (
        type(start_surface) is not int or start_surface < 0
    ):
        raise ValueError("start_surface selector must be a nonnegative integer")

    if segment is None and start_surface is None:
        raise ValueError(
            "a full native report requires a start_surface or segment selector"
        )
    selected_surface = segment.start_surface if segment is not None else start_surface
    if start_surface is not None and segment is not None:
        if start_surface != segment.start_surface:
            raise ValueError("start_surface and segment selectors disagree")
    assert selected_surface is not None

    selected_indices = tuple(
        index
        for index, marker in enumerate(markers)
        if marker.group(1) == "before"
        and int(marker.group(2)) == selected_surface
        and marker.group(3) == "after"
        and int(marker.group(4)) == selected_surface
    )
    if len(selected_indices) != 1:
        raise ValueError(
            "full native report must contain a unique surface-transfer marker "
            f"for selected start surface {selected_surface}"
        )
    selected_index = selected_indices[0]
    if selected_index + 1 >= len(markers):
        raise ValueError("selected transfer has no adjacent output surface state")

    output_marker = markers[selected_index + 1]
    output_surface = int(output_marker.group(2))
    output_marker_is_state = (
        output_marker.group(1) == "before"
        and output_marker.group(3) == "after"
        and int(output_marker.group(4)) == output_surface
    )
    expected_output_surface = (
        segment.end_surface if segment is not None else output_surface
    )
    if not output_marker_is_state or output_surface != expected_output_surface:
        raise ValueError(
            "selected transfer has no adjacent output surface state matching "
            f"surface {expected_output_surface}"
        )

    current_section = text[markers[selected_index].end() : output_marker.start()]
    next_marker_start = (
        markers[selected_index + 2].start()
        if selected_index + 2 < len(markers)
        else len(text)
    )
    output_section = text[output_marker.end() : next_marker_start]
    output_patterns = (
        (
            rf"^[ \t]*Starting delta X, Y size:[ \t]*"
            rf"{_FLOAT_TOKEN}[ \t]+{_FLOAT_TOKEN}[ \t]*$",
            "output starting sampling-interval",
        ),
        (
            rf"^[ \t]*Starting array X, Y size:[ \t]*"
            rf"{_FLOAT_TOKEN}[ \t]+{_FLOAT_TOKEN}[ \t]*$",
            "output starting array-width",
        ),
        (
            rf"^[ \t]*Starting pilot beam waist[ \t]+x, y:[ \t]*"
            rf"{_FLOAT_TOKEN}[ \t]+{_FLOAT_TOKEN}[ \t]*$",
            "output starting pilot-waist",
        ),
        (
            rf"^[ \t]*Starting pilot beam position[ \t]+x, y:[ \t]*"
            rf"{_FLOAT_TOKEN}[ \t]+{_FLOAT_TOKEN}[ \t]*$",
            "output starting pilot-position",
        ),
        (
            rf"^[ \t]*Starting pilot beam Rayleigh[ \t]+x, y:[ \t]*"
            rf"{_FLOAT_TOKEN}[ \t]+{_FLOAT_TOKEN}[ \t]*$",
            "output starting pilot-Rayleigh",
        ),
        (
            r"^[ \t]*X (?:Inside|Outside) Rayleigh range\.[ \t]*$",
            "output X-axis pilot-state",
        ),
        (
            r"^[ \t]*Y (?:Inside|Outside) Rayleigh range\.[ \t]*$",
            "output Y-axis pilot-state",
        ),
    )
    output_lines = tuple(
        _unique_match(output_section, pattern, label=label).group(0)
        for pattern, label in output_patterns
    )
    return "\n".join((current_section, *output_lines))


def parse_native_pop_report(
    text: str,
    *,
    start_surface: int | None = None,
    segment: SegmentSpec | None = None,
) -> NativePopReport:
    """Parse one transfer from a compact or explicitly selected native report."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("native report text must be nonempty")
    markers = tuple(_SURFACE_TRANSFER_MARKER.finditer(text))
    if markers:
        text = _selected_transfer_text(
            text,
            markers,
            start_surface=start_surface,
            segment=segment,
        )
    elif start_surface is not None or segment is not None:
        raise ValueError(
            "a selector can only be used with a full native report containing "
            "surface-transfer markers"
        )
    return _parse_compact_transfer(text)


def _float_equal(left: float, right: float) -> bool:
    scale = max(1.0, abs(float(left)), abs(float(right)))
    return abs(float(left) - float(right)) <= 64.0 * math.ulp(scale)


def validate_settings_readback(
    request: NativePopRequest,
    readback: NativeSettingsReadback,
) -> None:
    """Require every critical requested POP setting to survive API readback."""

    for setting in fields(_NativePopSettings):
        name = setting.name
        requested = getattr(request, name)
        actual = getattr(readback, name)
        matches = (
            _float_equal(requested, actual)
            if type(requested) is float and type(actual) is float
            else requested == actual and type(requested) is type(actual)
        )
        if not matches:
            raise ValueError(
                f"settings readback mismatch for {name}: requested={requested!r}, "
                f"actual={actual!r}"
            )


def _require_header_grid(
    header: RawZbfHeader,
    readback: NativeSettingsReadback,
    *,
    role: str,
) -> None:
    if (header.nx, header.ny) != (readback.nx, readback.ny):
        raise ValueError(
            f"sampling mismatch for {role} ZBF count: "
            f"header={(header.nx, header.ny)}, readback={(readback.nx, readback.ny)}"
        )
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (header.dx, header.dy)
    ):
        raise ValueError(f"sampling mismatch: {role} ZBF intervals must be positive")
    if type(header.is_polarized) is not int or header.is_polarized not in (0, 1):
        raise ValueError(
            f"sampling contract mismatch for {role} ZBF polarization: "
            f"is_polarized must be exactly 0 or 1, got {header.is_polarized!r}"
        )
    if header.is_polarized != int(readback.use_polarization):
        raise ValueError(f"sampling contract mismatch for {role} ZBF polarization")
    if not _float_equal(
        header.wavelength_vacuum_mm, readback.wavelength_vacuum_mm
    ):
        raise ValueError(
            f"wavelength mismatch for {role} ZBF: "
            f"header={header.wavelength_vacuum_mm!r}, "
            f"readback={readback.wavelength_vacuum_mm!r}"
        )
    if not _float_equal(header.refractive_index, readback.refractive_index):
        raise ValueError(
            f"refractive index mismatch for {role} ZBF: "
            f"header={header.refractive_index!r}, "
            f"readback={readback.refractive_index!r}"
        )


def _require_printed_grid(
    printed: PrintedSampling,
    header: RawZbfHeader,
    *,
    role: str,
) -> None:
    candidates = (
        ("X interval", printed.interval_mm[0], header.dx),
        ("Y interval", printed.interval_mm[1], header.dy),
        ("X width", printed.width_mm[0], header.nx * header.dx),
        ("Y width", printed.width_mm[1], header.ny * header.dy),
    )
    for name, report_number, candidate in candidates:
        if not report_number.contains(candidate):
            raise ValueError(
                f"sampling mismatch for {role} {name}: "
                f"report={report_number.text}, ZBF={candidate!r}"
            )


def validate_output_sampling(
    report: NativePopReport,
    readback: NativeSettingsReadback,
    source_header: RawZbfHeader,
    output_header: RawZbfHeader,
) -> None:
    """Close the API-readback, printed-grid, and binary-ZBF sampling loop."""

    _require_header_grid(source_header, readback, role="source")
    _require_header_grid(output_header, readback, role="output")
    source_widths = (
        source_header.nx * source_header.dx,
        source_header.ny * source_header.dy,
    )
    for axis, actual, expected in zip(
        ("x", "y"),
        source_widths,
        (readback.x_width_mm, readback.y_width_mm),
        strict=True,
    ):
        if not _float_equal(actual, expected):
            raise ValueError(
                f"sampling mismatch for source {axis}_width_mm: "
                f"ZBF={actual!r}, readback={expected!r}"
            )
    _require_printed_grid(report.input_sampling, source_header, role="input")
    _require_printed_grid(report.output_sampling, output_header, role="output")


def _require_report_pair(
    printed: NumberPair,
    actual: tuple[float, float],
    *,
    label: str,
) -> None:
    for axis, report_number, candidate in zip("xy", printed, actual, strict=True):
        if not report_number.contains(candidate):
            raise ValueError(
                f"pilot mismatch for {label} {axis}: "
                f"report={report_number.text}, ZBF={candidate!r}"
            )


def _inside(position_mm: float, rayleigh_mm: float) -> bool:
    return abs(position_mm) < rayleigh_mm


def _require_pilot(
    pilot: PrintedPilot,
    header: RawZbfHeader,
    *,
    role: str,
) -> None:
    _require_report_pair(pilot.waist_mm, (header.wx, header.wy), label=f"{role} waist")
    _require_report_pair(
        pilot.position_mm,
        (header.zx, header.zy),
        label=f"{role} position",
    )
    _require_report_pair(
        pilot.rayleigh_mm,
        (header.rx, header.ry),
        label=f"{role} Rayleigh",
    )
    actual_inside = (
        _inside(header.zx, header.rx),
        _inside(header.zy, header.ry),
    )
    if pilot.inside != actual_inside:
        raise ValueError(
            f"pilot mismatch for {role} inside/outside state: "
            f"report={pilot.inside}, ZBF={actual_inside}"
        )


def validate_native_transfer(
    report: NativePopReport,
    segment: SegmentSpec,
    source_header: RawZbfHeader,
    output_header: RawZbfHeader,
) -> None:
    """Validate branch, signed distance, and pilot state without naming a kernel."""

    if report.branch != segment.branch:
        raise ValueError(
            f"branch mismatch: report={report.branch}, segment={segment.branch}"
        )
    if segment.start_convention.surface != segment.start_surface:
        raise ValueError("axis-sign distance contract has the wrong start convention")
    if segment.end_convention.surface != segment.end_surface:
        raise ValueError("axis-sign distance contract has the wrong end convention")
    axis_sign = segment.start_convention.axis_sign
    if segment.end_convention.axis_sign != axis_sign:
        raise ValueError("axis-sign distance contract changes sign within the segment")
    expected_report_distance = axis_sign * segment.model_distance_mm
    if not report.signed_distance_mm.contains(expected_report_distance):
        raise ValueError(
            "axis-sign distance mismatch: "
            f"report={report.signed_distance_mm.text}, "
            f"expected={expected_report_distance!r}"
        )

    for axis, raw_delta in (
        ("x", output_header.zx - source_header.zx),
        ("y", output_header.zy - source_header.zy),
    ):
        if not report.signed_distance_mm.contains(raw_delta):
            raise ValueError(
                f"axis-sign distance mismatch on ZBF {axis}: "
                f"report={report.signed_distance_mm.text}, raw_delta={raw_delta!r}"
            )

    _require_pilot(report.start_pilot, source_header, role="start")
    _require_pilot(report.end_pilot, output_header, role="end")
    header_states = (
        _inside(source_header.zx, source_header.rx),
        _inside(output_header.zx, output_header.rx),
    )
    if header_states != _BRANCH_STATES[report.branch]:
        raise ValueError(
            f"branch mismatch with ZBF pilot states: branch={report.branch}, "
            f"states={header_states}"
        )


__all__ = [
    "NativePopReport",
    "NativePopRequest",
    "NativeSettingsReadback",
    "PrintedPilot",
    "PrintedSampling",
    "parse_native_pop_report",
    "validate_native_transfer",
    "validate_output_sampling",
    "validate_settings_readback",
]
