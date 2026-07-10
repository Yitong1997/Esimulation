"""Native POP-report geometry parsing for the fixed biconic free-space segments."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np

from .models import Branch, SegmentSpec
from .zbf_binary import LosslessZbf


_FLOAT_TOKEN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
_BRANCH_FROM_STATES = {
    ("Outside", "Outside"): "OO",
    ("Outside", "Inside"): "OI",
    ("Inside", "Outside"): "IO",
    ("Inside", "Inside"): "II",
}
_BRANCH_STATES = {
    "OO": (False, False),
    "OI": (False, True),
    "IO": (True, False),
}


def _readonly_matrix(values: object, *, shape: tuple[int, int]) -> np.ndarray:
    matrix = np.array(values, dtype=np.float64, order="C", copy=True)
    if matrix.shape != shape or not np.all(np.isfinite(matrix)):
        raise ValueError(f"matrix must be finite with shape {shape}")
    immutable = np.frombuffer(matrix.tobytes(order="C"), dtype=matrix.dtype).reshape(
        matrix.shape
    )
    immutable.setflags(write=False)
    return immutable


@dataclass(frozen=True)
class ReportNumber:
    text: str
    value: float
    last_digit_resolution: float

    def __post_init__(self) -> None:
        text = str(self.text).strip()
        if not text or not math.isfinite(self.value):
            raise ValueError("report number must retain a finite printed token")
        try:
            parsed = float(text)
        except ValueError as error:
            raise ValueError("report number token is not numeric") from error
        if parsed != float(self.value):
            raise ValueError("report number value does not match its printed token")
        if (
            not math.isfinite(self.last_digit_resolution)
            or self.last_digit_resolution <= 0.0
        ):
            raise ValueError("report number must retain positive printed resolution")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(
            self, "last_digit_resolution", float(self.last_digit_resolution)
        )

    @classmethod
    def from_token(cls, token: str) -> "ReportNumber":
        text = token.strip()
        try:
            decimal_value = Decimal(text)
            value = float(decimal_value)
        except (InvalidOperation, ValueError, OverflowError) as error:
            raise ValueError(f"invalid report number token: {text!r}") from error
        if not math.isfinite(value):
            raise ValueError("report number must be finite")
        resolution = float(Decimal(1).scaleb(decimal_value.as_tuple().exponent))
        return cls(text=text, value=value, last_digit_resolution=resolution)

    def contains(self, candidate: float) -> bool:
        if not math.isfinite(candidate):
            return False
        scale = max(1.0, abs(self.value), abs(candidate))
        parse_allowance = 32.0 * np.finfo(np.float64).eps * scale
        return abs(candidate - self.value) <= (
            0.5 * self.last_digit_resolution + parse_allowance
        )


@dataclass(frozen=True)
class ChiefRaySample:
    surface: int
    x_mm: ReportNumber
    y_mm: ReportNumber
    z_mm: ReportNumber
    l: float
    m: float
    n: float

    def __post_init__(self) -> None:
        if not np.all(np.isfinite([self.l, self.m, self.n])):
            raise ValueError("chief-ray direction cosines must be finite")


@dataclass(frozen=True)
class NativeSurfaceTransfer:
    surface: int
    side_before: str
    side_after: str
    orientation_before: np.ndarray
    orientation_after: np.ndarray
    signed_distance_mm: ReportNumber | None
    starting_pilot_waist_x_mm: ReportNumber | None
    starting_pilot_waist_y_mm: ReportNumber | None
    starting_pilot_position_x_mm: ReportNumber | None
    starting_pilot_position_y_mm: ReportNumber | None
    starting_pilot_rayleigh_x_mm: ReportNumber | None
    starting_pilot_rayleigh_y_mm: ReportNumber | None
    start_inside_x: bool | None
    start_inside_y: bool | None
    propagator_branch: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "orientation_before",
            _readonly_matrix(self.orientation_before, shape=(3, 3)),
        )
        object.__setattr__(
            self,
            "orientation_after",
            _readonly_matrix(self.orientation_after, shape=(3, 3)),
        )


@dataclass(frozen=True)
class NativeIntermediateTrace:
    chief_rays: tuple[ChiefRaySample, ...]
    surface_transfers: tuple[NativeSurfaceTransfer, ...]

    def transfer_for_surface(self, surface: int) -> NativeSurfaceTransfer:
        matches = tuple(
            transfer
            for transfer in self.surface_transfers
            if transfer.surface == surface
        )
        if len(matches) != 1:
            raise ValueError(
                f"native report must contain exactly one surface-transfer block for S{surface}"
            )
        return matches[0]


@dataclass(frozen=True)
class SegmentGeometry:
    segment_key: str
    start_surface: int
    end_surface: int
    branch: Branch
    axis_sign: int
    model_distance_mm: float
    report_signed_distance_mm: ReportNumber
    raw_z_start_mm: float
    raw_z_end_mm: float
    raw_pilot_delta_mm: float
    start_inside: bool
    end_inside: bool
    transverse_basis_change: np.ndarray
    report_sha256: str

    def __post_init__(self) -> None:
        if not np.all(
            np.isfinite(
                [
                    self.model_distance_mm,
                    self.raw_z_start_mm,
                    self.raw_z_end_mm,
                    self.raw_pilot_delta_mm,
                ]
            )
        ):
            raise ValueError("segment distances must be finite")
        if self.model_distance_mm <= 0.0:
            raise ValueError("model propagation distance must be positive")
        object.__setattr__(
            self,
            "transverse_basis_change",
            _readonly_matrix(self.transverse_basis_change, shape=(2, 2)),
        )

    @property
    def propagation_distance_mm(self) -> float:
        """The only distance authorized for physical propagation."""

        return self.model_distance_mm


def _parse_matrix(section: str, label: str) -> np.ndarray:
    match = re.search(
        rf"{re.escape(label)}:\s*\r?\n([^\r\n]+)\r?\n([^\r\n]+)\r?\n([^\r\n]+)",
        section,
    )
    if match is None:
        raise ValueError(f"native surface-transfer block is missing {label}")
    rows = []
    for row_text in match.groups():
        tokens = row_text.split()
        if len(tokens) != 3:
            raise ValueError(f"{label} must contain exactly three columns")
        try:
            rows.append([float(token) for token in tokens])
        except ValueError as error:
            raise ValueError(f"{label} contains a nonnumeric entry") from error
    return np.asarray(rows, dtype=np.float64)


def _optional_number(section: str, pattern: str) -> ReportNumber | None:
    match = re.search(pattern, section, flags=re.MULTILINE)
    return None if match is None else ReportNumber.from_token(match.group(1))


def _optional_pair(
    section: str, pattern: str
) -> tuple[ReportNumber | None, ReportNumber | None]:
    match = re.search(pattern, section, flags=re.MULTILINE)
    if match is None:
        return None, None
    return ReportNumber.from_token(match.group(1)), ReportNumber.from_token(match.group(2))


def _parse_chief_rays(text: str, first_transfer_start: int) -> tuple[ChiefRaySample, ...]:
    marker = re.search(r"^Chief ray data:\s*$", text, flags=re.MULTILINE)
    if marker is None or marker.end() >= first_transfer_start:
        raise ValueError("native report is missing its chief-ray block")
    samples: list[ChiefRaySample] = []
    for line in text[marker.end() : first_transfer_start].splitlines():
        tokens = line.split()
        if len(tokens) != 7:
            continue
        try:
            surface = int(tokens[0])
        except ValueError:
            continue
        try:
            sample = ChiefRaySample(
                surface=surface,
                x_mm=ReportNumber.from_token(tokens[1]),
                y_mm=ReportNumber.from_token(tokens[2]),
                z_mm=ReportNumber.from_token(tokens[3]),
                l=float(tokens[4]),
                m=float(tokens[5]),
                n=float(tokens[6]),
            )
        except ValueError as error:
            raise ValueError(f"invalid chief-ray row for surface {surface}") from error
        samples.append(sample)
    if not samples:
        raise ValueError("native chief-ray block contains no samples")
    if len({sample.surface for sample in samples}) != len(samples):
        raise ValueError("native chief-ray block contains duplicate surfaces")
    return tuple(samples)


def parse_native_intermediate_trace(text: str) -> NativeIntermediateTrace:
    """Parse only the native chief-ray and surface-transfer evidence we use."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("native report text is required")
    header_pattern = re.compile(
        r"^Surface transfer from (before|after)\s+(\d+)\s+to\s+(before|after)\s+(\d+)\s*$",
        flags=re.MULTILINE,
    )
    headers = tuple(header_pattern.finditer(text))
    if not headers:
        raise ValueError("native report contains no surface-transfer blocks")
    chief_rays = _parse_chief_rays(text, headers[0].start())

    transfers: list[NativeSurfaceTransfer] = []
    for index, header in enumerate(headers):
        side_before, start_surface_text, side_after, end_surface_text = header.groups()
        start_surface = int(start_surface_text)
        end_surface = int(end_surface_text)
        if start_surface != end_surface:
            raise ValueError("native surface-transfer block changes surface number")
        section_end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        section = text[header.end() : section_end]
        orientation_before = _parse_matrix(section, "Orientation matrix before")
        orientation_after = _parse_matrix(section, "Orientation matrix after")
        signed_distance = _optional_number(
            section, rf"^Propagating beam distance:\s*({_FLOAT_TOKEN})\s*$"
        )
        waist_x, waist_y = _optional_pair(
            section,
            rf"^Starting pilot beam waist\s+x, y:\s*({_FLOAT_TOKEN})\s+({_FLOAT_TOKEN})\s*$",
        )
        position_x, position_y = _optional_pair(
            section,
            rf"^Starting pilot beam position x, y:\s*({_FLOAT_TOKEN})\s+({_FLOAT_TOKEN})\s*$",
        )
        rayleigh_x, rayleigh_y = _optional_pair(
            section,
            rf"^Starting pilot beam Rayleigh x, y:\s*({_FLOAT_TOKEN})\s+({_FLOAT_TOKEN})\s*$",
        )
        state_x_match = re.search(
            r"^X (Inside|Outside) Rayleigh range\.\s*$", section, flags=re.MULTILINE
        )
        state_y_match = re.search(
            r"^Y (Inside|Outside) Rayleigh range\.\s*$", section, flags=re.MULTILINE
        )
        if (state_x_match is None) != (state_y_match is None):
            raise ValueError("native report has incomplete X/Y Rayleigh state")
        start_inside_x = (
            None if state_x_match is None else state_x_match.group(1) == "Inside"
        )
        start_inside_y = (
            None if state_y_match is None else state_y_match.group(1) == "Inside"
        )
        branch_match = re.search(
            r"^Using (Inside|Outside) to (Inside|Outside) propagator\.\s*$",
            section,
            flags=re.MULTILINE,
        )
        propagator_branch = (
            None
            if branch_match is None
            else _BRANCH_FROM_STATES.get(branch_match.groups())
        )
        if branch_match is not None and propagator_branch is None:
            raise ValueError("native report contains unsupported propagator branch")
        if signed_distance is not None and (
            waist_x is None
            or waist_y is None
            or position_x is None
            or position_y is None
            or rayleigh_x is None
            or rayleigh_y is None
            or start_inside_x is None
            or start_inside_y is None
        ):
            raise ValueError("native propagation block is missing pilot or Rayleigh evidence")
        transfers.append(
            NativeSurfaceTransfer(
                surface=start_surface,
                side_before=side_before,
                side_after=side_after,
                orientation_before=orientation_before,
                orientation_after=orientation_after,
                signed_distance_mm=signed_distance,
                starting_pilot_waist_x_mm=waist_x,
                starting_pilot_waist_y_mm=waist_y,
                starting_pilot_position_x_mm=position_x,
                starting_pilot_position_y_mm=position_y,
                starting_pilot_rayleigh_x_mm=rayleigh_x,
                starting_pilot_rayleigh_y_mm=rayleigh_y,
                start_inside_x=start_inside_x,
                start_inside_y=start_inside_y,
                propagator_branch=propagator_branch,
            )
        )
    if len({transfer.surface for transfer in transfers}) != len(transfers):
        raise ValueError("native report contains duplicate surface-transfer blocks")
    return NativeIntermediateTrace(
        chief_rays=chief_rays, surface_transfers=tuple(transfers)
    )


def _transverse_basis_change(
    start_orientation: np.ndarray, end_orientation: np.ndarray
) -> np.ndarray:
    start_transverse = np.asarray(start_orientation, dtype=np.float64)[:, :2]
    end_transverse = np.asarray(end_orientation, dtype=np.float64)[:, :2]
    return end_transverse.T @ start_transverse


def validate_parallel_segment(
    geometry: SegmentGeometry, *, segment: SegmentSpec
) -> None:
    if (
        geometry.segment_key != segment.key
        or geometry.start_surface != segment.start_surface
        or geometry.end_surface != segment.end_surface
    ):
        raise ValueError("segment geometry identity does not match SegmentSpec")
    if geometry.branch != segment.branch:
        raise ValueError("native report branch does not match SegmentSpec")
    expected_states = _BRANCH_STATES[segment.branch]
    if (geometry.start_inside, geometry.end_inside) != expected_states:
        raise ValueError("native report Rayleigh states do not match SegmentSpec branch")
    if geometry.axis_sign != segment.start_convention.axis_sign:
        raise ValueError("segment geometry axis sign does not match the surface registry")
    if segment.start_convention.axis_sign != segment.end_convention.axis_sign:
        raise ValueError("parallel segment endpoints do not share a local axis sign")
    if segment.start_convention.side != "after" or segment.end_convention.side != "after":
        raise ValueError("biconic ZBF segment endpoints must be after-surface fields")
    if geometry.model_distance_mm != segment.model_distance_mm:
        raise ValueError("segment model distance differs from SegmentSpec")
    if not geometry.report_signed_distance_mm.contains(
        geometry.axis_sign * geometry.model_distance_mm
    ):
        raise ValueError("axis-signed report distance does not match model distance")
    if not geometry.report_signed_distance_mm.contains(geometry.raw_pilot_delta_mm):
        raise ValueError("report distance does not match the independently stored raw pilot delta")
    if not np.allclose(
        geometry.transverse_basis_change,
        np.eye(2),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("segment transverse basis change is not the identity")


def load_segment_geometry(
    report_path: str | Path,
    *,
    segment: SegmentSpec,
    start_beam: LosslessZbf,
    end_beam: LosslessZbf,
) -> SegmentGeometry:
    path = Path(report_path)
    raw_report = path.read_bytes()
    try:
        if raw_report.startswith((b"\xff\xfe", b"\xfe\xff")):
            report_text = raw_report.decode("utf-16")
        else:
            report_text = raw_report.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("native POP report must be UTF-8 or BOM-marked UTF-16 text") from error
    trace = parse_native_intermediate_trace(report_text)
    start_transfer = trace.transfer_for_surface(segment.start_surface)
    end_transfer = trace.transfer_for_surface(segment.end_surface)
    if start_transfer.side_before != "before" or start_transfer.side_after != "after":
        raise ValueError("native report does not prove the start after-surface side")
    if end_transfer.side_before != "before" or end_transfer.side_after != "after":
        raise ValueError("native report does not prove the end after-surface side")
    if start_transfer.signed_distance_mm is None:
        raise ValueError("native start block does not report a signed propagation distance")
    if start_transfer.propagator_branch is None:
        raise ValueError("native start block does not report a literal propagator branch")
    if (
        start_transfer.start_inside_x is None
        or start_transfer.start_inside_y is None
        or end_transfer.start_inside_x is None
        or end_transfer.start_inside_y is None
    ):
        raise ValueError("native report does not contain both endpoint Rayleigh states")
    if start_transfer.start_inside_x != start_transfer.start_inside_y:
        raise ValueError("native start X/Y Rayleigh states disagree")
    if end_transfer.start_inside_x != end_transfer.start_inside_y:
        raise ValueError("native end X/Y Rayleigh states disagree")
    if start_transfer.propagator_branch != segment.branch:
        raise ValueError("native literal propagator branch does not match SegmentSpec branch")

    raw_z_start = float(start_beam.header.zx)
    raw_z_end = float(end_beam.header.zx)
    if not np.all(np.isfinite([raw_z_start, raw_z_end])):
        raise ValueError("ZBF raw pilot positions must be finite")
    raw_delta = raw_z_end - raw_z_start
    basis_change = _transverse_basis_change(
        start_transfer.orientation_after, end_transfer.orientation_after
    )
    geometry = SegmentGeometry(
        segment_key=segment.key,
        start_surface=segment.start_surface,
        end_surface=segment.end_surface,
        branch=segment.branch,
        axis_sign=segment.start_convention.axis_sign,
        model_distance_mm=segment.model_distance_mm,
        report_signed_distance_mm=start_transfer.signed_distance_mm,
        raw_z_start_mm=raw_z_start,
        raw_z_end_mm=raw_z_end,
        raw_pilot_delta_mm=raw_delta,
        start_inside=bool(start_transfer.start_inside_x),
        end_inside=bool(end_transfer.start_inside_x),
        transverse_basis_change=basis_change,
        report_sha256=hashlib.sha256(raw_report).hexdigest(),
    )
    validate_parallel_segment(geometry, segment=segment)
    return geometry


__all__ = [
    "ChiefRaySample",
    "NativeIntermediateTrace",
    "NativeSurfaceTransfer",
    "ReportNumber",
    "SegmentGeometry",
    "load_segment_geometry",
    "parse_native_intermediate_trace",
    "validate_parallel_segment",
]
