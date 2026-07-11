"""Physical Start=End identity and entrance-only complex calibration.

Only paired-ZBF physical total fields enter this module.  The sole fitted
degree of freedom is one entrance scalar.  Sample-value semantics are frozen
first from lossless power evidence; no spatial registration or endpoint fit is
implemented.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal

import numpy as np

from .artifacts import ArtifactHash, ArtifactRef, RunLayout, verify_artifact_ref
from .biconic_case import BICONIC_SEGMENTS
from .field_contract import MappedZbfField, reference_phases
from .geometry import ReportNumber
from .models import PointField2D, SampleValueConvention, UniformGrid2D
from .native_report import (
    NativePopRequest,
    NativeSettingsReadback,
    parse_native_pop_report,
    validate_native_transfer,
    validate_output_sampling,
    validate_settings_readback,
)
from .zbf_binary import HEADER_BYTES, RawZbfHeader


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_RUN_INSTANCE_RE = re.compile(r"[0-9a-f]{32}\Z")
_FLOAT_TOKEN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
_PILOT_ROUNDTRIP_RELATIVE_TOLERANCE = 1.0e-8
_PILOT_ROUNDTRIP_ABSOLUTE_TOLERANCE_MM = 1.0e-12
_PILOT_RAYLEIGH_RELATION_RELATIVE_TOLERANCE = 1.0e-8
_PILOT_RAYLEIGH_RELATION_ABSOLUTE_TOLERANCE_MM = 1.0e-12
_NEIGHBOURS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _logical_identifier(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{label} must be a nonempty logical identifier")
    return value.strip()


def _positive_float(value: object, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be a positive finite float")
    return value


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(values: np.ndarray, *, schema: str) -> str:
    array = np.ascontiguousarray(values)
    header = json.dumps(
        {
            "dtype": array.dtype.str,
            "schema": schema,
            "shape": list(array.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\x00" + array.tobytes(order="C")).hexdigest()


def physical_grid_sha256(grid: UniformGrid2D) -> str:
    """Bind full physical axes, including the sample-at-zero origin."""

    if not isinstance(grid, UniformGrid2D):
        raise ValueError("grid hash requires a UniformGrid2D")
    return _canonical_sha256(
        {
            "schema": "bts.identity_grid/v1",
            "nx": grid.nx,
            "ny": grid.ny,
            "x_sha256": _array_sha256(grid.x_mm, schema="axis_x_mm/v1"),
            "y_sha256": _array_sha256(grid.y_mm, schema="axis_y_mm/v1"),
        }
    )


def _float_equal(left: float, right: float) -> bool:
    scale = max(1.0, abs(float(left)), abs(float(right)))
    return abs(float(left) - float(right)) <= 64.0 * math.ulp(scale)


def _pilot_roundtrip_close(left_mm: float, right_mm: float) -> bool:
    return bool(
        np.isclose(
            left_mm,
            right_mm,
            rtol=_PILOT_ROUNDTRIP_RELATIVE_TOLERANCE,
            atol=_PILOT_ROUNDTRIP_ABSOLUTE_TOLERANCE_MM,
        )
    )


def _validate_pilot_rayleigh_relation(
    *,
    wavelength_vacuum_mm: float,
    refractive_index: float,
    rayleigh_mm: float,
    waist_mm: float,
    label: str,
) -> None:
    expected_rayleigh_mm = (
        np.pi * refractive_index * waist_mm**2 / wavelength_vacuum_mm
    )
    if not np.isclose(
        rayleigh_mm,
        expected_rayleigh_mm,
        rtol=_PILOT_RAYLEIGH_RELATION_RELATIVE_TOLERANCE,
        atol=_PILOT_RAYLEIGH_RELATION_ABSOLUTE_TOLERANCE_MM,
    ):
        raise ValueError(f"{label} pilot violates the Rayleigh relation")


@dataclass(frozen=True)
class IdentityNativeReport:
    """The one native Start=End line that defines physical power observables."""

    peak_irradiance_w_per_mm2: ReportNumber
    total_power_w: ReportNumber

    def __post_init__(self) -> None:
        if not isinstance(self.peak_irradiance_w_per_mm2, ReportNumber) or not isinstance(
            self.total_power_w, ReportNumber
        ):
            raise ValueError("identity summary requires two printed report numbers")
        if (
            self.peak_irradiance_w_per_mm2.value <= 0.0
            or self.total_power_w.value <= 0.0
        ):
            raise ValueError("identity summary observables must be positive")

    @classmethod
    def parse(cls, text: str) -> "IdentityNativeReport":
        if not isinstance(text, str) or not text.strip():
            raise ValueError("identity native report text must be nonempty")
        pattern = re.compile(
            rf"Peak Irradiance = ({_FLOAT_TOKEN}) Watts/Millimeters\^2, "
            rf"Total Power = ({_FLOAT_TOKEN}) Watts"
        )
        candidates = tuple(
            line.strip()
            for line in text.splitlines()
            if line.strip().startswith("Peak Irradiance =")
        )
        matches = tuple(pattern.fullmatch(line) for line in candidates)
        exact = tuple(match for match in matches if match is not None)
        if len(candidates) != 1 or len(exact) != 1:
            raise ValueError(
                "identity native report must contain exactly one exact physical-summary line"
            )
        match = exact[0]
        return cls(
            peak_irradiance_w_per_mm2=ReportNumber.from_token(match.group(1)),
            total_power_w=ReportNumber.from_token(match.group(2)),
        )

    @property
    def summary_sha256(self) -> str:
        return _canonical_sha256(
            {
                "peak_irradiance_token": self.peak_irradiance_w_per_mm2.text,
                "total_power_token": self.total_power_w.text,
            }
        )


@dataclass(frozen=True)
class SampleConventionPolicy:
    """Predeclared closure and separation limits in printed-resolution units."""

    maximum_closure_sigma: float = 3.0
    minimum_separation_sigma: float = 5.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            object.__setattr__(self, name, _positive_float(value, label=name))
        if self.minimum_separation_sigma <= self.maximum_closure_sigma:
            raise ValueError("sample-convention separation must exceed closure")

    @property
    def policy_sha256(self) -> str:
        return _canonical_sha256(
            {name: value.hex() for name, value in asdict(self).items()}
        )


def _synthetic_artifact(
    *,
    run_id: str,
    run_instance_uuid: str,
    case_id: str,
    role: str,
    stage: str,
) -> ArtifactRef:
    digest = _canonical_sha256(
        {
            "case_id": case_id,
            "role": role,
            "run_id": run_id,
            "run_instance_uuid": run_instance_uuid,
            "stage": stage,
        }
    )
    suffix = "ZBF" if role in {"source", "output"} else "dat"
    return ArtifactRef(
        run_id=run_id,
        run_instance_uuid=run_instance_uuid,
        producer_stage=stage,
        producer_case=case_id,
        relative_path=f"S07_S08/{case_id}/{stage}/{role}.{suffix}",
        byte_count=1,
        sha256=digest,
    )


@dataclass(frozen=True, init=False)
class SampleConventionProbe:
    """One current-run S7 Start=End report/ZBF dimensional test."""

    segment_key: str
    case_id: str
    repeat_id: str
    surface: int
    source_artifact: ArtifactRef
    output_artifact: ArtifactRef
    effective_cfg_artifact: ArtifactRef
    settings_artifact: ArtifactRef
    native_report_artifact: ArtifactRef
    header_sha256: str
    nx: int
    ny: int
    dx_mm: float
    dy_mm: float
    x_width_mm: float
    y_width_mm: float
    wavelength_number: int
    wavelength_vacuum_mm: float
    refractive_index: float
    field_number: int
    use_polarization: bool
    normalization_mode: str
    normalization_value: float
    pilot_zx_mm: float
    pilot_rx_mm: float
    pilot_wx_mm: float
    pilot_zy_mm: float
    pilot_ry_mm: float
    pilot_wy_mm: float
    report: IdentityNativeReport
    raw_energy: float
    raw_peak: float
    authoritative: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError(
            "SampleConventionProbe is created only from a current-run identity capture"
        )

    @classmethod
    def _create(cls, **values: object) -> "SampleConventionProbe":
        result = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(result, name, value)
        result.__post_init__()
        return result

    @classmethod
    def _synthetic(
        cls,
        *,
        case_id: str,
        repeat_id: str,
        nx: int,
        ny: int,
        dx_mm: float,
        dy_mm: float,
        raw_energy: float,
        raw_peak: float,
        report: IdentityNativeReport,
        run_id: str = "synthetic",
        run_instance_uuid: str = "a" * 32,
    ) -> "SampleConventionProbe":
        synthetic_waist_mm = 0.15
        synthetic_rayleigh_mm = (
            np.pi * 1.0 * synthetic_waist_mm**2 / 0.01064
        )
        refs = {
            role: _synthetic_artifact(
                run_id=run_id,
                run_instance_uuid=run_instance_uuid,
                case_id=case_id,
                role=role,
                stage="fixed_input" if role == "source" else "identity",
            )
            for role in ("source", "output", "effective_cfg", "settings", "report")
        }
        return cls._create(
            segment_key="S07_S08",
            case_id=case_id,
            repeat_id=repeat_id,
            surface=7,
            source_artifact=refs["source"],
            output_artifact=refs["output"],
            effective_cfg_artifact=refs["effective_cfg"],
            settings_artifact=refs["settings"],
            native_report_artifact=refs["report"],
            header_sha256=_canonical_sha256(
                {"case_id": case_id, "nx": nx, "synthetic_header": True}
            ),
            nx=nx,
            ny=ny,
            dx_mm=float(dx_mm),
            dy_mm=float(dy_mm),
            x_width_mm=float(nx * dx_mm),
            y_width_mm=float(ny * dy_mm),
            wavelength_number=1,
            wavelength_vacuum_mm=0.01064,
            refractive_index=1.0,
            field_number=1,
            use_polarization=False,
            normalization_mode="total_power",
            normalization_value=1.0,
            pilot_zx_mm=12.0,
            pilot_rx_mm=synthetic_rayleigh_mm,
            pilot_wx_mm=synthetic_waist_mm,
            pilot_zy_mm=12.0,
            pilot_ry_mm=synthetic_rayleigh_mm,
            pilot_wy_mm=synthetic_waist_mm,
            report=report,
            raw_energy=float(raw_energy),
            raw_peak=float(raw_peak),
            authoritative=False,
        )

    @classmethod
    def from_identity_capture(
        cls,
        layout: RunLayout,
        capture: object,
    ) -> "SampleConventionProbe":
        """Create one authoritative probe from verified S7 Start=End artifacts."""

        (
            segment,
            readback,
            _,
            header,
            summary,
            rewrite_path,
        ) = _load_identity_capture(layout, capture)
        if segment.key != "S07_S08" or readback.start_surface != 7:
            raise ValueError("sample convention is identified only at physical S7")
        if (readback.nx, readback.ny) not in {
            (1024, 1024),
            (2048, 2048),
            (4096, 4096),
        }:
            raise ValueError("sample convention requires N=1024/2048/4096")
        raw_energy, raw_peak = _raw_observables(rewrite_path, header)
        return cls._create(
            segment_key=segment.key,
            case_id=capture.case_id,
            repeat_id=capture.repeat_id,
            surface=readback.start_surface,
            source_artifact=capture.source_input,
            output_artifact=capture.output_zbfs[0].artifact,
            effective_cfg_artifact=capture.effective_cfg_artifact,
            settings_artifact=capture.settings_artifact,
            native_report_artifact=capture.report_artifact,
            header_sha256=hashlib.sha256(header.raw_bytes).hexdigest(),
            nx=header.nx,
            ny=header.ny,
            dx_mm=float(header.dx),
            dy_mm=float(header.dy),
            x_width_mm=float(readback.x_width_mm),
            y_width_mm=float(readback.y_width_mm),
            wavelength_number=readback.wavelength_number,
            wavelength_vacuum_mm=float(header.wavelength_vacuum_mm),
            refractive_index=float(header.refractive_index),
            field_number=readback.field_number,
            use_polarization=readback.use_polarization,
            normalization_mode=readback.normalization_mode,
            normalization_value=float(readback.normalization_value),
            pilot_zx_mm=float(header.zx),
            pilot_rx_mm=float(header.rx),
            pilot_wx_mm=float(header.wx),
            pilot_zy_mm=float(header.zy),
            pilot_ry_mm=float(header.ry),
            pilot_wy_mm=float(header.wy),
            report=summary,
            raw_energy=raw_energy,
            raw_peak=raw_peak,
            authoritative=True,
        )

    def __post_init__(self) -> None:
        for name in ("segment_key", "case_id", "repeat_id"):
            object.__setattr__(
                self, name, _logical_identifier(getattr(self, name), label=name)
            )
        if self.segment_key != "S07_S08" or self.surface != 7:
            raise ValueError("sample convention requires an S7 Start=End probe")
        if type(self.authoritative) is not bool:
            raise ValueError("sample-convention authority flag must be Boolean")
        refs = (
            self.source_artifact,
            self.output_artifact,
            self.effective_cfg_artifact,
            self.settings_artifact,
            self.native_report_artifact,
        )
        _refs_share_run(refs)
        if self.source_artifact.relative_path == self.output_artifact.relative_path:
            raise ValueError("sample-convention input and output must be distinct")
        for ref in refs[1:]:
            if ref.producer_stage != "identity" or ref.producer_case != self.case_id:
                raise ValueError("sample-convention evidence has the wrong producer")
        _require_sha256(self.header_sha256, label="sample-convention header")
        if (self.nx, self.ny) not in {
            (1024, 1024),
            (2048, 2048),
            (4096, 4096),
        }:
            raise ValueError("sample convention requires N=1024/2048/4096 square grids")
        for name in (
            "dx_mm",
            "dy_mm",
            "x_width_mm",
            "y_width_mm",
            "wavelength_vacuum_mm",
            "refractive_index",
            "normalization_value",
            "pilot_rx_mm",
            "pilot_wx_mm",
            "pilot_ry_mm",
            "pilot_wy_mm",
            "raw_energy",
            "raw_peak",
        ):
            object.__setattr__(
                self, name, _positive_float(getattr(self, name), label=name)
            )
        for name in ("pilot_zx_mm", "pilot_zy_mm"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite float")
        _validate_pilot_rayleigh_relation(
            wavelength_vacuum_mm=self.wavelength_vacuum_mm,
            refractive_index=self.refractive_index,
            rayleigh_mm=self.pilot_rx_mm,
            waist_mm=self.pilot_wx_mm,
            label="sample-convention X",
        )
        _validate_pilot_rayleigh_relation(
            wavelength_vacuum_mm=self.wavelength_vacuum_mm,
            refractive_index=self.refractive_index,
            rayleigh_mm=self.pilot_ry_mm,
            waist_mm=self.pilot_wy_mm,
            label="sample-convention Y",
        )
        for name in ("wavelength_number", "field_number"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.use_polarization is not False:
            raise ValueError("sample-convention formula requires a scalar ZBF")
        if not isinstance(self.normalization_mode, str) or not self.normalization_mode:
            raise ValueError("sample convention requires a normalization mode")
        if not isinstance(self.report, IdentityNativeReport):
            raise ValueError("sample convention requires a parsed identity report")
        if not _float_equal(self.x_width_mm, self.nx * self.dx_mm) or not _float_equal(
            self.y_width_mm, self.ny * self.dy_mm
        ):
            raise ValueError("sample-convention width does not match N times interval")
        if self.raw_peak > self.raw_energy * (1.0 + 64.0 * np.finfo(float).eps):
            raise ValueError("sample-convention raw peak exceeds its raw sum")

    @property
    def run_id(self) -> str:
        return self.output_artifact.run_id

    @property
    def run_instance_uuid(self) -> str:
        return self.output_artifact.run_instance_uuid

    @property
    def pixel_area_mm2(self) -> float:
        return self.dx_mm * self.dy_mm

    def predictions(self, hypothesis: SampleValueConvention) -> tuple[float, float]:
        if hypothesis == "cell_energy":
            return self.raw_energy, self.raw_peak / self.pixel_area_mm2
        if hypothesis == "point_value":
            return self.raw_energy * self.pixel_area_mm2, self.raw_peak
        raise ValueError("unknown sample-value hypothesis")

    def scores(self, hypothesis: SampleValueConvention) -> tuple[float, float]:
        total, peak = self.predictions(hypothesis)
        return (
            abs(total - self.report.total_power_w.value)
            / self.report.total_power_w.last_digit_resolution,
            abs(peak - self.report.peak_irradiance_w_per_mm2.value)
            / self.report.peak_irradiance_w_per_mm2.last_digit_resolution,
        )

    @property
    def evidence_sha256(self) -> str:
        float_names = (
            "dx_mm",
            "dy_mm",
            "x_width_mm",
            "y_width_mm",
            "wavelength_vacuum_mm",
            "refractive_index",
            "normalization_value",
            "pilot_zx_mm",
            "pilot_rx_mm",
            "pilot_wx_mm",
            "pilot_zy_mm",
            "pilot_ry_mm",
            "pilot_wy_mm",
            "raw_energy",
            "raw_peak",
        )
        return _canonical_sha256(
            {
                "artifacts": [
                    ref.to_dict()
                    for ref in (
                        self.source_artifact,
                        self.output_artifact,
                        self.effective_cfg_artifact,
                        self.settings_artifact,
                        self.native_report_artifact,
                    )
                ],
                "authoritative": self.authoritative,
                "case_id": self.case_id,
                "field_number": self.field_number,
                "floats": {name: getattr(self, name).hex() for name in float_names},
                "header_sha256": self.header_sha256,
                "normalization_mode": self.normalization_mode,
                "nx": self.nx,
                "ny": self.ny,
                "repeat_id": self.repeat_id,
                "report_sha256": self.report.summary_sha256,
                "segment_key": self.segment_key,
                "surface": self.surface,
                "use_polarization": self.use_polarization,
                "wavelength_number": self.wavelength_number,
            }
        )


@dataclass(frozen=True, init=False)
class SampleConventionResult:
    """Immutable evaluation of the three S7 report/ZBF dimensional probes."""

    probes: tuple[SampleConventionProbe, ...]
    policy: SampleConventionPolicy
    authoritative: bool = field(init=False)
    status: Literal["point_value", "cell_energy", "undecided"] = field(init=False)
    point_max_closure_sigma: float = field(init=False)
    cell_max_closure_sigma: float = field(init=False)
    point_min_separation_sigma: float = field(init=False)
    cell_min_separation_sigma: float = field(init=False)
    result_sha256: str = field(init=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError(
            "SampleConventionResult is created only by the fixed classification rule"
        )

    @classmethod
    def _create(
        cls,
        probes: tuple[SampleConventionProbe, ...],
        *,
        policy: SampleConventionPolicy,
        require_authoritative: bool,
    ) -> "SampleConventionResult":
        if require_authoritative and policy != SampleConventionPolicy():
            raise ValueError("formal sample convention has a fixed 3u/5u policy")
        if require_authoritative and not all(
            isinstance(probe, SampleConventionProbe) and probe.authoritative
            for probe in probes
        ):
            raise ValueError(
                "formal sample convention requires authoritative current-run probes"
            )
        result = object.__new__(cls)
        object.__setattr__(result, "probes", tuple(probes))
        object.__setattr__(result, "policy", policy)
        result.__post_init__()
        return result

    @classmethod
    def _synthetic(
        cls,
        probes: tuple[SampleConventionProbe, ...],
        *,
        policy: SampleConventionPolicy = SampleConventionPolicy(),
    ) -> "SampleConventionResult":
        if any(probe.authoritative for probe in probes):
            raise ValueError("synthetic classification cannot consume authoritative probes")
        return cls._create(
            tuple(probes), policy=policy, require_authoritative=False
        )

    def __post_init__(self) -> None:
        probes = tuple(self.probes)
        if len(probes) != 3 or not all(
            isinstance(probe, SampleConventionProbe) for probe in probes
        ):
            raise ValueError("sample convention requires exactly three S7 identity probes")
        if not isinstance(self.policy, SampleConventionPolicy):
            raise ValueError("sample convention requires a policy")
        authorities = {probe.authoritative for probe in probes}
        if len(authorities) != 1:
            raise ValueError("authoritative and synthetic probes cannot be mixed")
        if {probe.nx for probe in probes} != {1024, 2048, 4096}:
            raise ValueError("sample convention requires N=1024/2048/4096")
        if len({(probe.run_id, probe.run_instance_uuid) for probe in probes}) != 1:
            raise ValueError("sample-convention probes must share one current run")
        if len({probe.case_id for probe in probes}) != 3:
            raise ValueError("sample-convention probes must use distinct sampling cases")
        for role in (
            "source_artifact",
            "output_artifact",
            "settings_artifact",
            "native_report_artifact",
        ):
            refs = [getattr(probe, role).to_dict() for probe in probes]
            if len({_canonical_sha256(ref) for ref in refs}) != 3:
                raise ValueError(f"sample-convention {role} evidence must be distinct")
        first = probes[0]
        common_exact = (
            "surface",
            "wavelength_number",
            "field_number",
            "use_polarization",
            "normalization_mode",
        )
        if any(
            getattr(probe, name) != getattr(first, name)
            for probe in probes[1:]
            for name in common_exact
        ):
            raise ValueError("sample-convention probes change fixed physical settings")
        common_float = (
            "x_width_mm",
            "y_width_mm",
            "wavelength_vacuum_mm",
            "refractive_index",
            "normalization_value",
        )
        if any(
            not _float_equal(getattr(probe, name), getattr(first, name))
            for probe in probes[1:]
            for name in common_float
        ):
            raise ValueError(
                "sample-convention probes must preserve the S7 window, medium, "
                "and normalization"
            )
        common_pilot_mm = (
            "pilot_zx_mm",
            "pilot_rx_mm",
            "pilot_wx_mm",
            "pilot_zy_mm",
            "pilot_ry_mm",
            "pilot_wy_mm",
        )
        if any(
            not _pilot_roundtrip_close(
                getattr(probe, name), getattr(first, name)
            )
            for probe in probes[1:]
            for name in common_pilot_mm
        ):
            raise ValueError(
                "sample-convention pilot roundtrip exceeds the fixed tolerance"
            )

        point_scores = [
            score for probe in probes for score in probe.scores("point_value")
        ]
        cell_scores = [
            score for probe in probes for score in probe.scores("cell_energy")
        ]
        point_max = max(point_scores)
        cell_max = max(cell_scores)
        point_min = min(point_scores)
        cell_min = min(cell_scores)
        point_supported = (
            point_max <= self.policy.maximum_closure_sigma
            and cell_min > self.policy.minimum_separation_sigma
        )
        cell_supported = (
            cell_max <= self.policy.maximum_closure_sigma
            and point_min > self.policy.minimum_separation_sigma
        )
        status: Literal["point_value", "cell_energy", "undecided"]
        if point_supported == cell_supported:
            status = "undecided"
        else:
            status = "point_value" if point_supported else "cell_energy"
        authoritative = next(iter(authorities))
        object.__setattr__(self, "probes", probes)
        object.__setattr__(self, "authoritative", authoritative)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "point_max_closure_sigma", float(point_max))
        object.__setattr__(self, "cell_max_closure_sigma", float(cell_max))
        object.__setattr__(self, "point_min_separation_sigma", float(point_min))
        object.__setattr__(self, "cell_min_separation_sigma", float(cell_min))
        object.__setattr__(
            self,
            "result_sha256",
            _canonical_sha256(
                {
                    "authoritative": authoritative,
                    "cell_scores_hex": [value.hex() for value in cell_scores],
                    "point_scores_hex": [value.hex() for value in point_scores],
                    "policy_sha256": self.policy.policy_sha256,
                    "probe_sha256": [probe.evidence_sha256 for probe in probes],
                    "status": status,
                }
            ),
        )


def classify_sample_value_convention(
    probes: tuple[SampleConventionProbe, ...],
) -> SampleConventionResult:
    return SampleConventionResult._create(
        tuple(probes),
        policy=SampleConventionPolicy(),
        require_authoritative=True,
    )


@dataclass(frozen=True)
class IdentityPolicy:
    roi_threshold: float = 1e-6
    maximum_phase_rms_waves: float = 1e-6
    maximum_intensity_relative_l2_percent: float = 1e-4
    grid_rtol: float = 1e-10
    grid_atol_mm: float = 1e-12

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            object.__setattr__(self, name, _positive_float(value, label=name))
        if self.roi_threshold > 1.0:
            raise ValueError("roi_threshold cannot exceed one")

    @property
    def policy_sha256(self) -> str:
        return _canonical_sha256(
            {name: value.hex() for name, value in asdict(self).items()}
        )


def entrance_settings_sha256(
    readback: NativeSettingsReadback,
    *,
    input_zbf_sha256: str,
) -> str:
    """Hash entrance physics while excluding end/output routing."""

    if not isinstance(readback, NativeSettingsReadback):
        raise ValueError("entrance settings require a NativeSettingsReadback")
    input_hash = _require_sha256(input_zbf_sha256, label="entrance input ZBF")
    return _canonical_sha256(
        {
            "field_number": readback.field_number,
            "input_zbf_sha256": input_hash,
            "normalization_mode": readback.normalization_mode,
            "normalization_value_hex": readback.normalization_value.hex(),
            "nx": readback.nx,
            "ny": readback.ny,
            "refractive_index_hex": readback.refractive_index.hex(),
            "sample_size_enum": readback.sample_size_enum,
            "start_surface": readback.start_surface,
            "use_polarization": readback.use_polarization,
            "wavelength_number": readback.wavelength_number,
            "wavelength_vacuum_mm_hex": readback.wavelength_vacuum_mm.hex(),
            "x_width_mm_hex": readback.x_width_mm.hex(),
            "y_width_mm_hex": readback.y_width_mm.hex(),
        }
    )


def _refs_share_run(refs: tuple[ArtifactRef, ...]) -> tuple[str, str]:
    if not refs or not all(isinstance(ref, ArtifactRef) for ref in refs):
        raise ValueError("native evidence requires ArtifactRef objects")
    identities = {(ref.run_id, ref.run_instance_uuid) for ref in refs}
    if len(identities) != 1:
        raise ValueError("native evidence artifacts must share one run instance")
    return next(iter(identities))


def _read_native_text(path: Path) -> str:
    payload = Path(path).read_bytes()
    if not payload:
        raise ValueError("native report artifact is empty")
    if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
        return payload.decode("utf-16")
    if payload.startswith(b"\xef\xbb\xbf"):
        return payload.decode("utf-8-sig")
    if b"\x00" in payload:
        return payload.decode("utf-16-le")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("cp1252")


def _read_settings_readback(path: Path) -> NativeSettingsReadback:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("native settings artifact is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("native settings artifact must contain one JSON object")
    names = tuple(item.name for item in fields(NativeSettingsReadback))
    missing = tuple(name for name in names if name not in payload)
    if missing:
        raise ValueError(f"native settings artifact is missing fields: {missing}")
    try:
        return NativeSettingsReadback(**{name: payload[name] for name in names})
    except (TypeError, ValueError) as error:
        raise ValueError("native settings artifact violates the POP contract") from error


def _segment_for_capture(segment_key: str):
    matches = tuple(item for item in BICONIC_SEGMENTS if item.key == segment_key)
    if len(matches) != 1:
        raise ValueError("capture does not identify one fixed biconic segment")
    return matches[0]


def _verify_capture_artifacts(
    layout: RunLayout,
    capture: object,
    *,
    expected_stage: Literal["identity", "propagation"],
) -> tuple[Path, tuple[Path, ...], Path, Path, Path]:
    from .zos_runner import CapturedPopRun, CapturedZbf

    if not isinstance(layout, RunLayout) or not isinstance(capture, CapturedPopRun):
        raise ValueError("native binding requires a captured current-run POP")
    if capture.stage != expected_stage:
        raise ValueError(f"native binding requires a {expected_stage} capture")
    if not isinstance(capture.output_zbfs, tuple) or not all(
        isinstance(item, CapturedZbf) for item in capture.output_zbfs
    ):
        raise ValueError("native capture has no typed output-ZBF evidence")
    if not isinstance(capture.staged_input_hash, ArtifactHash):
        raise ValueError("native capture has no typed staged-input hash")
    produced = (
        capture.effective_cfg_artifact,
        capture.settings_artifact,
        capture.report_artifact,
        *(item.artifact for item in capture.output_zbfs),
    )
    refs = (capture.source_input, *produced)
    run_identity = _refs_share_run(refs)
    if run_identity != (layout.run_id, layout.run_instance_uuid):
        raise ValueError("capture artifacts do not belong to the supplied run layout")
    for ref in produced:
        if ref.producer_stage != expected_stage or ref.producer_case != capture.case_id:
            raise ValueError("captured native output has the wrong producer")
    source_path = verify_artifact_ref(
        layout,
        capture.source_input,
        expected_producer_stage=capture.source_input.producer_stage,
        expected_producer_case=capture.source_input.producer_case,
    )
    output_paths = tuple(
        verify_artifact_ref(
            layout,
            item.artifact,
            expected_producer_stage=expected_stage,
            expected_producer_case=capture.case_id,
        )
        for item in capture.output_zbfs
    )
    cfg_path = verify_artifact_ref(
        layout,
        capture.effective_cfg_artifact,
        expected_producer_stage=expected_stage,
        expected_producer_case=capture.case_id,
    )
    settings_path = verify_artifact_ref(
        layout,
        capture.settings_artifact,
        expected_producer_stage=expected_stage,
        expected_producer_case=capture.case_id,
    )
    report_path = verify_artifact_ref(
        layout,
        capture.report_artifact,
        expected_producer_stage=expected_stage,
        expected_producer_case=capture.case_id,
    )
    staged_hash = capture.staged_input_hash
    if (
        staged_hash.sha256 != capture.source_input.sha256
        or staged_hash.byte_count != capture.source_input.byte_count
    ):
        raise ValueError("staged input hash does not match the bound source artifact")
    return source_path, output_paths, cfg_path, settings_path, report_path


def _validate_captured_readback(
    capture: object,
    artifact_readback: NativeSettingsReadback,
    *,
    source_header: RawZbfHeader,
    output_name: str,
    start_surface: int,
    end_surface: int,
) -> None:
    captured_readback = capture.settings_readback
    if not isinstance(captured_readback, NativeSettingsReadback):
        raise ValueError("capture has no typed native settings readback")
    validate_settings_readback(
        NativePopRequest(**asdict(captured_readback)), artifact_readback
    )
    if not output_name.endswith(".ZBF"):
        raise ValueError("captured base output must retain its .ZBF name")
    expected = NativePopRequest(
        start_surface=start_surface,
        end_surface=end_surface,
        nx=source_header.nx,
        ny=source_header.ny,
        sample_size_enum=f"S_{source_header.nx}x{source_header.ny}",
        x_width_mm=float(source_header.nx * source_header.dx),
        y_width_mm=float(source_header.ny * source_header.dy),
        wavelength_number=artifact_readback.wavelength_number,
        wavelength_vacuum_mm=float(source_header.wavelength_vacuum_mm),
        refractive_index=float(source_header.refractive_index),
        field_number=artifact_readback.field_number,
        use_polarization=bool(source_header.is_polarized),
        normalization_mode=artifact_readback.normalization_mode,
        normalization_value=artifact_readback.normalization_value,
        input_beam_file=capture.staged_input_name,
        output_beam_file=output_name[:-4],
        save_output_beam=True,
        save_beam_at_all_surfaces=True,
    )
    validate_settings_readback(expected, artifact_readback)


def _require_header_readback(
    header: RawZbfHeader,
    readback: NativeSettingsReadback,
    *,
    role: str,
) -> None:
    if (header.nx, header.ny) != (readback.nx, readback.ny):
        raise ValueError(f"{role} ZBF count does not match identity settings")
    if header.units != 0:
        raise ValueError(f"{role} ZBF must use millimetre header units")
    if header.is_polarized not in (0, 1) or bool(header.is_polarized) != readback.use_polarization:
        raise ValueError(f"{role} ZBF polarization does not match identity settings")
    if not _float_equal(
        header.wavelength_vacuum_mm, readback.wavelength_vacuum_mm
    ) or not _float_equal(header.refractive_index, readback.refractive_index):
        raise ValueError(f"{role} ZBF wavelength or medium does not match settings")


def _pilot_tuple(header: RawZbfHeader) -> tuple[float, ...]:
    return (header.zx, header.rx, header.wx, header.zy, header.ry, header.wy)


def _validate_header_pilot(header: RawZbfHeader, *, label: str) -> None:
    for axis, position_mm, rayleigh_mm, waist_mm in (
        ("X", header.zx, header.rx, header.wx),
        ("Y", header.zy, header.ry, header.wy),
    ):
        if (
            not math.isfinite(position_mm)
            or not math.isfinite(rayleigh_mm)
            or rayleigh_mm <= 0.0
            or not math.isfinite(waist_mm)
            or waist_mm <= 0.0
        ):
            raise ValueError(f"{label} {axis} pilot header is not physical")
        _validate_pilot_rayleigh_relation(
            wavelength_vacuum_mm=header.wavelength_vacuum_mm,
            refractive_index=header.refractive_index,
            rayleigh_mm=rayleigh_mm,
            waist_mm=waist_mm,
            label=f"{label} {axis}",
        )


def _read_zbf_header(path: Path) -> RawZbfHeader:
    with Path(path).open("rb") as stream:
        raw = stream.read(HEADER_BYTES)
    return RawZbfHeader.from_bytes(raw)


def _load_identity_capture(
    layout: RunLayout,
    capture: object,
) -> tuple[
    object,
    NativeSettingsReadback,
    RawZbfHeader,
    RawZbfHeader,
    IdentityNativeReport,
    Path,
]:
    from .zos_runner import expected_output_names

    source_path, output_paths, _, settings_path, report_path = _verify_capture_artifacts(
        layout, capture, expected_stage="identity"
    )
    segment = _segment_for_capture(capture.segment_key)
    if len(capture.output_zbfs) != 2:
        raise ValueError("identity capture must contain base and numbered ZBFs")
    readback = _read_settings_readback(settings_path)
    expected_names = expected_output_names(
        readback.output_beam_file, segment.start_surface, segment.start_surface
    )
    actual_names = tuple(item.name for item in capture.output_zbfs)
    if actual_names != expected_names:
        raise ValueError("identity capture does not contain the exact anchored outputs")
    source_header = _read_zbf_header(source_path)
    rewrite_header = _read_zbf_header(output_paths[0])
    if rewrite_header != capture.output_zbfs[0].header:
        raise ValueError("identity ZBF bytes do not match captured header/hash evidence")
    if (
        readback.start_surface != segment.start_surface
        or readback.end_surface != segment.start_surface
    ):
        raise ValueError("identity capture does not prove the fixed Start=End side")
    _validate_captured_readback(
        capture,
        readback,
        source_header=source_header,
        output_name=capture.output_zbfs[0].name,
        start_surface=segment.start_surface,
        end_surface=segment.start_surface,
    )
    _require_header_readback(source_header, readback, role="source")
    _require_header_readback(rewrite_header, readback, role="rewrite")
    for axis, actual, expected in (
        ("source x", source_header.nx * source_header.dx, readback.x_width_mm),
        ("source y", source_header.ny * source_header.dy, readback.y_width_mm),
        ("rewrite x", rewrite_header.nx * rewrite_header.dx, readback.x_width_mm),
        ("rewrite y", rewrite_header.ny * rewrite_header.dy, readback.y_width_mm),
    ):
        if not _float_equal(actual, expected):
            raise ValueError(f"identity {axis} window does not match settings")
    _validate_header_pilot(source_header, label="identity source")
    _validate_header_pilot(rewrite_header, label="identity rewrite")
    if any(
        not _pilot_roundtrip_close(left, right)
        for left, right in zip(
            _pilot_tuple(source_header), _pilot_tuple(rewrite_header), strict=True
        )
    ):
        raise ValueError("identity pilot roundtrip exceeds the fixed tolerance")
    if rewrite_header.is_polarized or source_header.is_polarized:
        raise ValueError("identity physical-summary contract currently requires scalar ZBFs")
    summary = IdentityNativeReport.parse(_read_native_text(report_path))
    return (
        segment,
        readback,
        source_header,
        rewrite_header,
        summary,
        output_paths[0],
    )


def _raw_observables(path: Path, header: RawZbfHeader) -> tuple[float, float]:
    if header.is_polarized:
        raise ValueError("identity power observables require a scalar ZBF")
    count = header.nx * header.ny
    required_bytes = HEADER_BYTES + count * np.dtype("<c16").itemsize
    if Path(path).stat().st_size < required_bytes:
        raise ValueError("identity ZBF payload is incomplete")
    samples = np.memmap(
        path,
        dtype="<c16",
        mode="r",
        offset=HEADER_BYTES,
        shape=(count,),
    )
    total = 0.0
    peak = 0.0
    block_size = 1 << 20
    for start in range(0, count, block_size):
        intensity = np.abs(samples[start : start + block_size]) ** 2
        if not np.all(np.isfinite(intensity)):
            raise ValueError("identity ZBF contains non-finite payload intensity")
        total += float(np.sum(intensity, dtype=np.float64))
        peak = max(peak, float(np.max(intensity)))
    del samples
    if not math.isfinite(total) or total <= 0.0 or not math.isfinite(peak) or peak <= 0.0:
        raise ValueError("identity ZBF has no positive finite physical observables")
    return total, peak


def _identity_observable_scores(
    path: Path,
    header: RawZbfHeader,
    summary: IdentityNativeReport,
    hypothesis: SampleValueConvention,
) -> tuple[float, float]:
    raw_energy, raw_peak = _raw_observables(path, header)
    pixel_area = float(header.dx * header.dy)
    if hypothesis == "cell_energy":
        total, peak = raw_energy, raw_peak / pixel_area
    elif hypothesis == "point_value":
        total, peak = raw_energy * pixel_area, raw_peak
    else:
        raise ValueError("unknown sample-value hypothesis")
    return (
        abs(total - summary.total_power_w.value)
        / summary.total_power_w.last_digit_resolution,
        abs(peak - summary.peak_irradiance_w_per_mm2.value)
        / summary.peak_irradiance_w_per_mm2.last_digit_resolution,
    )


@dataclass(frozen=True, init=False)
class IdentityBinding:
    """Current-run proof of one independent Start=End rewrite."""

    segment_key: str
    case_id: str
    identity_repeat_id: str
    surface: int
    side: Literal["after"]
    input_artifact: ArtifactRef
    rewrite_artifact: ArtifactRef
    effective_cfg_artifact: ArtifactRef
    settings_artifact: ArtifactRef
    native_report_artifact: ArtifactRef
    entrance_settings_sha256: str
    convention_evidence_sha256: str
    sample_convention: SampleConventionResult
    input_grid_sha256: str
    rewrite_grid_sha256: str
    nx: int
    ny: int
    dx_mm: float
    dy_mm: float
    wavelength_vacuum_mm: float
    refractive_index: float

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("IdentityBinding is created only by from_capture")

    @classmethod
    def _create(cls, **values: object) -> "IdentityBinding":
        result = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(result, name, value)
        result.__post_init__()
        return result

    @classmethod
    def _synthetic(cls, **values: object) -> "IdentityBinding":
        """Test-only construction; production callers must use ``from_capture``."""

        return cls._create(**values)

    def __post_init__(self) -> None:
        for name in ("segment_key", "case_id", "identity_repeat_id"):
            object.__setattr__(
                self,
                name,
                _logical_identifier(getattr(self, name), label=name),
            )
        if type(self.surface) is not int or self.surface <= 0:
            raise ValueError("identity surface must be a positive integer")
        if self.side != "after":
            raise ValueError("identity binding must prove the after-surface side")
        refs = (
            self.input_artifact,
            self.rewrite_artifact,
            self.effective_cfg_artifact,
            self.settings_artifact,
            self.native_report_artifact,
        )
        _refs_share_run(refs)
        if self.input_artifact.relative_path == self.rewrite_artifact.relative_path:
            raise ValueError("identity input and rewrite artifacts must be distinct")
        for ref in refs[1:]:
            if ref.producer_stage != "identity" or ref.producer_case != self.case_id:
                raise ValueError("identity output evidence has the wrong producer")
        if self.input_artifact.producer_case != self.case_id:
            raise ValueError("identity input artifact has the wrong case")
        for name in (
            "entrance_settings_sha256",
            "convention_evidence_sha256",
            "input_grid_sha256",
            "rewrite_grid_sha256",
        ):
            object.__setattr__(
                self, name, _require_sha256(getattr(self, name), label=name)
            )
        if not isinstance(self.sample_convention, SampleConventionResult):
            raise ValueError("identity binding requires sample-convention evidence")
        if self.sample_convention.status == "undecided":
            raise ValueError("undecided sample convention blocks identity")
        if type(self.nx) is not int or type(self.ny) is not int or min(self.nx, self.ny) < 2:
            raise ValueError("identity sampling counts must be integers >= 2")
        for name in (
            "dx_mm",
            "dy_mm",
            "wavelength_vacuum_mm",
            "refractive_index",
        ):
            object.__setattr__(self, name, _positive_float(getattr(self, name), label=name))

    @property
    def sample_value_convention(self) -> SampleValueConvention:
        if self.sample_convention.status == "undecided":
            raise RuntimeError("undecided convention cannot enter identity")
        return self.sample_convention.status

    @property
    def binding_sha256(self) -> str:
        return _canonical_sha256(
            {
                "artifacts": [
                    ref.to_dict()
                    for ref in (
                        self.input_artifact,
                        self.rewrite_artifact,
                        self.effective_cfg_artifact,
                        self.settings_artifact,
                        self.native_report_artifact,
                    )
                ],
                "case_id": self.case_id,
                "convention_evidence_sha256": self.convention_evidence_sha256,
                "dx_mm_hex": self.dx_mm.hex(),
                "dy_mm_hex": self.dy_mm.hex(),
                "entrance_settings_sha256": self.entrance_settings_sha256,
                "input_grid_sha256": self.input_grid_sha256,
                "identity_repeat_id": self.identity_repeat_id,
                "nx": self.nx,
                "ny": self.ny,
                "refractive_index_hex": self.refractive_index.hex(),
                "rewrite_grid_sha256": self.rewrite_grid_sha256,
                "sample_convention_sha256": self.sample_convention.result_sha256,
                "segment_key": self.segment_key,
                "side": self.side,
                "surface": self.surface,
                "wavelength_vacuum_mm_hex": self.wavelength_vacuum_mm.hex(),
            }
        )

    @classmethod
    def from_capture(
        cls,
        layout: RunLayout,
        capture: object,
        *,
        convention_evidence_sha256: str,
        sample_convention: SampleConventionResult,
    ) -> "IdentityBinding":
        if not isinstance(sample_convention, SampleConventionResult):
            raise ValueError("identity binding requires sample-convention evidence")
        if not sample_convention.authoritative:
            raise ValueError("identity binding requires an authoritative convention result")
        if sample_convention.status == "undecided":
            raise ValueError("undecided sample convention blocks identity")
        (
            segment,
            settings,
            input_header,
            header,
            summary,
            rewrite_path,
        ) = _load_identity_capture(layout, capture)
        capture_run = (
            capture.source_input.run_id,
            capture.source_input.run_instance_uuid,
        )
        convention_run = (
            sample_convention.probes[0].run_id,
            sample_convention.probes[0].run_instance_uuid,
        )
        if capture_run != convention_run:
            raise ValueError("identity and convention evidence must share one current run")
        if max(
            _identity_observable_scores(
                rewrite_path, header, summary, sample_convention.status
            )
        ) > sample_convention.policy.maximum_closure_sigma:
            raise ValueError(
                "identity report and rewrite ZBF violate the decided sample convention"
            )
        input_grid = UniformGrid2D.centered(
            nx=input_header.nx,
            ny=input_header.ny,
            dx_mm=input_header.dx,
            dy_mm=input_header.dy,
        )
        rewrite_grid = UniformGrid2D.centered(
            nx=header.nx,
            ny=header.ny,
            dx_mm=header.dx,
            dy_mm=header.dy,
        )
        rewrite = capture.output_zbfs[0]
        return cls._create(
            segment_key=capture.segment_key,
            case_id=capture.case_id,
            identity_repeat_id=capture.repeat_id,
            surface=settings.start_surface,
            side="after",
            input_artifact=capture.source_input,
            rewrite_artifact=rewrite.artifact,
            effective_cfg_artifact=capture.effective_cfg_artifact,
            settings_artifact=capture.settings_artifact,
            native_report_artifact=capture.report_artifact,
            entrance_settings_sha256=entrance_settings_sha256(
                settings,
                input_zbf_sha256=capture.source_input.sha256,
            ),
            convention_evidence_sha256=convention_evidence_sha256,
            sample_convention=sample_convention,
            input_grid_sha256=physical_grid_sha256(input_grid),
            rewrite_grid_sha256=physical_grid_sha256(rewrite_grid),
            nx=input_header.nx,
            ny=input_header.ny,
            dx_mm=float(input_header.dx),
            dy_mm=float(input_header.dy),
            wavelength_vacuum_mm=float(input_header.wavelength_vacuum_mm),
            refractive_index=float(input_header.refractive_index),
        )


@dataclass(frozen=True, init=False)
class EntranceCalibration:
    c_entry: complex
    binding_sha256: str
    policy_sha256: str
    roi_mask_sha256: str
    sample_convention_sha256: str
    input_zbf_sha256: str
    rewrite_zbf_sha256: str
    segment_key: str
    case_id: str
    convention_evidence_sha256: str
    magnitude: float = field(init=False)
    phase_rad: float = field(init=False)
    calibration_sha256: str = field(init=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("EntranceCalibration is created only by identity evaluation")

    @classmethod
    def _create(
        cls,
        *,
        c_entry: complex,
        binding_sha256: str,
        policy_sha256: str,
        roi_mask_sha256: str,
        sample_convention_sha256: str,
        input_zbf_sha256: str,
        rewrite_zbf_sha256: str,
        segment_key: str,
        case_id: str,
        convention_evidence_sha256: str,
    ) -> "EntranceCalibration":
        result = object.__new__(cls)
        for name, value in {
            "c_entry": c_entry,
            "binding_sha256": binding_sha256,
            "policy_sha256": policy_sha256,
            "roi_mask_sha256": roi_mask_sha256,
            "sample_convention_sha256": sample_convention_sha256,
            "input_zbf_sha256": input_zbf_sha256,
            "rewrite_zbf_sha256": rewrite_zbf_sha256,
            "segment_key": segment_key,
            "case_id": case_id,
            "convention_evidence_sha256": convention_evidence_sha256,
        }.items():
            object.__setattr__(result, name, value)
        result.__post_init__()
        return result

    def __post_init__(self) -> None:
        scalar = complex(self.c_entry)
        if not np.isfinite(scalar.real) or not np.isfinite(scalar.imag):
            raise ValueError("entrance calibration scalar must be finite")
        object.__setattr__(self, "c_entry", scalar)
        for name in (
            "binding_sha256",
            "policy_sha256",
            "roi_mask_sha256",
            "sample_convention_sha256",
            "input_zbf_sha256",
            "rewrite_zbf_sha256",
            "convention_evidence_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        object.__setattr__(self, "segment_key", _logical_identifier(self.segment_key, label="segment_key"))
        object.__setattr__(self, "case_id", _logical_identifier(self.case_id, label="case_id"))
        magnitude = float(abs(scalar))
        phase = float(np.angle(scalar))
        object.__setattr__(self, "magnitude", magnitude)
        object.__setattr__(self, "phase_rad", phase)
        object.__setattr__(
            self,
            "calibration_sha256",
            _canonical_sha256(
                {
                    "binding_sha256": self.binding_sha256,
                    "c_entry_imag_hex": scalar.imag.hex(),
                    "c_entry_real_hex": scalar.real.hex(),
                    "case_id": self.case_id,
                    "convention_evidence_sha256": self.convention_evidence_sha256,
                    "input_zbf_sha256": self.input_zbf_sha256,
                    "policy_sha256": self.policy_sha256,
                    "rewrite_zbf_sha256": self.rewrite_zbf_sha256,
                    "roi_mask_sha256": self.roi_mask_sha256,
                    "sample_convention_sha256": self.sample_convention_sha256,
                    "segment_key": self.segment_key,
                }
            ),
        )


def _readonly_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2 or array.dtype != np.bool_:
        raise ValueError("identity ROI must be a two-dimensional Boolean mask")
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=np.bool_).reshape(array.shape)
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True, init=False)
class IdentityResult:
    binding: IdentityBinding
    policy: IdentityPolicy
    calibration: EntranceCalibration
    roi_mask: np.ndarray
    phase_rms_waves: float | None
    intensity_relative_l2_percent: float
    passed: bool = field(init=False)
    failure_reason: str | None = field(init=False)
    result_sha256: str = field(init=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("IdentityResult is created only by identity evaluation")

    @classmethod
    def _create(
        cls,
        *,
        binding: IdentityBinding,
        policy: IdentityPolicy,
        calibration: EntranceCalibration,
        roi_mask: np.ndarray,
        phase_rms_waves: float | None,
        intensity_relative_l2_percent: float,
    ) -> "IdentityResult":
        result = object.__new__(cls)
        for name, value in {
            "binding": binding,
            "policy": policy,
            "calibration": calibration,
            "roi_mask": roi_mask,
            "phase_rms_waves": phase_rms_waves,
            "intensity_relative_l2_percent": intensity_relative_l2_percent,
        }.items():
            object.__setattr__(result, name, value)
        result.__post_init__()
        return result

    def __post_init__(self) -> None:
        if not isinstance(self.binding, IdentityBinding) or not isinstance(self.policy, IdentityPolicy):
            raise ValueError("identity result requires binding and policy")
        if not isinstance(self.calibration, EntranceCalibration):
            raise ValueError("identity result requires an EntranceCalibration")
        if self.calibration.binding_sha256 != self.binding.binding_sha256:
            raise ValueError("identity calibration does not match its binding")
        if self.calibration.policy_sha256 != self.policy.policy_sha256:
            raise ValueError("identity calibration does not match its policy")
        roi = _readonly_mask(self.roi_mask)
        if roi.shape != (self.binding.ny, self.binding.nx) or not np.any(roi):
            raise ValueError("identity ROI does not match the bound sampling")
        roi_hash = _array_sha256(roi, schema="identity_roi_bool/v1")
        if roi_hash != self.calibration.roi_mask_sha256:
            raise ValueError("identity ROI does not match its calibration")
        if self.phase_rms_waves is not None and (
            not math.isfinite(self.phase_rms_waves) or self.phase_rms_waves < 0.0
        ):
            raise ValueError("identity phase RMS must be finite and nonnegative")
        if not math.isfinite(self.intensity_relative_l2_percent) or self.intensity_relative_l2_percent < 0.0:
            raise ValueError("identity intensity residual must be finite and nonnegative")
        if self.phase_rms_waves is None:
            passed = False
            reason = "identity_phase_undefined"
        else:
            passed = bool(
                self.phase_rms_waves <= self.policy.maximum_phase_rms_waves
                and self.intensity_relative_l2_percent
                <= self.policy.maximum_intensity_relative_l2_percent
            )
            reason = None if passed else "identity_residual_exceeds_policy"
        object.__setattr__(self, "roi_mask", roi)
        object.__setattr__(self, "passed", passed)
        object.__setattr__(self, "failure_reason", reason)
        object.__setattr__(
            self,
            "result_sha256",
            _canonical_sha256(
                {
                    "binding_sha256": self.binding.binding_sha256,
                    "calibration_sha256": self.calibration.calibration_sha256,
                    "failure_reason": reason,
                    "intensity_relative_l2_percent_hex": self.intensity_relative_l2_percent.hex(),
                    "passed": passed,
                    "phase_rms_waves_hex": None if self.phase_rms_waves is None else self.phase_rms_waves.hex(),
                    "policy_sha256": self.policy.policy_sha256,
                    "roi_mask_sha256": roi_hash,
                }
            ),
        )

    @property
    def roi_threshold(self) -> float:
        return self.policy.roi_threshold

    @property
    def roi_sample_count(self) -> int:
        return int(np.count_nonzero(self.roi_mask))

    @property
    def roi_mask_sha256(self) -> str:
        return self.calibration.roi_mask_sha256


def _load_propagation_capture(
    layout: RunLayout,
    capture: object,
) -> tuple[object, NativeSettingsReadback, RawZbfHeader, RawZbfHeader]:
    from .zos_runner import expected_output_names

    source_path, output_paths, _, settings_path, report_path = _verify_capture_artifacts(
        layout, capture, expected_stage="propagation"
    )
    segment = _segment_for_capture(capture.segment_key)
    if len(capture.output_zbfs) != 3:
        raise ValueError("propagation binding requires an exact adjacent transfer")
    readback = _read_settings_readback(settings_path)
    expected_names = expected_output_names(
        readback.output_beam_file, segment.start_surface, segment.end_surface
    )
    actual_names = tuple(item.name for item in capture.output_zbfs)
    if actual_names != expected_names:
        raise ValueError("propagation capture does not contain the exact anchored outputs")
    source_header = _read_zbf_header(source_path)
    endpoint_header = _read_zbf_header(output_paths[0])
    if endpoint_header != capture.output_zbfs[0].header:
        raise ValueError("propagation ZBF bytes do not match captured header/hash evidence")
    if (
        readback.start_surface != segment.start_surface
        or readback.end_surface != segment.end_surface
    ):
        raise ValueError("propagation readback is not the exact adjacent segment")
    _validate_captured_readback(
        capture,
        readback,
        source_header=source_header,
        output_name=capture.output_zbfs[0].name,
        start_surface=segment.start_surface,
        end_surface=segment.end_surface,
    )
    report = parse_native_pop_report(
        _read_native_text(report_path), segment=segment
    )
    validate_output_sampling(report, readback, source_header, endpoint_header)
    validate_native_transfer(report, segment, source_header, endpoint_header)
    return segment, readback, source_header, endpoint_header


@dataclass(frozen=True, init=False)
class PropagationBinding:
    """Current-run proof of the exact endpoint eligible for calibration."""

    segment_key: str
    case_id: str
    repeat_id: str
    source_input: ArtifactRef
    endpoint_artifact: ArtifactRef
    effective_cfg_artifact: ArtifactRef
    settings_artifact: ArtifactRef
    native_report_artifact: ArtifactRef
    entrance_settings_sha256: str
    convention_evidence_sha256: str
    sample_convention: SampleConventionResult
    wavelength_vacuum_mm: float
    refractive_index: float

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("PropagationBinding is created only by from_capture")

    @classmethod
    def _create(cls, **values: object) -> "PropagationBinding":
        result = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(result, name, value)
        result.__post_init__()
        return result

    @classmethod
    def _synthetic(cls, **values: object) -> "PropagationBinding":
        """Test-only construction; production callers must use ``from_capture``."""

        return cls._create(**values)

    def __post_init__(self) -> None:
        for name in ("segment_key", "case_id", "repeat_id"):
            object.__setattr__(self, name, _logical_identifier(getattr(self, name), label=name))
        refs = (
            self.source_input,
            self.endpoint_artifact,
            self.effective_cfg_artifact,
            self.settings_artifact,
            self.native_report_artifact,
        )
        _refs_share_run(refs)
        for ref in refs[1:]:
            if ref.producer_stage != "propagation" or ref.producer_case != self.case_id:
                raise ValueError("propagation endpoint evidence has the wrong producer")
        if self.source_input.producer_case != self.case_id:
            raise ValueError("propagation input has the wrong case")
        for name in ("entrance_settings_sha256", "convention_evidence_sha256"):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), label=name))
        if not isinstance(self.sample_convention, SampleConventionResult) or self.sample_convention.status == "undecided":
            raise ValueError("propagation requires decided sample-convention evidence")
        for name in ("wavelength_vacuum_mm", "refractive_index"):
            object.__setattr__(self, name, _positive_float(getattr(self, name), label=name))

    @property
    def binding_sha256(self) -> str:
        return _canonical_sha256(
            {
                "artifacts": [
                    ref.to_dict()
                    for ref in (
                        self.source_input,
                        self.endpoint_artifact,
                        self.effective_cfg_artifact,
                        self.settings_artifact,
                        self.native_report_artifact,
                    )
                ],
                "case_id": self.case_id,
                "convention_evidence_sha256": self.convention_evidence_sha256,
                "entrance_settings_sha256": self.entrance_settings_sha256,
                "refractive_index_hex": self.refractive_index.hex(),
                "repeat_id": self.repeat_id,
                "sample_convention_sha256": self.sample_convention.result_sha256,
                "segment_key": self.segment_key,
                "wavelength_vacuum_mm_hex": self.wavelength_vacuum_mm.hex(),
            }
        )

    @classmethod
    def from_capture(
        cls,
        layout: RunLayout,
        capture: object,
        *,
        convention_evidence_sha256: str,
        sample_convention: SampleConventionResult,
    ) -> "PropagationBinding":
        if not isinstance(sample_convention, SampleConventionResult):
            raise ValueError("propagation requires sample-convention evidence")
        if not sample_convention.authoritative:
            raise ValueError("propagation requires an authoritative convention result")
        if sample_convention.status == "undecided":
            raise ValueError("undecided sample convention blocks propagation")
        _, settings, source_header, endpoint_header = _load_propagation_capture(
            layout, capture
        )
        del endpoint_header
        capture_run = (
            capture.source_input.run_id,
            capture.source_input.run_instance_uuid,
        )
        convention_run = (
            sample_convention.probes[0].run_id,
            sample_convention.probes[0].run_instance_uuid,
        )
        if capture_run != convention_run:
            raise ValueError("propagation and convention evidence must share one current run")
        endpoint = capture.output_zbfs[0]
        return cls._create(
            segment_key=capture.segment_key,
            case_id=capture.case_id,
            repeat_id=capture.repeat_id,
            source_input=capture.source_input,
            endpoint_artifact=endpoint.artifact,
            effective_cfg_artifact=capture.effective_cfg_artifact,
            settings_artifact=capture.settings_artifact,
            native_report_artifact=capture.report_artifact,
            entrance_settings_sha256=entrance_settings_sha256(
                settings,
                input_zbf_sha256=capture.source_input.sha256,
            ),
            convention_evidence_sha256=convention_evidence_sha256,
            sample_convention=sample_convention,
            wavelength_vacuum_mm=float(source_header.wavelength_vacuum_mm),
            refractive_index=float(source_header.refractive_index),
        )


@dataclass(frozen=True)
class CalibratedEndpoint:
    physical: PointField2D
    endpoint_zbf_sha256: str
    calibration_sha256: str
    propagation_binding_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.physical, PointField2D):
            raise ValueError("calibrated endpoint must be a physical PointField2D")
        for name in (
            "endpoint_zbf_sha256",
            "calibration_sha256",
            "propagation_binding_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)


def _centered_grid_matches(
    field: MappedZbfField,
    binding: IdentityBinding,
    policy: IdentityPolicy,
    *,
    expected_grid_sha256: str,
) -> bool:
    grid = field.physical.grid
    expected = UniformGrid2D.centered(
        nx=binding.nx,
        ny=binding.ny,
        dx_mm=binding.dx_mm,
        dy_mm=binding.dy_mm,
    )
    return bool(
        physical_grid_sha256(grid) == expected_grid_sha256
        and np.allclose(grid.x_mm, expected.x_mm, rtol=policy.grid_rtol, atol=policy.grid_atol_mm)
        and np.allclose(grid.y_mm, expected.y_mm, rtol=policy.grid_rtol, atol=policy.grid_atol_mm)
    )


def _validate_mapped_physical(
    mapped: MappedZbfField,
    *,
    wavelength_vacuum_mm: float,
    refractive_index: float,
    label: str,
) -> None:
    expected_rayleigh = (
        np.pi * refractive_index * mapped.pilot.waist_mm**2 / wavelength_vacuum_mm
    )
    if not np.isclose(
        mapped.pilot.rayleigh_mm,
        expected_rayleigh,
        rtol=_PILOT_RAYLEIGH_RELATION_RELATIVE_TOLERANCE,
        atol=_PILOT_RAYLEIGH_RELATION_ABSOLUTE_TOLERANCE_MM,
    ):
        raise ValueError(f"{label} pilot violates the Rayleigh relation")
    expected = reference_phases(
        mapped.physical.grid,
        mapped.pilot,
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    for name, actual, predicted in (
        ("Q", mapped.references.q_rad, expected.q_rad),
        ("Phi", mapped.references.phi_rad, expected.phi_rad),
    ):
        if not np.allclose(actual, predicted, rtol=2e-13, atol=2e-12):
            raise ValueError(f"{label} paired {name} reference is inconsistent")
    reconstructed = mapped.reference_relative * np.exp(1j * mapped.references.phi_rad)
    scale = max(1.0, float(np.max(np.abs(reconstructed))))
    if not np.allclose(mapped.physical.values, reconstructed, rtol=2e-13, atol=2e-14 * scale):
        raise ValueError(f"{label} physical field is not chi times exp(+i Phi)")


def _validate_identity_evidence(
    input_field: MappedZbfField,
    rewrite_field: MappedZbfField,
    binding: IdentityBinding,
    policy: IdentityPolicy,
) -> None:
    if not isinstance(input_field, MappedZbfField) or not isinstance(rewrite_field, MappedZbfField):
        raise ValueError("identity requires paired-ZBF physical MappedZbfField inputs")
    if input_field is rewrite_field:
        raise ValueError("identity requires an independently captured rewrite field")
    if input_field.source_sha256 != binding.input_artifact.sha256 or rewrite_field.source_sha256 != binding.rewrite_artifact.sha256:
        raise ValueError("identity field hashes do not match current-run artifacts")
    if (
        input_field.convention_evidence_sha256 != binding.convention_evidence_sha256
        or rewrite_field.convention_evidence_sha256 != binding.convention_evidence_sha256
    ):
        raise ValueError("identity convention evidence does not match its binding")
    if (
        input_field.sample_value_convention != binding.sample_value_convention
        or rewrite_field.sample_value_convention != binding.sample_value_convention
    ):
        raise ValueError("identity sample convention does not match the decided evidence")
    if not _centered_grid_matches(
        input_field,
        binding,
        policy,
        expected_grid_sha256=binding.input_grid_sha256,
    ) or not _centered_grid_matches(
        rewrite_field,
        binding,
        policy,
        expected_grid_sha256=binding.rewrite_grid_sha256,
    ):
        raise ValueError("identity complete grid/sampling evidence does not match")
    _validate_mapped_physical(
        input_field,
        wavelength_vacuum_mm=binding.wavelength_vacuum_mm,
        refractive_index=binding.refractive_index,
        label="identity input",
    )
    _validate_mapped_physical(
        rewrite_field,
        wavelength_vacuum_mm=binding.wavelength_vacuum_mm,
        refractive_index=binding.refractive_index,
        label="identity rewrite",
    )


def _peak_connected_roi(intensity: np.ndarray, threshold: float) -> np.ndarray:
    peak_value = float(np.max(intensity))
    if not math.isfinite(peak_value) or peak_value <= 0.0:
        raise ValueError("identity input field must have finite nonzero intensity")
    maxima = intensity == peak_value
    seed_flat = int(np.argmax(intensity))
    seed = tuple(int(value) for value in np.unravel_index(seed_flat, intensity.shape))
    support = intensity >= threshold * peak_value

    def component(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
        selected = np.zeros(mask.shape, dtype=np.bool_)
        selected[start] = True
        queue: deque[tuple[int, int]] = deque([start])
        ny, nx = mask.shape
        while queue:
            row, column = queue.popleft()
            for drow, dcolumn in _NEIGHBOURS_8:
                neighbour = row + drow, column + dcolumn
                if (
                    0 <= neighbour[0] < ny
                    and 0 <= neighbour[1] < nx
                    and mask[neighbour]
                    and not selected[neighbour]
                ):
                    selected[neighbour] = True
                    queue.append(neighbour)
        return selected

    selected = component(support, seed)
    if np.any(maxima & ~selected):
        raise ValueError("identity input has disconnected equal global maxima")
    return selected


def evaluate_start_identity(
    input_field: MappedZbfField,
    rewrite_field: MappedZbfField,
    *,
    binding: IdentityBinding,
    policy: IdentityPolicy = IdentityPolicy(),
) -> IdentityResult:
    """Fit one analytic entrance scalar and evaluate the fixed gates."""

    if not isinstance(binding, IdentityBinding) or not isinstance(policy, IdentityPolicy):
        raise ValueError("identity requires an immutable binding and policy")
    _validate_identity_evidence(input_field, rewrite_field, binding, policy)
    source = input_field.physical.values
    rewrite = rewrite_field.physical.values
    pixel_area = input_field.physical.grid.pixel_area_mm2
    intensity = np.abs(source) ** 2
    roi = _peak_connected_roi(intensity, policy.roi_threshold)
    denominator = np.sum(np.abs(rewrite[roi]) ** 2, dtype=np.float64) * pixel_area
    if not math.isfinite(float(denominator)) or denominator <= 0.0:
        raise ValueError("identity rewrite has no finite nonzero ROI energy")
    numerator = np.sum(source[roi] * np.conj(rewrite[roi])) * pixel_area
    if not np.isfinite(numerator):
        raise ValueError("identity entrance overlap is not finite")
    c_entry = complex(numerator / denominator)
    aligned = c_entry * rewrite
    source_intensity = intensity[roi]
    aligned_intensity = np.abs(aligned[roi]) ** 2
    intensity_denominator = np.sum(source_intensity**2, dtype=np.float64) * pixel_area
    if not math.isfinite(float(intensity_denominator)) or intensity_denominator <= 0.0:
        raise ValueError("identity intensity norm is not finite and nonzero")
    intensity_numerator = np.sum(
        (aligned_intensity - source_intensity) ** 2,
        dtype=np.float64,
    ) * pixel_area
    intensity_percent = float(100.0 * np.sqrt(max(0.0, intensity_numerator / intensity_denominator)))
    phase_defined = bool(np.all(np.abs(source[roi]) > 0.0) and np.all(np.abs(aligned[roi]) > 0.0))
    phase_rms: float | None
    if phase_defined:
        residual = np.angle(aligned[roi] * np.conj(source[roi]))
        phase_rms = float(
            np.sqrt(
                np.sum(source_intensity * residual**2, dtype=np.float64)
                / np.sum(source_intensity, dtype=np.float64)
            )
            / (2.0 * np.pi)
        )
    else:
        phase_rms = None
    roi_hash = _array_sha256(roi, schema="identity_roi_bool/v1")
    calibration = EntranceCalibration._create(
        c_entry=c_entry,
        binding_sha256=binding.binding_sha256,
        policy_sha256=policy.policy_sha256,
        roi_mask_sha256=roi_hash,
        sample_convention_sha256=binding.sample_convention.result_sha256,
        input_zbf_sha256=binding.input_artifact.sha256,
        rewrite_zbf_sha256=binding.rewrite_artifact.sha256,
        segment_key=binding.segment_key,
        case_id=binding.case_id,
        convention_evidence_sha256=binding.convention_evidence_sha256,
    )
    return IdentityResult._create(
        binding=binding,
        policy=policy,
        calibration=calibration,
        roi_mask=roi,
        phase_rms_waves=phase_rms,
        intensity_relative_l2_percent=intensity_percent,
    )


def apply_entrance_calibration(
    endpoint: MappedZbfField,
    identity: IdentityResult,
    *,
    propagation: PropagationBinding,
) -> CalibratedEndpoint:
    """Apply a passed entrance scalar once to the exact captured endpoint."""

    if not isinstance(endpoint, MappedZbfField):
        raise ValueError("entrance calibration requires an uncalibrated MappedZbfField")
    if not isinstance(identity, IdentityResult) or not identity.passed:
        raise ValueError("a failed identity blocks propagation calibration")
    if not isinstance(propagation, PropagationBinding):
        raise ValueError("endpoint calibration requires propagation evidence")
    binding = identity.binding
    if (
        propagation.segment_key != binding.segment_key
        or propagation.case_id != binding.case_id
        or propagation.source_input.to_dict() != binding.input_artifact.to_dict()
        or propagation.entrance_settings_sha256 != binding.entrance_settings_sha256
        or propagation.convention_evidence_sha256 != binding.convention_evidence_sha256
        or propagation.sample_convention.result_sha256
        != binding.sample_convention.result_sha256
        or propagation.wavelength_vacuum_mm != binding.wavelength_vacuum_mm
        or propagation.refractive_index != binding.refractive_index
    ):
        raise ValueError("propagation binding does not match the entrance identity")
    if endpoint.source_sha256 != propagation.endpoint_artifact.sha256:
        raise ValueError("endpoint field does not match the captured propagation artifact")
    if (
        endpoint.convention_evidence_sha256 != binding.convention_evidence_sha256
        or endpoint.sample_value_convention != binding.sample_value_convention
    ):
        raise ValueError("endpoint physical-field convention does not match identity")
    _validate_mapped_physical(
        endpoint,
        wavelength_vacuum_mm=binding.wavelength_vacuum_mm,
        refractive_index=binding.refractive_index,
        label="propagation endpoint",
    )
    return CalibratedEndpoint(
        physical=PointField2D(
            identity.calibration.c_entry * endpoint.physical.values,
            endpoint.physical.grid,
        ),
        endpoint_zbf_sha256=endpoint.source_sha256,
        calibration_sha256=identity.calibration.calibration_sha256,
        propagation_binding_sha256=propagation.binding_sha256,
    )


__all__ = [
    "CalibratedEndpoint",
    "EntranceCalibration",
    "IdentityBinding",
    "IdentityNativeReport",
    "IdentityPolicy",
    "IdentityResult",
    "PropagationBinding",
    "SampleConventionPolicy",
    "SampleConventionProbe",
    "SampleConventionResult",
    "apply_entrance_calibration",
    "classify_sample_value_convention",
    "entrance_settings_sha256",
    "evaluate_start_identity",
    "physical_grid_sha256",
]
