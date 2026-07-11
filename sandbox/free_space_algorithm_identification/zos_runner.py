"""Fail-closed sustained ZOSPy capture for fixed-input POP segment runs."""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from .artifacts import (
    ArtifactHash,
    ArtifactRef,
    RunLayout,
    copy_file_once,
    hash_artifact,
    verify_artifact_ref,
    write_json_once,
)
from .biconic_case import BICONIC_SEGMENTS, FIXED_INPUT_SURFACES
from .native_report import (
    NativePopRequest,
    NativeSettingsReadback,
    validate_settings_readback,
)
from .zbf_binary import HEADER_BYTES, RawZbfHeader


_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9_.-]+\Z")
_SAMPLE_ENUM = re.compile(r"(?:.*\.)?S_(\d+)x(\d+)\Z")
_STAGE_OUTPUT_SUFFIXES = (
    "effective.CFG",
    "settings_readback.json",
    "report_raw.txt",
    "raw_grid.npy",
    "raw_grid.json",
    "messages.json",
)


def _safe_component(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SAFE_COMPONENT.fullmatch(value) is None
        or value in {".", ".."}
        or value.endswith(".")
    ):
        raise ValueError(f"{label} must be one safe path component")
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


def _positive_int(value: object, *, label: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _positive_float(value: object, *, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return parsed


def _float_equal(left: float, right: float) -> bool:
    scale = max(1.0, abs(float(left)), abs(float(right)))
    return abs(float(left) - float(right)) <= 64.0 * math.ulp(scale)


def _segment_for_key(key: str):
    matches = tuple(segment for segment in BICONIC_SEGMENTS if segment.key == key)
    if len(matches) != 1:
        raise ValueError(f"unknown fixed biconic segment: {key!r}")
    return matches[0]


@dataclass(frozen=True)
class SegmentPopRequest:
    """One fixed-input identity or adjacent biconic POP transfer request."""

    segment_key: str
    case_id: str
    repeat_id: str
    start_surface: int
    end_surface: int
    input_artifact: ArtifactRef
    input_producer_stage: str
    input_producer_case: str
    nx: int
    ny: int
    x_width_mm: float
    y_width_mm: float
    wavelength_number: int
    wavelength_vacuum_mm: float
    refractive_index: float
    field_number: int
    use_polarization: bool
    normalization_mode: Literal["total_power", "peak_irradiance"]
    normalization_value: float
    use_disk_storage: bool
    data_grid_index: int

    def __post_init__(self) -> None:
        segment_key = _safe_component(self.segment_key, label="segment_key")
        case_id = _safe_component(self.case_id, label="case_id")
        repeat_id = _safe_component(self.repeat_id, label="repeat_id")
        segment = _segment_for_key(segment_key)
        start = _positive_int(self.start_surface, label="start_surface")
        end = _positive_int(self.end_surface, label="end_surface")
        valid_end_surfaces = {segment.start_surface, segment.end_surface}
        if (
            start not in FIXED_INPUT_SURFACES
            or start != segment.start_surface
            or end not in valid_end_surfaces
        ):
            raise ValueError(
                "request must be a fixed biconic Start=End identity or exact segment"
            )
        if not isinstance(self.input_artifact, ArtifactRef):
            raise ValueError("input must be a run ArtifactRef")
        input_stage = _logical_identifier(
            self.input_producer_stage,
            label="input producer stage",
        )
        input_case = _logical_identifier(
            self.input_producer_case,
            label="input producer case",
        )
        nx = _positive_int(self.nx, label="nx", minimum=2)
        ny = _positive_int(self.ny, label="ny", minimum=2)
        if nx != ny or nx % 2 or ny % 2:
            raise ValueError("POP sampling must be an even square sample enumeration")
        x_width = _positive_float(self.x_width_mm, label="x_width_mm")
        y_width = _positive_float(self.y_width_mm, label="y_width_mm")
        wavelength = _positive_int(
            self.wavelength_number,
            label="wavelength_number",
        )
        wavelength_vacuum = _positive_float(
            self.wavelength_vacuum_mm,
            label="wavelength_vacuum_mm",
        )
        refractive_index = _positive_float(
            self.refractive_index,
            label="refractive_index",
        )
        field_number = _positive_int(self.field_number, label="field_number")
        if type(self.use_polarization) is not bool:
            raise ValueError("use_polarization must be boolean")
        if self.normalization_mode not in {"total_power", "peak_irradiance"}:
            raise ValueError("normalization_mode is unsupported")
        normalization = _positive_float(
            self.normalization_value,
            label="normalization_value",
        )
        if type(self.use_disk_storage) is not bool:
            raise ValueError("use_disk_storage must be boolean")
        data_grid_index = _positive_int(
            self.data_grid_index,
            label="data_grid_index",
            minimum=0,
        )
        object.__setattr__(self, "segment_key", segment_key)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "repeat_id", repeat_id)
        object.__setattr__(self, "input_producer_stage", input_stage)
        object.__setattr__(self, "input_producer_case", input_case)
        object.__setattr__(self, "x_width_mm", x_width)
        object.__setattr__(self, "y_width_mm", y_width)
        object.__setattr__(self, "wavelength_number", wavelength)
        object.__setattr__(self, "wavelength_vacuum_mm", wavelength_vacuum)
        object.__setattr__(self, "refractive_index", refractive_index)
        object.__setattr__(self, "field_number", field_number)
        object.__setattr__(self, "normalization_value", normalization)
        object.__setattr__(self, "data_grid_index", data_grid_index)

    @property
    def stage(self) -> Literal["identity", "propagation"]:
        return "identity" if self.start_surface == self.end_surface else "propagation"

    def output_prefix(self, run_id: str) -> str:
        run = _safe_component(run_id, label="run_id")
        return (
            f"P_{run}_{self.segment_key}_{self.case_id}_{self.stage}_"
            f"{self.repeat_id}"
        )

    def staged_input_name(self, run_id: str) -> str:
        run = _safe_component(run_id, label="run_id")
        return (
            f"I_{run}_{self.segment_key}_{self.case_id}_{self.stage}_"
            f"{self.repeat_id}.ZBF"
        )


def expected_output_names(
    prefix: str,
    start_surface: int,
    end_surface: int,
) -> tuple[str, ...]:
    """Return the only anchored save-all filenames accepted for one request."""

    stem = _safe_component(prefix, label="output prefix")
    start = _positive_int(start_surface, label="start_surface")
    end = _positive_int(end_surface, label="end_surface")
    surfaces = (start,) if start == end else (start, end)
    return (f"{stem}.ZBF", *(f"{stem}_{surface:04d}.ZBF" for surface in surfaces))


def _immutable_float_array(values: object, *, shape: tuple[int, int]) -> np.ndarray:
    array = np.array(values, dtype=np.float64, order="C", copy=True)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"raw DataGrid Values must be finite with shape {shape}")
    immutable = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        shape
    )
    immutable.setflags(write=False)
    return immutable


@dataclass(frozen=True)
class RawDataGridSnapshot:
    nx: int
    ny: int
    min_x_mm: float
    min_y_mm: float
    dx_mm: float
    dy_mm: float
    x_checkpoints: tuple[tuple[int, float], ...]
    y_checkpoints: tuple[tuple[int, float], ...]
    z_checkpoints: tuple[tuple[int, int, float], ...]
    values_checkpoints: tuple[tuple[int, int, float], ...]
    values_api_xy: np.ndarray
    values_package_yx: np.ndarray
    values_api_xy_sha256: str
    values_package_yx_sha256: str

    def __post_init__(self) -> None:
        nx = _positive_int(self.nx, label="raw grid nx", minimum=2)
        ny = _positive_int(self.ny, label="raw grid ny", minimum=2)
        api = _immutable_float_array(self.values_api_xy, shape=(nx, ny))
        package = _immutable_float_array(self.values_package_yx, shape=(ny, nx))
        if not np.array_equal(package, api.T):
            raise ValueError("package raw grid must be exactly one API XY transpose")
        api_hash = hashlib.sha256(api.tobytes(order="C")).hexdigest()
        package_hash = hashlib.sha256(package.tobytes(order="C")).hexdigest()
        if self.values_api_xy_sha256 != api_hash:
            raise ValueError("raw API XY array hash mismatch")
        if self.values_package_yx_sha256 != package_hash:
            raise ValueError("raw package YX array hash mismatch")
        object.__setattr__(self, "values_api_xy", api)
        object.__setattr__(self, "values_package_yx", package)


@dataclass(frozen=True)
class CapturedZbf:
    name: str
    artifact: ArtifactRef
    header: RawZbfHeader


@dataclass(frozen=True)
class CapturedMessage:
    source: Literal["analysis", "result", "application"]
    error_code: str
    text: str


@dataclass(frozen=True)
class CapturedPopRun:
    segment_key: str
    case_id: str
    repeat_id: str
    stage: Literal["identity", "propagation"]
    source_input: ArtifactRef
    staged_input_name: str
    staged_input_hash: ArtifactHash
    effective_cfg_artifact: ArtifactRef
    settings_readback: NativeSettingsReadback
    settings_artifact: ArtifactRef
    report_artifact: ArtifactRef
    output_zbfs: tuple[CapturedZbf, ...]
    raw_grid: RawDataGridSnapshot
    raw_grid_array_artifact: ArtifactRef
    raw_grid_metadata_artifact: ArtifactRef
    messages: tuple[CapturedMessage, ...]
    messages_artifact: ArtifactRef
    cleanup_errors: tuple[str, ...] = ()


class RunnerCleanupError(RuntimeError):
    def __init__(self, errors: tuple[str, ...]) -> None:
        self.errors = tuple(errors)
        super().__init__("runner cleanup failed: " + "; ".join(self.errors))


def _read_zbf_header(path: Path) -> RawZbfHeader:
    with path.open("rb") as stream:
        raw = stream.read(HEADER_BYTES)
    try:
        return RawZbfHeader.from_bytes(raw)
    except ValueError as error:
        raise ValueError(f"invalid ZBF header: {path.name}") from error


def _validate_input_artifact(
    layout: RunLayout,
    request: SegmentPopRequest,
) -> tuple[Path, RawZbfHeader, ArtifactHash]:
    source = verify_artifact_ref(
        layout,
        request.input_artifact,
        expected_producer_stage=request.input_producer_stage,
        expected_producer_case=request.input_producer_case,
    )
    header = _read_zbf_header(source)
    mismatches: list[str] = []
    if (header.nx, header.ny) != (request.nx, request.ny):
        mismatches.append("N")
    if not _float_equal(header.nx * header.dx, request.x_width_mm):
        mismatches.append("x_width")
    if not _float_equal(header.ny * header.dy, request.y_width_mm):
        mismatches.append("y_width")
    if header.units != 0:
        mismatches.append("millimetre_units")
    if header.is_polarized not in {0, 1} or bool(
        header.is_polarized
    ) != request.use_polarization:
        mismatches.append("polarization")
    if not _float_equal(
        header.wavelength_vacuum_mm,
        request.wavelength_vacuum_mm,
    ):
        mismatches.append("wavelength_vacuum_mm")
    if not _float_equal(header.refractive_index, request.refractive_index):
        mismatches.append("refractive_index")
    if mismatches:
        raise ValueError(
            "input ZBF header does not match request: " + ", ".join(mismatches)
        )
    return source, header, hash_artifact(source)


def _coordinate_close(actual: float, expected: float) -> bool:
    scale = max(1.0, abs(actual), abs(expected))
    return abs(actual - expected) <= 128.0 * math.ulp(scale)


def capture_raw_data_grid(
    data_grid: object,
    *,
    expected_header: RawZbfHeader | None = None,
) -> RawDataGridSnapshot:
    """Snapshot one live IAR_DataGrid using its raw API XY indexing."""

    try:
        nx = _positive_int(data_grid.Nx, label="raw grid Nx", minimum=2)
        ny = _positive_int(data_grid.Ny, label="raw grid Ny", minimum=2)
        min_x = float(data_grid.MinX)
        min_y = float(data_grid.MinY)
        dx = _positive_float(data_grid.Dx, label="raw grid Dx")
        dy = _positive_float(data_grid.Dy, label="raw grid Dy")
    except AttributeError as error:
        raise ValueError("raw IAR_DataGrid is missing coordinate metadata") from error
    if not all(math.isfinite(value) for value in (min_x, min_y)):
        raise ValueError("raw IAR_DataGrid coordinate origins must be finite")
    if nx % 2 or ny % 2:
        raise ValueError("raw IAR_DataGrid must use even sample-at-zero dimensions")

    x_indices = (0, nx // 2, nx - 1)
    y_indices = (0, ny // 2, ny - 1)
    try:
        x_checkpoints = tuple((index, float(data_grid.X(index))) for index in x_indices)
        y_checkpoints = tuple((index, float(data_grid.Y(index))) for index in y_indices)
    except (AttributeError, TypeError) as error:
        raise ValueError(
            "raw IAR_DataGrid is missing X/Y coordinate accessors"
        ) from error
    for index, actual in x_checkpoints:
        expected = min_x + index * dx
        if not _coordinate_close(actual, expected):
            raise ValueError("raw IAR_DataGrid X coordinate rule mismatch")
    for index, actual in y_checkpoints:
        expected = min_y + index * dy
        if not _coordinate_close(actual, expected):
            raise ValueError("raw IAR_DataGrid Y coordinate rule mismatch")
    if not _coordinate_close(x_checkpoints[1][1], 0.0) or not _coordinate_close(
        y_checkpoints[1][1], 0.0
    ):
        raise ValueError("raw IAR_DataGrid violates sample-at-zero coordinates")

    try:
        values_api = np.asarray(data_grid.Values, dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("raw IAR_DataGrid Values are unavailable") from error
    values_api = _immutable_float_array(values_api, shape=(nx, ny))
    checkpoint_indices = tuple(
        dict.fromkeys(
            (
                (0, 0),
                (nx // 2, ny // 2),
                (nx - 1, ny - 1),
                (min(1, nx - 1), max(0, ny - 2)),
            )
        )
    )
    try:
        z_checkpoints = tuple(
            (ix, iy, float(data_grid.Z(ix, iy)))
            for ix, iy in checkpoint_indices
        )
    except (AttributeError, TypeError) as error:
        raise ValueError("raw IAR_DataGrid is missing Z(ix,iy)") from error
    values_checkpoints = tuple(
        (ix, iy, float(values_api[ix, iy])) for ix, iy in checkpoint_indices
    )
    for z_value, values_value in zip(
        z_checkpoints,
        values_checkpoints,
        strict=True,
    ):
        if z_value[:2] != values_value[:2] or not _float_equal(
            z_value[2], values_value[2]
        ):
            raise ValueError("raw IAR_DataGrid Z does not equal Values")

    if expected_header is not None:
        if (nx, ny) != (expected_header.nx, expected_header.ny):
            raise ValueError("raw IAR_DataGrid count does not match output ZBF header")
        if not _float_equal(dx, expected_header.dx) or not _float_equal(
            dy, expected_header.dy
        ):
            raise ValueError(
                "raw IAR_DataGrid sampling does not match output ZBF header"
            )

    package = np.ascontiguousarray(values_api.T)
    api_hash = hashlib.sha256(values_api.tobytes(order="C")).hexdigest()
    package_hash = hashlib.sha256(package.tobytes(order="C")).hexdigest()
    return RawDataGridSnapshot(
        nx=nx,
        ny=ny,
        min_x_mm=min_x,
        min_y_mm=min_y,
        dx_mm=dx,
        dy_mm=dy,
        x_checkpoints=x_checkpoints,
        y_checkpoints=y_checkpoints,
        z_checkpoints=z_checkpoints,
        values_checkpoints=values_checkpoints,
        values_api_xy=values_api,
        values_package_yx=package,
        values_api_xy_sha256=api_hash,
        values_package_yx_sha256=package_hash,
    )


def _stage_relative_dir(request: SegmentPopRequest) -> str:
    return f"{request.segment_key}/{request.case_id}/{request.stage}"


def _stage_path(layout: RunLayout, request: SegmentPopRequest, name: str) -> Path:
    safe_name = _safe_component(name, label="stage artifact name")
    directory = (
        layout.run_dir / request.segment_key / request.case_id / request.stage
    ).resolve(strict=True)
    if not directory.is_relative_to(layout.run_dir):
        raise ValueError("capture stage must remain inside the run")
    return directory / safe_name


def _artifact_ref(
    layout: RunLayout,
    request: SegmentPopRequest,
    name: str,
) -> ArtifactRef:
    relative = f"{_stage_relative_dir(request)}/{name}"
    return ArtifactRef.from_file(
        layout,
        relative,
        producer_stage=request.stage,
        producer_case=request.case_id,
    )


def _ensure_stage_is_empty(
    layout: RunLayout,
    request: SegmentPopRequest,
    prefix: str,
) -> None:
    names = [f"{prefix}_{suffix}" for suffix in _STAGE_OUTPUT_SUFFIXES]
    names.extend(
        expected_output_names(prefix, request.start_surface, request.end_surface)
    )
    existing = tuple(
        name for name in names if _stage_path(layout, request, name).exists()
    )
    if existing:
        raise ValueError(f"run stage already contains capture artifacts: {existing}")


def _resolve_pop_dir(oss: object) -> Path:
    try:
        raw = oss.TheApplication.POPDir
    except AttributeError as error:
        raise RuntimeError(
            "connected OpticStudio system does not expose POPDir"
        ) from error
    path = Path(str(raw)).resolve(strict=True)
    if not path.is_dir():
        raise RuntimeError("connected OpticStudio POPDir is not a directory")
    return path


def _anchored_pop_outputs(pop_dir: Path, prefix: str) -> tuple[Path, ...]:
    pattern = re.compile(rf"{re.escape(prefix)}(?:_\d{{4}})?\.ZBF\Z")
    return tuple(
        sorted(
            (
                path
                for path in pop_dir.iterdir()
                if path.is_file() and pattern.fullmatch(path.name) is not None
            ),
            key=lambda path: path.name,
        )
    )


def _reject_stale_outputs(pop_dir: Path, prefix: str) -> None:
    stale = _anchored_pop_outputs(pop_dir, prefix)
    if stale:
        raise ValueError(
            "stale anchored POP output exists before run: "
            + ", ".join(path.name for path in stale)
        )


def _collect_exact_outputs(
    pop_dir: Path,
    prefix: str,
    expected_names: tuple[str, ...],
) -> tuple[Path, ...]:
    actual_paths = _anchored_pop_outputs(pop_dir, prefix)
    actual = {path.name: path for path in actual_paths}
    expected = set(expected_names)
    missing = tuple(name for name in expected_names if name not in actual)
    extra = tuple(sorted(set(actual) - expected))
    if missing:
        raise ValueError(f"missing anchored POP output: {missing}")
    if extra:
        raise ValueError(f"extra anchored POP output is forbidden: {extra}")
    return tuple(actual[name] for name in expected_names)


def _validate_output_header(
    header: RawZbfHeader,
    request: SegmentPopRequest,
    *,
    name: str,
) -> None:
    mismatches: list[str] = []
    if (header.nx, header.ny) != (request.nx, request.ny):
        mismatches.append("N")
    if header.units != 0:
        mismatches.append("millimetre_units")
    if header.is_polarized not in {0, 1} or bool(
        header.is_polarized
    ) != request.use_polarization:
        mismatches.append("polarization")
    if not _float_equal(
        header.wavelength_vacuum_mm,
        request.wavelength_vacuum_mm,
    ):
        mismatches.append("wavelength_vacuum_mm")
    if not _float_equal(header.refractive_index, request.refractive_index):
        mismatches.append("refractive_index")
    if mismatches:
        raise ValueError(
            f"output ZBF header mismatch for {name}: " + ", ".join(mismatches)
        )


def _parse_sample_enum(value: object, *, label: str) -> tuple[str, int, int]:
    text = str(value)
    match = _SAMPLE_ENUM.fullmatch(text)
    if match is None:
        raise ValueError(f"{label} is not a recognized POP sample enum: {text!r}")
    nx, ny = int(match.group(1)), int(match.group(2))
    return f"S_{nx}x{ny}", nx, ny


def _require_live_setting(
    actual: object,
    expected: object,
    *,
    label: str,
) -> None:
    matches = (
        _float_equal(float(actual), float(expected))
        if type(expected) is float
        else actual == expected and type(actual) is type(expected)
    )
    if not matches:
        raise ValueError(
            f"settings readback mismatch for {label}: "
            f"requested={expected!r}, actual={actual!r}"
        )


def _capture_settings_readback(
    analysis: object,
    oss: object,
    source_header: RawZbfHeader,
    source_hash: ArtifactHash,
    request: SegmentPopRequest,
    *,
    staged_input_name: str,
    output_prefix: str,
) -> tuple[NativeSettingsReadback, dict[str, object]]:
    try:
        settings = analysis.Settings
        start_surface = int(settings.StartSurface.GetSurfaceNumber())
        end_surface = int(settings.EndSurface.GetSurfaceNumber())
        x_enum, x_nx, x_ny = _parse_sample_enum(
            settings.XSampling,
            label="XSampling",
        )
        y_enum, y_nx, y_ny = _parse_sample_enum(
            settings.YSampling,
            label="YSampling",
        )
        wavelength = int(settings.Wavelength.GetWavelengthNumber())
        field_number = int(settings.Field.GetFieldNumber())
    except AttributeError as error:
        raise ValueError("live POP settings readback is incomplete") from error
    if x_enum != y_enum or (x_nx, x_ny, y_nx, y_ny) != (
        request.nx,
        request.ny,
        request.nx,
        request.ny,
    ):
        raise ValueError("settings readback sample enums do not match request")
    try:
        model_wavelength_um = float(
            oss.SystemData.Wavelengths.GetWavelength(wavelength).Wavelength
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "loaded model wavelength table readback is unavailable"
        ) from error
    model_wavelength_mm = 1.0e-3 * model_wavelength_um
    if not math.isfinite(model_wavelength_mm) or model_wavelength_mm <= 0.0:
        raise ValueError("loaded model wavelength must be positive and finite")
    if not _float_equal(
        model_wavelength_mm,
        source_header.wavelength_vacuum_mm,
    ):
        raise ValueError(
            "loaded model wavelength table does not match verified input ZBF header"
        )

    use_total = bool(settings.UseTotalPower)
    use_peak = bool(settings.UsePeakIrradiance)
    if use_total == use_peak:
        raise ValueError("settings readback normalization flags are ambiguous")
    normalization_mode = "total_power" if use_total else "peak_irradiance"
    normalization_value = float(
        settings.TotalPower if use_total else settings.PeakIrradiance
    )
    readback = NativeSettingsReadback(
        start_surface=start_surface,
        end_surface=end_surface,
        nx=x_nx,
        ny=y_ny,
        sample_size_enum=x_enum,
        x_width_mm=float(settings.XWidth),
        y_width_mm=float(settings.YWidth),
        wavelength_number=wavelength,
        wavelength_vacuum_mm=float(source_header.wavelength_vacuum_mm),
        refractive_index=float(source_header.refractive_index),
        field_number=field_number,
        use_polarization=bool(settings.UsePolarization),
        normalization_mode=normalization_mode,
        normalization_value=normalization_value,
        input_beam_file=str(settings.BeamTypeFilename),
        output_beam_file=str(settings.OutputBeamFile),
        save_output_beam=bool(settings.SaveOutputBeam),
        save_beam_at_all_surfaces=bool(settings.SaveBeamAtAllSurfaces),
    )
    requested = NativePopRequest(
        start_surface=request.start_surface,
        end_surface=request.end_surface,
        nx=request.nx,
        ny=request.ny,
        sample_size_enum=f"S_{request.nx}x{request.ny}",
        x_width_mm=request.x_width_mm,
        y_width_mm=request.y_width_mm,
        wavelength_number=request.wavelength_number,
        wavelength_vacuum_mm=request.wavelength_vacuum_mm,
        refractive_index=request.refractive_index,
        field_number=request.field_number,
        use_polarization=request.use_polarization,
        normalization_mode=request.normalization_mode,
        normalization_value=request.normalization_value,
        input_beam_file=staged_input_name,
        output_beam_file=output_prefix,
        save_output_beam=True,
        save_beam_at_all_surfaces=True,
    )
    validate_settings_readback(requested, readback)

    extras = {
        "beam_type": str(settings.BeamType),
        "beam_file": str(settings.BeamTypeFilename),
        "project": str(settings.Project),
        "surface_to_beam": float(settings.SurfaceToBeam),
        "separate_xy": bool(settings.SeparateXY),
        "use_disk_storage": bool(settings.UseDiskStorage),
        "x_sampling_raw": str(settings.XSampling),
        "y_sampling_raw": str(settings.YSampling),
        "model_wavelength_vacuum_mm": model_wavelength_mm,
        "wavelength_vacuum_mm_source": (
            "loaded_model_wavelength_table_and_verified_input_zbf_header"
        ),
        "refractive_index_source": "verified_input_zbf_header",
        "physical_contract_input_zbf_sha256": source_hash.sha256,
    }
    for label, actual, expected in (
        ("beam_type", extras["beam_type"], "File"),
        ("beam_file", extras["beam_file"], staged_input_name),
        ("project", extras["project"], "AlongBeam"),
        ("surface_to_beam", extras["surface_to_beam"], 0.0),
        ("separate_xy", extras["separate_xy"], False),
        ("use_disk_storage", extras["use_disk_storage"], request.use_disk_storage),
    ):
        _require_live_setting(actual, expected, label=label)
    return readback, {**asdict(readback), **extras}


def _write_npy_once(path: Path, values: np.ndarray) -> None:
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            np.save(stream, values, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        if created:
            path.unlink(missing_ok=True)
        raise


def _raw_grid_payload(snapshot: RawDataGridSnapshot) -> dict[str, object]:
    return {
        "nx": snapshot.nx,
        "ny": snapshot.ny,
        "min_x_mm": snapshot.min_x_mm,
        "min_y_mm": snapshot.min_y_mm,
        "dx_mm": snapshot.dx_mm,
        "dy_mm": snapshot.dy_mm,
        "x_checkpoints": [list(item) for item in snapshot.x_checkpoints],
        "y_checkpoints": [list(item) for item in snapshot.y_checkpoints],
        "z_checkpoints": [list(item) for item in snapshot.z_checkpoints],
        "values_checkpoints": [list(item) for item in snapshot.values_checkpoints],
        "values_api_xy_sha256": snapshot.values_api_xy_sha256,
        "values_package_yx_sha256": snapshot.values_package_yx_sha256,
        "api_array_order": "Values[x,y]",
        "package_array_order": "field[y,x]",
        "api_to_package_transform": "transpose_once",
        "grid_center_rule": "sample_at_zero",
    }


def _captured_message(source: str, message: object) -> CapturedMessage:
    error_code = str(getattr(message, "ErrorCode", ""))
    text = getattr(message, "Message", getattr(message, "Text", None))
    if text is None:
        text = str(message)
    return CapturedMessage(source=source, error_code=error_code, text=str(text))


def _capture_messages(
    analysis: object,
    result: object,
    zos: object,
) -> tuple[CapturedMessage, ...]:
    try:
        analysis_messages = tuple(analysis.messages)
        result_messages = tuple(result.messages)
        application_log = zos.retrieve_logs()
    except AttributeError as error:
        raise ValueError(
            "live POP messages or application log are unavailable"
        ) from error
    if not isinstance(application_log, str):
        raise ValueError("application log must be captured as text")
    captured = [
        *(_captured_message("analysis", message) for message in analysis_messages),
        *(_captured_message("result", message) for message in result_messages),
    ]
    captured.extend(
        CapturedMessage("application", "", line)
        for line in application_log.splitlines()
    )
    return tuple(captured)


def _capture_connected_run(
    layout: RunLayout,
    request: SegmentPopRequest,
    *,
    source_input: ArtifactRef,
    source_header: RawZbfHeader,
    source_hash: ArtifactHash,
    staged_input_name: str,
    output_prefix: str,
    pop_dir: Path,
    analysis: object,
    result: object,
    zos: object,
    oss: object,
) -> CapturedPopRun:
    prefix = output_prefix
    effective_name = f"{prefix}_effective.CFG"
    settings_name = f"{prefix}_settings_readback.json"
    report_name = f"{prefix}_report_raw.txt"
    raw_array_name = f"{prefix}_raw_grid.npy"
    raw_meta_name = f"{prefix}_raw_grid.json"
    messages_name = f"{prefix}_messages.json"

    effective_path = _stage_path(layout, request, effective_name)
    analysis.Settings.SaveTo(str(effective_path))
    effective_ref = _artifact_ref(layout, request, effective_name)

    readback, settings_payload = _capture_settings_readback(
        analysis,
        oss,
        source_header,
        source_hash,
        request,
        staged_input_name=staged_input_name,
        output_prefix=output_prefix,
    )
    settings_relative = f"{_stage_relative_dir(request)}/{settings_name}"
    write_json_once(layout, settings_relative, settings_payload)
    settings_ref = _artifact_ref(layout, request, settings_name)

    report_path = _stage_path(layout, request, report_name)
    analysis.Results.GetTextFile(str(report_path))
    report_ref = _artifact_ref(layout, request, report_name)

    expected_names = expected_output_names(
        output_prefix,
        request.start_surface,
        request.end_surface,
    )
    pop_outputs = _collect_exact_outputs(pop_dir, output_prefix, expected_names)
    output_zbfs: list[CapturedZbf] = []
    for name, pop_path in zip(expected_names, pop_outputs, strict=True):
        source_digest = hash_artifact(pop_path)
        relative = f"{_stage_relative_dir(request)}/{name}"
        copied_path = copy_file_once(layout, pop_path, relative)
        copied_digest = hash_artifact(copied_path)
        if copied_digest != source_digest:
            raise OSError(f"copied POP output hash mismatch: {name}")
        header = _read_zbf_header(copied_path)
        _validate_output_header(header, request, name=name)
        output_zbfs.append(
            CapturedZbf(
                name=name,
                artifact=_artifact_ref(layout, request, name),
                header=header,
            )
        )

    try:
        data_grid = analysis.Results.DataGrids[request.data_grid_index]
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError("requested raw IAR_DataGrid is unavailable") from error
    raw_grid = capture_raw_data_grid(
        data_grid,
        expected_header=output_zbfs[0].header,
    )
    raw_array_path = _stage_path(layout, request, raw_array_name)
    _write_npy_once(raw_array_path, raw_grid.values_api_xy)
    raw_array_ref = _artifact_ref(layout, request, raw_array_name)
    raw_meta_relative = f"{_stage_relative_dir(request)}/{raw_meta_name}"
    write_json_once(layout, raw_meta_relative, _raw_grid_payload(raw_grid))
    raw_meta_ref = _artifact_ref(layout, request, raw_meta_name)

    messages = _capture_messages(analysis, result, zos)
    messages_relative = f"{_stage_relative_dir(request)}/{messages_name}"
    write_json_once(
        layout,
        messages_relative,
        {"messages": [asdict(message) for message in messages]},
    )
    messages_ref = _artifact_ref(layout, request, messages_name)

    return CapturedPopRun(
        segment_key=request.segment_key,
        case_id=request.case_id,
        repeat_id=request.repeat_id,
        stage=request.stage,
        source_input=source_input,
        staged_input_name=staged_input_name,
        staged_input_hash=source_hash,
        effective_cfg_artifact=effective_ref,
        settings_readback=readback,
        settings_artifact=settings_ref,
        report_artifact=report_ref,
        output_zbfs=tuple(output_zbfs),
        raw_grid=raw_grid,
        raw_grid_array_artifact=raw_array_ref,
        raw_grid_metadata_artifact=raw_meta_ref,
        messages=messages,
        messages_artifact=messages_ref,
    )


def _default_factories() -> tuple[Callable[[], object], Callable[..., object]]:
    configured = os.environ.get("BTS_ZOSPY_PATH")
    candidate = Path(configured) if configured else Path(r"D:\BTS_ZosApi\ZOSPy-main")
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
    try:
        import zospy as zp
        from zospy.analyses.physicaloptics import PhysicalOpticsPropagation
    except ImportError as error:
        raise RuntimeError(
            "ZOSPy is unavailable; configure BTS_ZOSPY_PATH for a live run"
        ) from error
    return zp.ZOS, PhysicalOpticsPropagation


def _pop_kwargs(
    request: SegmentPopRequest,
    *,
    staged_input_name: str,
    output_prefix: str,
) -> dict[str, object]:
    use_total = request.normalization_mode == "total_power"
    return {
        "wavelength": request.wavelength_number,
        "field": request.field_number,
        "start_surface": request.start_surface,
        "end_surface": request.end_surface,
        "surface_to_beam": 0.0,
        "use_polarization": request.use_polarization,
        "separate_xy": False,
        "use_disk_storage": request.use_disk_storage,
        "beam_type": "File",
        "beam_file": staged_input_name,
        "x_sampling": request.nx,
        "y_sampling": request.ny,
        "x_width": request.x_width_mm,
        "y_width": request.y_width_mm,
        "use_total_power": use_total,
        "total_power": request.normalization_value if use_total else 1.0,
        "use_peak_irradiance": not use_total,
        "peak_irradiance": request.normalization_value if not use_total else 1.0,
        "show_as": "FalseColor",
        "data_type": "Irradiance",
        "project": "AlongBeam",
        "save_output_beam": True,
        "output_beam_file": output_prefix,
        "save_beam_at_all_surfaces": True,
        "auto_calculate_beam_sampling": False,
    }


def _cleanup_error(label: str, error: BaseException) -> str:
    return f"{label}: {type(error).__name__}: {error}"


def _active_analysis(wrapper: object | None) -> object | None:
    if wrapper is None:
        return None
    try:
        return wrapper.analysis
    except BaseException:
        return None


def _run_sustained_without_dataframe_unpack(
    wrapper: object,
    oss: object,
) -> object:
    """Run ZOSPy while bypassing only its eager high-level grid conversion."""

    try:
        original_get_data_grid = wrapper.get_data_grid
        wrapper.get_data_grid = lambda *args, **kwargs: None
    except (AttributeError, TypeError) as error:
        raise RuntimeError(
            "PhysicalOpticsPropagation cannot suppress high-level grid unpack"
        ) from error

    result: object | None = None
    primary: BaseException | None = None
    primary_traceback = None
    try:
        result = wrapper.run(oss, oncomplete="Sustain")
    except BaseException as error:
        primary = error
        primary_traceback = error.__traceback__
    try:
        wrapper.get_data_grid = original_get_data_grid
    except BaseException as restore_error:
        if primary is not None:
            primary.add_note(
                "runner wrapper cleanup error: "
                f"{type(restore_error).__name__}: {restore_error}"
            )
        else:
            raise RuntimeError(
                "failed to restore ZOSPy high-level grid adapter"
            ) from restore_error
    if primary is not None:
        raise primary.with_traceback(primary_traceback)
    if result is None:
        raise RuntimeError("sustained ZOSPy run returned no AnalysisResult")
    return result


def _copy_staged_input_exclusive(
    source: Path,
    destination: Path,
    expected_hash: ArtifactHash,
) -> ArtifactHash:
    """Create one owned POPDir input without overwriting any existing file."""

    created = False
    hasher = hashlib.sha256()
    byte_count = 0
    try:
        with source.open("rb") as input_stream, destination.open("xb") as output:
            created = True
            while True:
                block = input_stream.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                hasher.update(block)
                byte_count += len(block)
            output.flush()
            os.fsync(output.fileno())
        actual = ArtifactHash(byte_count=byte_count, sha256=hasher.hexdigest())
        if actual != expected_hash:
            raise OSError("staged POP input hash does not match run artifact")
        return actual
    except BaseException as primary:
        if created:
            try:
                destination.unlink(missing_ok=True)
            except BaseException as cleanup_error:
                primary.add_note(
                    "staged input rollback error: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        raise


def capture_segment_run(
    layout: RunLayout,
    request: SegmentPopRequest,
    *,
    zos_factory: Callable[[], object] | None = None,
    pop_factory: Callable[..., object] | None = None,
) -> CapturedPopRun:
    """Run and capture one sustained File-beam POP identity or transfer."""

    if not isinstance(layout, RunLayout):
        raise ValueError("capture requires a RunLayout")
    if not isinstance(request, SegmentPopRequest):
        raise ValueError("capture requires a SegmentPopRequest")
    source_path, source_header, source_hash = _validate_input_artifact(
        layout,
        request,
    )
    output_prefix = request.output_prefix(layout.run_id)
    staged_input_name = request.staged_input_name(layout.run_id)
    _ensure_stage_is_empty(layout, request, output_prefix)
    if zos_factory is None or pop_factory is None:
        default_zos, default_pop = _default_factories()
        zos_factory = default_zos if zos_factory is None else zos_factory
        pop_factory = default_pop if pop_factory is None else pop_factory

    zos: object | None = None
    oss: object | None = None
    wrapper: object | None = None
    analysis: object | None = None
    staged_path: Path | None = None
    staged_created = False
    pop_dir: Path | None = None
    captured: CapturedPopRun | None = None
    primary: BaseException | None = None
    primary_traceback = None
    cleanup_errors: list[str] = []

    try:
        zos = zos_factory()
        oss = zos.connect(mode="standalone")
        oss.load(str(layout.model_path))
        pop_dir = _resolve_pop_dir(oss)
        _reject_stale_outputs(pop_dir, output_prefix)
        staged_path = pop_dir / staged_input_name
        if staged_path.exists():
            raise ValueError(f"stale staged POP input exists: {staged_input_name}")
        staged_hash = _copy_staged_input_exclusive(
            source_path,
            staged_path,
            source_hash,
        )
        staged_created = True

        wrapper = pop_factory(
            **_pop_kwargs(
                request,
                staged_input_name=staged_input_name,
                output_prefix=output_prefix,
            )
        )
        try:
            result = _run_sustained_without_dataframe_unpack(wrapper, oss)
        finally:
            analysis = _active_analysis(wrapper)
        if analysis is None:
            raise RuntimeError("Sustain did not retain a live POP analysis")
        captured = _capture_connected_run(
            layout,
            request,
            source_input=request.input_artifact,
            source_header=source_header,
            source_hash=source_hash,
            staged_input_name=staged_input_name,
            output_prefix=output_prefix,
            pop_dir=pop_dir,
            analysis=analysis,
            result=result,
            zos=zos,
            oss=oss,
        )
    except BaseException as error:
        primary = error
        primary_traceback = error.__traceback__
    finally:
        if analysis is None:
            analysis = _active_analysis(wrapper)
        if analysis is not None:
            try:
                analysis.Close()
            except BaseException as error:
                cleanup_errors.append(_cleanup_error("close", error))
        if zos is not None:
            try:
                zos.disconnect()
            except BaseException as error:
                cleanup_errors.append(_cleanup_error("disconnect", error))
        if staged_created:
            try:
                if staged_path is None or pop_dir is None:
                    raise RuntimeError("POPDir unavailable during staged input cleanup")
                resolved = staged_path.resolve(strict=True)
                if not resolved.is_relative_to(pop_dir):
                    raise RuntimeError("staged input escaped POPDir")
                if hash_artifact(resolved) != source_hash:
                    raise RuntimeError("staged input changed before cleanup")
                resolved.unlink()
            except BaseException as error:
                cleanup_errors.append(_cleanup_error("staged_input", error))

    if primary is not None:
        if cleanup_errors:
            try:
                primary.runner_cleanup_errors = tuple(cleanup_errors)
            except BaseException:
                pass
        for cleanup_error in cleanup_errors:
            primary.add_note(f"runner cleanup error: {cleanup_error}")
        raise primary.with_traceback(primary_traceback)
    hard_cleanup = tuple(
        error
        for error in cleanup_errors
        if not error.startswith("staged_input:")
    )
    if hard_cleanup:
        raise RunnerCleanupError(tuple(cleanup_errors))
    if captured is None:
        raise RuntimeError("POP capture ended without a result")
    return replace(captured, cleanup_errors=tuple(cleanup_errors))


__all__ = [
    "CapturedMessage",
    "CapturedPopRun",
    "CapturedZbf",
    "RawDataGridSnapshot",
    "RunnerCleanupError",
    "SegmentPopRequest",
    "capture_raw_data_grid",
    "capture_segment_run",
    "expected_output_names",
]
