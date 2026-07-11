"""Immutable run-local artifacts and append-only stage receipts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from numbers import Integral
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_COMPONENT_RE = re.compile(r"[A-Za-z0-9_.-]+\Z")
_RECEIPT_NAME_RE = re.compile(r"([0-9]{4})\.json\Z")
_SEGMENTS = ("S07_S08", "S12_S13", "S13_S14")
_ROOT_HASH_NAME = "hashes.sha256"
_RUN_INSTANCE_RE = re.compile(r"[0-9a-f]{32}\Z")
_RESERVED_WRITER_PATHS = frozenset({_ROOT_HASH_NAME, "model/hashes.sha256"})


def _safe_component(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_COMPONENT_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be one safe path component")
    if value in {".", ".."} or value.endswith("."):
        raise ValueError(f"{label} must be one safe path component")
    return value


def _logical_identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a nonempty logical identifier")
    parsed = value.strip()
    if (
        not parsed
        or "/" in parsed
        or "\\" in parsed
        or "\x00" in parsed
        or parsed in {".", ".."}
    ):
        raise ValueError(f"{label} must be a nonempty logical identifier")
    return parsed


def _canonical_relative_path(relative_path: object) -> str:
    if isinstance(relative_path, PurePosixPath):
        text = relative_path.as_posix()
    elif isinstance(relative_path, Path):
        text = relative_path.as_posix()
    elif isinstance(relative_path, str):
        text = relative_path
    else:
        raise ValueError("artifact path must be a run-relative POSIX path")
    if (
        not text
        or "\\" in text
        or "\x00" in text
        or text.startswith("/")
        or re.match(r"^[A-Za-z]:", text) is not None
    ):
        raise ValueError("artifact path must be a run-relative POSIX path")
    pure = PurePosixPath(text)
    if (
        pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != text
        or any(_SAFE_COMPONENT_RE.fullmatch(part) is None for part in pure.parts)
    ):
        raise ValueError("artifact path must be a run-relative POSIX path")
    return pure.as_posix()


@dataclass(frozen=True)
class ArtifactHash:
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.byte_count, bool)
            or not isinstance(self.byte_count, Integral)
            or int(self.byte_count) < 0
        ):
            raise ValueError("artifact byte count must be a non-negative integer")
        if not isinstance(self.sha256, str) or _SHA256_RE.fullmatch(self.sha256) is None:
            raise ValueError("artifact SHA-256 must be lowercase 64-hex")
        object.__setattr__(self, "byte_count", int(self.byte_count))


@dataclass(frozen=True)
class RunLayout:
    run_id: str
    run_instance_uuid: str
    run_dir: Path
    manifest_path: Path
    provenance_path: Path
    model_dir: Path
    model_path: Path
    cfg_path: Path
    model_hash_manifest_path: Path
    receipts_dir: Path
    continuous_dir: Path
    baselines_dir: Path
    candidates_dir: Path
    comparisons_dir: Path
    final_report_path: Path
    root_hash_manifest_path: Path

    def __post_init__(self) -> None:
        _safe_component(self.run_id, label="run_id")
        if (
            not isinstance(self.run_instance_uuid, str)
            or _RUN_INSTANCE_RE.fullmatch(self.run_instance_uuid) is None
        ):
            raise ValueError("run instance UUID must be lowercase 32-hex")
        run_dir = Path(self.run_dir).resolve()
        if not run_dir.is_dir():
            raise ValueError("run directory must exist")
        for name, value in asdict(self).items():
            if name in {"run_id", "run_instance_uuid", "run_dir"}:
                continue
            path = Path(value).resolve(strict=False)
            if not path.is_relative_to(run_dir):
                raise ValueError("run layout paths must remain inside the run directory")
        object.__setattr__(self, "run_dir", run_dir)


@dataclass(frozen=True)
class ArtifactRef:
    run_id: str
    run_instance_uuid: str
    producer_stage: str
    producer_case: str
    relative_path: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        run_id = _safe_component(self.run_id, label="artifact run_id")
        if (
            not isinstance(self.run_instance_uuid, str)
            or _RUN_INSTANCE_RE.fullmatch(self.run_instance_uuid) is None
        ):
            raise ValueError("artifact run instance UUID must be lowercase 32-hex")
        stage = _logical_identifier(self.producer_stage, label="producer stage")
        case = _logical_identifier(self.producer_case, label="producer case")
        relative_path = _canonical_relative_path(self.relative_path)
        digest = ArtifactHash(self.byte_count, self.sha256)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "producer_stage", stage)
        object.__setattr__(self, "producer_case", case)
        object.__setattr__(self, "relative_path", relative_path)
        object.__setattr__(self, "byte_count", digest.byte_count)
        object.__setattr__(self, "sha256", digest.sha256)

    @classmethod
    def from_file(
        cls,
        layout: RunLayout,
        relative_path: str,
        *,
        producer_stage: str,
        producer_case: str,
    ) -> "ArtifactRef":
        canonical = _canonical_relative_path(relative_path)
        path = _resolve_existing_run_file(layout, canonical)
        digest = hash_artifact(path)
        return cls(
            run_id=layout.run_id,
            run_instance_uuid=layout.run_instance_uuid,
            producer_stage=producer_stage,
            producer_case=producer_case,
            relative_path=canonical,
            byte_count=digest.byte_count,
            sha256=digest.sha256,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _require_layout(layout: object) -> RunLayout:
    if not isinstance(layout, RunLayout):
        raise ValueError("operation requires a RunLayout")
    return layout


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(stat.S_ISLNK(metadata.st_mode) or attributes & reparse_flag)


def _reject_link_chain(run: RunLayout, candidate: Path) -> None:
    lexical = Path(os.path.abspath(candidate))
    root = Path(os.path.abspath(run.run_dir))
    while lexical != root:
        if _is_link_or_reparse(lexical):
            raise ValueError("artifact paths cannot contain a link or reparse point")
        parent = lexical.parent
        if parent == lexical:
            raise ValueError("artifact path must remain inside the current run")
        lexical = parent


def _resolve_new_run_file(layout: RunLayout, relative_path: object) -> tuple[str, Path]:
    run = _require_layout(layout)
    canonical = _canonical_relative_path(relative_path)
    pure = PurePosixPath(canonical)
    destination = run.run_dir.joinpath(*pure.parts)
    _reject_link_chain(run, destination.parent)
    if not destination.parent.is_dir():
        raise ValueError("artifact parent directory must be predeclared in the run layout")
    resolved_parent = destination.parent.resolve(strict=True)
    if not resolved_parent.is_relative_to(run.run_dir):
        raise ValueError("artifact path must remain inside the current run")
    return canonical, resolved_parent / destination.name


def _resolve_existing_run_file(layout: RunLayout, relative_path: object) -> Path:
    run = _require_layout(layout)
    canonical = _canonical_relative_path(relative_path)
    candidate = run.run_dir.joinpath(*PurePosixPath(canonical).parts)
    _reject_link_chain(run, candidate)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError("referenced artifact does not exist") from exc
    if not resolved.is_relative_to(run.run_dir) or not resolved.is_file():
        raise ValueError("referenced artifact must be a current-run regular file")
    return resolved


def _guard_not_finalized(layout: RunLayout, *, destination_relative: str) -> None:
    if (
        layout.root_hash_manifest_path.exists()
        and destination_relative != _ROOT_HASH_NAME
    ):
        raise RuntimeError("run is finalized by its root hash manifest")


def _json_bytes(payload: object) -> bytes:
    try:
        text = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("artifact JSON must be finite and serializable") from exc
    return (text + "\n").encode("utf-8")


def _write_bytes_exclusive(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def write_json_once(
    layout: RunLayout, relative_path: str, payload: object
) -> Path:
    """Write canonical UTF-8 JSON once to a predeclared run directory."""

    run = _require_layout(layout)
    canonical, destination = _resolve_new_run_file(run, relative_path)
    if canonical in _RESERVED_WRITER_PATHS:
        raise ValueError("hash-manifest paths are reserved for the manifest writer")
    _guard_not_finalized(run, destination_relative=canonical)
    if canonical == "provenance.json":
        _validate_provenance_payload(run, payload)
    _write_bytes_exclusive(destination, _json_bytes(payload))
    return destination


def _validate_provenance_payload(layout: RunLayout, payload: object) -> None:
    """Reject incomplete live reproducibility metadata before any file exists."""

    _verify_model_hash_manifest(layout)
    if not isinstance(payload, Mapping):
        raise ValueError("provenance must be a mapping")
    required = {
        "artifact_hashes",
        "captured_utc",
        "conventions",
        "git",
        "host",
        "pop_sample_enums",
        "run_id",
        "run_instance_uuid",
        "versions",
    }
    if not required.issubset(payload):
        raise ValueError("provenance is missing required reproducibility metadata")
    if payload["run_id"] != layout.run_id:
        raise ValueError("provenance run_id does not match the run layout")
    if payload["run_instance_uuid"] != layout.run_instance_uuid:
        raise ValueError("provenance run instance does not match the run layout")

    versions = payload["versions"]
    version_keys = {"opticstudio", "zos_api", "zospy", "python", "numpy", "scipy"}
    if not isinstance(versions, Mapping) or not version_keys.issubset(versions) or any(
        not isinstance(versions[key], str) or not versions[key].strip()
        for key in version_keys
    ):
        raise ValueError("provenance versions are incomplete")

    git = payload["git"]
    if (
        not isinstance(git, Mapping)
        or not isinstance(git.get("commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", git["commit"]) is None
        or not isinstance(git.get("dirty_paths"), list)
        or any(not isinstance(path, str) for path in git["dirty_paths"])
    ):
        raise ValueError("provenance Git metadata is incomplete")

    host = payload["host"]
    if (
        not isinstance(host, Mapping)
        or not isinstance(host.get("timezone"), str)
        or not host["timezone"].strip()
        or not isinstance(host.get("cpu"), str)
        or not host["cpu"].strip()
        or isinstance(host.get("physical_memory_bytes"), bool)
        or not isinstance(host.get("physical_memory_bytes"), Integral)
        or int(host["physical_memory_bytes"]) <= 0
    ):
        raise ValueError("provenance host metadata is incomplete")
    _parse_utc_timestamp(payload["captured_utc"], label="provenance capture")

    artifact_hashes = payload["artifact_hashes"]
    if not isinstance(artifact_hashes, Mapping):
        raise ValueError("provenance artifact hashes are incomplete")
    expected_hashes = {
        "model_sha256": hash_artifact(layout.model_path).sha256,
        "cfg_sha256": hash_artifact(layout.cfg_path).sha256,
    }
    if any(artifact_hashes.get(key) != value for key, value in expected_hashes.items()):
        raise ValueError("provenance model or CFG hash does not match the run copy")
    input_hashes = artifact_hashes.get("canonical_input_zbf_sha256")
    if (
        not isinstance(input_hashes, Mapping)
        or set(input_hashes) != {"S7", "S12", "S13"}
        or any(
            not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
            for value in input_hashes.values()
        )
    ):
        raise ValueError("provenance canonical input ZBF hashes are incomplete")

    conventions = payload["conventions"]
    convention_keys = {
        "axis_order",
        "grid_center",
        "phasor",
        "polarization",
        "power",
        "reflection",
        "surface_axis_signs",
    }
    if (
        not isinstance(conventions, Mapping)
        or not convention_keys.issubset(conventions)
        or any(
            not isinstance(conventions[key], str) or not conventions[key].strip()
            for key in convention_keys - {"surface_axis_signs"}
        )
    ):
        raise ValueError("provenance physical conventions are incomplete")
    axis_signs = conventions["surface_axis_signs"]
    if axis_signs != {"S7": -1, "S8": -1, "S12": 1, "S13": 1, "S14": 1}:
        raise ValueError("provenance surface axis signs are invalid")

    sample_enums = payload["pop_sample_enums"]
    if (
        not isinstance(sample_enums, list)
        or not sample_enums
        or any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) < 2
            for value in sample_enums
        )
        or len(sample_enums) != len(set(sample_enums))
    ):
        raise ValueError("provenance POP sample enumerations are invalid")


def copy_file_once(
    layout: RunLayout, source: str | Path, relative_path: str
) -> Path:
    """Copy exact bytes once and verify the completed run-local file."""

    run = _require_layout(layout)
    source_path = Path(source).resolve(strict=True)
    if not source_path.is_file():
        raise ValueError("copy source must be a regular file")
    canonical, destination = _resolve_new_run_file(run, relative_path)
    if canonical in _RESERVED_WRITER_PATHS:
        raise ValueError("hash-manifest paths are reserved for the manifest writer")
    _guard_not_finalized(run, destination_relative=canonical)
    created = False
    expected_hash = hashlib.sha256()
    byte_count = 0
    try:
        with source_path.open("rb") as source_stream, destination.open("xb") as output:
            created = True
            while True:
                block = source_stream.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                expected_hash.update(block)
                byte_count += len(block)
            output.flush()
            os.fsync(output.fileno())
        actual = hash_artifact(destination)
        if actual.byte_count != byte_count or actual.sha256 != expected_hash.hexdigest():
            raise OSError("copied artifact does not match the bytes read from source")
    except Exception:
        if created:
            destination.unlink(missing_ok=True)
        raise
    return destination


def hash_artifact(path: str | Path) -> ArtifactHash:
    """Hash one regular file without loading it into memory."""

    artifact = Path(path).resolve(strict=True)
    if not artifact.is_file():
        raise ValueError("artifact must be a regular file")
    hasher = hashlib.sha256()
    byte_count = 0
    with artifact.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
            byte_count += len(block)
    return ArtifactHash(byte_count=byte_count, sha256=hasher.hexdigest())


def _normalized_case_matrix(
    case_matrix: Mapping[str, Iterable[str]],
) -> dict[str, list[str]]:
    if not isinstance(case_matrix, Mapping) or set(case_matrix) != set(_SEGMENTS):
        raise ValueError("case matrix must contain exactly the three frozen segments")
    normalized: dict[str, list[str]] = {}
    for segment in _SEGMENTS:
        raw_cases = tuple(case_matrix[segment])
        cases = [_safe_component(case, label=f"{segment} case") for case in raw_cases]
        if not cases or len(cases) != len(set(cases)):
            raise ValueError("each segment requires distinct predeclared cases")
        normalized[segment] = cases
    return normalized


def create_run_layout(
    run_root: str | Path,
    run_id: str,
    *,
    model_source: str | Path,
    cfg_source: str | Path,
    manifest_payload: Mapping[str, object],
    case_matrix: Mapping[str, Iterable[str]],
) -> RunLayout:
    """Exclusively initialize one immutable diagnostic run directory.

    An initialization failure deliberately leaves a fail-closed partial directory;
    that run ID is permanently burned and must not be retried or repaired in place.
    """

    run_name = _safe_component(run_id, label="run_id")
    model_source_path = Path(model_source).resolve(strict=True)
    cfg_source_path = Path(cfg_source).resolve(strict=True)
    if not model_source_path.is_file() or not cfg_source_path.is_file():
        raise ValueError("model and CFG sources must be regular files")
    if not isinstance(manifest_payload, Mapping):
        raise ValueError("manifest payload must be a mapping")
    reserved = {"format_version", "run_id", "run_instance_uuid", "case_matrix"}
    if reserved.intersection(manifest_payload):
        raise ValueError("manifest payload cannot override immutable run fields")
    if "planned_stage_graph" not in manifest_payload:
        raise ValueError("manifest requires the complete planned stage graph")
    cases = _normalized_case_matrix(case_matrix)
    run_instance_uuid = uuid.uuid4().hex
    manifest = dict(manifest_payload)
    manifest.update(
        {
            "case_matrix": cases,
            "format_version": 1,
            "run_id": run_name,
            "run_instance_uuid": run_instance_uuid,
        }
    )
    _json_bytes(manifest)

    root = Path(run_root).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError("run root must be a directory")
    run_dir = root / run_name
    run_dir.mkdir(exist_ok=False)

    model_dir = run_dir / "model"
    receipts_dir = run_dir / "receipts"
    continuous_dir = run_dir / "continuous"
    baselines_dir = run_dir / "baselines"
    candidates_dir = run_dir / "candidates"
    comparisons_dir = run_dir / "comparisons"
    for directory in (
        model_dir,
        receipts_dir,
        continuous_dir,
        baselines_dir,
        candidates_dir,
        comparisons_dir,
    ):
        directory.mkdir(exist_ok=False)
    for segment, segment_cases in cases.items():
        segment_dir = run_dir / segment
        segment_dir.mkdir(exist_ok=False)
        for case in segment_cases:
            case_dir = segment_dir / case
            case_dir.mkdir(exist_ok=False)
            for stage_dir in ("input", "identity", "propagation"):
                (case_dir / stage_dir).mkdir(exist_ok=False)

    layout = RunLayout(
        run_id=run_name,
        run_instance_uuid=run_instance_uuid,
        run_dir=run_dir,
        manifest_path=run_dir / "manifest.json",
        provenance_path=run_dir / "provenance.json",
        model_dir=model_dir,
        model_path=model_dir / "system.zmx",
        cfg_path=model_dir / "source_native.CFG",
        model_hash_manifest_path=model_dir / "hashes.sha256",
        receipts_dir=receipts_dir,
        continuous_dir=continuous_dir,
        baselines_dir=baselines_dir,
        candidates_dir=candidates_dir,
        comparisons_dir=comparisons_dir,
        final_report_path=run_dir / "final_report.md",
        root_hash_manifest_path=run_dir / _ROOT_HASH_NAME,
    )
    write_json_once(layout, "manifest.json", manifest)
    copy_file_once(layout, model_source_path, "model/system.zmx")
    copy_file_once(layout, cfg_source_path, "model/source_native.CFG")
    write_hash_manifest(
        layout,
        relative_path="model/hashes.sha256",
        include_relative_paths=("model/system.zmx", "model/source_native.CFG"),
    )
    return layout


def verify_artifact_ref(
    layout: RunLayout,
    reference: ArtifactRef,
    *,
    expected_producer_stage: str,
    expected_producer_case: str,
) -> Path:
    """Verify current-run identity, producer identity, size, and hash."""

    run = _require_layout(layout)
    if not isinstance(reference, ArtifactRef):
        raise ValueError("artifact reference must be an ArtifactRef")
    if reference.run_id != run.run_id:
        raise ValueError("artifact reference does not belong to the current run")
    if reference.run_instance_uuid != run.run_instance_uuid:
        raise ValueError("artifact reference does not belong to this run instance")
    expected_stage = _logical_identifier(
        expected_producer_stage, label="expected producer stage"
    )
    expected_case = _logical_identifier(
        expected_producer_case, label="expected producer case"
    )
    if reference.producer_stage != expected_stage:
        raise ValueError("artifact producer stage does not match")
    if reference.producer_case != expected_case:
        raise ValueError("artifact producer case does not match")
    path = _resolve_existing_run_file(run, reference.relative_path)
    digest = hash_artifact(path)
    if digest.byte_count != reference.byte_count:
        raise ValueError("artifact byte count does not match its reference")
    if digest.sha256 != reference.sha256:
        raise ValueError("artifact hash does not match its reference")
    return path


def _parse_utc_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp") from exc
    return parsed


def write_stage_receipt(
    layout: RunLayout,
    *,
    sequence: int,
    stage: str,
    producer_case: str,
    inputs: tuple[ArtifactRef, ...],
    outputs: tuple[ArtifactRef, ...],
    gate_values: Mapping[str, object],
    gate_status: str,
    started_utc: str,
    ended_utc: str,
    exception_text: str | None = None,
) -> ArtifactRef:
    """Append the next uniquely numbered immutable stage receipt."""

    run = _require_layout(layout)
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, Integral)
        or not 1 <= int(sequence) <= 9999
    ):
        raise ValueError("receipt sequence must be an integer from 1 through 9999")
    sequence = int(sequence)
    existing_sequences: list[int] = []
    for path in run.receipts_dir.glob("*.json"):
        match = _RECEIPT_NAME_RE.fullmatch(path.name)
        if match is None:
            raise ValueError("receipt directory contains a noncanonical JSON filename")
        existing_sequences.append(int(match.group(1)))
    if len(existing_sequences) != len(set(existing_sequences)):
        raise ValueError("receipt directory contains duplicate sequence numbers")
    ordered = sorted(existing_sequences)
    if ordered != list(range(1, len(ordered) + 1)):
        raise ValueError("receipt sequence history is not contiguous")
    expected_sequence = len(ordered) + 1
    if sequence != expected_sequence:
        raise ValueError(f"next receipt sequence must be {expected_sequence}")

    stage_id = _logical_identifier(stage, label="stage")
    case_id = _logical_identifier(producer_case, label="producer case")
    if gate_status not in {"passed", "failed"}:
        raise ValueError("gate status must be passed or failed")
    start = _parse_utc_timestamp(started_utc, label="receipt start")
    end = _parse_utc_timestamp(ended_utc, label="receipt end")
    if end < start:
        raise ValueError("receipt end timestamp cannot precede its start")
    if gate_status == "failed":
        if not isinstance(exception_text, str) or not exception_text.strip():
            raise ValueError("a failed receipt requires exception text")
    elif exception_text is not None:
        raise ValueError("a passed receipt cannot contain exception text")
    if not isinstance(gate_values, Mapping):
        raise ValueError("gate values must be a mapping")
    for reference in inputs:
        verify_artifact_ref(
            run,
            reference,
            expected_producer_stage=reference.producer_stage,
            expected_producer_case=reference.producer_case,
        )
    for reference in outputs:
        if (
            reference.producer_stage != stage_id
            or reference.producer_case != case_id
        ):
            raise ValueError(
                "receipt output producer must match the receipt stage and case"
            )
        verify_artifact_ref(
            run,
            reference,
            expected_producer_stage=stage_id,
            expected_producer_case=case_id,
        )
    payload = {
        "ended_utc": ended_utc,
        "exception_text": exception_text,
        "gate_status": gate_status,
        "gate_values": dict(gate_values),
        "inputs": [reference.to_dict() for reference in inputs],
        "outputs": [reference.to_dict() for reference in outputs],
        "producer_case": case_id,
        "run_id": run.run_id,
        "run_instance_uuid": run.run_instance_uuid,
        "sequence": sequence,
        "stage": stage_id,
        "started_utc": started_utc,
    }
    filename = f"{sequence:04d}.json"
    relative = f"receipts/{filename}"
    write_json_once(run, relative, payload)
    return ArtifactRef.from_file(
        run,
        relative,
        producer_stage=stage_id,
        producer_case=case_id,
    )


def _listed_run_files(
    layout: RunLayout,
    *,
    excluded_relative: str,
) -> tuple[str, ...]:
    relative_paths: list[str] = []
    for path in layout.run_dir.rglob("*"):
        if _is_link_or_reparse(path):
            raise ValueError("hash manifests do not permit links or reparse points")
        if not path.is_file():
            continue
        relative = path.relative_to(layout.run_dir).as_posix()
        canonical = _canonical_relative_path(relative)
        if canonical != excluded_relative:
            relative_paths.append(canonical)
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("run contains duplicate canonical artifact paths")
    return tuple(sorted(relative_paths))


def _verify_model_hash_manifest(layout: RunLayout) -> None:
    """Bind the run-local model and CFG to their initialization hashes."""

    try:
        manifest = _resolve_existing_run_file(layout, "model/hashes.sha256")
        lines = manifest.read_text(encoding="ascii").splitlines()
    except (ValueError, UnicodeError) as exc:
        raise ValueError("model hash manifest is missing or malformed") from exc
    expected_paths = ("model/source_native.CFG", "model/system.zmx")
    recorded: list[tuple[str, str]] = []
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError("model hash manifest is malformed")
        relative = _canonical_relative_path(match.group(2))
        recorded.append((relative, match.group(1)))
    if tuple(relative for relative, _ in recorded) != expected_paths:
        raise ValueError("model hash manifest path set is invalid")
    for relative, expected_sha256 in recorded:
        actual = hash_artifact(_resolve_existing_run_file(layout, relative))
        if actual.sha256 != expected_sha256:
            raise ValueError(
                "model hash manifest hash mismatch for a run-local copy"
            )


def _verify_receipt_history(layout: RunLayout) -> tuple[dict[str, object], ...]:
    entries = tuple(sorted(layout.receipts_dir.iterdir(), key=lambda path: path.name))
    if any(
        not path.is_file() or _RECEIPT_NAME_RE.fullmatch(path.name) is None
        for path in entries
    ):
        raise ValueError("receipt directory contains a noncanonical entry")
    if [path.name for path in entries] != [
        f"{sequence:04d}.json" for sequence in range(1, len(entries) + 1)
    ]:
        raise ValueError("receipt sequence history is not contiguous")
    payloads: list[dict[str, object]] = []
    for sequence, path in enumerate(entries, start=1):
        if _is_link_or_reparse(path):
            raise ValueError("receipt paths cannot contain a link or reparse point")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("stage receipt is malformed") from exc
        if not isinstance(payload, dict):
            raise ValueError("stage receipt must be a JSON object")
        if (
            payload.get("sequence") != sequence
            or payload.get("run_id") != layout.run_id
            or payload.get("run_instance_uuid") != layout.run_instance_uuid
        ):
            raise ValueError("stage receipt identity or sequence is invalid")
        stage_id = _logical_identifier(payload.get("stage"), label="receipt stage")
        case_id = _logical_identifier(
            payload.get("producer_case"), label="receipt producer case"
        )
        status = payload.get("gate_status")
        exception = payload.get("exception_text")
        if status not in {"passed", "failed"}:
            raise ValueError("stage receipt gate status is invalid")
        has_exception = isinstance(exception, str) and bool(exception.strip())
        if (status == "failed") != has_exception:
            raise ValueError("stage receipt exception semantics are invalid")
        start = _parse_utc_timestamp(payload.get("started_utc"), label="receipt start")
        end = _parse_utc_timestamp(payload.get("ended_utc"), label="receipt end")
        if end < start:
            raise ValueError("stage receipt timestamps are invalid")
        for role in ("inputs", "outputs"):
            raw_refs = payload.get(role)
            if not isinstance(raw_refs, list):
                raise ValueError("stage receipt artifact lists are invalid")
            for raw in raw_refs:
                if not isinstance(raw, dict):
                    raise ValueError("stage receipt artifact reference is invalid")
                try:
                    reference = ArtifactRef(**raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError("stage receipt artifact reference is invalid") from exc
                if role == "outputs" and (
                    reference.producer_stage != stage_id
                    or reference.producer_case != case_id
                ):
                    raise ValueError("stage receipt output producer is invalid")
                verify_artifact_ref(
                    layout,
                    reference,
                    expected_producer_stage=reference.producer_stage,
                    expected_producer_case=reference.producer_case,
                )
        payloads.append(payload)
    return tuple(payloads)


def _verify_finalization_prerequisites(layout: RunLayout) -> None:
    try:
        provenance_path = _resolve_existing_run_file(layout, "provenance.json")
    except ValueError as exc:
        raise ValueError("provenance is required before finalization") from exc
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("provenance is missing or malformed") from exc
    _validate_provenance_payload(layout, provenance)
    try:
        final_report = _resolve_existing_run_file(layout, "final_report.md")
    except ValueError as exc:
        raise ValueError("final report is required before finalization") from exc
    receipts = _verify_receipt_history(layout)
    if not receipts:
        raise ValueError("final report requires a passed report receipt")
    last = receipts[-1]
    if last.get("stage") != "report" or last.get("gate_status") != "passed":
        raise ValueError("final report must be the output of the last passed receipt")
    report_refs = []
    for raw in last.get("outputs", []):
        reference = ArtifactRef(**raw)
        if reference.relative_path == "final_report.md":
            report_refs.append(reference)
    if len(report_refs) != 1:
        raise ValueError("final report receipt must bind exactly one final report")
    verified = verify_artifact_ref(
        layout,
        report_refs[0],
        expected_producer_stage="report",
        expected_producer_case=str(last["producer_case"]),
    )
    if verified != final_report:
        raise ValueError("final report receipt points to the wrong artifact")


def write_hash_manifest(
    layout: RunLayout,
    *,
    relative_path: str = _ROOT_HASH_NAME,
    include_relative_paths: Iterable[str] | None = None,
) -> Path:
    """Write a sorted SHA-256 manifest once, excluding the manifest itself."""

    run = _require_layout(layout)
    output_relative, output = _resolve_new_run_file(run, relative_path)
    _guard_not_finalized(run, destination_relative=output_relative)
    if output_relative == _ROOT_HASH_NAME:
        if include_relative_paths is not None:
            raise ValueError("the complete root hash manifest cannot use a subset")
        _verify_model_hash_manifest(run)
        _verify_finalization_prerequisites(run)
    if include_relative_paths is None:
        paths = _listed_run_files(run, excluded_relative=output_relative)
    else:
        paths = tuple(
            sorted({_canonical_relative_path(path) for path in include_relative_paths})
        )
        if output_relative in paths:
            raise ValueError("a hash manifest cannot include itself")
    lines: list[str] = []
    for relative in paths:
        artifact = _resolve_existing_run_file(run, relative)
        lines.append(f"{hash_artifact(artifact).sha256}  {relative}\n")
    _write_bytes_exclusive(output, "".join(lines).encode("ascii"))
    return output


def verify_hash_manifest(layout: RunLayout) -> tuple[str, ...]:
    """Verify the complete root manifest, including absence of added files."""

    run = _require_layout(layout)
    _verify_model_hash_manifest(run)
    _verify_finalization_prerequisites(run)
    manifest = _resolve_existing_run_file(run, _ROOT_HASH_NAME)
    try:
        lines = manifest.read_text(encoding="ascii").splitlines()
    except UnicodeError as exc:
        raise ValueError("hash manifest must be ASCII") from exc
    recorded: list[tuple[str, str]] = []
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError("hash manifest contains a malformed line")
        relative = _canonical_relative_path(match.group(2))
        if relative == _ROOT_HASH_NAME:
            raise ValueError("root hash manifest cannot include itself")
        recorded.append((relative, match.group(1)))
    recorded_paths = [relative for relative, _ in recorded]
    if recorded_paths != sorted(recorded_paths) or len(recorded_paths) != len(
        set(recorded_paths)
    ):
        raise ValueError("hash manifest paths must be sorted and unique")
    actual_paths = _listed_run_files(run, excluded_relative=_ROOT_HASH_NAME)
    if tuple(recorded_paths) != actual_paths:
        raise ValueError("hash manifest path set does not match the run")
    for relative, expected_sha256 in recorded:
        actual = hash_artifact(_resolve_existing_run_file(run, relative))
        if actual.sha256 != expected_sha256:
            raise ValueError(f"hash mismatch for {relative}")
    return tuple(recorded_paths)


__all__ = [
    "ArtifactHash",
    "ArtifactRef",
    "RunLayout",
    "copy_file_once",
    "create_run_layout",
    "hash_artifact",
    "verify_artifact_ref",
    "verify_hash_manifest",
    "write_hash_manifest",
    "write_json_once",
    "write_stage_receipt",
]
