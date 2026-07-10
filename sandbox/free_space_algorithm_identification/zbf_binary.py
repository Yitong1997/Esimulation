"""Bit-preserving Zemax Beam File helpers for propagation diagnostics."""

from __future__ import annotations

import hashlib
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .models import UniformGrid2D


INT_COUNT = 9
DOUBLE_COUNT = 20
HEADER_BYTES = INT_COUNT * 4 + DOUBLE_COUNT * 8
NX_OFFSET = 4
NY_OFFSET = 8
DX_OFFSET = 36
DY_OFFSET = 44

_NAMED_INT_FIELDS = ("version", "nx", "ny", "is_polarized", "units")
_NAMED_DOUBLE_FIELDS = (
    "dx",
    "dy",
    "zx",
    "rx",
    "wx",
    "zy",
    "ry",
    "wy",
    "wavelength_vacuum_mm",
    "refractive_index",
    "receiver_efficiency",
    "system_efficiency",
)


def _double_from_bits(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


@dataclass(frozen=True)
class RawZbfHeader:
    """The exact ZBF header bytes plus decoded integer and IEEE-754 words."""

    raw_bytes: bytes
    int_words: tuple[int, ...]
    double_bits: tuple[int, ...]

    def __post_init__(self) -> None:
        raw_bytes = bytes(self.raw_bytes)
        int_words = tuple(int(word) for word in self.int_words)
        double_bits = tuple(int(word) for word in self.double_bits)
        if len(raw_bytes) != HEADER_BYTES:
            raise ValueError(f"ZBF header must be exactly {HEADER_BYTES} bytes")
        if len(int_words) != INT_COUNT or len(double_bits) != DOUBLE_COUNT:
            raise ValueError("ZBF header word counts do not match the binary layout")

        parsed_int_words = struct.unpack_from("<9i", raw_bytes, 0)
        parsed_double_bits = struct.unpack_from("<20Q", raw_bytes, INT_COUNT * 4)
        if int_words != parsed_int_words or double_bits != parsed_double_bits:
            raise ValueError("ZBF header words do not match raw_bytes")

        object.__setattr__(self, "raw_bytes", raw_bytes)
        object.__setattr__(self, "int_words", int_words)
        object.__setattr__(self, "double_bits", double_bits)
        self._validate_required_values()

    @classmethod
    def from_bytes(cls, raw_bytes: bytes) -> "RawZbfHeader":
        raw = bytes(raw_bytes)
        if len(raw) != HEADER_BYTES:
            raise ValueError(f"ZBF header must be exactly {HEADER_BYTES} bytes")
        return cls(
            raw_bytes=raw,
            int_words=struct.unpack_from("<9i", raw, 0),
            double_bits=struct.unpack_from("<20Q", raw, INT_COUNT * 4),
        )

    def _validate_required_values(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError(f"ZBF dimensions must be positive: nx={self.nx}, ny={self.ny}")
        if not math.isfinite(self.wavelength_vacuum_mm) or self.wavelength_vacuum_mm <= 0:
            raise ValueError("ZBF wavelength must be positive and finite")
        if not math.isfinite(self.refractive_index) or self.refractive_index <= 0:
            raise ValueError("ZBF refractive index must be positive and finite")

    def _double(self, index: int) -> float:
        return _double_from_bits(self.double_bits[index])

    @property
    def version(self) -> int:
        return self.int_words[0]

    @property
    def nx(self) -> int:
        return self.int_words[1]

    @property
    def ny(self) -> int:
        return self.int_words[2]

    @property
    def is_polarized(self) -> int:
        return self.int_words[3]

    @property
    def units(self) -> int:
        return self.int_words[4]

    @property
    def dx(self) -> float:
        return self._double(0)

    @property
    def dy(self) -> float:
        return self._double(1)

    @property
    def zx(self) -> float:
        return self._double(2)

    @property
    def rx(self) -> float:
        return self._double(3)

    @property
    def wx(self) -> float:
        return self._double(4)

    @property
    def zy(self) -> float:
        return self._double(5)

    @property
    def ry(self) -> float:
        return self._double(6)

    @property
    def wy(self) -> float:
        return self._double(7)

    @property
    def wavelength_vacuum_mm(self) -> float:
        return self._double(8)

    @property
    def refractive_index(self) -> float:
        return self._double(9)

    @property
    def receiver_efficiency(self) -> float:
        return self._double(10)

    @property
    def system_efficiency(self) -> float:
        return self._double(11)


@dataclass(frozen=True)
class HeaderDifference:
    changed_named_fields: tuple[str, ...]
    changed_reserved_int_indices: tuple[int, ...]
    changed_reserved_double_indices: tuple[int, ...]


@dataclass(frozen=True)
class LosslessZbf:
    path: Path | None
    source_sha256: str
    header: RawZbfHeader
    ex: np.ndarray
    ey: np.ndarray | None
    trailing_bytes: bytes

    def __post_init__(self) -> None:
        path = None if self.path is None else Path(self.path).resolve()
        ex = np.array(self.ex, dtype=np.complex128, order="C", copy=True)
        ey = (
            None
            if self.ey is None
            else np.array(self.ey, dtype=np.complex128, order="C", copy=True)
        )
        trailing_bytes = bytes(self.trailing_bytes)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "ex", ex)
        object.__setattr__(self, "ey", ey)
        object.__setattr__(self, "trailing_bytes", trailing_bytes)
        if path is None:
            digest = hashlib.sha256(
                _serialize_exact_bytes(self.header, ex, ey, trailing_bytes)
            ).hexdigest()
            object.__setattr__(self, "source_sha256", digest)

    @property
    def grid(self) -> UniformGrid2D:
        return UniformGrid2D.centered(
            nx=self.header.nx,
            ny=self.header.ny,
            dx_mm=self.header.dx,
            dy_mm=self.header.dy,
        )


def read_lossless_zbf(path: str | Path) -> LosslessZbf:
    """Read a ZBF while retaining every byte not represented by field arrays."""

    source_path = Path(path).resolve()
    source_digest = hashlib.sha256()
    with source_path.open("rb") as stream:
        raw_header = stream.read(HEADER_BYTES)
        source_digest.update(raw_header)
        header = RawZbfHeader.from_bytes(raw_header)
        field_payload_bytes = _checked_payload_bytes(
            nx=header.nx, ny=header.ny, label="Ex"
        )
        remaining_bytes = _remaining_file_bytes(stream)
        if remaining_bytes < field_payload_bytes:
            raise ValueError(
                f"Incomplete ZBF Ex payload: expected {field_payload_bytes} bytes, "
                f"got {remaining_bytes} in {source_path}"
            )
        if header.is_polarized and remaining_bytes < 2 * field_payload_bytes:
            ey_bytes = remaining_bytes - field_payload_bytes
            raise ValueError(
                f"Incomplete ZBF Ey payload: expected {field_payload_bytes} bytes, "
                f"got {ey_bytes} in {source_path}"
            )
        ex = _read_complex_payload(
            stream,
            nx=header.nx,
            ny=header.ny,
            expected_bytes=field_payload_bytes,
            path=source_path,
            label="Ex",
            digest=source_digest,
        )
        ey = (
            _read_complex_payload(
                stream,
                nx=header.nx,
                ny=header.ny,
                expected_bytes=field_payload_bytes,
                path=source_path,
                label="Ey",
                digest=source_digest,
            )
            if header.is_polarized
            else None
        )
        trailing_bytes = stream.read()
        source_digest.update(trailing_bytes)
    return LosslessZbf(
        path=source_path,
        source_sha256=source_digest.hexdigest(),
        header=header,
        ex=ex,
        ey=ey,
        trailing_bytes=trailing_bytes,
    )


def write_lossless_zbf(path: str | Path, beam: LosslessZbf) -> None:
    """Write a ZBF by reusing its exact header bytes and serialized tail."""

    ny, nx = beam.ex.shape
    if (nx, ny) != (beam.header.nx, beam.header.ny):
        raise ValueError("Ex shape does not match ZBF header")
    if beam.header.is_polarized and beam.ey is None:
        raise ValueError("polarized ZBF requires Ey")
    if not beam.header.is_polarized and beam.ey is not None:
        raise ValueError("unpolarized ZBF cannot contain Ey")
    if beam.ey is not None and beam.ey.shape != (ny, nx):
        raise ValueError("Ey shape does not match ZBF header")
    with Path(path).open("wb") as stream:
        stream.write(beam.header.raw_bytes)
        _write_complex_payload(stream, beam.ex)
        if beam.ey is not None:
            _write_complex_payload(stream, beam.ey)
        stream.write(beam.trailing_bytes)


def patch_sampling_header(
    header: RawZbfHeader, *, nx: int, ny: int, dx: float, dy: float
) -> RawZbfHeader:
    """Return a header with only the four sampling words patched."""

    raw = bytearray(header.raw_bytes)
    struct.pack_into("<i", raw, NX_OFFSET, int(nx))
    struct.pack_into("<i", raw, NY_OFFSET, int(ny))
    struct.pack_into("<d", raw, DX_OFFSET, float(dx))
    struct.pack_into("<d", raw, DY_OFFSET, float(dy))
    return RawZbfHeader.from_bytes(bytes(raw))


def compare_headers(before: RawZbfHeader, after: RawZbfHeader) -> HeaderDifference:
    """Report named and reserved word changes using exact stored word bits."""

    changed_named_fields = tuple(
        name
        for index, name in enumerate(_NAMED_INT_FIELDS)
        if before.int_words[index] != after.int_words[index]
    ) + tuple(
        name
        for index, name in enumerate(_NAMED_DOUBLE_FIELDS)
        if before.double_bits[index] != after.double_bits[index]
    )
    changed_reserved_int_indices = tuple(
        index
        for index, (left, right) in enumerate(
            zip(before.int_words[5:], after.int_words[5:], strict=True), start=5
        )
        if left != right
    )
    changed_reserved_double_indices = tuple(
        index
        for index, (left, right) in enumerate(
            zip(before.double_bits[12:], after.double_bits[12:], strict=True), start=12
        )
        if left != right
    )
    return HeaderDifference(
        changed_named_fields=changed_named_fields,
        changed_reserved_int_indices=changed_reserved_int_indices,
        changed_reserved_double_indices=changed_reserved_double_indices,
    )


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without retaining it in memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_complex_payload(
    stream: BinaryIO,
    *,
    nx: int,
    ny: int,
    expected_bytes: int,
    path: Path,
    label: str,
    digest: object,
) -> np.ndarray:
    raw = stream.read(expected_bytes)
    if len(raw) != expected_bytes:
        raise ValueError(
            f"Incomplete ZBF {label} payload: expected {expected_bytes} bytes, "
            f"got {len(raw)} in {path}"
        )
    digest.update(raw)
    field = np.frombuffer(raw, dtype="<c16").astype(np.complex128, copy=True)
    return field.reshape(ny, nx)


def _checked_payload_bytes(*, nx: int, ny: int, label: str) -> int:
    itemsize = np.dtype("<c16").itemsize
    if nx > sys.maxsize // ny:
        raise ValueError(f"ZBF {label} payload size exceeds the platform limit")
    sample_count = nx * ny
    if sample_count > sys.maxsize // itemsize:
        raise ValueError(f"ZBF {label} payload size exceeds the platform limit")
    return sample_count * itemsize


def _remaining_file_bytes(stream: BinaryIO) -> int:
    position = stream.tell()
    end_position = stream.seek(0, 2)
    stream.seek(position)
    return end_position - position


def _complex_payload_bytes(field: np.ndarray) -> bytes:
    native = np.asarray(field, dtype=np.complex128).reshape(-1)
    return native.astype("<c16", copy=False).tobytes(order="C")


def _write_complex_payload(stream: BinaryIO, field: np.ndarray) -> None:
    stream.write(_complex_payload_bytes(field))


def _serialize_exact_bytes(
    header: RawZbfHeader,
    ex: np.ndarray,
    ey: np.ndarray | None,
    trailing_bytes: bytes,
) -> bytes:
    payload = bytearray(header.raw_bytes)
    payload.extend(_complex_payload_bytes(ex))
    if ey is not None:
        payload.extend(_complex_payload_bytes(ey))
    payload.extend(trailing_bytes)
    return bytes(payload)


__all__ = [
    "DOUBLE_COUNT",
    "DX_OFFSET",
    "DY_OFFSET",
    "HEADER_BYTES",
    "INT_COUNT",
    "NX_OFFSET",
    "NY_OFFSET",
    "HeaderDifference",
    "LosslessZbf",
    "RawZbfHeader",
    "compare_headers",
    "patch_sampling_header",
    "read_lossless_zbf",
    "sha256_file",
    "write_lossless_zbf",
]
