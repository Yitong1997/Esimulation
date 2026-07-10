from __future__ import annotations

import hashlib
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sandbox.free_space_algorithm_identification import zbf_binary as zbf_binary_module
from sandbox.free_space_algorithm_identification.models import UniformGrid2D
from sandbox.free_space_algorithm_identification.zbf_binary import (
    HEADER_BYTES,
    LosslessZbf,
    RawZbfHeader,
    compare_headers,
    patch_sampling_header,
    read_lossless_zbf,
    sha256_file,
    write_lossless_zbf,
)


def _double_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def write_synthetic_lossless_fixture(
    path: Path,
    *,
    reserved_ints: tuple[int, int, int, int] = (0, 0, 0, 0),
    reserved_double_bits: tuple[int, int, int, int, int, int, int, int] = (0,) * 8,
    trailing: bytes = b"",
    polarized: bool = False,
) -> Path:
    """Write an oracle ZBF using only explicit struct-packed words."""

    int_words = (17, 2, 2, int(polarized), 7, *reserved_ints)
    named_double_bits = tuple(
        _double_bits(value)
        for value in (
            0.5,
            0.75,
            -3.5,
            4.5,
            5.5,
            6.5,
            7.5,
            8.5,
            0.0006328,
            1.25,
            0.0,
            0.9,
        )
    )
    ex_payload_bits = (
        _double_bits(1.0),
        0x8000000000000000,
        0x7FF8000000000077,
        _double_bits(2.0),
        _double_bits(-3.25),
        _double_bits(4.5),
        0x0000000000000000,
        0x8000000000000000,
    )
    ey_payload_bits = (
        _double_bits(-1.0),
        _double_bits(0.25),
        _double_bits(2.5),
        _double_bits(-3.5),
        _double_bits(4.75),
        _double_bits(5.25),
        _double_bits(-6.5),
        _double_bits(7.5),
    )

    raw = bytearray()
    raw.extend(struct.pack("<9i", *int_words))
    raw.extend(struct.pack("<20Q", *(named_double_bits + reserved_double_bits)))
    raw.extend(struct.pack("<8Q", *ex_payload_bits))
    if polarized:
        raw.extend(struct.pack("<8Q", *ey_payload_bits))
    raw.extend(trailing)
    path.write_bytes(bytes(raw))
    return path


def test_roundtrip_preserves_all_unmodified_header_bits_and_trailing_bytes(
    tmp_path: Path,
) -> None:
    source = write_synthetic_lossless_fixture(
        tmp_path / "source.ZBF",
        reserved_ints=(11, 22, 33, 44),
        reserved_double_bits=(0x8000000000000000, 0x7FF8000000000042)
        + (0,) * 6,
        trailing=b"BTS-TAIL",
        polarized=True,
    )

    beam = read_lossless_zbf(source)
    output = tmp_path / "roundtrip.ZBF"
    write_lossless_zbf(output, beam)

    assert output.read_bytes() == source.read_bytes()
    assert beam.path == source.resolve()
    expected_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    assert beam.source_sha256 == expected_sha256
    assert sha256_file(source) == expected_sha256
    assert beam.header.raw_bytes == source.read_bytes()[:HEADER_BYTES]
    assert beam.trailing_bytes == b"BTS-TAIL"
    assert beam.ex.dtype == np.dtype(np.complex128)
    assert beam.ex.dtype.isnative
    assert beam.ey is not None
    assert beam.ey.dtype == np.dtype(np.complex128)
    assert beam.ey.dtype.isnative


def test_header_exposes_exact_semantic_mapping_and_registered_grid(tmp_path: Path) -> None:
    beam = read_lossless_zbf(write_synthetic_lossless_fixture(tmp_path / "source.ZBF"))
    header = beam.header

    assert header.version == 17
    assert header.nx == 2
    assert header.ny == 2
    assert header.is_polarized == 0
    assert header.units == 7
    assert header.dx == 0.5
    assert header.dy == 0.75
    assert header.zx == -3.5
    assert header.rx == 4.5
    assert header.wx == 5.5
    assert header.zy == 6.5
    assert header.ry == 7.5
    assert header.wy == 8.5
    assert header.wavelength_vacuum_mm == 0.0006328
    assert header.refractive_index == 1.25
    assert header.receiver_efficiency == 0.0
    assert header.system_efficiency == 0.9
    assert isinstance(beam.grid, UniformGrid2D)
    assert beam.grid.x_mm.tolist() == [-0.5, 0.0]
    assert beam.grid.y_mm.tolist() == [-0.75, 0.0]


def test_patch_sampling_changes_only_nx_ny_dx_dy(tmp_path: Path) -> None:
    source = read_lossless_zbf(write_synthetic_lossless_fixture(tmp_path / "source.ZBF"))

    patched = patch_sampling_header(source.header, nx=16, ny=16, dx=0.125, dy=0.25)
    diff = compare_headers(source.header, patched)

    assert diff.changed_named_fields == ("nx", "ny", "dx", "dy")
    assert diff.changed_reserved_int_indices == ()
    assert diff.changed_reserved_double_indices == ()
    assert patched.nx == 16
    assert patched.ny == 16
    assert patched.dx == 0.125
    assert patched.dy == 0.25
    assert source.header.nx == 2
    assert source.header.ny == 2
    changed_byte_indices = {
        index
        for index, (before, after) in enumerate(
            zip(source.header.raw_bytes, patched.raw_bytes, strict=True)
        )
        if before != after
    }
    allowed_byte_indices = set(range(4, 12)) | set(range(36, 52))
    assert changed_byte_indices <= allowed_byte_indices


def test_compare_headers_uses_ieee_bits_and_reports_reserved_word_indices(
    tmp_path: Path,
) -> None:
    source = read_lossless_zbf(write_synthetic_lossless_fixture(tmp_path / "source.ZBF"))
    changed_raw = bytearray(source.header.raw_bytes)
    struct.pack_into("<Q", changed_raw, 36 + 10 * 8, 0x8000000000000000)
    struct.pack_into("<i", changed_raw, 4 * (5 + 1), 123)
    struct.pack_into("<Q", changed_raw, 36 + (12 + 1) * 8, 0x7FF8000000000042)
    changed = RawZbfHeader.from_bytes(bytes(changed_raw))

    diff = compare_headers(source.header, changed)

    assert diff.changed_named_fields == ("receiver_efficiency",)
    assert diff.changed_reserved_int_indices == (6,)
    assert diff.changed_reserved_double_indices == (13,)


@pytest.mark.parametrize(
    ("offset", "fmt", "value", "message"),
    [
        (4, "<i", 0, "dimensions"),
        (8, "<i", -1, "dimensions"),
        (36 + 8 * 8, "<d", 0.0, "wavelength"),
        (36 + 8 * 8, "<Q", 0x7FF8000000000042, "wavelength"),
        (36 + 9 * 8, "<d", -1.0, "refractive index"),
    ],
)
def test_header_rejects_nonpositive_or_nonfinite_required_values(
    tmp_path: Path, offset: int, fmt: str, value: int | float, message: str
) -> None:
    source = write_synthetic_lossless_fixture(tmp_path / "source.ZBF")
    raw = bytearray(source.read_bytes()[:HEADER_BYTES])
    struct.pack_into(fmt, raw, offset, value)

    with pytest.raises(ValueError, match=message):
        RawZbfHeader.from_bytes(bytes(raw))


def test_header_and_reader_reject_truncated_input(tmp_path: Path) -> None:
    source = write_synthetic_lossless_fixture(tmp_path / "source.ZBF")
    raw = source.read_bytes()

    with pytest.raises(ValueError, match="header"):
        RawZbfHeader.from_bytes(raw[: HEADER_BYTES - 1])

    truncated = tmp_path / "truncated.ZBF"
    truncated.write_bytes(raw[:-1])
    with pytest.raises(ValueError, match="Ex payload"):
        read_lossless_zbf(truncated)


def test_reader_rejects_impossible_declared_payload_before_platform_overflow(
    tmp_path: Path,
) -> None:
    source = write_synthetic_lossless_fixture(tmp_path / "source.ZBF")
    raw_header = bytearray(source.read_bytes()[:HEADER_BYTES])
    struct.pack_into("<i", raw_header, 4, 2_147_483_647)
    struct.pack_into("<i", raw_header, 8, 2_147_483_647)
    impossible = tmp_path / "impossible.ZBF"
    impossible.write_bytes(bytes(raw_header))

    with pytest.raises(ValueError, match="Ex payload"):
        read_lossless_zbf(impossible)


def test_reader_hashes_the_exact_open_snapshot_without_reopening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_synthetic_lossless_fixture(
        tmp_path / "source.ZBF", polarized=True, trailing=b"SNAPSHOT"
    )
    expected_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(zbf_binary_module, "sha256_file", lambda _path: "0" * 64)

    beam = read_lossless_zbf(source)

    assert beam.source_sha256 == expected_sha256


def test_write_rejects_shape_and_polarization_mismatches(tmp_path: Path) -> None:
    unpolarized = read_lossless_zbf(
        write_synthetic_lossless_fixture(tmp_path / "unpolarized.ZBF")
    )
    polarized = read_lossless_zbf(
        write_synthetic_lossless_fixture(tmp_path / "polarized.ZBF", polarized=True)
    )

    with pytest.raises(ValueError, match="Ex shape does not match ZBF header"):
        write_lossless_zbf(
            tmp_path / "bad-shape.ZBF",
            replace(unpolarized, ex=np.ones((1, 2), dtype=np.complex128)),
        )
    with pytest.raises(ValueError, match="polarized ZBF requires Ey"):
        write_lossless_zbf(tmp_path / "missing-ey.ZBF", replace(polarized, ey=None))
    with pytest.raises(ValueError, match="Ey shape does not match ZBF header"):
        write_lossless_zbf(
            tmp_path / "bad-ey-shape.ZBF",
            replace(polarized, ey=np.ones((1, 2), dtype=np.complex128)),
        )
    with pytest.raises(ValueError, match="unpolarized ZBF cannot contain Ey"):
        write_lossless_zbf(
            tmp_path / "unexpected-ey.ZBF",
            replace(unpolarized, ey=np.ones((2, 2), dtype=np.complex128)),
        )


def test_in_memory_beam_hashes_exact_serialization_without_a_path(tmp_path: Path) -> None:
    source = write_synthetic_lossless_fixture(
        tmp_path / "source.ZBF", polarized=True, trailing=b"TAIL"
    )
    disk_beam = read_lossless_zbf(source)

    memory_beam = LosslessZbf(
        path=None,
        source_sha256="",
        header=disk_beam.header,
        ex=disk_beam.ex,
        ey=disk_beam.ey,
        trailing_bytes=disk_beam.trailing_bytes,
    )

    assert memory_beam.path is None
    assert memory_beam.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
