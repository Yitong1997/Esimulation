# Zemax POP Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reproducible benchmark that compares the local POP API with Zemax POP through ZOSAPI for direct Gaussian input and non-ideal ZBF input.

**Architecture:** Add source/io and benchmark layers around the existing POP propagation path. Implement ZBF parsing and a `ZbfSource` adapter first, then add shared comparison utilities, a local POP runner, a ZOSAPI runner, a manual benchmark entry point, and opt-in pytest benchmark tests.

**Tech Stack:** Python, NumPy, PROPER, pytest, ZOSPy/ZOSAPI, existing `pop` package APIs.

---

### Task 0: Prepare Execution Branch

**Files:**
- No source files.

**Step 1: Check worktree status**

Run:

```powershell
git status --short
```

Expected: existing unrelated dirty files may be present. Do not modify or stage them.

**Step 2: Create an implementation branch**

Run:

```powershell
git switch -c codex/zemax-pop-benchmark
```

Expected: branch switches successfully.

**Step 3: Commit boundary**

No commit for this task.

---

### Task 1: Add Package-Local ZBF I/O Tests

**Files:**
- Create: `tests/test_zbf_io.py`
- Create later: `pop/io/zbf.py`
- Modify later: `pop/io/__init__.py`

**Step 1: Write failing tests**

Create `tests/test_zbf_io.py` with synthetic ZBF helpers and these tests:

```python
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from pop.io.zbf import ZbfField, read_zbf, write_zbf, zbf_reference_phase


def _write_minimal_zbf(
    path: Path,
    ex: np.ndarray,
    *,
    dx: float = 0.25,
    dy: float = 0.25,
    zx: float = 2.0,
    zy: float = 2.0,
    rayleigh: float = 10.0,
    waist: float = 1.5,
    wavelength: float = 0.01,
    index: float = 1.0,
    polarized: bool = False,
) -> None:
    ny, nx = ex.shape
    ints = [1, nx, ny, int(polarized), 0, 0, 0, 0, 0]
    dbls = [
        dx, dy,
        zx, rayleigh, waist,
        zy, rayleigh, waist,
        wavelength, index,
        0.0, 0.0,
        *([0.0] * 8),
    ]
    flat = ex.reshape(-1)
    interleaved = np.empty(2 * flat.size, dtype="<f8")
    interleaved[0::2] = flat.real
    interleaved[1::2] = flat.imag
    with path.open("wb") as f:
        f.write(struct.pack("<9i", *ints))
        f.write(struct.pack("<20d", *dbls))
        f.write(interleaved.tobytes())
        if polarized:
            f.write(interleaved.tobytes())


def test_read_zbf_preserves_complex_field_and_header(tmp_path: Path) -> None:
    ex = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex128)
    path = tmp_path / "known.ZBF"
    _write_minimal_zbf(path, ex, dx=0.125, dy=0.25)

    zbf = read_zbf(path)

    assert isinstance(zbf, ZbfField)
    assert zbf.nx == 2
    assert zbf.ny == 2
    assert zbf.dx == 0.125
    assert zbf.dy == 0.25
    assert zbf.wavelength == 0.01
    np.testing.assert_array_equal(zbf.ex, ex)
    np.testing.assert_allclose(zbf.amplitude, np.abs(ex))


def test_write_zbf_round_trips(tmp_path: Path) -> None:
    ex = np.array([[1 + 0j, 0 + 1j], [2 - 1j, -3 + 0.5j]], dtype=np.complex128)
    zbf = ZbfField(
        path=None,
        version=1,
        nx=2,
        ny=2,
        is_polarized=0,
        units=0,
        dx=0.5,
        dy=0.5,
        zx=0.0,
        rx=0.0,
        wx=1.0,
        zy=0.0,
        ry=0.0,
        wy=1.0,
        wavelength=0.01,
        index=1.0,
        receiver_efficiency=0.0,
        system_efficiency=0.0,
        ex=ex,
        ey=None,
    )
    path = tmp_path / "roundtrip.ZBF"

    write_zbf(path, zbf)
    actual = read_zbf(path)

    np.testing.assert_array_equal(actual.ex, ex)
    assert actual.dx == 0.5


def test_reference_phase_uses_spherical_header_metadata(tmp_path: Path) -> None:
    ex = np.ones((3, 3), dtype=np.complex128)
    path = tmp_path / "ref.ZBF"
    _write_minimal_zbf(path, ex, dx=0.5, dy=0.5, zx=2.0, zy=2.0, rayleigh=4.0)
    zbf = read_zbf(path)

    phase = zbf_reference_phase(zbf)

    xg, yg = np.meshgrid(zbf.x_coords, zbf.y_coords)
    radius = zbf.zx * (1.0 + (zbf.rx / zbf.zx) ** 2)
    expected = (2.0 * np.pi * zbf.index / zbf.wavelength) * (
        np.sqrt(radius**2 + xg**2 + yg**2) - abs(radius)
    )
    np.testing.assert_allclose(phase, expected)


def test_read_zbf_marks_polarized_input(tmp_path: Path) -> None:
    path = tmp_path / "pol.ZBF"
    _write_minimal_zbf(path, np.ones((2, 2), dtype=np.complex128), polarized=True)

    zbf = read_zbf(path)

    assert zbf.is_polarized == 1
    assert zbf.ey is not None
```

**Step 2: Run tests to verify failure**

Run:

```powershell
pytest tests/test_zbf_io.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pop.io.zbf'`.

**Step 3: Commit**

Do not commit until Task 2 implements the module and tests pass.

---

### Task 2: Implement ZBF I/O and Reference Phase

**Files:**
- Create: `pop/io/zbf.py`
- Modify: `pop/io/__init__.py`
- Test: `tests/test_zbf_io.py`

**Step 1: Add implementation**

Create `pop/io/zbf.py` with:

```python
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

N_HEADER_INTS = 9
N_HEADER_DBLS = 20


@dataclass
class ZbfField:
    path: Optional[Path]
    version: int
    nx: int
    ny: int
    is_polarized: int
    units: int
    dx: float
    dy: float
    zx: float
    rx: float
    wx: float
    zy: float
    ry: float
    wy: float
    wavelength: float
    index: float
    receiver_efficiency: float
    system_efficiency: float
    ex: np.ndarray
    ey: np.ndarray | None = None

    @property
    def x_coords(self) -> np.ndarray:
        return (np.arange(self.nx) - self.nx / 2.0 + 0.5) * self.dx

    @property
    def y_coords(self) -> np.ndarray:
        return (np.arange(self.ny) - self.ny / 2.0 + 0.5) * self.dy

    @property
    def amplitude(self) -> np.ndarray:
        amp_sq = np.abs(self.ex) ** 2
        if self.ey is not None:
            amp_sq = amp_sq + np.abs(self.ey) ** 2
        return np.sqrt(amp_sq)

    @property
    def physical_field(self) -> np.ndarray:
        return self.ex * np.exp(1j * zbf_reference_phase(self))


def read_zbf(path: str | Path) -> ZbfField:
    path = Path(path)
    with path.open("rb") as f:
        ints_raw = f.read(N_HEADER_INTS * 4)
        if len(ints_raw) != N_HEADER_INTS * 4:
            raise ValueError(f"Incomplete ZBF integer header: {path}")
        ints = struct.unpack("<9i", ints_raw)
        version, nx, ny, is_polarized, units = ints[:5]
        if nx <= 0 or ny <= 0:
            raise ValueError(f"Invalid ZBF dimensions: nx={nx}, ny={ny}")

        dbls_raw = f.read(N_HEADER_DBLS * 8)
        if len(dbls_raw) != N_HEADER_DBLS * 8:
            raise ValueError(f"Incomplete ZBF double header: {path}")
        dbls = struct.unpack("<20d", dbls_raw)

        ex = _read_complex_grid(f, nx, ny, path, "Ex")
        ey = _read_complex_grid(f, nx, ny, path, "Ey") if is_polarized else None

    return ZbfField(
        path=path,
        version=version,
        nx=nx,
        ny=ny,
        is_polarized=is_polarized,
        units=units,
        dx=float(dbls[0]),
        dy=float(dbls[1]),
        zx=float(dbls[2]),
        rx=float(dbls[3]),
        wx=float(dbls[4]),
        zy=float(dbls[5]),
        ry=float(dbls[6]),
        wy=float(dbls[7]),
        wavelength=float(dbls[8]),
        index=float(dbls[9]),
        receiver_efficiency=float(dbls[10]),
        system_efficiency=float(dbls[11]),
        ex=ex,
        ey=ey,
    )


def write_zbf(path: str | Path, zbf: ZbfField) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if zbf.ex.shape != (zbf.ny, zbf.nx):
        raise ValueError(f"Ex shape {zbf.ex.shape} does not match {(zbf.ny, zbf.nx)}")
    with path.open("wb") as f:
        f.write(struct.pack("<9i", zbf.version, zbf.nx, zbf.ny, zbf.is_polarized, zbf.units, 0, 0, 0, 0))
        f.write(struct.pack(
            "<20d",
            zbf.dx, zbf.dy,
            zbf.zx, zbf.rx, zbf.wx,
            zbf.zy, zbf.ry, zbf.wy,
            zbf.wavelength, zbf.index,
            zbf.receiver_efficiency, zbf.system_efficiency,
            *([0.0] * 8),
        ))
        _write_complex_grid(f, zbf.ex)
        if zbf.is_polarized:
            if zbf.ey is None:
                raise ValueError("Polarized ZBF requires ey")
            _write_complex_grid(f, zbf.ey)


def zbf_reference_phase(zbf: ZbfField) -> np.ndarray:
    x_grid, y_grid = np.meshgrid(zbf.x_coords, zbf.y_coords)
    phase = np.zeros((zbf.ny, zbf.nx), dtype=np.float64)
    if zbf.wavelength <= 0:
        return phase
    rcx = _curvature_radius(zbf.zx, zbf.rx)
    rcy = _curvature_radius(zbf.zy, zbf.ry)
    k = 2.0 * np.pi * zbf.index / zbf.wavelength
    if np.isfinite(rcx) and np.isfinite(rcy) and np.isclose(rcx, rcy, rtol=5e-8, atol=1e-8):
        return k * _signed_spherical_opd(x_grid**2 + y_grid**2, 0.5 * (rcx + rcy))
    if np.isfinite(rcx):
        phase += k * _signed_spherical_opd(x_grid**2, rcx)
    if np.isfinite(rcy):
        phase += k * _signed_spherical_opd(y_grid**2, rcy)
    return phase


def _read_complex_grid(f, nx: int, ny: int, path: Path, label: str) -> np.ndarray:
    raw = f.read(nx * ny * 2 * 8)
    if len(raw) != nx * ny * 2 * 8:
        raise ValueError(f"Incomplete ZBF {label} data: {path}")
    pairs = np.frombuffer(raw, dtype="<f8")
    return (pairs[0::2] + 1j * pairs[1::2]).reshape(ny, nx)


def _write_complex_grid(f, field: np.ndarray) -> None:
    flat = np.asarray(field, dtype=np.complex128).reshape(-1)
    pairs = np.empty(2 * flat.size, dtype="<f8")
    pairs[0::2] = flat.real
    pairs[1::2] = flat.imag
    f.write(pairs.tobytes())


def _curvature_radius(waist_position: float, rayleigh_range: float) -> float:
    z = float(waist_position)
    if abs(z) < 1e-15:
        return float("inf")
    zr = float(rayleigh_range)
    return z * (1.0 + (zr / z) ** 2)


def _signed_spherical_opd(transverse_sq: np.ndarray, radius: float) -> np.ndarray:
    if not np.isfinite(radius) or abs(radius) < 1e-15:
        return np.zeros_like(transverse_sq, dtype=np.float64)
    abs_radius = abs(radius)
    return np.sign(radius) * (np.sqrt(abs_radius**2 + transverse_sq) - abs_radius)
```

Modify `pop/io/__init__.py`:

```python
from .zbf import ZbfField, read_zbf, write_zbf, zbf_reference_phase

__all__ = [
    "load_zmx",
    "to_optiland",
    "GlobalSurfaceDefinition",
    "ZbfField",
    "read_zbf",
    "write_zbf",
    "zbf_reference_phase",
]
```

**Step 2: Run tests**

Run:

```powershell
pytest tests/test_zbf_io.py -v
```

Expected: PASS.

**Step 3: Commit**

Run:

```powershell
git add pop/io/zbf.py pop/io/__init__.py tests/test_zbf_io.py
git commit -m "feat: add zbf io utilities"
```

Expected: commit succeeds with only these files.

---

### Task 3: Add ZBF Source Adapter Tests

**Files:**
- Create: `tests/test_zbf_source.py`
- Modify later: `pop/source.py`
- Modify later: `pop/__init__.py`

**Step 1: Write failing tests**

Create `tests/test_zbf_source.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import proper
from pop import ZbfSource
from pop.io.zbf import ZbfField, write_zbf, zbf_reference_phase


def _field(ex: np.ndarray, *, zx: float = 0.0, rayleigh: float = 10.0, waist: float = 2.0) -> ZbfField:
    ny, nx = ex.shape
    return ZbfField(
        path=None,
        version=1,
        nx=nx,
        ny=ny,
        is_polarized=0,
        units=0,
        dx=0.25,
        dy=0.25,
        zx=zx,
        rx=rayleigh,
        wx=waist,
        zy=zx,
        ry=rayleigh,
        wy=waist,
        wavelength=0.01,
        index=1.0,
        receiver_efficiency=0.0,
        system_efficiency=0.0,
        ex=ex,
        ey=None,
    )


def test_zbf_source_reference_relative_writes_ex_to_proper_wfarr(tmp_path: Path) -> None:
    ex = np.array([[1 + 0j, 0.5 + 0.25j], [0.25 - 0.5j, -1 + 0j]], dtype=np.complex128)
    zbf = _field(ex, zx=0.0)
    path = tmp_path / "input.ZBF"
    write_zbf(path, zbf)

    source = ZbfSource(path, reference_mode="reference_relative")
    amplitude, phase, pilot, wfo = source.create_initial_wavefront()

    np.testing.assert_allclose(amplitude, np.abs(ex))
    np.testing.assert_allclose(phase, np.angle(ex))
    np.testing.assert_allclose(proper.prop_shift_center(wfo.wfarr), ex)
    assert pilot.wavelength_um == pytest.approx(10.0)
    assert pilot.waist_radius_mm == pytest.approx(2.0)


def test_zbf_source_physical_mode_returns_physical_phase(tmp_path: Path) -> None:
    ex = np.ones((3, 3), dtype=np.complex128)
    zbf = _field(ex, zx=2.0, rayleigh=4.0)
    path = tmp_path / "physical.ZBF"
    write_zbf(path, zbf)

    source = ZbfSource(path, reference_mode="physical")
    amplitude, phase, _pilot, _wfo = source.create_initial_wavefront()

    np.testing.assert_allclose(amplitude, np.ones((3, 3)))
    np.testing.assert_allclose(np.exp(1j * phase), np.exp(1j * zbf_reference_phase(zbf)))


def test_zbf_source_rejects_polarized_input_by_default(tmp_path: Path) -> None:
    zbf = _field(np.ones((2, 2), dtype=np.complex128))
    zbf.is_polarized = 1
    zbf.ey = zbf.ex.copy()
    path = tmp_path / "polarized.ZBF"
    write_zbf(path, zbf)

    with pytest.raises(ValueError, match="polarized"):
        ZbfSource(path).create_initial_wavefront()


def test_zbf_source_rejects_astigmatic_header_by_default(tmp_path: Path) -> None:
    zbf = _field(np.ones((2, 2), dtype=np.complex128))
    zbf.wy = 3.0
    path = tmp_path / "astigmatic.ZBF"
    write_zbf(path, zbf)

    with pytest.raises(ValueError, match="astigmatic"):
        ZbfSource(path).create_initial_wavefront()
```

**Step 2: Run tests to verify failure**

Run:

```powershell
pytest tests/test_zbf_source.py -v
```

Expected: FAIL because `ZbfSource` is not exported.

**Step 3: Commit**

Do not commit until Task 4 implements the source.

---

### Task 4: Implement ZbfSource Without Changing Core Propagation

**Files:**
- Modify: `pop/source.py`
- Modify: `pop/__init__.py`
- Test: `tests/test_zbf_source.py`

**Step 1: Add implementation to `pop/source.py`**

Add `ZbfSource` after `CustomSource` or before the debug plotting helper:

```python
@dataclass
class ZbfSource:
    zbf_path: str | Path
    reference_mode: str = "reference_relative"
    allow_polarized_ex_only: bool = False
    allow_astigmatic_approximation: bool = False
    radial_rtol: float = 1e-6
    radial_atol: float = 1e-9

    def __post_init__(self) -> None:
        mode = str(self.reference_mode).strip().lower()
        if mode not in {"reference_relative", "physical"}:
            raise ValueError("reference_mode must be 'reference_relative' or 'physical'")
        self.reference_mode = mode
        self.zbf_path = Path(self.zbf_path)

    @property
    def wavelength_um(self) -> float:
        zbf = read_zbf(self.zbf_path)
        return zbf.wavelength * 1e3

    def create_initial_wavefront(self):
        import proper
        from .io.zbf import read_zbf, zbf_reference_phase

        zbf = read_zbf(self.zbf_path)
        if zbf.is_polarized and not self.allow_polarized_ex_only:
            raise ValueError("ZBF polarized input is not supported by ZbfSource by default")
        self._validate_radial_header(zbf)

        pilot_beam = self._pilot_from_zbf(zbf)
        residual_field = np.asarray(zbf.ex, dtype=np.complex128)
        physical_field = residual_field * np.exp(1j * zbf_reference_phase(zbf))

        if self.reference_mode == "physical":
            wfarr_field = physical_field
            phase = np.angle(physical_field)
        else:
            wfarr_field = residual_field
            phase = np.angle(physical_field)

        amplitude = np.abs(physical_field)
        wavelength_m = zbf.wavelength * 1e-3
        sampling_m = zbf.dx * 1e-3
        beam_diameter_m = 2.0 * pilot_beam.waist_radius_mm * 1e-3
        beam_ratio = beam_diameter_m / (zbf.nx * sampling_m)
        beam_ratio = max(min(float(beam_ratio), 1.0), 1e-6)

        wfo = proper.prop_begin(beam_diameter_m, wavelength_m, zbf.nx, beam_ratio)
        wfo.w0 = pilot_beam.waist_radius_mm * 1e-3
        wfo.z_Rayleigh = pilot_beam.rayleigh_length_mm * 1e-3
        wfo.z = float(zbf.zx) * 1e-3
        wfo.z_w0 = 0.0
        wfo._dx = sampling_m
        if abs(zbf.zx) < proper.rayleigh_factor * max(abs(zbf.rx), 1e-15):
            wfo.beam_type_old = "INSIDE_"
            wfo.reference_surface = "PLANAR"
        else:
            wfo.beam_type_old = "OUTSIDE"
            wfo.reference_surface = "SPHERI"
        wfo.wfarr = proper.prop_shift_center(wfarr_field)
        return amplitude, phase, pilot_beam, wfo
```

Add private helpers:

```python
    def _validate_radial_header(self, zbf) -> None:
        pairs = [("dx", zbf.dx, zbf.dy), ("z", zbf.zx, zbf.zy), ("rayleigh", zbf.rx, zbf.ry), ("waist", zbf.wx, zbf.wy)]
        for name, x_value, y_value in pairs:
            if not np.isclose(x_value, y_value, rtol=self.radial_rtol, atol=self.radial_atol):
                if not self.allow_astigmatic_approximation:
                    raise ValueError(f"ZBF astigmatic header is not supported: {name} differs")

    def _pilot_from_zbf(self, zbf) -> PilotBeamParams:
        q = complex(float(zbf.zx), float(zbf.rx))
        return PilotBeamParams.from_q_parameter(q, zbf.wavelength * 1e3, current_refractive_index=float(zbf.index))
```

At the top of `pop/source.py`, import `Path`:

```python
from pathlib import Path
```

If importing `read_zbf` inside the `wavelength_um` property is needed, add the import inside the property to avoid import cycles.

**Step 2: Export from `pop/__init__.py`**

Modify imports:

```python
from .source import CustomSource, GaussianSource, ZbfSource
```

Add to `__all__`:

```python
"ZbfSource",
```

**Step 3: Run source tests**

Run:

```powershell
pytest tests/test_zbf_source.py -v
```

Expected: PASS.

**Step 4: Run ZBF I/O tests**

Run:

```powershell
pytest tests/test_zbf_io.py tests/test_zbf_source.py -v
```

Expected: PASS.

**Step 5: Commit**

Run:

```powershell
git add pop/source.py pop/__init__.py tests/test_zbf_source.py
git commit -m "feat: add zbf source adapter"
```

Expected: commit succeeds.

---

### Task 5: Add Shared Comparison Utilities

**Files:**
- Create: `sandbox/zemax_pop_benchmark/__init__.py`
- Create: `sandbox/zemax_pop_benchmark/comparison.py`
- Create: `tests/test_zemax_pop_comparison.py`

**Step 1: Write failing comparison tests**

Create `tests/test_zemax_pop_comparison.py` with synthetic fields:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np

from pop.io.zbf import ZbfField
from sandbox.zemax_pop_benchmark.comparison import compare_pop_state_to_zbf


class Dummy:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _zbf(ex: np.ndarray) -> ZbfField:
    ny, nx = ex.shape
    return ZbfField(
        path=None,
        version=1,
        nx=nx,
        ny=ny,
        is_polarized=0,
        units=0,
        dx=0.25,
        dy=0.25,
        zx=0.0,
        rx=0.0,
        wx=1.0,
        zy=0.0,
        ry=0.0,
        wy=1.0,
        wavelength=0.01,
        index=1.0,
        receiver_efficiency=0.0,
        system_efficiency=0.0,
        ex=ex,
        ey=None,
    )


def test_compare_reports_zero_residual_when_fields_match() -> None:
    ex = np.array([[1 + 0j, 1j], [-1 + 0j, 1 - 1j]], dtype=np.complex128)
    state = Dummy(
        surface_index=3,
        position="entrance",
        proper_wfo=None,
        pilot_beam_params=Dummy(wavelength_um=10.0),
        grid_sampling=Dummy(sampling_mm=0.25),
    )

    result = compare_pop_state_to_zbf(
        state=state,
        pop_reference_relative=ex,
        pop_reference_phase=np.zeros(ex.shape),
        zbf=_zbf(ex),
        surface_name="S3",
        mask_threshold=0.0,
    )

    assert result.summary["phase_rms_waves"] < 1e-12
    assert result.summary["relative_intensity_rms"] < 1e-12


def test_compare_removes_phase_piston() -> None:
    ex = np.ones((2, 2), dtype=np.complex128)
    result = compare_pop_state_to_zbf(
        state=Dummy(
            surface_index=4,
            position="exit",
            pilot_beam_params=Dummy(wavelength_um=10.0),
            grid_sampling=Dummy(sampling_mm=0.25),
        ),
        pop_reference_relative=np.exp(1j * 0.25) * ex,
        pop_reference_phase=np.zeros(ex.shape),
        zbf=_zbf(ex),
        surface_name="S4",
        mask_threshold=0.0,
    )

    assert abs(result.summary["phase_piston_rad"] - 0.25) < 1e-12
    assert result.summary["phase_rms_waves"] < 1e-12
```

**Step 2: Run tests to verify failure**

Run:

```powershell
pytest tests/test_zemax_pop_comparison.py -v
```

Expected: FAIL because `sandbox.zemax_pop_benchmark.comparison` does not exist.

**Step 3: Implement comparison utilities**

Create `sandbox/zemax_pop_benchmark/__init__.py` as an empty package marker.

Create `sandbox/zemax_pop_benchmark/comparison.py` with:

- `ComparisonResult` dataclass containing `summary`, `fields`, and `residuals`;
- `_remove_piston`;
- `_masked_rms`;
- `_masked_peak_to_valley`;
- `compare_pop_state_to_zbf`.

The function should:

1. Use `zbf.ex` as Zemax reference-relative field.
2. Compare `pop_reference_relative * conj(zbf.ex)`.
3. Normalize intensity residuals by each peak intensity.
4. Mask on both normalized intensity arrays.
5. Remove piston from phase residual before RMS/PV.
6. Include sampling and wavelength metadata in `summary`.

**Step 4: Run tests**

Run:

```powershell
pytest tests/test_zemax_pop_comparison.py -v
```

Expected: PASS.

**Step 5: Commit**

Run:

```powershell
git add sandbox/zemax_pop_benchmark/__init__.py sandbox/zemax_pop_benchmark/comparison.py tests/test_zemax_pop_comparison.py
git commit -m "feat: add zemax pop comparison utilities"
```

Expected: commit succeeds.

---

### Task 6: Add Benchmark Config and Local POP Runner

**Files:**
- Create: `sandbox/zemax_pop_benchmark/config.py`
- Create: `sandbox/zemax_pop_benchmark/popapi_runner.py`
- Create: `tests/test_zemax_pop_benchmark_config.py`

**Step 1: Write failing tests**

Create `tests/test_zemax_pop_benchmark_config.py`:

```python
from __future__ import annotations

from pathlib import Path

from sandbox.zemax_pop_benchmark.config import BenchmarkConfig, PopSamplingConfig


def test_benchmark_config_serializes_paths() -> None:
    config = BenchmarkConfig(
        zmx_path=Path("system.zmx"),
        output_dir=Path("output/run"),
        sampling=PopSamplingConfig(grid_size=128, physical_size_mm=64.0),
    )

    payload = config.to_jsonable()

    assert payload["zmx_path"] == "system.zmx"
    assert payload["sampling"]["grid_size"] == 128
```

**Step 2: Run test to verify failure**

Run:

```powershell
pytest tests/test_zemax_pop_benchmark_config.py -v
```

Expected: FAIL because config module does not exist.

**Step 3: Implement config**

Create dataclasses:

- `PopSamplingConfig(grid_size, physical_size_mm, beam_diam_fraction=None)`;
- `GaussianInputConfig(wavelength_um, w0_mm, z0_mm=0.0)`;
- `ZbfInputConfig(path, reference_mode="reference_relative")`;
- `ComparisonConfig(surface_indices=None, pop_position="entrance", mask_threshold=0.1, sampling_rtol=1e-4, sampling_atol_mm=5e-9)`;
- `BenchmarkConfig(zmx_path, output_dir, sampling, gaussian, zbf_input=None, comparison=ComparisonConfig())`.

Each dataclass should have `to_jsonable()`.

**Step 4: Implement `popapi_runner.py`**

Add `PopApiRunner` with:

- `run_gaussian_direct(config)`;
- `run_zbf_input(config)`.

Implementation should use public POP APIs only:

```python
system = pop.load_zmx(str(config.zmx_path))
source = pop.GaussianSource(...)
result = pop.propagate(system, source, options=...)
```

For ZBF:

```python
source = pop.ZbfSource(config.zbf_input.path, reference_mode=config.zbf_input.reference_mode)
```

**Step 5: Run config tests**

Run:

```powershell
pytest tests/test_zemax_pop_benchmark_config.py -v
```

Expected: PASS.

**Step 6: Commit**

Run:

```powershell
git add sandbox/zemax_pop_benchmark/config.py sandbox/zemax_pop_benchmark/popapi_runner.py tests/test_zemax_pop_benchmark_config.py
git commit -m "feat: add benchmark config and local runner"
```

Expected: commit succeeds.

---

### Task 7: Add ZOSAPI Runner With Fake Tests

**Files:**
- Create: `sandbox/zemax_pop_benchmark/zosapi_runner.py`
- Create: `tests/test_zosapi_runner.py`

**Step 1: Write fake tests**

Create `tests/test_zosapi_runner.py` that monkeypatches the POP analysis class:

```python
from __future__ import annotations

from pathlib import Path

from sandbox.zemax_pop_benchmark.config import BenchmarkConfig, GaussianInputConfig, PopSamplingConfig
from sandbox.zemax_pop_benchmark.zosapi_runner import build_gaussian_pop_kwargs, build_zbf_pop_kwargs


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        zmx_path=Path("system.zmx"),
        output_dir=Path("output/run"),
        sampling=PopSamplingConfig(grid_size=128, physical_size_mm=64.0),
        gaussian=GaussianInputConfig(wavelength_um=10.64, w0_mm=29.0),
    )


def test_build_gaussian_pop_kwargs_sets_gaussian_waist() -> None:
    kwargs = build_gaussian_pop_kwargs(_config(), output_stem="gaussian_out")

    assert kwargs["beam_type"] == "GaussianWaist"
    assert kwargs["beam_parameters"]["Waist X"] == 29.0
    assert kwargs["beam_parameters"]["Waist Y"] == 29.0
    assert kwargs["x_sampling"] == 128
    assert kwargs["x_width"] == 64.0
    assert kwargs["save_output_beam"] is True
    assert kwargs["output_beam_file"] == "gaussian_out"


def test_build_zbf_pop_kwargs_sets_file_beam() -> None:
    kwargs = build_zbf_pop_kwargs(_config(), beam_file="input.ZBF", output_stem="zbf_out")

    assert kwargs["beam_type"] == "File"
    assert kwargs["beam_file"] == "input.ZBF"
    assert kwargs["save_output_beam"] is True
```

**Step 2: Run test to verify failure**

Run:

```powershell
pytest tests/test_zosapi_runner.py -v
```

Expected: FAIL because module does not exist.

**Step 3: Implement `zosapi_runner.py`**

Implement pure builder functions first:

- `build_gaussian_pop_kwargs(config, output_stem, data_type="Irradiance")`;
- `build_zbf_pop_kwargs(config, beam_file, output_stem, data_type="Irradiance")`.

Then add `ZosPopRunner`:

- imports `zospy` lazily;
- imports `PhysicalOpticsPropagation` lazily;
- connects and loads `.zmx`;
- copies ZBF input to Zemax beam folder for File mode;
- calls POP with builder kwargs;
- returns a dataclass with output directory and expected ZBF paths.

Do not connect to Zemax in unit tests.

**Step 4: Run fake tests**

Run:

```powershell
pytest tests/test_zosapi_runner.py -v
```

Expected: PASS.

**Step 5: Commit**

Run:

```powershell
git add sandbox/zemax_pop_benchmark/zosapi_runner.py tests/test_zosapi_runner.py
git commit -m "feat: add zosapi pop runner"
```

Expected: commit succeeds.

---

### Task 8: Add Manual Benchmark Runner

**Files:**
- Create: `sandbox/zemax_pop_benchmark/run_biconic_zemax_pop_benchmark.py`
- Modify: `sandbox/zemax_pop_benchmark/__init__.py` if exports are useful.

**Step 1: Implement CLI runner**

Create a script with:

- default config for `sandbox/Zemax_baseline/biconic_focus_test_expand_validation.zmx`;
- options:
  - `--mode gaussian`;
  - `--mode zbf`;
  - `--mode both`;
  - `--input-zbf path`;
  - `--output-dir path`;
  - `--grid-size 1024`;
  - `--physical-size-mm 348.0`;
- calls `ZosPopRunner`;
- calls `PopApiRunner`;
- calls comparison utilities;
- writes `config.json` and `summary.json`.

**Step 2: Run syntax check**

Run:

```powershell
python -m py_compile sandbox/zemax_pop_benchmark/run_biconic_zemax_pop_benchmark.py
```

Expected: no output and exit code 0.

**Step 3: Run import check**

Run:

```powershell
python -c "import sandbox.zemax_pop_benchmark.run_biconic_zemax_pop_benchmark as r; print(r.__name__)"
```

Expected: prints module name without connecting to Zemax.

**Step 4: Commit**

Run:

```powershell
git add sandbox/zemax_pop_benchmark/run_biconic_zemax_pop_benchmark.py sandbox/zemax_pop_benchmark/__init__.py
git commit -m "feat: add biconic zemax pop benchmark runner"
```

Expected: commit succeeds.

---

### Task 9: Add Opt-In Pytest Benchmark Tests

**Files:**
- Create: `tests/benchmark/test_zemax_pop_benchmark.py`
- Modify: no production files expected.

**Step 1: Create benchmark tests**

Create `tests/benchmark/test_zemax_pop_benchmark.py`:

```python
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("BTS_RUN_ZEMAX_BENCHMARK") != "1",
    reason="Set BTS_RUN_ZEMAX_BENCHMARK=1 to run Zemax-dependent benchmark tests",
)


def test_gaussian_direct_benchmark_smoke(tmp_path: Path) -> None:
    from sandbox.zemax_pop_benchmark.run_biconic_zemax_pop_benchmark import run_benchmark

    result = run_benchmark(mode="gaussian", output_dir=tmp_path, grid_size=128)

    assert result["modes"]["gaussian_direct"]["surface_count"] >= 1


def test_zbf_input_nonideal_benchmark_smoke_requires_input(tmp_path: Path) -> None:
    from sandbox.zemax_pop_benchmark.run_biconic_zemax_pop_benchmark import run_benchmark

    with pytest.raises(ValueError, match="input_zbf"):
        run_benchmark(mode="zbf", output_dir=tmp_path, grid_size=128)
```

**Step 2: Run skipped tests**

Run:

```powershell
pytest tests/benchmark/test_zemax_pop_benchmark.py -v
```

Expected: all tests SKIPPED unless `BTS_RUN_ZEMAX_BENCHMARK=1` is set.

**Step 3: Run local non-Zemax tests**

Run:

```powershell
pytest tests/test_zbf_io.py tests/test_zbf_source.py tests/test_zemax_pop_comparison.py tests/test_zemax_pop_benchmark_config.py tests/test_zosapi_runner.py -v
```

Expected: PASS.

**Step 4: Commit**

Run:

```powershell
git add tests/benchmark/test_zemax_pop_benchmark.py
git commit -m "test: add opt-in zemax pop benchmark tests"
```

Expected: commit succeeds.

---

### Task 10: Run Final Verification

**Files:**
- No source changes expected.

**Step 1: Run local test suite added by this plan**

Run:

```powershell
pytest tests/test_zbf_io.py tests/test_zbf_source.py tests/test_zemax_pop_comparison.py tests/test_zemax_pop_benchmark_config.py tests/test_zosapi_runner.py tests/benchmark/test_zemax_pop_benchmark.py -v
```

Expected:

- local tests PASS;
- Zemax benchmark tests SKIPPED unless explicitly enabled.

**Step 2: Run compile check for benchmark package**

Run:

```powershell
python -m compileall pop sandbox/zemax_pop_benchmark tests
```

Expected: no syntax errors.

**Step 3: Check git status**

Run:

```powershell
git status --short
```

Expected: only unrelated pre-existing dirty files remain.

**Step 4: Final note**

Do not claim Zemax parity has been measured unless a real `BTS_RUN_ZEMAX_BENCHMARK=1` run was executed and its output was inspected.
