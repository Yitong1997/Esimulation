# Zemax POP Free-Space Propagator Identification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a reproducible black-box experiment that identifies which fixed propagation operator best explains Zemax POP on S7→S8, S12→S13, and S13→S14, using the corresponding start-plane ZBF physical field as the common input and exact scalar propagation as the accuracy baseline.

**Architecture:** All new behavior lives in an isolated diagnostic package under `sandbox/free_space_algorithm_identification/`. The package separates lossless ZBF I/O, physical-field/reference contracts, numerical propagation, ZOS-API capture, comparison/decision logic, and orchestration; generated reports and ZBFs live under a unique untracked run directory. Production `pop/` and PROPER code remain untouched until the three-segment evidence identifies a root cause.

**Tech Stack:** Python 3.13, NumPy 2.3.5, SciPy 1.17.0 (`scipy.fft`, stable unit-circle `scipy.signal.ZoomFFT`, small-oracle `czt` only), pytest, mpmath, ZOSPy/ZOS-API, Matplotlib, JSON/NPZ/ZBF artifacts, PowerShell on Windows.

## Global Constraints

- Before Task 1 execution, invoke `superpowers:using-git-worktrees`; never implement in the current dirty worktree.
- Execute each implementation task with `superpowers:test-driven-development`, and invoke `superpowers:verification-before-completion` before any completion claim.
- Do not modify any file under `pop/`, `proper_v3.3.4_python/proper/`, or `angular_spectrum_method/` in this plan.
- Do not use POP API intermediate fields as candidate inputs. Every candidate for a segment consumes the same physical field reconstructed from that segment's start ZBF.
- Preserve the original ZBF header bit-for-bit except `nx`, `ny`, `dx`, and `dy`; resample `Ex` and `Ey` together and preserve all reserved words and trailing bytes.
- The fresh biconic ZBF preflight must prove `is_polarized == 0` before scalar candidate ranking. The codec and derivation layers still preserve polarized fixtures losslessly; if any real input is polarized, stop and approve a Jones-field extension instead of silently discarding `Ey`.
- Freeze whether ZBF payloads are continuous point values or cell-energy samples through an independent cross-sampling power/identity probe. Every payload-to-field and field-to-payload conversion takes that explicit convention; per-case entrance constants may not silently absorb an unresolved area factor.
- Use `complex128` and the CPU NumPy/SciPy implementation for canonical results. GPU or `complex64` results may not establish the accuracy baseline.
- Fix coordinate center, axis directions, reflection parity, phasor convention, model distance, and reference formulas before viewing candidate residuals.
- Raw ZBF phasor conversion is global, not surface-specific: every directly read `Ex` is converted to the common `exp(-i omega t)` convention as `conj(Ex)`. Real coordinate pullbacks and reflection parity never cancel that anti-linear conversion. S7/S8 use `axis_sign=-1`; S12/S13/S14 use `axis_sign=+1`.
- The even ZBF grid is sample-at-zero: `x_i=(i-Nx/2)dx`, `y_j=(j-Ny/2)dy`. This is fixed independently by the OpticStudio `IAR_DataGrid` and ZBF format contracts and must be revalidated from raw `DataGrid` metadata. Never use the half-step-shifted ZOSPy DataFrame labels to reconstruct phase or resample a field.
- Use model physical distances 368.600000 mm, 608.600000 mm, and 2.000000 mm for propagation. Use ZBF pilot-to-waist distances only for STW/WTS sampling.
- Candidate comparison may remove one global phase piston. It may not fit amplitude, shift, flip, rotation, tilt, defocus, astigmatism, quartic phase, Zernike terms, propagation distance, reference radius, or a residual scale coefficient.
- The primary region is the central connected component at `I_Z / I_Z.max() >= 1e-3`; repeat conclusions at `1e-2` and `1e-6`.
- A case that fails input derivation, coordinate/reference validation, start identity, report/header validation, or sampling convergence stops at that gate and may not enter algorithm ranking.
- Never overwrite a run directory or collect ZBFs with a broad historical glob. Every ZOS run uses a unique anchored prefix and all captured files are hashed immediately.
- Default pytest runs are offline. Live native Zemax smoke tests require `BTS_RUN_ZEMAX_BENCHMARK=1`; 2048/4096 runs additionally require `BTS_RUN_ZEMAX_HIGH_SAMPLING=1`.
- Generated high-resolution ZBFs, native reports, CFG files, arrays, and plots are not committed. Commit only code, tests, small summary JSON files, and the final scientific report.
- If the experiment does not uniquely distinguish candidates within the prescribed uncertainty, report “当前采样与输入条件下不可判别”; do not select the smallest numerical error.

## File Structure

Create the following focused package:

```text
sandbox/free_space_algorithm_identification/
  __init__.py              # public diagnostic API only
  models.py                # immutable grids, fields, segment and result dataclasses
  biconic_case.py          # fixed S7/S8/S12/S13/S14 registry and baseline paths
  zbf_binary.py            # bit-preserving ZBF codec and header diffs
  field_contract.py        # coordinate, phasor, Q/Phi, pilot and physical-field rules
  geometry.py              # native-report frame/distance validation
  fourier.py               # continuous FFT convention, CZT evaluation and resampling
  derived_inputs.py        # fixed-window interpolation and fixed-step zero padding
  sampling.py              # segment-specific Zemax and exact-baseline matrices
  asm.py                   # band-limited exact Helmholtz angular spectrum
  rayleigh_sommerfeld.py   # sparse RS-I direct quadrature
  fresnel.py               # independent physical-field Fresnel propagation
  candidates.py            # F_Q, R_Phi|Q and R_Phi|Phi fixed operators
  metrics.py               # fixed ROI, piston, physical complex-field metrics
  decision.py              # uncertainty, 3u/5u and three-segment decisions
  artifacts.py             # immutable layouts, JSON, provenance and hashes
  native_report.py         # native POP report parser and assertions
  zos_runner.py            # sustained ZOSPy segment capture
  identity.py              # entrance-only complex calibration and identity gate
  interventions.py         # conditional reference/pilot and weak-sideband probes
  pipeline.py              # gate-controlled execution graph
  report.py                # academic Chinese final report
  cli.py                   # staged command-line entry point
  .gitignore               # ignores output/
```

Create offline tests under:

```text
tests/free_space_identification/
  conftest.py
  test_models.py
  test_zbf_binary.py
  test_field_contract.py
  test_geometry.py
  test_fourier.py
  test_derived_inputs.py
  test_sampling.py
  test_metrics_decision.py
  test_asm.py
  test_rayleigh_sommerfeld.py
  test_fresnel_candidates.py
  fixtures/
    native_oo_report.txt
    native_oi_report.txt
    native_io_report.txt
  test_artifacts.py
  test_native_report.py
  test_zos_runner.py
  test_identity.py
  test_interventions.py
  test_pipeline_report.py
  test_static_fixtures.py
  test_zemax_live.py
```

The following existing files are read-only evidence or implementation references, not modification targets:

- `sandbox/zemax_pop_benchmark/zosapi_runner.py`
- `pop/io/zbf.py`
- `sandbox/diagnostics/s7_s8_phase_root_cause.md` until Task 16

No diagnostic module may import the ignored, untracked `sandbox/biconic_focus_baseline_utils.py`; Task 3 owns a minimal tracked native-report parser.

---

### Task 1: Immutable experiment models and the fixed biconic segment registry

**Files:**
- Create: `sandbox/free_space_algorithm_identification/__init__.py`
- Create: `sandbox/free_space_algorithm_identification/models.py`
- Create: `sandbox/free_space_algorithm_identification/biconic_case.py`
- Create: `sandbox/free_space_algorithm_identification/.gitignore`
- Create: `tests/free_space_identification/conftest.py`
- Create: `tests/free_space_identification/test_models.py`

**Interfaces:**
- Produces: `UniformGrid2D`, `PointField2D`, `SurfaceConvention`, `SegmentSpec`, `SamplingCase`, `BICONIC_SEGMENTS`, and `resolve_biconic_baseline()`.
- Consumes: no new diagnostic modules.

- [ ] **Step 1: Write the failing registry and grid tests**

```python
from sandbox.free_space_algorithm_identification.biconic_case import BICONIC_SEGMENTS
from sandbox.free_space_algorithm_identification.models import UniformGrid2D


def test_registry_contains_only_true_free_space_pairs() -> None:
    assert [(s.start_surface, s.end_surface) for s in BICONIC_SEGMENTS] == [
        (7, 8), (12, 13), (13, 14)
    ]
    assert [s.branch for s in BICONIC_SEGMENTS] == ["OO", "OI", "IO"]


def test_even_grid_uses_the_prevalidated_sample_at_zero_convention() -> None:
    grid = UniformGrid2D.centered(nx=4, ny=4, dx_mm=0.5, dy_mm=0.25)
    assert grid.x_mm.tolist() == [-1.0, -0.5, 0.0, 0.5]
    assert grid.y_mm.tolist() == [-0.5, -0.25, 0.0, 0.25]


def test_segment_axis_and_phasor_conventions_are_fixed() -> None:
    by_key = {s.key: s for s in BICONIC_SEGMENTS}
    assert all(s.start_convention.side == "after" and s.end_convention.side == "after"
               for s in BICONIC_SEGMENTS)
    assert by_key["S07_S08"].start_convention.axis_sign == -1
    assert by_key["S12_S13"].start_convention.axis_sign == 1
    assert all(not hasattr(s.start_convention, "conjugate")
               and not hasattr(s.end_convention, "conjugate")
               for s in BICONIC_SEGMENTS)
```

- [ ] **Step 2: Run the tests and verify the import failure**

Run:

```powershell
python -m pytest tests/free_space_identification/test_models.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'sandbox.free_space_algorithm_identification'`.

- [ ] **Step 3: Implement the immutable types and registry**

```python
# models.py
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import numpy as np

Branch = Literal["OO", "OI", "IO"]
DerivationStrategy = Literal[
    "exact_copy", "fourier_refine_fixed_window", "zero_extend_fixed_sampling",
    "zero_extend_then_fourier_refine", "chained_zemax_output"
]
SamplingPurpose = Literal[
    "native", "input_resolution", "output_resolution",
    "combined_resolution", "window_control"
]
SourceKind = Literal["native_zbf", "derived_zbf", "chained_zemax_output"]
SampleValueConvention = Literal["point_value", "cell_energy"]


@dataclass(frozen=True)
class UniformGrid2D:
    x_mm: np.ndarray
    y_mm: np.ndarray

    @classmethod
    def centered(cls, *, nx: int, ny: int, dx_mm: float, dy_mm: float) -> "UniformGrid2D":
        if nx < 2 or ny < 2 or dx_mm <= 0 or dy_mm <= 0:
            raise ValueError("grid dimensions must be at least two and sampling must be positive")
        x = (np.arange(nx, dtype=np.float64) - nx // 2) * dx_mm
        y = (np.arange(ny, dtype=np.float64) - ny // 2) * dy_mm
        return cls(x_mm=x, y_mm=y)

    @property
    def nx(self) -> int:
        return int(self.x_mm.size)

    @property
    def ny(self) -> int:
        return int(self.y_mm.size)

    @property
    def dx_mm(self) -> float:
        return float(self.x_mm[1] - self.x_mm[0])

    @property
    def dy_mm(self) -> float:
        return float(self.y_mm[1] - self.y_mm[0])

    @property
    def pixel_area_mm2(self) -> float:
        return self.dx_mm * self.dy_mm


@dataclass(frozen=True)
class PointField2D:
    values: np.ndarray
    grid: UniformGrid2D

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.complex128)
        if values.shape != (self.grid.ny, self.grid.nx):
            raise ValueError("field shape does not match grid")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class SurfaceConvention:
    surface: int
    side: Literal["after"]
    axis_sign: Literal[-1, 1]


@dataclass(frozen=True)
class SegmentSpec:
    key: str
    start_surface: int
    end_surface: int
    branch: Branch
    model_distance_mm: float
    source_zbf_name: str
    target_zbf_name: str
    start_convention: SurfaceConvention
    end_convention: SurfaceConvention


@dataclass(frozen=True)
class NativeSurfaceSource:
    surface: int
    role: Literal["fresh_continuous", "historical_preflight"]


@dataclass(frozen=True)
class CaseOutputSource:
    producer_case: str
    surface: int


SamplingSource = NativeSurfaceSource | CaseOutputSource


@dataclass(frozen=True)
class SamplingCase:
    case_id: str
    segment_key: str
    source_kind: SourceKind
    strategy: DerivationStrategy
    purpose: SamplingPurpose
    nx: int
    ny: int
    dx_mm: float
    dy_mm: float
    source: SamplingSource
    expected_output_dx_mm: float
    expected_output_dy_mm: float
    depends_on_case: str | None = None
    repeat_count: int = 1
```

```python
# biconic_case.py
from pathlib import Path
from .models import SegmentSpec, SurfaceConvention

S7 = SurfaceConvention(7, "after", -1)
S8 = SurfaceConvention(8, "after", -1)
S12 = SurfaceConvention(12, "after", 1)
S13 = SurfaceConvention(13, "after", 1)
S14 = SurfaceConvention(14, "after", 1)

BICONIC_SEGMENTS = (
    SegmentSpec("S07_S08", 7, 8, "OO", 368.600000,
                "biconic_focus_test_0007.ZBF", "biconic_focus_test_0008.ZBF", S7, S8),
    SegmentSpec("S12_S13", 12, 13, "OI", 608.600000,
                "biconic_focus_test_0012.ZBF", "biconic_focus_test_0013.ZBF", S12, S13),
    SegmentSpec("S13_S14", 13, 14, "IO", 2.000000,
                "biconic_focus_test_0013.ZBF", "biconic_focus_test_0014.ZBF", S13, S14),
)


def resolve_biconic_baseline(baseline_dir: Path) -> Path:
    path = baseline_dir.resolve()
    required = [
        "biconic_focus_test.zmx", "biconic_focus_test.CFG",
        "biconic_focus_test.txt", "biconic_phase.txt",
        "biconic_focus_test_0007.ZBF",
        "biconic_focus_test_0008.ZBF", "biconic_focus_test_0012.ZBF",
        "biconic_focus_test_0013.ZBF", "biconic_focus_test_0014.ZBF",
    ]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing baseline files: {missing}")
    return path
```

Set `.gitignore` to exactly:

```text
output/
```

`UniformGrid2D.__post_init__()` and `PointField2D.__post_init__()` must copy their arrays, validate finiteness/shape/uniform spacing, and set the stored NumPy arrays read-only so the frozen dataclasses are actually immutable. `tests/free_space_identification/conftest.py` contains only environment-gated `baseline_dir` and `baseline_report` pytest fixtures; it must not import modules that are created in later tasks. Task-specific synthetic builders and direct-sum oracles live in the corresponding test file so Task 1 remains independently runnable.

The sample-at-zero even-grid constructor is fixed independently by the installed OpticStudio `IAR_DataGrid` and ZBF format contracts, plus existing raw DataGrid evidence; it is not selected from endpoint error. Task 3 encodes the fail-closed validation contract, and Task 12 must revalidate it from raw ZOS-API metadata before the formal run. Failure blocks the experiment and does not trigger a trial of the half-pixel convention against endpoint error.

- [ ] **Step 4: Run the model tests**

Run:

```powershell
python -m pytest tests/free_space_identification/test_models.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit Task 1**

```powershell
git add sandbox/free_space_algorithm_identification tests/free_space_identification/test_models.py tests/free_space_identification/conftest.py
git commit -m "test: define free-space identification contracts"
```

### Task 2: Bit-preserving ZBF codec and immutable header audit

**Files:**
- Create: `sandbox/free_space_algorithm_identification/zbf_binary.py`
- Create: `tests/free_space_identification/test_zbf_binary.py`

**Interfaces:**
- Consumes: `UniformGrid2D` from Task 1.
- Produces: `RawZbfHeader`, `LosslessZbf`, `HeaderDifference`, `read_lossless_zbf()`, `write_lossless_zbf()`, `patch_sampling_header()`, `compare_headers()`, and `sha256_file()`.

- [ ] **Step 1: Write failing tests containing nonzero reserved words, negative zero, NaN payload, polarization, and trailing bytes**

Define `write_synthetic_lossless_fixture()` locally in `test_zbf_binary.py` with `struct.pack`, including explicit integer/double words and complex payload bytes; it must not call the codec under test to construct its oracle file.

```python
def test_roundtrip_preserves_all_unmodified_header_bits_and_trailing_bytes(tmp_path):
    source = write_synthetic_lossless_fixture(
        tmp_path / "source.ZBF",
        reserved_ints=(11, 22, 33, 44),
        reserved_double_bits=(0x8000000000000000, 0x7FF8000000000042) + (0,) * 6,
        trailing=b"BTS-TAIL",
        polarized=True,
    )
    beam = read_lossless_zbf(source)
    output = tmp_path / "roundtrip.ZBF"
    write_lossless_zbf(output, beam)
    assert output.read_bytes() == source.read_bytes()


def test_patch_sampling_changes_only_nx_ny_dx_dy(tmp_path):
    source = read_lossless_zbf(write_synthetic_lossless_fixture(tmp_path / "source.ZBF"))
    patched = patch_sampling_header(source.header, nx=16, ny=16, dx=0.125, dy=0.25)
    diff = compare_headers(source.header, patched)
    assert diff.changed_named_fields == ("nx", "ny", "dx", "dy")
    assert diff.changed_reserved_int_indices == ()
    assert diff.changed_reserved_double_indices == ()
```

- [ ] **Step 2: Run the codec tests and verify failure**

Run:

```powershell
python -m pytest tests/free_space_identification/test_zbf_binary.py -q
```

Expected: collection fails because `zbf_binary.py` does not exist.

- [ ] **Step 3: Implement the codec by patching raw bytes instead of repacking the whole header**

Use these exact binary constants and offsets:

```python
INT_COUNT = 9
DOUBLE_COUNT = 20
HEADER_BYTES = INT_COUNT * 4 + DOUBLE_COUNT * 8  # 196
NX_OFFSET = 4
NY_OFFSET = 8
DX_OFFSET = 36
DY_OFFSET = 44
```

`RawZbfHeader` stores `raw_bytes`, `int_words: tuple[int, ...]`, and `double_bits: tuple[int, ...]`, and exposes read-only semantic properties with this exact mapping:

```text
int[0:5]    version, nx, ny, is_polarized, units
int[5:9]    four reserved integer words
double[0:12] dx, dy, zx, rx, wx, zy, ry, wy,
             wavelength_vacuum_mm, refractive_index,
             receiver_efficiency, system_efficiency
double[12:20] eight reserved double words
```

Floating semantic properties decode from raw bits on access; header comparison uses the stored integer/IEEE-754 bit words, not floating equality. Validate positive dimensions, wavelength, refractive index, and payload length without normalizing negative zero, NaN payloads, or reserved values.

The core write path must follow this code:

```python
@dataclass(frozen=True)
class LosslessZbf:
    path: Path | None
    source_sha256: str
    header: RawZbfHeader
    ex: np.ndarray
    ey: np.ndarray | None
    trailing_bytes: bytes


def patch_sampling_header(header, *, nx, ny, dx, dy):
    raw = bytearray(header.raw_bytes)
    struct.pack_into("<i", raw, NX_OFFSET, int(nx))
    struct.pack_into("<i", raw, NY_OFFSET, int(ny))
    struct.pack_into("<d", raw, DX_OFFSET, float(dx))
    struct.pack_into("<d", raw, DY_OFFSET, float(dy))
    return RawZbfHeader.from_bytes(bytes(raw))


def write_lossless_zbf(path, beam):
    ny, nx = beam.ex.shape
    if (nx, ny) != (beam.header.nx, beam.header.ny):
        raise ValueError("Ex shape does not match ZBF header")
    if beam.header.is_polarized and beam.ey is None:
        raise ValueError("polarized ZBF requires Ey")
    if not beam.header.is_polarized and beam.ey is not None:
        raise ValueError("unpolarized ZBF cannot contain Ey")
    with Path(path).open("wb") as stream:
        stream.write(beam.header.raw_bytes)
        _write_complex_payload(stream, beam.ex)
        if beam.ey is not None:
            _write_complex_payload(stream, beam.ey)
        stream.write(beam.trailing_bytes)
```

`read_lossless_zbf()` must set the resolved source path and SHA-256, copy `Ex` and `Ey` into native-endian `complex128`, retain the original 196 header bytes, and preserve all bytes after the expected field payload as `trailing_bytes`. For an in-memory synthetic object, compute `source_sha256` from the exact header/payload/tail byte serialization and permit `path=None`; downstream code uses the stored hash rather than reopening `beam.path`.

- [ ] **Step 4: Run the codec tests and existing semantic-reader tests**

Run:

```powershell
python -m pytest tests/free_space_identification/test_zbf_binary.py tests/test_zbf_io.py -k "not reference_phase_uses_spherical_header_metadata" -q
```

Expected: all new codec tests and unaffected existing ZBF I/O tests pass. The excluded legacy test predates commit `1373b4c` and still expects Gaussian-curvature phase inside the Rayleigh range; the approved contract and new Task 3 tests require a planar reference there. Do not modify production ZBF code to satisfy that stale expectation.

- [ ] **Step 5: Commit Task 2**

```powershell
git add sandbox/free_space_algorithm_identification/zbf_binary.py tests/free_space_identification/test_zbf_binary.py
git commit -m "feat: add lossless ZBF diagnostic codec"
```

### Task 2A: Correct the global raw-ZBF phasor contract before Task 3

This is a migration task for the already-started execution branch whose Task 1 commit `615172e` and Task 2 head `aec1215` still contain the superseded surface-specific flag. The amended Task 1 specification above is already correct for a fresh execution, so a fresh run skips Task 2A after verifying the corrected Task 1 test is green. On the current branch, this task is required by the pre-Task-3 physics audit: the earlier flag had no evidence outside the old plan and incorrectly treated a real coordinate reflection as an anti-linear phasor conversion. Existing project contracts and the user-provided POP/ZBF reference-frame validation require uniform `conj(Ex)` for every directly read raw ZBF.

**Files:**
- Modify: `sandbox/free_space_algorithm_identification/models.py`
- Modify: `sandbox/free_space_algorithm_identification/biconic_case.py`
- Modify: `tests/free_space_identification/test_models.py`

- [ ] **Step 1: Write the failing regression before changing the models**

Change the registry test so that `SurfaceConvention` contains only `surface`, `side`, and `axis_sign`; assert that no start or end convention exposes a `conjugate` attribute. Add an analytic regression showing that a real-coordinate pullback commutes with complex conjugation but is not itself conjugation for a nontrivial complex field.

- [ ] **Step 2: Run the focused test and record RED**

```powershell
python -m pytest tests/free_space_identification/test_models.py -q
```

Expected on the current `aec1215` execution branch: the new contract test fails because the existing dataclass still exposes the invalid surface-specific flag. Expected on a fresh execution of the amended Task 1: the test is already green, no migration edit or extra commit is made, and execution proceeds to Task 3.

- [ ] **Step 3: Remove the surface-specific flag and update the registry**

`SurfaceConvention` must contain only the physical side and local-axis sign. Construct S7/S8 with `axis_sign=-1` and S12/S13/S14 with `axis_sign=+1`. Do not add another per-surface phasor switch. Task 3 will perform the single global conversion `reference_relative = np.conj(point_payload)`.

- [ ] **Step 4: Run focused and accumulated diagnostic tests**

```powershell
python -m pytest tests/free_space_identification/test_models.py tests/free_space_identification/test_zbf_binary.py -q
python -m pytest tests/test_zbf_io.py tests/test_zbf_source.py -k "not reference_phase_uses_spherical_header_metadata" -W ignore::DeprecationWarning -q
```

Expected: all focused tests pass; the known stale legacy test remains the only deselection.

- [ ] **Step 5: Commit Task 2A**

```powershell
git add -f sandbox/free_space_algorithm_identification/models.py sandbox/free_space_algorithm_identification/biconic_case.py tests/free_space_identification/test_models.py
git commit -m "fix: make raw ZBF conjugation a global contract"
```

### Task 3: Physical-field, coordinate, reference-phase, and pilot-state contract

**Files:**
- Create: `sandbox/free_space_algorithm_identification/field_contract.py`
- Create: `sandbox/free_space_algorithm_identification/geometry.py`
- Create: `tests/free_space_identification/test_field_contract.py`
- Create: `tests/free_space_identification/test_geometry.py`
- Create: `tests/free_space_identification/test_static_fixtures.py`

**Interfaces:**
- Consumes: `LosslessZbf`, `SegmentSpec`, `UniformGrid2D`, and `PointField2D`.
- Produces: `PilotState`, `MappedZbfField`, `ReportNumber`, `SegmentGeometry`, `RawGridEvidence`, `ConventionValidation`, `zbf_payload_to_point_values()`, `point_values_to_zbf_payload()`, `pilot_from_zbf()`, `quadratic_reference_phase()`, `spherical_reference_phase()`, `physical_field_from_zbf()`, `parse_native_intermediate_trace()`, `load_segment_geometry()`, `validate_parallel_segment()`, `validate_raw_grid_contract()`, and `validate_convention_validation()`.

- [ ] **Step 1: Write failing tests for the exact Q/Phi formulas and fixed surface mappings**

```python
def test_reference_uses_signed_waist_distance_not_gaussian_curvature():
    grid = UniformGrid2D.centered(nx=4, ny=4, dx_mm=0.1, dy_mm=0.1)
    q = quadratic_reference_phase(grid, wavelength_vacuum_mm=0.02,
                                  refractive_index=2.0,
                                  signed_waist_distance_mm=-2.0)
    phi = spherical_reference_phase(grid, wavelength_vacuum_mm=0.02,
                                    refractive_index=2.0,
                                    signed_waist_distance_mm=-2.0)
    x, y = np.meshgrid(grid.x_mm, grid.y_mm)
    k = 2 * np.pi * 2.0 / 0.02
    np.testing.assert_allclose(q, k * (x*x + y*y) / -4.0)
    np.testing.assert_allclose(phi, -k * (np.sqrt(4.0 + x*x + y*y) - 2.0))


def test_plane_reference_is_zero_inside_rayleigh_range():
    pilot = PilotState(zeta_mm=-0.015, rayleigh_mm=0.761, waist_mm=0.0508)
    grid = UniformGrid2D.centered(nx=8, ny=8, dx_mm=0.01, dy_mm=0.01)
    phases = reference_phases(grid, pilot, wavelength_vacuum_mm=0.01064,
                              refractive_index=1.0)
    np.testing.assert_array_equal(phases.phi_rad, 0.0)
    np.testing.assert_array_equal(phases.q_rad, 0.0)


def test_common_physical_mapping_is_fixed_by_surface_registry():
    beam = make_small_unpolarized_lossless_zbf()
    convention = SurfaceConvention(7, "after", -1)
    mapped = physical_field_from_zbf(
        beam, convention=convention,
        convention_validation=make_in_memory_authoritative_validation_fixture(),
        sample_value_convention="point_value")
    grid = UniformGrid2D.centered(nx=beam.header.nx, ny=beam.header.ny,
                                 dx_mm=beam.header.dx, dy_mm=beam.header.dy)
    x, y = np.meshgrid(grid.x_mm, grid.y_mm)
    zeta = -beam.header.zx
    expected_phi = (2*np.pi*beam.header.refractive_index /
                    beam.header.wavelength_vacuum_mm) * (
        np.sqrt(zeta*zeta+x*x+y*y)-abs(zeta)
    )
    expected = np.conj(beam.ex) * np.exp(1j * expected_phi)
    np.testing.assert_allclose(mapped.physical.values, expected)
```

Define `make_small_unpolarized_lossless_zbf()` locally in `test_field_contract.py`; construct its raw header independently rather than round-tripping through `physical_field_from_zbf()`.
Also add `test_point_and_cell_payload_conversions_have_the_expected_area_factor`, `test_cell_payload_roundtrip_recovers_the_same_point_field`, and parameterized raw-ZBF physical-field oracles covering all five registered surfaces, both axis signs, positive/negative `zeta`, inside/outside/boundary Rayleigh states, and nontrivial complex payloads. Every raw-ZBF oracle uses `conj(Ex)`; no surface-specific switch exists.

Add fail-closed raw-grid evidence tests. A valid even grid must satisfy `MinX=-(Nx/2)Dx`, `MinY=-(Ny/2)Dy`, `X(Nx/2)=Y(Ny/2)=0`, `X(i)=MinX+iDx`, `Y(j)=MinY+jDy`, and `Z(ix,iy)=Values[ix,iy]`. Reject the half-step-shifted ZOSPy DataFrame labels. The API array order is `[x,y]`; conversion to the package/ZBF `[y,x]` order is one explicit transpose.

- [ ] **Step 2: Run the contract tests and verify failure**

Run:

```powershell
python -m pytest tests/free_space_identification/test_field_contract.py tests/free_space_identification/test_geometry.py -q
```

Expected: import failure for the missing modules.

- [ ] **Step 3: Implement explicit pilot and physical-field mappings**

```python
@dataclass(frozen=True)
class PilotState:
    zeta_mm: float
    rayleigh_mm: float
    waist_mm: float

    def __post_init__(self):
        if not np.all(np.isfinite([self.zeta_mm, self.rayleigh_mm, self.waist_mm])):
            raise ValueError("pilot values must be finite")
        if self.rayleigh_mm <= 0.0 or self.waist_mm <= 0.0:
            raise ValueError("pilot Rayleigh distance and waist must be positive")

    @property
    def inside(self) -> bool:
        return abs(self.zeta_mm) < self.rayleigh_mm


@dataclass(frozen=True)
class ReferencePhases:
    q_rad: np.ndarray
    phi_rad: np.ndarray


@dataclass(frozen=True)
class MappedZbfField:
    physical: PointField2D
    reference_relative: np.ndarray
    references: ReferencePhases
    pilot: PilotState
    source_sha256: str
    convention_evidence_sha256: str
    sample_value_convention: SampleValueConvention


def pilot_from_zbf(beam: LosslessZbf, convention: SurfaceConvention) -> PilotState:
    h = beam.header
    if h.units != 0:
        raise ValueError("biconic physical contract requires ZBF millimetre units")
    if not np.allclose([h.zx, h.rx, h.wx], [h.zy, h.ry, h.wy],
                       rtol=1e-10, atol=1e-12):
        raise ValueError("axisymmetric reference contract is not satisfied")
    expected_rayleigh = (np.pi * h.refractive_index * h.wx**2 /
                         h.wavelength_vacuum_mm)
    if not np.isclose(h.rx, expected_rayleigh, rtol=1e-8, atol=1e-12):
        raise ValueError("ZBF Rayleigh distance and waist are inconsistent")
    zeta = convention.axis_sign * h.zx
    return PilotState(zeta_mm=zeta, rayleigh_mm=h.rx, waist_mm=h.wx)


def quadratic_reference_phase(grid, *, wavelength_vacuum_mm, refractive_index,
                              signed_waist_distance_mm):
    if signed_waist_distance_mm == 0.0:
        return np.zeros((grid.ny, grid.nx), dtype=np.float64)
    x, y = np.meshgrid(grid.x_mm, grid.y_mm)
    return (2*np.pi*refractive_index/wavelength_vacuum_mm) * (x*x+y*y) / (2*signed_waist_distance_mm)


def spherical_reference_phase(grid, *, wavelength_vacuum_mm, refractive_index,
                              signed_waist_distance_mm):
    if signed_waist_distance_mm == 0.0:
        return np.zeros((grid.ny, grid.nx), dtype=np.float64)
    x, y = np.meshgrid(grid.x_mm, grid.y_mm)
    z = signed_waist_distance_mm
    r2 = x*x + y*y
    sag = r2 / (np.sqrt(z*z+r2) + abs(z))
    return (2*np.pi*refractive_index/wavelength_vacuum_mm) * np.sign(z) * sag


def reference_phases(grid, pilot, *, wavelength_vacuum_mm, refractive_index):
    if pilot.inside:
        zeros = np.zeros((grid.ny, grid.nx), dtype=np.float64)
        return ReferencePhases(q_rad=zeros.copy(), phi_rad=zeros)
    return ReferencePhases(
        q_rad=quadratic_reference_phase(
            grid, wavelength_vacuum_mm=wavelength_vacuum_mm,
            refractive_index=refractive_index,
            signed_waist_distance_mm=pilot.zeta_mm),
        phi_rad=spherical_reference_phase(
            grid, wavelength_vacuum_mm=wavelength_vacuum_mm,
            refractive_index=refractive_index,
            signed_waist_distance_mm=pilot.zeta_mm),
    )


def zbf_payload_to_point_values(values, grid, *, sample_value_convention):
    values = np.asarray(values, dtype=np.complex128)
    if sample_value_convention == "point_value":
        return values.copy()
    if sample_value_convention == "cell_energy":
        return values / np.sqrt(grid.pixel_area_mm2)
    raise ValueError("unknown ZBF sample-value convention")


def point_values_to_zbf_payload(values, grid, *, sample_value_convention):
    values = np.asarray(values, dtype=np.complex128)
    if sample_value_convention == "point_value":
        return values.copy()
    if sample_value_convention == "cell_energy":
        return values * np.sqrt(grid.pixel_area_mm2)
    raise ValueError("unknown ZBF sample-value convention")


def physical_field_from_zbf(
    beam, *, convention, convention_validation, sample_value_convention
):
    validate_convention_validation(convention_validation, surface=convention.surface)
    if beam.header.is_polarized:
        raise NotImplementedError("biconic scalar ranking requires an approved Jones-field extension")
    grid = UniformGrid2D.centered(nx=beam.header.nx, ny=beam.header.ny,
                                 dx_mm=beam.header.dx, dy_mm=beam.header.dy)
    pilot = pilot_from_zbf(beam, convention)
    refs = reference_phases(
        grid, pilot, wavelength_vacuum_mm=beam.header.wavelength_vacuum_mm,
        refractive_index=beam.header.refractive_index,
    )
    point_payload = zbf_payload_to_point_values(
        beam.ex, grid, sample_value_convention=sample_value_convention)
    reference_relative = np.conj(point_payload)
    physical = PointField2D(reference_relative * np.exp(1j*refs.phi_rad), grid)
    return MappedZbfField(physical=physical, reference_relative=reference_relative,
                          references=refs, pilot=pilot,
                          source_sha256=beam.source_sha256,
                          convention_evidence_sha256=convention_validation.evidence_sha256,
                          sample_value_convention=sample_value_convention)
```

The reference arrays and stored reference-relative array are copied and marked read-only in their dataclass post-initializers.

`geometry.py` must contain its own minimal, tracked parser for the native chief-ray and surface-transfer blocks. It may not import the ignored and untracked `sandbox/biconic_focus_baseline_utils.py`. Preserve every parsed distance token as `ReportNumber(text, value, last_digit_resolution)` so comparison tolerance follows the report's printed precision rather than an arbitrary `isclose` default. Define immutable `SegmentGeometry` records and validate that each segment's transverse basis change is the identity to `1e-9`, that the report's inside/outside states match `SegmentSpec.branch`, and that the physical propagation distance comes from model/report geometry rather than a ZBF pilot-position difference.

Native report distances remain signed in each local ZBF axis. Validate `axis_sign * report_signed_distance == model_distance` and independently validate `report_signed_distance == raw_z_end - raw_z_start`, using the report token's half-last-digit interval plus a small floating parse allowance. Persist all three quantities. In particular, S7→S8 has raw pilot delta about `-368.600001354865 mm`, while the report prints approximately `-368.6 mm`; propagation still uses exactly `368.600000 mm`.

- [ ] **Step 4: Add static evidence and a fail-closed convention-validation contract**

The historical-fixture portion is skipped unless both environment variables are set:

```powershell
$env:BTS_FREE_SPACE_ZBF_DIR='D:\BTS\.worktrees\residual-phase\sandbox\Zemax_baseline'
$env:BTS_FREE_SPACE_REPORT='D:\BTS\.worktrees\residual-phase\sandbox\Zemax_baseline\biconic_focus_test.txt'
```

It must assert the mapped pilot values and signed geometry from the design specification. The native report proves surface side, orientation, signed distance, and OO/OI/IO classification; it does not by itself prove raw complex phasor conversion or the array origin, and the analytic `biconic_phase.txt` must not be mislabeled as a five-surface native phase grid.

Add immutable `RawGridEvidence` and `ConventionValidation` types. `RawGridEvidence` stores the raw ZOS-API `Nx/Ny/MinX/MinY/Dx/Dy`, selected `X/Y/Z/Values` checkpoints, the exact raw-grid array hash, input/output ZBF hashes, model/CFG hashes, run id, evidence origin `synthetic_test|live_zosapi`, and the explicit API `[x,y]` to package `[y,x]` transpose. `validate_raw_grid_contract()` enforces sample-at-zero and rejects half-step labels. `ConventionValidation` records the five `after` sides, axis signs, uniform raw-ZBF `conj(Ex)` phasor contract, raw-grid evidence hashes, report/model/CFG hashes, phase unit, origin, `authoritative` flag, and validation status. Its `evidence_sha256` is the canonical JSON hash of all those fields, not a caller-supplied free string.

`validate_convention_validation()` is fail-closed: missing raw-grid evidence, hash mismatch, a half-step center, a per-surface conjugation switch, a branch/axis mismatch, unknown phase unit, or `authoritative=False` raises and prevents formal physical-field derivation. Unit tests may build an in-memory self-consistent authoritative fixture to exercise the mapping function, but that object has no `ArtifactRef` and cannot enter the receipt-driven pipeline. Any serialized synthetic or smoke receipt has `authoritative=false`; Task 14 rejects it before deserialization into a formal `ConventionValidation`. The static historical-fixture test never emits a receipt, so a pytest skip cannot unlock a live run.

The sample-at-zero rule is independently fixed by the installed OpticStudio manual's `IAR_DataGrid` and ZBF format sections and by the existing raw S4/S6 DataGrid evidence. The formal live run must nevertheless capture raw DataGrid evidence directly through ZOS-API before any main derivation; it may not use `zospy.analyses.base.AnalysisResult.get_data_grid()` or its internal `zospy.utils.zputils.unpack_datagrid()` DataFrame path. Task 12 creates the live evidence and Task 14 requires its immutable receipt before the fresh continuous field can unlock candidate inputs.

- [ ] **Step 5: Run contract, geometry, and existing reference-frame tests**

Run:

```powershell
python -m pytest tests/free_space_identification/test_field_contract.py tests/free_space_identification/test_geometry.py tests/free_space_identification/test_static_fixtures.py tests/test_zbf_source.py -k "not reference_phase_uses_spherical_header_metadata" -q
```

Expected: all new contract/geometry/static tests and the unaffected existing source tests pass; the one known stale Gaussian-curvature assertion remains deselected and is not modified.

- [ ] **Step 6: Commit Task 3**

```powershell
git add sandbox/free_space_algorithm_identification/field_contract.py sandbox/free_space_algorithm_identification/geometry.py tests/free_space_identification/test_field_contract.py tests/free_space_identification/test_geometry.py tests/free_space_identification/test_static_fixtures.py
git commit -m "test: lock physical ZBF field contract"
```

### Task 3A: Harden immutable ZBF fields and nominal grid spacing

This short corrective task closes two dependency risks found while implementing Task 3. It is limited to three high-information regression items and is not a new test matrix.

**Files:**
- Modify: `sandbox/free_space_algorithm_identification/models.py`
- Modify: `sandbox/free_space_algorithm_identification/zbf_binary.py`
- Modify: `tests/free_space_identification/test_models.py`
- Modify: `tests/free_space_identification/test_zbf_binary.py`
- Modify: `tests/free_space_identification/test_field_contract.py` only to move existing invalid-object expectations to construction time; add no test item.

- [ ] **Step 1: Write three focused failing regressions**

1. `LosslessZbf.ex/ey` must use immutable backing storage; item assignment and `setflags(write=True)` both fail.
2. A disk-origin `LosslessZbf` whose stored SHA-256 no longer matches its exact header/payload/tail serialization is rejected. This prevents `dataclasses.replace()` from creating a changed field with stale provenance.
3. For a centered `N=4096` grid with the native S12 interval, `grid.dx_mm` and `grid.dy_mm` reproduce the supplied IEEE-754 values exactly rather than subtracting two large edge coordinates.

- [ ] **Step 2: Record RED, then implement the minimum correction**

Store copied Ex/Ey arrays over immutable byte buffers and validate shape/polarization in `LosslessZbf.__post_init__()` without rejecting preserved NaN payload bits. Recompute the exact serialization hash incrementally over header, immutable Ex/Ey byte buffers, and tail rather than materializing a second full serialized copy; set it for `path=None` and require equality for `path!=None`. In `UniformGrid2D`, cache the nominal step from the sample-at-zero center and its adjacent sample during construction; the public spacing properties return that cached value.

- [ ] **Step 3: Run the focused accumulated gate**

```powershell
python -m pytest tests/free_space_identification/test_models.py tests/free_space_identification/test_zbf_binary.py tests/free_space_identification/test_field_contract.py -q
```

- [ ] **Step 4: Commit Task 3A**

```powershell
git add -f sandbox/free_space_algorithm_identification/models.py sandbox/free_space_algorithm_identification/zbf_binary.py tests/free_space_identification/test_models.py tests/free_space_identification/test_zbf_binary.py tests/free_space_identification/test_field_contract.py
git commit -m "fix: harden diagnostic field provenance"
```

### Task 4: Continuous Fourier convention, CZT evaluation, and band-limited resampling

**Files:**
- Modify: `pytest.ini`
- Create: `sandbox/free_space_algorithm_identification/fourier.py`
- Create: `tests/free_space_identification/test_fourier.py`

**Interfaces:**
- Consumes: `UniformGrid2D` and `PointField2D`.
- Produces: `Spectrum2D`, `forward_continuous_spectrum()`, `evaluate_spectrum_czt()`, `evaluate_field_fourier_czt()`, `resample_bandlimited()`, `resample_lanczos_complex()`, `resample_cubic_complex()`, `point_to_cell_energy()`, and `cell_energy_to_point()`.

- [ ] **Step 1: Write failing direct-sum and common-node tests**

Define `smooth_test_field()`, `direct_inverse_spectrum_sum()`, `direct_forward_field_sum()`, `analytic_bandlimited_complex_field()`, `evaluate_analytic_bandlimited_field()`, and `relative_l2()` locally in `test_fourier.py`. The direct oracles must use explicit dense exponential matrices and physical `dx*dy` or `dfx*dfy`, never SciPy FFT/CZT.

```python
def test_czt_matches_direct_inverse_sum_on_arbitrary_uniform_grid():
    field = smooth_test_field(nx=8, ny=10, dx=0.2, dy=0.15)
    spectrum = forward_continuous_spectrum(field)
    output = UniformGrid2D.centered(nx=7, ny=9, dx_mm=0.07, dy_mm=0.08)
    actual = evaluate_spectrum_czt(spectrum, output, batch_size=3)
    expected = direct_inverse_spectrum_sum(spectrum, output)
    np.testing.assert_allclose(actual.values, expected.values, rtol=1e-11, atol=1e-12)


def test_point_cell_energy_roundtrip_preserves_physical_power():
    field = smooth_test_field(nx=8, ny=8, dx=0.2, dy=0.3)
    samples = point_to_cell_energy(field)
    restored = cell_energy_to_point(samples, field.grid)
    np.testing.assert_allclose(restored.values, field.values, rtol=0, atol=1e-14)
    assert np.isclose(np.sum(abs(samples)**2),
                      np.sum(abs(field.values)**2) * field.grid.pixel_area_mm2)


def test_forward_czt_matches_direct_continuous_fourier_sum():
    field = smooth_test_field(nx=8, ny=10, dx=0.2, dy=0.15)
    fx = -0.73 + np.arange(7) * 0.11
    fy = -0.51 + np.arange(9) * 0.09
    actual = evaluate_field_fourier_czt(field, fx, fy, batch_size=3)
    expected = direct_forward_field_sum(field, fx, fy)
    np.testing.assert_allclose(actual.values, expected, rtol=1e-11, atol=1e-12)


def test_forward_fft_matches_an_independent_natural_grid_sum():
    field = smooth_test_field(nx=7, ny=6, dx=0.2, dy=0.15)
    actual = forward_continuous_spectrum(field)
    expected = direct_forward_field_sum(field, actual.fx_cpm, actual.fy_cpm)
    np.testing.assert_allclose(actual.values, expected, rtol=1e-11, atol=1e-12)


def test_lanczos_and_cubic_checks_interpolate_complex_field_not_wrapped_phase():
    field = analytic_bandlimited_complex_field()
    target = UniformGrid2D.centered(nx=15, ny=17, dx_mm=0.08, dy_mm=0.07)
    lanczos = resample_lanczos_complex(field, target, lobes=8)
    cubic = resample_cubic_complex(field, target)
    expected = evaluate_analytic_bandlimited_field(target)
    assert relative_l2(lanczos.values, expected) < 2e-4
    assert relative_l2(cubic.values, expected) < 2e-3
```

- [ ] **Step 2: Run the Fourier tests and verify failure**

Run:

```powershell
python -m pytest tests/free_space_identification/test_fourier.py -m "not slow" -q
```

Expected: import failure for `fourier.py`.

- [ ] **Step 3: Implement the continuous FFT and separable inverse-CZT convention**

```python
@dataclass(frozen=True)
class Spectrum2D:
    values: np.ndarray
    fx_cpm: np.ndarray
    fy_cpm: np.ndarray
    source_grid: UniformGrid2D

    def __post_init__(self):
        values = np.array(self.values, dtype=np.complex128, copy=True)
        fx = np.array(self.fx_cpm, dtype=np.float64, copy=True)
        fy = np.array(self.fy_cpm, dtype=np.float64, copy=True)
        if values.shape != (fy.size, fx.size):
            raise ValueError("spectrum shape does not match frequency axes")
        # Also require finite values and finite, strictly increasing, uniform axes
        # with at least two points; store all three arrays over immutable backing
        # with object.__setattr__.


def forward_continuous_spectrum(field, *, workers=-1):
    values = scipy.fft.fftshift(
        scipy.fft.fft2(scipy.fft.ifftshift(field.values), workers=workers)
    ) * field.grid.pixel_area_mm2
    fx = scipy.fft.fftshift(scipy.fft.fftfreq(field.grid.nx, field.grid.dx_mm))
    fy = scipy.fft.fftshift(scipy.fft.fftfreq(field.grid.ny, field.grid.dy_mm))
    return Spectrum2D(values=values, fx_cpm=fx, fy_cpm=fy, source_grid=field.grid)


def _build_inverse_zoom(frequencies, coordinates):
    df = float(frequencies[1] - frequencies[0])
    zoom = scipy.signal.ZoomFFT(
        frequencies.size,
        [float(coordinates[0]), float(coordinates[-1])],
        m=coordinates.size,
        fs=1.0/df,
        endpoint=True,
    )
    phase = np.exp(2j*np.pi*frequencies[0]*coordinates)
    return zoom, phase, df


def _apply_inverse_zoom(values, zoom, phase, df, *, axis):
    transformed = np.conj(zoom(np.conj(values), axis=axis))
    shape = [1] * transformed.ndim
    shape[axis] = phase.size
    return transformed * phase.reshape(shape) * df


def evaluate_spectrum_czt(spectrum, output_grid, *, batch_size=128):
    xzoom, xphase, dfx = _build_inverse_zoom(spectrum.fx_cpm, output_grid.x_mm)
    after_x = np.empty((spectrum.fy_cpm.size, output_grid.nx), dtype=np.complex128)
    for y0 in range(0, spectrum.fy_cpm.size, batch_size):
        ys = slice(y0, min(y0 + batch_size, spectrum.fy_cpm.size))
        after_x[ys, :] = _apply_inverse_zoom(
            spectrum.values[ys, :], xzoom, xphase, dfx, axis=1)
    yzoom, yphase, dfy = _build_inverse_zoom(spectrum.fy_cpm, output_grid.y_mm)
    out = np.empty((output_grid.ny, output_grid.nx), dtype=np.complex128)
    for x0 in range(0, output_grid.nx, batch_size):
        xs = slice(x0, min(x0 + batch_size, output_grid.nx))
        out[:, xs] = _apply_inverse_zoom(
            after_x[:, xs], yzoom, yphase, dfy, axis=0)
    return PointField2D(out, output_grid)
```

For arbitrary **uniformly spaced, strictly increasing** output frequency axes, use the paired forward convention. Nonuniform or descending axes are rejected; a future direct-sum/NUFFT interface would be a distinct operator and is outside this task:

```python
def _forward_czt_axis(values, coordinates, frequencies, *, axis):
    dx = float(coordinates[1] - coordinates[0])
    x0 = float(coordinates[0])
    zoom = scipy.signal.ZoomFFT(
        coordinates.size,
        [float(frequencies[0]), float(frequencies[-1])],
        m=frequencies.size,
        fs=1.0/dx,
        endpoint=True,
    )
    transformed = zoom(values, axis=axis)
    phase = np.exp(-2j*np.pi*frequencies*x0)
    shape = [1] * transformed.ndim
    shape[axis] = frequencies.size
    return transformed * phase.reshape(shape) * dx
```

`evaluate_field_fourier_czt()` applies this function along X and Y and returns a `Spectrum2D` whose axes are the requested `fx/fy`; `evaluate_spectrum_czt()` returns `PointField2D`. Implement `resample_bandlimited()` as `forward_continuous_spectrum()` followed by `evaluate_spectrum_czt()`.

Require every coordinate/frequency axis to be one-dimensional, finite, strictly increasing, uniformly spaced, and at least two points. A single axis-normalization helper returns the canonical `start + arange(M)*step` sequence after a small documented ULP-based consistency check; that exact returned array must drive `ZoomFFT`, the external phase factor, and the returned coordinates. Do not let those three consumers use separately rounded nodes. Require `batch_size` to be a positive integer. These contracts need only one compact rejection test covering nonuniform/descending axes and invalid batch size.

Rename the public batching argument to `batch_size`. Prebuild and reuse one `ZoomFFT` object per axis. For the X transform, read at most `batch_size` input rows and write into the single `Ny×Mx` intermediate array. For the Y transform, read at most `batch_size` intermediate columns and write into a preallocated `My×Mx` output. Add an instrumented transformer fake that asserts neither stage ever receives a larger slab. Do not call raw `czt(w=exp(i theta))` for canonical unit-circle production transforms: at N=12288 its accumulated unit-circle error exceeds the `1e-10` common-node gate on this environment.

Add a production-parameter stability regression using one-dimensional `N=12288` axes or a thin rectangular batch, never a `12288^2` field. Compare the natural grid against FFT/IFFT and evaluate 9–17 predeclared native/quarter-S13 coordinates with compensated direct sums, requiring relative complex error at most `1e-10`. Only the zero coordinate is assumed to coincide automatically with the natural grid; do not use nearest neighbours as common nodes. The test is marked `slow` but runs before a field can be labeled an exact baseline.

Register the `slow` marker in `pytest.ini`; normal diagnostic commands use `-m "not slow"`, and Task 15 invokes the production regression explicitly.

Implement the independent continuousization checks on real and imaginary parts of the same slow complex field: separable normalized Lanczos with exactly 8 or 12 lobes, and `scipy.ndimage.map_coordinates(..., order=3, mode="constant", cval=0)` for cubic sensitivity. They never operate on amplitude/phase separately and never define the canonical field. Task 7 compares Fourier, Lanczos-8, Lanczos-12, and cubic results on common nodes; only Fourier is the main definition, while the cross-method spread enters `u_input`.

Document `resample_bandlimited()` as periodic trigonometric interpolation of the supplied finite sample grid. Fixed-window refinement is valid only when X and Y separately satisfy `N_target*delta_target = N_source*delta_source`; finite-window expansion remains the explicit zero-extension operation in Task 5.

- [ ] **Step 4: Run Fourier tests including rectangular X/Y sampling**

Run:

```powershell
python -m pytest tests/free_space_identification/test_fourier.py -m "not slow" -q
```

Expected: all normal Fourier tests pass with common-node relative complex-field error below `1e-10`; the N=12288 production regression is executed under the Task 15 memory gate.

- [ ] **Step 5: Commit Task 4**

```powershell
git add pytest.ini sandbox/free_space_algorithm_identification/fourier.py tests/free_space_identification/test_fourier.py
git commit -m "feat: add continuous Fourier and CZT primitives"
```

### Task 5: Bit-faithful derived ZBF inputs and branch-specific sampling matrices

**Files:**
- Modify: `sandbox/free_space_algorithm_identification/models.py`
- Create: `sandbox/free_space_algorithm_identification/derived_inputs.py`
- Create: `sandbox/free_space_algorithm_identification/sampling.py`
- Create: `tests/free_space_identification/test_derived_inputs.py`
- Create: `tests/free_space_identification/test_sampling.py`

**Interfaces:**
- Consumes: the lossless codec, slow reference-relative `Ex/Ey`, the fixed segment registry, and the continuous Fourier resampler.
- Produces: `DerivedInputValidation`, `DerivedInputResult`, `derive_zbf_input()`, `validate_derived_input()`, `build_segment_sampling_cases()`, and `write_sampling_manifest()`.

- [ ] **Step 1: Write failing strategy, overlap, and sampling-law tests**

The tests must include these names and assertions:

```text
test_native_case_is_a_byte_exact_copy
test_fixed_window_refinement_recovers_original_complex_nodes
test_fixed_step_extension_preserves_original_ex_and_ey_exactly
test_zero_extension_rejects_excessive_edge_energy
test_derived_input_never_applies_power_normalization
test_only_nx_ny_dx_dy_and_payload_may_change
test_s7_s8_fixed_window_refines_input_and_output_sampling
test_s12_s13_fixed_window_n_does_not_refine_stw_output_sampling
test_s12_s13_window_expansion_refines_stw_output_sampling
test_s13_s14_fixed_window_does_not_refine_wts_output_sampling
test_s13_s14_high_resolution_input_depends_on_chained_s12_s13_output
test_native_s13_interpolation_is_not_labeled_physical_convergence
```

The S12→S13 invariant must be tested numerically, not only by checking a label:

```python
def test_s12_s13_fixed_window_n_does_not_refine_stw_output_sampling():
    d0 = stw_output_sampling_mm(
        wavelength_vacuum_mm=0.01064, refractive_index=1.0,
        waist_distance_mm=608.615263635412,
        n=1024, input_dx_mm=0.6339798584,
    )
    d1 = stw_output_sampling_mm(
        wavelength_vacuum_mm=0.01064, refractive_index=1.0,
        waist_distance_mm=608.615263635412,
        n=2048, input_dx_mm=0.6339798584 / 2,
    )
    assert d1 == pytest.approx(d0, rel=1e-14)
```

- [ ] **Step 2: Run the tests and verify the missing-module failures**

```powershell
python -m pytest tests/free_space_identification/test_derived_inputs.py tests/free_space_identification/test_sampling.py -q
```

Expected: import failures for `derived_inputs.py` and `sampling.py`.

- [ ] **Step 3: Implement the five permitted derivations without changing physical normalization**

`derive_zbf_input()` has this exact call contract:

```python
def derive_zbf_input(
    source_path: str | Path,
    output_path: str | Path,
    *,
    target_grid: UniformGrid2D,
    strategy: DerivationStrategy,
    convention: SurfaceConvention,
    sample_value_convention: SampleValueConvention,
    max_edge_energy_fraction: float = 1e-10,
) -> DerivedInputResult:
```

`SamplingCase.source` remains a logical `NativeSurfaceSource` or `CaseOutputSource` in the frozen manifest. The pipeline resolves it to a verified current-run `ArtifactRef` immediately before calling `derive_zbf_input()`; the path argument is never serialized as a guessed future file path.

Implement the strategies as follows:

- `exact_copy`: require identical `N`, `dx`, and `dy`, then use `shutil.copy2`; verify byte identity.
- `fourier_refine_fixed_window`: resample the slow ZBF `Ex` and `Ey` separately with the same Fourier operator; require `N_target dx_target = N_source dx_source` independently in X/Y.
- `zero_extend_fixed_sampling`: require unchanged `dx/dy`, insert the original arrays at indices `M//2-N//2 : M//2+N//2`, and set every added sample to exact complex zero.
- `zero_extend_then_fourier_refine`: first perform exact centered zero extension at the original sampling, then Fourier-refine the extended slow field.
- `chained_zemax_output`: byte-copy the named upstream output and record its upstream run/case hash; never re-encode it.

Decode source payloads to point values using the frozen sample-value convention for every Fourier interpolation. Under `point_value`, fixed-window common raw nodes remain unchanged; under `cell_energy`, their deterministic raw factor is `sqrt(dA_target/dA_source)` while decoded point values remain unchanged. This area conversion is not a fitted power normalization.

For `zero_extend_fixed_sampling`, `dx`, `dy`, and therefore `dA` are unchanged: copy the original raw Ex/Ey arrays directly and bit-for-bit into the centered overlap, then fill only the new samples with exact complex zero. Do not divide by and multiply by `sqrt(dA)` on this path, because that numerically redundant round trip can change a payload by one ulp. Exact-step checks use the nominal cached `UniformGrid2D.dx_mm/dy_mm` from Task 3A and the raw header bits, not subtraction of two large edge coordinates.

Do not crop a window, apply an apodizer, interpolate wrapped phase, interpolate a physical total field whose carrier is under-sampled, or normalize power. Patch only the four sampling header fields; preserve all untouched header bits and trailing bytes through the lossless writer.

- [ ] **Step 4: Implement and enforce the derived-input gates**

For each component and both axes, persist the source/output hashes, raw header diff, slow-field overlap error, physical-field overlap phase error, intensity error, energy error, and edge-energy fraction. The hard gates are:

```text
unexpected header-byte changes                 = 0
fixed-window decoded point-field common-node relative L2 <= 1e-10
fixed-window physical-phase RMS                 <= 1e-8 wave
fixed-window normalized-intensity RMS            <= 1e-6 percent
pixel-area-weighted relative energy error        <= 1e-10
fixed-step overlap Ex/Ey difference              = 0 bitwise because dA is unchanged
added fixed-step samples                         = exact complex zero
outer 5-percent edge-energy target               <= 1e-12
outer 5-percent edge-energy hard gate            <= 1e-10
```

If the edge gate fails, stop that case; do not introduce a window function to make it pass.

- [ ] **Step 5: Build the fixed predeclared matrix for the three propagation branches**

Use the exact pilot distances from the design and compute X/Y independently. The minimum matrix is:

All sampling-law functions take vacuum wavelength and refractive index explicitly and use `lambda_medium = wavelength_vacuum_mm / refractive_index`; tests include `n != 1` even though this biconic case has `n=1`.

| Segment | Sequence | Input construction | Purpose |
|---|---|---|---|
| S7→S8 | `native`, `R2`, `R4` | same S7 window; N=1024, 2048, 4096 | input and output sampling convergence |
| S7→S8 | `W2` | N=2048 at original S7 `dx/dy`, zero extended | finite-window control |
| S12→S13 | `ZI0`, `ZI1`, `ZI2` | fixed S12 window; N=1024, 2048, 4096 | input discretization only |
| S12→S13 | `ZO0`, `ZO1`, `ZO2` | fixed S12 `dx/dy`; N=1024, 2048, 4096 | S13 output sampling |
| S12→S13 | `ZJ2` | N=4096, `dx/dy` halved, window doubled | joint check |
| S13→S14 | `native` | native S13 ZBF | native control |
| S13→S14 | `input_R2`, `input_R4` | corresponding fresh S12→S13 `ZO1/ZO2` S13 outputs | recover input-waist information |
| S13→S14 | `interp_sensitivity_R2`, `interp_sensitivity_R4` | Fourier interpolation of the fresh native S13 field | sensitivity only; never convergence truth |
| S13→S14 | `output_O2` | native S13 zero-extended at fixed `dx/dy` | output sampling only |
| S13→S14 | `combined` | fresh `input_R2` S13 output, then fixed-step extension to N=4096 | input and output sampling |

The two branch laws are mandatory assertions:

\[
\Delta_{8,j}=\frac{D_8}{D_7}\Delta_{7,j},
\qquad
\Delta_{13,j}=\frac{\lambda D_{12}}{N\Delta_{12,j}},
\qquad
\Delta_{14,j}=\frac{\lambda D_{14}}{N\Delta_{13,j}}.
\]

The S12 fixed-window `ZI` sequence improves input discretization but leaves the focal output interval unchanged. The high-resolution S13 inputs used to establish S13→S14 convergence must come from the matching fresh upstream S12→S13 ZOS run. Fourier interpolation of the native five-sample-waist S13 field is allowed only as a labeled sensitivity check and cannot establish physical convergence.

Set `repeat_count=2` for every native 1024 segment case and for the highest planned sampling case on each independent convergence axis; other trend cases use `repeat_count=1`. Repeats share byte-identical input and settings but use fresh ZOS connections and distinct anchored output prefixes. The manifest freezes these counts before live execution.

Compare Zemax convergence on a common terminal grid by first removing the same terminal `Phi_t` from every physical field, resampling that slow complex field, and restoring `Phi_t`; never interpolate wrapped phase. Test ZI and ZO axes separately. For S12→S13, both highest-level ZI and ZO phase differences must be at most `2e-5` wave. For all segments, complex, intensity, and power differences must decrease on the last two levels and enter their separate `u_Zemax_sampling`; this numerical gate permits a provisional ROI. After candidate generation, a separate final stability stage repeats 3u/5u decisions at the top two Zemax levels and all `Omega_-2/-3/-6` regions. Any changed status makes the sequence non-converged and stops unique ranking; this deferred check avoids a dependency cycle.

- [ ] **Step 6: Run derived-input and matrix tests**

```powershell
python -m pytest tests/free_space_identification/test_derived_inputs.py tests/free_space_identification/test_sampling.py tests/free_space_identification/test_zbf_binary.py -q
```

Expected: all tests pass, including polarized X/Y-asymmetric fixtures.

- [ ] **Step 7: Commit Task 5**

```powershell
git add sandbox/free_space_algorithm_identification/models.py sandbox/free_space_algorithm_identification/derived_inputs.py sandbox/free_space_algorithm_identification/sampling.py tests/free_space_identification/test_derived_inputs.py tests/free_space_identification/test_sampling.py
git commit -m "feat: define faithful ZBF sampling matrices"
```

### Task 6: Frozen comparison regions, physical complex-field metrics, and 3u/5u decisions

**Files:**
- Create: `sandbox/free_space_algorithm_identification/metrics.py`
- Create: `sandbox/free_space_algorithm_identification/decision.py`
- Create: `tests/free_space_identification/test_metrics_decision.py`

**Interfaces:**
- Consumes: mapped physical fields on a common terminal grid and convergence deltas from later tasks.
- Produces: `FrozenRoiSet`, `ComparisonMetrics`, `MetricUncertainty`, `PairDecision`, `build_frozen_rois()`, `compare_physical_fields()`, `symmetric_complex_distance()`, `combine_uncertainty()`, and `classify_pair()`.

- [ ] **Step 1: Write failing ROI and no-fitting metric tests**

Define the small two-component fields, peak indices, and full-ROI constructor locally in `test_metrics_decision.py`; none may call the ROI selection function under test to compute its expected mask.

```python
def test_roi_is_the_peak_connected_component_of_only_the_zemax_reference():
    zemax = two_component_reference_field()
    candidate = candidate_with_a_different_bright_component()
    rois = build_frozen_rois(
        zemax, reference_zbf_sha256="a"*64,
        thresholds=(1e-3, 1e-2, 1e-6))
    assert rois.primary.threshold == 1e-3
    assert rois.primary.mask[zemax_peak_index()]
    assert not rois.primary.mask[secondary_component_index()]
    assert np.array_equal(rois.primary.mask,
                          build_frozen_rois(
                              zemax, reference_zbf_sha256="a"*64,
                              thresholds=(1e-3, 1e-2, 1e-6)).primary.mask)


def test_symmetric_distance_retains_real_amplitude_error():
    a = np.ones((4, 4), dtype=np.complex128)
    b = 2.0 * a
    mask = np.ones_like(a, dtype=bool)
    assert symmetric_complex_distance(a, b, mask=mask, pixel_area_mm2=1.0) > 0.0


def test_only_one_phase_piston_is_removed():
    reference = smooth_test_field(nx=16, ny=16, dx=0.1, dy=0.1)
    x, y = np.meshgrid(reference.grid.x_mm, reference.grid.y_mm)
    candidate = PointField2D(reference.values * np.exp(1j*(0.4 + 0.2*x*x)), reference.grid)
    result = compare_physical_fields(candidate, reference, frozen_full_roi(reference.grid))
    assert result.phase_rms_waves > 1e-4
```

Also add `test_roi_threshold_dependent_decision_is_undecided` and `test_candidate_pair_uncertainty_excludes_zemax_repeat_error`.

- [ ] **Step 2: Run metric tests and verify import failure**

```powershell
python -m pytest tests/free_space_identification/test_metrics_decision.py -q
```

Expected: collection fails for the missing modules.

- [ ] **Step 3: Implement the fixed physical metrics from the approved design**

Use these immutable result contracts:

```python
@dataclass(frozen=True)
class FrozenRoi:
    threshold: float
    mask: np.ndarray
    mask_sha256: str
    reference_zbf_sha256: str
    grid: UniformGrid2D


@dataclass(frozen=True)
class FrozenRoiSet:
    primary: FrozenRoi
    checks: tuple[FrozenRoi, FrozenRoi]


@dataclass(frozen=True)
class ComparisonMetrics:
    threshold: float
    piston_rad: float
    complex_relative_l2: float
    symmetric_distance: float
    phase_rms_waves: float
    phase_pv_waves: float
    intensity_relative_l2: float
    full_window_power_relative_error: float
    outside_roi_energy_fraction: float
    coherence_magnitude: float
    unwrap_status: Literal["not_needed", "unique", "ambiguous"]


@dataclass(frozen=True)
class MetricUncertainty:
    complex_distance: float
    phase_waves: float
    intensity_relative: float
    power_relative: float


class PairDecision(str, Enum):
    CONSISTENT = "consistent"
    UNDECIDED = "undecided"
    UNDECIDED_ROI_SENSITIVITY = "undecided_roi_sensitivity"
    EXCLUDED = "excluded"
```

The public signature is `build_frozen_rois(reference: PointField2D, *, reference_zbf_sha256: str, thresholds: tuple[float, float, float] = (1e-3, 1e-2, 1e-6)) -> FrozenRoiSet`; the artifact hash is mandatory.

Build each ROI from the highest-sampling Zemax endpoint that has already passed identity and convergence. Use the eight-connected component containing the global intensity peak. Persist the endpoint ZBF hash, grid, threshold, mask hash, and selection rule. A candidate may never define or alter the ROI.

Predeclare the primary ROI source as S7→S8 `R4`, S12→S13 `ZO2` after independent ZI convergence, and S13→S14 `combined`. If the installed Zemax sampling enum makes a named case impossible, use its highest manifest predecessor and record the downgrade before comparisons; never choose the source by inspecting which ROI favors a candidate.

For each candidate, compute one piston from the primary ROI,

\[
\alpha=\arg\sum_{\Omega_{-3}}U_mU_Z^*\,\Delta A,
\qquad \widetilde U_m=U_m e^{-i\alpha},
\]

and reuse that same piston for all three ROIs and all metrics. Implement exactly:

\[
\epsilon_E=\sqrt{\frac{\sum_\Omega|\widetilde U_m-U_Z|^2\Delta A}
{\sum_\Omega|U_Z|^2\Delta A}},
\]

\[
\epsilon_{\phi,\mathrm{waves}}=
\sqrt{\frac{\sum_\Omega I_Z[\operatorname{Arg}(\widetilde U_mU_Z^*)/(2\pi)]^2\Delta A}
{\sum_\Omega I_Z\Delta A}},
\]

\[
\epsilon_I=\sqrt{\frac{\sum_\Omega(|U_m|^2-|U_Z|^2)^2\Delta A}
{\sum_\Omega|U_Z|^4\Delta A}}.
\]

Also report full-window power error, energy outside the primary ROI, and complex coherence. If the ratio crosses the phase branch, use a deterministic peak-anchored two-dimensional unwrap only to add integer cycles; mark phase metrics undecidable if the unwrap is not unique. The complex metric remains valid. Candidate decisions must be repeated on `Omega_-2`, `Omega_-3`, and `Omega_-6`; if support/exclusion status changes with threshold, return `undecided_roi_sensitivity` and stop unique attribution.

The symmetric field distance used for candidate separation is

\[
D(A,B)=\min_\theta\sqrt{
\frac{2\sum_\Omega|Ae^{-i\theta}-B|^2\Delta A}
{\sum_\Omega(|A|^2+|B|^2)\Delta A}}.
\]

Do not independently normalize `A` and `B`. Do not fit amplitude, shift, flip, rotation, tilt, defocus, astigmatism, quartic phase, a Zernike basis, propagation distance, reference radius, or fingerprint scale.

- [ ] **Step 4: Implement dimensioned uncertainty and the exact gray zone**

Maintain separate uncertainty records for complex distance, phase waves, relative intensity, and power. Split them into candidate-specific numerical error `u_m`, Zemax observation/repeat error `u_Z`, and one common coordinate/input-projection term `u_common`. For candidate-to-observation comparison use

\[
u_{Zm}=u_Z+u_m+u_{\rm common},
\]

where `u_m=u_input+u_grid+u_window+u_output`. For candidate-to-candidate separation use

\[
u_{AB}=u_A+u_B+u_{\rm common},
\]

and do not insert Zemax repeat/sampling error into `u_AB`. For two Zemax repeats use their own readback, identity, coordinate, and repeat components exactly once. This separation is applied independently to all four metric dimensions.

Use nonzero numerical floors `1e-12` for complex distance and `1e-10` wave for phase, recorded as numerical bounds rather than statistical standard deviations. `classify_pair()` must return:

```text
D <= 3u       consistent
3u < D <= 5u  undecided
D > 5u        excluded/separated
```

A unique candidate requires both `D(Z,A) <= 3u_ZA` and, for every other candidate B, `D(Z,B) > 5u_ZB` and `D(A,B) > 5u_AB`. The smallest error alone is never a decision. Apply analogous gates separately to phase, intensity, power, and mandatory analytic identities.

- [ ] **Step 5: Add the S13→S14 analytic structure gate**

For `R_Phi_given_Q`, assert pointwise intensity and power equality to `F_Q` within their numerical bounds, and assert that their phase ratio equals the precomputed `Phi14-Q14` plus the one allowed piston. A failure above `5u` excludes this candidate even if one scalar phase RMS happens to be small. Do not apply this identity to `R_Phi_given_Phi`, which can change intensity.

- [ ] **Step 6: Run the metric/decision tests**

```powershell
python -m pytest tests/free_space_identification/test_metrics_decision.py -q
```

Expected: all tests pass, including exact checks at `3u`, just above `3u`, exactly `5u`, and just above `5u`.

- [ ] **Step 7: Commit Task 6**

```powershell
git add sandbox/free_space_algorithm_identification/metrics.py sandbox/free_space_algorithm_identification/decision.py tests/free_space_identification/test_metrics_decision.py
git commit -m "feat: add physical field decisions with uncertainty"
```

### Task 7: Band-limited exact Helmholtz angular-spectrum baseline

**Files:**
- Modify: `sandbox/free_space_algorithm_identification/sampling.py`
- Create: `sandbox/free_space_algorithm_identification/asm.py`
- Modify: `tests/free_space_identification/test_sampling.py`
- Create: `tests/free_space_identification/test_asm.py`

**Interfaces:**
- Consumes: a sufficiently sampled physical `PointField2D`, a target `UniformGrid2D`, and the continuous Fourier/CZT primitives.
- Produces: `AsmDiagnostics`, `helmholtz_delta_k()`, `matsushima_bandlimit_mask()`, `estimate_exact_peak_bytes()`, `available_memory_bytes_windows()`, `apply_helmholtz_transfer_inplace()`, and `propagate_bl_asm()`.

- [ ] **Step 1: Write failing analytic-kernel and direct-transform tests**

Include all of the following:

```text
test_fft_bin_plane_wave_acquires_exact_helmholtz_phase
test_evanescent_branch_decays_and_never_grows
test_negative_distance_is_rejected_for_this_experiment
test_stable_delta_k_matches_mpmath_80_digit_paraxial_reference
test_bandlimit_keeps_inside_mode_and_removes_outside_mode
test_clipped_spectral_energy_is_reported
test_small_same_grid_result_matches_explicit_fft_kernel_ifft
test_arbitrary_output_grid_matches_direct_inverse_spectral_sum
test_rectangular_sampling_uses_independent_x_and_y_frequencies
test_low_na_gaussian_matches_the_analytic_gaussian_solution
test_peak_memory_estimate_counts_fft_zoom_intermediate_and_output
test_memory_gate_fails_before_allocating_a_large_field
```

The plane-wave and evanescent tests must compare the full complex ratio, not only intensity. Use `mpmath` at 80 decimal digits as the independent cancellation oracle; Windows `np.longdouble` is not wider than float64 on this host. The analytic Gaussian test may remove one known global axial phase, but it may not fit defocus or beam radius.

- [ ] **Step 2: Run the ASM tests and verify import failure**

```powershell
python -m pytest tests/free_space_identification/test_asm.py tests/free_space_identification/test_fourier.py -m "not slow" -q
```

Expected: collection fails for `asm.py`.

- [ ] **Step 3: Implement the stable Helmholtz branch and finite-window bandlimit**

Let `lambda_medium = wavelength_vacuum_mm / refractive_index`, `k=2π/lambda_medium`, `kx=2πfx`, `ky=2πfy`, and `kappa2=kx²+ky²`. Use

```text
kappa2 <= k2: kz = +sqrt(k2-kappa2)
kappa2 >  k2: kz = +i sqrt(kappa2-k2)
Re(kz) >= 0 and Im(kz) >= 0
```

For all bins calculate the carrier-removed phase with the stable identity

\[
\Delta k=k_z-k=-\frac{\kappa^2}{k_z+k},
\qquad H=\exp(i d\Delta k).
\]

For an evanescent bin `kz=i alpha`, this same kernel is the full complex value `exp(-alpha*d) exp(-i*k*d)`, not merely its decay magnitude. Record bins that underflow in complex128. Never select the negative-imaginary branch. Reject `distance_mm <= 0` in this experiment.

The predeclared rectangular bandlimit is

\[
f_{\mathrm{lim},j}=\min\left[\frac{1}{2\Delta j},
\frac{1}{\lambda_{\rm medium}\sqrt{1+(2d/L_j)^2}}\right],
\quad j\in\{x,y\}.
\]

Persist both cutoffs, the exact two-dimensional Boolean mask definition, and

\[
\eta_{\rm clipped}=\frac{\sum|\widehat U|^2(1-W)}{\sum|\widehat U|^2}.
\]

The bandlimit is not evidence of window convergence. If its field effect exceeds the uncertainty allocation, expand the input window.

For each accepted top window, evaluate the same output once with the predeclared mask and once with the unmasked spectrum solely to quantify the mask effect. Require primary-ROI mask-induced phase RMS at most `1e-6` wave; record its complex/intensity/power changes in the corresponding uncertainty components. If this allocation fails, enlarge the window and start a new exact-baseline case rather than disabling the mask.

- [ ] **Step 4: Implement memory-bounded propagation to arbitrary output nodes**

`propagate_bl_asm()` must:

1. call `forward_continuous_spectrum()` once;
2. apply the bandlimit and Helmholtz kernel in-place in row batches;
3. evaluate the propagated spectrum directly on the requested output grid with `evaluate_spectrum_czt()`;
4. avoid simultaneously retaining separate full-size input, spectrum, kernel, propagated spectrum, and output arrays;
5. persist peak resident-memory estimates and actual array shapes/dtypes.

The caller constructs a finely sampled physical field by first interpolating the slow ZBF field and only then multiplying the analytic `exp(i Phi_s)`. It must never form an under-sampled physical carrier on the native S7 or S12 grid and then interpolate it.

For S12, construct four independently labeled continuous inputs from periodic Fourier interpolation, Lanczos-8, Lanczos-12, and cubic complex interpolation before multiplying `exp(i Phi12)`. Propagate all four at the common accepted grid/output nodes. Fourier is canonical; the maximum Fourier/Lanczos spread is the input-continuousization gate and enters `u_input`, while cubic is reported only as sensitivity. None may be selected after viewing Zemax error.

- [ ] **Step 5: Encode the exact-baseline convergence matrices**

Use model distances, not pilot-position differences:

| Segment | Step-size sequence | Window sequence |
|---|---|---|
| S7→S8 | L=256 mm; N=6144, 8192, 10240, 12288 | (L,N)=(256,10240),(320,12800) at dx=0.025 mm |
| S12→S13 | L=256 mm; N=6144, 8192, 10240, 12288 | dx=dy=0.025 mm; (L,N)=(192,7680),(224,8960),(256,10240) |
| S13→S14 | L=4.234 mm; N=2048, 4096, 8192 | fix dx=4.234/4096 mm; N=4096,6144,8192, giving displayed L=4.234,6.351,8.468 mm |

For every segment, use ZoomFFT evaluation to the same predeclared terminal metric grid for every step/window level before applying Task 6 metrics. Never compare arrays on their own changing grids. For S12→S13 additionally evaluate native S13 spacing, half spacing, and quarter spacing. Common-node complex error must be at most `1e-10`; common-node phase error must be at most `1e-9` wave. The S12 hard gates are:

```text
input continuousization cross-method phase RMS <= 3e-6 wave
N=10240 versus N=12288 phase RMS              <= 3e-6 wave
L=224 versus L=256 phase RMS                  <= 1e-6 wave
N=10240 versus N=12288 normalized intensity RMS <= 0.003 percent
L=224 versus L=256 normalized intensity RMS     <= 0.001 percent
```

The `N=6144` and `L=192` S12 cases are trend diagnostics and cannot define truth. For S7, use N=10240↔12288 for `u_grid` and (256,10240)↔(320,12800) for `u_window`. For S12 use N=10240↔12288 and L=224↔256. For S13 use N=4096↔8192 at fixed L for `u_grid` and N=6144↔8192 at fixed dx for `u_window`. Apply the same frozen ROI, one piston, phase/intensity/power definitions from Task 6. Store separate complex, phase, intensity, and power deltas; for every segment, the conservative summed exact-baseline phase uncertainty must be at most `1e-5` wave before the result is labeled an accuracy baseline.

Before any N≥8192 allocation, calculate a conservative peak including the input physical field, FFT work/output, one mutable spectrum, row-batched transfer data, both ZoomFFT convolution workspaces, the `Ny×Mx` intermediate, and `My×Mx` output. Query Windows available physical RAM with standard-library `ctypes`/`GlobalMemoryStatusEx` and require `available_bytes >= 1.3 * estimated_peak_bytes`. Persist both values. If the gate fails, write a failed numerical receipt and stop; paging or disk-backed arrays may not establish the canonical exact baseline.

- [ ] **Step 6: Run offline ASM tests**

```powershell
python -m pytest tests/free_space_identification/test_asm.py tests/free_space_identification/test_fourier.py -m "not slow" -q
```

Expected: all ASM and Fourier tests pass. Command-line smoke execution is added only after `cli.py` exists in Task 14.

- [ ] **Step 7: Commit Task 7**

```powershell
git add sandbox/free_space_algorithm_identification/sampling.py sandbox/free_space_algorithm_identification/asm.py tests/free_space_identification/test_sampling.py tests/free_space_identification/test_asm.py
git commit -m "feat: add convergent Helmholtz baseline"
```

### Task 8: Sparse first Rayleigh–Sommerfeld cross-check

**Files:**
- Create: `sandbox/free_space_algorithm_identification/rayleigh_sommerfeld.py`
- Create: `tests/free_space_identification/test_rayleigh_sommerfeld.py`

**Interfaces:**
- Consumes: the identical finely sampled physical input used by ASM and 9–17 predeclared target points.
- Produces: `RsPointResult`, `RsDiagnostics`, `rs1_kernel()`, `select_rs_points()`, and `propagate_rs1_points()`.

- [ ] **Step 1: Write failing strict-kernel and blocked-sum tests**

```text
test_axial_removed_kernel_times_exp_ikd_equals_full_kernel
test_on_axis_kernel_has_the_rs1_sign_and_far_field_limit
test_stable_r_minus_d_matches_high_precision_value
test_blocked_sum_matches_explicit_small_double_sum
test_rectangular_grid_uses_dx_dy_and_cell_center_weights
test_trapezoid_edge_rule_is_reported_as_quadrature_sensitivity
test_smooth_gaussian_converges_under_delta_halving
test_sparse_result_reports_only_point_set_errors
```

Use `mpmath` at 80 decimal digits for the independent `R-d` and kernel-phase cancellation oracles; do not use `np.longdouble` on Windows.

- [ ] **Step 2: Run the tests and verify import failure**

```powershell
python -m pytest tests/free_space_identification/test_rayleigh_sommerfeld.py -q
```

Expected: collection fails for `rayleigh_sommerfeld.py`.

- [ ] **Step 3: Implement the approved RS-I kernel with stable axial-phase removal**

For `exp(-iωt)`, forward `exp(+ikz)`, parallel planes, and positive model distance `d`, implement

\[
R=\sqrt{d^2+\Delta x^2+\Delta y^2},
\qquad R-d=\frac{\Delta x^2+\Delta y^2}{R+d},
\]

\[
h'_{\rm RS1}=\frac{d(1-ikR)}{2\pi R^3}\exp[i k(R-d)].
\]

The non-removed form is `h' exp(ikd)`. Include exactly one source-plane obliquity factor `d/R`; do not multiply a second target-plane factor and do not reverse `1-ikR` to manufacture a phase match.

- [ ] **Step 4: Implement complex128 sparse quadrature with bounded memory**

The ZBF/FFT samples represent the same half-open cell-centered grid used by the continuous Fourier sum, so the canonical RS Riemann quadrature uses uniform cell weights and physical `dx*dy`. Run a second composite-trapezoid edge rule as quadrature sensitivity and include its spread in the sparse-point uncertainty; do not silently mix the two conventions. Use row-blocked source evaluation and pairwise or compensated complex summation. Never allocate an array shaped `n_target × ny × nx`. Use the model distances `368.600000`, `608.600000`, and `2.000000` mm.

Select 9–17 points from the predeclared high-intensity ROI: center, symmetric X/Y axis points, and diagonal points whose reference intensity remains above the fixed threshold. Persist coordinates before evaluating either ASM or RS.

- [ ] **Step 5: Add the formal three-segment point-set convergence gates**

Use the identical finest-input definitions as ASM. For S7→S8 run L=256 mm at N=10240 and 12288. For S12→S13 run L=256 mm at N=8192, 10240, and 12288. For S13→S14 run L=4.234 mm at N=4096 and 8192. Use the same predeclared center/axis/diagonal point coordinates within each segment's fixed ROI and require for every segment:

```text
highest two RS levels point-set phase RMS       <= 5e-6 wave
finest RS versus finest ASM point-set phase RMS <= 5e-6 wave
finest RS versus finest ASM max phase           <= 5e-6 wave
max_m abs(|U_RS|/|U_ASM|-1)                     <= 1e-4
```

Because ASM and RS already use the same axial-carrier removal, save the raw complex ratio at every point and apply no fitted point-set piston in the hard kernel/sign gate. A one-piston diagnostic may be reported separately but cannot replace the raw gate. Point RMS is equal-weight over the predeclared high-intensity points; the amplitude limit is the maximum pointwise relative magnitude error shown above. Apply the same amplitude definition to highest-two-level RS convergence.

Call these values “稀疏 RS-I 点集误差”. They do not bound full-field `epsilon_E`, intensity, power, or unmeasured points, and they share the same input-continuousization uncertainty as ASM. If final classification depends on unsampled full-field structure, add a separately converged full-field RS convolution or enlarge the uncertainty and mark that evidence unverified. A segment that misses its RS matrix is explicitly `RS sparse unverified`; S12 evidence may not be extrapolated to S7 or S13.

- [ ] **Step 6: Run offline RS tests**

```powershell
python -m pytest tests/free_space_identification/test_rayleigh_sommerfeld.py -q
```

Expected: all RS kernel and quadrature tests pass. Command-line smoke execution is added in Task 14.

- [ ] **Step 7: Commit Task 8**

```powershell
git add sandbox/free_space_algorithm_identification/rayleigh_sommerfeld.py tests/free_space_identification/test_rayleigh_sommerfeld.py
git commit -m "feat: add sparse RS-I verification"
```

### Task 9: Independent scaled Fresnel propagation and the three fixed reference candidates

**Files:**
- Create: `sandbox/free_space_algorithm_identification/fresnel.py`
- Create: `sandbox/free_space_algorithm_identification/candidates.py`
- Create: `tests/free_space_identification/test_fresnel_candidates.py`

**Interfaces:**
- Consumes: the same physical start field, start pilot only, fixed model distance, the corresponding current-run same-case captured target-ZBF paired reference, and actual X/Y grids.
- Produces: `CandidateResult`, `propagate_scaled_fresnel()`, `propagate_ptp_fresnel()`, `scaled_dft_cell_samples()`, `run_stock_proper_fq()`, `candidate_f_q()`, `candidate_r_phi_given_q()`, and `candidate_r_phi_given_phi()`.

- [ ] **Step 1: Write failing direct-integral, normalization, ordering, and structural tests**

```text
test_scaled_fresnel_czt_matches_dense_complex_direct_integral
test_ptp_fresnel_mode_matches_explicit_discrete_periodic_fft_kernel
test_zero_padded_ptp_converges_to_scaled_finite_domain_fresnel
test_scaled_dft_roundtrip_and_power_for_rectangular_grid
test_scaled_dft_output_sampling_uses_x_and_y_separately
test_stw_phase_is_after_dft_and_wts_phase_is_before_dft
test_candidate_target_q_uses_predicted_pilot_not_target_fit
test_candidate_phi_target_hash_matches_the_same_captured_case
test_all_candidates_receive_the_same_physical_input_hash
test_stock_proper_fq_matches_independent_fresnel_on_each_branch
test_stock_proper_restores_globals_and_sets_beam_type_old
test_s13_s14_r_phi_given_q_has_fq_intensity_exactly
test_s13_s14_r_phi_given_q_phase_is_phi14_minus_q14
test_r_phi_given_phi_replaces_every_applicable_internal_phase
```

The dense direct-integral oracle must compare absolute complex amplitude; it may not normalize the endpoint or compare phase alone.

- [ ] **Step 2: Run the candidate tests and verify import failure**

```powershell
python -m pytest tests/free_space_identification/test_fresnel_candidates.py -q
```

Expected: collection fails for `fresnel.py` and `candidates.py`.

- [ ] **Step 3: Implement the independent physical-field Fresnel operator**

Every candidate returns:

```python
@dataclass(frozen=True)
class CandidateResult:
    segment_key: str
    operator_id: Literal["H", "F_Q", "R_Phi_given_Q", "R_Phi_given_Phi"]
    input_sha256: str
    input_grid_sha256: str
    output: PointField2D
    predicted_target_zeta_mm: float
    diagnostics: dict[str, float | str | bool]
```

The pipeline rejects a segment matrix unless all candidate `input_sha256` and `input_grid_sha256` values are identical.

With the common axial factor `exp(ikd)` removed consistently, implement

\[
U_t(x_t,y_t)=\frac{1}{i\lambda d}
e^{ik(x_t^2+y_t^2)/(2d)}
\iint U_s(x_s,y_s)e^{ik(x_s^2+y_s^2)/(2d)}
e^{-i2\pi(x_sx_t+y_sy_t)/(\lambda d)}dx_sdy_s.
\]

Use `evaluate_field_fourier_czt()` at `fx=x_t/(lambda*d)` and `fy=y_t/(lambda*d)`. Retain `1/(i lambda d)`, both chirps, and the physical integration area. This scaled finite-domain operator must close in absolute complex amplitude against the dense small-grid integral oracle.

Test the same-grid paraxial transfer function `exp[-i d(kx²+ky²)/(2k)]` separately against periodic FFT-bin modes and an explicit discrete FFT kernel. It is a periodic convolution and is not required to equal an unpadded finite-domain integral on a small grid. Demonstrate their continuous equivalence only after zero-padding and independent step/window convergence.

- [ ] **Step 4: Implement the cell-energy-scaled STW/WTS building blocks**

For signed distance `d`, natural output sampling is

\[
\Delta x'=\frac{\lambda|d|}{N_x\Delta x},
\qquad \Delta y'=\frac{\lambda|d|}{N_y\Delta y}.
\]

Convert point values to cell-energy samples before the centered transform. For `d>0`, use centered `fft2/sqrt(Nx Ny)`; for `d<0`, use centered `sqrt(Nx Ny)*ifft2`. Divide by the output cell-area square root afterward. This normalization is physical and may not be replaced by endpoint amplitude fitting.

`D_d` is the PROPER unitary DFT convention. Relative to the complete carrier-removed scaled Fresnel integral on its natural grid it has the predeclared unit-modulus factor `-i*sgn(d)`; record and remove only that known factor when cross-checking the two definitions. `propagate_scaled_fresnel()` always retains `1/(i lambda d)` and absolute amplitude. Never add or omit the factor case-by-case after seeing an endpoint.

Define

\[
q_d=\frac{k(x^2+y^2)}{2d},
\qquad
\varphi_d=k\operatorname{sgn}(d)
\frac{x^2+y^2}{\sqrt{d^2+x^2+y^2}+|d|}.
\]

Then implement `S_Q=M_q(output) D_d`, `W_Q=D_d M_q(input)`, `S_Phi=M_phi(output) D_d`, and `W_Phi=D_d M_phi(input)`. STW phase is after the transform; WTS phase is before it. `T_F(p)` remains the standard same-grid PTP Fresnel transfer function.

- [ ] **Step 5: Encode each branch and candidate explicitly**

Let `a=-zeta_s` and `b=zeta_s+d_model`. `b` is predicted from the start pilot and model distance; it is never overwritten with the observed target pilot. For each sampling case, boundary `Phi_t` comes from the corresponding current-run captured target ZBF header paired with that endpoint field and grid; validate and store that target hash. It is never borrowed from a historical or different-resolution target.

```text
F_Q,OO       = M_Qt   W_Q(b)   S_Q(a)   M_-Qs   Us
F_Q,OI       =         T_F(b)   S_Q(a)   M_-Qs   Us
F_Q,IO       = M_Qt   W_Q(b)   T_F(a)             Us

R_Phi|Q,OO   = M_Phit W_Q(b)   S_Q(a)   M_-Phis Us
R_Phi|Q,OI   =         T_F(b)   S_Q(a)   M_-Phis Us
R_Phi|Q,IO   = M_Phit W_Q(b)   T_F(a)             Us

R_Phi|Phi,OO = M_Phit W_Phi(b) S_Phi(a) M_-Phis Us
R_Phi|Phi,OI =         T_F(b)   S_Phi(a) M_-Phis Us
R_Phi|Phi,IO = M_Phit W_Phi(b) T_F(a)             Us
```

Do not create a result-tunable generic phase-family function. These nine paths are explicit audited functions. Never lift a PROPER `wfarr` with a ZBF reference; stock PROPER is lifted only with its paired `Q` reference.

- [ ] **Step 6: Add the stock-PROPER cross-check without endpoint information leakage**

Initialize a fresh stock PROPER wavefront with start-plane `z=0`, `z_w0=-zeta_s`, `z_Rayleigh=rx=ry`, `w0=wx=wy`, explicit `dx`, and asserted `ngrid`. Set both `reference_surface="SPHERI"/"PLANAR"` and `beam_type_old="OUTSIDE"/"INSIDE_"` from the prevalidated classification, then set `wfarr=M_-Qs Us`, shifted with PROPER's own center convention. Save all touched PROPER module globals, force `proper.phase_offset=False` to match the carrier-removed convention, propagate exactly `d_model` once, lift endpoint `wfarr` only with `Q_t` computed from `zeta_s+d_model`, and restore globals in `finally`. Add a regression that begins with deliberately polluted `phase_offset` and `beam_type_old` and still selects the required OO/OI/IO branch.

Because stock PROPER has one square sampling while the ZBF has slightly different X/Y intervals, run two predeclared variants at the same N: X-priority uses `dx_square=dy_square=dx_ZBF`, and Y-priority uses `dx_square=dy_square=dy_ZBF`. Band-limit-resample the rectangular ZBF slow field `chi_Phi` onto each square grid, then form the Q-relative PROPER input explicitly as `wfarr_Q = chi_Phi * exp[i(Phi_s-Q_s)]`; never resample the under-sampled physical carrier. After propagation, keep the output `wfarr_Q` as the slow Q-relative field, band-limit-evaluate it onto the common target grid, and only there multiply `exp(+i Q_t_pred)` to recover the physical field. Never remove `Q_t` from `wfarr` and never lift it with `Phi_t`. Persist both square variants; their difference enters the PROPER implementation uncertainty. Do not inspect the Zemax endpoint before choosing a variant. The independent X/Y-aware scaled Fresnel field is the theoretical `F_Q` candidate; stock PROPER is an implementation cross-check.

Run independent Fresnel and all fixed candidates on the same three-segment step/window matrices defined in Task 7, evaluate every level directly on the common target grid, and form separate `u_input/u_grid/u_window/u_output` components. On the frozen `Omega_-3`, use the same one piston and intensity-weighted physical-phase metric as Task 6. Require PROPER versus independent Fresnel phase RMS at most `1e-5` wave and record their complex, intensity, and power differences as implementation uncertainty. Task 14, only after every candidate field exists, additionally requires the phase implementation uncertainty to be less than one tenth of the same-segment `Omega_-3` minimum phase-candidate separation; that runtime separability check is not an offline unit-test oracle. Otherwise mark `F_Q` implementation-unresolved and do not use it to attribute a Zemax algorithm.

- [ ] **Step 7: Run offline candidate tests**

```powershell
python -m pytest tests/free_space_identification/test_fresnel_candidates.py tests/free_space_identification/test_fourier.py -m "not slow" -q
```

Expected: all Fresnel normalization, branch-ordering, PROPER-cross-check, and structural-identity tests pass. Command-line smoke execution is added in Task 14.

- [ ] **Step 8: Commit Task 9**

```powershell
git add sandbox/free_space_algorithm_identification/fresnel.py sandbox/free_space_algorithm_identification/candidates.py tests/free_space_identification/test_fresnel_candidates.py
git commit -m "feat: add fixed Fresnel reference candidates"
```

### Task 10: Immutable run artifacts, provenance, hashes, and stage receipts

**Files:**
- Create: `sandbox/free_space_algorithm_identification/artifacts.py`
- Create: `tests/free_space_identification/test_artifacts.py`

**Interfaces:**
- Consumes: planned segment/case definitions and paths produced by every later stage.
- Produces: `ArtifactRef`, `RunLayout`, `create_run_layout()`, `write_json_once()`, `copy_file_once()`, `hash_artifact()`, `write_hash_manifest()`, `verify_hash_manifest()`, `write_stage_receipt()`, and `verify_artifact_ref()`.

- [ ] **Step 1: Write failing exclusivity, exact-path, and provenance tests**

```text
test_create_run_refuses_an_existing_run_id
test_write_json_once_refuses_overwrite
test_copy_file_once_hashes_the_copied_bytes
test_artifact_ref_cannot_escape_the_run_root
test_artifact_ref_records_producer_case_relative_path_and_sha256
test_stage_receipts_are_append_only_distinct_files
test_manifest_is_a_plan_not_a_mutable_status_file
test_hash_manifest_is_sorted_excludes_itself_and_detects_tampering
test_provenance_records_versions_git_state_memory_timezone_and_conventions
```

- [ ] **Step 2: Run the tests and verify import failure**

```powershell
python -m pytest tests/free_space_identification/test_artifacts.py -q
```

Expected: collection fails for `artifacts.py`.

- [ ] **Step 3: Implement the immutable layout with exclusive creation**

Create the following directories under a caller-supplied `run_root/run_id`; use exclusive directory/file creation and refuse pre-existing destinations. The listing names files at their eventual locations, but initialization must not pre-create empty files:

```text
manifest.json
provenance.json
model/system.zmx
model/source_native.CFG
model/hashes.sha256
receipts/
continuous/
S07_S08/<case_id>/{input,identity,propagation}/
S12_S13/<case_id>/{input,identity,propagation}/
S13_S14/<case_id>/{input,identity,propagation}/
baselines/
candidates/
comparisons/
final_report.md
```

`manifest.json` is written once and contains the planned stage graph and complete predeclared case matrix. Results never rewrite it. Each completed or failed stage writes a unique monotonically numbered receipt such as `receipts/0007_identity_S12_S13_ZO1.json`, containing input artifact references, output artifact references, gate values, gate status, start/end timestamps, and exception text when applicable.

At `init`, create only directories, `manifest.json`, the run-local model/CFG copies, and `model/hashes.sha256`. Write `provenance.json` once during the first live connection, when the actual OpticStudio/ZOS versions and supported sampling enums are available. Write `final_report.md` only in Task 16. After the final report and its receipt exist, generate the root `hashes.sha256` over sorted run-relative POSIX paths while excluding that root hash file itself; the read-only `verify` command rehashes it. Nothing is written after the root hash except external console output.

`ArtifactRef` contains `producer_stage`, `producer_case`, run-relative POSIX path, byte count, and SHA-256. `verify_artifact_ref()` resolves the path, proves it remains inside the run root, rehashes it, and rejects cross-run dependencies.

- [ ] **Step 4: Capture complete reproducibility metadata**

Write once: OpticStudio, ZOS-API/ZOSPy, Python, NumPy and SciPy versions; Git commit and dirty-state paths; model/CFG/input hashes; local timezone and UTC times; phasor, grid-center, reflection and axis conventions; polarization and power settings; CPU/RAM information; and the actual POP sample-size enum values exposed by the connected OpticStudio version. Never put a live Python/.NET object into an artifact or return value.

- [ ] **Step 5: Run artifact tests**

```powershell
python -m pytest tests/free_space_identification/test_artifacts.py -q
```

Expected: all tests pass, including Windows path traversal and overwrite cases.

- [ ] **Step 6: Commit Task 10**

```powershell
git add sandbox/free_space_algorithm_identification/artifacts.py tests/free_space_identification/test_artifacts.py
git commit -m "feat: add immutable diagnostic artifacts"
```

### Task 11: Native POP report parser and request/header validation

**Files:**
- Create: `sandbox/free_space_algorithm_identification/native_report.py`
- Create: `tests/free_space_identification/fixtures/native_oo_report.txt`
- Create: `tests/free_space_identification/fixtures/native_oi_report.txt`
- Create: `tests/free_space_identification/fixtures/native_io_report.txt`
- Create: `tests/free_space_identification/test_native_report.py`

**Interfaces:**
- Consumes: native POP text, explicit ZOS settings readback, requested sampling, and source/output ZBF headers.
- Produces: `NativePopReport`, `NativeSettingsReadback`, `parse_native_pop_report()`, `validate_native_transfer()`, `validate_settings_readback()`, and `validate_output_sampling()`.

- [ ] **Step 1: Add three minimal report fixtures and failing parser tests**

The fixtures must preserve literal OO, OI, and IO transfer blocks extracted from the read-only historical `biconic_focus_test.txt`, redacting only unrelated model paths. They test parser semantics before live access; Task 15 validates the parser again on entirely new reports rather than treating these historical fixtures as experimental evidence. Tests must verify:

```text
test_oo_report_parses_signed_negative_s7_s8_distance
test_oi_report_parses_literal_branch_total_distance_and_rounded_sampling
test_io_report_parses_literal_branch_total_distance_and_rounded_sampling
test_report_collects_every_low_sampling_warning
test_settings_readback_not_report_text_proves_actual_start_and_end
test_request_report_and_zbf_sampling_must_all_agree
test_report_branch_must_agree_with_zbf_rayleigh_classification
test_propagator_label_is_not_interpreted_as_kernel_identity
```

- [ ] **Step 2: Run the report tests and verify import failure**

```powershell
python -m pytest tests/free_space_identification/test_native_report.py -q
```

Expected: collection fails for `native_report.py`.

- [ ] **Step 3: Implement strict parse records without inferring omitted settings**

Parse and retain only what the native TXT actually states: literal composite OO/OI/IO propagator labels, total signed transfer distance, printed X/Y widths and intervals, printed pilot quantities, and all warning lines. Derive the STW/PTP/WTS sub-distances analytically from the start pilot plus model distance and label them “解析推导值”; do not claim that the TXT directly reported them. Native TXT files often omit explicit Start/End, sample enum/count, wavelength/field, normalization, and polarization. Obtain those exclusively from the separately saved API settings readback and effective CFG. Treat settings readback plus ZBF header as authoritative for sample count; use rounded TXT width/interval only as a tolerance check.

For each segment validate both

\[
s_{\rm axis}\,d_{\rm report}=d_{\rm model}
\]

and

\[
d_{\rm report}=z_{x,t}^{\rm raw}-z_{x,s}^{\rm raw}
=z_{y,t}^{\rm raw}-z_{y,s}^{\rm raw}.
\]

Thus the native S7→S8 report distance is about `-368.6 mm`, while the common-axis model distance is `+368.6 mm`. Use independent tolerances fixed before candidate comparison.

- [ ] **Step 4: Enforce request/readback/report/header consistency**

Reject the run if any of these differ: requested and read-back Start/End; requested and actual sample enum; file-input N/dx/dy and read-back width; report input/output grid and ZBF header; pilot parameters and branch; requested wavelength/field/polarization/normalization; or expected output filename. Preserve all warnings in the receipt. The phrase “Using Outside-to-Inside propagator” establishes a branch and sampling structure only; it is never evidence that the internal kernel is Fresnel, ASM, or RS.

- [ ] **Step 5: Run report tests**

```powershell
python -m pytest tests/free_space_identification/test_native_report.py -q
```

Expected: all three branch fixtures pass and every altered-field negative test fails closed.

- [ ] **Step 6: Commit Task 11**

```powershell
git add sandbox/free_space_algorithm_identification/native_report.py tests/free_space_identification/fixtures tests/free_space_identification/test_native_report.py
git commit -m "feat: validate native POP reports"
```

### Task 12: Sustained ZOSPy capture for native continuous and segment runs

**Files:**
- Modify: `sandbox/free_space_algorithm_identification/biconic_case.py`
- Create: `sandbox/free_space_algorithm_identification/zos_runner.py`
- Create: `tests/free_space_identification/test_zos_runner.py`
- Create: `tests/free_space_identification/test_zemax_live.py`

**Interfaces:**
- Consumes: the immutable run layout, copied ZMX/native CFG, explicit segment requests, and exact input ZBF artifact references.
- Produces: `NativeContinuousRequest`, `SegmentPopRequest`, `RawDataGridSnapshot`, `CoordinatePhasorProbe`, `CapturedPopRun`, `capture_raw_data_grid()`, `capture_coordinate_phasor_probe()`, `capture_native_continuous()`, `capture_segment_run()`, and `expected_output_names()`.

- [ ] **Step 1: Write failing fake-ZOS call-order and stale-output tests**

Use fakes, not a live OpticStudio instance, to prove:

```text
test_segment_run_uses_sustain_and_captures_before_close
test_continuous_run_uses_new_analysis_loadfrom_and_one_apply
test_continuous_run_does_not_call_the_high_level_wrapper_run
test_settings_saved_before_text_then_zbf_then_messages
test_each_capture_failure_still_closes_and_disconnects
test_close_failure_still_disconnects_without_hiding_primary_exception
test_return_value_contains_paths_and_values_not_live_objects
test_exact_same_name_stale_file_is_rejected_before_run
test_output_collection_accepts_only_the_anchored_exact_name_set
test_missing_base_endpoint_or_expected_surface_file_fails
test_input_artifact_to_popdir_copy_hashes_match
test_popdir_output_to_run_copy_hashes_match
test_application_log_is_read_only_while_connection_is_alive
test_staged_input_cleanup_failure_cannot_prevent_close_or_disconnect
test_raw_datagrid_capture_never_calls_analysisresult_get_data_grid_or_unpack_datagrid
test_raw_datagrid_uses_sample_at_zero_and_api_xy_order
test_raw_datagrid_values_are_transposed_exactly_once_for_zbf_yx_order
test_half_step_dataframe_labels_fail_the_coordinate_gate
test_coordinate_phasor_probe_precedes_native_continuous_capture
```

- [ ] **Step 2: Run runner tests and verify import failure**

```powershell
python -m pytest tests/free_space_identification/test_zos_runner.py -q
```

Expected: collection fails for `zos_runner.py`.

- [ ] **Step 3: Freeze the exact continuous-output name set**

Add to `biconic_case.py`:

```python
NATIVE_SAVED_SURFACES = (1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15)
```

For prefix `P`, collect only `P.ZBF` and `P_0001.ZBF`, `P_0002.ZBF`, and the remaining explicitly enumerated surface names. Refuse any pre-existing exact output, missing expected output, or unexpected file whose name matches anchored regex `^escaped_P(?:_\d{4})?\.ZBF$`. Never use a broad historical glob.

- [ ] **Step 4: Implement native continuous replay through the low-level ZOS API path**

The bundled POP wrapper does not consume CFG/TXT arguments passed to `run()`. Therefore continuous replay must:

1. connect and load the copied run-local ZMX;
2. import `new_analysis` from `zospy.analyses.base` and create a raw Physical Optics Propagation analysis;
3. call `Settings.LoadFrom(source_native.CFG)` before execution;
4. override only the unique output prefix/routing and required save flags;
5. read back and assert all critical native settings before execution;
6. call `ApplyAndWaitForCompletion()` exactly once;
7. capture using the common sustained-analysis routine;
8. close and disconnect in `finally` blocks.

Do not load the CFG after a wrapper run and do not reconstruct the native continuous settings from memory. The baseline directory contains the native `biconic_focus_test.CFG`; copy and hash it into the run before use.

Before Apply, validate the loaded CFG against the loaded model and historical native report: wavelength number/value, field, start/end, polarization, normalization, N/X/Y widths, Gaussian/file source parameters, saved-surface behavior, and output routing. Record `baseline_cfg_mismatch` and stop on any physical-setting conflict; only the output prefix and save routing may differ in the fresh replay.

- [ ] **Step 5: Implement segment File-beam runs through the high-level wrapper with explicit settings**

For a segment, resolve the input `ArtifactRef`, copy it into the connected application's `POPDir` under a unique run/case/repeat-specific basename, and verify source/staged SHA-256 equality. Construct `PhysicalOpticsPropagation` with explicit Start/End, `surface_to_beam=0`, beam type File using only that basename, actual N/X/Y widths, `auto_calculate_beam_sampling=False`, AlongBeam projection, polarization, normalization settings read from the native control, save-output enabled, and a unique anchored output prefix. Run with

```python
result = wrapper.run(oss, oncomplete="Sustain")
```

The sustained live object is `wrapper.analysis`. The common capture helper accepts the active analysis object rather than assuming a high-level `result`. For the low-level continuous path, serialize messages from the active analysis; for the high-level segment path, additionally snapshot `AnalysisResult.messages`. Capture in this order while connected:

```text
analysis.Settings.SaveTo(effective.CFG)
settings API readback -> settings_readback.json
analysis.Results.GetTextFile(report_raw.txt)
exact expected ZBF copy -> immediate SHA-256 -> lossless header read
active-analysis messages, optional segment AnalysisResult snapshot, and connected application log -> messages.json
analysis.Close()
zos.disconnect()
```

On error, preserve the primary exception, attempt Close, and always disconnect. The returned dataclass contains only immutable paths, hashes, parsed values, and messages.

Never call `AnalysisResult.get_data_grid()` or `zospy.utils.zputils.unpack_datagrid()` for physical coordinates. While the analysis is sustained, read `analysis.Results.DataGrids[k]` directly and persist `Nx/Ny/MinX/MinY/Dx/Dy`, `X(0)/X(Nx//2)/X(Nx-1)`, `Y(0)/Y(Ny//2)/Y(Ny-1)`, selected `Z(ix,iy)` and `Values[ix,iy]` checkpoints, and the raw `Values` array. Validate the official `X=MinX+iDx`, `Y=MinY+jDy`, sample-at-zero center, and `Z=Values` relations before converting API `[x,y]` to package/ZBF `[y,x]` with exactly one transpose.

Use nested cleanup in this fixed order: finish/abort capture, close `wrapper.analysis` or raw `live` exactly once, disconnect, and only then remove the exact staged input file created by this run after resolving and proving it lies inside `POPDir`. A staged-file deletion failure cannot prevent Close/disconnect and cannot obscure an earlier primary exception; record it as a separate cleanup error. Never enumerate or clean unrelated POP files. The run-local input artifact remains immutable.

- [ ] **Step 6: Add the live coordinate and raw-phasor representation probe**

Before the fresh native continuous run can unlock any candidate input, run a small Start=End File-beam probe in the copied model using a planar-reference ZBF with a predeclared, band-limited, non-centrosymmetric complex pattern. Include several known phase plateaus/slopes below `0.4 rad` so radians and waves differ well beyond the numerical tolerance without phase wrapping. Read raw `EXPhase` DataGrid values and the saved output ZBF from the same sustained analysis. The probe must establish, without endpoint propagation fitting:

```text
raw DataGrid sample-at-zero coordinates
raw API Values[x,y] -> ZBF Ex[y,x] by one transpose
no X/Y exchange or additional left/right/up/down flip
the raw phase unit from setting/document readback; if unavailable, radians versus waves
must be classified by a predeclared 3u/5u known-input test or remain undecided
raw DataGrid EXPhase follows arg(raw ZBF Ex) in Zemax's own phasor convention,
up to one phase piston and the predeclared numerical tolerance
```

Convert the declared raw phase unit to radians and compare periodic unit phasors; do not unwrap or fit spatial terms. This probe verifies the API/ZBF representation relation; the conversion to the common external `exp(-i omega t)` convention remains the fixed analytic `conj(Ex)` contract and is not selected by this residual.

`CoordinatePhasorProbe.to_raw_grid_evidence()` binds the raw snapshot to the exact model, CFG, input ZBF, output ZBF, run id, and raw-grid array hashes. Only the connected live path may set origin `live_zosapi`. `CoordinatePhasorProbe.to_convention_validation()` combines that evidence with the Task 3 geometry/phasor contract and produces canonical JSON plus `evidence_sha256`. Task 10 writes both the evidence artifact and a stage receipt whose `ArtifactRef` hashes match that canonical JSON. Fakes and optional smoke fixtures always produce `authoritative=false`.

Any missing raw field, half-step center, unknown phase unit, axis-order ambiguity, hash mismatch, or phase mismatch fails closed. The probe must use a unique prefix, must not write into the historical baseline directory, and must not use the high-level DataFrame coordinates.

- [ ] **Step 7: Add identity-output fallback support without weakening side validation**

Request Start=End first. If no usable base output is produced, run a controlled all-surfaces capture and select only the exact numbered start-surface output after report/settings evidence proves it is on the same physical side as the input. Record the fallback path in the receipt; do not silently switch.

- [ ] **Step 8: Add a gated live smoke test**

`test_zemax_live.py` is skipped only when `BTS_RUN_ZEMAX_BENCHMARK` is not `1` or `BTS_FREE_SPACE_BASELINE_DIR` was not supplied. The test resolves and copies the model/CFG from that explicit read-only directory into a temporary run. Once explicitly enabled, connection, license, model-load, coordinate-phasor probe, capture, and validation failures fail the test rather than becoming skips. It captures the small asymmetric identity probe and checks raw DataGrid/settings/report/ZBF/header/hash closure. It may never write into the historical baseline directory.

- [ ] **Step 9: Run offline runner tests**

```powershell
python -m pytest tests/free_space_identification/test_zos_runner.py -q
```

Expected: all fake-ZOS failure-order tests pass; no OpticStudio process is required.

- [ ] **Step 10: Commit Task 12**

```powershell
git add sandbox/free_space_algorithm_identification/biconic_case.py sandbox/free_space_algorithm_identification/zos_runner.py tests/free_space_identification/test_zos_runner.py tests/free_space_identification/test_zemax_live.py
git commit -m "feat: capture sustained native POP runs"
```

### Task 13: Start-plane identity and entrance-only complex calibration

**Files:**
- Create: `sandbox/free_space_algorithm_identification/identity.py`
- Create: `tests/free_space_identification/test_identity.py`
- Modify: `tests/free_space_identification/test_zemax_live.py`

**Interfaces:**
- Consumes: the input ZBF physical field, a same-surface Zemax rewrite, its native report/settings proof, and one frozen identity ROI.
- Produces: `SampleConventionProbe`, `SampleConventionResult`, `classify_sample_value_convention()`, `IdentityPolicy`, `EntranceCalibration`, `IdentityResult`, `evaluate_start_identity()`, and `apply_entrance_calibration()`.

- [ ] **Step 1: Write failing calibration, non-compensation, and reuse tests**

```text
test_identity_recovers_a_known_complex_scalar
test_grid_order_or_sampling_mismatch_is_rejected
test_shift_flip_rotation_and_defocus_are_not_compensated
test_endpoint_uses_the_exact_entrance_calibration
test_no_endpoint_fit_api_exists
test_failed_identity_blocks_segment_propagation
test_same_side_fallback_requires_native_report_proof
test_numerical_failure_does_not_trigger_the_missing-output_fallback
test_cross_sampling_probe_separates_point_and_cell_area_laws
test_per_case_entrance_constants_cannot_replace_convention_classification
test_ambiguous_sample_convention_blocks_main_derivations
```

- [ ] **Step 2: Run identity tests and verify import failure**

```powershell
python -m pytest tests/free_space_identification/test_identity.py -q
```

Expected: collection fails for `identity.py`.

- [ ] **Step 3: Implement the cross-sampling sample-value convention gate**

From the fresh continuous S7 ZBF, construct predeclared N=1024/2048/4096 same-window same-slow-field identity probes under both hypotheses:

```text
point hypothesis payload at common nodes: E_N = E_1024
cell hypothesis payload at common nodes:  E_N = E_1024 sqrt(dA_N/dA_1024)
```

Run Start=End with identical normalization settings and fresh connections. Compare API/report total power, peak irradiance, payload sums, pixel-area-weighted sums, and the entrance-calibration magnitude ratios against the two analytic area-scaling laws. Use separate uncertainties from repeats/readback and require the supported hypothesis to close within 3u while the other is separated by more than 5u. A per-case free complex constant cannot itself choose the convention. If normalization hides both power laws or the candidates are not separated, record `sample_value_convention_undecided` and stop before main ZBF derivation or propagation.

Persist the unique result as an immutable pre-experiment receipt. Every later call to payload conversion, derivation, identity, Fresnel normalization, and comparison reads that receipt and passes the convention explicitly.

- [ ] **Step 4: Implement the single entrance calibration with a fixed direction**

Let `U_in` be the physical field reconstructed from the input file and `U_rewrite` the physical field Zemax writes at the same start plane. Freeze

\[
\Omega_{\rm id}=\text{the eight-connected component containing the input peak in }
\{I_{\rm in}/I_{\max}\ge10^{-6}\}.
\]

Only the input field defines this region. Define

\[
c_{\rm entry}=
\frac{\sum_\Omega U_{\rm in}\,U_{\rm rewrite}^*\Delta A}
{\sum_\Omega |U_{\rm rewrite}|^2\Delta A},
\]

so that `c_entry * U_rewrite` is in the input field's units and phase convention. Persist the complex value, magnitude, phase, ROI, and exact input/rewrite hashes. Apply this same `c_entry` to the corresponding Zemax propagation endpoint before comparison. Candidate fields are not rescaled, and no endpoint amplitude or piston fit is exposed by this module; the later metric layer still removes only its one permitted comparison piston.

- [ ] **Step 5: Enforce the identity gate**

Require exact `nx/ny`, matching physical side, and `dx/dy` closure with fixed `rtol=1e-10` and `atol=1e-12 mm` against both the request and input header. Phase uses `I_in ΔA` weights. Define the intensity gate as

\[
100\sqrt{\frac{\sum_{\Omega_{\rm id}}
(|c_{\rm entry}U_{\rm rewrite}|^2-|U_{\rm in}|^2)^2\Delta A}
{\sum_{\Omega_{\rm id}}|U_{\rm in}|^4\Delta A}}.
\]

Require:

```text
input-intensity-weighted phase RMS after c_entry <= 1e-6 wave
relative intensity L2 after c_entry × 100        <= 1e-4 percent
```

Do not search a translation, reflection, rotation, scale, tilt, defocus, higher-order phase, or alternate grid center. If Start=End produces no output, the fallback in Task 12 is allowed only when settings/report evidence proves the numbered surface file lies on the identical side. A numerically bad identity result is a hard failure, not a reason to switch files.

- [ ] **Step 6: Run identity tests**

```powershell
python -m pytest tests/free_space_identification/test_identity.py -q
```

Expected: all tests pass, including a test proving that a deliberately different terminal amplitude cannot alter `c_entry`.

- [ ] **Step 7: Commit Task 13**

```powershell
git add sandbox/free_space_algorithm_identification/identity.py tests/free_space_identification/test_identity.py tests/free_space_identification/test_zemax_live.py
git commit -m "feat: gate Zemax input identity"
```

### Task 14: Strict staged pipeline and command-line interface

**Files:**
- Create: `sandbox/free_space_algorithm_identification/interventions.py`
- Create: `sandbox/free_space_algorithm_identification/pipeline.py`
- Create: `sandbox/free_space_algorithm_identification/cli.py`
- Create: `tests/free_space_identification/test_interventions.py`
- Create: `tests/free_space_identification/test_pipeline_report.py`

**Interfaces:**
- Consumes: all Task 1–13 modules and immutable receipts.
- Produces: `CompositeIntervention`, `SidebandProbe`, `PipelineStage`, `PipelineOutcome`, `run_stage()`, `next_allowed_stages()`, and CLI subcommands `init`, `verify`, `status`, `capture-native`, `probe-sample-values`, `prepare-inputs`, `run-zemax`, `baselines`, `compare`, and `decide`.

- [ ] **Step 1: Write failing stage-order, dependency, and stop-condition tests**

```text
test_preflight_failure_never_calls_zemax
test_manifest_is_frozen_before_any_live_backend_call
test_coordinate_phasor_failure_blocks_native_capture_and_all_derivations
test_coordinate_phasor_receipt_rejects_half_step_dataframe_labels
test_synthetic_or_skipped_coordinate_evidence_cannot_unlock_live_pipeline
test_convention_validation_hashes_bind_model_cfg_input_output_and_raw_grid
test_actual_inputs_resolve_to_fresh_continuous_artifacts
test_sample_value_convention_is_frozen_before_main_derivation
test_every_case_runs_identity_before_propagation
test_identity_failure_blocks_only_that_case_and_downstream_ranking
test_report_or_header_failure_blocks_comparison
test_continuous_restart_mismatch_sets_blocked_hidden_state
test_failed_zemax_convergence_cannot_freeze_the_roi
test_4096_memory_failure_retries_disk_storage_once
test_nonmemory_failure_is_not_retried
test_s12_zo_output_is_the_declared_s13_high_resolution_input
test_chained_s13_is_registered_only_after_upstream_hash_validation
test_native_and_highest_accepted_restarts_have_two_independent_runs
test_restart_repeat_distance_enters_zemax_uncertainty
test_intervention_parameters_are_frozen_before_natural_zemax_outputs
test_intervention_preselection_never_reads_fresh_target_artifacts
test_reference_variant_is_labeled_composite_pilot_intervention
test_sideband_frequency_selection_uses_only_external_candidate_predictions
test_sideband_half_amplitude_checks_first_order_linearity
test_undecided_result_does_not_choose_the_smallest_error
test_mixed_segment_matches_are_not_averaged
test_failure_receipt_cannot_be_interpreted_as_completed
```

- [ ] **Step 2: Run pipeline tests and verify import failure**

```powershell
python -m pytest tests/free_space_identification/test_pipeline_report.py -q
```

Expected: collection fails for `pipeline.py` and `cli.py`.

- [ ] **Step 3: Implement the exact stage graph**

The only valid order is:

```text
offline_preflight
→ freeze_manifest
→ coordinate_phasor_probe
→ fresh_native_continuous
→ sample_value_convention_probe
→ resolve_intervention_parameters_from_fresh_start_only
→ derive_available_seed_inputs
→ native_identity_and_restart
→ continuous_restart_gate
→ s7_s12_high_sampling_identity_and_segments
→ validate_and_register_chained_s13_inputs
→ derive_remaining_s13_inputs
→ s13_high_sampling_identity_and_segments
→ zemax_numerical_convergence_and_provisional_roi
→ exact_baselines_and_candidates
→ comparisons
→ decision_stability_across_top_zemax_levels_and_rois
→ natural_dimensioned_uncertainty_and_3u_5u_decision
→ [if analytically inseparable] composite_reference_pilot_intervention
→ [if still inseparable] weak_sideband_response
→ final_dimensioned_uncertainty_and_3u_5u_decision
```

The historical baseline ZBF/report/CFG files are read-only preflight fixtures. `coordinate_phasor_probe` consumes only the run-local model/CFG copies and its predeclared asymmetric planar-reference input; it writes the immutable `ConventionValidation` artifact and stage receipt required by every later raw-ZBF physical-field mapping. Before accepting it, the pipeline verifies `authoritative=true`, origin `live_zosapi`, stage name, run id, OpticStudio/ZOS versions, and the `ArtifactRef` hashes for the model, CFG, input ZBF, output ZBF, raw-grid array, canonical validation JSON, and receipt. A synthetic/smoke artifact, a skipped static test, or a free in-memory object cannot satisfy this dependency. A missing raw DataGrid, half-step-shifted coordinate label, unknown phase unit, axis-order ambiguity, phasor mismatch, or hash mismatch stops before `fresh_native_continuous`. After `fresh_native_continuous`, every S7/S12/S13 native restart source must be an `ArtifactRef` produced by that fresh case. `derive_available_seed_inputs` may derive S7/S12 refinements and native-S13 window controls, but it may not fabricate chained S13 hashes. Only after S12→S13 `ZO1/ZO2` propagation, copy/hash/header validation, and passed receipts may `validate_and_register_chained_s13_inputs` create those `ArtifactRef` records and unlock S13→S14 high-resolution identity runs.

For continuous/restart closure compare `c_entry * U_t,restart` against `U_t,continuous`; never apply `c_entry` to the continuous endpoint. Before high-sampling runs, freeze dedicated `Omega_-2/-3/-6` regions from the continuous endpoint alone and use the same complex-distance, phase, intensity, power, and separate 3u/5u formulas as candidate comparisons. This dedicated control ROI avoids depending on the later high-sampling candidate ROI. If start identity passes but this calibrated endpoint comparison is excluded, set `blocked_hidden_state`; if it lies in any 3u–5u gray zone, set `undecided_hidden_state`. Both stop continuous-kernel attribution with the interpretation “输出依赖是否完全由物理总复振幅确定尚未闭合”. Do not infer the continuous-run kernel from restart results. If multiple candidates remain inside the later candidate gray zone, set `undecided`; never promote the smallest error.

Run every native 1024 restart twice from the identical input bytes and settings using distinct prefixes and fresh connections. Also repeat the highest accepted sampling case for each segment. Their direct endpoint complex distance, phase, intensity, and power differences enter the corresponding `u_Zemax`; a repeat is never aligned by anything beyond the permitted piston.

After all candidate fields exist, compare the stored PROPER-versus-independent-Fresnel implementation uncertainty with the smallest relevant candidate separation. It must be below one tenth of that separation as well as below the absolute `1e-5`-wave gate; otherwise `F_Q` remains implementation-unresolved.

- [ ] **Step 4: Implement retry and memory policy**

A 4096 POP run that fails with a recognized OpticStudio memory/storage error may be retried exactly once with identical physical settings and `use_disk_storage=True`. Other errors are not retried. If the installed sample-size enumeration does not support 4096, retain 2048 as the Zemax maximum, record the limitation, enlarge the Zemax sampling uncertainty, and allow the decision layer to return only what remains separable within the enlarged numerical uncertainty bounds. Never emulate a global POP transform with tiles.

- [ ] **Step 5: Implement CLI parsing and receipt-driven resumption**

All subcommands require `--run-dir` except `init`. `init` requires `--baseline-dir`, `--output-root`, and `--run-id`, and accepts `--run-ref` to write one exclusive untracked text file containing the absolute run directory for later independent shells. `run-zemax` requires `--tier native|high|intervention`; high and intervention tiers also require `BTS_RUN_ZEMAX_HIGH_SAMPLING=1`. Any live command requires `BTS_RUN_ZEMAX_BENCHMARK=1`. `verify` rehashes every referenced artifact and validates every stage dependency without changing files. `status --field terminal_state|next_stage` reads verified receipts and prints exactly one machine-readable token to stdout without mutation; diagnostics go to stderr. Exit codes are fixed: `0` means the requested stage passed, `2` means a scientifically valid terminal gate (`undecided`, `undecided_hidden_state`, `blocked_hidden_state`, or a failed physics/numerical gate) was recorded, and `1` means an operational/program failure. Receipts, not console phrasing, determine resumability. An exit code 2 skips downstream propagation but never prevents Task 16 from rendering the terminal report.

Task 7–9 use direct offline unit tests before the CLI exists. After this task, optional synthetic CLI smoke runs must first create a complete synthetic run fixture through `init`; every smoke receipt states `authoritative=false` and cannot satisfy a real-run convergence dependency.

`prepare-inputs` first resolves and writes the fresh-start-only `intervention_preselection` receipt, then derives inputs whose logical producers are already available; it cannot read a target field while resolving intervention parameters.

- [ ] **Step 6: Implement the two conditionally triggered discrimination experiments**

Before `freeze_manifest`, freeze only the intervention selection rules, factor/frequency candidate sets, analytic thresholds, and maximum case count. After fresh continuous capture and the sample-value convention gate—but before any segment restart or natural endpoint comparison—resolve the concrete parameters from only the fresh S7/S12/S13 start artifacts plus the analytic Helmholtz-versus-Fresnel transfer-phase difference, then write an immutable `intervention_preselection` receipt. The resolver must not open fresh continuous target artifacts; a fake target that raises on access proves this. Never select parameters from Zemax residuals or endpoint candidate rankings.

For the composite reference/pilot intervention, consider signed pilot distances `zeta_prime = 2*zeta_s` then `zeta_prime = zeta_s/2` in that fixed order and retain the first whose new slow field satisfies a four-samples-per-cycle margin over `Omega_-6`. Construct

\[
\chi_s'=\exp[i(\Phi_s-\Phi_s')]\chi_s
\]

so the external physical field is unchanged, and intentionally patch only `zx/zy` in addition to the normal payload. Keep `rx/ry/wx/wy`, window, sampling, power, and coordinates fixed. Run a new identity gate and record every changed pilot classification, propagator branch, output grid, and warning. Unless API readback proves all pilot and sampling states are independently locked, label this only “ZBF 参考编码与导引光束状态复合干预”, never a pure reference-gauge test.

For weak sidebands, predeclare candidate fractions `{1/16, 1/8, 3/16, 1/4}` of each supported input Nyquist frequency and directions X, Y, and diagonal. Retain the lowest three modes that remain below one-quarter of the physical-carrier Nyquist limit and whose external `H` versus `F_Q` differential predictions exceed `10u`; if none qualify, record insufficient analytic separability. For each selected wavevector run

\[
U_s^{(\kappa,\epsilon)}=U_s\exp[i\epsilon\cos(\kappa\cdot r)]
\]

at fixed `epsilon=0.01 rad` and `0.005 rad`, plus the unperturbed control. Require the two differential responses to scale 2:1 within the numerical bound before using first-order phase response. Compare Zemax only with the no-fit exact-square-root and Fresnel-quadratic differential predictions. Every intervention has its own start identity and sampling convergence gate.

Expose these only as `run-zemax --tier intervention`, enabled when the natural decision receipt is `needs_intervention` and `BTS_RUN_ZEMAX_HIGH_SAMPLING=1`. A natural `undecided` result is not converted to a best candidate; it triggers the frozen intervention graph or remains undecidable if no safe separating probe exists.

- [ ] **Step 7: Run pipeline and all offline diagnostic tests**

```powershell
python -m pytest tests/free_space_identification/test_pipeline_report.py -q
python -m pytest tests/free_space_identification/test_interventions.py -q
python -m pytest tests/free_space_identification -m "not slow" -q
```

Expected: all tests pass; live tests remain skipped.

- [ ] **Step 8: Commit Task 14**

```powershell
git add sandbox/free_space_algorithm_identification/interventions.py sandbox/free_space_algorithm_identification/pipeline.py sandbox/free_space_algorithm_identification/cli.py tests/free_space_identification/test_interventions.py tests/free_space_identification/test_pipeline_report.py
git commit -m "feat: orchestrate gated propagator identification"
```

### Task 15: Execute the new native controls, sampling matrix, and exact baselines

**Files:**
- Generate only under: `sandbox/free_space_algorithm_identification/output/<run_id>/`
- Do not modify source code while interpreting results; implementation corrections require a failing offline test and a separate code commit before rerunning with a new run ID.

- [ ] **Step 1: Run the complete offline verification before connecting to OpticStudio**

```powershell
python -m pytest tests/free_space_identification -m "not slow" -q
python -m pytest tests/free_space_identification/test_fourier.py -m slow -q
python -m pytest tests/test_zbf_io.py tests/test_zbf_source.py -k "not reference_phase_uses_spherical_header_metadata" -q
python -m compileall sandbox/free_space_algorithm_identification tests/free_space_identification
git diff --check
```

Expected: all offline tests pass and live tests are skipped.

- [ ] **Step 2: Run the gated live smoke test**

```powershell
$Workspace = (git rev-parse --show-toplevel).Trim()
$env:BTS_RUN_ZEMAX_BENCHMARK='1'
$env:BTS_FREE_SPACE_BASELINE_DIR='D:\BTS\.worktrees\residual-phase\sandbox\Zemax_baseline'
python -m pytest tests/free_space_identification/test_zemax_live.py -q -s
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

Expected: a unique temporary run passes settings/report/ZBF/hash capture and start identity. On failure, stop before the full matrix.

- [ ] **Step 3: Create one immutable real run and verify its frozen plan**

```powershell
$Workspace = (git rev-parse --show-toplevel).Trim()
$RunId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ') + '_' + [guid]::NewGuid().ToString('N').Substring(0,8)
$Root = Join-Path $Workspace 'sandbox\free_space_algorithm_identification\output'
$Base = 'D:\BTS\.worktrees\residual-phase\sandbox\Zemax_baseline'
$RunRef = Join-Path $Root 'authorized_s7_s14_run.ref'

python -m sandbox.free_space_algorithm_identification.cli init `
  --baseline-dir $Base --output-root $Root --run-id $RunId --run-ref $RunRef
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$RunDir = (Get-Content -LiteralPath $RunRef -Raw).Trim()
python -m sandbox.free_space_algorithm_identification.cli verify --run-dir $RunDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

Expected: the run-local ZMX and native CFG hashes match their copied sources and the manifest contains the entire matrix before any Zemax call.

- [ ] **Step 4: Capture a fresh native continuous control, then derive inputs from it**

```powershell
$Workspace = (git rev-parse --show-toplevel).Trim()
$RunRef = Join-Path $Workspace 'sandbox\free_space_algorithm_identification\output\authorized_s7_s14_run.ref'
$RunDir = (Get-Content -LiteralPath $RunRef -Raw).Trim()
$env:BTS_RUN_ZEMAX_BENCHMARK='1'
$env:BTS_FREE_SPACE_BASELINE_DIR='D:\BTS\.worktrees\residual-phase\sandbox\Zemax_baseline'
python -m sandbox.free_space_algorithm_identification.cli capture-native --run-dir $RunDir
$StageCode = $LASTEXITCODE
if ($StageCode -eq 1) { exit 1 }
if ($StageCode -eq 2) {
  python -m sandbox.free_space_algorithm_identification.cli verify --run-dir $RunDir
  exit 0
}
python -m sandbox.free_space_algorithm_identification.cli probe-sample-values --run-dir $RunDir
$StageCode = $LASTEXITCODE
if ($StageCode -eq 1) { exit 1 }
if ($StageCode -eq 2) {
  python -m sandbox.free_space_algorithm_identification.cli verify --run-dir $RunDir
  exit 0
}
python -m sandbox.free_space_algorithm_identification.cli prepare-inputs --run-dir $RunDir
$StageCode = $LASTEXITCODE
if ($StageCode -eq 1) { exit 1 }
python -m sandbox.free_space_algorithm_identification.cli verify --run-dir $RunDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($StageCode -eq 2) { exit 0 }
```

Expected: new native reports and ZBFs exist for every declared saved surface; the point-value versus cell-energy convention is uniquely frozen; S7/S12/S13 derivations resolve to those new hashes, not historical baseline ZBFs. Any convention/header/reference/edge-energy failure stops execution.

- [ ] **Step 5: Run all native identity/restart cases and the continuous/restart gate**

```powershell
$Workspace = (git rev-parse --show-toplevel).Trim()
$RunRef = Join-Path $Workspace 'sandbox\free_space_algorithm_identification\output\authorized_s7_s14_run.ref'
$RunDir = (Get-Content -LiteralPath $RunRef -Raw).Trim()
$env:BTS_RUN_ZEMAX_BENCHMARK='1'
python -m sandbox.free_space_algorithm_identification.cli run-zemax --run-dir $RunDir --tier native
$StageCode = $LASTEXITCODE
if ($StageCode -eq 1) { exit 1 }
python -m sandbox.free_space_algorithm_identification.cli verify --run-dir $RunDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($StageCode -eq 2) {
  Write-Output 'Native stage reached a scientific terminal receipt; downstream propagation is skipped, but Task 16 must still render the report.'
  exit 0
}
```

Expected: each segment has a passed start identity before propagation. If a calibrated native restart endpoint is excluded or undecided against its continuous counterpart, the pipeline terminates as `blocked_hidden_state` or `undecided_hidden_state` and skips kernel attribution.

- [ ] **Step 6: Run the authorized high-sampling Zemax matrix**

```powershell
$Workspace = (git rev-parse --show-toplevel).Trim()
$RunRef = Join-Path $Workspace 'sandbox\free_space_algorithm_identification\output\authorized_s7_s14_run.ref'
$RunDir = (Get-Content -LiteralPath $RunRef -Raw).Trim()
$env:BTS_RUN_ZEMAX_BENCHMARK='1'
$env:BTS_RUN_ZEMAX_HIGH_SAMPLING='1'
$NextStage = (python -m sandbox.free_space_algorithm_identification.cli status --run-dir $RunDir --field next_stage).Trim()
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($NextStage -ne 'high') {
  python -m sandbox.free_space_algorithm_identification.cli verify --run-dir $RunDir
  exit 0
}
python -m sandbox.free_space_algorithm_identification.cli run-zemax --run-dir $RunDir --tier high
$StageCode = $LASTEXITCODE
if ($StageCode -eq 1) { exit 1 }
python -m sandbox.free_space_algorithm_identification.cli verify --run-dir $RunDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($StageCode -eq 2) {
  Write-Output 'High-sampling stage reached a scientific terminal receipt; Task 16 remains required.'
  exit 0
}
```

Expected: ZI and ZO axes are reported separately, every high-resolution input passes its own identity gate, and S13→S14 chained cases point to the matching upstream ZO outputs. A 4096 limitation is recorded rather than hidden.

- [ ] **Step 7: Run exact Helmholtz/RS, Fresnel/PROPER, and fixed candidate matrices**

```powershell
$Workspace = (git rev-parse --show-toplevel).Trim()
$RunRef = Join-Path $Workspace 'sandbox\free_space_algorithm_identification\output\authorized_s7_s14_run.ref'
$RunDir = (Get-Content -LiteralPath $RunRef -Raw).Trim()
$env:BTS_RUN_ZEMAX_BENCHMARK='1'
$env:BTS_RUN_ZEMAX_HIGH_SAMPLING='1'
$NextStage = (python -m sandbox.free_space_algorithm_identification.cli status --run-dir $RunDir --field next_stage).Trim()
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($NextStage -ne 'baselines') {
  python -m sandbox.free_space_algorithm_identification.cli verify --run-dir $RunDir
  exit 0
}
python -m sandbox.free_space_algorithm_identification.cli baselines --run-dir $RunDir
$StageCode = $LASTEXITCODE
if ($StageCode -eq 1) { exit 1 }
if ($StageCode -eq 2) {
  python -m sandbox.free_space_algorithm_identification.cli verify --run-dir $RunDir
  exit 0
}
python -m sandbox.free_space_algorithm_identification.cli compare --run-dir $RunDir
$StageCode = $LASTEXITCODE
if ($StageCode -eq 1) { exit 1 }
if ($StageCode -eq 2) {
  python -m sandbox.free_space_algorithm_identification.cli verify --run-dir $RunDir
  exit 0
}
python -m sandbox.free_space_algorithm_identification.cli decide --run-dir $RunDir
$DecisionCode = $LASTEXITCODE
if ($DecisionCode -eq 1) { exit 1 }
$TerminalState = (python -m sandbox.free_space_algorithm_identification.cli status --run-dir $RunDir --field terminal_state).Trim()
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($TerminalState -eq 'needs_intervention') {
  python -m sandbox.free_space_algorithm_identification.cli run-zemax --run-dir $RunDir --tier intervention
  $InterventionCode = $LASTEXITCODE
  if ($InterventionCode -eq 1) { exit 1 }
  if ($InterventionCode -eq 0) {
    python -m sandbox.free_space_algorithm_identification.cli compare --run-dir $RunDir
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    python -m sandbox.free_space_algorithm_identification.cli decide --run-dir $RunDir
    $DecisionCode = $LASTEXITCODE
    if ($DecisionCode -eq 1) { exit 1 }
  }
}
python -m sandbox.free_space_algorithm_identification.cli verify --run-dir $RunDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
exit 0
```

Expected: either every prescribed convergence/identity/structure gate passes and a dimensioned decision is produced, or the pipeline stops with the exact failed gate. Do not edit a parameter in the existing run to obtain a smaller residual; a changed experiment requires a new run ID and manifest.

### Task 16: Produce the academic Chinese report and update root-cause status

**Files:**
- Create: `sandbox/free_space_algorithm_identification/report.py`
- Modify: `sandbox/free_space_algorithm_identification/cli.py`
- Modify: `sandbox/free_space_algorithm_identification/pipeline.py`
- Modify: `tests/free_space_identification/test_pipeline_report.py`
- Create after a verified terminal real run: `sandbox/diagnostics/zemax_free_space_propagator_identification_report.md`
- Modify after a verified terminal real run: `sandbox/diagnostics/s7_s8_phase_root_cause.md`

**Interfaces:**
- Consumes: only verified immutable artifacts and a verified terminal receipt, including `complete`, `undecided`, `undecided_hidden_state`, `blocked_hidden_state`, identity failure, sampling failure, or numerical-baseline failure.
- Produces: CLI subcommand `report`, `render_scientific_report()`, run-local `final_report.md`, and the repository scientific summary.

- [ ] **Step 1: Write failing report-content and terminology tests**

Require the report to contain source hashes, the point-value/cell-energy convention evidence, physical-field/reference equations, segment tables, convergence and uncertainty tables, exact/RS and Fresnel/PROPER checks, every candidate's `D`, `u`, `D/u`, phase/intensity/power gates, the S13→S14 structural identity, continuous/restart status, and direct artifact links. Reject missing units and the informal terms `winner`, `trick`, `ledger`, and “最小误差故胜出”.

- [ ] **Step 2: Implement deterministic report rendering**

Use formal Chinese academic prose and the notation of the approved design. When producing plots, invoke `making-report-plots-readable` and show convergence axes separately; never combine fixed-window and fixed-step runs into one unlabeled resolution curve. The report must explicitly state:

- ZBF physical fields are reconstructed only with their paired `Phi_ZBF` reference;
- PROPER `wfarr` is reconstructed only with `Q_PROPER`;
- matching `H` means “在这些输入和不确定度内与非旁轴标量传播等效”, not that Zemax internally uses ASM or RS;
- sparse RS evidence is “稀疏 RS-I 点集核验”, not a full-field RS RMS;
- segment-dependent matches are a branch-dependent or mixed rule, not an averaged single algorithm;
- unresolved candidates are “当前采样与输入条件下不可判别”.

A failed or blocked run still produces a rigorous report containing all evidence acquired before the gate and the exact missing evidence; report generation must not require a successful unique decision.

- [ ] **Step 3: Render and verify the real report**

```powershell
$Workspace = (git rev-parse --show-toplevel).Trim()
$RunRef = Join-Path $Workspace 'sandbox\free_space_algorithm_identification\output\authorized_s7_s14_run.ref'
$RunDir = (Get-Content -LiteralPath $RunRef -Raw).Trim()
python -m pytest tests/free_space_identification/test_pipeline_report.py -q
python -m sandbox.free_space_algorithm_identification.cli report --run-dir $RunDir
python -m sandbox.free_space_algorithm_identification.cli verify --run-dir $RunDir
```

Expected: report generation is deterministic from receipts; it does not rerun propagation or mutate previous artifacts.

- [ ] **Step 4: Update the existing S7–S8 root-cause document only from the verified decision**

Add a dated section linking the new report and stating one of: uniquely supported candidate, propagation-branch-dependent rule, `blocked_hidden_state`, `undecided_hidden_state`, or undecidable. Preserve contrary evidence and failed gates. Do not replace earlier observations with a post hoc narrative.

If and only if the three-segment evidence uniquely localizes a production-code error, stop this identification plan and invoke `superpowers:brainstorming` followed by `superpowers:writing-plans` for a separate minimal correction. That correction must be test-driven and pass the same exact Helmholtz/RS, independent Fresnel, identity, and three-segment gates. If evidence is undecidable or indicates hidden Zemax state, do not change production propagation code.

- [ ] **Step 5: Run final verification and commit only code plus the scientific summary**

```powershell
python -m pytest tests/free_space_identification -m "not slow" -q
python -m pytest tests/test_zbf_io.py tests/test_zbf_source.py -k "not reference_phase_uses_spherical_header_metadata" -q
git diff --check
git diff --exit-code -- pop proper_v3.3.4_python/proper angular_spectrum_method sandbox/biconic_focus_baseline_utils.py sandbox/zemax_pop_benchmark/zosapi_runner.py
git add sandbox/free_space_algorithm_identification/report.py sandbox/free_space_algorithm_identification/cli.py sandbox/free_space_algorithm_identification/pipeline.py tests/free_space_identification/test_pipeline_report.py sandbox/diagnostics/zemax_free_space_propagator_identification_report.md sandbox/diagnostics/s7_s8_phase_root_cause.md
git commit -m "docs: report free-space propagator identification"
```

Expected: all tests pass; production propagation directories are unchanged by this plan; generated ZBF/CFG/NPZ/plot artifacts remain untracked in the immutable run directory.
