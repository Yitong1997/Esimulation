"""Diagnostic full-Helmholtz angular-spectrum propagation.

The phasor convention is ``exp(-i omega t)`` and a forward axial plane wave is
``exp(+i k z)``.  The large workspace carries an explicitly carrier-removed
field only internally; both returned ``PointField2D`` objects are physical
total fields with ``exp(+i k d)`` restored.  This module never uses ``Q`` or
``prop_qphase``.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Callable

import numpy as np
import scipy.fft

from .fourier import (
    _evaluate_spectrum_czt_owned_normalized,
    _forward_continuous_spectrum_owned_inplace,
)
from .metrics import FrozenRoi, FrozenRoiSet
from .models import PointField2D, SegmentSpec, UniformGrid2D


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TWO_PI = 2.0 * np.pi
_MEMORY_SAFETY_FACTOR = 1.3
_CENTRAL_PERIOD_RULE = "[-Lx/2,Lx/2) x [-Ly/2,Ly/2)"
_SOURCE_GRID_RULE = "sample_at_zero_centered_v1"
_BANDLIMIT_RULE_VERSION = "matsushima_two_ellipse_nyquist_v1"


def _require_positive_finite(value: object, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be positive and finite")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return parsed


def _require_positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return parsed


def _require_shape(shape: object, *, label: str) -> tuple[int, int]:
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError(f"{label} must be a (ny, nx) tuple")
    ny = _require_positive_integer(shape[0], label=f"{label} ny")
    nx = _require_positive_integer(shape[1], label=f"{label} nx")
    if ny < 2 or nx < 2:
        raise ValueError(f"{label} dimensions must contain at least two samples")
    return ny, nx


def _immutable_array(values: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    contiguous = np.asarray(values, dtype=dtype, order="C")
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    result = result.reshape(contiguous.shape)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class HelmholtzBranchEvaluation:
    kz_per_mm: np.ndarray
    delta_k_per_mm: np.ndarray
    evanescent: np.ndarray


@dataclass(frozen=True)
class HelmholtzTransferEvaluation:
    values: np.ndarray
    branch: HelmholtzBranchEvaluation
    underflow_count: int


@dataclass(frozen=True)
class MatsushimaBandlimit:
    mask: np.ndarray
    f_mx_cpm: float
    f_my_cpm: float
    nyquist_x_cpm: float
    nyquist_y_cpm: float
    mask_sha256: str
    rule_version: str
    rule_sha256: str

    def __post_init__(self) -> None:
        mask = _immutable_array(self.mask, dtype=np.dtype(np.bool_))
        object.__setattr__(self, "mask", mask)


@dataclass(frozen=True)
class AsmPropagationEvidence:
    """Frozen manifest values paired with the read-back values used here."""

    segment_key: str
    model_distance_mm: float
    requested_distance_mm: float
    readback_distance_mm: float
    frozen_model_sha256: str
    readback_model_sha256: str
    frozen_settings_sha256: str
    readback_settings_sha256: str
    frozen_start_artifact_sha256: str
    readback_start_artifact_sha256: str
    frozen_end_artifact_sha256: str
    readback_end_artifact_sha256: str
    start_wavelength_vacuum_mm: float
    end_wavelength_vacuum_mm: float
    start_refractive_index: float
    end_refractive_index: float
    uniform_medium_asserted: bool
    zbf_length_unit: str
    grid_length_unit: str

    def __post_init__(self) -> None:
        if not isinstance(self.segment_key, str) or not self.segment_key:
            raise ValueError("evidence segment key must be nonempty")
        for label in (
            "model_distance_mm",
            "requested_distance_mm",
            "readback_distance_mm",
            "start_wavelength_vacuum_mm",
            "end_wavelength_vacuum_mm",
            "start_refractive_index",
            "end_refractive_index",
        ):
            _require_positive_finite(getattr(self, label), label=label)
        for label in (
            "frozen_model_sha256",
            "readback_model_sha256",
            "frozen_settings_sha256",
            "readback_settings_sha256",
            "frozen_start_artifact_sha256",
            "readback_start_artifact_sha256",
            "frozen_end_artifact_sha256",
            "readback_end_artifact_sha256",
        ):
            digest = getattr(self, label)
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        if not isinstance(self.uniform_medium_asserted, bool):
            raise ValueError("uniform-medium evidence must be Boolean")
        if not isinstance(self.zbf_length_unit, str) or not isinstance(
            self.grid_length_unit, str
        ):
            raise ValueError("length units must be strings")

    @property
    def canonical_sha256(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class AsmAllocationPlan:
    source_shape: tuple[int, int]
    target_shape: tuple[int, int]
    fft_shift_batch_rows: int
    transfer_batch_rows: int
    czt_batch_size: int
    components: tuple[tuple[str, int], ...]
    phase_peaks: tuple[tuple[str, int], ...]
    estimated_peak_bytes: int
    required_available_bytes: int
    safety_factor: float = _MEMORY_SAFETY_FACTOR

    def __post_init__(self) -> None:
        names = [name for name, _ in self.components]
        if len(names) != len(set(names)) or any(
            not isinstance(name, str) or not name or value < 0
            for name, value in self.components
        ):
            raise ValueError("allocation components must have unique names and sizes")
        phase_names = [name for name, _ in self.phase_peaks]
        if len(phase_names) != len(set(phase_names)) or any(
            not isinstance(name, str) or not name or value <= 0
            for name, value in self.phase_peaks
        ):
            raise ValueError("allocation phases must have unique names and positive peaks")
        if self.estimated_peak_bytes != max(value for _, value in self.phase_peaks):
            raise ValueError("allocation estimate does not match its phase peaks")
        if self.required_available_bytes != int(
            math.ceil(self.safety_factor * self.estimated_peak_bytes)
        ):
            raise ValueError("allocation safety-factor total is inconsistent")


@dataclass(frozen=True)
class AsmBandlimitDiagnostics:
    f_mx_cpm: float
    f_my_cpm: float
    nyquist_x_cpm: float
    nyquist_y_cpm: float
    mask_sha256: str
    clipped_source_spectrum_energy_fraction: float
    rule_version: str
    rule_sha256: str
    interpretation: str = "finite-window alias control; not a physical candidate"


@dataclass(frozen=True)
class FullBlClosureDiagnostics:
    complex_relative_l2: float
    phase_rms_waves: float
    normalized_intensity_relative_l2: float
    relative_power_error: float
    within_predeclared_budgets: bool


@dataclass(frozen=True)
class AsmDiagnostics:
    segment_key: str
    model_distance_mm: float
    wavelength_medium_mm: float
    k_per_mm: float
    axial_carrier_nominal_rad: float
    axial_carrier_reduced_rad: float
    axial_carrier_bound_rad: float
    evanescent_bin_count: int
    evanescent_underflow_count: int
    bandlimit: AsmBandlimitDiagnostics
    full_bl_closure: FullBlClosureDiagnostics
    central_period_lx_mm: float
    central_period_ly_mm: float
    central_period_rule: str
    central_period_sha256: str
    source_grid_rule: str
    source_grid_sha256: str
    roi_threshold: float
    roi_mask_sha256: str
    roi_reference_zbf_sha256: str
    builder_observed_shape: tuple[int, int]
    builder_observed_dtype: str
    evidence_sha256: str
    available_memory_bytes: int


@dataclass(frozen=True)
class AsmPropagationResult:
    h_full: PointField2D
    h_bl: PointField2D
    diagnostics: AsmDiagnostics
    allocation_plan: AsmAllocationPlan


def helmholtz_delta_k(
    kappa2_per_mm2: np.ndarray, *, k_per_mm: float
) -> HelmholtzBranchEvaluation:
    """Return the forward/decaying Helmholtz branch and stable ``kz-k``."""

    k = _require_positive_finite(k_per_mm, label="k_per_mm")
    kappa2 = np.array(kappa2_per_mm2, dtype=np.float64, copy=True)
    if kappa2.size == 0 or not np.all(np.isfinite(kappa2)) or np.any(kappa2 < 0.0):
        raise ValueError("kappa squared must be a nonempty finite non-negative array")
    k_squared = k * k
    if not math.isfinite(k_squared):
        raise ValueError("k squared must remain finite")
    evanescent = kappa2 > k_squared
    kz = np.empty(kappa2.shape, dtype=np.complex128)
    kz[~evanescent] = np.sqrt(k_squared - kappa2[~evanescent])
    kz[evanescent] = 1j * np.sqrt(kappa2[evanescent] - k_squared)
    delta_k = -kappa2 / (kz + k)
    if not np.all(np.isfinite(kz)) or not np.all(np.isfinite(delta_k)):
        raise ValueError("Helmholtz branch evaluation must remain finite")
    return HelmholtzBranchEvaluation(
        kz_per_mm=kz,
        delta_k_per_mm=delta_k,
        evanescent=evanescent,
    )


def helmholtz_transfer(
    kappa2_per_mm2: np.ndarray, *, k_per_mm: float, distance_mm: float
) -> HelmholtzTransferEvaluation:
    """Return the carrier-removed transfer, preserving evanescent decay."""

    distance = _require_positive_finite(distance_mm, label="distance_mm")
    branch = helmholtz_delta_k(kappa2_per_mm2, k_per_mm=k_per_mm)
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            values = np.exp(1j * distance * branch.delta_k_per_mm)
    except FloatingPointError as exc:
        raise ValueError("Helmholtz transfer must remain finite") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("Helmholtz transfer must remain finite")
    underflow = int(np.count_nonzero(branch.evanescent & (values == 0.0j)))
    return HelmholtzTransferEvaluation(
        values=values, branch=branch, underflow_count=underflow
    )


def _bandlimit_parameters(
    *,
    wavelength_medium_mm: float,
    distance_mm: float,
    lx_mm: float,
    ly_mm: float,
    dx_mm: float,
    dy_mm: float,
) -> tuple[float, float, float, float]:
    wavelength = _require_positive_finite(
        wavelength_medium_mm, label="wavelength_medium_mm"
    )
    distance = _require_positive_finite(distance_mm, label="distance_mm")
    lx = _require_positive_finite(lx_mm, label="lx_mm")
    ly = _require_positive_finite(ly_mm, label="ly_mm")
    dx = _require_positive_finite(dx_mm, label="dx_mm")
    dy = _require_positive_finite(dy_mm, label="dy_mm")
    f_mx = 1.0 / (wavelength * math.sqrt(1.0 + (2.0 * distance / lx) ** 2))
    f_my = 1.0 / (wavelength * math.sqrt(1.0 + (2.0 * distance / ly) ** 2))
    return f_mx, f_my, 1.0 / (2.0 * dx), 1.0 / (2.0 * dy)


def _bandlimit_rule_sha256(
    *,
    wavelength_medium_mm: float,
    distance_mm: float,
    lx_mm: float,
    ly_mm: float,
    dx_mm: float,
    dy_mm: float,
    f_mx_cpm: float,
    f_my_cpm: float,
    nyquist_x_cpm: float,
    nyquist_y_cpm: float,
) -> str:
    """Hash the exact two-ellipse rule and all parameters used to apply it."""

    payload = json.dumps(
        {
            "ellipse_x": "(fx/fMx)^2 + (lambda_medium*fy)^2 <= 1",
            "ellipse_y": "(lambda_medium*fx)^2 + (fy/fMy)^2 <= 1",
            "nyquist": "abs(fx)<=1/(2*dx) and abs(fy)<=1/(2*dy)",
            "parameters": {
                "distance_mm": distance_mm,
                "dx_mm": dx_mm,
                "dy_mm": dy_mm,
                "f_mx_cpm": f_mx_cpm,
                "f_my_cpm": f_my_cpm,
                "lx_mm": lx_mm,
                "ly_mm": ly_mm,
                "nyquist_x_cpm": nyquist_x_cpm,
                "nyquist_y_cpm": nyquist_y_cpm,
                "wavelength_medium_mm": wavelength_medium_mm,
            },
            "version": _BANDLIMIT_RULE_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bandlimit_mask_values(
    fx_cpm: np.ndarray,
    fy_cpm: np.ndarray,
    *,
    wavelength_medium_mm: float,
    f_mx_cpm: float,
    f_my_cpm: float,
    nyquist_x_cpm: float,
    nyquist_y_cpm: float,
) -> np.ndarray:
    fx = fx_cpm[np.newaxis, :]
    fy = fy_cpm[:, np.newaxis]
    ellipse_x = (fx / f_mx_cpm) ** 2 + (wavelength_medium_mm * fy) ** 2 <= 1.0
    ellipse_y = (wavelength_medium_mm * fx) ** 2 + (fy / f_my_cpm) ** 2 <= 1.0
    nyquist = (np.abs(fx) <= nyquist_x_cpm) & (
        np.abs(fy) <= nyquist_y_cpm
    )
    return np.asarray(ellipse_x & ellipse_y & nyquist, dtype=np.bool_)


def _validated_frequency_axis(axis: object, *, label: str) -> np.ndarray:
    values = np.array(axis, dtype=np.float64, copy=True)
    if (
        values.ndim != 1
        or values.size < 2
        or not np.all(np.isfinite(values))
        or not np.all(np.diff(values) > 0.0)
    ):
        raise ValueError(f"{label} must be a finite increasing frequency axis")
    steps = np.diff(values)
    if not np.allclose(steps, steps[0], rtol=1e-12, atol=0.0):
        raise ValueError(f"{label} must be uniformly spaced")
    return values


def matsushima_bandlimit_mask(
    fx_cpm: np.ndarray,
    fy_cpm: np.ndarray,
    *,
    wavelength_medium_mm: float,
    distance_mm: float,
    lx_mm: float,
    ly_mm: float,
    dx_mm: float,
    dy_mm: float,
) -> MatsushimaBandlimit:
    """Build the two-ellipse intersection and independent Nyquist rectangle."""

    fx = _validated_frequency_axis(fx_cpm, label="fx_cpm")
    fy = _validated_frequency_axis(fy_cpm, label="fy_cpm")
    wavelength = _require_positive_finite(
        wavelength_medium_mm, label="wavelength_medium_mm"
    )
    parameters = _bandlimit_parameters(
        wavelength_medium_mm=wavelength,
        distance_mm=distance_mm,
        lx_mm=lx_mm,
        ly_mm=ly_mm,
        dx_mm=dx_mm,
        dy_mm=dy_mm,
    )
    f_mx, f_my, nyquist_x, nyquist_y = parameters
    mask = _bandlimit_mask_values(
        fx,
        fy,
        wavelength_medium_mm=wavelength,
        f_mx_cpm=f_mx,
        f_my_cpm=f_my,
        nyquist_x_cpm=nyquist_x,
        nyquist_y_cpm=nyquist_y,
    )
    return MatsushimaBandlimit(
        mask=mask,
        f_mx_cpm=f_mx,
        f_my_cpm=f_my,
        nyquist_x_cpm=nyquist_x,
        nyquist_y_cpm=nyquist_y,
        mask_sha256=hashlib.sha256(mask.tobytes(order="C")).hexdigest(),
        rule_version=_BANDLIMIT_RULE_VERSION,
        rule_sha256=_bandlimit_rule_sha256(
            wavelength_medium_mm=wavelength,
            distance_mm=float(distance_mm),
            lx_mm=float(lx_mm),
            ly_mm=float(ly_mm),
            dx_mm=float(dx_mm),
            dy_mm=float(dy_mm),
            f_mx_cpm=f_mx,
            f_my_cpm=f_my,
            nyquist_x_cpm=nyquist_x,
            nyquist_y_cpm=nyquist_y,
        ),
    )


def estimate_exact_peak_bytes(
    *,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    fft_shift_batch_rows: int,
    transfer_batch_rows: int,
    czt_batch_size: int,
) -> AsmAllocationPlan:
    """Conservatively account for the canonical in-memory ASM calculation."""

    ny, nx = _require_shape(source_shape, label="source_shape")
    my, mx = _require_shape(target_shape, label="target_shape")
    shift_rows = _require_positive_integer(
        fft_shift_batch_rows, label="fft_shift_batch_rows"
    )
    transfer_rows = _require_positive_integer(
        transfer_batch_rows, label="transfer_batch_rows"
    )
    czt_batch = _require_positive_integer(czt_batch_size, label="czt_batch_size")
    source_points = ny * nx
    target_points = my * mx
    complex_bytes = np.dtype(np.complex128).itemsize
    real_bytes = np.dtype(np.float64).itemsize
    boolean_bytes = np.dtype(np.bool_).itemsize
    x_fft_length = scipy.fft.next_fast_len(nx + mx - 1)
    y_fft_length = scipy.fft.next_fast_len(ny + my - 1)
    shift_block = max(
        min(shift_rows, ny // 2) * nx,
        min(shift_rows, ny) * (nx // 2),
    )
    transfer_points = min(transfer_rows, ny) * nx
    target_complex_bytes = target_points * complex_bytes
    target_real_bytes = target_points * real_bytes
    target_boolean_bytes = target_points * boolean_bytes
    components = (
        ("builder_physical_workspace", source_points * complex_bytes),
        ("builder_slow_field_allowance", source_points * complex_bytes),
        ("builder_phi_allowance", source_points * real_bytes),
        ("numpy_fft_scratch", source_points * complex_bytes),
        ("owned_mutable_spectrum", source_points * complex_bytes),
        ("fft_shift_block", shift_block * complex_bytes),
        (
            "transfer_row_temporaries",
            transfer_points
            * (real_bytes + 3 * complex_bytes + boolean_bytes),
        ),
        (
            "czt_x_convolution_workspaces",
            2 * min(czt_batch, ny) * x_fft_length * complex_bytes,
        ),
        (
            "czt_y_convolution_workspaces",
            2 * min(czt_batch, mx) * y_fft_length * complex_bytes,
        ),
        ("czt_intermediate", ny * mx * complex_bytes),
        ("czt_result", target_complex_bytes),
        ("retained_previous_physical_output", target_complex_bytes),
        ("pointfield_np_array_copy", target_complex_bytes),
        ("pointfield_tobytes_transient", target_complex_bytes),
        ("both_retained_physical_outputs", 2 * target_complex_bytes),
        ("closure_aligned_field", target_complex_bytes),
        ("closure_intensity_fields", 2 * target_real_bytes),
        (
            "closure_boolean_indexing_temporaries",
            4 * target_complex_bytes,
        ),
        ("closure_phase_temporaries", 2 * target_real_bytes),
        ("closure_roi_mask_resident", target_boolean_bytes),
    )
    component = dict(components)
    phase_peaks = (
        (
            "builder",
            component["builder_physical_workspace"]
            + component["builder_slow_field_allowance"]
            + component["builder_phi_allowance"],
        ),
        (
            "fft_and_transfer",
            component["owned_mutable_spectrum"]
            + component["numpy_fft_scratch"]
            + max(
                component["fft_shift_block"],
                component["transfer_row_temporaries"],
            ),
        ),
        (
            "czt_and_pointfield_construction",
            component["owned_mutable_spectrum"]
            + component["czt_x_convolution_workspaces"]
            + component["czt_y_convolution_workspaces"]
            + component["czt_intermediate"]
            + component["czt_result"]
            + component["retained_previous_physical_output"]
            + component["pointfield_np_array_copy"]
            + component["pointfield_tobytes_transient"],
        ),
        (
            "full_bl_closure",
            component["owned_mutable_spectrum"]
            + component["both_retained_physical_outputs"]
            + component["closure_aligned_field"]
            + component["closure_intensity_fields"]
            + component["closure_boolean_indexing_temporaries"]
            + component["closure_phase_temporaries"]
            + component["closure_roi_mask_resident"],
        ),
    )
    estimated = max(value for _, value in phase_peaks)
    return AsmAllocationPlan(
        source_shape=(ny, nx),
        target_shape=(my, mx),
        fft_shift_batch_rows=shift_rows,
        transfer_batch_rows=transfer_rows,
        czt_batch_size=czt_batch,
        components=components,
        phase_peaks=phase_peaks,
        estimated_peak_bytes=estimated,
        required_available_bytes=int(math.ceil(_MEMORY_SAFETY_FACTOR * estimated)),
    )


def available_memory_bytes_windows() -> int:
    """Return available physical bytes from ``GlobalMemoryStatusEx``."""

    if os.name != "nt":
        raise OSError("Windows GlobalMemoryStatusEx is required")

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = (
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        )

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
    available = int(status.ullAvailPhys)
    if available <= 0:
        raise OSError("GlobalMemoryStatusEx returned no available physical memory")
    return available


def _validate_evidence(
    segment: SegmentSpec, evidence: AsmPropagationEvidence
) -> tuple[float, float]:
    if not isinstance(segment, SegmentSpec):
        raise ValueError("segment must be a SegmentSpec")
    if not isinstance(evidence, AsmPropagationEvidence):
        raise ValueError("evidence must be an AsmPropagationEvidence")
    distance = _require_positive_finite(
        segment.model_distance_mm, label="SegmentSpec.model_distance_mm"
    )
    if evidence.segment_key != segment.key:
        raise ValueError("evidence segment key does not match SegmentSpec")
    for label, value in (
        ("evidence model distance", evidence.model_distance_mm),
        ("requested distance", evidence.requested_distance_mm),
        ("read-back distance", evidence.readback_distance_mm),
    ):
        if not math.isclose(float(value), distance, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{label} does not match SegmentSpec.model_distance_mm")
    for label, frozen, readback in (
        ("model", evidence.frozen_model_sha256, evidence.readback_model_sha256),
        (
            "settings",
            evidence.frozen_settings_sha256,
            evidence.readback_settings_sha256,
        ),
        (
            "start artifact",
            evidence.frozen_start_artifact_sha256,
            evidence.readback_start_artifact_sha256,
        ),
        (
            "end artifact",
            evidence.frozen_end_artifact_sha256,
            evidence.readback_end_artifact_sha256,
        ),
    ):
        if frozen != readback:
            raise ValueError(f"{label} hash does not match frozen evidence")
    if not evidence.uniform_medium_asserted:
        raise ValueError("ASM requires an asserted uniform medium")
    if evidence.zbf_length_unit != "mm" or evidence.grid_length_unit != "mm":
        raise ValueError("ZBF and grid length units must both be millimetres")
    if evidence.start_wavelength_vacuum_mm != evidence.end_wavelength_vacuum_mm:
        raise ValueError("endpoint vacuum wavelengths do not match")
    if evidence.start_refractive_index != evidence.end_refractive_index:
        raise ValueError("endpoint refractive indices do not match")
    wavelength_medium = (
        evidence.start_wavelength_vacuum_mm / evidence.start_refractive_index
    )
    wavelength_medium = _require_positive_finite(
        wavelength_medium, label="medium wavelength"
    )
    return wavelength_medium, _TWO_PI / wavelength_medium


def _central_period_receipt(
    source_grid: UniformGrid2D, target_grid: UniformGrid2D
) -> tuple[float, float, str]:
    lx = source_grid.nx * source_grid.dx_mm
    ly = source_grid.ny * source_grid.dy_mm
    if (
        np.any(target_grid.x_mm < -lx / 2.0)
        or np.any(target_grid.x_mm >= lx / 2.0)
        or np.any(target_grid.y_mm < -ly / 2.0)
        or np.any(target_grid.y_mm >= ly / 2.0)
    ):
        raise ValueError("ASM target nodes must lie inside the half-open central period")
    payload = json.dumps(
        {
            "lx_mm": lx,
            "ly_mm": ly,
            "rule": _CENTRAL_PERIOD_RULE,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return lx, ly, hashlib.sha256(payload).hexdigest()


def _grid_is_identical(left: UniformGrid2D, right: UniformGrid2D) -> bool:
    return bool(
        isinstance(left, UniformGrid2D)
        and isinstance(right, UniformGrid2D)
        and np.array_equal(left.x_mm, right.x_mm)
        and np.array_equal(left.y_mm, right.y_mm)
    )


def _validate_source_grid_and_hash(source_grid: UniformGrid2D) -> str:
    """Require the registered even-grid sample-at-zero coordinate rule."""

    expected_x = (
        np.arange(source_grid.nx, dtype=np.float64) - source_grid.nx // 2
    ) * source_grid.dx_mm
    expected_y = (
        np.arange(source_grid.ny, dtype=np.float64) - source_grid.ny // 2
    ) * source_grid.dy_mm
    if not np.array_equal(source_grid.x_mm, expected_x) or not np.array_equal(
        source_grid.y_mm, expected_y
    ):
        raise ValueError(
            "ASM source grid must use the exact sample-at-zero centered rule"
        )
    header = json.dumps(
        {
            "dx_mm": source_grid.dx_mm,
            "dy_mm": source_grid.dy_mm,
            "nx": source_grid.nx,
            "ny": source_grid.ny,
            "rule": _SOURCE_GRID_RULE,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hasher = hashlib.sha256()
    hasher.update(header)
    hasher.update(np.asarray(source_grid.x_mm, dtype="<f8").tobytes(order="C"))
    hasher.update(np.asarray(source_grid.y_mm, dtype="<f8").tobytes(order="C"))
    return hasher.hexdigest()


def _validate_frozen_rois(
    frozen_rois: FrozenRoiSet,
    *,
    target_grid: UniformGrid2D,
    evidence: AsmPropagationEvidence,
) -> FrozenRoi:
    if not isinstance(frozen_rois, FrozenRoiSet):
        raise ValueError("ASM closure requires a frozen Zemax ROI set")
    primary = frozen_rois.primary
    if not _grid_is_identical(primary.grid, target_grid):
        raise ValueError("frozen ROI grid does not match the ASM target grid")
    if primary.reference_zbf_sha256 != evidence.frozen_end_artifact_sha256:
        raise ValueError("frozen ROI reference hash does not match the endpoint ZBF")
    if primary.mask.flags.writeable:
        raise ValueError("frozen ROI mask must remain immutable")
    return primary


def _full_bl_closure(
    h_full: PointField2D, h_bl: PointField2D, primary_roi_mask: np.ndarray
) -> FullBlClosureDiagnostics:
    mask = np.asarray(primary_roi_mask)
    if mask.dtype != np.bool_ or mask.shape != h_full.values.shape or not np.any(mask):
        raise ValueError("primary ROI mask must be a nonempty target-shaped Boolean array")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            full = h_full.values
            bandlimited = h_bl.values
            overlap = np.sum(bandlimited[mask] * np.conj(full[mask]))
            piston = 0.0 if overlap == 0.0j else float(np.angle(overlap))
            aligned = bandlimited * np.exp(-1j * piston)
            full_intensity = np.abs(full) ** 2
            bl_intensity = np.abs(bandlimited) ** 2
            field_denominator = float(np.sum(np.abs(full[mask]) ** 2))
            intensity_denominator = float(np.sum(full_intensity[mask] ** 2))
            full_power = float(np.sum(full_intensity))
            if field_denominator <= 0.0 or intensity_denominator <= 0.0 or full_power <= 0.0:
                raise ValueError("full-versus-bandlimited closure requires nonzero power")
            complex_relative_l2 = float(
                np.sqrt(
                    np.sum(np.abs(aligned[mask] - full[mask]) ** 2)
                    / field_denominator
                )
            )
            residual_waves = np.angle(aligned[mask] * np.conj(full[mask])) / _TWO_PI
            phase_rms = float(
                np.sqrt(
                    np.sum(full_intensity[mask] * residual_waves**2)
                    / field_denominator
                )
            )
            intensity_relative_l2 = float(
                np.sqrt(
                    np.sum((bl_intensity[mask] - full_intensity[mask]) ** 2)
                    / intensity_denominator
                )
            )
            power_error = abs(float(np.sum(bl_intensity)) - full_power) / full_power
    except FloatingPointError as exc:
        raise ValueError("full-versus-bandlimited metrics must remain finite") from exc
    metrics = np.asarray(
        [complex_relative_l2, phase_rms, intensity_relative_l2, power_error]
    )
    if not np.all(np.isfinite(metrics)) or np.any(metrics < 0.0):
        raise ValueError("full-versus-bandlimited metrics must remain finite")
    within = bool(
        complex_relative_l2 <= 1e-8
        and phase_rms <= 1e-6
        and intensity_relative_l2 <= 1e-6
        and power_error <= 1e-8
    )
    return FullBlClosureDiagnostics(
        complex_relative_l2=complex_relative_l2,
        phase_rms_waves=phase_rms,
        normalized_intensity_relative_l2=intensity_relative_l2,
        relative_power_error=power_error,
        within_predeclared_budgets=within,
    )


def _axial_phase_bound_rad(
    segment: SegmentSpec, evidence: AsmPropagationEvidence, k_per_mm: float
) -> float:
    distance_spread = max(
        abs(evidence.model_distance_mm - segment.model_distance_mm),
        abs(evidence.requested_distance_mm - segment.model_distance_mm),
        abs(evidence.readback_distance_mm - segment.model_distance_mm),
    )
    wavelength_spread = abs(
        evidence.start_wavelength_vacuum_mm
        - evidence.end_wavelength_vacuum_mm
    )
    index_spread = abs(
        evidence.start_refractive_index - evidence.end_refractive_index
    )
    wavelength = evidence.start_wavelength_vacuum_mm
    index = evidence.start_refractive_index
    distance = segment.model_distance_mm
    evidence_bound = _TWO_PI * (
        abs(index / wavelength) * distance_spread
        + abs(distance / wavelength) * index_spread
        + abs(distance * index / wavelength**2) * wavelength_spread
    )
    nominal = k_per_mm * distance
    rounding_bound = 32.0 * np.finfo(np.float64).eps * (abs(nominal) + 1.0)
    return float(evidence_bound + rounding_bound)


def propagate_helmholtz_pair(
    *,
    segment: SegmentSpec,
    source_grid: UniformGrid2D,
    source_shape: tuple[int, int],
    target_grid: UniformGrid2D,
    evidence: AsmPropagationEvidence,
    physical_field_builder: Callable[[], tuple[np.ndarray, UniformGrid2D]],
    frozen_rois: FrozenRoiSet,
    fft_shift_batch_rows: int = 64,
    transfer_batch_rows: int = 64,
    czt_batch_size: int = 128,
    available_memory_query: Callable[[], int] | None = None,
) -> AsmPropagationResult:
    """Propagate one owned physical field through full and bandlimited kernels."""

    if not isinstance(source_grid, UniformGrid2D) or not isinstance(
        target_grid, UniformGrid2D
    ):
        raise ValueError("source and target metadata must be UniformGrid2D values")
    shape = _require_shape(source_shape, label="source_shape")
    if shape != (source_grid.ny, source_grid.nx):
        raise ValueError("source shape metadata does not match source grid")
    if source_grid.nx % 2 or source_grid.ny % 2:
        raise ValueError("canonical ASM source grids require even dimensions")
    source_grid_hash = _validate_source_grid_and_hash(source_grid)
    wavelength_medium, k_per_mm = _validate_evidence(segment, evidence)
    lx, ly, period_hash = _central_period_receipt(source_grid, target_grid)
    primary_roi = _validate_frozen_rois(
        frozen_rois, target_grid=target_grid, evidence=evidence
    )
    if not callable(physical_field_builder):
        raise ValueError("physical_field_builder must be callable")

    plan = estimate_exact_peak_bytes(
        source_shape=shape,
        target_shape=(target_grid.ny, target_grid.nx),
        fft_shift_batch_rows=fft_shift_batch_rows,
        transfer_batch_rows=transfer_batch_rows,
        czt_batch_size=czt_batch_size,
    )
    query = available_memory_bytes_windows if available_memory_query is None else available_memory_query
    if not callable(query):
        raise ValueError("available_memory_query must be callable")
    available_raw = query()
    if (
        isinstance(available_raw, (bool, np.bool_))
        or not isinstance(available_raw, Integral)
        or int(available_raw) < 0
    ):
        raise ValueError("available memory query must return non-negative bytes")
    available = int(available_raw)
    if available < plan.required_available_bytes:
        raise MemoryError(
            "available physical memory is below the 1.3 * estimated ASM peak"
        )

    built = physical_field_builder()
    if not isinstance(built, tuple) or len(built) != 2:
        raise ValueError("physical field builder must return (owned_array, exact_grid)")
    workspace, built_grid = built
    if not _grid_is_identical(built_grid, source_grid):
        raise ValueError("physical field builder grid does not match source metadata")
    if (
        not isinstance(workspace, np.ndarray)
        or workspace.dtype != np.complex128
        or workspace.shape != shape
        or not workspace.flags.c_contiguous
        or not workspace.flags.writeable
        or not workspace.flags.owndata
        or not np.all(np.isfinite(workspace))
    ):
        raise ValueError(
            "builder must return one owned, C-contiguous, writeable complex128 array"
        )

    fx, fy = _forward_continuous_spectrum_owned_inplace(
        workspace, source_grid, shift_batch_rows=plan.fft_shift_batch_rows
    )
    f_mx, f_my, nyquist_x, nyquist_y = _bandlimit_parameters(
        wavelength_medium_mm=wavelength_medium,
        distance_mm=segment.model_distance_mm,
        lx_mm=lx,
        ly_mm=ly,
        dx_mm=source_grid.dx_mm,
        dy_mm=source_grid.dy_mm,
    )
    bandlimit_rule_hash = _bandlimit_rule_sha256(
        wavelength_medium_mm=wavelength_medium,
        distance_mm=segment.model_distance_mm,
        lx_mm=lx,
        ly_mm=ly,
        dx_mm=source_grid.dx_mm,
        dy_mm=source_grid.dy_mm,
        f_mx_cpm=f_mx,
        f_my_cpm=f_my,
        nyquist_x_cpm=nyquist_x,
        nyquist_y_cpm=nyquist_y,
    )
    mask_hasher = hashlib.sha256()
    total_source_energy = 0.0
    clipped_source_energy = 0.0
    evanescent_count = 0
    underflow_count = 0
    for row0 in range(0, source_grid.ny, plan.transfer_batch_rows):
        rows = slice(row0, min(row0 + plan.transfer_batch_rows, source_grid.ny))
        mask_row = _bandlimit_mask_values(
            fx,
            fy[rows],
            wavelength_medium_mm=wavelength_medium,
            f_mx_cpm=f_mx,
            f_my_cpm=f_my,
            nyquist_x_cpm=nyquist_x,
            nyquist_y_cpm=nyquist_y,
        )
        mask_hasher.update(mask_row.tobytes(order="C"))
        try:
            with np.errstate(over="raise", invalid="raise"):
                energy = np.abs(workspace[rows, :]) ** 2
                row_total = float(np.sum(energy, dtype=np.float64))
                row_clipped = float(np.sum(energy[~mask_row], dtype=np.float64))
        except FloatingPointError as exc:
            raise ValueError("source spectrum energy must remain finite") from exc
        total_source_energy += row_total
        clipped_source_energy += row_clipped
        if not math.isfinite(total_source_energy) or not math.isfinite(
            clipped_source_energy
        ):
            raise ValueError("source spectrum energy must remain finite")
        kappa2 = (2.0 * np.pi * fx[np.newaxis, :]) ** 2 + (
            2.0 * np.pi * fy[rows, np.newaxis]
        ) ** 2
        transfer = helmholtz_transfer(
            kappa2,
            k_per_mm=k_per_mm,
            distance_mm=segment.model_distance_mm,
        )
        evanescent_count += int(np.count_nonzero(transfer.branch.evanescent))
        underflow_count += transfer.underflow_count
        workspace[rows, :] *= transfer.values
    if total_source_energy <= 0.0:
        raise ValueError("source spectrum must have finite nonzero energy")
    clipped_fraction = clipped_source_energy / total_source_energy
    if not math.isfinite(clipped_fraction) or not 0.0 <= clipped_fraction <= 1.0:
        raise ValueError("clipped source-spectrum fraction must be finite")

    carrier_nominal = k_per_mm * segment.model_distance_mm
    carrier_reduced = float(np.remainder(carrier_nominal, _TWO_PI))
    carrier = np.exp(1j * carrier_reduced)
    full_values, output_grid = _evaluate_spectrum_czt_owned_normalized(
        workspace,
        fx,
        fy,
        target_grid,
        batch_size=plan.czt_batch_size,
    )
    full_values *= carrier
    h_full = PointField2D(full_values, output_grid)
    del full_values

    for row0 in range(0, source_grid.ny, plan.transfer_batch_rows):
        rows = slice(row0, min(row0 + plan.transfer_batch_rows, source_grid.ny))
        mask_row = _bandlimit_mask_values(
            fx,
            fy[rows],
            wavelength_medium_mm=wavelength_medium,
            f_mx_cpm=f_mx,
            f_my_cpm=f_my,
            nyquist_x_cpm=nyquist_x,
            nyquist_y_cpm=nyquist_y,
        )
        workspace[rows, :] *= mask_row
    bl_values, bl_grid = _evaluate_spectrum_czt_owned_normalized(
        workspace,
        fx,
        fy,
        target_grid,
        batch_size=plan.czt_batch_size,
    )
    bl_values *= carrier
    h_bl = PointField2D(bl_values, bl_grid)
    del bl_values
    closure = _full_bl_closure(h_full, h_bl, primary_roi.mask)
    diagnostics = AsmDiagnostics(
        segment_key=segment.key,
        model_distance_mm=segment.model_distance_mm,
        wavelength_medium_mm=wavelength_medium,
        k_per_mm=k_per_mm,
        axial_carrier_nominal_rad=carrier_nominal,
        axial_carrier_reduced_rad=carrier_reduced,
        axial_carrier_bound_rad=_axial_phase_bound_rad(segment, evidence, k_per_mm),
        evanescent_bin_count=evanescent_count,
        evanescent_underflow_count=underflow_count,
        bandlimit=AsmBandlimitDiagnostics(
            f_mx_cpm=f_mx,
            f_my_cpm=f_my,
            nyquist_x_cpm=nyquist_x,
            nyquist_y_cpm=nyquist_y,
            mask_sha256=mask_hasher.hexdigest(),
            clipped_source_spectrum_energy_fraction=clipped_fraction,
            rule_version=_BANDLIMIT_RULE_VERSION,
            rule_sha256=bandlimit_rule_hash,
        ),
        full_bl_closure=closure,
        central_period_lx_mm=lx,
        central_period_ly_mm=ly,
        central_period_rule=_CENTRAL_PERIOD_RULE,
        central_period_sha256=period_hash,
        source_grid_rule=_SOURCE_GRID_RULE,
        source_grid_sha256=source_grid_hash,
        roi_threshold=primary_roi.threshold,
        roi_mask_sha256=primary_roi.mask_sha256,
        roi_reference_zbf_sha256=primary_roi.reference_zbf_sha256,
        builder_observed_shape=tuple(int(value) for value in workspace.shape),
        builder_observed_dtype=workspace.dtype.name,
        evidence_sha256=evidence.canonical_sha256,
        available_memory_bytes=available,
    )
    return AsmPropagationResult(
        h_full=h_full,
        h_bl=h_bl,
        diagnostics=diagnostics,
        allocation_plan=plan,
    )


__all__ = [
    "AsmAllocationPlan",
    "AsmBandlimitDiagnostics",
    "AsmDiagnostics",
    "AsmPropagationEvidence",
    "AsmPropagationResult",
    "FullBlClosureDiagnostics",
    "HelmholtzBranchEvaluation",
    "HelmholtzTransferEvaluation",
    "MatsushimaBandlimit",
    "available_memory_bytes_windows",
    "estimate_exact_peak_bytes",
    "helmholtz_delta_k",
    "helmholtz_transfer",
    "matsushima_bandlimit_mask",
    "propagate_helmholtz_pair",
]
