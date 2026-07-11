from __future__ import annotations

import hashlib
from dataclasses import replace

import mpmath as mp
import numpy as np
import pytest

from sandbox.free_space_algorithm_identification.asm import (
    AsmPropagationEvidence,
    estimate_exact_peak_bytes,
    helmholtz_delta_k,
    helmholtz_transfer,
    matsushima_bandlimit_mask,
    propagate_helmholtz_pair,
)
from sandbox.free_space_algorithm_identification.biconic_case import BICONIC_SEGMENTS
from sandbox.free_space_algorithm_identification.fourier import (
    _forward_continuous_spectrum_owned_inplace,
    forward_continuous_spectrum,
)
from sandbox.free_space_algorithm_identification.metrics import (
    FrozenRoiSet,
    build_frozen_rois,
)
from sandbox.free_space_algorithm_identification.models import (
    PointField2D,
    SegmentSpec,
    UniformGrid2D,
)


def _segment(key: str) -> SegmentSpec:
    return next(segment for segment in BICONIC_SEGMENTS if segment.key == key)


def _evidence(
    segment: SegmentSpec, *, wavelength_vacuum_mm: float = 0.5
) -> AsmPropagationEvidence:
    hashes = [f"{value:064x}" for value in range(1, 5)]
    return AsmPropagationEvidence(
        segment_key=segment.key,
        model_distance_mm=segment.model_distance_mm,
        requested_distance_mm=segment.model_distance_mm,
        readback_distance_mm=segment.model_distance_mm,
        frozen_model_sha256=hashes[0],
        readback_model_sha256=hashes[0],
        frozen_settings_sha256=hashes[1],
        readback_settings_sha256=hashes[1],
        frozen_start_artifact_sha256=hashes[2],
        readback_start_artifact_sha256=hashes[2],
        frozen_end_artifact_sha256=hashes[3],
        readback_end_artifact_sha256=hashes[3],
        start_wavelength_vacuum_mm=wavelength_vacuum_mm,
        end_wavelength_vacuum_mm=wavelength_vacuum_mm,
        start_refractive_index=1.0,
        end_refractive_index=1.0,
        uniform_medium_asserted=True,
        zbf_length_unit="mm",
        grid_length_unit="mm",
    )


def _frozen_rois(
    grid: UniformGrid2D, evidence: AsmPropagationEvidence
) -> FrozenRoiSet:
    reference = PointField2D(
        np.ones((grid.ny, grid.nx), dtype=np.complex128), grid
    )
    return build_frozen_rois(
        reference,
        reference_zbf_sha256=evidence.frozen_end_artifact_sha256,
    )


def _direct_continuous_spectrum(
    values: np.ndarray,
    grid: UniformGrid2D,
    fx_cpm: np.ndarray,
    fy_cpm: np.ndarray,
) -> np.ndarray:
    x_kernel = np.exp(-2j * np.pi * np.outer(grid.x_mm, fx_cpm))
    y_kernel = np.exp(-2j * np.pi * np.outer(fy_cpm, grid.y_mm))
    return (y_kernel @ values @ x_kernel) * grid.pixel_area_mm2


def _direct_inverse(
    spectrum: np.ndarray,
    fx_cpm: np.ndarray,
    fy_cpm: np.ndarray,
    output: UniformGrid2D,
) -> np.ndarray:
    x_kernel = np.exp(2j * np.pi * np.outer(fx_cpm, output.x_mm))
    y_kernel = np.exp(2j * np.pi * np.outer(output.y_mm, fy_cpm))
    dfx = float(fx_cpm[1] - fx_cpm[0])
    dfy = float(fy_cpm[1] - fy_cpm[0])
    return (y_kernel @ spectrum @ x_kernel) * dfx * dfy


def test_helmholtz_branches_stable_delta_k_and_positive_distance() -> None:
    k_per_mm = 10.0
    kappa2 = np.array([9.0, 121.0])
    branch = helmholtz_delta_k(kappa2, k_per_mm=k_per_mm)
    assert branch.evanescent.tolist() == [False, True]
    assert branch.kz_per_mm[0] == pytest.approx(np.sqrt(91.0) + 0j)
    assert branch.kz_per_mm[1] == pytest.approx(1j * np.sqrt(21.0))
    assert np.all(branch.kz_per_mm.real >= 0.0)
    assert np.all(branch.kz_per_mm.imag >= 0.0)

    distance_mm = 0.3
    transfer = helmholtz_transfer(
        kappa2, k_per_mm=k_per_mm, distance_mm=distance_mm
    )
    restored_ratio = transfer.values * np.exp(
        1j * np.remainder(k_per_mm * distance_mm, 2.0 * np.pi)
    )
    np.testing.assert_allclose(
        restored_ratio,
        np.exp(1j * distance_mm * branch.kz_per_mm),
        rtol=2e-15,
        atol=2e-15,
    )
    alpha = float(branch.kz_per_mm[1].imag)
    assert transfer.values[1] == pytest.approx(
        np.exp(-alpha * distance_mm) * np.exp(-1j * k_per_mm * distance_mm)
    )

    underflow = helmholtz_transfer(
        np.array([k_per_mm**2 + 10_000.0**2]),
        k_per_mm=k_per_mm,
        distance_mm=0.1,
    )
    assert underflow.underflow_count == 1
    assert underflow.values[0] == 0.0j
    for bad_distance in (0.0, -1.0):
        with pytest.raises(ValueError, match="positive"):
            helmholtz_transfer(
                np.array([1.0]),
                k_per_mm=k_per_mm,
                distance_mm=bad_distance,
            )

    mp.mp.dps = 80
    large_k = 1.0e12
    tiny_kappa2 = 1.0
    stable = helmholtz_delta_k(
        np.array([tiny_kappa2]), k_per_mm=large_k
    ).delta_k_per_mm[0].real
    oracle = mp.sqrt(mp.mpf(str(large_k)) ** 2 - mp.mpf(1)) - mp.mpf(
        str(large_k)
    )
    assert stable == pytest.approx(float(oracle), rel=2e-16, abs=0.0)
    assert np.sqrt(large_k**2 - tiny_kappa2) - large_k == 0.0


def test_rectangular_matsushima_mask_is_two_ellipse_intersection() -> None:
    wavelength_medium_mm = 0.01064
    distance_mm = 2.0
    lx_mm = ly_mm = 4.234
    dx_mm = dy_mm = lx_mm / 4096
    f_m = 1.0 / (
        wavelength_medium_mm
        * np.sqrt(1.0 + (2.0 * distance_mm / lx_mm) ** 2)
    )
    frequencies = np.array([-f_m, 0.0, f_m])
    result = matsushima_bandlimit_mask(
        frequencies,
        frequencies,
        wavelength_medium_mm=wavelength_medium_mm,
        distance_mm=distance_mm,
        lx_mm=lx_mm,
        ly_mm=ly_mm,
        dx_mm=dx_mm,
        dy_mm=dy_mm,
    )
    assert result.f_mx_cpm == pytest.approx(68.3185, rel=2e-6)
    assert result.f_my_cpm == pytest.approx(result.f_mx_cpm)
    assert result.mask[1, 2]
    assert result.mask[2, 1]
    assert not result.mask[2, 2]
    assert np.hypot(result.f_mx_cpm, result.f_my_cpm) == pytest.approx(
        96.6169, rel=2e-6
    )
    diagonal_single_axis_limit = 1.0 / np.sqrt(
        1.0 / result.f_mx_cpm**2 + wavelength_medium_mm**2
    )
    assert diagonal_single_axis_limit == pytest.approx(55.2612, rel=2e-6)
    assert result.mask_sha256 == hashlib.sha256(
        result.mask.tobytes(order="C")
    ).hexdigest()
    assert result.rule_version == "matsushima_two_ellipse_nyquist_v1"
    assert len(result.rule_sha256) == 64

    nyquist_control = matsushima_bandlimit_mask(
        np.array([-6.0, 0.0, 6.0]),
        np.array([-2.0, 0.0, 2.0]),
        wavelength_medium_mm=0.01,
        distance_mm=0.1,
        lx_mm=1000.0,
        ly_mm=1000.0,
        dx_mm=0.1,
        dy_mm=0.2,
    )
    assert nyquist_control.f_mx_cpm > nyquist_control.nyquist_x_cpm
    assert nyquist_control.f_my_cpm > nyquist_control.nyquist_y_cpm
    assert not np.any(nyquist_control.mask[:, (0, 2)])
    assert nyquist_control.mask[1, 1]


def test_small_rectangular_full_and_bl_fields_match_explicit_dft() -> None:
    segment = replace(_segment("S13_S14"), model_distance_mm=0.3)
    source = UniformGrid2D.centered(nx=8, ny=6, dx_mm=0.2, dy_mm=0.25)
    x, y = np.meshgrid(source.x_mm, source.y_mm)
    values = (
        0.8
        + 0.11 * x
        - 0.07 * y
        + 0.05j * x * y
        + 0.2 * np.exp(2j * np.pi * (1.25 * x - 2.0 * y))
    )
    target = UniformGrid2D.centered(nx=5, ny=4, dx_mm=0.17, dy_mm=0.2)
    evidence = _evidence(segment, wavelength_vacuum_mm=0.5)
    rois = _frozen_rois(target, evidence)

    owned_spectrum = values.copy(order="C")
    owned_fx, owned_fy = _forward_continuous_spectrum_owned_inplace(
        owned_spectrum, source, shift_batch_rows=2
    )
    immutable_spectrum = forward_continuous_spectrum(PointField2D(values, source))
    np.testing.assert_array_equal(owned_fx, immutable_spectrum.fx_cpm)
    np.testing.assert_array_equal(owned_fy, immutable_spectrum.fy_cpm)
    np.testing.assert_allclose(
        owned_spectrum, immutable_spectrum.values, rtol=3e-15, atol=3e-15
    )

    result = propagate_helmholtz_pair(
        segment=segment,
        source_grid=source,
        source_shape=values.shape,
        target_grid=target,
        evidence=evidence,
        physical_field_builder=lambda: (values.copy(order="C"), source),
        frozen_rois=rois,
        fft_shift_batch_rows=2,
        transfer_batch_rows=2,
        czt_batch_size=2,
        available_memory_query=lambda: 1 << 50,
    )

    fx = np.fft.fftshift(np.fft.fftfreq(source.nx, source.dx_mm))
    fy = np.fft.fftshift(np.fft.fftfreq(source.ny, source.dy_mm))
    source_spectrum = _direct_continuous_spectrum(values, source, fx, fy)
    fx2, fy2 = np.meshgrid(fx, fy)
    k_per_mm = 2.0 * np.pi / 0.5
    kappa2 = (2.0 * np.pi * fx2) ** 2 + (2.0 * np.pi * fy2) ** 2
    kz = np.empty_like(kappa2, dtype=np.complex128)
    propagating = kappa2 <= k_per_mm**2
    kz[propagating] = np.sqrt(k_per_mm**2 - kappa2[propagating])
    kz[~propagating] = 1j * np.sqrt(kappa2[~propagating] - k_per_mm**2)
    physical_kernel = np.exp(1j * segment.model_distance_mm * kz)

    lx_mm = source.nx * source.dx_mm
    ly_mm = source.ny * source.dy_mm
    f_mx = 1.0 / (
        0.5 * np.sqrt(1.0 + (2.0 * segment.model_distance_mm / lx_mm) ** 2)
    )
    f_my = 1.0 / (
        0.5 * np.sqrt(1.0 + (2.0 * segment.model_distance_mm / ly_mm) ** 2)
    )
    mask = (
        ((fx2 / f_mx) ** 2 + (0.5 * fy2) ** 2 <= 1.0)
        & ((0.5 * fx2) ** 2 + (fy2 / f_my) ** 2 <= 1.0)
        & (np.abs(fx2) <= 1.0 / (2.0 * source.dx_mm))
        & (np.abs(fy2) <= 1.0 / (2.0 * source.dy_mm))
    )
    expected_full = _direct_inverse(
        source_spectrum * physical_kernel, fx, fy, result.h_full.grid
    )
    expected_bl = _direct_inverse(
        source_spectrum * physical_kernel * mask, fx, fy, result.h_bl.grid
    )
    np.testing.assert_allclose(
        result.h_full.values, expected_full, rtol=3e-12, atol=3e-12
    )
    np.testing.assert_allclose(
        result.h_bl.values, expected_bl, rtol=3e-12, atol=3e-12
    )
    assert result.diagnostics.axial_carrier_reduced_rad == pytest.approx(
        np.remainder(k_per_mm * segment.model_distance_mm, 2.0 * np.pi)
    )
    assert result.diagnostics.bandlimit.mask_sha256 == hashlib.sha256(
        mask.tobytes(order="C")
    ).hexdigest()
    assert result.diagnostics.bandlimit.rule_version == (
        "matsushima_two_ellipse_nyquist_v1"
    )
    assert len(result.diagnostics.bandlimit.rule_sha256) == 64
    assert result.diagnostics.builder_observed_shape == values.shape
    assert result.diagnostics.builder_observed_dtype == "complex128"
    assert result.diagnostics.roi_threshold == rois.primary.threshold
    assert result.diagnostics.roi_mask_sha256 == rois.primary.mask_sha256
    assert result.diagnostics.roi_reference_zbf_sha256 == (
        evidence.frozen_end_artifact_sha256
    )
    assert result.diagnostics.source_grid_rule == "sample_at_zero_centered_v1"
    assert len(result.diagnostics.source_grid_sha256) == 64

    builder_called = False

    def forbidden_builder():
        nonlocal builder_called
        builder_called = True
        raise AssertionError("period gate must run before the builder")

    outside = UniformGrid2D(
        x_mm=np.array([lx_mm / 2.0 - 0.1, lx_mm / 2.0]),
        y_mm=np.array([-0.1, 0.0]),
    )
    outside_rois = _frozen_rois(outside, evidence)
    with pytest.raises(ValueError, match="central period"):
        propagate_helmholtz_pair(
            segment=segment,
            source_grid=source,
            source_shape=values.shape,
            target_grid=outside,
            evidence=evidence,
            physical_field_builder=forbidden_builder,
            frozen_rois=outside_rois,
            available_memory_query=lambda: 1 << 50,
        )
    assert not builder_called

    translated_source = UniformGrid2D(
        x_mm=source.x_mm + 0.01,
        y_mm=source.y_mm,
    )
    with pytest.raises(ValueError, match="sample-at-zero"):
        propagate_helmholtz_pair(
            segment=segment,
            source_grid=translated_source,
            source_shape=values.shape,
            target_grid=target,
            evidence=evidence,
            physical_field_builder=forbidden_builder,
            frozen_rois=rois,
            available_memory_query=lambda: 1 << 50,
        )
    assert not builder_called


def test_low_na_gaussian_and_full_bl_closure() -> None:
    segment = _segment("S13_S14")
    wavelength_mm = 0.0005
    source = UniformGrid2D.centered(nx=64, ny=48, dx_mm=0.05, dy_mm=0.06)
    x, y = np.meshgrid(source.x_mm, source.y_mm)
    waist_mm = 0.3
    input_values = np.exp(-(x**2 + y**2) / waist_mm**2).astype(np.complex128)
    evidence = _evidence(segment, wavelength_vacuum_mm=wavelength_mm)
    rois = _frozen_rois(source, evidence)

    def builder_with_immutable_roi_check():
        with pytest.raises(ValueError):
            rois.primary.mask[0, 0] = False
        return input_values.copy(order="C"), source

    result = propagate_helmholtz_pair(
        segment=segment,
        source_grid=source,
        source_shape=input_values.shape,
        target_grid=source,
        evidence=evidence,
        physical_field_builder=builder_with_immutable_roi_check,
        frozen_rois=rois,
        fft_shift_batch_rows=8,
        transfer_batch_rows=8,
        czt_batch_size=8,
        available_memory_query=lambda: 1 << 50,
    )

    k_per_mm = 2.0 * np.pi / wavelength_mm
    rayleigh_mm = k_per_mm * waist_mm**2 / 2.0
    normalized_distance = segment.model_distance_mm / rayleigh_mm
    paraxial_envelope = np.exp(
        -(x**2 + y**2)
        / (waist_mm**2 * (1.0 + 1j * normalized_distance))
    ) / (1.0 + 1j * normalized_distance)
    analytic = paraxial_envelope * np.exp(
        1j * np.remainder(k_per_mm * segment.model_distance_mm, 2.0 * np.pi)
    )
    relative_l2 = np.linalg.norm(result.h_full.values - analytic) / np.linalg.norm(
        analytic
    )
    assert relative_l2 < 2e-8

    closure = result.diagnostics.full_bl_closure
    assert closure.complex_relative_l2 <= 1e-8
    assert closure.phase_rms_waves <= 1e-6
    assert closure.normalized_intensity_relative_l2 <= 1e-6
    assert closure.relative_power_error <= 1e-8
    assert closure.within_predeclared_budgets


def test_evidence_and_memory_fail_before_physical_field_builder() -> None:
    segment = _segment("S13_S14")
    source = UniformGrid2D.centered(nx=4, ny=6, dx_mm=0.2, dy_mm=0.15)
    target = UniformGrid2D.centered(nx=3, ny=4, dx_mm=0.2, dy_mm=0.15)
    valid = _evidence(segment)
    bad_evidence = (
        replace(valid, end_wavelength_vacuum_mm=0.51),
        replace(valid, end_refractive_index=1.01),
        replace(valid, grid_length_unit="m"),
        replace(valid, requested_distance_mm=segment.model_distance_mm + 0.1),
        replace(valid, readback_model_sha256="f" * 64),
        replace(valid, uniform_medium_asserted=False),
    )

    calls = 0

    def sentinel_builder():
        nonlocal calls
        calls += 1
        raise AssertionError("a failed preflight must not invoke the builder")

    common = dict(
        segment=segment,
        source_grid=source,
        source_shape=(source.ny, source.nx),
        target_grid=target,
        physical_field_builder=sentinel_builder,
        frozen_rois=_frozen_rois(target, valid),
        available_memory_query=lambda: 1 << 50,
    )
    for evidence in bad_evidence:
        with pytest.raises(ValueError):
            propagate_helmholtz_pair(evidence=evidence, **common)
    assert calls == 0

    plan = estimate_exact_peak_bytes(
        source_shape=(source.ny, source.nx),
        target_shape=(target.ny, target.nx),
        fft_shift_batch_rows=2,
        transfer_batch_rows=3,
        czt_batch_size=2,
    )
    components = dict(plan.components)
    assert {
        "builder_physical_workspace",
        "builder_slow_field_allowance",
        "builder_phi_allowance",
        "numpy_fft_scratch",
        "owned_mutable_spectrum",
        "fft_shift_block",
        "transfer_row_temporaries",
        "czt_x_convolution_workspaces",
        "czt_y_convolution_workspaces",
        "czt_intermediate",
        "czt_result",
        "retained_previous_physical_output",
        "pointfield_np_array_copy",
        "pointfield_tobytes_transient",
        "both_retained_physical_outputs",
        "closure_aligned_field",
        "closure_intensity_fields",
        "closure_boolean_indexing_temporaries",
        "closure_phase_temporaries",
    } <= set(components)
    phase_peaks = dict(plan.phase_peaks)
    assert set(phase_peaks) == {
        "builder",
        "fft_and_transfer",
        "czt_and_pointfield_construction",
        "full_bl_closure",
    }
    assert phase_peaks["builder"] == sum(
        components[name]
        for name in (
            "builder_physical_workspace",
            "builder_slow_field_allowance",
            "builder_phi_allowance",
        )
    )
    assert phase_peaks["fft_and_transfer"] == (
        components["owned_mutable_spectrum"]
        + components["numpy_fft_scratch"]
        + max(
            components["fft_shift_block"],
            components["transfer_row_temporaries"],
        )
    )
    assert phase_peaks["czt_and_pointfield_construction"] == sum(
        components[name]
        for name in (
            "owned_mutable_spectrum",
            "czt_x_convolution_workspaces",
            "czt_y_convolution_workspaces",
            "czt_intermediate",
            "czt_result",
            "retained_previous_physical_output",
            "pointfield_np_array_copy",
            "pointfield_tobytes_transient",
        )
    )
    assert phase_peaks["full_bl_closure"] == sum(
        components[name]
        for name in (
            "owned_mutable_spectrum",
            "both_retained_physical_outputs",
            "closure_aligned_field",
            "closure_intensity_fields",
            "closure_boolean_indexing_temporaries",
            "closure_phase_temporaries",
            "closure_roi_mask_resident",
        )
    )
    assert plan.estimated_peak_bytes == max(phase_peaks.values())
    assert plan.required_available_bytes == int(
        np.ceil(1.3 * plan.estimated_peak_bytes)
    )

    target_dominant = estimate_exact_peak_bytes(
        source_shape=(4, 4),
        target_shape=(1000, 1000),
        fft_shift_batch_rows=2,
        transfer_batch_rows=2,
        czt_batch_size=2,
    )
    target_complex_bytes = 1000 * 1000 * np.dtype(np.complex128).itemsize
    assert target_dominant.estimated_peak_bytes >= 7 * target_complex_bytes

    wrong_hash_rois = build_frozen_rois(
        PointField2D(
            np.ones((target.ny, target.nx), dtype=np.complex128), target
        ),
        reference_zbf_sha256="e" * 64,
    )
    wrong_grid = UniformGrid2D.centered(
        nx=target.nx,
        ny=target.ny,
        dx_mm=target.dx_mm * 1.1,
        dy_mm=target.dy_mm,
    )
    wrong_grid_rois = _frozen_rois(wrong_grid, valid)
    for frozen_rois in (wrong_hash_rois, wrong_grid_rois):
        with pytest.raises(ValueError, match="ROI"):
            propagate_helmholtz_pair(
                evidence=valid,
                frozen_rois=frozen_rois,
                **{
                    key: value
                    for key, value in common.items()
                    if key != "frozen_rois"
                },
            )
    assert calls == 0

    with pytest.raises(MemoryError, match="1.3"):
        propagate_helmholtz_pair(
            evidence=valid,
            available_memory_query=lambda: 0,
            **{key: value for key, value in common.items() if key != "available_memory_query"},
        )
    assert calls == 0
