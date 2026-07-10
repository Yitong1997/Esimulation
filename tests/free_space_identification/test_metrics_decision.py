from __future__ import annotations

import hashlib

import numpy as np
import pytest

from sandbox.free_space_algorithm_identification.decision import (
    MetricUncertainty,
    PairDecision,
    classify_pair,
    combine_uncertainty,
    resolve_roi_decisions,
    s13_s14_r_phi_given_q_gate,
)
from sandbox.free_space_algorithm_identification.metrics import (
    FrozenRoi,
    FrozenRoiSet,
    build_frozen_rois,
    compare_physical_fields,
    symmetric_complex_distance,
)
from sandbox.free_space_algorithm_identification.models import (
    PointField2D,
    UniformGrid2D,
)


_REFERENCE_SHA256 = "a" * 64


def _mask_sha256(mask: np.ndarray) -> str:
    canonical = np.asarray(mask, dtype=np.bool_, order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _frozen_roi(
    grid: UniformGrid2D,
    mask: np.ndarray,
    *,
    threshold: float = 1e-3,
) -> FrozenRoi:
    return FrozenRoi(
        threshold=threshold,
        mask=mask,
        mask_sha256=_mask_sha256(mask),
        reference_zbf_sha256=_REFERENCE_SHA256,
        grid=grid,
    )


def _full_roi_set(grid: UniformGrid2D) -> FrozenRoiSet:
    mask = np.ones((grid.ny, grid.nx), dtype=bool)
    return FrozenRoiSet(
        primary=_frozen_roi(grid, mask, threshold=1e-3),
        checks=(
            _frozen_roi(grid, mask, threshold=1e-2),
            _frozen_roi(grid, mask, threshold=1e-6),
        ),
    )


def test_roi_is_frozen_from_only_the_reference_peak_component() -> None:
    grid = UniformGrid2D.centered(nx=6, ny=6, dx_mm=0.2, dy_mm=0.3)
    values = np.zeros((6, 6), dtype=np.complex128)
    values[1, 1] = 10.0
    values[1, 2] = 2.0
    values[1, 3] = 0.2
    values[4, 4] = 9.0
    reference = PointField2D(values, grid)

    rois = build_frozen_rois(
        reference,
        reference_zbf_sha256=_REFERENCE_SHA256,
        thresholds=(1e-3, 1e-2, 1e-6),
    )

    expected_primary = np.zeros((6, 6), dtype=bool)
    expected_primary[1, 1:3] = True
    expected_low = expected_primary.copy()
    expected_low[1, 3] = True
    assert rois.primary.threshold == 1e-3
    assert tuple(roi.threshold for roi in rois.checks) == (1e-2, 1e-6)
    np.testing.assert_array_equal(rois.primary.mask, expected_primary)
    np.testing.assert_array_equal(rois.checks[0].mask, expected_primary)
    np.testing.assert_array_equal(rois.checks[1].mask, expected_low)
    assert not any(roi.mask[4, 4] for roi in (rois.primary, *rois.checks))
    assert rois.primary.mask_sha256 == _mask_sha256(expected_primary)
    assert rois.primary.reference_zbf_sha256 == _REFERENCE_SHA256
    with pytest.raises(ValueError):
        rois.primary.mask.setflags(write=True)
    with pytest.raises(ValueError, match="SHA-256"):
        build_frozen_rois(reference, reference_zbf_sha256="not-a-hash")
    with pytest.raises(ValueError, match="distinct"):
        build_frozen_rois(
            reference,
            reference_zbf_sha256=_REFERENCE_SHA256,
            thresholds=(1e-3, 1e-3, 1e-6),
        )


def test_exactly_one_primary_roi_piston_is_reused_without_fitting_error() -> None:
    grid = UniformGrid2D.centered(nx=12, ny=10, dx_mm=0.1, dy_mm=0.2)
    x, y = np.meshgrid(grid.x_mm, grid.y_mm)
    reference_values = np.exp(-(x * x + 0.7 * y * y)) * np.exp(0.15j * y)
    reference = PointField2D(reference_values, grid)
    rois = _full_roi_set(grid)

    pure_piston = PointField2D(reference.values * np.exp(0.4j), grid)
    closed = compare_physical_fields(pure_piston, reference, rois)
    assert len(closed) == 3
    assert all(metric.piston_rad == pytest.approx(0.4) for metric in closed)
    assert max(metric.complex_relative_l2 for metric in closed) < 2e-15
    assert max(metric.phase_rms_waves for metric in closed) < 2e-15

    candidate = PointField2D(
        1.25 * reference.values * np.exp(1j * (0.4 + 0.2 * x * x)), grid
    )
    metrics = compare_physical_fields(candidate, reference, rois)
    assert len({metric.piston_rad for metric in metrics}) == 1
    assert metrics[0].complex_relative_l2 > 0.2
    assert metrics[0].intensity_relative_l2 > 0.5
    assert metrics[0].phase_rms_waves > 1e-4
    assert metrics[0].full_window_power_relative_error == pytest.approx(0.5625)


def test_peak_anchored_unwrap_is_deterministic_and_ambiguity_fails_closed() -> None:
    grid = UniformGrid2D.centered(nx=5, ny=2, dx_mm=0.2, dy_mm=0.3)
    reference_values = np.ones((2, 5), dtype=np.complex128)
    reference_values[:, 0] = 2.0
    reference = PointField2D(reference_values, grid)
    phase = np.broadcast_to(np.array([0.1, 2.6, 5.1, 7.6, 10.1]), (2, 5))
    candidate = PointField2D(reference.values * np.exp(1j * phase), grid)
    roi = _frozen_roi(grid, np.ones((2, 5), dtype=bool))

    first = compare_physical_fields(candidate, reference, roi)
    second = compare_physical_fields(candidate, reference, roi)
    assert first.unwrap_status == "unique"
    assert first.phase_pv_waves == pytest.approx(10.0 / (2.0 * np.pi))
    assert first == second

    ambiguous_grid = UniformGrid2D.centered(nx=2, ny=2, dx_mm=0.2, dy_mm=0.3)
    ambiguous_reference = PointField2D(
        np.array([[2.0, 1.0], [2.0, 1.0]], dtype=np.complex128),
        ambiguous_grid,
    )
    exact_pi = np.array([[0.0, np.pi], [0.0, np.pi]])
    ambiguous_candidate = PointField2D(
        ambiguous_reference.values * np.exp(1j * exact_pi), ambiguous_grid
    )
    ambiguous_roi = _frozen_roi(
        ambiguous_grid, np.ones((2, 2), dtype=bool)
    )
    ambiguous = compare_physical_fields(
        ambiguous_candidate, ambiguous_reference, ambiguous_roi
    )
    assert ambiguous.unwrap_status == "ambiguous"
    assert np.isnan(ambiguous.phase_rms_waves)
    assert np.isnan(ambiguous.phase_pv_waves)
    with pytest.raises(ValueError, match="finite"):
        classify_pair(ambiguous.phase_rms_waves, 1e-6)

    inconsistent_phase = np.array(
        [[0.0, 2.0 * np.pi / 3.0], [-2.0 * np.pi / 3.0, 0.0]]
    )
    inconsistent = compare_physical_fields(
        PointField2D(
            ambiguous_reference.values * np.exp(1j * inconsistent_phase),
            ambiguous_grid,
        ),
        ambiguous_reference,
        ambiguous_roi,
    )
    assert inconsistent.unwrap_status == "ambiguous"
    assert np.isnan(inconsistent.phase_rms_waves)

    with_zero = ambiguous_candidate.values.copy()
    with_zero[0, 0] = 0.0
    zero_result = compare_physical_fields(
        PointField2D(with_zero, ambiguous_grid),
        ambiguous_reference,
        ambiguous_roi,
    )
    assert zero_result.unwrap_status == "ambiguous"
    assert np.isnan(zero_result.phase_rms_waves)


def test_symmetric_distance_keeps_amplitude_and_common_grid_is_strict() -> None:
    rows, columns = np.indices((4, 5))
    a = 0.7 + 0.13 * columns - 0.04 * rows + 1j * (0.2 + 0.08 * rows)
    mask = np.ones_like(a, dtype=bool)
    assert symmetric_complex_distance(
        a * np.exp(1.2j), a, mask=mask, pixel_area_mm2=0.06
    ) < 2e-15
    assert symmetric_complex_distance(
        2.0 * a, a, mask=mask, pixel_area_mm2=0.06
    ) == pytest.approx(np.sqrt(2.0 / 5.0))
    orthogonal_a = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    orthogonal_b = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    assert symmetric_complex_distance(
        orthogonal_a,
        orthogonal_b,
        mask=np.ones((2, 2), dtype=bool),
        pixel_area_mm2=0.06,
    ) == pytest.approx(np.sqrt(2.0))

    grid = UniformGrid2D.centered(nx=5, ny=4, dx_mm=0.2, dy_mm=0.3)
    changed_grid = UniformGrid2D.centered(
        nx=5, ny=4, dx_mm=np.nextafter(0.2, np.inf), dy_mm=0.3
    )
    with pytest.raises(ValueError, match="identical grid"):
        compare_physical_fields(
            PointField2D(a, changed_grid),
            PointField2D(a, grid),
            _full_roi_set(grid),
        )


def test_dimensioned_uncertainty_and_exact_three_u_five_u_boundaries() -> None:
    zemax = MetricUncertainty(1.0, 2.0, 3.0, 4.0)
    candidate = MetricUncertainty(0.5, 0.25, 0.125, 0.0625)
    common = MetricUncertainty(0.1, 0.2, 0.3, 0.4)
    combined = combine_uncertainty(
        comparison="zemax_candidate",
        zemax=zemax,
        candidate=candidate,
        common=common,
    )
    assert combined == MetricUncertainty(1.6, 2.45, 3.425, 4.4625)

    floor_only = combine_uncertainty(
        comparison="candidate_pair",
        candidate_a=MetricUncertainty(0.0, 0.0, 0.0, 0.0),
        candidate_b=MetricUncertainty(0.0, 0.0, 0.0, 0.0),
        common=MetricUncertainty(0.0, 0.0, 0.0, 0.0),
    )
    assert floor_only == MetricUncertainty(1e-12, 1e-10, 0.0, 0.0)

    assert classify_pair(3.0, 1.0) is PairDecision.CONSISTENT
    assert classify_pair(np.nextafter(3.0, np.inf), 1.0) is PairDecision.UNDECIDED
    assert classify_pair(5.0, 1.0) is PairDecision.UNDECIDED
    assert classify_pair(np.nextafter(5.0, np.inf), 1.0) is PairDecision.EXCLUDED
    with pytest.raises(ValueError, match="finite"):
        MetricUncertainty(np.nan, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        MetricUncertainty(0.0, -1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="positive"):
        classify_pair(0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        classify_pair(np.nan, 1.0)


def test_roi_sensitivity_and_candidate_pair_budget_exclude_zemax_repeat() -> None:
    decisions = tuple(classify_pair(value, 1.0) for value in (2.0, 4.0, 6.0))
    assert resolve_roi_decisions(decisions) is PairDecision.UNDECIDED_ROI_SENSITIVITY
    assert resolve_roi_decisions(
        (PairDecision.EXCLUDED,) * 3
    ) is PairDecision.EXCLUDED

    candidate_a = MetricUncertainty(0.2, 0.3, 0.4, 0.5)
    candidate_b = MetricUncertainty(0.4, 0.5, 0.6, 0.7)
    common = MetricUncertainty(0.1, 0.1, 0.1, 0.1)
    zemax_repeat = MetricUncertainty(9.0, 9.0, 9.0, 9.0)
    pair = combine_uncertainty(
        comparison="candidate_pair",
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        common=common,
    )
    np.testing.assert_allclose(
        [
            pair.complex_distance,
            pair.phase_waves,
            pair.intensity_relative,
            pair.power_relative,
        ],
        [0.7, 0.9, 1.1, 1.3],
        rtol=0.0,
        atol=2e-16,
    )
    with pytest.raises(ValueError, match="Zemax"):
        combine_uncertainty(
            comparison="candidate_pair",
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            common=common,
            zemax=zemax_repeat,
        )


def test_s13_s14_r_phi_given_q_structural_identity_is_a_hard_gate() -> None:
    grid = UniformGrid2D.centered(nx=4, ny=4, dx_mm=0.2, dy_mm=0.3)
    x, y = np.meshgrid(grid.x_mm, grid.y_mm)
    f_q = PointField2D(
        (1.0 + 0.1 * x - 0.05 * y) * np.exp(0.2j * x), grid
    )
    phi_minus_q = 0.3 * x * x - 0.2 * y
    r_phi_given_q = PointField2D(
        f_q.values * np.exp(1j * (phi_minus_q + 0.7)), grid
    )
    roi = _frozen_roi(grid, np.ones((4, 4), dtype=bool))
    uncertainty = MetricUncertainty(1e-9, 1e-9, 1e-9, 1e-9)

    passed = s13_s14_r_phi_given_q_gate(
        r_phi_given_q,
        f_q,
        phi_minus_q,
        roi,
        uncertainty=uncertainty,
        candidate_model="R_Phi_given_Q",
    )
    assert passed.overall is PairDecision.CONSISTENT
    assert passed.intensity is PairDecision.CONSISTENT
    assert passed.power is PairDecision.CONSISTENT
    assert passed.phase is PairDecision.CONSISTENT

    changed_values = r_phi_given_q.values.copy()
    changed_values[0, 0] *= 1.01
    failed = s13_s14_r_phi_given_q_gate(
        PointField2D(changed_values, grid),
        f_q,
        phi_minus_q,
        roi,
        uncertainty=MetricUncertainty(1e-4, 1e-4, 1e-4, 1e-4),
        candidate_model="R_Phi_given_Q",
    )
    assert failed.intensity is PairDecision.EXCLUDED
    assert failed.overall is PairDecision.EXCLUDED

    phase_error = PointField2D(
        r_phi_given_q.values * np.exp(0.02j * x), grid
    )
    failed_phase = s13_s14_r_phi_given_q_gate(
        phase_error,
        f_q,
        phi_minus_q,
        roi,
        uncertainty=MetricUncertainty(1e-5, 1e-5, 1e-5, 1e-5),
        candidate_model="R_Phi_given_Q",
    )
    assert failed_phase.intensity is PairDecision.CONSISTENT
    assert failed_phase.power is PairDecision.CONSISTENT
    assert failed_phase.phase is PairDecision.EXCLUDED
    assert failed_phase.overall is PairDecision.EXCLUDED

    with pytest.raises(ValueError, match="R_Phi_given_Q"):
        s13_s14_r_phi_given_q_gate(
            r_phi_given_q,
            f_q,
            phi_minus_q,
            roi,
            uncertainty=uncertainty,
            candidate_model="R_Phi_given_Phi",
        )
