from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sandbox.free_space_algorithm_identification.biconic_case import (
    BICONIC_SEGMENTS,
    S7,
    S8,
    S12,
    S13,
    S14,
)
from sandbox.free_space_algorithm_identification.field_contract import pilot_from_zbf
from sandbox.free_space_algorithm_identification.geometry import load_segment_geometry
from sandbox.free_space_algorithm_identification.zbf_binary import (
    read_lossless_zbf,
    sha256_file,
)


_EXPECTED_ZBF_SHA256 = {
    7: "2941271c278e346820ab51b1821e3d5b533ec5186cfa6ef1851d2dd948103341",
    8: "88a64ad3e916ab3180fac364070528998b98a290fc3ee60cfcce956a5c47217d",
    12: "b276a77eb7566acfa70e65b452b3b8025f53fedf9c6fcc834f2940fddb55a2d1",
    13: "f0c8dde821ed9cfc0afad8fddd6a8af60f76372fd8211f4daf4761fcbc5eb5e2",
    14: "4014c8a43e31f9416bc202e3cc049687e044e91d4e6d31e8e909d37d874a79c5",
}


def _load_beams(baseline_dir: Path) -> dict[int, object]:
    return {
        surface: read_lossless_zbf(
            baseline_dir / f"biconic_focus_test_{surface:04d}.ZBF"
        )
        for surface in (7, 8, 12, 13, 14)
    }


def test_historical_zbf_pilots_match_the_fixed_design_mapping(
    baseline_dir: Path,
    baseline_report: Path,
) -> None:
    # Requesting both fixtures is intentional: either missing environment variable
    # skips this historical preflight and cannot create validation evidence.
    assert baseline_report.is_file()
    beams = _load_beams(baseline_dir)
    conventions = {7: S7, 8: S8, 12: S12, 13: S13, 14: S14}
    expected_zeta_mm = {
        7: 239.982966226840,
        8: 608.582967581705,
        12: -608.615263635412,
        13: -0.015263632634,
        14: 1.984736370234,
    }
    expected_inside = {7: False, 8: False, 12: False, 13: True, 14: False}

    assert {
        surface: beam.source_sha256 for surface, beam in beams.items()
    } == _EXPECTED_ZBF_SHA256
    for surface, beam in beams.items():
        pilot = pilot_from_zbf(beam, conventions[surface])
        assert pilot.zeta_mm == pytest.approx(
            expected_zeta_mm[surface], rel=0.0, abs=1e-12
        )
        assert pilot.inside is expected_inside[surface]


def test_historical_report_locks_signed_geometry_but_not_phasor_evidence(
    baseline_dir: Path,
    baseline_report: Path,
) -> None:
    beams = _load_beams(baseline_dir)
    expected_report_signed_mm = {
        "S07_S08": -368.6,
        "S12_S13": 608.6,
        "S13_S14": 2.0,
    }
    expected_raw_delta_mm = {
        "S07_S08": -368.600001354865,
        "S12_S13": 608.600000002778,
        "S13_S14": 2.000000002868,
    }

    for segment in BICONIC_SEGMENTS:
        geometry = load_segment_geometry(
            baseline_report,
            segment=segment,
            start_beam=beams[segment.start_surface],
            end_beam=beams[segment.end_surface],
        )
        assert geometry.model_distance_mm == segment.model_distance_mm
        assert geometry.propagation_distance_mm == segment.model_distance_mm
        assert geometry.report_signed_distance_mm.value == expected_report_signed_mm[
            segment.key
        ]
        assert geometry.raw_pilot_delta_mm == pytest.approx(
            expected_raw_delta_mm[segment.key], rel=0.0, abs=1e-12
        )
        np.testing.assert_allclose(
            geometry.transverse_basis_change,
            np.eye(2),
            rtol=0.0,
            atol=1e-9,
        )

    assert sha256_file(baseline_report) == (
        "d25608db0552df8764d9f022115e9469dbe407ad16d2debad5d32fe4c78a56ee"
    )
    assert sha256_file(baseline_dir / "biconic_focus_test.zmx") == (
        "ca90da9cc8fff9371ee249837e04d8d93c4bc27c8602c46331d053a3f5cc20e1"
    )
    assert sha256_file(baseline_dir / "biconic_focus_test.CFG") == (
        "446210aa3baf6dcc7c576d0a7b6cc9470619813a0aeab03e0f975e06c3dcc90f"
    )
    analytic_phase = baseline_dir / "biconic_phase.txt"
    assert analytic_phase.is_file()
    assert sha256_file(analytic_phase) == (
        "1a9d0d5045e97bea5f27bc9e88c5b45b2c837cb24bc8c63e8578175b7ec887d7"
    )
    # Static preflight has no raw ZOS-API DataGrid and therefore cannot emit or
    # stand in for an authoritative convention-validation receipt.
    assert not list(baseline_dir.glob("*convention_validation*receipt*"))
