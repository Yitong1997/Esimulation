from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest

from sandbox.free_space_algorithm_identification.artifacts import ArtifactRef
from sandbox.free_space_algorithm_identification.field_contract import (
    MappedZbfField,
    PilotState,
    reference_phases,
)
from sandbox.free_space_algorithm_identification.identity import (
    CalibratedEndpoint,
    IdentityBinding,
    IdentityNativeReport,
    IdentityPolicy,
    PropagationBinding,
    SampleConventionPolicy,
    SampleConventionProbe,
    SampleConventionResult,
    apply_entrance_calibration,
    classify_sample_value_convention,
    entrance_settings_sha256,
    evaluate_start_identity,
    physical_grid_sha256,
)
from sandbox.free_space_algorithm_identification.models import (
    PointField2D,
    UniformGrid2D,
)
from sandbox.free_space_algorithm_identification.native_report import (
    NativeSettingsReadback,
)


_INPUT_SHA = "1" * 64
_REWRITE_SHA = "2" * 64
_ENDPOINT_SHA = "3" * 64
_CONVENTION_SHA = "6" * 64
_ENTRANCE_SHA = "7" * 64
_RUN_UUID = "a" * 32
_WAVELENGTH_MM = 0.01064
_INDEX = 1.0


def _grid(*, dx_mm: float = 0.08) -> UniformGrid2D:
    return UniformGrid2D.centered(nx=48, ny=40, dx_mm=dx_mm, dy_mm=0.07)


def _values(grid: UniformGrid2D) -> np.ndarray:
    xx, yy = np.meshgrid(grid.x_mm, grid.y_mm)
    envelope = np.exp(-0.7 * ((xx / 0.72) ** 2 + (yy / 0.58) ** 2))
    phase = 0.11 * xx - 0.07 * yy + 0.025 * xx * yy
    return envelope * np.exp(1j * phase)


def _mapped(
    values: np.ndarray,
    grid: UniformGrid2D,
    source_sha256: str,
    *,
    convention_sha256: str = _CONVENTION_SHA,
) -> MappedZbfField:
    waist = 0.05
    pilot = PilotState(
        zeta_mm=12.0,
        rayleigh_mm=np.pi * _INDEX * waist**2 / _WAVELENGTH_MM,
        waist_mm=waist,
    )
    references = reference_phases(
        grid,
        pilot,
        wavelength_vacuum_mm=_WAVELENGTH_MM,
        refractive_index=_INDEX,
    )
    physical = np.asarray(values, dtype=np.complex128)
    return MappedZbfField(
        physical=PointField2D(physical, grid),
        reference_relative=physical * np.exp(-1j * references.phi_rad),
        references=references,
        pilot=pilot,
        source_sha256=source_sha256,
        convention_evidence_sha256=convention_sha256,
        sample_value_convention="cell_energy",
    )


def _ref(
    relative_path: str,
    sha256: str,
    *,
    stage: str,
    case: str = "N1024_fixed",
) -> ArtifactRef:
    return ArtifactRef(
        run_id="runA",
        run_instance_uuid=_RUN_UUID,
        producer_stage=stage,
        producer_case=case,
        relative_path=relative_path,
        byte_count=123,
        sha256=sha256,
    )


def _convention_probe(
    n: int,
    *,
    hypothesis: str = "cell_energy",
    total_token: str = "1.000000",
    peak_token: str = "0.0024906",
) -> SampleConventionProbe:
    width_mm = 256.0
    step_mm = width_mm / n
    pixel_area_mm2 = step_mm**2
    total = float(total_token)
    peak = float(peak_token)
    if hypothesis == "cell_energy":
        raw_energy = total
        raw_peak = peak * pixel_area_mm2
    elif hypothesis == "point_value":
        raw_energy = total / pixel_area_mm2
        raw_peak = peak
    else:
        raw_energy = 0.37
        raw_peak = 0.13 * pixel_area_mm2
    return SampleConventionProbe._synthetic(
        case_id=f"N{n}_fixed",
        repeat_id=f"I{n}",
        nx=n,
        ny=n,
        dx_mm=step_mm,
        dy_mm=step_mm,
        raw_energy=raw_energy,
        raw_peak=raw_peak,
        report=IdentityNativeReport.parse(
            "Peak Irradiance = "
            f"{peak_token} Watts/Millimeters^2, Total Power = {total_token} Watts"
        ),
    )


def _cell_convention() -> SampleConventionResult:
    return SampleConventionResult._synthetic(
        tuple(_convention_probe(n) for n in (1024, 2048, 4096))
    )


def _probe_with_pilot_roundtrip(
    probe: SampleConventionProbe,
    *,
    position_relative_drift: float,
    waist_relative_drift: float,
    rayleigh_relative_error: float = 0.0,
) -> SampleConventionProbe:
    waist_x = probe.pilot_wx_mm * (1.0 + waist_relative_drift)
    waist_y = probe.pilot_wy_mm * (1.0 + waist_relative_drift)
    rayleigh_x = (
        np.pi
        * probe.refractive_index
        * waist_x**2
        / probe.wavelength_vacuum_mm
    ) * (1.0 + rayleigh_relative_error)
    rayleigh_y = (
        np.pi
        * probe.refractive_index
        * waist_y**2
        / probe.wavelength_vacuum_mm
    ) * (1.0 + rayleigh_relative_error)
    values = {item.name: getattr(probe, item.name) for item in fields(probe)}
    values.update(
        pilot_zx_mm=probe.pilot_zx_mm * (1.0 + position_relative_drift),
        pilot_rx_mm=rayleigh_x,
        pilot_wx_mm=waist_x,
        pilot_zy_mm=probe.pilot_zy_mm * (1.0 + position_relative_drift),
        pilot_ry_mm=rayleigh_y,
        pilot_wy_mm=waist_y,
    )
    return SampleConventionProbe._create(**values)


def _binding(
    grid: UniformGrid2D,
    *,
    sample_convention=None,
) -> IdentityBinding:
    return IdentityBinding._synthetic(
        segment_key="S07_S08",
        case_id="N1024_fixed",
        identity_repeat_id="I0",
        surface=7,
        side="after",
        input_artifact=_ref(
            "S07_S08/N1024_fixed/input/seed.ZBF",
            _INPUT_SHA,
            stage="fixed_input",
        ),
        rewrite_artifact=_ref(
            "S07_S08/N1024_fixed/identity/rewrite.ZBF",
            _REWRITE_SHA,
            stage="identity",
        ),
        effective_cfg_artifact=_ref(
            "S07_S08/N1024_fixed/identity/effective.CFG",
            "4" * 64,
            stage="identity",
        ),
        settings_artifact=_ref(
            "S07_S08/N1024_fixed/identity/settings.json",
            "5" * 64,
            stage="identity",
        ),
        native_report_artifact=_ref(
            "S07_S08/N1024_fixed/identity/report.txt",
            "8" * 64,
            stage="identity",
        ),
        entrance_settings_sha256=_ENTRANCE_SHA,
        convention_evidence_sha256=_CONVENTION_SHA,
        sample_convention=_cell_convention()
        if sample_convention is None
        else sample_convention,
        input_grid_sha256=physical_grid_sha256(grid),
        rewrite_grid_sha256=physical_grid_sha256(grid),
        nx=grid.nx,
        ny=grid.ny,
        dx_mm=grid.dx_mm,
        dy_mm=grid.dy_mm,
        wavelength_vacuum_mm=_WAVELENGTH_MM,
        refractive_index=_INDEX,
    )


def _propagation(
    binding: IdentityBinding,
    *,
    entrance_settings_digest: str | None = None,
) -> PropagationBinding:
    return PropagationBinding._synthetic(
        segment_key=binding.segment_key,
        case_id=binding.case_id,
        repeat_id="P0",
        source_input=binding.input_artifact,
        endpoint_artifact=_ref(
            "S07_S08/N1024_fixed/propagation/endpoint.ZBF",
            _ENDPOINT_SHA,
            stage="propagation",
        ),
        effective_cfg_artifact=_ref(
            "S07_S08/N1024_fixed/propagation/effective.CFG",
            "9" * 64,
            stage="propagation",
        ),
        settings_artifact=_ref(
            "S07_S08/N1024_fixed/propagation/settings.json",
            "e" * 64,
            stage="propagation",
        ),
        native_report_artifact=_ref(
            "S07_S08/N1024_fixed/propagation/report.txt",
            "f" * 64,
            stage="propagation",
        ),
        entrance_settings_sha256=(
            binding.entrance_settings_sha256
            if entrance_settings_digest is None
            else entrance_settings_digest
        ),
        convention_evidence_sha256=binding.convention_evidence_sha256,
        sample_convention=binding.sample_convention,
        wavelength_vacuum_mm=binding.wavelength_vacuum_mm,
        refractive_index=binding.refractive_index,
    )


def test_power_evidence_freezes_sample_semantics_then_recovers_one_scalar() -> None:
    summary = IdentityNativeReport.parse(
        "header\n"
        "Peak Irradiance = 0.0024906 Watts/Millimeters^2, "
        "Total Power = 1.000000 Watts\n"
        "footer\n"
    )
    assert summary.peak_irradiance_w_per_mm2.value == 0.0024906
    assert summary.peak_irradiance_w_per_mm2.last_digit_resolution == 1e-7
    assert summary.total_power_w.value == 1.0
    assert summary.total_power_w.last_digit_resolution == 1e-6
    with pytest.raises(ValueError, match="exactly one"):
        IdentityNativeReport.parse(
            "Peak Irradiance = 0.0024906 Watts/Millimeters^2, "
            "Total Power = 1.000000 Watts\n"
            "Peak Irradiance = 0.0024906 Watts/Millimeters^2, "
            "Total Power = 1.000000 Watts"
        )
    with pytest.raises(ValueError, match="exactly one"):
        IdentityNativeReport.parse(
            "Peak Irradiance = 0.0024906 Watts/mm^2, Total Power = 1 Watts"
        )

    cell = _cell_convention()
    assert cell.status == "cell_energy"
    assert not cell.authoritative
    assert cell.cell_max_closure_sigma < 1e-8
    assert cell.point_min_separation_sigma > 5.0
    measured_roundtrip = (
        _probe_with_pilot_roundtrip(
            cell.probes[0],
            position_relative_drift=0.0,
            waist_relative_drift=0.0,
        ),
        _probe_with_pilot_roundtrip(
            cell.probes[1],
            position_relative_drift=2.22344e-9,
            waist_relative_drift=2.22346e-9,
        ),
        _probe_with_pilot_roundtrip(
            cell.probes[2],
            position_relative_drift=-2.22344e-9,
            waist_relative_drift=-2.22346e-9,
        ),
    )
    assert SampleConventionResult._synthetic(measured_roundtrip).status == "cell_energy"
    excessive_roundtrip = measured_roundtrip[:2] + (
        _probe_with_pilot_roundtrip(
            cell.probes[2],
            position_relative_drift=2.0e-7,
            waist_relative_drift=2.0e-7,
        ),
    )
    with pytest.raises(ValueError, match="pilot roundtrip"):
        SampleConventionResult._synthetic(excessive_roundtrip)
    with pytest.raises(ValueError, match="Rayleigh relation"):
        _probe_with_pilot_roundtrip(
            cell.probes[1],
            position_relative_drift=2.22344e-9,
            waist_relative_drift=2.22346e-9,
            rayleigh_relative_error=2.0e-7,
        )
    with pytest.raises(TypeError, match="current-run identity capture"):
        SampleConventionProbe()
    with pytest.raises(TypeError, match="init=False"):
        replace(cell, status="point_value")
    with pytest.raises(ValueError, match="authoritative current-run"):
        classify_sample_value_convention(cell.probes)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        classify_sample_value_convention(
            cell.probes,
            policy=SampleConventionPolicy(  # type: ignore[call-arg]
                maximum_closure_sigma=30.0,
                minimum_separation_sigma=50.0,
            ),
        )
    foreign_probe = SampleConventionProbe._synthetic(
        case_id="N4096_fixed",
        repeat_id="I4096",
        nx=4096,
        ny=4096,
        dx_mm=0.0625,
        dy_mm=0.0625,
        raw_energy=1.0,
        raw_peak=0.0024906 * 0.0625**2,
        report=summary,
        run_id="another_run",
    )
    with pytest.raises(ValueError, match="one current run"):
        SampleConventionResult._synthetic(
            (_convention_probe(1024), _convention_probe(2048), foreign_probe)
        )

    point = SampleConventionResult._synthetic(
        tuple(
            _convention_probe(n, hypothesis="point_value")
            for n in (1024, 2048, 4096)
        )
    )
    assert point.status == "point_value"
    ambiguous = SampleConventionResult._synthetic(
        tuple(
            _convention_probe(n, hypothesis="neither")
            for n in (1024, 2048, 4096)
        ),
        policy=SampleConventionPolicy(
            maximum_closure_sigma=3.0,
            minimum_separation_sigma=5.0,
        ),
    )
    assert ambiguous.status == "undecided"
    with pytest.raises(ValueError, match="undecided"):
        _binding(_grid(), sample_convention=ambiguous)

    grid = _grid()
    input_values = _values(grid)
    expected = 1.23 * np.exp(0.37j)
    result = evaluate_start_identity(
        _mapped(input_values, grid, _INPUT_SHA),
        _mapped(input_values / expected, grid, _REWRITE_SHA),
        binding=_binding(grid),
    )
    assert result.passed
    assert result.phase_rms_waves is not None and result.phase_rms_waves < 2e-16
    assert result.intensity_relative_l2_percent < 2e-13
    assert result.calibration.c_entry == pytest.approx(expected, rel=2e-15)
    assert result.calibration.magnitude == pytest.approx(abs(expected))
    assert result.calibration.phase_rad == pytest.approx(np.angle(expected))
    assert not result.roi_mask.flags.writeable

    island_grid = UniformGrid2D.centered(nx=8, ny=8, dx_mm=0.1, dy_mm=0.1)
    islands = np.zeros((8, 8), dtype=np.complex128)
    islands[4, 4] = 2.0
    islands[4, 5] = 1.0
    islands[0, 0] = 1.5
    island_result = evaluate_start_identity(
        _mapped(islands, island_grid, _INPUT_SHA),
        _mapped(islands / expected, island_grid, _REWRITE_SHA),
        binding=_binding(island_grid),
    )
    assert island_result.roi_sample_count == 2
    assert island_result.roi_mask[4, 4] and island_result.roi_mask[4, 5]
    assert not island_result.roi_mask[0, 0]

    islands[0, 0] = 2.0
    with pytest.raises(ValueError, match="disconnected equal global maxima"):
        evaluate_start_identity(
            _mapped(islands, island_grid, _INPUT_SHA),
            _mapped(islands / expected, island_grid, _REWRITE_SHA),
            binding=_binding(island_grid),
        )


@pytest.mark.parametrize("defect", ("grid", "shift", "flip", "tilt", "defocus"))
def test_grid_spatial_or_reference_defects_are_not_compensated(defect: str) -> None:
    grid = _grid()
    input_values = _values(grid)
    rewritten_grid = grid
    rewritten = input_values.copy()
    if defect == "grid":
        rewritten_grid = UniformGrid2D(
            x_mm=grid.x_mm + 0.37,
            y_mm=grid.y_mm - 0.22,
        )
        rewritten = _values(rewritten_grid)
    elif defect == "shift":
        rewritten = np.roll(rewritten, 1, axis=1)
    elif defect == "flip":
        rewritten = rewritten[:, ::-1]
    else:
        xx, yy = np.meshgrid(grid.x_mm, grid.y_mm)
        phase = 0.02 * xx if defect == "tilt" else 0.025 * (xx * xx + yy * yy)
        rewritten *= np.exp(1j * phase)

    if defect == "grid":
        with pytest.raises(ValueError, match="grid|sampling"):
            evaluate_start_identity(
                _mapped(input_values, grid, _INPUT_SHA),
                _mapped(rewritten, rewritten_grid, _REWRITE_SHA),
                binding=_binding(grid),
            )
    else:
        result = evaluate_start_identity(
            _mapped(input_values, grid, _INPUT_SHA),
            _mapped(rewritten, grid, _REWRITE_SHA),
            binding=_binding(grid),
        )
        assert not result.passed
        assert result.failure_reason == "identity_residual_exceeds_policy"

    mapped = _mapped(input_values, grid, _INPUT_SHA)
    inconsistent = replace(
        mapped,
        physical=PointField2D(mapped.physical.values * np.exp(0.1j), grid),
    )
    with pytest.raises(ValueError, match="chi times"):
        evaluate_start_identity(
            inconsistent,
            _mapped(input_values, grid, _REWRITE_SHA),
            binding=_binding(grid),
        )
    with pytest.raises(ValueError, match="independently captured"):
        evaluate_start_identity(mapped, mapped, binding=_binding(grid))

    binding = _binding(grid)
    with pytest.raises(TypeError, match="from_capture"):
        replace(binding, rewrite_artifact=binding.input_artifact)


def test_exact_endpoint_binding_applies_the_entrance_calibration_once() -> None:
    grid = _grid()
    input_values = _values(grid)
    expected = 0.83 * np.exp(-0.21j)
    binding = _binding(grid)
    identity = evaluate_start_identity(
        _mapped(input_values, grid, _INPUT_SHA),
        _mapped(input_values / expected, grid, _REWRITE_SHA),
        binding=binding,
    )
    endpoint_values = (0.7 + 0.4j) * input_values / expected
    endpoint = _mapped(endpoint_values, grid, _ENDPOINT_SHA)
    propagation = _propagation(binding)
    calibrated = apply_entrance_calibration(
        endpoint,
        identity,
        propagation=propagation,
    )
    assert isinstance(calibrated, CalibratedEndpoint)
    np.testing.assert_allclose(
        calibrated.physical.values,
        (0.7 + 0.4j) * input_values,
        rtol=3e-15,
        atol=3e-15,
    )
    assert calibrated.endpoint_zbf_sha256 == _ENDPOINT_SHA
    assert calibrated.calibration_sha256 == identity.calibration.calibration_sha256

    with pytest.raises(ValueError, match="endpoint field"):
        apply_entrance_calibration(
            _mapped(endpoint_values, grid, "0" * 64),
            identity,
            propagation=propagation,
        )
    with pytest.raises(ValueError, match="propagation binding"):
        apply_entrance_calibration(
            endpoint,
            identity,
            propagation=_propagation(
                binding,
                entrance_settings_digest="0" * 64,
            ),
        )
    with pytest.raises(ValueError, match="uncalibrated MappedZbfField"):
        apply_entrance_calibration(
            calibrated,  # type: ignore[arg-type]
            identity,
            propagation=propagation,
        )

    settings = NativeSettingsReadback(
        start_surface=7,
        end_surface=7,
        nx=48,
        ny=40,
        sample_size_enum="S_48x40",
        x_width_mm=48.0 * grid.dx_mm,
        y_width_mm=40.0 * grid.dy_mm,
        wavelength_number=1,
        wavelength_vacuum_mm=_WAVELENGTH_MM,
        refractive_index=_INDEX,
        field_number=1,
        use_polarization=False,
        normalization_mode="total_power",
        normalization_value=1.0,
        input_beam_file="identity_unique.ZBF",
        output_beam_file="identity_output",
        save_output_beam=True,
        save_beam_at_all_surfaces=True,
    )
    entrance_hash = entrance_settings_sha256(settings, input_zbf_sha256=_INPUT_SHA)
    assert entrance_settings_sha256(
        replace(
            settings,
            end_surface=8,
            input_beam_file="propagation_unique.ZBF",
            output_beam_file="propagation_output",
        ),
        input_zbf_sha256=_INPUT_SHA,
    ) == entrance_hash
    assert entrance_settings_sha256(
        replace(settings, normalization_value=2.0),
        input_zbf_sha256=_INPUT_SHA,
    ) != entrance_hash
    with pytest.raises(TypeError, match="from_capture"):
        IdentityBinding()
    with pytest.raises(TypeError, match="from_capture"):
        PropagationBinding()


def test_failed_identity_status_cannot_be_forged_and_blocks_endpoint() -> None:
    grid = _grid()
    input_values = _values(grid)
    xx, _ = np.meshgrid(grid.x_mm, grid.y_mm)
    bad_rewrite = input_values * np.exp(0.03j * xx)
    binding = _binding(grid)
    failed = evaluate_start_identity(
        _mapped(input_values, grid, _INPUT_SHA),
        _mapped(bad_rewrite, grid, _REWRITE_SHA),
        binding=binding,
    )
    assert not failed.passed
    with pytest.raises(TypeError, match="init=False"):
        replace(failed, passed=True, failure_reason=None)
    with pytest.raises(TypeError, match="identity evaluation"):
        replace(
            failed,
            phase_rms_waves=0.0,
            intensity_relative_l2_percent=0.0,
        )
    with pytest.raises(ValueError, match="failed identity"):
        apply_entrance_calibration(
            _mapped(input_values, grid, _ENDPOINT_SHA),
            failed,
            propagation=_propagation(binding),
        )

    strict = IdentityPolicy(
        roi_threshold=1e-6,
        maximum_phase_rms_waves=1e-18,
        maximum_intensity_relative_l2_percent=1e-14,
        grid_rtol=1e-10,
        grid_atol_mm=1e-12,
    )
    noisy = input_values * np.exp(1e-12j * xx)
    assert not evaluate_start_identity(
        _mapped(input_values, grid, _INPUT_SHA),
        _mapped(noisy, grid, _REWRITE_SHA),
        binding=binding,
        policy=strict,
    ).passed

    good = evaluate_start_identity(
        _mapped(input_values, grid, _INPUT_SHA),
        _mapped(input_values, grid, _REWRITE_SHA),
        binding=binding,
    )
    with pytest.raises(TypeError, match="identity evaluation"):
        replace(good.calibration, c_entry=99.0 + 3.0j)
