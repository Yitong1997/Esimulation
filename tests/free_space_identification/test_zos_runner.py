from __future__ import annotations

import json
import struct
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sandbox.free_space_algorithm_identification.zos_runner as runner_module
from sandbox.free_space_algorithm_identification.artifacts import (
    ArtifactRef,
    copy_file_once,
    create_run_layout,
    hash_artifact,
    verify_artifact_ref,
)
from sandbox.free_space_algorithm_identification.biconic_case import (
    BICONIC_SEGMENTS,
)
from sandbox.free_space_algorithm_identification.zos_runner import (
    CapturedPopRun,
    RunnerCleanupError,
    SegmentPopRequest,
    capture_segment_run,
    expected_output_names,
)


class _PrimaryBoom(RuntimeError):
    pass


def _zbf_header_bytes(
    *,
    nx: int = 4,
    ny: int = 4,
    dx_mm: float = 0.1,
    dy_mm: float = 0.2,
    is_polarized: int = 0,
) -> bytes:
    ints = (1, nx, ny, is_polarized, 0, 0, 0, 0, 0)
    doubles = (
        dx_mm,
        dy_mm,
        -2.0,
        1.0,
        0.05,
        -2.0,
        1.0,
        0.05,
        0.0006328,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    return struct.pack("<9i20d", *ints, *doubles)


def _layout_and_input(
    tmp_path: Path,
    *,
    segment_key: str = "S07_S08",
    case_id: str = "caseA",
    nx: int = 4,
    ny: int = 4,
    dx_mm: float = 0.1,
    dy_mm: float = 0.2,
) -> tuple[object, ArtifactRef]:
    model = tmp_path / "source.zmx"
    cfg = tmp_path / "source.CFG"
    seed = tmp_path / "seed.ZBF"
    model.write_bytes(b"fixed model")
    cfg.write_bytes(b"fixed cfg")
    seed.write_bytes(
        _zbf_header_bytes(
            nx=nx,
            ny=ny,
            dx_mm=dx_mm,
            dy_mm=dy_mm,
        )
    )
    layout = create_run_layout(
        tmp_path / "runs",
        "runA",
        model_source=model,
        cfg_source=cfg,
        manifest_payload={"planned_stage_graph": ["identity", "propagation"]},
        case_matrix={
            "S07_S08": (case_id,),
            "S12_S13": (case_id,),
            "S13_S14": (case_id,),
        },
    )
    relative = f"{segment_key}/{case_id}/input/seed.ZBF"
    copy_file_once(layout, seed, relative)
    reference = ArtifactRef.from_file(
        layout,
        relative,
        producer_stage="fixed_input",
        producer_case=case_id,
    )
    return layout, reference


def _request(
    reference: ArtifactRef,
    *,
    segment_key: str = "S07_S08",
    start_surface: int = 7,
    end_surface: int = 8,
    case_id: str = "caseA",
) -> SegmentPopRequest:
    return SegmentPopRequest(
        segment_key=segment_key,
        case_id=case_id,
        repeat_id="R0",
        start_surface=start_surface,
        end_surface=end_surface,
        input_artifact=reference,
        input_producer_stage="fixed_input",
        input_producer_case=case_id,
        nx=4,
        ny=4,
        x_width_mm=0.4,
        y_width_mm=0.8,
        wavelength_number=1,
        wavelength_vacuum_mm=0.0006328,
        refractive_index=1.0,
        field_number=1,
        use_polarization=False,
        normalization_mode="total_power",
        normalization_value=1.0,
        use_disk_storage=False,
        data_grid_index=0,
    )


class _SurfaceNumber:
    def __init__(self, value: int, environment: "_FakeEnvironment") -> None:
        self.value = value
        self.environment = environment

    def GetSurfaceNumber(self) -> int:
        self.environment.events.append("readback")
        return self.value


class _FakeGrid:
    def __init__(
        self,
        kwargs: dict[str, object],
        *,
        half_step: bool,
        z_mismatch: bool,
    ) -> None:
        self.Nx = int(kwargs["x_sampling"])
        self.Ny = int(kwargs["y_sampling"])
        self.Dx = float(kwargs["x_width"]) / self.Nx
        self.Dy = float(kwargs["y_width"]) / self.Ny
        offset = 0.5 * self.Dx if half_step else 0.0
        self.MinX = -(self.Nx // 2) * self.Dx + offset
        self.MinY = -(self.Ny // 2) * self.Dy
        ix, iy = np.indices((self.Nx, self.Ny))
        self.Values = 100.0 * ix + 3.0 * iy + 0.25
        self.z_mismatch = z_mismatch

    def X(self, index: int) -> float:
        return self.MinX + index * self.Dx

    def Y(self, index: int) -> float:
        return self.MinY + index * self.Dy

    def Z(self, ix: int, iy: int) -> float:
        offset = 1.0 if self.z_mismatch else 0.0
        return float(self.Values[ix, iy] + offset)


class _FakeSettings:
    def __init__(
        self,
        kwargs: dict[str, object],
        environment: "_FakeEnvironment",
    ) -> None:
        self.environment = environment
        self.StartSurface = _SurfaceNumber(int(kwargs["start_surface"]), environment)
        self.EndSurface = _SurfaceNumber(int(kwargs["end_surface"]), environment)
        self.SurfaceToBeam = kwargs["surface_to_beam"]
        self.UsePolarization = kwargs["use_polarization"]
        self.SeparateXY = kwargs["separate_xy"]
        self.UseDiskStorage = kwargs["use_disk_storage"]
        self.BeamType = kwargs["beam_type"]
        self.BeamTypeFilename = kwargs["beam_file"]
        size = int(kwargs["x_sampling"])
        self.XSampling = f"S_{size}x{size}"
        self.YSampling = f"S_{size}x{size}"
        self.XWidth = kwargs["x_width"]
        self.YWidth = kwargs["y_width"]
        self.UseTotalPower = kwargs["use_total_power"]
        self.TotalPower = (
            2.0
            if environment.fail_stage == "normalization_readback"
            else kwargs["total_power"]
        )
        self.UsePeakIrradiance = kwargs["use_peak_irradiance"]
        self.PeakIrradiance = kwargs["peak_irradiance"]
        self.Project = kwargs["project"]
        self.SaveOutputBeam = kwargs["save_output_beam"]
        self.OutputBeamFile = kwargs["output_beam_file"]
        self.SaveBeamAtAllSurfaces = kwargs["save_beam_at_all_surfaces"]
        self.Wavelength = SimpleNamespace(
            GetWavelengthNumber=lambda: int(kwargs["wavelength"])
        )
        self.Field = SimpleNamespace(GetFieldNumber=lambda: int(kwargs["field"]))

    def SaveTo(self, path: str) -> None:
        self.environment.events.append("save_cfg")
        if self.environment.fail_stage == "save_cfg":
            raise _PrimaryBoom("save_cfg")
        Path(path).write_bytes(b"effective cfg")


class _FakeResults:
    def __init__(
        self,
        kwargs: dict[str, object],
        environment: "_FakeEnvironment",
    ) -> None:
        self.environment = environment
        self._grids = (
            _FakeGrid(
                kwargs,
                half_step=environment.fail_stage == "half_step",
                z_mismatch=environment.fail_stage == "z_mismatch",
            ),
        )

    @property
    def DataGrids(self) -> tuple[_FakeGrid, ...]:
        self.environment.events.append("raw_grid")
        return self._grids

    def GetTextFile(self, path: str) -> None:
        self.environment.events.append("report")
        if self.environment.fail_stage == "report":
            raise _PrimaryBoom("report")
        Path(path).write_text("native POP report\n", encoding="utf-8")


class _FakeAnalysis:
    def __init__(
        self,
        kwargs: dict[str, object],
        environment: "_FakeEnvironment",
    ) -> None:
        self.environment = environment
        self.Settings = _FakeSettings(kwargs, environment)
        self.Results = _FakeResults(kwargs, environment)

    @property
    def messages(self) -> tuple[object, ...]:
        self.environment.events.append("analysis_messages")
        return (SimpleNamespace(ErrorCode="A1", Message="analysis message"),)

    def Close(self) -> None:
        self.environment.events.append("close")
        if self.environment.close_fails:
            raise RuntimeError("close cleanup")


class _FakeResult:
    def __init__(self, environment: "_FakeEnvironment") -> None:
        self.environment = environment

    @property
    def messages(self) -> tuple[object, ...]:
        self.environment.events.append("result_messages")
        return (SimpleNamespace(ErrorCode="R1", Message="result message"),)


class _FakeWrapper:
    def __init__(
        self,
        kwargs: dict[str, object],
        environment: "_FakeEnvironment",
    ) -> None:
        self.kwargs = kwargs
        self.environment = environment
        self.analysis = _FakeAnalysis(kwargs, environment)

    def run(self, oss: object, *, oncomplete: str) -> _FakeResult:
        del oss
        self.environment.events.append(f"run:{oncomplete}")
        if self.environment.fail_stage == "run":
            raise _PrimaryBoom("run")
        prefix = str(self.kwargs["output_beam_file"])
        names = expected_output_names(
            prefix,
            int(self.kwargs["start_surface"]),
            int(self.kwargs["end_surface"]),
        )
        if self.environment.output_mode == "missing":
            names = names[:-1]
        for name in names:
            (self.environment.pop_dir / name).write_bytes(
                _zbf_header_bytes(
                    nx=int(self.kwargs["x_sampling"]),
                    ny=int(self.kwargs["y_sampling"]),
                    dx_mm=float(self.kwargs["x_width"])
                    / int(self.kwargs["x_sampling"]),
                    dy_mm=float(self.kwargs["y_width"])
                    / int(self.kwargs["y_sampling"]),
                )
            )
        if self.environment.output_mode == "extra":
            (self.environment.pop_dir / f"{prefix}_9999.ZBF").write_bytes(
                _zbf_header_bytes()
            )
        if self.environment.simulate_high_level_unpack:
            self.get_data_grid(cell_origin="bottom_left")
        return _FakeResult(self.environment)

    def get_data_grid(self, *, cell_origin: str) -> None:
        assert cell_origin == "bottom_left"
        self.environment.events.append("high_level_grid_unpack")
        raise AssertionError("high-level DataFrame grid path is forbidden")


class _FakeSystem:
    def __init__(self, environment: "_FakeEnvironment") -> None:
        self.environment = environment
        self.TheApplication = SimpleNamespace(POPDir=str(environment.pop_dir))
        wavelength_um = (
            0.5 if environment.fail_stage == "model_wavelength" else 0.6328
        )

        def get_wavelength(number: int) -> object:
            assert number == 1
            environment.events.append("model_wavelength")
            return SimpleNamespace(Wavelength=wavelength_um)

        self.SystemData = SimpleNamespace(
            Wavelengths=SimpleNamespace(GetWavelength=get_wavelength)
        )

    def load(self, path: str) -> None:
        assert Path(path).is_file()
        self.environment.events.append("load")


class _FakeZos:
    def __init__(self, environment: "_FakeEnvironment") -> None:
        self.environment = environment
        self.system = _FakeSystem(environment)

    def connect(self, *, mode: str) -> _FakeSystem:
        assert mode == "standalone"
        self.environment.events.append("connect")
        return self.system

    def retrieve_logs(self) -> str:
        self.environment.events.append("application_logs")
        return "application log"

    def disconnect(self) -> None:
        self.environment.events.append("disconnect")
        if self.environment.disconnect_fails:
            raise RuntimeError("disconnect cleanup")


class _FakeEnvironment:
    def __init__(
        self,
        pop_dir: Path,
        *,
        output_mode: str = "ok",
        fail_stage: str | None = None,
        close_fails: bool = False,
        disconnect_fails: bool = False,
        simulate_high_level_unpack: bool = False,
    ) -> None:
        self.pop_dir = pop_dir
        self.pop_dir.mkdir()
        self.output_mode = output_mode
        self.fail_stage = fail_stage
        self.close_fails = close_fails
        self.disconnect_fails = disconnect_fails
        self.simulate_high_level_unpack = simulate_high_level_unpack
        self.events: list[str] = []
        self.wrapper_kwargs: dict[str, object] | None = None

    def zos_factory(self) -> _FakeZos:
        return _FakeZos(self)

    def pop_factory(self, **kwargs: object) -> _FakeWrapper:
        self.events.append("wrapper_init")
        self.wrapper_kwargs = dict(kwargs)
        return _FakeWrapper(dict(kwargs), self)


@pytest.mark.parametrize(
    ("segment_key", "start_surface", "end_surface", "stage"),
    (
        ("S07_S08", 7, 7, "identity"),
        ("S07_S08", 7, 8, "propagation"),
        ("S12_S13", 12, 13, "propagation"),
        ("S13_S14", 13, 14, "propagation"),
    ),
)
def test_request_supports_identity_and_three_fixed_input_segments(
    tmp_path: Path,
    segment_key: str,
    start_surface: int,
    end_surface: int,
    stage: str,
) -> None:
    layout, reference = _layout_and_input(tmp_path, segment_key=segment_key)
    request = _request(
        reference,
        segment_key=segment_key,
        start_surface=start_surface,
        end_surface=end_surface,
    )

    assert request.stage == stage
    assert expected_output_names("P", start_surface, end_surface) == (
        ("P.ZBF", f"P_{start_surface:04d}.ZBF")
        if start_surface == end_surface
        else (
            "P.ZBF",
            f"P_{start_surface:04d}.ZBF",
            f"P_{end_surface:04d}.ZBF",
        )
    )
    if start_surface != end_surface:
        segment = next(item for item in BICONIC_SEGMENTS if item.key == segment_key)
        assert (request.start_surface, request.end_surface) == (
            segment.start_surface,
            segment.end_surface,
        )
    segment = next(item for item in BICONIC_SEGMENTS if item.key == segment_key)
    other_end = (
        segment.end_surface if request.stage == "identity" else segment.start_surface
    )
    other_stage = replace(request, end_surface=other_end)
    assert other_stage.stage != request.stage
    assert other_stage.output_prefix(layout.run_id) != request.output_prefix(
        layout.run_id
    )
    assert other_stage.staged_input_name(layout.run_id) != request.staged_input_name(
        layout.run_id
    )

    bad = replace(
        request,
        nx=8,
        ny=8,
        x_width_mm=0.8,
        y_width_mm=1.6,
    )
    connected = False

    def forbidden_connection() -> object:
        nonlocal connected
        connected = True
        raise AssertionError("input header mismatch must fail before connection")

    with pytest.raises(ValueError, match="input ZBF header"):
        capture_segment_run(
            layout,
            bad,
            zos_factory=forbidden_connection,
            pop_factory=lambda **_: None,
        )
    assert not connected

    with pytest.raises(ValueError, match="fixed biconic"):
        replace(request, end_surface=end_surface + 2)

    environment = _FakeEnvironment(tmp_path / "POP_stale_input")
    source_path = verify_artifact_ref(
        layout,
        reference,
        expected_producer_stage="fixed_input",
        expected_producer_case="caseA",
    )
    stale_input = environment.pop_dir / request.staged_input_name(layout.run_id)
    stale_input.write_bytes(source_path.read_bytes())
    stale_hash = hash_artifact(stale_input)
    with pytest.raises(ValueError, match="stale staged POP input"):
        capture_segment_run(
            layout,
            request,
            zos_factory=environment.zos_factory,
            pop_factory=environment.pop_factory,
        )
    assert stale_input.is_file()
    assert hash_artifact(stale_input) == stale_hash
    assert "disconnect" in environment.events


def test_capture_uses_sustain_exact_order_and_one_raw_xy_transpose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, reference = _layout_and_input(tmp_path)
    request = _request(reference)
    environment = _FakeEnvironment(
        tmp_path / "POP",
        simulate_high_level_unpack=True,
    )
    real_copy = runner_module.copy_file_once
    real_unlink = Path.unlink

    def observed_copy(*args: object, **kwargs: object) -> Path:
        source = Path(args[1])
        if source.parent == environment.pop_dir:
            environment.events.append("copy_zbf")
        return real_copy(*args, **kwargs)

    monkeypatch.setattr(runner_module, "copy_file_once", observed_copy)

    def observed_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path.name == request.staged_input_name(layout.run_id):
            environment.events.append("input_cleanup")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", observed_unlink)
    captured = capture_segment_run(
        layout,
        request,
        zos_factory=environment.zos_factory,
        pop_factory=environment.pop_factory,
    )

    assert isinstance(captured, CapturedPopRun)
    kwargs = environment.wrapper_kwargs
    assert kwargs is not None
    assert kwargs == {
        "wavelength": 1,
        "field": 1,
        "start_surface": 7,
        "end_surface": 8,
        "surface_to_beam": 0.0,
        "use_polarization": False,
        "separate_xy": False,
        "use_disk_storage": False,
        "beam_type": "File",
        "beam_file": request.staged_input_name(layout.run_id),
        "x_sampling": 4,
        "y_sampling": 4,
        "x_width": 0.4,
        "y_width": 0.8,
        "use_total_power": True,
        "total_power": 1.0,
        "use_peak_irradiance": False,
        "peak_irradiance": 1.0,
        "show_as": "FalseColor",
        "data_type": "Irradiance",
        "project": "AlongBeam",
        "save_output_beam": True,
        "output_beam_file": request.output_prefix(layout.run_id),
        "save_beam_at_all_surfaces": True,
        "auto_calculate_beam_sampling": False,
    }
    first = {name: environment.events.index(name) for name in set(environment.events)}
    assert first["run:Sustain"] < first["save_cfg"]
    assert first["save_cfg"] < first["readback"] < first["report"]
    assert first["save_cfg"] < first["model_wavelength"] < first["report"]
    assert first["report"] < first["copy_zbf"] < first["raw_grid"]
    assert first["raw_grid"] < first["analysis_messages"]
    assert first["application_logs"] < first["close"] < first["disconnect"]
    assert first["disconnect"] < first["input_cleanup"]
    assert "high_level_grid_unpack" not in environment.events

    expected_names = expected_output_names(
        request.output_prefix(layout.run_id), 7, 8
    )
    assert tuple(output.name for output in captured.output_zbfs) == expected_names
    source_path = verify_artifact_ref(
        layout,
        reference,
        expected_producer_stage="fixed_input",
        expected_producer_case="caseA",
    )
    assert captured.staged_input_hash == hash_artifact(source_path)
    for output in captured.output_zbfs:
        path = verify_artifact_ref(
            layout,
            output.artifact,
            expected_producer_stage="propagation",
            expected_producer_case="caseA",
        )
        assert hash_artifact(path).sha256 == output.artifact.sha256
        assert output.header.nx == output.header.ny == 4

    api_xy = 100.0 * np.indices((4, 4))[0] + 3.0 * np.indices((4, 4))[1] + 0.25
    np.testing.assert_array_equal(captured.raw_grid.values_api_xy, api_xy)
    np.testing.assert_array_equal(captured.raw_grid.values_package_yx, api_xy.T)
    assert not captured.raw_grid.values_api_xy.flags.writeable
    assert not captured.raw_grid.values_package_yx.flags.writeable
    assert captured.raw_grid.x_checkpoints[1] == (2, 0.0)
    assert captured.raw_grid.y_checkpoints[1] == (2, 0.0)
    assert captured.raw_grid.z_checkpoints == captured.raw_grid.values_checkpoints
    assert captured.cleanup_errors == ()
    assert not (environment.pop_dir / request.staged_input_name(layout.run_id)).exists()

    forbidden_types = {
        _FakeZos,
        _FakeSystem,
        _FakeWrapper,
        _FakeAnalysis,
        _FakeResults,
        _FakeGrid,
        _FakeResult,
    }
    assert all(
        type(getattr(captured, field.name)) not in forbidden_types
        for field in fields(captured)
    )
    settings_path = verify_artifact_ref(
        layout,
        captured.settings_artifact,
        expected_producer_stage="propagation",
        expected_producer_case="caseA",
    )
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert settings["project"] == "AlongBeam"
    assert settings["beam_file"] == request.staged_input_name(layout.run_id)
    assert settings["wavelength_vacuum_mm"] == 0.0006328
    assert settings["refractive_index"] == 1.0
    assert settings["normalization_mode"] == "total_power"
    assert settings["normalization_value"] == 1.0
    assert settings["wavelength_vacuum_mm_source"] == (
        "loaded_model_wavelength_table_and_verified_input_zbf_header"
    )
    assert settings["refractive_index_source"] == "verified_input_zbf_header"
    assert settings["physical_contract_input_zbf_sha256"] == (
        captured.staged_input_hash.sha256
    )


@pytest.mark.parametrize("mode", ("stale", "missing", "extra"))
def test_output_collection_rejects_stale_missing_and_extra_anchored_names(
    tmp_path: Path,
    mode: str,
) -> None:
    layout, reference = _layout_and_input(tmp_path)
    request = _request(reference)
    output_mode = "ok" if mode == "stale" else mode
    environment = _FakeEnvironment(tmp_path / "POP", output_mode=output_mode)
    if mode == "stale":
        prefix = request.output_prefix(layout.run_id)
        (environment.pop_dir / f"{prefix}_0007.ZBF").write_bytes(
            _zbf_header_bytes()
        )

    with pytest.raises(ValueError, match=mode):
        capture_segment_run(
            layout,
            request,
            zos_factory=environment.zos_factory,
            pop_factory=environment.pop_factory,
        )

    assert "disconnect" in environment.events
    if mode == "stale":
        assert "run:Sustain" not in environment.events
    if mode != "stale":
        assert environment.events.index("close") < environment.events.index(
            "disconnect"
        )
    assert not (environment.pop_dir / request.staged_input_name(layout.run_id)).exists()


@pytest.mark.parametrize(
    "scenario",
    (
        "run",
        "save_cfg",
        "report",
        "half_step",
        "z_mismatch",
        "normalization_readback",
        "model_wavelength",
        "close_only",
        "primary_plus_cleanup",
        "unlink_only",
    ),
)
def test_failures_close_disconnect_preserve_primary_and_separate_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    layout, reference = _layout_and_input(tmp_path)
    request = _request(reference)
    failure_stages = {
        "run",
        "save_cfg",
        "report",
        "half_step",
        "z_mismatch",
        "normalization_readback",
        "model_wavelength",
    }
    fail_stage = scenario if scenario in failure_stages else None
    if scenario == "primary_plus_cleanup":
        fail_stage = "report"
    environment = _FakeEnvironment(
        tmp_path / "POP",
        fail_stage=fail_stage,
        close_fails=scenario in {"close_only", "primary_plus_cleanup"},
        disconnect_fails=scenario == "primary_plus_cleanup",
    )
    original_unlink = Path.unlink
    if scenario in {"primary_plus_cleanup", "unlink_only"}:

        def selective_unlink(path: Path, *args: object, **kwargs: object) -> None:
            if path.name == request.staged_input_name(layout.run_id):
                raise PermissionError("staged input cleanup")
            original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", selective_unlink)

    if scenario == "unlink_only":
        captured = capture_segment_run(
            layout,
            request,
            zos_factory=environment.zos_factory,
            pop_factory=environment.pop_factory,
        )
        assert captured.cleanup_errors == (
            "staged_input: PermissionError: staged input cleanup",
        )
    elif scenario == "close_only":
        with pytest.raises(RunnerCleanupError, match="close") as caught:
            capture_segment_run(
                layout,
                request,
                zos_factory=environment.zos_factory,
                pop_factory=environment.pop_factory,
            )
        assert caught.value.errors == ("close: RuntimeError: close cleanup",)
    else:
        expected = (
            ValueError
            if scenario in {
                "half_step",
                "z_mismatch",
                "normalization_readback",
                "model_wavelength",
            }
            else _PrimaryBoom
        )
        with pytest.raises(expected) as caught:
            capture_segment_run(
                layout,
                request,
                zos_factory=environment.zos_factory,
                pop_factory=environment.pop_factory,
            )
        if scenario == "primary_plus_cleanup":
            assert str(caught.value) == "report"
            notes = tuple(getattr(caught.value, "__notes__", ()))
            assert any("close cleanup" in note for note in notes)
            assert any("disconnect cleanup" in note for note in notes)
            assert any("staged input cleanup" in note for note in notes)
            assert caught.value.runner_cleanup_errors == (
                "close: RuntimeError: close cleanup",
                "disconnect: RuntimeError: disconnect cleanup",
                "staged_input: PermissionError: staged input cleanup",
            )

    assert "close" in environment.events
    assert "disconnect" in environment.events
    assert environment.events.index("close") < environment.events.index("disconnect")
