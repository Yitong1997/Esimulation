from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from sandbox.free_space_algorithm_identification.artifacts import (
    ArtifactRef,
    copy_file_once,
    create_run_layout,
)
from sandbox.free_space_algorithm_identification.zbf_binary import (
    HEADER_BYTES,
    RawZbfHeader,
)
from sandbox.free_space_algorithm_identification.zos_runner import (
    SegmentPopRequest,
    capture_segment_run,
)


def test_live_1024_fixed_input_identity_capture() -> None:
    if os.environ.get("BTS_RUN_ZEMAX_BENCHMARK") != "1":
        pytest.skip("set BTS_RUN_ZEMAX_BENCHMARK=1 for the explicit live smoke")
    required = {
        "baseline": os.environ.get("BTS_FREE_SPACE_BASELINE_DIR"),
        "input": os.environ.get("BTS_FREE_SPACE_INPUT_ZBF"),
        "run_root": os.environ.get("BTS_FREE_SPACE_RUN_ROOT"),
    }
    if any(not value for value in required.values()):
        pytest.fail(
            "enabled live smoke requires explicit baseline, input, and run-root paths"
        )

    baseline = Path(str(required["baseline"])).resolve(strict=True)
    input_zbf = Path(str(required["input"])).resolve(strict=True)
    run_root = Path(str(required["run_root"])).resolve(strict=True)
    model = baseline / "biconic_focus_test.zmx"
    cfg = baseline / "biconic_focus_test.CFG"
    if not model.is_file() or not cfg.is_file() or not input_zbf.is_file():
        pytest.fail("explicit live model, CFG, and input ZBF must be regular files")

    run_id = f"live1024_{uuid.uuid4().hex}"
    layout = create_run_layout(
        run_root,
        run_id,
        model_source=model,
        cfg_source=cfg,
        manifest_payload={"planned_stage_graph": ["identity"]},
        case_matrix={
            "S07_S08": ("live1024",),
            "S12_S13": ("live1024",),
            "S13_S14": ("live1024",),
        },
    )
    relative = "S07_S08/live1024/input/fixed_1024.ZBF"
    copy_file_once(layout, input_zbf, relative)
    reference = ArtifactRef.from_file(
        layout,
        relative,
        producer_stage="fixed_input",
        producer_case="live1024",
    )
    with input_zbf.open("rb") as stream:
        header = RawZbfHeader.from_bytes(stream.read(HEADER_BYTES))
    if header.nx != 1024 or header.ny != 1024:
        pytest.fail("live smoke accepts only an explicit 1024x1024 fixed input")

    captured = capture_segment_run(
        layout,
        SegmentPopRequest(
            segment_key="S07_S08",
            case_id="live1024",
            repeat_id="R0",
            start_surface=7,
            end_surface=7,
            input_artifact=reference,
            input_producer_stage="fixed_input",
            input_producer_case="live1024",
            nx=header.nx,
            ny=header.ny,
            x_width_mm=header.nx * header.dx,
            y_width_mm=header.ny * header.dy,
            wavelength_number=1,
            wavelength_vacuum_mm=header.wavelength_vacuum_mm,
            refractive_index=header.refractive_index,
            field_number=1,
            use_polarization=bool(header.is_polarized),
            normalization_mode="total_power",
            normalization_value=1.0,
            use_disk_storage=False,
            data_grid_index=0,
        ),
    )

    assert captured.stage == "identity"
    assert len(captured.output_zbfs) == 2
    assert captured.raw_grid.nx == captured.raw_grid.ny == 1024
    assert captured.cleanup_errors == ()
