from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from sandbox.free_space_algorithm_identification.artifacts import (
    ArtifactRef,
    copy_file_once,
    create_run_layout,
    hash_artifact,
    verify_artifact_ref,
    verify_hash_manifest,
    write_hash_manifest,
    write_json_once,
    write_stage_receipt,
)


def _create_layout(tmp_path: Path, run_id: str = "run_001"):
    source_model = tmp_path / "source_system.zmx"
    source_cfg = tmp_path / "source_native.CFG"
    source_model.write_bytes(b"ZMX\x00model\r\n")
    source_cfg.write_bytes(b"CFG\x00settings\r\n")
    return create_run_layout(
        tmp_path / "runs",
        run_id,
        model_source=source_model,
        cfg_source=source_cfg,
        manifest_payload={
            "planned_stage_graph": ["identity", "propagation", "comparison"],
            "physical_conventions": {
                "phasor": "exp(-i omega t)",
                "grid_center": "sample_at_zero",
                "reflection": "real_coordinate_only",
            },
        },
        case_matrix={
            "S07_S08": ("native", "R2"),
            "S12_S13": ("ZI0", "ZO2"),
            "S13_S14": ("native", "combined"),
        },
    )


def test_create_run_layout_is_exclusive_and_captures_init_metadata(
    tmp_path: Path,
) -> None:
    layout = _create_layout(tmp_path)
    assert layout.run_dir == (tmp_path / "runs" / "run_001").resolve()
    assert layout.manifest_path.is_file()
    assert not layout.provenance_path.exists()
    assert not layout.final_report_path.exists()
    assert layout.model_path.read_bytes() == b"ZMX\x00model\r\n"
    assert layout.cfg_path.read_bytes() == b"CFG\x00settings\r\n"
    for segment, cases in {
        "S07_S08": ("native", "R2"),
        "S12_S13": ("ZI0", "ZO2"),
        "S13_S14": ("native", "combined"),
    }.items():
        for case in cases:
            for stage_dir in ("input", "identity", "propagation"):
                assert (layout.run_dir / segment / case / stage_dir).is_dir()
    for directory in (
        layout.receipts_dir,
        layout.continuous_dir,
        layout.baselines_dir,
        layout.candidates_dir,
        layout.comparisons_dir,
    ):
        assert directory.is_dir()

    manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_version"] == 1
    assert manifest["run_id"] == "run_001"
    assert manifest["planned_stage_graph"] == [
        "identity",
        "propagation",
        "comparison",
    ]
    assert manifest["case_matrix"]["S12_S13"] == ["ZI0", "ZO2"]
    assert manifest["physical_conventions"]["phasor"] == "exp(-i omega t)"

    model_hash_lines = layout.model_hash_manifest_path.read_text(
        encoding="ascii"
    ).splitlines()
    assert [line.split("  ", 1)[1] for line in model_hash_lines] == [
        "model/source_native.CFG",
        "model/system.zmx",
    ]
    assert hash_artifact(layout.model_path).sha256 == model_hash_lines[1][:64]
    assert hash_artifact(layout.cfg_path).sha256 == model_hash_lines[0][:64]

    with pytest.raises(ValueError, match="provenance"):
        write_json_once(layout, "provenance.json", {"run_id": layout.run_id})
    assert not layout.provenance_path.exists()

    model_digest = hash_artifact(layout.model_path)
    cfg_digest = hash_artifact(layout.cfg_path)
    provenance = {
        "run_id": layout.run_id,
        "versions": {
            "opticstudio": "2024 R1",
            "zos_api": "23.12.05",
            "zospy": "2.1.5",
            "python": "3.13.11",
            "numpy": "2.3.5",
            "scipy": "1.x",
        },
        "git": {"commit": "a" * 40, "dirty_paths": []},
        "host": {
            "timezone": "Asia/Shanghai",
            "cpu": "test-cpu",
            "physical_memory_bytes": 32 * 1024**3,
        },
        "captured_utc": "2026-07-11T00:00:00Z",
        "artifact_hashes": {
            "model_sha256": model_digest.sha256,
            "cfg_sha256": cfg_digest.sha256,
        },
        "conventions": {
            **manifest["physical_conventions"],
            "axis_order": "array_y_x",
            "polarization": "native_control",
            "power": "native_control",
        },
        "pop_sample_enums": [1024, 2048, 4096],
    }
    write_json_once(layout, "provenance.json", provenance)
    assert json.loads(layout.provenance_path.read_text(encoding="utf-8")) == provenance

    with pytest.raises(FileExistsError):
        _create_layout(tmp_path)


def test_once_writers_reject_overwrite_and_windows_path_escape(
    tmp_path: Path,
) -> None:
    layout = _create_layout(tmp_path)
    output = write_json_once(
        layout, "comparisons/metrics.json", {"status": "predeclared"}
    )
    assert output == layout.comparisons_dir / "metrics.json"
    with pytest.raises(FileExistsError):
        write_json_once(layout, "comparisons/metrics.json", {"status": "changed"})

    source = tmp_path / "input.ZBF"
    source.write_bytes(b"ZBF\x00payload\xff")
    copied = copy_file_once(layout, source, "continuous/input.ZBF")
    assert copied.read_bytes() == source.read_bytes()
    digest = hash_artifact(copied)
    assert digest.byte_count == len(source.read_bytes())
    assert len(digest.sha256) == 64
    with pytest.raises(FileExistsError):
        copy_file_once(layout, source, "continuous/input.ZBF")

    for escaped in (
        "../escape.json",
        r"..\escape.json",
        "/absolute.json",
        "C:/absolute.json",
        r"C:\absolute.json",
    ):
        with pytest.raises(ValueError, match="run-relative"):
            write_json_once(layout, escaped, {})
        with pytest.raises(ValueError, match="run-relative"):
            copy_file_once(layout, source, escaped)
    assert not (tmp_path / "runs" / "escape.json").exists()


def test_artifact_refs_bind_current_run_and_receipts_are_append_only(
    tmp_path: Path,
) -> None:
    layout = _create_layout(tmp_path, "run_A")
    other_layout = _create_layout(tmp_path, "run_B")
    source = tmp_path / "restart.ZBF"
    source.write_bytes(b"restart-physical-field")
    copy_file_once(layout, source, "continuous/restart.ZBF")
    copy_file_once(other_layout, source, "continuous/restart.ZBF")

    reference = ArtifactRef.from_file(
        layout,
        "continuous/restart.ZBF",
        producer_stage="fresh_native_continuous",
        producer_case="S12_S13:ZO2",
    )
    assert reference.run_id == "run_A"
    assert reference.producer_stage == "fresh_native_continuous"
    assert reference.producer_case == "S12_S13:ZO2"
    assert reference.relative_path == "continuous/restart.ZBF"
    assert reference.byte_count == len(source.read_bytes())
    assert len(reference.sha256) == 64
    assert verify_artifact_ref(
        layout,
        reference,
        expected_producer_stage="fresh_native_continuous",
        expected_producer_case="S12_S13:ZO2",
    ) == layout.continuous_dir / "restart.ZBF"

    with pytest.raises(ValueError, match="producer stage"):
        verify_artifact_ref(
            layout,
            reference,
            expected_producer_stage="identity",
            expected_producer_case="S12_S13:ZO2",
        )
    with pytest.raises(ValueError, match="producer case"):
        verify_artifact_ref(
            layout,
            reference,
            expected_producer_stage="fresh_native_continuous",
            expected_producer_case="S13_S14:input_R4",
        )
    with pytest.raises(ValueError, match="current run"):
        verify_artifact_ref(
            other_layout,
            reference,
            expected_producer_stage="fresh_native_continuous",
            expected_producer_case="S12_S13:ZO2",
        )
    for bad_reference in (
        replace(reference, byte_count=reference.byte_count + 1),
        replace(reference, sha256="f" * 64),
    ):
        with pytest.raises(ValueError, match="artifact"):
            verify_artifact_ref(
                layout,
                bad_reference,
                expected_producer_stage="fresh_native_continuous",
                expected_producer_case="S12_S13:ZO2",
            )
    with pytest.raises(ValueError, match="run-relative"):
        replace(reference, relative_path=r"..\other_run\restart.ZBF")

    identity_source = tmp_path / "identity.ZBF"
    identity_source.write_bytes(b"identity-output")
    copy_file_once(layout, identity_source, "continuous/identity.ZBF")
    identity_output = ArtifactRef.from_file(
        layout,
        "continuous/identity.ZBF",
        producer_stage="identity",
        producer_case="S12_S13:ZO2",
    )
    with pytest.raises(ValueError, match="output producer"):
        write_stage_receipt(
            layout,
            sequence=1,
            stage="identity",
            producer_case="S12_S13:ZO2",
            inputs=(reference,),
            outputs=(reference,),
            gate_values={"phase_rms_waves": 2e-7},
            gate_status="passed",
            started_utc="2026-07-11T00:00:00Z",
            ended_utc="2026-07-11T00:00:01Z",
        )

    receipt_1 = write_stage_receipt(
        layout,
        sequence=1,
        stage="identity",
        producer_case="S12_S13:ZO2",
        inputs=(reference,),
        outputs=(identity_output,),
        gate_values={"phase_rms_waves": 2e-7},
        gate_status="passed",
        started_utc="2026-07-11T00:00:00Z",
        ended_utc="2026-07-11T00:00:01Z",
    )
    propagation_source = tmp_path / "propagation.ZBF"
    propagation_source.write_bytes(b"propagation-output")
    copy_file_once(layout, propagation_source, "continuous/propagation.ZBF")
    propagation_output = ArtifactRef.from_file(
        layout,
        "continuous/propagation.ZBF",
        producer_stage="propagation",
        producer_case="S12_S13:ZO2",
    )
    receipt_2 = write_stage_receipt(
        layout,
        sequence=2,
        stage="propagation",
        producer_case="S12_S13:ZO2",
        inputs=(identity_output,),
        outputs=(propagation_output,),
        gate_values={"memory_gate": False},
        gate_status="failed",
        started_utc="2026-07-11T00:00:02Z",
        ended_utc="2026-07-11T00:00:03Z",
        exception_text="insufficient physical memory",
    )
    assert receipt_1.relative_path != receipt_2.relative_path
    assert receipt_1.relative_path.startswith("receipts/0001_")
    assert receipt_2.relative_path.startswith("receipts/0002_")
    assert verify_artifact_ref(
        layout,
        receipt_2,
        expected_producer_stage="propagation",
        expected_producer_case="S12_S13:ZO2",
    ).is_file()
    receipt_payload = json.loads(
        (layout.run_dir / receipt_2.relative_path).read_text(encoding="utf-8")
    )
    assert receipt_payload["inputs"][0]["sha256"] == identity_output.sha256
    assert receipt_payload["outputs"][0]["sha256"] == propagation_output.sha256
    assert receipt_payload["exception_text"] == "insufficient physical memory"
    with pytest.raises(ValueError, match="next receipt sequence"):
        write_stage_receipt(
            layout,
            sequence=2,
            stage="comparison",
            producer_case="S12_S13:ZO2",
            inputs=(),
            outputs=(),
            gate_values={},
            gate_status="passed",
            started_utc="2026-07-11T00:00:04Z",
            ended_utc="2026-07-11T00:00:05Z",
        )


def test_root_hash_manifest_is_sorted_self_excluding_and_detects_tamper(
    tmp_path: Path,
) -> None:
    prefinal_tamper = _create_layout(tmp_path, "run_prefinal_tamper")
    prefinal_tamper.model_path.write_bytes(b"tampered-before-finalization")
    with pytest.raises(ValueError, match="model hash manifest"):
        write_hash_manifest(prefinal_tamper)
    assert not prefinal_tamper.root_hash_manifest_path.exists()

    layout = _create_layout(tmp_path)
    write_json_once(layout, "comparisons/summary.json", {"value": 1})
    manifest_path = write_hash_manifest(layout)
    lines = manifest_path.read_text(encoding="ascii").splitlines()
    paths = [line.split("  ", 1)[1] for line in lines]
    assert paths == sorted(paths)
    assert "hashes.sha256" not in paths
    assert "model/hashes.sha256" in paths
    assert verify_hash_manifest(layout) == tuple(paths)

    with pytest.raises(RuntimeError, match="finalized"):
        write_json_once(layout, "comparisons/late.json", {"late": True})

    layout.model_path.write_bytes(b"tampered-model")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_hash_manifest(layout)
