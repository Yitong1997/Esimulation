from pathlib import Path

from .models import SegmentSpec, SurfaceConvention


S7 = SurfaceConvention(7, "after", -1, False)
S8 = SurfaceConvention(8, "after", -1, False)
S12 = SurfaceConvention(12, "after", 1, True)
S13 = SurfaceConvention(13, "after", 1, True)
S14 = SurfaceConvention(14, "after", 1, True)

BICONIC_SEGMENTS = (
    SegmentSpec(
        "S07_S08",
        7,
        8,
        "OO",
        368.600000,
        "biconic_focus_test_0007.ZBF",
        "biconic_focus_test_0008.ZBF",
        S7,
        S8,
    ),
    SegmentSpec(
        "S12_S13",
        12,
        13,
        "OI",
        608.600000,
        "biconic_focus_test_0012.ZBF",
        "biconic_focus_test_0013.ZBF",
        S12,
        S13,
    ),
    SegmentSpec(
        "S13_S14",
        13,
        14,
        "IO",
        2.000000,
        "biconic_focus_test_0013.ZBF",
        "biconic_focus_test_0014.ZBF",
        S13,
        S14,
    ),
)


def resolve_biconic_baseline(baseline_dir: Path) -> Path:
    path = baseline_dir.resolve()
    required = [
        "biconic_focus_test.zmx",
        "biconic_focus_test.CFG",
        "biconic_focus_test.txt",
        "biconic_phase.txt",
        "biconic_focus_test_0007.ZBF",
        "biconic_focus_test_0008.ZBF",
        "biconic_focus_test_0012.ZBF",
        "biconic_focus_test_0013.ZBF",
        "biconic_focus_test_0014.ZBF",
    ]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing baseline files: {missing}")
    return path
