"""Configuration objects for Zemax POP benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass
class PopSamplingConfig:
    grid_size: int = 1024
    physical_size_mm: float = 348.0
    beam_diam_fraction: float | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return _jsonable_dataclass(self)


@dataclass
class GaussianInputConfig:
    wavelength_um: float = 10.64
    w0_mm: float = 29.0
    z0_mm: float = 0.0

    def to_jsonable(self) -> dict[str, Any]:
        return _jsonable_dataclass(self)


@dataclass
class ZbfInputConfig:
    path: Path
    reference_mode: str = "reference_relative"

    def to_jsonable(self) -> dict[str, Any]:
        return _jsonable_dataclass(self)


@dataclass
class ComparisonConfig:
    surface_indices: Sequence[int] | None = None
    # The POP analysis files are labelled by the incident surface boundary;
    # retain the established entrance mapping for the benchmark geometry.
    pop_position: str = "entrance"
    mask_threshold: float = 0.1
    sampling_rtol: float = 1e-4
    sampling_atol_mm: float = 5e-9

    def to_jsonable(self) -> dict[str, Any]:
        return _jsonable_dataclass(self)


@dataclass
class BenchmarkConfig:
    zmx_path: Path
    output_dir: Path
    sampling: PopSamplingConfig
    gaussian: GaussianInputConfig = field(default_factory=GaussianInputConfig)
    zbf_input: ZbfInputConfig | None = None
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    num_rays: int = 10000
    coordinate_priority: str = "x"
    auto_asm: bool = True
    reconstruction_mask_ratio: float | None = 0.95
    phase_method: str = "griddata"
    zernike_terms: int | None = 5
    sampling_sigma: float = 4.0
    auto_resample: bool = False
    auto_unwarp_at_incident: bool = True
    auto_unwarp_surface_indices: Sequence[int] | str | None = None
    print_status: bool = False

    def __post_init__(self) -> None:
        self.zmx_path = Path(self.zmx_path)
        self.output_dir = Path(self.output_dir)

    def to_jsonable(self) -> dict[str, Any]:
        return _jsonable_dataclass(self)


def _jsonable_dataclass(obj: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name in getattr(obj, "__dataclass_fields__", {}):
        payload[name] = _jsonable_value(getattr(obj, name))
    return payload


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_jsonable"):
        return value.to_jsonable()
    if isinstance(value, tuple):
        return [_jsonable_value(v) for v in value]
    if isinstance(value, list):
        return [_jsonable_value(v) for v in value]
    return value
