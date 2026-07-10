from dataclasses import dataclass, field
from typing import Literal

import numpy as np


Branch = Literal["OO", "OI", "IO"]
DerivationStrategy = Literal[
    "exact_copy",
    "fourier_refine_fixed_window",
    "zero_extend_fixed_sampling",
    "zero_extend_then_fourier_refine",
    "chained_zemax_output",
]
SamplingPurpose = Literal[
    "native",
    "input_resolution",
    "output_resolution",
    "combined_resolution",
    "window_control",
]
SourceKind = Literal["native_zbf", "derived_zbf", "chained_zemax_output"]
SampleValueConvention = Literal["point_value", "cell_energy"]


@dataclass(frozen=True)
class UniformGrid2D:
    x_mm: np.ndarray
    y_mm: np.ndarray
    _dx_mm: float = field(init=False, repr=False)
    _dy_mm: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        x = np.array(self.x_mm, dtype=np.float64, copy=True)
        y = np.array(self.y_mm, dtype=np.float64, copy=True)

        nominal_steps: dict[str, float] = {}
        for name, axis in (("x_mm", x), ("y_mm", y)):
            if axis.ndim != 1 or axis.size < 2:
                raise ValueError(f"{name} must be a one-dimensional array with at least two samples")
            if not np.all(np.isfinite(axis)):
                raise ValueError(f"{name} must contain only finite values")

            steps = np.diff(axis)
            center = axis.size // 2
            nominal_step = (
                axis[center + 1] - axis[center]
                if center + 1 < axis.size
                else axis[center] - axis[center - 1]
            )
            if nominal_step <= 0.0 or not np.allclose(
                steps, nominal_step, rtol=1e-12, atol=0.0
            ):
                raise ValueError(f"{name} must be uniformly spaced in increasing order")
            nominal_steps[name] = float(nominal_step)

        x = np.frombuffer(x.tobytes(order="C"), dtype=x.dtype)
        y = np.frombuffer(y.tobytes(order="C"), dtype=y.dtype)
        x.setflags(write=False)
        y.setflags(write=False)
        object.__setattr__(self, "x_mm", x)
        object.__setattr__(self, "y_mm", y)
        object.__setattr__(self, "_dx_mm", nominal_steps["x_mm"])
        object.__setattr__(self, "_dy_mm", nominal_steps["y_mm"])

    @classmethod
    def centered(
        cls, *, nx: int, ny: int, dx_mm: float, dy_mm: float
    ) -> "UniformGrid2D":
        if nx < 2 or ny < 2 or dx_mm <= 0 or dy_mm <= 0:
            raise ValueError(
                "grid dimensions must be at least two and sampling must be positive"
            )
        x = (np.arange(nx, dtype=np.float64) - nx // 2) * dx_mm
        y = (np.arange(ny, dtype=np.float64) - ny // 2) * dy_mm
        return cls(x_mm=x, y_mm=y)

    @property
    def nx(self) -> int:
        return int(self.x_mm.size)

    @property
    def ny(self) -> int:
        return int(self.y_mm.size)

    @property
    def dx_mm(self) -> float:
        return self._dx_mm

    @property
    def dy_mm(self) -> float:
        return self._dy_mm

    @property
    def pixel_area_mm2(self) -> float:
        return self.dx_mm * self.dy_mm


@dataclass(frozen=True)
class PointField2D:
    values: np.ndarray
    grid: UniformGrid2D

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.complex128, copy=True)
        if values.shape != (self.grid.ny, self.grid.nx):
            raise ValueError("field shape does not match grid")
        if not np.all(np.isfinite(values)):
            raise ValueError("field values must be finite")
        values = np.frombuffer(
            values.tobytes(order="C"), dtype=values.dtype
        ).reshape(values.shape)
        values.setflags(write=False)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class SurfaceConvention:
    surface: int
    side: Literal["after"]
    axis_sign: Literal[-1, 1]


@dataclass(frozen=True)
class SegmentSpec:
    key: str
    start_surface: int
    end_surface: int
    branch: Branch
    model_distance_mm: float
    source_zbf_name: str
    target_zbf_name: str
    start_convention: SurfaceConvention
    end_convention: SurfaceConvention


@dataclass(frozen=True)
class NativeSurfaceSource:
    surface: int
    role: Literal["fresh_continuous", "historical_preflight"]


@dataclass(frozen=True)
class CaseOutputSource:
    producer_case: str
    surface: int


SamplingSource = NativeSurfaceSource | CaseOutputSource


@dataclass(frozen=True)
class SamplingCase:
    case_id: str
    segment_key: str
    source_kind: SourceKind
    strategy: DerivationStrategy
    purpose: SamplingPurpose
    nx: int
    ny: int
    dx_mm: float
    dy_mm: float
    source: SamplingSource
    expected_output_dx_mm: float
    expected_output_dy_mm: float
    depends_on_case: str | None = None
    repeat_count: int = 1
