"""ZMX loader for POP."""

from __future__ import annotations

from typing import List, Any

from sequential_system.zmx_parser import ZmxParser
from sequential_system.coordinate_system import (
    GlobalSurfaceDefinition,
    SurfaceTraversalAlgorithm,
    ZemaxToOptilandConverter,
)


def load_zmx(path: str) -> List[GlobalSurfaceDefinition]:
    parser = ZmxParser(path)
    data_model = parser.parse()
    traversal = SurfaceTraversalAlgorithm(data_model)
    return traversal.traverse()


def to_optiland(
    surfaces: List[GlobalSurfaceDefinition],
    wavelength_um: float = 0.55,
    entrance_pupil_diameter: float = 10.0,
) -> Any:
    converter = ZemaxToOptilandConverter(
        surfaces,
        wavelength=wavelength_um,
        entrance_pupil_diameter=entrance_pupil_diameter,
    )
    return converter.convert()


__all__ = ["load_zmx", "to_optiland", "GlobalSurfaceDefinition"]
