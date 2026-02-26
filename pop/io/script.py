"""
Export POP System to Python script.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any, List, Optional
import numpy as np

from pop.system import System
from sequential_system.coordinate_system import GlobalSurfaceDefinition


def _format_value(value: Any) -> str:
    """Format a value for Python code."""
    if isinstance(value, np.ndarray):
        # Format numpy array as list, then wrap in np.array
        # Use repr to get strict float formatting, but might be too verbose
        # Use simple list conversion
        if value.ndim == 0:
            return f"{float(value)}"
        return f"np.array({value.tolist()})"
    if isinstance(value, (float, np.float64, np.float32)):
        if np.isinf(value):
            return "np.inf"
        return f"{value}"
    if isinstance(value, str):
        return repr(value)
    return repr(value)


def _generate_imports() -> str:
    """Generate necessary imports."""
    return """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
Generated POP Optical System Script
\"\"\"

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Adjust path to find 'pop' package if needed
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Assuming pop is installed or in path
try:
    from pop import System, propagate, GaussianSource
    from sequential_system.coordinate_system import GlobalSurfaceDefinition
except ImportError:
    # Fallback to relative import if running from within the repo structure
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pop import System, propagate, GaussianSource
    from sequential_system.coordinate_system import GlobalSurfaceDefinition

"""


def _generate_surface_code(surface: GlobalSurfaceDefinition, idx: int) -> str:
    """Generate code to create a GlobalSurfaceDefinition."""
    # We want to generate something like:
    # s0 = GlobalSurfaceDefinition(
    #     index=0,
    #     ...
    # )
    
    code = f"    # Surface {surface.index}\n"
    code += f"    s{idx} = GlobalSurfaceDefinition(\n"
    code += f"        index={surface.index},\n"
    code += f"        surface_type={_format_value(surface.surface_type)},\n"
    code += f"        vertex_position={_format_value(surface.vertex_position)},\n"
    code += f"        orientation={_format_value(surface.orientation)},\n"
    code += f"        radius={_format_value(surface.radius)},\n"
    code += f"        conic={_format_value(surface.conic)},\n"
    code += f"        is_mirror={_format_value(surface.is_mirror)},\n"
    code += f"        semi_aperture={_format_value(surface.semi_aperture)},\n"
    code += f"        material={_format_value(surface.material)},\n"
    
    # Optional fields
    if hasattr(surface, 'radius_x') and not np.isinf(surface.radius_x):
         code += f"        radius_x={_format_value(surface.radius_x)},\n"
    if hasattr(surface, 'conic_x') and surface.conic_x != 0.0:
         code += f"        conic_x={_format_value(surface.conic_x)},\n"
    if hasattr(surface, 'thickness'):
         code += f"        thickness={_format_value(surface.thickness)},\n"
    if hasattr(surface, 'comment') and surface.comment:
         code += f"        comment={_format_value(surface.comment)},\n"
    if hasattr(surface, 'focal_length') and not np.isinf(surface.focal_length):
         code += f"        focal_length={_format_value(surface.focal_length)},\n"
    # asphere_coeffs
    if hasattr(surface, 'asphere_coeffs') and surface.asphere_coeffs:
         code += f"        asphere_coeffs={_format_value(surface.asphere_coeffs)},\n"

    code += "    )\n"
    return code


def _generate_system_code(system: System) -> str:
    """Generate the function to build the system."""
    code = "def build_system() -> System:\n"
    code += "    surfaces = []\n\n"
    
    surface_vars = []
    for i, surface in enumerate(system.surfaces):
        code += _generate_surface_code(surface, i)
        code += f"    surfaces.append(s{i})\n\n"
        surface_vars.append(f"s{i}")
    
    code += f"    return System(name={_format_value(system.name)}, surfaces=surfaces)\n"
    return code


def _generate_main_block() -> str:
    """Generate the main block for running the script."""
    return """
def main():
    system = build_system()
    print(f"Built System: {system.name}")
    print(f"Number of surfaces: {len(system.surfaces)}")
    
    # --- Example Propagation Setup (Modify as needed) ---
    # source = GaussianSource(
    #     wavelength_um=1.0,
    #     w0_mm=5.0,
    #     grid_size=512,
    #     z0_mm=-100.0, # Adjust based on your system entry
    #     physical_size_mm=50.0
    # )
    #
    # result = propagate(system, source, debug=True)
    # result.plot_optical_axis_3d(show=True)
    # ----------------------------------------------------

if __name__ == "__main__":
    main()
"""


def save_system_as_script(system: System, filename: str) -> None:
    """
    Export the System object to a Python script.
    
    Args:
        system: The System object to export.
        filename: The path to save the script to.
    """
    content = _generate_imports()
    content += "\n\n"
    content += _generate_system_code(system)
    content += "\n"
    content += _generate_main_block()
    
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"System exported to {path.resolve()}")
