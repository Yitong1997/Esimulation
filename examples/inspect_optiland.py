
import sys
import os
import inspect

# Adjust path to include src
sys.path.insert(0, os.path.abspath('src'))
sys.path.insert(0, os.path.abspath('optiland-master'))

from optiland.surfaces.standard_surface import Surface
from optiland.geometries.standard import StandardGeometry
from optiland.materials import IdealMaterial
from optiland.coordinate_system import CoordinateSystem

def inspect_surface():
    print("Inspecting optiland.surfaces.Surface methods:\n")
    
    # Create a dummy surface
    cs = CoordinateSystem()
    geo = StandardGeometry(coordinate_system=cs, radius=100.0)
    mat = IdealMaterial(n=1.5)
    surf = Surface(geometry=geo, material_post=mat)
    
    methods = inspect.getmembers(surf, predicate=inspect.ismethod)
    for name, method in methods:
        if not name.startswith('__'):
            print(f"Method: {name}")

if __name__ == "__main__":
    inspect_surface()
