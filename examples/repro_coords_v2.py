
import numpy as np
import sys
import os

# Adjust path to include src
sys.path.insert(0, os.path.abspath('src'))

from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
from optiland.rays import RealRays

def test_coordinates():
    print("Testing ElementRaytracer Coordinate Transformation...")
    
    # 1. Define Global Configuration
    # Mirror at (0, 100, 1000)
    vertex_global = (0.0, 100.0, 1000.0)
    R_mirror = 2000.0
    
    # Entrance Beam at (0, 102, 900)
    # Direction: Tilted towards mirror slightly
    # Vector to mirror: (0, -2, 100)
    # Let's say beam direction is exactly along Z for simplicity first, 
    # OR tilted. User says "tilted".
    # Let's define a beam starting at (0, 100, 900) pointing to (0, 100, 1000) -> Z direction.
    # To make it key: let's tilt the beam.
    # Start at (0, 0, 0). Mirror at (10, 0, 100).
    # Beam Direction = Normalize(10, 0, 100).
    
    start_pos = (0.0, 0.0, 0.0)
    mirror_pos = (10.0, 0.0, 100.0)
    
    direction = np.array(mirror_pos) - np.array(start_pos)
    norm = np.linalg.norm(direction)
    direction = direction / norm
    direction = tuple(direction)
    
    print(f"Global Start: {start_pos}")
    print(f"Global Mirror: {mirror_pos}")
    print(f"Global Direction: {direction}")
    
    # 2. Define Surface
    # Need to verify if SurfaceDefinition assumes tilt is pre-calculated.
    # In hybrid_element_propagator, tilt is calculated relative to beam.
    # If beam points directly at mirror center, and mirror faces -Z (global), 
    # then relative tilt exists.
    
    # Let's assume Global Mirror Normal is (0, 0, -1).
    # Beam Direction is approx (0.1, 0, 1).
    # Relative tilt is needed.
    # But for this test, we only care about POSITION coordinates in optic.
    
    surf_def = SurfaceDefinition(
        surface_type='mirror',
        radius=R_mirror,
        thickness=0.0,
        vertex_position=mirror_pos,
        tilt_x=0.0, # Just dummy values
        tilt_y=0.0
    )
    
    # 3. Create Raytracer
    raytracer = ElementRaytracer(
        surfaces=[surf_def],
        wavelength=0.6328,
        chief_ray_direction=direction,
        entrance_position=start_pos
    )
    
    # 4. Inspect Optic Surfaces
    # index 0: Object
    # index 1: Stop (at 0,0,0)
    # index 2: Mirror
    
    surfaces = raytracer.optic.surface_group.surfaces
    print(f"\nInspect Surfaces (Total {len(surfaces)}):")
    
    for i, s in enumerate(surfaces):
        geo = getattr(s, 'geometry', None)
        cs = getattr(geo, 'cs', None)
        if cs:
            print(f"Surface {i} Coords: x={cs.x}, y={cs.y}, z={cs.z}")
            
            # Check correctness
            if i == 2:
                # Expected Local Position:
                # Since beam points from Start to Mirror, Mirror should be at (0, 0, distance) in Local Frame.
                dist = np.sqrt(10**2 + 100**2)
                print(f"  Expected 'z' (dist): {dist:.4f}")
                print(f"  Expected 'x', 'y': ~0.0")
                
                # Check actual
                # cs.x, cs.y, cs.z might be arrays or scalars
                x_val = float(cs.x)
                y_val = float(cs.y)
                z_val = float(cs.z)
                
                if abs(z_val - dist) < 1e-3 and abs(x_val) < 1e-3 and abs(y_val) < 1e-3:
                    print("  [SUCCESS] Surface coordinates are LOCAL and correct.")
                else:
                    print(f"  [FAILURE] Surface coordinates do not match expectation.")
                    print(f"  Gap: x={x_val}, y={y_val}, z={z_val-dist}")
                    
                    # Check if they look Global
                    if abs(x_val - mirror_pos[0]) < 1e-3:
                         print("  [DETECTED] Coordinates match GLOBAL X position!")
        else:
            print(f"Surface {i}: No CS found (Object/Stop?)")

if __name__ == "__main__":
    test_coordinates()
