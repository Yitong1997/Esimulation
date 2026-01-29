
import numpy as np
import sys
import os
import time

# Adjust path to include src
sys.path.insert(0, os.path.abspath('src'))

from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition

def test_coordinates():
    print("START TEST", flush=True)
    
    # 1. Define Global Configuration
    vertex_global = (0.0, 100.0, 1000.0)
    R_mirror = 2000.0
    start_pos = (0.0, 0.0, 0.0)
    mirror_pos = (10.0, 0.0, 100.0) # Using small numbers to be clear
    
    direction = np.array(mirror_pos) - np.array(start_pos)
    norm = np.linalg.norm(direction)
    direction = tuple(direction / norm)
    
    # 2. Define Surface
    surf_def = SurfaceDefinition(
        surface_type='mirror',
        radius=R_mirror,
        thickness=0.0,
        vertex_position=mirror_pos,
        tilt_x=0.0, 
        tilt_y=0.0
    )
    
    # 3. Create Raytracer
    raytracer = ElementRaytracer(
        surfaces=[surf_def],
        wavelength=0.6328,
        chief_ray_direction=direction,
        entrance_position=start_pos
    )
    
    time.sleep(0.1) # Wait for init prints
    
    # 4. Inspect Optic Surfaces
    surfaces = raytracer.optic.surface_group.surfaces
    
    print(f"NUM_SURFACES: {len(surfaces)}", flush=True)
    
    for i, s in enumerate(surfaces):
        geo = getattr(s, 'geometry', None)
        cs = getattr(geo, 'cs', None)
        if cs:
            # Format explicitly
            x_val = float(cs.x)
            y_val = float(cs.y)
            z_val = float(cs.z)
            print(f"SURF_{i}_POS: {x_val:.4f}, {y_val:.4f}, {z_val:.4f}", flush=True)
            
            if i == 2:
                dist = np.sqrt(10**2 + 100**2)
                expected_z = dist
                print(f"EXPECTED_Z: {expected_z:.4f}", flush=True)
                
                if abs(z_val - expected_z) < 1.0 and abs(x_val) < 1.0:
                     print("RESULT: LOCAL coords confirmed.", flush=True)
                elif abs(x_val - mirror_pos[0]) < 1.0:
                     print("RESULT: GLOBAL coords detected!", flush=True)
                else:
                     print("RESULT: Unknown coords.", flush=True)

if __name__ == "__main__":
    test_coordinates()
