import sys
import numpy as np
import os

# Ensure src is in path
sys.path.insert(0, r'd:\BTS\src')

# Mock optiland if not fully installed or to control behavior? 
# No, we should use the installed optiland.
# But we need to make sure we don't fail on imports.

try:
    from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
    from optiland.rays import RealRays
    import optiland
    print(f"Optiland version: {optiland.__version__ if hasattr(optiland, '__version__') else 'unknown'}")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_check():
    print("Initializing test...")
    
    # Define an OAP-like surface
    # Parabolic mirror, focal length 1000mm -> R = 2*f = 2000mm. Concave -> R=-2000
    # Off-axis: we can just set a tilt or an offset.
    # The check relies on _exit_dir_local being set, which happens in trace_chief_ray -> _finalize_optic.
    
    surf = SurfaceDefinition(
        surface_type='mirror',
        radius=-2000.0,
        thickness=0.0, # distance to next surface
        material='mirror',
        conic=-1.0, # Parabola
        tilt_x=0.0,
        tilt_y=0.1, # Some tilt (radians) roughly 5.7 degrees
        vertex_position=[0.0, 50.0, 0.0] # Off-axis position
    )
    
    # Tracer
    # Entrance position at 0,0,0
    print("Creating ElementRaytracer...")
    tracer = ElementRaytracer(
        surfaces=[surf],
        wavelength=0.6328,
        entrance_position=[0.0, 0.0, 0.0]
    )
    
    # Rays
    input_rays = RealRays(
        x=[0.0], y=[0.0], z=[0.0],
        L=[0.0], M=[0.0], N=[1.0],
        intensity=[1.0], wavelength=[0.6328]
    )
    
    # Trace
    print("Calling trace()...")
    tracer.trace(input_rays)
    
    print("Trace complete.")

if __name__ == "__main__":
    test_check()
