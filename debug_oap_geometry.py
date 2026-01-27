import sys
from pathlib import Path
sys.path.insert(0, r'd:\BTS\src')
import numpy as np
import bts
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition

# Setup standard OAP1
F1 = -100
R1 = -2 * F1 # 200
D1 = 50.0

# Define Surface
surf_def = SurfaceDefinition(
    surface_type='mirror',
    vertex_position=(0, 50, 0), # Vertex at y=50, z=0
    radius=R1,
    conic=-1.0,
    tilt_x=0.0,
    tilt_y=0.0, 
    material='MIRROR'
)

# Setup Raytracer
raytracer = ElementRaytracer(
    surfaces=[surf_def],
    wavelength=0.633,
    chief_ray_direction=(0,0,1), # Input +Z
    entrance_position=(0,50,0),
    exit_chief_direction=(0,0,-1) # Guess -Z
)

from optiland.rays import RealRays

# Input Ray (Global Coords). Chief Ray at (0, 50, 0)
input_rays = RealRays(
    x=[0.0], y=[50.0], z=[0.0],
    L=[0.0], M=[0.0], N=[1.0], # +Z direction
    wavelength=[0.633],
    intensity=[1.0]
)

# Trace
print("Tracing OAP1...")
out_rays = raytracer.trace(input_rays)
print(f"Output Z: {out_rays.z[0]:.4f}")
print(f"Output Direction (N): {out_rays.N[0]:.4f}")

if out_rays.N[0] < 0:
    print("CONCLUSION: Light Reflects Backward (-Z)")
else:
    print("CONCLUSION: Light Refracts Forward (+Z)")
