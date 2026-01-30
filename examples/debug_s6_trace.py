
import sys
from pathlib import Path
import numpy as np

# Add paths
current_file = Path(__file__).resolve()
project_root = Path(r"d:\BTS")
sys.path.insert(0, str(project_root / 'src'))

from bts.io import load_zmx
from wavefront_to_rays.element_raytracer import ElementRaytracer
from optiland.rays import RealRays
from hybrid_optical_propagation.hybrid_element_propagator import HybridElementPropagator

def debug_s6():
    zmx_path = project_root / 'optiland-master' / 'tests' / 'zemax_files' / 'complicated_fold_mirrors_setup_v2.zmx'
    print(f"Loading {zmx_path}")
    system = load_zmx(str(zmx_path))
    
    # Global Surface 6 corresponds to index 6 in inspection list (S6)
    # It is a Mirror at (60, 34, 80).
    # Previous Surface (S5) exit is Entrance for S6.
    # We simulate the S6 trace in isolation.
    
    # Setup Entrance Parameters (Simulating Exit from S5)
    # S5 is at (0, 40, 80). Ray goes along +X (1, 0, 0).
    entrance_pos = (0, 40, 80)
    entrance_dir = (1, 0, 0)
    
    # Get Global Surface 6
    gs6 = system.get_global_surfaces()[6]
    print(f"Surface 6 Vertex: {gs6.vertex_position}")
    
    # Create Dummy Hybrid Propagator to use helper methods if needed (or just use logic from it)
    # But we can use ElementRaytracer directly if we prepare SurfaceDefinition.
    
    # Manually create SurfaceDefinition from GlobalSurfaceDefinition
    # We reuse logic from HybridElementPropagator._create_surface_definition
    
    propagator = HybridElementPropagator(wavelength_um=0.6328, debug=True)
    
    # Mock Entrance Axis
    class MockAxis:
        def __init__(self, pos, direction):
            class Vec:
                def __init__(self, v): self.v = np.array(v)
                def to_array(self): return self.v
            self.position = Vec(pos)
            self.direction = Vec(direction)
            
    ent_axis = MockAxis(entrance_pos, entrance_dir)
    # Exit Axis (Expected Reflection from S6)
    # S6 reflects (1, 0, 0) to (0, -1, 0).
    # Exit Position = Intersection of Chief Ray (1,0,0) with S6.
    # Ray x=t, y=40, z=80. S6 Vertex (60, 34, 80). Normal roughly (+X).
    # Intersect at (54, 40, 80).
    expected_exit_pos = (54, 40, 80)
    expected_exit_dir = (0, -1, 0)
    
    exit_axis = MockAxis(expected_exit_pos, expected_exit_dir)
    
    surface_def = propagator._create_surface_definition(gs6, ent_axis, exit_axis)
    print(f"Created SurfaceDef: Type={surface_def.surface_type}, Radius={surface_def.radius}")
    print(f"  Tilt X={surface_def.tilt_x*180/np.pi:.2f}, Tilt Y={surface_def.tilt_y*180/np.pi:.2f}")
    
    # Construct ElementRaytracer
    tracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=0.6328,
        chief_ray_direction=entrance_dir,
        entrance_position=entrance_pos,
        exit_chief_direction=expected_exit_dir,
        exit_position=expected_exit_pos,
        debug=True
    )
    
    # Trace Chief Ray explicitly
    tracer.trace_chief_ray()
    # print(f"Traced Chief Ray. Intersection Local: {tracer._chief_intersection_local}")
    
    # Create Input Rays (Global)
    # 1. Chief Ray (0, 40, 80)
    # 2. Offset Ray (0, 40, 80+1) (Check Z offset)
    x = [0.0]
    y = [40.0]
    z = [80.0]
    L = [1.0]
    M = [0.0]
    N = [0.0]
    
    input_rays = RealRays(x=x, y=y, z=z, L=L, M=M, N=N, intensity=[1], wavelength=[0.6328])
    
    print("\nTracing Input Rays (Global)...")
    output_rays = tracer.trace(input_rays)
    
    print("\nOutput Rays (Exit Local Frame):")
    print(f"  x: {output_rays.x}")
    print(f"  y: {output_rays.y}")
    print(f"  z: {output_rays.z}")
    print(f"  L: {output_rays.L}")
    print(f"  M: {output_rays.M}")
    print(f"  N: {output_rays.N}")
    
    # Check if y is near 0 (Correct behavior if exit_pos matches beam)
    # expected_exit_pos was calculated as (54, 40, 80).
    # Ray hits at (60, 40, 80).
    # Difference (6, 0, 0) in Global.
    # In Exit Frame (aligned -Y), difference should be in X/Z components, not Y.
    # So Y should be 0.
    
    if np.allclose(output_rays.y, 0, atol=1e-3):
        print("\n[SUCCESS] Output Y is near 0. Correct coordinates!")
    else:
        print(f"\n[FAIL] Output Y={output_rays.y[0]} (Expected ~0). Offset still present?")

if __name__ == "__main__":
    debug_s6()
