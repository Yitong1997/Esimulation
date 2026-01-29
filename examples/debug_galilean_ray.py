
import sys
import os
import numpy as np

# Adjust path to include src
sys.path.insert(0, os.path.abspath('src'))
sys.path.insert(0, os.path.abspath('optiland-master'))

from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
from optiland.rays import RealRays

def trace_galilean_chief_ray():
    print("DEBUG: Tracing Galilean Chief Ray\n")
    
    # --- Parameters ---
    WAVELENGTH_UM = 1.0
    F1 = -1000.0
    F2 = 2000.0
    D1 = 100.0
    D2 = 100.0
    L = F2 + F1 # 1000.0
    # Wait, in the test code: L_MM = F2_MM + F1_MM = 2000 + (-1000) = 1000.
    # IN CODE VIEWED EARLIER:
    # 48: - 两镜间距: L = 300 - 100 = 200 mm (Comment)
    # 110: L_MM = F2_MM + F1_MM (Code)
    # The parameters in the code lines 97-108:
    # F1 = -1000, F2 = 2000.
    # L = 1000.
    
    # Coordinates:
    # Source: z=1000. (y=0 ?) Source is at (0,0,1000)
    # OAP1: z=500. Vertex at (0, D1, 500).
    # OAP2: z = 500 - L = -500. Vertex at (0, D2, -500).
    # NOTE: In test code viewed earlier:
    # 264: z=500-l_mm = 500 - 1000 = -500.
    # But F1=-1000, F2=2000.
    # Focus 1: Vertex(500) + f(-1000)/2 = 0? No f=R/2?
    # f is focal length. f=-1000.
    # Focus = Vertex + f? 
    # For a mirror, Focus is at f/2? No, f IS the focal length.
    # R = 2*f.
    # Vertex is at z.
    # Parabolic Mirror (Conic -1). 
    # Optiland Convention: 
    # Light +Z. R<0 (Concave). F = R/2. F<0. Focus at z+F.
    # Here Light -Z. 
    # OAP1 Convex. F1=-1000. 
    # If Light goes -Z, and mirror is Convex (bulges towards light), 
    # Center of Curvature is in -Z direction (behind mirror relative to light).
    # So R should be negative in Local Frame (if Z is normal).
    
    # Let's perform step-by-step tracing using ElementRaytracer logic
    
    # 1. Source -> OAP1
    print("--- Step 1: Source -> OAP1 ---")
    
    # Input Ray
    source_pos = (0.0, 0.0, 1000.0)
    ray_dir = (0.0, 0.0, -1.0) # -Z direction
    
    print(f"Input Pos: {source_pos}")
    print(f"Input Dir: {ray_dir}")
    
    # OAP1 Definition
    # Code: radius = -2 * F1 = -2 * (-1000) = 2000.
    # "radius=r1_mm" in test.
    # Convex Mirror. R > 0.
    oap1_def = SurfaceDefinition(
        surface_type='mirror',
        radius=2000.0,  # R=2000. Convex.
        conic=-1.0,
        vertex_position=(0.0, 100.0, 500.0)
    )
    
    tracer1 = ElementRaytracer(
        surfaces=[oap1_def],
        wavelength=WAVELENGTH_UM,
        chief_ray_direction=ray_dir,
        entrance_position=source_pos
    )
    
    # Trace to get exit direction
    exit_dir1 = tracer1.get_exit_chief_ray_direction()
    exit_pos_global1 = tracer1.get_global_chief_ray_intersection()
    
    print(f"OAP1 Global Hit Pos: {exit_pos_global1}")
    print(f"OAP1 Exit Dir: {exit_dir1}")
    
    # CHECK VIRTUAL FOCUS
    # Ray: P + t * D
    # Focus Plane z = -500
    # t = (-500 - Pz) / Dz
    P = np.array(exit_pos_global1)
    D = np.array(exit_dir1)
    t_focus = (-500.0 - P[2]) / D[2]
    P_focus = P + t_focus * D
    
    expected_focus = np.array([0.0, 100.0, -500.0])
    error_focus = P_focus - expected_focus
    print(f"Virtual Focus Hit: {P_focus}")
    print(f"Expected Focus: {expected_focus}")
    print(f"Focus Error: {error_focus} (Norm: {np.linalg.norm(error_focus):.6e})")
    
    # 2. OAP1 -> OAP2
    print("\n--- Step 2: OAP1 -> OAP2 ---")
    
    # OAP2 Definition
    # Code: radius = -2 * F2 = -4000? 
    # Code: r2_mm = -2 * F2_MM = -4000.
    # F2=2000.
    # "radius=4000" in line 265 of viewed file? 
    # Wait, line 265 says "radius=4000".
    # Line 222 says "r2_mm = -2 * f2_mm".
    # User updated the file? Or I misread?
    # Let's trust the hardcoded value if present, or variable.
    # Line 265 in view: `radius=4000, #全局中凸，相对于-Z光线凹`.
    # OK, R=4000.
    
    oap2_def = SurfaceDefinition(
        surface_type='mirror',
        radius=4000.0,
        conic=-1.0, 
        vertex_position=(0.0, 100.0, 1500.0), # Physical position
        tilt_x=np.pi # Face the incoming light (from +Z)
    )
    
    tracer2 = ElementRaytracer(
        surfaces=[oap2_def],
        wavelength=WAVELENGTH_UM,
        chief_ray_direction=exit_dir1,
        entrance_position=exit_pos_global1
    )
    
    exit_dir2 = tracer2.get_exit_chief_ray_direction()
    exit_pos_global2 = tracer2.get_global_chief_ray_intersection()
    
    print(f"OAP2 Global Hit Pos: {exit_pos_global2}")
    print(f"OAP2 Exit Dir: {exit_dir2}")
    
    # Check if (0,0,1)
    L, M, N = exit_dir2
    tilt = np.sqrt(L**2 + M**2)
    print(f"\nTilt Magnitude: {tilt:.8e}")
    if tilt < 1e-6 and N > 0:
        print("SUCCESS: Exit direction is effectively (0,0,1).")
    elif tilt < 1e-6 and N < 0:
        print("WARNING: Exit direction is (0,0,-1) (Backwards).")
    else:
        print("FAILURE: Exit direction is tilted.")

if __name__ == "__main__":
    trace_galilean_chief_ray()
