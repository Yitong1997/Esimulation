
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

import bts
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
from optiland.rays import RealRays

def test_radius_sign():
    print("Testing Radius Sign Convention...")
    
    # Create a simple system with one spherical mirror
    # defined with POSITIVE radius (which bts doc says is Concave/Focusing)
    radius = 1000.0  # f = 500 mm expected if concave
    
    # Input beam: Incident along +Z
    # Raytracer expects incident chief ray direction. 
    # Default is (0,0,1). Entrance position at (0,0,0).
    
    # Define mirror surface at Z=100
    mirror_surf = SurfaceDefinition(
        surface_type='mirror',
        radius=radius,
        thickness=0.0,
        material='mirror',
        semi_aperture=50.0,
        vertex_position=(0, 0, 100)
    )
    
    raytracer = ElementRaytracer(
        surfaces=[mirror_surf],
        wavelength=0.633,
        chief_ray_direction=(0, 0, 1),
        entrance_position=(0, 0, 0)
    )
    
    # Create a test ray parallel to axis, offset by y=10
    # If focusing (concave), reflected ray should point towards axis
    input_rays = RealRays(
        x=[0.0], y=[10.0], z=[0.0],
        L=[0.0], M=[0.0], N=[1.0], # Parallel to Z
        intensity=[1.0], wavelength=[0.633]
    )
    
    print("Tracing ray at y=10...")
    try:
        raytracer.trace(input_rays)
        output_rays = raytracer.get_output_rays()
        
        # Output ray direction (L, M, N)
        # Note: Output rays are in Exit Local Coordinate System.
        # We need to transform them to Global to check direction easily, 
        # OR understand the exit local system.
        # The Exit Surface is placed at the intersection (0,0,100).
        # The exit chief ray direction for a mirror at z=100 facing -z would be (0,0,-1).
        # So Exit Local Z matches Global -Z? No, Exit Local Z aligns with Exit Chief Ray.
        # So if Exit Chief Ray is (0,0,-1), then Local Z=(0,0,1) corresponds to Global (0,0,-1).
        
        # Let's look at Global Direction
        # But wait, trace() returns rays in Exit Local.
        # We can see `get_exit_chief_ray_direction()` to know global exit direction.
        
        exit_dir_global = raytracer.get_exit_chief_ray_direction()
        print(f"Exit Chief Ray Direction (Global): {exit_dir_global}")
        
        # Transform output rays to Global for analysis
        # Using ElementRaytracer internals (hacky but effective for debug)
        R_exit = raytracer.get_exit_rotation_matrix()
        # Ray Position in Exit Local
        x_loc = output_rays.x[0]
        y_loc = output_rays.y[0]
        z_loc = output_rays.z[0]
        L_loc = output_rays.L[0] 
        M_loc = output_rays.M[0]
        N_loc = output_rays.N[0]
        
        print(f"Output Ray (Local): Pos=({x_loc:.3f}, {y_loc:.3f}, {z_loc:.3f}), Dir=({L_loc:.3f}, {M_loc:.3f}, {N_loc:.3f})")
        
        # Typically for "perfect" reflection of chief ray (0,0,1) -> (0,0,-1)
        # Exit Local Z is (0,0,-1).
        # So Local L=0, M=0, N=1 means Global (0,0,-1).
        
        # Our test ray was at y=10.
        # If concave, it should converge.
        # In Global frame:
        #  - It hits mirror at (0, 10, ~100)
        #  - It reflects. If concave, focal point at z=50 (approx, f=R/2=500? No f=R/2=500. Center at -900).
        # Wait, if R=1000 and Center in -Z (front), then Center is at Z = 100 - 1000 = -900.
        # Focal point at Z = 100 - 500 = -400.
        # Ray goes from (0,10,100) to (0,0,-400).
        # dz = -500. dy = -10. 
        # Slope dy/dz = -10/-500 = 0.02.
        # Global M roughly = 0.02 * Global N? 
        # Global N is negative approx -1. So Global M approx -0.02.
        
        # If convex (Center at +Z, R=1000, Center at 1100).
        # Focal point at 100 + 500 = 600. (Virtual focus).
        # Ray reflects as if coming from (0,0,600).
        # Goes from (0,10,100) away from (0,0,600).
        # Vector: (0, 10, 100) - (0, 0, 600) = (0, 10, -500).
        # Direction is (0, 10, -500).
        # Slope dy/dz = 10/-500 = -0.02.
        # Global M approx 0.02? No.
        # Normalized: (0, 0.02, -1). 
        # Global M is positive?
        # Wait.
        # Concave: Ray converges to axis. y goes 10 -> 0. M should be opposite sign to y?
        # Convex: Ray diverges from axis. y goes 10 -> 20. M has same sign as y?
        
        # Let's check Local M.
        # Exit Local X,Y aligned with Global X,Y (if R is simple flip).
        # compute_rotation_matrix checks Y alignment.
        # If Z global = -1. Local Z = Global Z * -1.
        # R = [X_loc, Y_loc, Z_loc].
        # Z_loc = (0,0,-1).
        # ref = (0,1,0). X_loc = ref x Z_loc = (1,0,0) x (0,0,-1) = (0,-1,0)? No.
        # (0,1,0) x (0,0,-1) = (-1, 0, 0).
        # Y_loc = Z_loc x X_loc = (0,0,-1) x (-1,0,0) = (0,1,0).
        # R = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]. (180 deg rot around Y).
        # X flips, Z flips, Y stays.
        
        # If R=[-1, 0, 0; 0, 1, 0; 0, 0, -1]
        # Then Global = R @ Local.
        # Global X = -Local X
        # Global Y = Local Y
        # Global Z = -Local Z
        
        # If Concave: Global M is Negative (pointing to axis). -> Local M is Negative.
        # If Convex: Global M is Positive (pointing away). -> Local M is Positive.
        
        print(f"Local M (Slope): {M_loc}")
        if M_loc < 0:
            print("CONCLUSION: CONCAVE (Focusing) - Sign Convention Matches BTS Doc (Positive R = Concave)")
        else:
            print("CONCLUSION: CONVEX (Diverging) - Sign Convention Mismatch! (Positive R = Convex)")

    except Exception as e:
        print(f"Trace failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_radius_sign()
