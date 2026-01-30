import numpy as np
import optiland
from optiland.surfaces import Surface
from optiland.geometries import StandardGeometry
from optiland.materials import IdealMaterial
from optiland.rays import RealRays
from optiland.coordinate_system import CoordinateSystem

def test_independent_surface_refraction():
    print("Testing Independent Surface Refraction...")
    
    # Create input ray along Z axis
    wavelength = 0.55
    rays = RealRays(
        x=[0.0], y=[0.0], z=[0.0],
        L=[0.0], M=[0.0], N=[1.0],
        intensity=[1.0], wavelength=[wavelength]
    )
    
    # Create an independent surface (tilted to force refraction if index changes)
    # Tilt 45 degrees
    cs = CoordinateSystem(rx=np.radians(45))
    geometry = StandardGeometry(coordinate_system=cs, radius=np.inf)
    
    # Case 1: Material Post = 1.0 (Air)
    # Expected: No refraction if input is air.
    rays_1 = RealRays(
        x=[0.0], y=[0.0], z=[0.0],
        L=[0.0], M=[0.0], N=[1.0],
        intensity=[1.0], wavelength=[wavelength]
    )
    
    mat_air = IdealMaterial(n=1.0)
    surf_air = Surface(
        previous_surface=None,
        geometry=geometry,
        material_post=mat_air
    )
    
    print("\n--- Trace 1: Target Material = Air (n=1.0) ---")
    surf_air.trace(rays_1)
    print(f"Input Dir: (0, 0, 1)")
    print(f"Output Dir: ({rays_1.L[0]:.4f}, {rays_1.M[0]:.4f}, {rays_1.N[0]:.4f})")
    
    # Case 2: Material Post = 1.5 (Glass)
    # Expected: Refraction if optiland assumes input is n=1.0
    rays_2 = RealRays(
        x=[0.0], y=[0.0], z=[0.0],
        L=[0.0], M=[0.0], N=[1.0],
        intensity=[1.0], wavelength=[wavelength]
    )
    
    mat_glass = IdealMaterial(n=1.5)
    surf_glass = Surface(
        previous_surface=None,
        geometry=geometry,
        material_post=mat_glass
    )
    
    print("\n--- Trace 2: Target Material = Glass (n=1.5) ---")
    surf_glass.trace(rays_2)
    print(f"Input Dir: (0, 0, 1)")
    print(f"Output Dir: ({rays_2.L[0]:.4f}, {rays_2.M[0]:.4f}, {rays_2.N[0]:.4f})")
    
    # Case 3: Linked Surface
    # Create a system: Surf A (n=1.5) -> Surf B (n=1.5)
    # Ray starts in glass (after Surf A), hits Surf B. Should be no refraction.
    
    print("\n--- Trace 3: In Glass (n=1.5) -> Out Glass (n=1.5) ---")
    
    # Surf A: Just launches rays into n=1.5
    rays_3 = RealRays(
        x=[0.0], y=[0.0], z=[0.0],
        L=[0.0], M=[0.0], N=[1.0],
        intensity=[1.0], wavelength=[wavelength]
    )
    
    surf_A = Surface(
        previous_surface=None,
        geometry=StandardGeometry(CoordinateSystem(), np.inf), # Flat at z=0
        material_post=mat_glass # Post is 1.5
    )
    
    # Trace though A to enter medium
    surf_A.trace(rays_3)
    print("After Surf A (entering glass):")
    print(f"Output Dir: ({rays_3.L[0]:.4f}, {rays_3.M[0]:.4f}, {rays_3.N[0]:.4f})")
    
    # Surf B: Tilted, n=1.5 (same as A)
    # Should be NO refraction because n_in (from A) = n_out (from B) = 1.5
    
    # Move B slightly forward
    cs_b = CoordinateSystem(z=10, rx=np.radians(45))
    geometry_b = StandardGeometry(coordinate_system=cs_b, radius=np.inf)
    
    surf_B = Surface(
        previous_surface=surf_A, # Link to A
        geometry=geometry_b,
        material_post=mat_glass
    )
    
    surf_B.trace(rays_3)
    print("After Surf B (staying in glass):")
    print(f"Output Dir: ({rays_3.L[0]:.4f}, {rays_3.M[0]:.4f}, {rays_3.N[0]:.4f})")
    is_parallel_3 = np.isclose(rays_3.L[0], 0) and np.isclose(rays_3.M[0], 0) and np.isclose(rays_3.N[0], 1)
    print(f"Refraction Occurred? {not is_parallel_3}")
    
    # Case 4: Independent Surface simulating Case 3
    # If we don't link previous_surface, does Optiland know n_in?
    # Hint: It probably assumes n_in=1.00 if prev is None.
    
    print("\n--- Trace 4: Independent Surface Simulation (n=1.5 -> n=1.5) ---")
    rays_4 = RealRays(
        x=[0.0], y=[0.0], z=[0.0],
        L=[0.0], M=[0.0], N=[1.0],
        intensity=[1.0], wavelength=[wavelength]
    )
    
    # We want to simulate staying in 1.5. 
    # If we create an independent surface with n=1.5, and trace...
    surf_indep_glass = Surface(
        previous_surface=None,
        geometry=geometry, # Tilted
        material_post=mat_glass
    )
    
    surf_indep_glass.trace(rays_4)
    print(f"Output Dir: ({rays_4.L[0]:.4f}, {rays_4.M[0]:.4f}, {rays_4.N[0]:.4f})")
    is_parallel_4 = np.isclose(rays_4.L[0], 0) and np.isclose(rays_4.M[0], 0) and np.isclose(rays_4.N[0], 1)
    print(f"Refraction Occurred? {not is_parallel_4}")
    
    if not is_parallel_4:
        print(">> Independent surface assumes n_in=1.0 (Air) by default!")

if __name__ == "__main__":
    test_independent_surface_refraction()
