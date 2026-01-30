
import sys
import numpy as np
import optiland.backend as be
from optiland.surfaces import Surface
from optiland.geometries import StandardGeometry
from optiland.coordinate_system import CoordinateSystem
from optiland.rays import RealRays
from optiland.materials import IdealMaterial
from optiland.interactions import RefractiveReflectiveModel

def reproduce():
    # Setup
    # Incident ray along Global +Y: (0, 1, 0)
    # Start at origin for simplicity
    rays = RealRays(x=0, y=-1, z=0, L=0, M=1, N=0, wavelength=0.55, intensity=1.0)
    
    # Define Geometry with User's params
    # Tilt X = 45 deg = pi/4
    # Tilt Y = -pi
    rx = np.pi/4
    ry = -np.pi
    
    cs = CoordinateSystem(x=0, y=0, z=0, rx=rx, ry=ry, rz=0)
    
    # Plane Mirror (Infinite radius)
    geometry = StandardGeometry(coordinate_system=cs, radius=np.inf)
    
    # Materials
    mat_pre = IdealMaterial(n=1.0)
    mat_post = IdealMaterial(n=1.0)
    
    # Interaction Model: Reflective
    interaction = RefractiveReflectiveModel(
        parent_surface=None,
        is_reflective=True
    )
    
    # Surface
    surface = Surface(
        previous_surface=None,
        material_post=mat_post,
        geometry=geometry,
        interaction_model=interaction
    )
    # Hack to set material_pre since previous_surface is None
    # In Surface.trace: material_pre getter uses previous_surface.material_post or self.material_post
    # If we want reflection in air, mat_post=1 is fine.
    
    print("Initial Rays:")
    print(f"L={rays.L}, M={rays.M}, N={rays.N}")
    
    # Trace
    # Surface trace calls:
    # 1. geometry.localize(rays)
    # 2. interaction (reflection)
    # 3. geometry.globalize(rays)
    
    surface.trace(rays)
    
    print("\nFinal Rays:")
    print(f"L={rays.L}, M={rays.M}, N={rays.N}")
    
    # Manual Check of expected logic
    print("\n--- Manual Transform Check ---")
    
    # 1. Localize
    # v_loc = Rx(-rx) @ Ry(-ry) @ Rz(-rz) @ v_glob
    v_in = np.array([0, 1, 0])
    
    # Rz(0)
    v1 = v_in
    
    # Ry(-(-pi)) = Ry(pi)
    # Ry matrix: [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
    # cos(pi) = -1, sin(pi) = 0
    # Ry(pi) = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
    # Ry(pi) @ [0, 1, 0] = [0, 1, 0]
    v2 = np.array([0, 1, 0])
    
    # Rx(-pi/4) = Rx(-45)
    # Rx matrix: [[1, 0, 0], [0, cos, -sin], [0, sin, cos]]
    # cos(-45) = 0.707, sin(-45) = -0.707
    # Rx(-45) = [[1, 0, 0], [0, 0.707, 0.707], [0, -0.707, 0.707]]
    # Rx(-45) @ [0, 1, 0] = [0, 0.707, -0.707]
    v_loc = np.array([0, 0.70710678, -0.70710678])
    
    print(f"Manual Local Ray: {v_loc}")
    
    # 2. Reflect
    # Normal is (0, 0, 1) in local
    # L_out = L_in - 2(L_in . N)N
    # L_in dot N = -0.707
    # L_out = (0, 0.707, -0.707) - 2(-0.707)(0, 0, 1) = (0, 0.707, 0.707)
    v_out_loc = np.array([0, 0.70710678, 0.70710678])
    
    print(f"Manual Reflected Local Ray: {v_out_loc}")
    
    # 3. Globalize
    # v_glob = Rz(rz) @ Ry(ry) @ Rx(rx) @ v_out_loc
    
    # Rx(45)
    # Rx(45) = [[1, 0, 0], [0, 0.707, -0.707], [0, 0.707, 0.707]]
    # Rx(45) @ [0, 0.707, 0.707]
    # x = 0
    # y = 0.707*0.707 - 0.707*0.707 = 0
    # z = 0.707*0.707 + 0.707*0.707 = 0.5 + 0.5 = 1
    v3 = np.array([0, 0, 1])
    
    # Ry(-pi)
    # Ry(-pi) = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
    # Ry(-pi) @ [0, 0, 1] = [0, 0, -1]
    
    # Rz(0)
    v_final = np.array([0, 0, -1])
    
    print(f"Manual Final Ray: {v_final}")
    
    # Compare with Optiland Trace
    print(f"\nMatch? L: {np.isclose(v_final[0], rays.L.item(), atol=1e-5)}, M: {np.isclose(v_final[1], rays.M.item(), atol=1e-5)}, N: {np.isclose(v_final[2], rays.N.item(), atol=1e-5)}")

if __name__ == "__main__":
    reproduce()
