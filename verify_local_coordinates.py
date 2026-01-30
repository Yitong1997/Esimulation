
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from hybrid_optical_propagation.hybrid_element_propagator import HybridElementPropagator
from sequential_system.coordinate_tracking import OpticalAxisState, RayDirection, Position3D
from sequential_system.coordinate_system import GlobalSurfaceDefinition
from scipy.spatial.transform import Rotation

def create_rotation_matrix(rx, ry, rz):
    return Rotation.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()

def run_test_case(case_name, entrance_dir, surface_orientation_euler):
    print(f"\n--- Test Case: {case_name} ---")
    
    # 1. Setup Entrance Axis
    entrance_pos = np.array([0, 0, 0])
    entrance_axis = OpticalAxisState(
        position=Position3D.from_array(entrance_pos),
        direction=RayDirection.from_array(entrance_dir),
        path_length=0.0
    )
    print(f"Entrance Direction (Global): {entrance_dir}")
    
    # 2. Setup Surface
    # surface_orientation_euler is (rx, ry, rz) in degrees
    surf_rot = create_rotation_matrix(*surface_orientation_euler)
    surface = GlobalSurfaceDefinition(
        index=1,
        surface_type="mirror",
        vertex_position=np.array([0, 0, 0]),
        orientation=surf_rot,
        thickness=0,
        radius=np.inf,
        material="mirror",
        semi_aperture=5.0, # aperture=10 -> semi=5
        is_mirror=True
    )
    print(f"Surface Orientation (Global Euler XYZ deg): {surface_orientation_euler}")
    
    # 3. Create HybridElementPropagator instance (mocking)
    propagator = HybridElementPropagator(wavelength_um=0.55)
    
    # 4. Call _create_surface_definition (accessing private method for verification)
    # We need a dummy exit axis (not used for tilt calculation in the new logic)
    exit_axis = OpticalAxisState(position=Position3D(0,0,1), direction=RayDirection(0,0,1), path_length=0.0) 
    
    try:
        surface_def = propagator._create_surface_definition(surface, entrance_axis, exit_axis)
        
        print("\n[Resulting SurfaceDefinition Properties]")
        print(f"tilt_x (rad): {surface_def.tilt_x:.6f} -> {np.degrees(surface_def.tilt_x):.2f} deg")
        print(f"tilt_y (rad): {surface_def.tilt_y:.6f} -> {np.degrees(surface_def.tilt_y):.2f} deg")
        
        # Verify interpretation
        # If Entrance is Z (0,0,1) and Surface is 45 deg X, we expect tilt_x ~ 45.
        # If Entrance is 45 deg Y, Surface is 0 deg, we expect relative tilt.
        
        return surface_def
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Case 1: Standard Incidence
    # Incident: +Z
    # Surface: 45 deg X tilt (reflects +Z to +Y)
    run_test_case(
        "Standard Incidence",
        entrance_dir=np.array([0, 0, 1]),
        surface_orientation_euler=(45, 0, 0)
    )
    
    # Case 2: Tilted Incidence
    # Incident: +Y (0, 1, 0) -> Corresponds to 90 deg X rotation of beam? No, beam dir.
    # Surface: 45 deg X tilt
    # Relative angle calculation:
    # Beam along +Y. Surface normal (from 45 deg X) is (0, -sin(45), cos(45)).
    # Angle between Beam and Normal?
    # Beam (0, 1, 0). Normal (0, -0.707, 0.707). Dot = -0.707 (135 deg).
    # This should result in a different tilt in the Local Frame.
    run_test_case(
        "Tilted Incidence (Beam along +Y)",
        entrance_dir=np.array([0, 1, 0]),
        surface_orientation_euler=(45, 0, 0)
    )

    # Case 3: Matched Tilt (Normal Incidence)
    # Incident: Tilted 45 deg X (0, 0.707, 0.707) -- wait, that's beam direction.
    # Surface: Tilted 45 deg X. Normal is (0, -0.707, 0.707).
    # If beam is along normal...
    # Normal of surface (45 deg X rotation of (0,0,1)) -> (0, -sin(45), cos(45)).
    # Let's set beam opposite to normal: (0, sin(45), -cos(45)).
    # Then it should be normal incidence --> tilt ~ 0.
    
    # Surface Normal for 45 deg X:
    # R_x(45) @ [0,0,1] = [0, -sin45, cos45] (Global Z axis rotated)
    # Actually standard surface normal is +Z in local.
    # If we rotate surface by 45 X, normal becomes [0, -s, c].
    
    # Let's try Beam = Surface Axis.
    # Beam along [0, -0.707, 0.707].
    # Surface oriented [45, 0, 0].
    # Then beam is parallel to surface normal (Z).
    # Expected relative tilt: 0.
     
    s45 = np.sin(np.radians(45))
    c45 = np.cos(np.radians(45))
    
    # Orientation 45 deg X. "Z" axis points to (0, -s, c).
    run_test_case(
        "Matched Orientation (Normal Incidence)",
        entrance_dir=np.array([0, -s45, c45]),  # Along the surface Z-axis
        surface_orientation_euler=(45, 0, 0)
    )

    # 4. Test Case: OAP2 Configuration (-Z Incidence, 180 deg Y-Tilt)
    # Incident: -Z (0, 0, -1)
    # Surface: 180 deg Y tilt (Z -> -Z)
    # Expected relative tilt: 0 (since both flipped)
    run_test_case(
        "OAP2 Configuration (-Z Incidence, 180 deg Y-Tilt)",
        entrance_dir=np.array([0, 0, -1]),
        surface_orientation_euler=(0, 180, 0)
    )

if __name__ == "__main__":
    main()
