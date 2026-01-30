

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import bts
from sequential_system.coordinate_system import CoordinateBreakProcessor

def test_manual_tilt_z_support():
    """Verify that OpticalSystem.add_surface supports tilt_z and computes orientation correctly."""
    
    system = bts.OpticalSystem("Test Tilt Z")
    
    # 1. Add a surface with pure Tilt Z = 90 deg
    system.add_surface(
        z=100.0,
        tilt_z=90.0,
        radius=np.inf
    )
    
    assert system.num_surfaces == 1
    
    # Check SurfaceDefinition
    surf_def = system._surfaces[0]
    assert surf_def.tilt_z == 90.0
    
    # Check GlobalSurfaceDefinition
    global_surf = system.get_global_surfaces()[0]
    
    # Expected Orientation: Rz(90) @ Ry(0) @ Rx(0)
    # Rz(90) = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    expected_matrix = CoordinateBreakProcessor.rotation_matrix_xyz(0, 0, np.radians(90))
    
    print("\nExpected Matrix (Rz=90):")
    print(expected_matrix)
    print("\nActual Matrix:")
    print(global_surf.orientation)
    
    np.testing.assert_allclose(
        global_surf.orientation, 
        expected_matrix, 
        atol=1e-7, 
        err_msg="Orientation matrix mismatch for pure Tilt Z"
    )
    
    # 2. Add a surface with combined Tilt X=90, Tilt Z=90
    # Order should be Rz @ Ry @ Rx
    # Rx(90): [[1,0,0],[0,0,-1],[0,1,0]]
    # Rz(90): [[0,-1,0],[1,0,0],[0,0,1]]
    # R = [[0,-1,0],[1,0,0],[0,0,1]] @ [[1,0,0],[0,0,-1],[0,1,0]]
    #   = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    
    system.add_flat_mirror(
        z=200.0,
        tilt_x=90.0,
        tilt_z=90.0
    )
    
    assert system.num_surfaces == 2
    global_surf_2 = system.get_global_surfaces()[1]
    
    expected_matrix_2 = CoordinateBreakProcessor.rotation_matrix_xyz(
        np.radians(90), 0, np.radians(90)
    )
    
    print("\nExpected Matrix (Tx=90, Tz=90):")
    print(expected_matrix_2)
    print("\nActual Matrix:")
    print(global_surf_2.orientation)
    
    np.testing.assert_allclose(
        global_surf_2.orientation, 
        expected_matrix_2, 
        atol=1e-7, 
        err_msg="Orientation matrix mismatch for combined Tilt X/Z"
    )

if __name__ == "__main__":
    try:
        test_manual_tilt_z_support()
        print("\nTest PASSED: tilt_z support verified.")
    except Exception as e:
        print(f"\nTest FAILED: {e}")
        import traceback
        traceback.print_exc()
