
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from wavefront_to_rays.element_raytracer import compute_rotation_matrix

def test_rotation_matrix_for_y_axis():
    print("--- Testing compute_rotation_matrix for +Y Incidence ---")
    
    # 1. Start with Incidence along +Y (0, 1, 0)
    # This simulates the "Entrance Chief Ray Direction" being +Y.
    entrance_dir_global = np.array([0, 1, 0], dtype=np.float64)
    
    print(f"Entrance Direction (Global): {entrance_dir_global}")
    
    # 2. Compute Rotation Matrix
    # R maps Local -> Global.
    # Columns are X_local, Y_local, Z_local in Global coords.
    R = compute_rotation_matrix(tuple(entrance_dir_global))
    
    print("\nRotation Matrix R (Cols = Local Axes):")
    print(R)
    
    print(f"X_local (Col 0): {R[:, 0]}")
    print(f"Y_local (Col 1): {R[:, 1]}")
    print(f"Z_local (Col 2): {R[:, 2]}")
    
    # 3. Simulate Reflection 45 deg X
    # Incident +Y. Reflects to +Z.
    exit_dir_global = np.array([0, 0, 1], dtype=np.float64)
    print(f"\nExit Direction (Global): {exit_dir_global}")
    
    # 4. Compute Exit Direction in Local Frame
    # Local = R.T @ Global
    exit_dir_local = R.T @ exit_dir_global
    
    print(f"\nExit Direction (Local) = R.T @ Exit_Global:")
    print(exit_dir_local)
    
    # Check expectation
    expected = np.array([0, 1, 0]) # User says this should be (0, 1, 0)
    
    if np.allclose(exit_dir_local, expected, atol=1e-6):
        print("\n✅ MATCHES User Expectation (0, 1, 0)")
    else:
        print(f"\n❌ DOES NOT MATCH User Expectation. Got {exit_dir_local}")

    # 5. Proposed Fix Verification
    # If we change ref to (0, 0, 1) (Global Z)
    # Then:
    # z_local = (0, 1, 0)
    # ref = (0, 0, 1)
    # x_local = cross(ref, z_local) = Z x Y = (-1, 0, 0)
    # y_local = cross(z_local, x_local) = Y x (-X) = Z = (0, 0, 1)
    # y_local[1] = 0. Not < 0. No flip.
    # R = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
    # R.T = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
    # Local Exit = R.T @ (0, 0, 1)
    # X: [-1, 0, 0] . Z = 0
    # Y: [0, 0, 1] . Z = 1
    # Z: [0, 1, 0] . Z = 0
    # Result: (0, 1, 0).
    # This would work.

if __name__ == "__main__":
    test_rotation_matrix_for_y_axis()
