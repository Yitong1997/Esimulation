
import numpy as np
from scipy.spatial.transform import Rotation

def compute_rotation_matrix(chief_ray_direction):
    z_local = np.array(chief_ray_direction, dtype=np.float64)
    z_local = z_local / np.linalg.norm(z_local)
    
    if abs(z_local[1]) > 0.999999: # Updated threshold
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
        
    x_local = np.cross(ref, z_local)
    x_local = x_local / np.linalg.norm(x_local)
    y_local = np.cross(z_local, x_local)
    
    # Fix Y projection
    if y_local[1] < 0:
        x_local = -x_local
        y_local = -y_local
        
    R = np.column_stack([x_local, y_local, z_local])
    return R

def unit_vector(v):
    return v / np.linalg.norm(v)

print("--- Verification ---")

# Case 1: Retrograde Propagation (Light -Z)
beam_dir = np.array([0, 0, -1])
R_beam = compute_rotation_matrix(beam_dir)
print(f"Beam Direction (Global): {beam_dir}")
print(f"R_beam (Local->Global):\n{R_beam}")

# Surface: Default Orientation (Normal +Z)
R_surf = np.eye(3) # Global Surf Z = (0,0,1)

# R_rel calculation
# Surface in Beam Frame?
R_rel = R_beam.T @ R_surf
print(f"R_rel:\n{R_rel}")

# Euler Angles
euler = Rotation.from_matrix(R_rel).as_euler('yxz', degrees=True)
print(f"Euler (y, x, z) deg: {euler}")
# Should be 180, 0, 0

# Optiland Surface Axis in Beam Frame
# Default Optiland Surf Z is (0,0,1)
# Apply Tilt R_tilt = Ry(180)
# Axis_local_tilted = R_tilt @ (0,0,1)
R_tilt = Rotation.from_euler('yxz', euler, degrees=True).as_matrix()
axis_in_beam_frame = R_tilt @ np.array([0, 0, 1])
print(f"Surface Z-Axis in Beam Frame: {axis_in_beam_frame}")

# Transform back to Global Frame
# Axis_global = R_beam @ Axis_in_beam_frame
axis_global = R_beam @ axis_in_beam_frame
print(f"Surface Z-Axis in Global Frame: {axis_global}")

# Check consistency
# Input Surface Z was (0,0,1)
print(f"Original Surface Z (Global): {R_surf[:, 2]}")
print(f"Is preserved? {np.allclose(axis_global, R_surf[:, 2], atol=1e-6)}")
