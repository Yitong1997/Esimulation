import numpy as np
from scipy.spatial.transform import Rotation

def _avoid_exact_45_degrees(angle: float) -> float:
    return angle

def _normalize_vector(v):
    return v / np.linalg.norm(v)

def compute_rotation_matrix(chief_ray_direction):
    z_local = np.array(chief_ray_direction, dtype=np.float64)
    z_local = _normalize_vector(z_local)
    
    if abs(z_local[1]) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    
    x_local = np.cross(ref, z_local)
    x_local = _normalize_vector(x_local)
    y_local = np.cross(z_local, x_local)
    
    if y_local[1] < 0:
        x_local = -x_local
        y_local = -y_local
    
    R = np.column_stack([x_local, y_local, z_local])
    return R

def verify_transformation(incident_dir, surface_normal_global):
    print(f"Incident Dir: {incident_dir}")
    print(f"Surface Normal Global: {surface_normal_global}")
    
    # 1. R_beam
    R_beam = compute_rotation_matrix(incident_dir)
    print(f"R_beam:\n{R_beam}")
    
    # 2. R_surf
    # Construct R_surf such that Z axis points to surface_normal_global
    # Use compute_rotation_matrix for consistency or just Rotation to vector
    # Let's say R_surf maps (0,0,1) to surface_normal_global
    # We can use compute_rotation_matrix to get A valid frame for surface
    R_surf = compute_rotation_matrix(surface_normal_global)
    print(f"R_surf:\n{R_surf}")
    
    # 3. R_rel
    R_rel = R_beam.T @ R_surf
    print(f"R_rel:\n{R_rel}")
    
    # 4. Extract Euler angles (Current Implementation)
    euler_angles = Rotation.from_matrix(R_rel).as_euler('xyz', degrees=False)
    tilt_x = euler_angles[0]
    tilt_y = euler_angles[1]
    tilt_z = euler_angles[2]
    print(f"Computed Tilts (xyz): tx={np.degrees(tilt_x):.2f}, ty={np.degrees(tilt_y):.2f}, tz={np.degrees(tilt_z):.2f}")
    
    # 5. Verify consistency with Optiland assumption
    # Optiland assumption from ElementRaytracer doc: n = Ry(ry) @ Rx(rx) @ [0, 0, 1]
    # which is R_optiland = Ry(ry) @ Rx(rx)
    
    Rx = Rotation.from_euler('x', tilt_x, degrees=False).as_matrix()
    Ry = Rotation.from_euler('y', tilt_y, degrees=False).as_matrix()
    
    # If optiland applies Rx then Ry (intrinsic?) or Ry then Rx?
    # ElementRaytracer._direction_to_rotation_angles says: n = Ry(ry) @ Rx(rx) @ [0, 0, 1]
    # So R_reconstructed = Ry @ Rx
    
    R_reconstructed = Ry @ Rx
    
    # Check if R_reconstructed * [0,0,1] matches R_rel * [0,0,1]
    # i.e. does the normal match in the local frame?
    
    n_local_actual = R_rel @ [0, 0, 1]
    n_local_reconstructed = R_reconstructed @ [0, 0, 1]
    
    print(f"Normal Local Actual: {n_local_actual}")
    print(f"Normal Local Reconstructed: {n_local_reconstructed}")
    
    error = np.linalg.norm(n_local_actual - n_local_reconstructed)
    print(f"Reconstruction Error: {error:.2e}")
    if error > 1e-6:
        print("FAIL: Significant error in normal reconstruction.")
    else:
        print("PASS: Normal reconstruction is close.")

    # Check if extracting 'yx' order is better?
    # Note: intrinsic 'yxz' means Ry(y) @ Rx(x) @ Rz(z)
    # If optiland effectively does Ry @ Rx, then 'yxz' is the correct intrinsic sequence 
    # if we assume z rotation is 0.
    
    euler_angles_yx = Rotation.from_matrix(R_rel).as_euler('yxz', degrees=False)
    
    ty_yx = euler_angles_yx[0]
    tx_yx = euler_angles_yx[1]
    tz_yx = euler_angles_yx[2]
    
    print(f"Computed Tilts (yxz): tx={np.degrees(tx_yx):.2f}, ty={np.degrees(ty_yx):.2f}, tz={np.degrees(tz_yx):.2f}")

    Rx_yx = Rotation.from_euler('x', tx_yx, degrees=False).as_matrix()
    Ry_yx = Rotation.from_euler('y', ty_yx, degrees=False).as_matrix()
    R_rec_yx = Ry_yx @ Rx_yx
    
    n_local_rec_yx = R_rec_yx @ [0, 0, 1]
    error_yx = np.linalg.norm(n_local_actual - n_local_rec_yx)
    print(f"Reconstruction Error (yxz order): {error_yx:.2e}")
    
print("-" * 50)
print("Test Case 1: Incident Z, Surface Normal Z (Identity)")
verify_transformation((0,0,1), (0,0,1))

print("-" * 50)
print("Test Case 2: Incident Z, Surface Normal tilted 45 deg around X")
normal = (0, -np.sin(np.pi/4), np.cos(np.pi/4)) # Tilted back?
verify_transformation((0,0,1), normal)

print("-" * 50)
print("Test Case 3: Incident X, Surface Normal -X (90 deg rot)")
verify_transformation((1,0,0), (-1,0,0))

print("-" * 50)
print("Test Case 4: Incident (0, 1, 1), Surface Normal (0, -1, 1)")
verify_transformation(_normalize_vector(np.array([0, 1, 1])), _normalize_vector(np.array([0, -1, 1])))

