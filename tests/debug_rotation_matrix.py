
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from wavefront_to_rays.element_raytracer import compute_rotation_matrix

def test_rotation_smoothness():
    output = []
    output.append("Testing Rotation Matrix Smoothness around Z axis (Small Angles)...")
    
    # Test small rotations around X axis
    thetas = np.linspace(0, 1e-4, 11) # 0 to 0.1 mrad
    
    output.append(f"{'Theta (rad)':<15} {'R[1,1] (cos)':<15} {'R[1,2]':<15} {'R[2,1]':<15} {'R[2,2]':<15}")
    for theta in thetas:
        direction = (0, np.sin(theta), np.cos(theta))
        R = compute_rotation_matrix(direction)
        output.append(f"{theta:<15.2e} {R[1,1]:<15.6f} {R[1,2]:<15.6e} {R[2,1]:<15.6e} {R[2,2]:<15.6f}")

    output.append("\nTesting Rotation Matrix Smoothness around Y axis (Small Angles)...")
    
    output.append(f"{'Theta (rad)':<15} {'R[0,0]':<15} {'R[0,2]':<15} {'R[2,0]':<15} {'R[2,2]':<15}")
    for theta in thetas:
        direction = (np.sin(theta), 0, np.cos(theta))
        R = compute_rotation_matrix(direction)
        output.append(f"{theta:<15.2e} {R[0,0]:<15.6f} {R[0,2]:<15.6e} {R[2,0]:<15.6e} {R[2,2]:<15.6f}")
        
    output.append("\n------------------------------------------------------------")
    output.append("Checking for Discontinuity at z_local[1] = 0.9 switch point")
    output.append("------------------------------------------------------------")
    
    theta_crit = np.arcsin(0.9)
    eps = 1e-6
    thetas_crit = [theta_crit - eps, theta_crit, theta_crit + eps]
    
    output.append(f"{'Theta (rad)':<15} {'Ref Axis':<10} {'R[0,0]':<12} {'R[1,1]':<12} {'R[2,2]':<12}")
    for theta in thetas_crit:
        direction = (0, np.sin(theta), np.cos(theta))
        R = compute_rotation_matrix(direction)
        ref_axis = "X" if abs(direction[1]) > 0.9 else "Y"
        output.append(f"{theta:<15.6f} {ref_axis:<10} {R[0,0]:<12.4f} {R[1,1]:<12.4f} {R[2,2]:<12.4f}")

    output.append("\n------------------------------------------------------------")
    output.append("Checking for Discontinuity at y_local[1] < 0 flip condition")
    output.append("------------------------------------------------------------")

    output.append("Scanning around Z = (0, 1, 0) [Vertical Up]...")
    theta_up = np.pi/2
    eps = 1e-4
    thetas_scan = np.linspace(theta_up - eps, theta_up + eps, 11)
    
    output.append(f"{'Theta (rad)':<15} {'Dir Y':<15} {'R[0,1]':<15} {'R[1,1] (y_y)':<15}")
    
    for theta in thetas_scan:
        direction = (0, np.sin(theta), np.cos(theta))
        R = compute_rotation_matrix(direction)
        y_local = R[:, 1]
        output.append(f"{theta:<15.6f} {direction[1]:<15.6f} {y_local[0]:<15.6f} {y_local[1]:<15.6e}")

    output.append("\nScanning around Z = (0, -1, 0) [Vertical Down]...")
    theta_down = -np.pi/2
    thetas_scan = np.linspace(theta_down - eps, theta_down + eps, 11)
    
    output.append(f"{'Theta (rad)':<15} {'Dir Y':<15} {'R[0,1]':<15} {'R[1,1] (y_y)':<15}")
    for theta in thetas_scan:
        direction = (0, np.sin(theta), np.cos(theta))
        R = compute_rotation_matrix(direction)
        y_local = R[:, 1]
        output.append(f"{theta:<15.6f} {direction[1]:<15.6f} {y_local[0]:<15.6f} {y_local[1]:<15.6e}")

    with open('d:/BTS/tests/debug_results.log', 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))

if __name__ == "__main__":
    test_rotation_smoothness()
