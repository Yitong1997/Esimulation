
import sys
import os
import numpy as np

# Add src and optiland to path
sys.path.insert(0, os.path.abspath("optiland-master"))

from optiland.rays import RealRays
import optiland.backend as be

def test_rotation_precision():
    print("Testing Rotation Precision...")
    
    # Create a ray at [0, 1, 0]
    rays = RealRays(x=[0], y=[1], z=[0], L=[0], M=[0], N=[1], intensity=[1], wavelength=[0.55])
    
    print(f"Initial Position: x={rays.x[0]:.20f}, y={rays.y[0]:.20f}, z={rays.z[0]:.20f}")
    
    # Rotate by 180 degrees (pi) around X axis
    angle = np.pi
    print(f"Rotating by pi ({angle:.20f}) around X axis")
    
    rays.rotate_x(angle)
    
    print(f"Final Position:   x={rays.x[0]:.20f}, y={rays.y[0]:.20f}, z={rays.z[0]:.20f}")
    
    # Expected: y -> -1, z -> 0
    # Check z
    z_val = float(rays.z[0])
    print(f"Z error: {z_val}")
    
    if abs(z_val) > 1e-15:
        print("FAIL: Significant floating point error detected.")
    else:
        print("PASS: Precision is acceptable.")

if __name__ == "__main__":
    test_rotation_precision()
