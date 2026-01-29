
import numpy as np
from scipy.spatial.transform import Rotation as R

def optiland_style_matrix(rx, ry, rz):
    """
    Copy of logic from optiland.coordinate_system.CoordinateSystem.get_rotation_matrix
    R = Rz @ Ry @ Rx
    """
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    return Rz @ Ry @ Rx

def processor_style_matrix(rx, ry, rz):
    """
    Copy of logic from sequential_system.coordinate_system.CoordinateBreakProcessor.rotation_matrix_xyz
    Identical to optiland style in code structure
    """
    c, s = np.cos(rx), np.sin(rx)
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c]
    ])
    
    c, s = np.cos(ry), np.sin(ry)
    Ry = np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ])
    
    c, s = np.cos(rz), np.sin(rz)
    Rz = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    return Rz @ Ry @ Rx

def check_consistency():
    print("Checking Rotation Matrix Consistency...")
    
    # Test Case 1: 180 degree rotation around X (The case in question)
    rx, ry, rz = np.pi, 0, 0
    print(f"\nTest Case 1: rx={rx}, ry={ry}, rz={rz}")
    
    # Optiland / Processor Matrix
    m_opt = optiland_style_matrix(rx, ry, rz)
    print("Optiland Matrix:")
    print(m_opt)
    
    # Scipy Matrix (Intrinsic 'xyz')
    # Use degrees for easier reading if needed, but R.from_euler uses radians by default
    r_scipy = R.from_euler('xyz', [rx, ry, rz])
    m_scipy = r_scipy.as_matrix()
    print("Scipy 'xyz' Matrix:")
    print(m_scipy)
    
    diff = np.abs(m_opt - m_scipy).max()
    print(f"Max Difference: {diff}")
    
    if diff < 1e-10:
        print("✅ CONSISTENT")
    else:
        print("❌ INCONSISTENT")

    # Test Case 2: Mixed Rotation
    rx, ry, rz = 0.1, 0.2, 0.3
    print(f"\nTest Case 2: rx={rx}, ry={ry}, rz={rz}")
    
    m_opt = optiland_style_matrix(rx, ry, rz)
    m_scipy = R.from_euler('xyz', [rx, ry, rz]).as_matrix()
    
    diff = np.abs(m_opt - m_scipy).max()
    print(f"Max Difference: {diff}")
    
    if diff < 1e-10:
        print("✅ CONSISTENT")
    else:
        print("❌ INCONSISTENT")
        
    print("\n---------------------------------------------------")
    print("Checking Euler Angle Extraction Round-Trip")
    # Simulate the pipeline:
    # 1. ZMX Processor creates matrix M (using processor_style_matrix)
    # 2. Converter uses scipy 'xyz' to get angles: ang = R.from_matrix(M).as_euler('xyz')
    # 3. Optiland builds matrix M' from ang (using optiland_style_matrix)
    
    # Original rotations (Zemax inputs)
    rx_in, ry_in, rz_in = np.deg2rad(10), np.deg2rad(20), np.deg2rad(30)
    
    # 1. Active Matrix
    M_global = processor_style_matrix(rx_in, ry_in, rz_in)
    
    # 2. Extraction
    r_extract = R.from_matrix(M_global)
    extracted_euler = r_extract.as_euler('xyz')
    print(f"Input Euler: {[rx_in, ry_in, rz_in]}")
    print(f"Extracted Euler: {extracted_euler}")
    
    # 3. Reconstruction in Optiland
    M_reconstructed = optiland_style_matrix(*extracted_euler)
    
    diff_roundtrip = np.abs(M_global - M_reconstructed).max()
    print(f"Round Trip Matrix Difference: {diff_roundtrip}")
    if diff_roundtrip < 1e-10:
         print("✅ ROUND TRIP SUCCESSFUL")
    else:
         print("❌ ROUND TRIP FAILED")

if __name__ == "__main__":
    check_consistency()
