
import numpy as np

def check_coordinates():
    n = 512
    sampling_mm = 0.1
    physical_size_mm = n * sampling_mm
    half_size = physical_size_mm / 2
    
    print(f"Grid Size: {n}")
    print(f"Sampling: {sampling_mm} mm")
    print(f"Physical Size: {physical_size_mm} mm")
    
    # 1. Correct (PROPER compatible) method
    coords_arange = (np.arange(n) - n // 2) * sampling_mm
    
    # 2. Incorrect (Linspace) method used in code
    coords_linspace = np.linspace(-half_size, half_size, n)
    
    print(f"\nMethod 1 (arange):")
    print(f"  Start: {coords_arange[0]}")
    print(f"  End: {coords_arange[-1]}")
    print(f"  Step: {coords_arange[1] - coords_arange[0]}")
    print(f"  Contains 0.0? {np.any(np.abs(coords_arange) < 1e-10)}")
    if np.any(np.abs(coords_arange) < 1e-10):
        print(f"  Index of 0.0: {np.argmin(np.abs(coords_arange))}")
        
    print(f"\nMethod 2 (linspace):")
    print(f"  Start: {coords_linspace[0]}")
    print(f"  End: {coords_linspace[-1]}")
    print(f"  Step: {coords_linspace[1] - coords_linspace[0]}")
    print(f"  Contains 0.0? {np.any(np.abs(coords_linspace) < 1e-10)}")
    
    diff = coords_linspace - coords_arange
    print(f"\nMax difference: {np.max(np.abs(diff))}")

if __name__ == "__main__":
    check_coordinates()
