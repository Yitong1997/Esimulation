
import sys
import os
import numpy as np

# Add project root to path
# Assuming this file is in d:\BTS\tests\unit\
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.hybrid_optical_propagation.hybrid_element_propagator import HybridElementPropagator
    from src.hybrid_optical_propagation.data_models import GridSampling, PropagationState, PilotBeamParams
    from sequential_system.coordinate_tracking import OpticalAxisState, Position3D, RayDirection
except ImportError as e:
    print(f"Import failed: {e}")
    # Fallback for manual run location
    sys.path.append(os.path.join(project_root, 'src'))
    from hybrid_optical_propagation.hybrid_element_propagator import HybridElementPropagator
    from hybrid_optical_propagation.data_models import GridSampling, PropagationState, PilotBeamParams
    # Mocking sequential_system if not available or complex import
    from sequential_system.coordinate_tracking import OpticalAxisState, Position3D, RayDirection

def test_central_ray_zero_tilt():
    """Verify that a perfect flat wavefront results in a central ray with exactly 0,0 direction."""
    
    wavelength_um = 0.5
    grid_size = 512
    physical_size_mm = 10.0
    
    print(f"Testing Grid Size: {grid_size}, Size: {physical_size_mm}mm")
    
    # 1. Setup Grid and Wavefront
    sampling_mm = physical_size_mm / grid_size
    grid_sampling = GridSampling(grid_size, physical_size_mm, sampling_mm)
    
    # Flat wavefront (Approximating a perfect plane wave)
    amplitude = np.ones((grid_size, grid_size))
    # Phase is exactly 0 everywhere
    phase = np.zeros((grid_size, grid_size))
    
    # 2. Setup Propagator
    propagator = HybridElementPropagator(wavelength_um, num_rays=100)
    
    # 3. Setup Optical Axis (On-axis)
    # Using mock objects if real ones are complex
    try:
        entrance_axis = OpticalAxisState(
            position=Position3D(0,0,0),
            direction=RayDirection(0,0,1),
            path_length=0
        )
    except:
        # Minimal mock if needed
        class MockPosition:
            def to_array(self): return np.array([0.,0.,0.])
        class MockAxis:
            position = MockPosition()
        entrance_axis = MockAxis()
    
    # 4. Sample Rays
    # We call the internal method _sample_rays_from_wavefront directly
    rays = propagator._sample_rays_from_wavefront(
        amplitude=amplitude,
        phase=phase,
        grid_sampling=grid_sampling,
        entrance_axis=entrance_axis,
        pilot_beam_params=None # No pilot beam for this test
    )
    
    x = np.asarray(rays.x)
    y = np.asarray(rays.y)
    L = np.asarray(rays.L)
    M = np.asarray(rays.M)
    
    # 5. Verify Center Ray
    # Find the ray closest to 0,0
    r_sq = x**2 + y**2
    center_idx = np.argmin(r_sq)
    
    print(f"Center Ray Index: {center_idx}")
    print(f"Center Ray Position: x={x[center_idx]}, y={y[center_idx]}")
    print(f"Center Ray Direction: L={L[center_idx]}, M={M[center_idx]}")
    
    # Check Position is EXACTLY 0 (due to grid alignment)
    # For even grid size, center index n//2 coordinate should be exactly 0.0
    if abs(x[center_idx]) > 1e-12:
        print(f"FAILED: Center X should be 0, got {x[center_idx]}")
    else:
        print("PASSED: Center X is 0")
        
    if abs(y[center_idx]) > 1e-12:
        print(f"FAILED: Center Y should be 0, got {y[center_idx]}")
    else:
         print("PASSED: Center Y is 0")
    
    # Check Direction is EXACTLY 0 (due to flat phase and no interpolation)
    if abs(L[center_idx]) > 1e-10:
        print(f"FAILED: Center L should be 0, got {L[center_idx]}")
    else:
        print("PASSED: Center L is 0")
        
    if abs(M[center_idx]) > 1e-10:
        print(f"FAILED: Center M should be 0, got {M[center_idx]}")
    else:
        print("PASSED: Center M is 0")

if __name__ == "__main__":
    test_central_ray_zero_tilt()
