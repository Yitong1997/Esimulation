import sys
import numpy as np

# Add src to path
sys.path.insert(0, r'd:\BTS\src')
sys.path.insert(0, r'd:\BTS\optiland-master')

# We need to mock classes because importing them might require complex dependencies
# However, importing HybridElementPropagator is necessary.
# Let's try to import real classes first if possible, if not found, use mocks.

try:
    from hybrid_optical_propagation.hybrid_element_propagator import HybridElementPropagator
    from hybrid_optical_propagation.data_models import GridSampling
except ImportError:
    print("Failed to import modules. Please check paths.")
    sys.exit(1)

class MockVector3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def to_array(self):
        return np.array([self.x, self.y, self.z])

class MockOpticalAxisState:
    def __init__(self, position, direction):
        self.position = position
        self.direction = direction

def test_negative_z_ray_sampling():
    print("Testing Negative Z Ray Sampling...")
    
    # Setup
    wavelength_um = 0.633
    propagator = HybridElementPropagator(wavelength_um=wavelength_um, num_rays=100)
    
    # 1. Create GridSampling
    grid_size = 128
    physical_size_mm = 10.0
    sampling_mm = physical_size_mm / grid_size
    grid_sampling = GridSampling(
        grid_size=grid_size,
        physical_size_mm=physical_size_mm,
        sampling_mm=sampling_mm
    )
    
    # 2. Create Amplitude and Phase
    # Flat phase -> rays should be parallel to geometric direction
    # Use gaussian amplitude
    y, x = np.ogrid[-grid_size/2:grid_size/2, -grid_size/2:grid_size/2]
    r2 = x*x + y*y
    amplitude = np.exp(-r2 / (20**2))
    
    # Flat phase
    phase = np.zeros((grid_size, grid_size))
    
    # 3. Create OpticalAxisState with NEGATIVE Z direction
    # Direction: -Z
    direction = MockVector3D(0.0, 0.0, -1.0)
    # Position doesn't matter much for direction vector test
    position = MockVector3D(0.0, 0.0, 100.0)
    entrance_axis = MockOpticalAxisState(position, direction)
    
    # 4. Call _sample_rays_from_wavefront
    rays = propagator._sample_rays_from_wavefront(
        amplitude=amplitude,
        phase=phase,
        grid_sampling=grid_sampling,
        entrance_axis=entrance_axis
    )
    
    # 5. Verify N component
    # Since phase is flat, L and M should be near 0.
    # N should be near -1 because of the -Z direction.
    
    print(f"Mean L: {np.mean(rays.L)}")
    print(f"Mean M: {np.mean(rays.M)}")
    print(f"Mean N: {np.mean(rays.N)}")
    
    if np.all(rays.N < 0):
        print("SUCCESS: All rays have negative N component.")
    else:
        print("FAILURE: Some rays have non-negative N component.")
        print(f"Max N: {np.max(rays.N)}")
        sys.exit(1)
        
    if np.isclose(np.mean(rays.N), -1.0, atol=1e-3):
        print("SUCCESS: Mean N is approximately -1.0.")
    else:
        print(f"FAILURE: Mean N is {np.mean(rays.N)}, expected -1.0")
        sys.exit(1)

    # 6. Verify Positive Z Case
    print("\nTesting Positive Z Ray Sampling...")
    direction_pos = MockVector3D(0.0, 0.0, 1.0)
    entrance_axis_pos = MockOpticalAxisState(position, direction_pos)
    
    rays_pos = propagator._sample_rays_from_wavefront(
        amplitude=amplitude,
        phase=phase,
        grid_sampling=grid_sampling,
        entrance_axis=entrance_axis_pos
    )
    
    print(f"Mean N (Pos): {np.mean(rays_pos.N)}")
    
    if np.all(rays_pos.N > 0):
        print("SUCCESS: All rays have positive N component for +Z.")
    else:
        print("FAILURE: Some rays have non-positive N component for +Z.")
        sys.exit(1)

if __name__ == "__main__":
    test_negative_z_ray_sampling()
