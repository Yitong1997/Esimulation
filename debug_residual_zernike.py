
import sys
import sys
import numpy as np
# import zernike - removed, using internal function

# Add project root to path
project_root = r'd:\BTS'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary modules
from tests.integration.离轴抛物面镜传输误差标准测试文件 import run_oap_test
import matplotlib.pyplot as plt

def zernike_projection(phase_grid, n_terms=15):
    """
    Simple Zernike decomposition.
    Assumes phase_grid is square and defined over a unit circle.
    Returns coefficients for first n_terms (Noll index).
    """
    ny, nx = phase_grid.shape
    y, x = np.mgrid[-1:1:ny*1j, -1:1:nx*1j]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    mask = r <= 1.0
    valid_phase = phase_grid[mask]
    
    # We will build a matrix of Zernike polynomials and solve least squares
    # This is slow but fine for debug
    
    # Noll indices mapping (n, m)
    # 1: 0,0 (Piston)
    # 2: 1,1 (Tilt X)
    # 3: 1,-1 (Tilt Y)
    # 4: 2,0 (Defocus)
    # 5: 2,-2 (Astigmatism Oblique)
    # 6: 2,2 (Astigmatism Vertical)
    # 7: 3,-1 (Coma Vertical)
    # 8: 3,1 (Coma Horizontal)
    # ... simplified for first few terms
    
    zernikes = []
    names = []
    
    # 1. Piston (n=0, m=0)
    zernikes.append(np.ones_like(valid_phase))
    names.append("Z1 (Piston)")
    
    # 2. Tilt X (n=1, m=1) -> r*cos(theta) = x
    zernikes.append(x[mask])
    names.append("Z2 (Tilt X)")
    
    # 3. Tilt Y (n=1, m=-1) -> r*sin(theta) = y
    zernikes.append(y[mask])
    names.append("Z3 (Tilt Y)")
    
    # 4. Defocus (n=2, m=0) -> 2r^2 - 1
    zernikes.append(2*r[mask]**2 - 1)
    names.append("Z4 (Defocus)")
    
    # 5. Astigmatism Oblique (n=2, m=-2) -> r^2 * sin(2theta)
    zernikes.append(r[mask]**2 * np.sin(2*theta[mask]))
    names.append("Z5 (Astig 45)")
    
    # 6. Astigmatism Vertical (n=2, m=2) -> r^2 * cos(2theta)
    zernikes.append(r[mask]**2 * np.cos(2*theta[mask]))
    names.append("Z6 (Astig 0/90)")
    
    # 7. Coma Vertical (n=3, m=-1) -> (3r^3 - 2r) sin(theta)
    zernikes.append((3*r[mask]**3 - 2*r[mask]) * np.sin(theta[mask]))
    names.append("Z7 (Coma Y)")
    
    # 8. Coma Horizontal (n=3, m=1) -> (3r^3 - 2r) cos(theta)
    zernikes.append((3*r[mask]**3 - 2*r[mask]) * np.cos(theta[mask]))
    names.append("Z8 (Coma X)")
    
    # 9. Spherical (n=4, m=0) -> 6r^4 - 6r^2 + 1
    zernikes.append(6*r[mask]**4 - 6*r[mask]**2 + 1)
    names.append("Z11 (Spherical)") # Often Z11 in Noll
    
    Z = np.array(zernikes).T
    coeffs, _, _, _ = np.linalg.lstsq(Z, valid_phase, rcond=None)
    
    return names, coeffs

def analyze_residual():
    print("Running OAP test to analyze residual...")
    
    # We need to capture the residual phase. 
    # Since run_oap_test returns the final wavefront, but we want the "Residual" used inside the propagation,
    # we might need to rely on the fact that the final phase *is* the residual if Pilot Beam removal isn't perfect,
    # OR we can inspect the final phase directly.
    # The current residual ~2.5 waves is likely dominated by the mismatch we want to find.
    #
    # Wait, the test function `run_oap_test` returns a result dict with `final_wavefront`.
    # But `final_wavefront.phase` IS the reconstructed phase. 
    # If the propagation was perfect, `final_wavefront.phase` (after piston/tilt removal) should be 0 for a perfect OAP focusing a perfect Gaussian?
    # No, OAP focuses to a point.
    # If we are looking at the Collimated->Focused path?
    # Test file says: RADIUS=2000, f=1000. Source w0=10 (large). 
    # Usually OAP is used to collimate or focus.
    # If the setup is focusing, the output wavefront should be spherical (converging).
    # The Pilot Beam should also be spherical (converging).
    # The difference (residual) should be 0.
    # So analyzing `final_wavefront.phase` vs `PilotBeam phase` (which is removed relative to plane?)
    #
    # Actually, `bts.simulate` returns a wavefront object.
    # Let's check `bts/wavefront.py` or similar to see what `.phase` contains.
    # Usually it's the phase relative to the current reference surface (sphere or plane).
    # For a converging beam, usually PROPER converts to a Spherical reference.
    # So `.phase` should be the residual from that Sphere.
    # Meaning `final_wavefront.phase` IS the aberration.
    
    result = run_oap_test(verbose=False)
    
    # Get the residual phase from the result (implied)
    # We need to access the wavefront object. 
    # The run_oap_test doesn't return the object directly in dict, but let's mod it or assume we can hack it
    # Actually run_oap_test calls bts.simulate which returns result.
    # The result object has get_final_wavefront().
    # But run_oap_test only returns the dict summary.
    # I will modify run_oap_test slightly or run the steps manually?
    #
    # Easier: Just import bts and re-run the setup manually here.
    
    import bts
    
    RADIUS_MM = 2000.0
    SURFACE_Y_MM = 100.0
    WAVELENGTH_UM = 0.633
    W0_MM = 10.0
    GRID_SIZE = 256
    
    system = bts.OpticalSystem("OAP Debug")
    system.add_parabolic_mirror(x=0.0, y=SURFACE_Y_MM, z=10000.0, radius=RADIUS_MM)
    
    source = bts.GaussianSource(wavelength_um=WAVELENGTH_UM, w0_mm=W0_MM, grid_size=GRID_SIZE)
    
    print("Simulating...")
    sim_result = bts.simulate(system, source)
    wf = sim_result.get_final_wavefront()
    
    phase = wf.phase
    amp = wf.amplitude
    
    # Mask usually valid where amplitude is significant
    threshold = 0.01 * np.max(amp)
    valid_mask = amp > threshold
    
    # Unwrap phase if needed (though it claims to be unwrapped)
    # bts wavefront phase is "non-wrapped real numbers"
    
    # Remove Piston/Tilt first (standard procedure)
    y, x = np.indices(phase.shape)
    y = y[valid_mask]
    x = x[valid_mask]
    p = phase[valid_mask]
    
    A = np.c_[np.ones_like(x), x, y]
    C, _, _, _ = np.linalg.lstsq(A, p, rcond=None)
    phase_removed_tilt = phase - (C[0] + C[1]*np.indices(phase.shape)[1] + C[2]*np.indices(phase.shape)[0])
    
    print(f"\nRMS after Piston/Tilt removal: {np.std(phase_removed_tilt[valid_mask]) / (2*np.pi) * 1000:.3f} milli-waves")
    
    # Zernike Decomposition on the residual
    # We need to crop/scale the valid region to unit circle for Zernike
    # Find bounding box of valid mask
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    phases_crop = phase_removed_tilt[rmin:rmax+1, cmin:cmax+1]
    
    # Resample to square for simple projection if needed, or just use grid
    # Our simple zernike_projection assumes filled circle in the square grid
    # Let's reshape/pad phases_crop to square
    h, w = phases_crop.shape
    size = max(h, w)
    square_phase = np.zeros((size, size))
    # Center it
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    square_phase[y_off:y_off+h, x_off:x_off+w] = phases_crop
    
    names, coeffs = zernike_projection(square_phase)
    
    print("\nDominant Zernike Terms (in waves):")
    # Coeffs are in radians. Convert to waves.
    for n, c in zip(names, coeffs):
        print(f"{n}: {c / (2*np.pi):.4f} waves")

if __name__ == '__main__':
    analyze_residual()
