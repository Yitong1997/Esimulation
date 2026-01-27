
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath('src'))

from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
from optiland.rays import RealRays
from hybrid_optical_propagation.data_models import PilotBeamParams, GridSampling

def test_sign_consistency():
    print("Testing Sign Consistency for Diverging Beam (R > 0)...")
    
    # 1. Define Diverging Beam
    wavelength_um = 0.6328
    wavelength_mm = wavelength_um * 1e-3
    w0_mm = 1.0
    z0_mm = 100.0 # Waist is 100mm behind current position
    # R at z=100 will be approx 100 mm (Diverging)
    
    pilot = PilotBeamParams.from_gaussian_source(wavelength_um, w0_mm, -z0_mm)
    print(f"Pilot R: {pilot.curvature_radius_mm:.4f} mm")
    
    if pilot.curvature_radius_mm <= 0:
        print("ERROR: Expected Positive R for Diverging Beam")
        return

    # 2. Calculate Theoretical Pilot Phase at r=1.0mm
    r = 1.0
    k = 2 * np.pi / wavelength_mm
    pilot_phase_theoretical = k * r**2 / (2 * pilot.curvature_radius_mm)
    print(f"Theoretical Pilot Phase at r={r}mm: {pilot_phase_theoretical:.4f} rad (Positive)")

    # 3. Simulate Ray Trace OPD
    # Diverging geometric rays from point source at z=-100 to plane z=0
    # Center ray length L0 = 100.
    # Edge ray at r=1: L1 = sqrt(100^2 + 1^2) approx 100 + 0.005
    # OPD = L1 - L0 = 0.005 mm
    
    opd_geometric_mm = np.sqrt(z0_mm**2 + r**2) - z0_mm
    print(f"Geometric OPD at r={r}mm: {opd_geometric_mm:.6f} mm (Positive)")
    
    # 4. Check Code Formula for Pilot OPD
    # hybrid_element_propagator.py line 327: pilot_opd_mm = -r_sq_out / (2 * R_out)
    pilot_opd_code_mm = - (r**2) / (2 * pilot.curvature_radius_mm)
    print(f"Code Pilot OPD calculation: {pilot_opd_code_mm:.6f} mm (Negative)")
    
    if np.sign(pilot_opd_code_mm) != np.sign(opd_geometric_mm):
        print(">>> CONFIRMED ISSUE 1: Code Pilot OPD sign is OPPOSITE to Geometric OPD.")
    else:
        print(">>> Issue 1 not reproduced?")

    # 5. Check Reconstructor Sign
    # reconstructor.py line 336: phase = -2 * pi * opd
    # If we feed geometric OPD (Positive)
    recon_phase = -2 * np.pi * (opd_geometric_mm / wavelength_mm)
    print(f"Reconstructed Phase from Geometric OPD: {recon_phase:.4f} rad (Negative)")
    
    if np.sign(recon_phase) != np.sign(pilot_phase_theoretical):
        print(">>> CONFIRMED ISSUE 2: Reconstructor inverts phase (Positive OPD -> Negative Phase).")
        print("    This contradicts Pilot Phase (Positive for Positive R).")
    else:
        print(">>> Issue 2 not reproduced?")

if __name__ == "__main__":
    test_sign_consistency()
