
import sys
import numpy as np

# Add project root to path
project_root = r'd:\BTS'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tests.integration.离轴抛物面镜传输误差标准测试文件 import run_oap_test
import bts

print("Running OAP test to inspect Phase and Pilot Beam...")

# Run test with z_mm=1000.0 (focal length configuration, output should be nearly collimated if source was point)
# But source is w0=10mm (collimated). So output should be focusing.
result = run_oap_test(verbose=False, z_mm=1000.0)

print("\n--- DEBUG INSPECTION ---")
if result['success'] or True: # Inspect regardless of success
    # Access via bts not explicitly exposed, but run_oap_test returns dict summary.
    # We need the direct result object.
    # We will simulate manually using the same parameters as run_oap_test.
    
    WAVELENGTH_UM = 0.633
    W0_MM = 10.0
    RADIUS_MM = 2000.0
    SURFACE_Y_MM = 100.0
    Z_MM = 1000.0
    GRID_SIZE = 256
    
    system = bts.OpticalSystem("OAP Debug")
    system.add_parabolic_mirror(x=0.0, y=SURFACE_Y_MM, z=Z_MM, radius=RADIUS_MM)
    source = bts.GaussianSource(wavelength_um=WAVELENGTH_UM, w0_mm=W0_MM, grid_size=GRID_SIZE)
    
    sim_result = bts.simulate(system, source, verbose=False)
    wf = sim_result.get_final_wavefront()
    
    # 1. Pilot Beam Curvature
    R = wf.pilot_beam.curvature_radius_mm
    print(f"Pilot Beam Curvature R: {R:.4f} mm")
    
    # 2. Phase Statistics
    phase = wf.phase
    amp = wf.amplitude
    valid_mask = amp > 0.01 * np.max(amp)
    valid_phase = phase[valid_mask]
    
    p_min, p_max = np.min(valid_phase), np.max(valid_phase)
    p_pv = p_max - p_min
    p_rms = np.std(valid_phase)
    
    print(f"Phase PV: {p_pv:.4f} rad ({p_pv/(2*np.pi):.4f} waves)")
    print(f"Phase RMS: {p_rms:.4f} rad ({p_rms/(2*np.pi):.4f} waves)")
    
    # 3. Check Detrended (Tilt Removed)
    y, x = np.indices(phase.shape)
    y = y[valid_mask]
    x = x[valid_mask]
    p = valid_phase
    A = np.column_stack([np.ones_like(x), x, y])
    C, _, _, _ = np.linalg.lstsq(A, p, rcond=None)
    phase_detrended = p - (C[0] + C[1]*x + C[2]*y)
    
    detrend_rms = np.std(phase_detrended)
    print(f"Detrended (Tilt Removed) RMS: {detrend_rms/(2*np.pi):.4f} waves")
    
    # 4. Check Defocus Removed (Spherical Fit)
    # Fit Z4: 2r^2 - 1. Or just r^2
    # Simple paraboloid fit: z = A + Bx + Cy + D(x^2+y^2)
    # Using indices as proxy for coordinates (assume square pixel)
    r2 = x**2 + y**2
    A2 = np.column_stack([np.ones_like(x), x, y, r2])
    C2, _, _, _ = np.linalg.lstsq(A2, p, rcond=None)
    phase_no_defocus = p - (C2[0] + C2[1]*x + C2[2]*y + C2[3]*r2)
    
    no_defocus_rms = np.std(phase_no_defocus)
    print(f"Defocus Removed RMS: {no_defocus_rms/(2*np.pi):.4f} waves")
    
    print(f"Pilot Beam q: {wf.pilot_beam.q_parameter}")
