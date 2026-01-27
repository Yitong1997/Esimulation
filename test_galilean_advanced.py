import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Add project root to path
project_root = Path(r'd:\BTS')
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tests.integration.伽利略式离轴抛物面扩束镜传输误差标准测试文件 import run_galilean_oap_expander_test
import bts

def gaussian_1d(x, a, x0, w, offset):
    return a * np.exp(-2 * ((x - x0) / w)**2) + offset

def measure_beam_quality(wavefront):
    """
    Measure beam size (w) and divergence (theta).
    """
    amp = wavefront.amplitude
    phase = wavefront.phase
    grid = wavefront.grid
    
    # 1. Measure Near Field Beam Size (w_out)
    # Fit Gaussian to central cut
    n = grid.grid_size
    sampling = grid.physical_size_mm / n
    coords = np.linspace(-n/2, n/2-1, n) * sampling
    
    # Find peak center
    y_idx, x_idx = np.unravel_index(np.argmax(amp), amp.shape)
    
    # X-cut and Y-cut
    x_cut = amp[y_idx, :]
    y_cut = amp[:, x_idx]
    
    try:
        popt_x, _ = curve_fit(gaussian_1d, coords, x_cut, p0=[np.max(x_cut), coords[x_idx], 5.0, 0])
        w_x = abs(popt_x[2])
        
        popt_y, _ = curve_fit(gaussian_1d, coords, y_cut, p0=[np.max(y_cut), coords[y_idx], 5.0, 0])
        w_y = abs(popt_y[2])
    except:
        w_x, w_y = 0, 0
        
    w_out_avg = (w_x + w_y) / 2
    
    # 2. Measure Divergence
    # Method: Propagate to Far Field (Fraunhofer) using FFT
    # Divergence angle theta = w_f / f_lens (if using Fourier lens)
    # Equivalent to FFT magnitude scaled.
    # Coordinates in far field: u = x / (lambda * z) -> theta = u * lambda = x/z
    # Grid in freq domain: df = 1 / (N * dx)
    # Theta grid = df * lambda
    
    complex_field = wavefront.get_complex_amplitude()
    fft_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(complex_field)))
    fft_amp = np.abs(fft_field)
    
    # Freq sampling
    wavelength_mm = wavefront.wavelength_um / 1000.0
    df = 1.0 / grid.physical_size_mm  # Spatial frequency increment (1/mm)
    theta_res_rad = df * wavelength_mm # Angular resolution (rad)
    
    theta_coords = np.linspace(-n/2, n/2-1, n) * theta_res_rad
    
    # Fit Gaussian to Far Field Amplitude
    y_idx_fft, x_idx_fft = np.unravel_index(np.argmax(fft_amp), fft_amp.shape)
    x_cut_fft = fft_amp[y_idx_fft, :]
    
    try:
        popt_div, _ = curve_fit(gaussian_1d, theta_coords, x_cut_fft, 
                                p0=[np.max(x_cut_fft), theta_coords[x_idx_fft], 0.001, 0])
        theta_div = abs(popt_div[2]) # Half divergence angle (1/e^2 radius in angle)
    except:
        theta_div = 0
        
    return w_out_avg, theta_div * 1000.0 # mrad

def run_advanced_verification():
    print("Running Advanced Galilean Expander Verification...")
    print("="*60)
    
    # Run the standard test to get the result object
    # We need to modify the standard function to return the full result object or run simulation manually
    # The standard function returns a dict.
    # Let's run simulation manually to get the objects.
    
    WAVELENGTH_UM = 0.633
    W0_MM = 5.0
    F1 = -100.0
    F2 = 300.0
    D1 = 50.0
    MAG = -F2/F1 # 3.0
    D2 = MAG * D1 # 150.0
    L = F1 + F2 # 200.0
    
    EXPECTED_W_OUT = W0_MM * MAG # 15.0 mm
    EXPECTED_DIV = (WAVELENGTH_UM / (np.pi * EXPECTED_W_OUT * 1000)) * 1000 # mrad (approx, if collimated at waist)
    # Ideally divergence is 0 if perfectly collimated, but Gaussian has minimum diffraction divergence.
    # Theta_0 = lambda / (pi * w0)
    
    print(f"Design Parameters:")
    print(f"  Input w0: {W0_MM} mm")
    print(f"  Magnification: {MAG}x")
    print(f"  Expected Output w0: {EXPECTED_W_OUT:.3f} mm")
    print(f"  Expected Diffraction Limit Divergence: {EXPECTED_DIV:.6f} mrad")
    print("-" * 60)
    
    system = bts.OpticalSystem("Galilean Lens Expander")
    # Use Paraxial Lenses to verify expansion logic without geometric reflection complexity
    # Lens 1: Diverging (f = -100)
    system.add_paraxial_lens(x=0.0, y=0.0, z=0.0, focal_length=F1)
    # Lens 2: Collimating (f = 300)
    system.add_paraxial_lens(x=0.0, y=0.0, z=L, focal_length=F2)
    
    # Ensure large enough grid for expanded beam (w=15mm -> 4w=60mm)
    # Standard grid 256 over ? 
    # Source default physical size is 4*w0 = 20mm. 
    # At output, beam is 15mm. 20mm window is too small! Beam will be clipped!
    # ISSUE DETECTED IN STANDARD TEST?
    # Standard test source: Physical Size = 4 * 5.0 = 20.0 mm.
    # Output Beam w = 15.0 mm. 4w = 60.0 mm.
    # If physical size stays 20mm, the beam is heavily clipped.
    # PROPER/Hybrid usually maintains *grid sampling interval* or *physical size* depending on propagation.
    # "local_raytracing" maps the input field to the surface.
    # If the surface aperture / grid definitions don't expand, we have a problem.
    # The standard test might be failing silently or purely inspecting the central part.
    
    # Let's force a larger grid physical size for the source to accommodate the output?
    # No, usually we want size to match the beam.
    # If we start with 20mm, expand 3x, we effectively need 60mm at output.
    # Does the code handle grid expansion?
    # HybridElementPropagator: "Output grid physical size is determined by..."
    # Usually it projects pixel-by-pixel. If input pixels capture 20mm, output pixels capture the mapped area.
    # If magnification is 3x, the mapped area should be 60mm.
    # Let's see what the simulation produces.
    
    source = bts.GaussianSource(wavelength_um=WAVELENGTH_UM, w0_mm=W0_MM, grid_size=512)
    # Force larger physical size if needed? 
    # Let's trust the propagator handling first.
    
    result = bts.simulate(system, source, verbose=True)
    
    if not result.success:
        print(f"Simulation FAILED: {result.error_message}")
        try:
            # Try to get surfaces anyway to see how far it got
            print(f"Surfaces processed: {len(result.surfaces)}")
            for s in result.surfaces:
                print(f"  {s.index}: {s.name}")
        except:
            pass
        return

    final_wf = result.get_final_wavefront()
    phys_size = final_wf.grid.physical_size_mm
    print(f"Output Grid Physical Size: {phys_size:.3f} mm")
    
    w_out, theta_out = measure_beam_quality(final_wf)
    
    print(f"\nMeasured Results:")
    print(f"  Output Beam Size (w): {w_out:.3f} mm (Error: {(w_out - EXPECTED_W_OUT)/EXPECTED_W_OUT*100:.2f}%)")
    print(f"  Output Divergence: {theta_out:.6f} mrad")
    
    resid_rms = final_wf.get_residual_rms_waves() * 1000
    print(f"  Residual RMS: {resid_rms:.3f} milli-waves")
    
    # Check clipping
    if phys_size < 3 * w_out:
        print("WARNING: Grid size too small for output beam! Beam might be clipped.")
    
    
if __name__ == '__main__':
    run_advanced_verification()
