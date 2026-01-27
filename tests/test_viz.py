
import sys
import os
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'd:\\BTS\\src'))

from hybrid_simulation.simulator import HybridSimulator

def test_debug_viz():
    print("Initializing Simulator with Debug Mode...")
    sim = HybridSimulator(verbose=True)
    sim.set_debug_mode(True)
    
    # Load or setup a simple OAP system
    # Since I don't have the ZMX file handy, I'll programmatically add an OAP
    
    sim.set_source(wavelength_um=0.6328, w0_mm=5.0)
    
    # Add an OAP (Parabolic Mirror)
    # 90 degree OAP: 
    # Focal length = 100mm
    # Off-axis distance = 100mm (parent vertex to optical axis)
    # In global coords:
    # Entrance/Source at (0, 0, 0) -> propagating +Z
    # Mirror vertex at (0, 100, 100) (if parent vertex is at (0, 100, 200)?)
    # Let's just use a simple tilted flat mirror or a simple OAP setup.
    
    # Let's use the helper provided in simulator if available or just manual add
    # sim.add_flat_mirror(z=100, tilt_x=45) 
    
    # Trying a simple propagation
    sim.add_flat_mirror(z=50, tilt_x=45) 
    
    print("Running Simulation...")
    try:
        sim.run()
        print("Simulation finished.")
    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_debug_viz()
