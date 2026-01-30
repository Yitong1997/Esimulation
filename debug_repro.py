
import sys
from pathlib import Path
import numpy as np

# Setup paths
current_file = Path(__file__).resolve()
project_root = Path("d:/BTS")
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))
sys.path.insert(0, str(project_root / 'proper_v3.3.4_python'))

import bts
from wavefront_to_rays import ElementRaytracer

def inspection_script():
    try:
        zmx_dir = project_root / 'optiland-master' / 'tests' / 'zemax_files'
    zmx_file = zmx_dir / 'complicated_fold_mirrors_setup_v2.zmx'
    
    print(f"Loading {zmx_file}")
    system = bts.load_zmx(str(zmx_file))
    
    # Inspect Global Surfaces
    print("\n[Global Surfaces Inspection]")
    s8 = None
    for s in system.surfaces:
        print(f"Idx {s.index}: {s.surface_type}, R={s.radius}, T={s.thickness}, Mat={s.material}")
        print(f"  Pos: {s.vertex_position}")
        print(f"  Z-Axis: {s.orientation[:, 2]}")
        if s.index == 8:
            s8 = s
            
    if not s8:
        print("Surface 8 not found!")
        return

    # Create dummy source just to run propagator logic (partially) or invoke ElementRaytracer manually
    # We can try to simulate what hybrid_element_propagator does.
    # It creates an ElementRaytracer for a "Element".
    # Since we don't know exactly how they are grouped, let's look at `hybrid_propagator.py` or use `system` structure.
    # But usually `bts.load_zmx` returns a `OpticalSystem` which is a list of surfaces? 
    # No, `bts.load_zmx` returns `ZmxOpticalSystem`? Np, it returns `bts.OpticalSystem` (from `bts/__init__.py`).
    # Wait. `load_zmx` in `bts/io.py` returns `OpticalSystem`.
    
    # Actually `bts.simulate` uses `HybridOpticalPropagator`.
    # It traverses the system.
    
    # Let's peek into the grouping.
    # The system is likely just a list of GlobalSurfaceDefinition.
    # `HybridOpticalPropagator` groups them.
    from hybrid_optical_propagation.hybrid_propagator import HybridOpticalPropagator
    from hybrid_optical_propagation.data_models import SourceDefinition
    
    source = SourceDefinition(wavelength_um=0.6328, w0_mm=1.0, z0_mm=0.0)
    propagator = bts.HybridOpticalPropagator(
        optical_system=system,
        source=source,
        propagation_method="local_raytracing"
    )
    
    print("\n[Propagator Elements Inspection]")
    for i, elem in enumerate(propagator.elements):
        print(f"Element {i} (Type: {type(elem).__name__})")
        if hasattr(elem, 'surfaces'):
            # It's a HybridElementPropagator
            print(f"  Surfaces: {[s.index for s in elem.surfaces]}")
            # Check if Surface 8 is here
            if any(s.index == 8 for s in elem.surfaces):
                print("  -> Contains Surface 8!")
                # Run `ElementRaytracer` init logic explicitly to debug
                # We need input beam direction.
                # Propagator tracks beam.
                # But we can just assume beam direction based on previous surfaces. Or just let it run.
                
                # Let's try to infer beam direction at simple fold mirror setup.
                # Assuming forward trace.
                pass

    # Actually running trace might be complex due to dependencies.
    # But checking `s8.orientation` above is already very useful.
    # If Z-axis is (0,0,1) but after mirrors, it confirms CoordinateSystem issue.

    except Exception as e:
        print(f"\n[ERROR] Script failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        inspection_script()
    except Exception as e:
        print(f"Global Error: {e}")
