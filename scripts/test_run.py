import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'optiland-master')

print("Starting test...")
sys.stdout.flush()

try:
    from hybrid_optical_propagation import (
        SourceDefinition,
        HybridOpticalPropagator,
        load_optical_system_from_zmx,
    )
    print("Imports successful")
    sys.stdout.flush()
    
    zmx_file = "optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx"
    optical_system = load_optical_system_from_zmx(zmx_file)
    print(f"Loaded optical system with {len(optical_system)} surfaces")
    sys.stdout.flush()
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
