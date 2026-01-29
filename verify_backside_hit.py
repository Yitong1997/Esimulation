
import sys
import numpy as np

sys.path.insert(0, 'd:/BTS/optiland-master')

try:
    from optiland.surfaces import Surface
    from optiland.geometries import StandardGeometry
    from optiland.rays import RealRays
    from optiland.coordinate_system import CoordinateSystem
except ImportError:
    print("Optiland not found. Cannot run verification.")
    sys.exit(1)

def verify_backside_hit():
    print("Verifying Backside Hit Logic...")
    
    # Setup dummy coordinate system
    cs = CoordinateSystem(x=0, y=0, z=0)
    
    # CASE 1: Standard Frontside Hit
    R = 1000.0
    geometry = StandardGeometry(cs, radius=R, conic=0)
    
    print("\n--- CASE 1: Frontside Hit (Ray traveling +Z) ---")
    try:
        surf = Surface(geometry)
        
        # Ray 1: Frontside
        # Ray at y=10, z=-10, N=1
        r1 = RealRays(x=[0], y=[10], z=[-10], L=[0], M=[0], N=[1], intensity=[1], wavelength=[0.55])
        surf.trace(r1)
        print(f"In(y=10, N=1):")
        print(f"  Out Direction: (L={r1.L[0]:.6f}, M={r1.M[0]:.6f}, N={r1.N[0]:.6f})")
        print(f"  Out Position:  (x={r1.x[0]:.3f}, y={r1.y[0]:.3f}, z={r1.z[0]:.3f})")
        
        if r1.M[0] > 0:
            print("  -> Diverging (Convex). CORRECT.")
        else:
            print("  -> Converging (Concave). UNEXPECTED.")
    except Exception as e:
        print(f"Trace Failed: {e}")

    # CASE 2: Backside Hit (User's Case)
    print("\n--- CASE 2: Backside Hit (Ray traveling -Z) ---")
    try:
        # Ray 2: Backside
        # Ray at y=10, z=10, N=-1
        # Surface is defined with R=1000 (Bulging to +Z).
        # Ray comes from +Z (z=10).
        r2 = RealRays(x=[0], y=[10], z=[10], L=[0], M=[0], N=[-1], intensity=[1], wavelength=[0.55])
        surf.trace(r2)
        print(f"In(y=10, N=-1):")
        print(f"  Out Direction: (L={r2.L[0]:.6f}, M={r2.M[0]:.6f}, N={r2.N[0]:.6f})")
        print(f"  Out Position:  (x={r2.x[0]:.3f}, y={r2.y[0]:.3f}, z={r2.z[0]:.3f})")
        
        # Check if reflected back to +Z
        if r2.N[0] > 0.9: 
             if r2.M[0] < 0:
                 print("  -> Converging (Concave). CONFIRMED: Backside R>0 is Concave.")
             else:
                 print("  -> Diverging (Convex). UNEXPECTED.")
        else:
            print(f"  -> Ray did not reflect properly? N={r2.N[0]}")
            
    except Exception as e:
        print(f"Trace Failed: {e}")

if __name__ == "__main__":
    verify_backside_hit()
