from optiland.optic import Optic
from optiland.raytrace import RealRayTracer
import numpy as np

def verify_optiland_coords():
    print("Verifying Optiland Coordinate System...")
    optic = Optic()
    
    # Set aperture to ensure no blocking
    optic.set_aperture(aperture_type='EPD', value=100.0)
    optic.set_field_type(field_type='angle')
    optic.add_field(y=0, x=0)
    optic.add_wavelength(value=0.5, is_primary=True)
    
    # 1. Setup a simple system:
    # Ray along +Z (0,0,1)
    
    optic.add_surface(index=0, thickness=0.0) # Object
    optic.add_surface(index=1, thickness=0.0, is_stop=True) # Stop at 0,0,0
    
    # Mirror at (0,0,10), Tilted 45 degrees around X
    # Result should be +Y direction (0, 1, 0)
    
    optic.add_surface(
        index=2,
        radius=np.inf,
        material='mirror',
        rx=np.deg2rad(45),
        z=10.0
    )
    
    tracer = RealRayTracer(optic)
    rays = tracer.trace_generic(Hx=0, Hy=0, Px=0, Py=0, wavelength=0.5)
    
    if len(rays.L) == 0:
        print("ERROR: No rays returned.")
        return

    L = rays.L[0]
    M = rays.M[0]
    N = rays.N[0]
    
    print(f"Output Direction (L,M,N): ({L:.4f}, {M:.4f}, {N:.4f})")
    
    if np.isclose(L, 0) and np.isclose(M, 1) and np.isclose(N, 0, atol=1e-5):
        print("CONCLUSION: Optiland returns output rays in the SYSTEM GLOBAL coordinate frame (Result is +Y).")
    elif np.isclose(N, 1, atol=1e-5):
        print("CONCLUSION: Optiland output seems unmodified or local? (Result is +Z)")
    else:
        print("CONCLUSION: Optiland returns something else.")

if __name__ == "__main__":
    verify_optiland_coords()
