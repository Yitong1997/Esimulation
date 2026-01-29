import numpy as np
import sys
import os

# Mock classes to simulate the logic without importing the whole heavy system
class PilotBeamParams:
    def __init__(self, wavelength_um, w0_mm, z0_mm, q=None):
        self.wavelength_um = wavelength_um
        wavelength_mm = wavelength_um * 1e-3
        if q is None:
            z_R = np.pi * w0_mm**2 / wavelength_mm
            z = -z0_mm
            self.q_parameter = z + 1j * z_R
        else:
            self.q_parameter = q
            
    @property
    def spot_size_mm(self):
        wavelength_mm = self.wavelength_um * 1e-3
        inv_q = 1.0 / self.q_parameter
        imag_part = np.imag(inv_q)
        w_sq = -wavelength_mm / (np.pi * imag_part)
        return np.sqrt(w_sq) if w_sq > 0 else 0.0

    @property
    def curvature_radius_mm(self):
        inv_q = 1.0 / self.q_parameter
        real_part = np.real(inv_q)
        if abs(real_part) < 1e-15:
            return np.inf
        return 1.0 / real_part

    def propagate(self, distance_mm):
        return PilotBeamParams(self.wavelength_um, 0, 0, q=self.q_parameter + distance_mm)
        
    def apply_mirror(self, radius_mm):
        if np.isinf(radius_mm):
            return self
        A, B, C, D = 1, 0, 2/radius_mm, 1
        q_new = (A * self.q_parameter + B) / (C * self.q_parameter + D)
        return PilotBeamParams(self.wavelength_um, 0, 0, q=q_new)

def test_galilean_expander_pilot_beam():
    print("Verifying Galilean Expander Pilot Beam Propagation...")
    
    # Parameters from test file
    WAVELENGTH_UM = 1.0
    W0_MM = 5.0
    
    # OAP1 (Convex)
    F1 = -1000.0
    R1 = 2 * F1 # -2000? No. 
    # In test file: r1_mm = -2 * f1_mm = -2 * (-1000) = 2000.
    # Convex mirror has positive radius in optiland (center of curvature to the right, light goes right)
    # BUT wait. Mirror reflects light. If light goes +Z, hits mirror, reflects to -Z.
    # Convex mirror bulges towards -Z. Center of curvature is at +Z. So R > 0.
    # So R1 = 2000 is correct for Convex.
    R1 = 2000.0
    
    # Off-axis distance
    d1 = 50.0 # From comments
    
    # OAP2 (Concave)
    F2 = 2000.0
    # In test file: r2_mm = -2 * f2_mm = -4000.
    # radius = -r2_mm = 4000? No.
    # radius = -(-4000) = 4000.
    # Wait. Concave mirror. Light comes from +Z (after reflection from OAP1).
    # If light travels -Z, and mirror is Concave (bulges to -Z, center of curvature at -Z).
    # In optiland, if R < 0, center is at -Z.
    # So for light traveling -Z, hitting R<0 surface...
    # Optiland conventions are tricky for sequential rays. 
    # But usually "Concave" means focusing.
    # Let's assume the math derived: R_eff should be negative for focusing mirror?
    # Our previous derivation: R_eff = -2 * Dist.
    # The code calculates R_eff = R + d^2/R.
    # If we need R_eff < 0, we need R < 0.
    # So OAP2 should have R < 0. (Concave).
    # Let's use R2 = -4000.
    R2 = -4000.0
    d2 = 100.0 # Match Magnification M=2 (1000->2000)
    
    # Distances
    # Distance to focus for OAP1 (Convex): D1 = |R_eff1| / 2
    # R_eff1 = R1 + d1**2/R1
    R_eff1 = R1 + d1**2/R1
    D1 = abs(R_eff1) / 2
    print(f"OAP1: R={R1}, d={d1}, R_eff={R_eff1:.4f}, Dist_to_Focus={D1:.4f}")
    
    # Distance to focus for OAP2 (Concave): D2 = |R_eff2| / 2
    # R_eff2 = R2 + d2**2/R2
    R_eff2 = R2 + d2**2/R2
    D2 = abs(R_eff2) / 2
    print(f"OAP2: R={R2}, d={d2}, R_eff={R_eff2:.4f}, Dist_to_Focus={D2:.4f}")
    
    # Separation L (Physical distance along ray)
    L_ray = D2 - D1
    print(f"Propagating distance L_ray = {L_ray:.4f}")
    
    # --- Propagation ---
    
    # 0. Initial Beam (Collimated)
    # The Source creates a beam with waist at z=-1000 (1000mm before OAP1).
    # Wait, usually source z0 is distance *from* waist.
    # If z0 = 1000, waist is at current position + 1000? Or -1000?
    # DataModels: z = -z0_mm.
    # If z0=1000, z=-1000. Waist is 1000mm *upstream*.
    # So beam at surface is expanding.
    # BUT Galilean expander requires input to be COLLIMATED (waist at OAP1).
    # Usually z0=0.
    # Let's assume z0=0 for ideal case.
    pb = PilotBeamParams(WAVELENGTH_UM, W0_MM, 0.0)
    print(f"Start: R={pb.curvature_radius_mm}, w={pb.spot_size_mm}")
    
    # 1. OAP1 (Convex)
    # Apply Mirror
    pb = pb.apply_mirror(R_eff1)
    print(f"After OAP1: R={pb.curvature_radius_mm:.4f}, w={pb.spot_size_mm:.4f}")
    
    # 2. Propagate
    pb = pb.propagate(L_ray)
    print(f"After Prop {L_ray:.4f}: R={pb.curvature_radius_mm:.4f}, w={pb.spot_size_mm:.4f}")
    
    # Check if R matches D2?
    # It should match D2 (since virtual source is at distance D1 behind OAP1, total distance D1 + (D2-D1) = D2).
    # Since OAP1 is convex, output is diverging, radius should be positive D2.
    
    # 3. OAP2 (Concave)
    # Apply Mirror
    pb = pb.apply_mirror(R_eff2)
    print(f"After OAP2: R={pb.curvature_radius_mm:.4f}, w={pb.spot_size_mm:.4f}")
    
    if abs(pb.curvature_radius_mm) > 1e10:
        print("RESULT: Collimated! (PASS)")
    else:
        print("RESULT: NOT Collimated. (FAIL)")

if __name__ == "__main__":
    test_galilean_expander_pilot_beam()
