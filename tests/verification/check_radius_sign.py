
import sys
from pathlib import Path
project_root = Path('d:/BTS')
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))
sys.path.insert(0, str(project_root / 'proper_v3.3.4_python'))
import bts

def check_radius_sign():
    # Parameters from test
    F1_MM = -1000.0             # OAP1 焦距 (mm)，负值表示凸面
    r1_mm = -2 * F1_MM          # r1 = 2000
    
    F2_MM = 2000.0              # OAP2 焦距 (mm)，正值表示凹面
    r2_mm = -2 * F2_MM          # r2 = -4000
    
    print(f"Input r1_mm: {r1_mm}")
    print(f"Input r2_mm: {r2_mm}")
    
    system = bts.OpticalSystem("Sign Check")
    
    # OAP1
    system.add_parabolic_mirror(
        x=0.0, y=50, z=0,
        radius=r1_mm, 
    )
    
    # OAP2
    system.add_parabolic_mirror(
        x=0.0, y=150, z=1000,
        radius=-r2_mm,  # Test file uses -r2_mm = 4000
    )
    
    print("\n--- Surface Properties in system._surfaces ---")
    vals = system._surfaces
    
    for i, s in enumerate(vals):
        print(f"Surface {i}: Radius = {s.radius}")

if __name__ == "__main__":
    check_radius_sign()
