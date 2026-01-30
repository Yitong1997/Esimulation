
import sys
from pathlib import Path

# Add paths
current_file = Path(__file__).resolve()
project_root = Path(r"d:\BTS")
sys.path.insert(0, str(project_root / 'src'))

from sequential_system.zmx_parser import ZmxParser

def dump_raw():
    zmx_path = project_root / 'optiland-master' / 'tests' / 'zemax_files' / 'complicated_fold_mirrors_setup_v2.zmx'
    print(f"Loading {zmx_path}")
    
    parser = ZmxParser(str(zmx_path))
    zmx_data = parser.parse()
    
    output_file = project_root / 'raw_zmx_dump.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Raw ZMX Surface Data:\n")
        sorted_indices = sorted(zmx_data.surfaces.keys())
        for idx in sorted_indices:
            surf = zmx_data.surfaces[idx]
            f.write(f"Surface {idx}: {surf.surface_type}\n")
            f.write(f"  Comment: {surf.comment}\n")
            f.write(f"  Thickness: {surf.thickness}\n")
            f.write(f"  Radius: {surf.radius}\n")
            f.write(f"  Material: {surf.material}\n")
            
            # Check for Coordinate Break params
            if surf.surface_type == 'coordinate_break':
                f.write(f"  Decenter X: {getattr(surf, 'decenter_x', 0)}\n")
                f.write(f"  Decenter Y: {getattr(surf, 'decenter_y', 0)}\n")
                f.write(f"  Tilt X: {getattr(surf, 'tilt_x_deg', 0)}\n")
                f.write(f"  Tilt Y: {getattr(surf, 'tilt_y_deg', 0)}\n")
                f.write(f"  Tilt Z: {getattr(surf, 'tilt_z_deg', 0)}\n")
                f.write(f"  Order: {getattr(surf, 'order', 0)}\n")
            
            f.write("-" * 40 + "\n")
            
    print(f"Dumped raw data to {output_file}")

if __name__ == "__main__":
    dump_raw()
