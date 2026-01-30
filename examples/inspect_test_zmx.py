
import sys
from pathlib import Path
import numpy as np

# Add paths
current_file = Path(__file__).resolve()
project_root = Path(r"d:\BTS")
sys.path.insert(0, str(project_root / 'src'))

import bts
from bts.io import load_zmx

def inspect():
    zmx_path = project_root / 'optiland-master' / 'tests' / 'zemax_files' / 'complicated_fold_mirrors_setup_v2.zmx'
    print(f"Loading {zmx_path}")
    try:
        system = load_zmx(str(zmx_path))
    except Exception as e:
        print(f"Failed to load ZMX: {e}")
        return

    output_file = project_root / 'inspection_result.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Global Surfaces Info:\n")
        global_surfaces = system.get_global_surfaces()
        for i, surf in enumerate(global_surfaces):
            f.write(f"Surface {i}: {surf.surface_type}\n")
            f.write(f"  Vertex: {surf.vertex_position}\n")
            f.write(f"  Normal (Local Z): {surf.surface_normal} (pointing towards incident side)\n")
            f.write(f"  Orientation (Axes Columns [X, Y, Z]):\n{surf.orientation}\n")
            f.write(f"  Radius: {surf.radius}\n")
            f.write(f"  Is Mirror: {surf.is_mirror}\n")
            f.write(f"  Material: {surf.material}\n")
            f.write("-" * 40 + "\n")
    print(f"Output written to {output_file}")

if __name__ == "__main__":
    inspect()
