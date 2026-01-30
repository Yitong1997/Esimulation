
import sys
from pathlib import Path
import os

# 获取项目根目录
current_file = Path(__file__).resolve()
project_root = current_file.parent

# 添加源代码和依赖库到 Python 路径
sys.path.insert(0, str(project_root / 'src'))                # BTS 源码
sys.path.insert(0, str(project_root / 'optiland-master'))    # Optiland 几何光线追迹库
sys.path.insert(0, str(project_root / 'proper_v3.3.4_python')) # PROPER 物理光学传播库

import bts
import matplotlib.pyplot as plt

def main():
    print("BTS Debug Run")
    zmx_dir = project_root / 'optiland-master' / 'tests' / 'zemax_files'
    zmx_file = zmx_dir / 'complicated_fold_mirrors_setup_v2.zmx'
    
    if not zmx_file.exists():
        print(f"Error: {zmx_file} not found")
        return

    try:
        system = bts.load_zmx(str(zmx_file))
    except Exception as e:
        print(f"Load failed: {e}")
        return

    wavelength = 0.6328
    waist_radius = 2.0
    grid_size = 256 # Smaller grid for faster debug

    source = bts.GaussianSource(
        wavelength_um=wavelength,
        w0_mm=waist_radius,
        grid_size=grid_size,
        z0_mm=0.0,
        physical_size_mm=8*waist_radius,
        beam_diam_fraction=0.25
    )

    print("\nStarting Simulation with DEBUG=True...")
    try:
        result = bts.simulate(
            system, 
            source, 
            propagation_method="local_raytracing", 
            debug=True  # ENABLE DEBUG
        )
        print("Simulation done.")
    except Exception as e:
        print(f"Simulation Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
