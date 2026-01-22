"""
检查 ZMX 文件的表面定义，确认表面编号和面形
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))

import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from hybrid_optical_propagation import load_optical_system_from_zmx

print("=" * 70)
print("ZMX 文件表面定义")
print("=" * 70)

zmx_file = "optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx"
optical_system = load_optical_system_from_zmx(zmx_file)

print(f"\n共 {len(optical_system)} 个表面:\n")
print(f"{'Index':>5} | {'Name':^20} | {'Type':^12} | {'Radius':>12} | {'Thickness':>10} | {'Is Mirror':^9}")
print("-" * 80)

for i, surface in enumerate(optical_system):
    name = getattr(surface, 'name', '') or ''
    surf_type = getattr(surface, 'surface_type', 'STANDARD')
    radius = surface.radius
    thickness = surface.thickness
    is_mirror = surface.is_mirror
    
    radius_str = f"{radius:.2f}" if not np.isinf(radius) else "Infinity"
    
    print(f"{i:>5} | {name:^20} | {surf_type:^12} | {radius_str:>12} | {thickness:>10.2f} | {str(is_mirror):^9}")

print("\n" + "=" * 70)
print("表面详细信息")
print("=" * 70)

for i, surface in enumerate(optical_system):
    name = getattr(surface, 'name', '') or f'Surface {i}'
    print(f"\n【Surface {i}】{name}")
    print(f"  类型: {getattr(surface, 'surface_type', 'STANDARD')}")
    print(f"  曲率半径: {surface.radius}")
    print(f"  厚度: {surface.thickness}")
    print(f"  是否反射镜: {surface.is_mirror}")
    print(f"  顶点位置: {surface.vertex_position}")
    print(f"  表面法向量: {surface.surface_normal}")
    if hasattr(surface, 'tilt_x') and surface.tilt_x != 0:
        print(f"  倾斜 X: {np.degrees(surface.tilt_x):.2f}°")
    if hasattr(surface, 'tilt_y') and surface.tilt_y != 0:
        print(f"  倾斜 Y: {np.degrees(surface.tilt_y):.2f}°")
