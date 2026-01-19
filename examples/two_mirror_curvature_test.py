"""
双反射镜系统验证

验证高斯光束经过两个球面反射镜后，PROPER 物理光学仿真与 ABCD 矩阵法的一致性。
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')

from sequential_system import (
    SequentialOpticalSystem,
    GaussianBeamSource,
    SphericalMirror,
)

# 1. 定义光源（束腰在起点前 100mm）
source = GaussianBeamSource(wavelength=1.064, w0=5.0, z0=-100.0)

# 2. 创建系统并添加两个球面反射镜
system = SequentialOpticalSystem(source, grid_size=512, beam_ratio=0.3)

system.add_sampling_plane(distance=0.0, name="Input")
system.add_surface(SphericalMirror(
    radius_of_curvature=300.0,   # 凹面镜，f=150mm
    thickness=120.0,
    semi_aperture=20.0,
    name="M1",
))
system.add_sampling_plane(distance=120.0, name="After M1")
system.add_surface(SphericalMirror(
    radius_of_curvature=-400.0,  # 凸面镜，f=-200mm
    thickness=150.0,
    semi_aperture=25.0,
    name="M2",
))
system.add_sampling_plane(distance=270.0, name="Output")

# 3. 运行仿真并绘图
results = system.run(plot=True, save_plot="two_mirror_test.png", show_plot=False)
print("✓ 保存: two_mirror_test.png\n")

# 4. 对比 PROPER 与 ABCD
print(f"{'采样面':<12} {'PROPER w (mm)':<15} {'ABCD w (mm)':<15} {'误差 (%)':<10}")
print("-" * 55)

for result in results:
    abcd = system.get_abcd_result(result.distance)
    err = abs(result.beam_radius - abcd.w) / abcd.w * 100
    print(f"{result.name:<12} {result.beam_radius:<15.4f} {abcd.w:<15.4f} {err:<10.2f}")
