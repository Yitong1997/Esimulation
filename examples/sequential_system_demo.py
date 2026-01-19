"""
离轴抛物面反射式激光扩束镜仿真示例

系统配置：
- OAP1: f=50mm 凹面镜，将准直光聚焦
- OAP2: f=150mm 凹面镜，将光束重新准直并扩束
- 放大倍率: f2/f1 = 3x

仿真说明：
- 使用 PROPER 进行物理光学衍射传播
- 光束半径等参数从实际仿真波前计算得到
- ABCD 计算仅作为参考对比
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示窗口
import matplotlib.pyplot as plt

from sequential_system import (
    SequentialOpticalSystem,
    GaussianBeamSource,
    ParabolicMirror,
)
from sequential_system.visualization import plot_sampling_results

# 光源：1064nm Nd:YAG，束腰 15mm（准直光束）
source = GaussianBeamSource(wavelength=1.064, w0=15.0, z0=0.0)

# 扩束镜参数
f1 = 50.0   # OAP1 焦距 (mm)
f2 = 150.0  # OAP2 焦距 (mm)
magnification = f2 / f1  # 3x 扩束

# 创建系统（使用较大网格以容纳大光束）
system = SequentialOpticalSystem(source, grid_size=512, beam_ratio=0.3)

# 采样面：入射光束
system.add_sampling_plane(distance=0.0, name="Input")

# OAP1：聚焦镜
system.add_surface(ParabolicMirror(
    parent_focal_length=f1,
    thickness=f1 + f2,
    semi_aperture=25.0,
    name="OAP1",
))

# 采样面：共焦点
system.add_sampling_plane(distance=f1, name="Focus")

# OAP2：准直镜
system.add_surface(ParabolicMirror(
    parent_focal_length=f2,
    thickness=100.0,
    semi_aperture=60.0,
    name="OAP2",
))

# 采样面：输出光束
system.add_sampling_plane(distance=f1 + f2 + 100.0, name="Output")

# 运行仿真
print(f"=== OAP Beam Expander ({magnification:.0f}x) ===")
print("Running PROPER simulation...")
results = system.run()

# 输出仿真结果
print("\n--- Simulation Results (from wavefront) ---")
for result in results:
    print(f"  {result.name}: w = {result.beam_radius:.3f} mm, "
          f"sampling = {result.sampling:.4f} mm/pixel, "
          f"WFE RMS = {result.wavefront_rms:.4f} waves")

# ABCD 参考值
print("\n--- ABCD Reference ---")
abcd_input = system.get_abcd_result(0.0)
abcd_output = system.get_abcd_result(f1 + f2 + 100.0)
print(f"  Input:  w = {abcd_input.w:.3f} mm")
print(f"  Output: w = {abcd_output.w:.3f} mm")
print(f"  Magnification: {abcd_output.w / abcd_input.w:.2f}x")

# 可视化：光路布局
print("\n--- Visualization ---")
fig1, ax1 = system.draw_layout(show=False)
fig1.savefig("beam_expander_layout.png", dpi=150)
plt.close(fig1)
print("Saved: beam_expander_layout.png")

# 可视化：采样面强度和相位
fig2, axes2 = plot_sampling_results(results, show=False)
fig2.savefig("beam_expander_results.png", dpi=150)
plt.close(fig2)
print("Saved: beam_expander_results.png")

print("\nDone!")
