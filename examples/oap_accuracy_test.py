"""
离轴抛物面（OAP）精度测试

使用 BTS 主函数 API 测试离轴抛物面镜的仿真精度。
离轴量通过绝对坐标 (x, y, z) 中的 y 值指定。

测试参数：
- 曲率半径：500 mm（焦距 250 mm）
- 离轴量：125 mm（d/f = 0.5，出射角约 27°）
- 波长：0.633 μm (He-Ne)
- 束腰半径：3.0 mm

作者：混合光学仿真项目
"""

import sys
import os
import numpy as np

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ============================================================
# 1. 导入与初始化
# ============================================================
import bts

# ============================================================
# 2. 仿真参数
# ============================================================
wavelength_um = 0.633    # He-Ne 激光波长
w0_mm = 3.0              # 束腰半径
grid_size = 256          # 网格大小

# 离轴抛物面参数
radius_mm = 500.0        # 曲率半径 R = 2f
focal_length_mm = radius_mm / 2  # 焦距 f = R/2 = 250 mm
off_axis_mm = 125.0      # 离轴量（d/f = 0.5）
mirror_z_mm = 100.0      # 镜面 Z 位置

# 计算理论出射角
exit_angle_deg = np.degrees(np.arctan(off_axis_mm / focal_length_mm))
print(f"理论出射角: {exit_angle_deg:.2f}°")

# 计算瑞利距离
wavelength_mm = wavelength_um * 1e-3
z_R = np.pi * w0_mm**2 / wavelength_mm
print(f"瑞利距离: z_R = {z_R:.2f} mm")
print(f"z/z_R = {mirror_z_mm/z_R:.3f} (近场条件: < 1)")

# ============================================================
# 3. 定义光学系统
# ============================================================
system = bts.OpticalSystem("OAP Accuracy Test")

# 添加离轴抛物面镜
# 关键：通过 y 坐标指定离轴量，不使用专门的 off_axis_distance 参数
system.add_parabolic_mirror(
    z=mirror_z_mm,
    y=off_axis_mm,        # 离轴量通过 y 坐标指定
    radius=radius_mm,
    semi_aperture=w0_mm * 4,
)

# ============================================================
# 4. 定义光源
# ============================================================
source = bts.GaussianSource(
    wavelength_um=wavelength_um,
    w0_mm=w0_mm,
    grid_size=grid_size,
    physical_size_mm=8 * w0_mm,
)

# ============================================================
# 5. 系统信息展示
# ============================================================
print("\n" + "="*60)
print("系统信息")
print("="*60)
system.print_info()
source.print_info()

# ============================================================
# 6. 执行仿真
# ============================================================
print("\n" + "="*60)
print("执行仿真")
print("="*60)

result = bts.simulate(system, source, verbose=True)

# ============================================================
# 7. 结果分析
# ============================================================
print("\n" + "="*60)
print("仿真结果")
print("="*60)

result.summary()

# 获取最终波前数据
final_wf = result.get_final_wavefront()
rms_waves = final_wf.get_residual_rms_waves()
pv_waves = final_wf.get_residual_pv_waves()

print(f"\n最终波前误差:")
print(f"  相位残差 RMS: {rms_waves*1000:.3f} milli-waves")
print(f"  相位残差 PV:  {pv_waves:.4f} waves")

# ============================================================
# 8. 结果保存
# ============================================================
save_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'oap_test')
os.makedirs(save_dir, exist_ok=True)

# 保存概览图
result.plot_all(
    save_path=os.path.join(save_dir, 'oap_overview.png'),
    show=False,
)
print(f"\n概览图已保存到: {save_dir}/oap_overview.png")

# 保存完整结果
result.save(save_dir)
print(f"完整结果已保存到: {save_dir}/")

print("\n" + "="*60)
print("测试完成")
print("="*60)
