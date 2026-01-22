"""
分析 PROPER 边缘相位的来源

关键问题：
- PROPER 计算的相位在边缘区域不是纯二次形式
- 这是 PROPER 的正常行为还是问题？

可能的原因：
1. 高斯光束的精确相位公式不是纯二次形式
2. PROPER 的衍射计算引入了额外相位
3. 数值边界效应
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))

import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import proper

print("=" * 80)
print("PROPER 边缘相位分析")
print("=" * 80)

# 创建一个简单的高斯光束，不经过任何传播
wavelength_um = 0.55
wavelength_m = wavelength_um * 1e-6
w0_mm = 5.0
w0_m = w0_mm * 1e-3
grid_size = 512
physical_size_mm = 40.0
physical_size_m = physical_size_mm * 1e-3

# 初始化 PROPER
beam_diameter_m = 2 * w0_m
beam_diam_fraction = beam_diameter_m / physical_size_m

wfo = proper.prop_begin(
    beam_diameter_m,
    wavelength_m,
    grid_size,
    beam_diam_fraction,
)

# 设置高斯光束参数
wfo.w0 = w0_m
wfo.z_Rayleigh = np.pi * w0_m**2 / wavelength_m
wfo.z = 0.0  # 在束腰处
wfo.z_w0 = 0.0

print(f"束腰半径: {w0_mm} mm")
print(f"瑞利长度: {wfo.z_Rayleigh * 1e3:.2f} mm")
print(f"网格尺寸: {physical_size_mm} mm")
print(f"边缘距离 / 束腰半径: {physical_size_mm/2 / w0_mm:.1f}")

# 获取初始相位
phase_initial = proper.prop_get_phase(wfo)
amp_initial = proper.prop_get_amplitude(wfo)

# 创建坐标网格
sampling_m = proper.prop_get_sampling(wfo)
sampling_mm = sampling_m * 1e3
half_size = sampling_mm * grid_size / 2
coords = np.linspace(-half_size, half_size, grid_size)
X, Y = np.meshgrid(coords, coords)
r_grid = np.sqrt(X**2 + Y**2)

print(f"\n初始相位范围: [{np.min(phase_initial):.6f}, {np.max(phase_initial):.6f}] rad")
print(f"初始振幅范围: [{np.min(amp_initial):.9f}, {np.max(amp_initial):.6f}]")

print("\n" + "=" * 60)
print("【分析 1】在束腰处的相位")
print("=" * 60)

# 在束腰处，高斯光束的相位应该是 0（平面波前）
# 但 PROPER 可能有 Gouy 相位

# 检查不同半径处的相位
r_values = [0, 5, 10, 15, 20]
print("\n不同半径处的相位（在束腰处）:")
for r_target in r_values:
    mask = np.abs(r_grid - r_target) < 0.5
    if np.sum(mask) > 0:
        mean_phase = np.mean(phase_initial[mask])
        mean_amp = np.mean(amp_initial[mask])
        print(f"  r ≈ {r_target:2d} mm: 相位 = {mean_phase:.6f} rad, 振幅 = {mean_amp:.6f}")

print("\n" + "=" * 60)
print("【分析 2】传播一段距离后的相位")
print("=" * 60)

# 传播 40 mm（与测试案例相同）
proper.prop_propagate(wfo, 0.04)  # 40 mm = 0.04 m

phase_after = proper.prop_get_phase(wfo)
amp_after = proper.prop_get_amplitude(wfo)

print(f"传播后相位范围: [{np.min(phase_after):.6f}, {np.max(phase_after):.6f}] rad")
print(f"传播后振幅范围: [{np.min(amp_after):.9f}, {np.max(amp_after):.6f}]")

# 更新坐标（采样可能变化）
sampling_m = proper.prop_get_sampling(wfo)
sampling_mm = sampling_m * 1e3
half_size = sampling_mm * grid_size / 2
coords = np.linspace(-half_size, half_size, grid_size)
X, Y = np.meshgrid(coords, coords)
r_grid = np.sqrt(X**2 + Y**2)
r_sq_grid = X**2 + Y**2

print(f"\n传播后采样: {sampling_mm:.6f} mm/pixel")
print(f"传播后网格尺寸: {sampling_mm * grid_size:.2f} mm")

# 计算 Pilot Beam 相位
z_mm = 40.0
z_R_mm = wfo.z_Rayleigh * 1e3
R_mm = z_mm * (1 + (z_R_mm / z_mm)**2)  # 严格公式
wavelength_mm = wavelength_um * 1e-3
k = 2 * np.pi / wavelength_mm

pilot_phase = k * r_sq_grid / (2 * R_mm)
gouy_phase = np.arctan(z_mm / z_R_mm)

print(f"\nPilot Beam 曲率半径: {R_mm:.2f} mm")
print(f"Gouy 相位: {gouy_phase:.6f} rad = {gouy_phase/(2*np.pi)*1000:.4f} milli-waves")

# 比较
print("\n不同半径处的相位比较（传播后）:")
for r_target in r_values:
    mask = np.abs(r_grid - r_target) < 0.5
    if np.sum(mask) > 0:
        mean_proper = np.mean(phase_after[mask])
        mean_pilot = np.mean(pilot_phase[mask])
        mean_expected = mean_pilot + gouy_phase
        diff = (mean_proper - mean_expected) / (2 * np.pi) * 1000
        print(f"  r ≈ {r_target:2d} mm: PROPER = {mean_proper:.6f}, "
              f"Pilot+Gouy = {mean_expected:.6f}, 差异 = {diff:.4f} milli-waves")

print("\n" + "=" * 60)
print("【分析 3】高斯光束的精确相位公式")
print("=" * 60)

# 高斯光束的精确相位公式（不是近轴近似）
# φ(r,z) = k*z + k*r²/(2*R(z)) - ψ(z)
# 其中 ψ(z) = arctan(z/z_R) 是 Gouy 相位
# R(z) = z * (1 + (z_R/z)²) 是曲率半径

# 但这个公式假设 r << z，在边缘区域可能不准确

# 更精确的公式考虑了高阶项
# 在远离光轴的区域，相位可能有额外的贡献

# 检查 PROPER 相位是否包含高阶项
# 拟合 PROPER 相位为 a*r² + b*r⁴ + c 的形式

r_flat = r_grid.flatten()
phase_flat = phase_after.flatten()

# 只使用有效区域（振幅 > 0.001 * max）
amp_flat = amp_after.flatten()
valid_mask = amp_flat > 0.001 * np.max(amp_flat)

r_valid = r_flat[valid_mask]
phase_valid = phase_flat[valid_mask]

# 二次拟合
A2 = np.column_stack([r_valid**2, np.ones_like(r_valid)])
coeffs2, _, _, _ = np.linalg.lstsq(A2, phase_valid, rcond=None)
a2, c2 = coeffs2

# 四次拟合
A4 = np.column_stack([r_valid**2, r_valid**4, np.ones_like(r_valid)])
coeffs4, _, _, _ = np.linalg.lstsq(A4, phase_valid, rcond=None)
a4, b4, c4 = coeffs4

print(f"二次拟合: phase = {a2:.9f} * r² + {c2:.9f}")
print(f"四次拟合: phase = {a4:.9f} * r² + {b4:.12f} * r⁴ + {c4:.9f}")

# 计算残差
residual2 = phase_valid - (a2 * r_valid**2 + c2)
residual4 = phase_valid - (a4 * r_valid**2 + b4 * r_valid**4 + c4)

print(f"\n二次拟合残差 RMS: {np.std(residual2)/(2*np.pi)*1000:.6f} milli-waves")
print(f"四次拟合残差 RMS: {np.std(residual4)/(2*np.pi)*1000:.6f} milli-waves")

# 检查四次项的贡献
r_max = np.max(r_valid)
quartic_contribution = b4 * r_max**4
print(f"\n四次项在 r = {r_max:.2f} mm 处的贡献: {quartic_contribution:.6f} rad = {quartic_contribution/(2*np.pi)*1000:.4f} milli-waves")

print("\n" + "=" * 80)
print("【结论】")
print("=" * 80)
print(f"""
分析结果：

1. PROPER 计算的相位在有效区域内（振幅 > 0.001 * max）：
   - 二次拟合残差: {np.std(residual2)/(2*np.pi)*1000:.4f} milli-waves
   - 四次拟合残差: {np.std(residual4)/(2*np.pi)*1000:.4f} milli-waves

2. 四次项系数: {b4:.12f}
   - 在 r = {r_max:.2f} mm 处贡献: {quartic_contribution/(2*np.pi)*1000:.4f} milli-waves

3. 结论：
   - 如果四次拟合残差显著小于二次拟合残差，说明 PROPER 相位包含高阶项
   - 这是高斯光束的正常物理行为，不是误差
   - Pilot Beam 只使用二次近似，在边缘区域会有偏差
""")
