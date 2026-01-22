"""
调试 PROPER 参考面切换行为

PROPER 在什么条件下从 PLANAR 切换到 SPHERI？
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import proper


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


print_section("PROPER 参考面切换条件")

wavelength_m = 0.55e-6
grid_size = 256
physical_size_m = 0.02  # 20 mm

# 测试不同的 beam_ratio
for beam_ratio in [0.1, 0.3, 0.5]:
    print(f"\n--- beam_ratio = {beam_ratio} ---")
    
    wfo = proper.prop_begin(physical_size_m, wavelength_m, grid_size, beam_ratio)
    
    print(f"初始状态:")
    print(f"  wfo.w0 = {wfo.w0 * 1e3:.2f} mm")
    print(f"  wfo.z_Rayleigh = {wfo.z_Rayleigh * 1e3:.2f} mm")
    print(f"  rayleigh_factor = {proper.rayleigh_factor}")
    print(f"  切换阈值 = {proper.rayleigh_factor * wfo.z_Rayleigh * 1e3:.2f} mm")
    print(f"  reference_surface = {wfo.reference_surface}")
    
    # 传播不同距离
    for dist_mm in [10, 100, 1000, 10000, 100000, 1000000]:
        wfo2 = proper.prop_begin(physical_size_m, wavelength_m, grid_size, beam_ratio)
        proper.prop_propagate(wfo2, dist_mm * 1e-3)
        
        z_over_threshold = dist_mm / (proper.rayleigh_factor * wfo2.z_Rayleigh * 1e3)
        print(f"  传播 {dist_mm:>7} mm: z/阈值 = {z_over_threshold:.4f}, ref = {wfo2.reference_surface}")


print_section("检查 PROPER 内部的瑞利长度计算")

# PROPER 的瑞利长度计算
# z_R = π × w0² / λ

for beam_ratio in [0.1, 0.3, 0.5]:
    wfo = proper.prop_begin(physical_size_m, wavelength_m, grid_size, beam_ratio)
    
    # PROPER 计算的瑞利长度
    z_R_proper = wfo.z_Rayleigh
    
    # 我们期望的瑞利长度（基于 beam_ratio）
    w0_expected = physical_size_m * beam_ratio / 2
    z_R_expected = np.pi * w0_expected**2 / wavelength_m
    
    print(f"\nbeam_ratio = {beam_ratio}:")
    print(f"  PROPER w0 = {wfo.w0 * 1e3:.2f} mm")
    print(f"  期望 w0 = {w0_expected * 1e3:.2f} mm")
    print(f"  PROPER z_R = {z_R_proper * 1e3:.2f} mm")
    print(f"  期望 z_R = {z_R_expected * 1e3:.2f} mm")


print_section("使用 ZMX 文件中的实际参数")

# 从 debug_surface3_step_by_step.py 的输出中获取的参数
# Surface_3 entrance:
#   Pilot Beam 曲率半径: 509793655.76 mm
#   这意味着 z/z_R 非常小

# 让我们模拟这种情况
wavelength_um = 0.55
wavelength_m = wavelength_um * 1e-6
w0_mm = 5.0  # 假设的束腰半径
w0_m = w0_mm * 1e-3
z_R_mm = np.pi * w0_mm**2 / (wavelength_um * 1e-3)
z_R_m = z_R_mm * 1e-3

print(f"假设的高斯光束参数:")
print(f"  w0 = {w0_mm} mm")
print(f"  z_R = {z_R_mm:.2f} mm")

# 传播 40mm（与 ZMX 文件中的距离类似）
propagation_mm = 40.0
z_over_z_R = propagation_mm / z_R_mm

print(f"\n传播 {propagation_mm} mm:")
print(f"  z/z_R = {z_over_z_R:.6f}")
print(f"  严格曲率半径 R = z × (1 + (z_R/z)²) = {propagation_mm * (1 + (z_R_mm/propagation_mm)**2):.2f} mm")

# 这解释了为什么曲率半径如此巨大：z << z_R 时，R ≈ z_R²/z


print_section("关键发现")

print("""
1. PROPER 的参考面切换条件：
   - 当 |z - z_w0| < rayleigh_factor × z_Rayleigh 时，使用 PLANAR
   - 当 |z - z_w0| >= rayleigh_factor × z_Rayleigh 时，使用 SPHERI
   
2. 在我们的测试中，瑞利长度非常大（~143000 mm），所以即使传播 40mm，
   仍然在 PLANAR 区域内。

3. 这意味着 PROPER 的残差相位就是绝对相位（因为参考面是平面）。

4. 但是，PROPER 的残差相位是通过 np.angle() 计算的，范围是 [-π, π]！
   这就是折叠的来源！

5. 当波前曲率较大时（即使参考面是 PLANAR），wfarr 中的相位可能超过 [-π, π]，
   但 prop_get_phase 返回的是 np.angle(wfarr)，会被折叠。
""")


print_section("验证：PROPER 的 prop_get_phase 是否使用 np.angle")

# 创建一个有大相位的波前
wfo = proper.prop_begin(0.02, 0.55e-6, 256, 0.5)

# 手动添加一个大相位
n = 256
x = np.linspace(-0.01, 0.01, n)
X, Y = np.meshgrid(x, x)
r_sq = X**2 + Y**2

# 添加一个球面相位（模拟透镜效果）
k = 2 * np.pi / 0.55e-6
R = 0.1  # 100mm 曲率半径
phase_to_add = k * r_sq / (2 * R)

print(f"添加的相位范围: [{np.min(phase_to_add):.2f}, {np.max(phase_to_add):.2f}] rad")
print(f"添加的相位范围（波长数）: [{np.min(phase_to_add)/(2*np.pi):.2f}, {np.max(phase_to_add)/(2*np.pi):.2f}] waves")

# 乘以相位
proper.prop_multiply(wfo, np.exp(1j * phase_to_add))

# 提取相位
extracted_phase = proper.prop_get_phase(wfo)

print(f"\n提取的相位范围: [{np.min(extracted_phase):.2f}, {np.max(extracted_phase):.2f}] rad")
print(f"提取的相位范围（波长数）: [{np.min(extracted_phase)/(2*np.pi):.2f}, {np.max(extracted_phase)/(2*np.pi):.2f}] waves")

if np.max(extracted_phase) - np.min(extracted_phase) < 2 * np.pi + 0.1:
    print(f"\n[确认] prop_get_phase 返回的是折叠相位（范围 < 2π）！")
else:
    print(f"\n[意外] prop_get_phase 返回的相位范围超过 2π")

# 直接检查 wfarr
wfarr_centered = proper.prop_shift_center(wfo.wfarr)
wfarr_phase = np.angle(wfarr_centered)

print(f"\nnp.angle(wfarr) 范围: [{np.min(wfarr_phase):.2f}, {np.max(wfarr_phase):.2f}] rad")

# 比较
diff = extracted_phase - wfarr_phase
print(f"prop_get_phase 与 np.angle(wfarr) 的差异: {np.max(np.abs(diff)):.6f} rad")
