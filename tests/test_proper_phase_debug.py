"""
PROPER 相位处理机制深度调试

分析 PROPER 内部的参考面机制和相位存储方式。
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import proper
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# 测试 1：简单高斯光束传播，检查 PROPER 内部状态
# ============================================================

print_section("测试 1：PROPER 内部状态分析")

wavelength = 1.064e-6  # m
w0 = 10e-3  # m (10 mm)
grid_size = 256
beam_ratio = 0.5

# 初始化波前
wfo = proper.prop_begin(4 * w0, wavelength, grid_size, beam_ratio)

print(f"初始化后:")
print(f"  grid_size: {wfo.ngrid}")
print(f"  wavelength: {wfo.lamda * 1e6:.3f} μm")
print(f"  sampling: {proper.prop_get_sampling(wfo) * 1e3:.4f} mm/pixel")
print(f"  reference_surface: {wfo.reference_surface}")
print(f"  z: {wfo.z * 1e3:.3f} mm")
print(f"  z_w0: {wfo.z_w0 * 1e3:.3f} mm")
print(f"  beam_type_old: {wfo.beam_type_old}")

# 创建高斯光束
n = grid_size
sampling_m = proper.prop_get_sampling(wfo)
half_size = sampling_m * n / 2
coords = np.linspace(-half_size, half_size, n)
X, Y = np.meshgrid(coords, coords)
R_sq = X**2 + Y**2

# 高斯振幅
amplitude = np.exp(-R_sq / w0**2)
gaussian_field = proper.prop_shift_center(amplitude.astype(complex))
wfo.wfarr = wfo.wfarr * gaussian_field

print(f"\n应用高斯振幅后:")
print(f"  reference_surface: {wfo.reference_surface}")
print(f"  z: {wfo.z * 1e3:.3f} mm")
print(f"  z_w0: {wfo.z_w0 * 1e3:.3f} mm")


# ============================================================
# 测试 2：传播后检查参考面变化
# ============================================================

print_section("测试 2：传播后参考面变化")

# 传播 50mm
prop_distance = 50e-3  # m
proper.prop_propagate(wfo, prop_distance)

print(f"传播 {prop_distance*1e3:.0f} mm 后:")
print(f"  reference_surface: {wfo.reference_surface}")
print(f"  z: {wfo.z * 1e3:.3f} mm")
print(f"  z_w0: {wfo.z_w0 * 1e3:.3f} mm")

# 获取相位
phase_proper = proper.prop_get_phase(wfo)
amplitude_proper = proper.prop_get_amplitude(wfo)

# 分析相位
phase_centered = proper.prop_shift_center(phase_proper)
amp_centered = proper.prop_shift_center(amplitude_proper)

# 掩模
mask = amp_centered > 0.1 * np.max(amp_centered)

print(f"\n相位分析:")
print(f"  相位范围: [{np.min(phase_centered[mask]):.2f}, {np.max(phase_centered[mask]):.2f}] rad")
print(f"  相位 PV: {np.max(phase_centered[mask]) - np.min(phase_centered[mask]):.2f} rad")

# 理论相位（高斯光束传播后的球面波前）
# 传播 50mm 后，波前曲率半径 R = z + zR^2/z
zR = np.pi * w0**2 / wavelength  # 瑞利距离
z = prop_distance
R_theory = z + zR**2 / z if z != 0 else float('inf')
print(f"\n理论预测:")
print(f"  瑞利距离 zR: {zR * 1e3:.0f} mm")
print(f"  波前曲率半径 R: {R_theory * 1e3:.0f} mm")

# 理论相位
k = 2 * np.pi / wavelength
if np.isfinite(R_theory):
    phase_theory = -k * R_sq / (2 * R_theory)
    phase_theory_centered = phase_theory - np.mean(phase_theory[mask])
else:
    phase_theory_centered = np.zeros_like(R_sq)

print(f"  理论相位范围: [{np.min(phase_theory_centered[mask]):.2f}, {np.max(phase_theory_centered[mask]):.2f}] rad")


# ============================================================
# 测试 3：检查 PROPER 的参考球面补偿
# ============================================================

print_section("测试 3：PROPER 参考球面机制")

print("""
PROPER 使用参考球面跟踪机制：
- 当 reference_surface == "SPHERI" 时，PROPER 内部存储的是相对于参考球面的相位偏差
- 参考球面的曲率半径 R_ref = z - z_w0
- 真实相位 = 存储相位 + 参考球面相位

这意味着：
- 对于理想高斯光束，存储的相位偏差应该接近零
- 我们需要加上参考球面的相位才能得到真实的绝对相位
""")

# 计算参考球面的曲率半径
R_ref = wfo.z - wfo.z_w0
print(f"参考球面曲率半径 R_ref = z - z_w0 = {R_ref * 1e3:.3f} mm")
print(f"理论波前曲率半径 R_theory = {R_theory * 1e3:.0f} mm")

# 参考球面相位
if abs(R_ref) > 1e-10:
    phase_ref = -k * R_sq / (2 * R_ref)
    print(f"参考球面相位范围: [{np.min(phase_ref[mask]):.2f}, {np.max(phase_ref[mask]):.2f}] rad")
else:
    phase_ref = np.zeros_like(R_sq)
    print("参考球面曲率半径接近零，无参考球面相位")

# 真实绝对相位 = 存储相位 + 参考球面相位
phase_absolute = phase_centered + phase_ref
print(f"\n真实绝对相位范围: [{np.min(phase_absolute[mask]):.2f}, {np.max(phase_absolute[mask]):.2f}] rad")
print(f"理论相位范围: [{np.min(phase_theory_centered[mask]):.2f}, {np.max(phase_theory_centered[mask]):.2f}] rad")

# 比较
diff = phase_absolute - phase_theory_centered
print(f"\n差异 (绝对相位 - 理论相位):")
print(f"  范围: [{np.min(diff[mask]):.4f}, {np.max(diff[mask]):.4f}] rad")
print(f"  RMS: {np.sqrt(np.mean(diff[mask]**2)):.4f} rad")


# ============================================================
# 测试 4：应用透镜后的相位变化
# ============================================================

print_section("测试 4：应用透镜后的相位变化")

# 重新初始化
wfo2 = proper.prop_begin(4 * w0, wavelength, grid_size, beam_ratio)
gaussian_field = proper.prop_shift_center(amplitude.astype(complex))
wfo2.wfarr = wfo2.wfarr * gaussian_field

print(f"初始状态:")
print(f"  reference_surface: {wfo2.reference_surface}")
print(f"  z: {wfo2.z * 1e3:.3f} mm")
print(f"  z_w0: {wfo2.z_w0 * 1e3:.3f} mm")

# 应用透镜（焦距 -50mm，凸面镜）
f = -50e-3  # m
proper.prop_lens(wfo2, f)

print(f"\n应用透镜 (f={f*1e3:.0f}mm) 后:")
print(f"  reference_surface: {wfo2.reference_surface}")
print(f"  z: {wfo2.z * 1e3:.3f} mm")
print(f"  z_w0: {wfo2.z_w0 * 1e3:.3f} mm")

# 获取相位
phase_after_lens = proper.prop_get_phase(wfo2)
phase_after_lens_centered = proper.prop_shift_center(phase_after_lens)

print(f"\n相位分析:")
print(f"  相位范围: [{np.min(phase_after_lens_centered[mask]):.2f}, {np.max(phase_after_lens_centered[mask]):.2f}] rad")

# 传播 50mm
proper.prop_propagate(wfo2, 50e-3)

print(f"\n传播 50mm 后:")
print(f"  reference_surface: {wfo2.reference_surface}")
print(f"  z: {wfo2.z * 1e3:.3f} mm")
print(f"  z_w0: {wfo2.z_w0 * 1e3:.3f} mm")

phase_after_prop = proper.prop_get_phase(wfo2)
phase_after_prop_centered = proper.prop_shift_center(phase_after_prop)
amp_after_prop = proper.prop_get_amplitude(wfo2)
amp_after_prop_centered = proper.prop_shift_center(amp_after_prop)
mask2 = amp_after_prop_centered > 0.1 * np.max(amp_after_prop_centered)

print(f"\n相位分析:")
print(f"  相位范围: [{np.min(phase_after_prop_centered[mask2]):.2f}, {np.max(phase_after_prop_centered[mask2]):.2f}] rad")


# ============================================================
# 测试 5：检查 _record_wavefront 中的相位补偿逻辑
# ============================================================

print_section("测试 5：检查 system.py 中的相位补偿逻辑")

# 读取 system.py 中的 _record_wavefront 方法
print("""
在 system.py 的 _record_wavefront 方法中，我们尝试补偿参考球面相位：

```python
if reference_surface == "SPHERI":
    R_ref_m = wfo.z - wfo.z_w0
    if abs(R_ref_m) > 1e-10:
        phase_ref = -k * r² / (2 * R_ref)
        phase_absolute = phase_relative + phase_ref
```

但是这里有一个问题：
- PROPER 的 z 和 z_w0 是在 prop_begin 时设置的
- prop_lens 会修改 z_w0（虚拟束腰位置）
- 但 z 只在 prop_propagate 时更新

让我们检查 prop_lens 对 z_w0 的影响...
""")

# 重新测试，详细跟踪 z 和 z_w0
wfo3 = proper.prop_begin(4 * w0, wavelength, grid_size, beam_ratio)
gaussian_field = proper.prop_shift_center(amplitude.astype(complex))
wfo3.wfarr = wfo3.wfarr * gaussian_field

print(f"步骤 0 - 初始化:")
print(f"  z = {wfo3.z * 1e3:.3f} mm, z_w0 = {wfo3.z_w0 * 1e3:.3f} mm")
print(f"  R_ref = z - z_w0 = {(wfo3.z - wfo3.z_w0) * 1e3:.3f} mm")

proper.prop_lens(wfo3, -50e-3)
print(f"\n步骤 1 - 应用透镜 (f=-50mm):")
print(f"  z = {wfo3.z * 1e3:.3f} mm, z_w0 = {wfo3.z_w0 * 1e3:.3f} mm")
print(f"  R_ref = z - z_w0 = {(wfo3.z - wfo3.z_w0) * 1e3:.3f} mm")

proper.prop_propagate(wfo3, 50e-3)
print(f"\n步骤 2 - 传播 50mm:")
print(f"  z = {wfo3.z * 1e3:.3f} mm, z_w0 = {wfo3.z_w0 * 1e3:.3f} mm")
print(f"  R_ref = z - z_w0 = {(wfo3.z - wfo3.z_w0) * 1e3:.3f} mm")

proper.prop_lens(wfo3, float('inf'))  # 平面镜
print(f"\n步骤 3 - 应用平面镜:")
print(f"  z = {wfo3.z * 1e3:.3f} mm, z_w0 = {wfo3.z_w0 * 1e3:.3f} mm")
print(f"  R_ref = z - z_w0 = {(wfo3.z - wfo3.z_w0) * 1e3:.3f} mm")

proper.prop_propagate(wfo3, 50e-3)
print(f"\n步骤 4 - 传播 50mm:")
print(f"  z = {wfo3.z * 1e3:.3f} mm, z_w0 = {wfo3.z_w0 * 1e3:.3f} mm")
print(f"  R_ref = z - z_w0 = {(wfo3.z - wfo3.z_w0) * 1e3:.3f} mm")

proper.prop_lens(wfo3, 150e-3)
print(f"\n步骤 5 - 应用透镜 (f=150mm):")
print(f"  z = {wfo3.z * 1e3:.3f} mm, z_w0 = {wfo3.z_w0 * 1e3:.3f} mm")
print(f"  R_ref = z - z_w0 = {(wfo3.z - wfo3.z_w0) * 1e3:.3f} mm")

proper.prop_propagate(wfo3, 100e-3)
print(f"\n步骤 6 - 传播 100mm:")
print(f"  z = {wfo3.z * 1e3:.3f} mm, z_w0 = {wfo3.z_w0 * 1e3:.3f} mm")
print(f"  R_ref = z - z_w0 = {(wfo3.z - wfo3.z_w0) * 1e3:.3f} mm")


# ============================================================
# 总结
# ============================================================

print_section("总结")

print("""
关键发现：

1. PROPER 使用参考球面跟踪机制
   - reference_surface = "SPHERI" 表示使用球面参考
   - 存储的相位是相对于参考球面的偏差
   - 参考球面曲率半径 R_ref = z - z_w0

2. prop_lens 会修改 z_w0
   - 这会改变参考球面的曲率半径
   - 但不会改变 z（当前位置）

3. 问题根源
   - 在 _record_wavefront 中，我们使用 R_ref = z - z_w0 来补偿参考球面
   - 但 z_w0 在经过透镜后已经被修改
   - 这导致补偿的参考球面与实际的参考球面不匹配

4. 可能的解决方案
   - 方案 A：不补偿参考球面，直接使用 PROPER 存储的相对相位
   - 方案 B：跟踪真实的波前曲率半径（使用 ABCD 矩阵）
   - 方案 C：在每个采样点重新计算正确的参考球面
""")
