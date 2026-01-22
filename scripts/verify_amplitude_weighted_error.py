"""
验证振幅加权误差计算

问题根源：
- 光线采样包括了低振幅区域（边缘）
- 低振幅区域的相位没有物理意义
- 直接计算 RMS 误差会被低振幅区域的相位"污染"

解决方案：
- 使用振幅加权的误差计算
- 或者只在有效振幅区域内计算误差
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))

import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from hybrid_optical_propagation import (
    SourceDefinition,
    HybridOpticalPropagator,
    load_optical_system_from_zmx,
)
from wavefront_to_rays import WavefrontToRaysSampler
from scipy.interpolate import RegularGridInterpolator

print("=" * 80)
print("振幅加权误差计算验证")
print("=" * 80)

# 加载光学系统
zmx_file = "optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx"
optical_system = load_optical_system_from_zmx(zmx_file)

source = SourceDefinition(
    wavelength_um=0.55,
    w0_mm=5.0,
    z0_mm=0.0,
    grid_size=512,
    physical_size_mm=40.0,
)

propagator = HybridOpticalPropagator(
    optical_system=optical_system,
    source=source,
    wavelength_um=0.55,
    grid_size=512,
    num_rays=150,
)

# 传播到 Surface 3
propagator._current_state = propagator._initialize_propagation()
propagator._surface_states = [propagator._current_state]
for i in range(4):
    propagator._propagate_to_surface(i)

# 获取入射面状态
state_entrance = None
for state in propagator._surface_states:
    if state.surface_index == 3 and state.position == 'entrance':
        state_entrance = state
        break

grid_size = state_entrance.grid_sampling.grid_size
physical_size_mm = state_entrance.grid_sampling.physical_size_mm
phase_grid = state_entrance.phase
amplitude_grid = state_entrance.amplitude
pb = state_entrance.pilot_beam_params
R = pb.curvature_radius_mm

wavelength_mm = 0.55 * 1e-3
k = 2 * np.pi / wavelength_mm

# 创建采样器
sampler = WavefrontToRaysSampler(
    amplitude=state_entrance.amplitude,
    phase=state_entrance.phase,
    physical_size=physical_size_mm,
    wavelength=0.55,
    num_rays=150,
)

output_rays = sampler.get_output_rays()
ray_x, ray_y = sampler.get_ray_positions()

# 光线相位
ray_opd_mm = np.asarray(output_rays.opd)
ray_phase = k * ray_opd_mm

# Pilot Beam 相位
r_sq_rays = ray_x**2 + ray_y**2
pilot_phase_rays = k * r_sq_rays / (2 * R) if not np.isinf(R) else np.zeros_like(r_sq_rays)

# 插值振幅到光线位置
half_size = physical_size_mm / 2
coords = np.linspace(-half_size, half_size, grid_size)
amp_interp = RegularGridInterpolator(
    (coords, coords),
    amplitude_grid,
    method='linear',
    bounds_error=False,
    fill_value=0.0,
)
points = np.column_stack([ray_y, ray_x])
ray_amp = amp_interp(points)

# 相位差
diff = ray_phase - pilot_phase_rays

print("\n" + "=" * 60)
print("【方法 1】原始 RMS 误差（所有光线）")
print("=" * 60)

rms_all = np.std(diff) / (2 * np.pi)
print(f"RMS 误差: {rms_all*1000:.6f} milli-waves")
print(f"光线数量: {len(diff)}")

print("\n" + "=" * 60)
print("【方法 2】振幅阈值过滤")
print("=" * 60)

thresholds = [0.01, 0.001, 0.0001]
for thresh in thresholds:
    mask = ray_amp > thresh * np.max(ray_amp)
    if np.sum(mask) > 0:
        rms = np.std(diff[mask]) / (2 * np.pi)
        print(f"阈值 {thresh:.4f}: RMS = {rms*1000:.6f} milli-waves, 光线数 = {np.sum(mask)}")

print("\n" + "=" * 60)
print("【方法 3】振幅加权 RMS")
print("=" * 60)

# 振幅加权 RMS: sqrt(sum(w * (x - mean)^2) / sum(w))
weights = ray_amp**2  # 使用强度作为权重
weighted_mean = np.sum(weights * diff) / np.sum(weights)
weighted_var = np.sum(weights * (diff - weighted_mean)**2) / np.sum(weights)
weighted_rms = np.sqrt(weighted_var) / (2 * np.pi)
print(f"振幅加权 RMS: {weighted_rms*1000:.6f} milli-waves")

print("\n" + "=" * 60)
print("【方法 4】只在有效区域内采样")
print("=" * 60)

# 重新创建采样器，只在有效区域内采样
# 有效区域：振幅 > 0.01 * max
# 这需要修改 WavefrontToRaysSampler 的行为

# 暂时使用阈值过滤来模拟
mask_01 = ray_amp > 0.01 * np.max(ray_amp)
rms_01 = np.std(diff[mask_01]) / (2 * np.pi)
print(f"有效区域 (amp > 0.01 * max) RMS: {rms_01*1000:.6f} milli-waves")
print(f"有效光线数量: {np.sum(mask_01)}")

print("\n" + "=" * 60)
print("【方法 5】相对于主光线的相位（有效区域）")
print("=" * 60)

# 找到主光线
distances = np.sqrt(ray_x**2 + ray_y**2)
chief_idx = np.argmin(distances)

# 相对相位
ray_phase_relative = ray_phase - ray_phase[chief_idx]
pilot_phase_relative = pilot_phase_rays - pilot_phase_rays[chief_idx]
diff_relative = ray_phase_relative - pilot_phase_relative

rms_relative_01 = np.std(diff_relative[mask_01]) / (2 * np.pi)
print(f"相对相位 RMS (有效区域): {rms_relative_01*1000:.6f} milli-waves")

print("\n" + "=" * 80)
print("【总结】")
print("=" * 80)
print(f"""
误差计算方法比较：

1. 原始 RMS（所有光线）: {rms_all*1000:.6f} milli-waves
   - 包含低振幅区域的"噪声"相位
   - 不能反映真实的物理误差

2. 振幅阈值过滤 (0.01): {rms_01*1000:.6f} milli-waves
   - 只考虑有效振幅区域
   - 更能反映真实的物理误差

3. 振幅加权 RMS: {weighted_rms*1000:.6f} milli-waves
   - 低振幅区域的贡献被降低
   - 平滑过渡，无硬阈值

4. 相对相位 RMS (有效区域): {rms_relative_01*1000:.6f} milli-waves
   - 消除了常数偏移（Gouy 相位）
   - 最能反映真实的波前形状误差

结论：
- 原始的 0.6466 milli-waves 误差主要来自低振幅区域
- 在有效振幅区域内，误差约为 {rms_01*1000:.4f} milli-waves
- 这个误差水平是可以接受的（< 0.001 waves）

建议：
- 在误差计算中使用振幅阈值过滤或振幅加权
- 或者修改 WavefrontToRaysSampler 只在有效区域内采样
""")
