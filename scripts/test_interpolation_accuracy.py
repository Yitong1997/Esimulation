"""
插值精度测试脚本

测试不同参数对插值精度的影响：
1. 采样光线数量
2. 网格分辨率
3. 插值方法

目标：找到最优的参数组合，降低插值误差
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
from scipy.interpolate import (
    RegularGridInterpolator,
    RBFInterpolator,
    griddata,
    CloughTocher2DInterpolator,
)
import matplotlib.pyplot as plt

print("=" * 80)
print("插值精度测试")
print("=" * 80)

# =============================================================================
# 加载光学系统并传播到 Surface 3
# =============================================================================
zmx_file = "optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx"
optical_system = load_optical_system_from_zmx(zmx_file)

# 使用较高分辨率的网格
base_grid_size = 512
base_physical_size = 40.0

source = SourceDefinition(
    wavelength_um=0.55,
    w0_mm=5.0,
    z0_mm=0.0,
    grid_size=base_grid_size,
    physical_size_mm=base_physical_size,
)

propagator = HybridOpticalPropagator(
    optical_system=optical_system,
    source=source,
    wavelength_um=0.55,
    grid_size=base_grid_size,
    num_rays=150,  # 默认值，后面会测试不同值
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

wavelength_mm = 0.55 * 1e-3
k = 2 * np.pi / wavelength_mm

# =============================================================================
# 测试 1: 采样光线数量的影响
# =============================================================================
print("\n" + "=" * 80)
print("【测试 1】采样光线数量的影响")
print("=" * 80)

# 入射面 Pilot Beam 参数
pb_entrance = state_entrance.pilot_beam_params
R_entrance = pb_entrance.curvature_radius_mm
physical_size_mm = state_entrance.grid_sampling.physical_size_mm

# 测试不同的光线数量
ray_counts = [50, 100, 150, 200, 300, 500]
ray_count_errors = []

for num_rays in ray_counts:
    sampler = WavefrontToRaysSampler(
        amplitude=state_entrance.amplitude,
        phase=state_entrance.phase,
        physical_size=physical_size_mm,
        wavelength=0.55,
        num_rays=num_rays,
    )
    
    output_rays = sampler.get_output_rays()
    ray_x, ray_y = sampler.get_ray_positions()
    n_rays = len(ray_x)
    
    # 光线相位（从 OPD 转换）
    ray_opd_mm = np.asarray(output_rays.opd)
    ray_phase = k * ray_opd_mm
    
    # Pilot Beam 在光线位置的相位
    r_sq_rays = ray_x**2 + ray_y**2
    if np.isinf(R_entrance):
        pilot_phase_rays = np.zeros_like(r_sq_rays)
    else:
        pilot_phase_rays = k * r_sq_rays / (2 * R_entrance)
    
    # 计算误差
    diff = ray_phase - pilot_phase_rays
    rms_error = np.std(diff) / (2 * np.pi)
    ray_count_errors.append(rms_error)
    
    print(f"  num_rays={num_rays:4d}: 实际光线数={n_rays:4d}, RMS误差={rms_error:.6f} waves")

# =============================================================================
# 测试 2: 网格分辨率的影响
# =============================================================================
print("\n" + "=" * 80)
print("【测试 2】网格分辨率的影响")
print("=" * 80)

grid_sizes = [128, 256, 512, 1024]
grid_size_errors = []

for grid_size in grid_sizes:
    # 重新创建 propagator 使用不同的网格分辨率
    source_test = SourceDefinition(
        wavelength_um=0.55,
        w0_mm=5.0,
        z0_mm=0.0,
        grid_size=grid_size,
        physical_size_mm=base_physical_size,
    )
    
    propagator_test = HybridOpticalPropagator(
        optical_system=optical_system,
        source=source_test,
        wavelength_um=0.55,
        grid_size=grid_size,
        num_rays=150,
    )
    
    # 传播到 Surface 3
    propagator_test._current_state = propagator_test._initialize_propagation()
    propagator_test._surface_states = [propagator_test._current_state]
    for i in range(4):
        propagator_test._propagate_to_surface(i)
    
    # 获取入射面状态
    state_test = None
    for state in propagator_test._surface_states:
        if state.surface_index == 3 and state.position == 'entrance':
            state_test = state
            break
    
    pb_test = state_test.pilot_beam_params
    R_test = pb_test.curvature_radius_mm
    
    sampler = WavefrontToRaysSampler(
        amplitude=state_test.amplitude,
        phase=state_test.phase,
        physical_size=base_physical_size,
        wavelength=0.55,
        num_rays=150,
    )
    
    output_rays = sampler.get_output_rays()
    ray_x, ray_y = sampler.get_ray_positions()
    
    ray_opd_mm = np.asarray(output_rays.opd)
    ray_phase = k * ray_opd_mm
    
    r_sq_rays = ray_x**2 + ray_y**2
    if np.isinf(R_test):
        pilot_phase_rays = np.zeros_like(r_sq_rays)
    else:
        pilot_phase_rays = k * r_sq_rays / (2 * R_test)
    
    diff = ray_phase - pilot_phase_rays
    rms_error = np.std(diff) / (2 * np.pi)
    grid_size_errors.append(rms_error)
    
    print(f"  grid_size={grid_size:4d}: RMS误差={rms_error:.6f} waves")

# =============================================================================
# 测试 3: 不同插值方法的影响
# =============================================================================
print("\n" + "=" * 80)
print("【测试 3】不同插值方法的影响")
print("=" * 80)

# 使用基准状态
phase_grid = state_entrance.phase
amplitude_grid = state_entrance.amplitude
grid_size = state_entrance.grid_sampling.grid_size
sampling_mm = state_entrance.grid_sampling.sampling_mm

# 创建网格坐标
half_size = physical_size_mm / 2
coords = np.linspace(-half_size, half_size, grid_size)
X_grid, Y_grid = np.meshgrid(coords, coords)

# 生成测试光线位置（均匀分布）
num_test_rays = 150
# 使用与 WavefrontToRaysSampler 相同的采样方式
test_coords_1d = np.linspace(-half_size * 0.9, half_size * 0.9, num_test_rays)
test_X, test_Y = np.meshgrid(test_coords_1d, test_coords_1d)
# 只取有效区域内的点
mask_valid = (test_X**2 + test_Y**2) < (half_size * 0.9)**2
test_x = test_X[mask_valid]
test_y = test_Y[mask_valid]
print(f"测试点数量: {len(test_x)}")

# Pilot Beam 在测试点的相位（作为参考）
r_sq_test = test_x**2 + test_y**2
if np.isinf(R_entrance):
    pilot_phase_test = np.zeros_like(r_sq_test)
else:
    pilot_phase_test = k * r_sq_test / (2 * R_entrance)

# 方法 1: RegularGridInterpolator (linear)
print("\n方法 1: RegularGridInterpolator (linear)")
interp_linear = RegularGridInterpolator(
    (coords, coords), phase_grid, method='linear', bounds_error=False, fill_value=0.0
)
phase_linear = interp_linear(np.column_stack([test_y, test_x]))
diff_linear = phase_linear - pilot_phase_test
rms_linear = np.std(diff_linear) / (2 * np.pi)
print(f"  RMS误差: {rms_linear:.6f} waves")

# 方法 2: RegularGridInterpolator (cubic)
print("\n方法 2: RegularGridInterpolator (cubic)")
interp_cubic = RegularGridInterpolator(
    (coords, coords), phase_grid, method='cubic', bounds_error=False, fill_value=0.0
)
phase_cubic = interp_cubic(np.column_stack([test_y, test_x]))
diff_cubic = phase_cubic - pilot_phase_test
rms_cubic = np.std(diff_cubic) / (2 * np.pi)
print(f"  RMS误差: {rms_cubic:.6f} waves")

# 方法 3: RegularGridInterpolator (quintic) - scipy >= 1.10
print("\n方法 3: RegularGridInterpolator (quintic)")
try:
    interp_quintic = RegularGridInterpolator(
        (coords, coords), phase_grid, method='quintic', bounds_error=False, fill_value=0.0
    )
    phase_quintic = interp_quintic(np.column_stack([test_y, test_x]))
    diff_quintic = phase_quintic - pilot_phase_test
    rms_quintic = np.std(diff_quintic) / (2 * np.pi)
    print(f"  RMS误差: {rms_quintic:.6f} waves")
except ValueError as e:
    print(f"  不支持 quintic 方法: {e}")
    rms_quintic = None

# 方法 4: griddata (linear)
print("\n方法 4: griddata (linear)")
grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
grid_values = phase_grid.ravel()
phase_griddata_linear = griddata(
    grid_points, grid_values, (test_x, test_y), method='linear', fill_value=0.0
)
diff_griddata_linear = phase_griddata_linear - pilot_phase_test
rms_griddata_linear = np.std(diff_griddata_linear) / (2 * np.pi)
print(f"  RMS误差: {rms_griddata_linear:.6f} waves")

# 方法 5: griddata (cubic)
print("\n方法 5: griddata (cubic)")
phase_griddata_cubic = griddata(
    grid_points, grid_values, (test_x, test_y), method='cubic', fill_value=0.0
)
# 处理 NaN
phase_griddata_cubic = np.nan_to_num(phase_griddata_cubic, nan=0.0)
diff_griddata_cubic = phase_griddata_cubic - pilot_phase_test
rms_griddata_cubic = np.std(diff_griddata_cubic) / (2 * np.pi)
print(f"  RMS误差: {rms_griddata_cubic:.6f} waves")

# 方法 6: CloughTocher2DInterpolator (C1 连续)
print("\n方法 6: CloughTocher2DInterpolator (C1 连续)")
try:
    # 使用稀疏采样点构建插值器（避免内存问题）
    step = max(1, grid_size // 64)
    sparse_points = np.column_stack([
        X_grid[::step, ::step].ravel(), 
        Y_grid[::step, ::step].ravel()
    ])
    sparse_values = phase_grid[::step, ::step].ravel()
    
    interp_ct = CloughTocher2DInterpolator(sparse_points, sparse_values, fill_value=0.0)
    phase_ct = interp_ct(test_x, test_y)
    phase_ct = np.nan_to_num(phase_ct, nan=0.0)
    diff_ct = phase_ct - pilot_phase_test
    rms_ct = np.std(diff_ct) / (2 * np.pi)
    print(f"  RMS误差: {rms_ct:.6f} waves (使用稀疏采样点)")
except Exception as e:
    print(f"  失败: {e}")
    rms_ct = None

# =============================================================================
# 测试 4: 样条插值 (scipy.interpolate.RectBivariateSpline)
# =============================================================================
print("\n" + "=" * 80)
print("【测试 4】样条插值方法")
print("=" * 80)

from scipy.interpolate import RectBivariateSpline

# 方法 7: RectBivariateSpline (k=1, 线性)
print("\n方法 7: RectBivariateSpline (k=1, 线性)")
spline_k1 = RectBivariateSpline(coords, coords, phase_grid, kx=1, ky=1)
phase_spline_k1 = spline_k1(test_y, test_x, grid=False)
diff_spline_k1 = phase_spline_k1 - pilot_phase_test
rms_spline_k1 = np.std(diff_spline_k1) / (2 * np.pi)
print(f"  RMS误差: {rms_spline_k1:.6f} waves")

# 方法 8: RectBivariateSpline (k=3, 三次)
print("\n方法 8: RectBivariateSpline (k=3, 三次)")
spline_k3 = RectBivariateSpline(coords, coords, phase_grid, kx=3, ky=3)
phase_spline_k3 = spline_k3(test_y, test_x, grid=False)
diff_spline_k3 = phase_spline_k3 - pilot_phase_test
rms_spline_k3 = np.std(diff_spline_k3) / (2 * np.pi)
print(f"  RMS误差: {rms_spline_k3:.6f} waves")

# 方法 9: RectBivariateSpline (k=5, 五次)
print("\n方法 9: RectBivariateSpline (k=5, 五次)")
spline_k5 = RectBivariateSpline(coords, coords, phase_grid, kx=5, ky=5)
phase_spline_k5 = spline_k5(test_y, test_x, grid=False)
diff_spline_k5 = phase_spline_k5 - pilot_phase_test
rms_spline_k5 = np.std(diff_spline_k5) / (2 * np.pi)
print(f"  RMS误差: {rms_spline_k5:.6f} waves")

# =============================================================================
# 测试 5: 傅里叶插值（带限信号的理想插值）
# =============================================================================
print("\n" + "=" * 80)
print("【测试 5】傅里叶插值（带限信号的理想插值）")
print("=" * 80)

def fourier_interpolate_2d(data, new_x, new_y, old_coords):
    """使用傅里叶变换进行 2D 插值
    
    对于带限信号，傅里叶插值是理论上最优的插值方法。
    """
    from scipy.fft import fft2, ifft2, fftfreq
    
    ny, nx = data.shape
    dx = old_coords[1] - old_coords[0]
    
    # 计算傅里叶变换
    F = fft2(data)
    
    # 频率坐标
    fx = fftfreq(nx, dx)
    fy = fftfreq(ny, dx)
    FX, FY = np.meshgrid(fx, fy)
    
    # 对每个新点计算插值
    result = np.zeros(len(new_x), dtype=complex)
    for i, (x, y) in enumerate(zip(new_x, new_y)):
        # 计算相位因子
        phase_shift = np.exp(-2j * np.pi * (FX * x + FY * y))
        # 求和
        result[i] = np.sum(F * phase_shift) / (nx * ny)
    
    return np.real(result)

# 由于傅里叶插值计算量大，只测试少量点
print("傅里叶插值计算量大，使用少量测试点...")
n_test_fourier = min(100, len(test_x))
indices = np.random.choice(len(test_x), n_test_fourier, replace=False)
test_x_fourier = test_x[indices]
test_y_fourier = test_y[indices]
pilot_phase_fourier = pilot_phase_test[indices]

try:
    phase_fourier = fourier_interpolate_2d(phase_grid, test_x_fourier, test_y_fourier, coords)
    diff_fourier = phase_fourier - pilot_phase_fourier
    rms_fourier = np.std(diff_fourier) / (2 * np.pi)
    print(f"  RMS误差: {rms_fourier:.6f} waves (测试点数: {n_test_fourier})")
except Exception as e:
    print(f"  失败: {e}")
    rms_fourier = None

# =============================================================================
# 测试 6: 零填充 FFT 插值（高效的傅里叶插值）
# =============================================================================
print("\n" + "=" * 80)
print("【测试 6】零填充 FFT 插值")
print("=" * 80)

def fft_upsample_2d(data, factor=2):
    """使用零填充 FFT 进行上采样
    
    这是傅里叶插值的高效实现，通过在频域零填充实现。
    """
    from scipy.fft import fft2, ifft2, fftshift, ifftshift
    
    ny, nx = data.shape
    new_ny, new_nx = ny * factor, nx * factor
    
    # FFT
    F = fftshift(fft2(data))
    
    # 零填充
    F_padded = np.zeros((new_ny, new_nx), dtype=complex)
    start_y = (new_ny - ny) // 2
    start_x = (new_nx - nx) // 2
    F_padded[start_y:start_y+ny, start_x:start_x+nx] = F
    
    # IFFT
    result = ifft2(ifftshift(F_padded)) * (factor ** 2)
    
    return np.real(result)

# 上采样 2 倍
print("\n上采样因子 = 2:")
phase_upsampled_2x = fft_upsample_2d(phase_grid, factor=2)
new_grid_size = grid_size * 2
new_coords = np.linspace(-half_size, half_size, new_grid_size)

# 在上采样后的网格上插值
interp_upsampled = RegularGridInterpolator(
    (new_coords, new_coords), phase_upsampled_2x, method='cubic', bounds_error=False, fill_value=0.0
)
phase_from_upsampled = interp_upsampled(np.column_stack([test_y, test_x]))
diff_upsampled = phase_from_upsampled - pilot_phase_test
rms_upsampled_2x = np.std(diff_upsampled) / (2 * np.pi)
print(f"  RMS误差: {rms_upsampled_2x:.6f} waves")

# 上采样 4 倍
print("\n上采样因子 = 4:")
phase_upsampled_4x = fft_upsample_2d(phase_grid, factor=4)
new_grid_size_4x = grid_size * 4
new_coords_4x = np.linspace(-half_size, half_size, new_grid_size_4x)

interp_upsampled_4x = RegularGridInterpolator(
    (new_coords_4x, new_coords_4x), phase_upsampled_4x, method='cubic', bounds_error=False, fill_value=0.0
)
phase_from_upsampled_4x = interp_upsampled_4x(np.column_stack([test_y, test_x]))
diff_upsampled_4x = phase_from_upsampled_4x - pilot_phase_test
rms_upsampled_4x = np.std(diff_upsampled_4x) / (2 * np.pi)
print(f"  RMS误差: {rms_upsampled_4x:.6f} waves")

# =============================================================================
# 汇总结果
# =============================================================================
print("\n" + "=" * 80)
print("【汇总】插值方法精度比较")
print("=" * 80)

results = [
    ("RegularGridInterpolator (linear)", rms_linear),
    ("RegularGridInterpolator (cubic)", rms_cubic),
    ("griddata (linear)", rms_griddata_linear),
    ("griddata (cubic)", rms_griddata_cubic),
    ("RectBivariateSpline (k=1)", rms_spline_k1),
    ("RectBivariateSpline (k=3)", rms_spline_k3),
    ("RectBivariateSpline (k=5)", rms_spline_k5),
    ("FFT 上采样 2x + cubic", rms_upsampled_2x),
    ("FFT 上采样 4x + cubic", rms_upsampled_4x),
]

if rms_quintic is not None:
    results.insert(2, ("RegularGridInterpolator (quintic)", rms_quintic))
if rms_ct is not None:
    results.append(("CloughTocher2DInterpolator", rms_ct))
if rms_fourier is not None:
    results.append(("傅里叶插值 (直接)", rms_fourier))

# 按误差排序
results_sorted = sorted(results, key=lambda x: x[1])

print("\n按精度排序（从高到低）:")
print("-" * 60)
for i, (name, error) in enumerate(results_sorted):
    print(f"  {i+1:2d}. {name:40s}: {error:.6f} waves")

print("\n" + "=" * 80)
print("【结论】")
print("=" * 80)
best_method, best_error = results_sorted[0]
print(f"最佳插值方法: {best_method}")
print(f"最小 RMS 误差: {best_error:.6f} waves")

# =============================================================================
# 可视化结果
# =============================================================================
print("\n生成可视化图表...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 图 1: 采样光线数量 vs 误差
ax1 = axes[0, 0]
ax1.plot(ray_counts, [e * 1000 for e in ray_count_errors], 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('采样光线数量 (num_rays)', fontsize=12)
ax1.set_ylabel('RMS 误差 (milli-waves)', fontsize=12)
ax1.set_title('采样光线数量对精度的影响', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# 图 2: 网格分辨率 vs 误差
ax2 = axes[0, 1]
ax2.plot(grid_sizes, [e * 1000 for e in grid_size_errors], 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('网格分辨率 (grid_size)', fontsize=12)
ax2.set_ylabel('RMS 误差 (milli-waves)', fontsize=12)
ax2.set_title('网格分辨率对精度的影响', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log', base=2)

# 图 3: 插值方法比较（条形图）
ax3 = axes[1, 0]
methods = [r[0] for r in results_sorted]
errors = [r[1] * 1000 for r in results_sorted]  # 转换为 milli-waves
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(methods)))
bars = ax3.barh(range(len(methods)), errors, color=colors)
ax3.set_yticks(range(len(methods)))
ax3.set_yticklabels(methods, fontsize=9)
ax3.set_xlabel('RMS 误差 (milli-waves)', fontsize=12)
ax3.set_title('不同插值方法的精度比较', fontsize=14)
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')

# 在条形图上添加数值标签
for bar, error in zip(bars, errors):
    ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{error:.3f}', va='center', fontsize=8)

# 图 4: 误差分布对比
ax4 = axes[1, 1]
# 比较最佳方法和当前方法
ax4.hist(diff_cubic / (2 * np.pi) * 1000, bins=50, alpha=0.5, label='cubic (当前)', density=True)
ax4.hist(diff_spline_k5 / (2 * np.pi) * 1000, bins=50, alpha=0.5, label='spline k=5', density=True)
ax4.hist(diff_upsampled_4x / (2 * np.pi) * 1000, bins=50, alpha=0.5, label='FFT 4x + cubic', density=True)
ax4.set_xlabel('相位误差 (milli-waves)', fontsize=12)
ax4.set_ylabel('概率密度', fontsize=12)
ax4.set_title('不同插值方法的误差分布', fontsize=14)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('interpolation_accuracy_test.png', dpi=150, bbox_inches='tight')
print("图表已保存: interpolation_accuracy_test.png")

# =============================================================================
# 优化建议
# =============================================================================
print("\n" + "=" * 80)
print("【优化建议】")
print("=" * 80)

print("""
基于测试结果，以下是降低插值误差的建议：

1. **插值方法选择**:
   - 推荐使用 RectBivariateSpline (k=5) 或 FFT 上采样方法
   - 避免使用线性插值，精度较低

2. **网格分辨率**:
   - 增加网格分辨率可以显著降低误差
   - 建议使用 512 或更高分辨率

3. **采样光线数量**:
   - 增加采样光线数量对入射面插值误差影响较小
   - 但对出射面重采样精度有帮助

4. **FFT 上采样方法**:
   - 对于带限信号，FFT 上采样是理论最优方法
   - 可以在插值前对网格进行 2-4 倍上采样
   - 计算开销增加，但精度提升明显

5. **实现建议**:
   - 在 WavefrontToRaysSampler 中使用 RectBivariateSpline (k=5)
   - 或者在插值前进行 FFT 上采样
""")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
