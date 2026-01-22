"""
调试 ZMX 文件测试中的相位折叠问题

分析 Surface_3 入射面的相位是如何被折叠的。
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import proper

from hybrid_optical_propagation import (
    create_propagator_from_zmx,
    SourceDefinition,
)


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================================
# 加载 ZMX 文件并创建传播器
# ============================================================================

print_section("加载 ZMX 文件并创建传播器")

zmx_path = "optiland-master/tests/zemax_files/simple_fold_mirror_up.zmx"

wavelength_um = 0.55
grid_size = 256
physical_size_mm = 30.0
w0_mm = 5.0

source = SourceDefinition(
    wavelength_um=wavelength_um,
    w0_mm=w0_mm,
    z0_mm=0.0,
    grid_size=grid_size,
    physical_size_mm=physical_size_mm,
)

propagator = create_propagator_from_zmx(
    filepath=zmx_path,
    source=source,
    wavelength_um=wavelength_um,
)

print(f"ZMX 文件: {zmx_path}")
print(f"波长: {wavelength_um} μm")
print(f"网格大小: {grid_size}")
print(f"物理尺寸: {physical_size_mm} mm")
print(f"初始束腰半径: {w0_mm} mm")
print(f"表面数量: {len(propagator.optical_system)}")


# ============================================================================
# 执行传播
# ============================================================================

print_section("执行传播")

result = propagator.propagate()

print(f"传播成功: {result.success}")
if not result.success:
    print(f"错误信息: {result.error_message}")

print(f"记录的状态数量: {len(result.surface_states)}")

# 打印每个状态的信息
for state in result.surface_states:
    print(f"\nSurface_{state.surface_index} {state.position}:")
    print(f"  振幅范围: [{np.min(state.amplitude):.6f}, {np.max(state.amplitude):.6f}]")
    print(f"  相位范围: [{np.min(state.phase):.6f}, {np.max(state.phase):.6f}] rad")
    print(f"  相位范围（波长数）: [{np.min(state.phase)/(2*np.pi):.6f}, {np.max(state.phase)/(2*np.pi):.6f}] waves")
    
    # 检查相位是否被折叠
    phase_range = np.max(state.phase) - np.min(state.phase)
    if phase_range < 2 * np.pi + 0.1:
        print(f"  [WARNING] 相位范围 < 2π，可能被折叠！")
    
    # 检查 PROPER 状态
    if state.proper_wfo is not None:
        wfo = state.proper_wfo
        print(f"  PROPER 状态:")
        print(f"    reference_surface: {wfo.reference_surface}")
        print(f"    z: {wfo.z * 1e3:.2f} mm")
        print(f"    z_w0: {wfo.z_w0 * 1e3:.2f} mm")
        print(f"    z_Rayleigh: {wfo.z_Rayleigh * 1e3:.2f} mm")
        
        # 检查 PROPER 残差相位
        residual_phase = proper.prop_get_phase(wfo)
        print(f"    残差相位范围: [{np.min(residual_phase):.6f}, {np.max(residual_phase):.6f}] rad")


# ============================================================================
# 分析 Pilot Beam 和 PROPER 的高斯光束参数是否一致
# ============================================================================

print_section("分析 Pilot Beam 和 PROPER 的高斯光束参数")

# 找到 Surface_3 entrance 的状态
surface_3_entrance = None
for state in result.surface_states:
    if state.surface_index == 3 and state.position == 'entrance':
        surface_3_entrance = state
        break

if surface_3_entrance is None:
    print("未找到 Surface_3 entrance 状态")
else:
    wfo = surface_3_entrance.proper_wfo
    pilot = surface_3_entrance.pilot_beam_params
    
    print("PROPER 高斯光束参数:")
    print(f"  w0 (束腰半径): {wfo.w0 * 1e3:.6f} mm")
    print(f"  z_Rayleigh (瑞利长度): {wfo.z_Rayleigh * 1e3:.2f} mm")
    print(f"  z (当前位置): {wfo.z * 1e3:.2f} mm")
    print(f"  z_w0 (束腰位置): {wfo.z_w0 * 1e3:.2f} mm")
    print(f"  z - z_w0 (相对于束腰的距离): {(wfo.z - wfo.z_w0) * 1e3:.2f} mm")
    
    print("\nPilot Beam 高斯光束参数:")
    print(f"  waist_radius_mm (束腰半径): {pilot.waist_radius_mm:.6f} mm")
    print(f"  rayleigh_length_mm (瑞利长度): {pilot.rayleigh_length_mm:.2f} mm")
    print(f"  waist_position_mm (束腰位置): {pilot.waist_position_mm:.2f} mm")
    print(f"  curvature_radius_mm (曲率半径): {pilot.curvature_radius_mm:.2f} mm")
    
    # 比较参数
    print("\n参数比较:")
    w0_proper = wfo.w0 * 1e3
    w0_pilot = pilot.waist_radius_mm
    print(f"  束腰半径: PROPER={w0_proper:.2f} mm, Pilot={w0_pilot:.2f} mm")
    print(f"  束腰半径差异: {abs(w0_proper - w0_pilot):.6f} mm ({abs(w0_proper - w0_pilot)/w0_pilot*100:.2f}%)")
    
    z_R_proper = wfo.z_Rayleigh * 1e3
    z_R_pilot = pilot.rayleigh_length_mm
    print(f"  瑞利长度: PROPER={z_R_proper:.2f} mm, Pilot={z_R_pilot:.2f} mm")
    print(f"  瑞利长度差异: {abs(z_R_proper - z_R_pilot):.2f} mm ({abs(z_R_proper - z_R_pilot)/z_R_pilot*100:.2f}%)")
    
    # PROPER 的 z - z_w0 应该等于 Pilot Beam 的传播距离
    z_rel_proper = (wfo.z - wfo.z_w0) * 1e3
    # Pilot Beam 的传播距离 = 当前位置 - 束腰位置
    # 但 Pilot Beam 没有直接存储传播距离，需要从曲率半径反推
    print(f"  PROPER 相对传播距离 (z - z_w0): {z_rel_proper:.2f} mm")
    print(f"  Pilot Beam 束腰位置: {pilot.waist_position_mm:.2f} mm")
    
    # 关键问题：参数不一致！
    if abs(w0_proper - w0_pilot) / w0_pilot > 0.01:
        print(f"\n[严重问题] PROPER 和 Pilot Beam 的束腰半径不一致！")
        print(f"  这会导致曲率半径计算错误，进而导致相位解包裹失败。")


# ============================================================================
# 理论分析：Pilot Beam 在入射面的相位应该是多少？
# ============================================================================

print_section("理论分析：Pilot Beam 在入射面的相位")

if surface_3_entrance is not None:
    pilot = surface_3_entrance.pilot_beam_params
    grid_sampling = surface_3_entrance.grid_sampling
    
    # Pilot Beam 相位公式: φ_pilot(r) = k × r² / (2 × R)
    # 其中 R 是曲率半径（使用严格公式）
    
    wavelength_mm = wavelength_um * 1e-3
    k = 2 * np.pi / wavelength_mm  # 波数 (1/mm)
    
    R_pilot = pilot.curvature_radius_mm
    print(f"Pilot Beam 曲率半径 R: {R_pilot:.2f} mm")
    
    # 计算网格边缘的最大 r²
    half_size = grid_sampling.physical_size_mm / 2
    r_max = half_size * np.sqrt(2)  # 对角线
    r_sq_max = r_max**2
    
    print(f"网格半尺寸: {half_size:.2f} mm")
    print(f"最大 r (对角线): {r_max:.2f} mm")
    print(f"最大 r^2: {r_sq_max:.2f} mm^2")
    
    # 理论 Pilot Beam 相位范围
    if np.isinf(R_pilot):
        phi_pilot_max = 0.0
    else:
        phi_pilot_max = k * r_sq_max / (2 * R_pilot)
    
    print(f"\n理论 Pilot Beam 最大相位: {phi_pilot_max:.6f} rad = {phi_pilot_max/(2*np.pi):.6f} waves")
    
    # 实际计算 Pilot Beam 相位网格
    pilot_phase_grid = pilot.compute_phase_grid(
        grid_sampling.grid_size,
        grid_sampling.physical_size_mm,
    )
    
    print(f"\n实际 Pilot Beam 相位网格:")
    print(f"  最小值: {np.min(pilot_phase_grid):.6f} rad")
    print(f"  最大值: {np.max(pilot_phase_grid):.6f} rad")
    print(f"  范围: {np.max(pilot_phase_grid) - np.min(pilot_phase_grid):.6f} rad = {(np.max(pilot_phase_grid) - np.min(pilot_phase_grid))/(2*np.pi):.6f} waves")
    
    # 为什么 Pilot Beam 相位这么小？
    print(f"\n分析：为什么 Pilot Beam 相位这么小？")
    print(f"  因为曲率半径 R = {R_pilot:.2f} mm 非常大")
    print(f"        = {k * r_sq_max / (2 * R_pilot):.6f} rad")


# ============================================================================
# 理论分析：PROPER 提取的相位应该是多少？
# ============================================================================

print_section("理论分析：PROPER 提取的相位")

if surface_3_entrance is not None:
    wfo = surface_3_entrance.proper_wfo
    
    # PROPER 存储的是相对于参考面的残差
    # 对于理想高斯光束，残差应该接近零
    
    print(f"PROPER 参考面类型: {wfo.reference_surface}")
    
    if wfo.reference_surface == "PLANAR":
        print(f"\nPLANAR 参考面：参考相位 = 0")
        print(f"所以 PROPER 存储的相位 = 绝对相位")
        
        # 理论上，理想高斯光束的绝对相位是什么？
        # φ_gaussian(r) = k × r² / (2 × R_gaussian)
        # 其中 R_gaussian 是高斯光束的曲率半径（严格公式）
        
        z = (wfo.z - wfo.z_w0) * 1e3  # mm
        z_R = wfo.z_Rayleigh * 1e3  # mm
        
        if abs(z) < 1e-10:
            R_gaussian = np.inf
        else:
            R_gaussian = z * (1 + (z_R / z)**2)
        
        print(f"\n理论高斯光束参数:")
        print(f"  z (相对于束腰): {z:.2f} mm")
        print(f"  z_R (瑞利长度): {z_R:.2f} mm")
        print(f"  z/z_R: {z/z_R if z_R > 0 else 'inf':.6f}")
        print(f"  R_gaussian (严格公式): {R_gaussian:.2f} mm")
        
        # 理论高斯光束相位范围
        if np.isinf(R_gaussian):
            phi_gaussian_max = 0.0
        else:
            phi_gaussian_max = k * r_sq_max / (2 * R_gaussian)
        
        print(f"\n理论高斯光束最大相位: {phi_gaussian_max:.6f} rad = {phi_gaussian_max/(2*np.pi):.6f} waves")
        
        # 但是 PROPER 提取的相位是折叠的！
        residual_phase = proper.prop_get_phase(wfo)
        print(f"\nPROPER 提取的相位 (prop_get_phase):")
        print(f"  最小值: {np.min(residual_phase):.6f} rad")
        print(f"  最大值: {np.max(residual_phase):.6f} rad")
        print(f"  范围: {np.max(residual_phase) - np.min(residual_phase):.6f} rad")
        
        # 检查是否被折叠
        if np.max(residual_phase) - np.min(residual_phase) > 2 * np.pi - 0.1:
            print(f"\n[确认] PROPER 相位范围接近 2π，说明发生了折叠！")
            print(f"  理论相位范围应该是 {phi_gaussian_max:.6f} rad = {phi_gaussian_max/(2*np.pi):.2f} waves")
            print(f"  但 PROPER 返回的是折叠相位，范围被限制在 [-π, π]")
        
        # 比较 Pilot Beam 和高斯光束的曲率半径
        print(f"\n曲率半径比较:")
        print(f"  Pilot Beam R: {R_pilot:.2f} mm")
        print(f"  高斯光束 R (严格公式): {R_gaussian:.2f} mm")
        print(f"  差异: {abs(R_pilot - R_gaussian):.2f} mm ({abs(R_pilot - R_gaussian)/R_gaussian*100:.4f}%)")
        
    elif wfo.reference_surface == "SPHERI":
        R_ref = (wfo.z - wfo.z_w0) * 1e3  # PROPER 远场近似
        print(f"\nSPHERI 参考面：R_ref = z - z_w0 = {R_ref:.2f} mm")
        print(f"参考相位 = -k × r² / (2 × R_ref)")


# ============================================================================
# 关键问题分析：为什么 PROPER 相位范围是 2π？
# ============================================================================

print_section("关键问题分析：为什么 PROPER 相位范围是 2π？")

if surface_3_entrance is not None:
    wfo = surface_3_entrance.proper_wfo
    
    # 检查 wfarr
    wfarr_centered = proper.prop_shift_center(wfo.wfarr)
    wfarr_amplitude = np.abs(wfarr_centered)
    wfarr_phase = np.angle(wfarr_centered)
    
    print(f"wfarr 振幅:")
    print(f"  最小值: {np.min(wfarr_amplitude):.6f}")
    print(f"  最大值: {np.max(wfarr_amplitude):.6f}")
    
    print(f"\nwfarr 相位 (np.angle, 折叠):")
    print(f"  最小值: {np.min(wfarr_phase):.6f} rad")
    print(f"  最大值: {np.max(wfarr_phase):.6f} rad")
    
    # 检查相位梯度
    grad_y, grad_x = np.gradient(wfarr_phase)
    valid_mask = wfarr_amplitude > 0.01 * np.max(wfarr_amplitude)
    
    print(f"\n相位梯度 (检测 2π 跳变):")
    print(f"  X 方向最大梯度: {np.max(np.abs(grad_x[valid_mask])):.6f} rad/pixel")
    print(f"  Y 方向最大梯度: {np.max(np.abs(grad_y[valid_mask])):.6f} rad/pixel")
    
    if np.max(np.abs(grad_x[valid_mask])) > np.pi or np.max(np.abs(grad_y[valid_mask])) > np.pi:
        print(f"  [确认] 存在 2π 跳变（梯度 > π）")
    else:
        print(f"  [INFO] 未检测到明显的 2π 跳变")
    
    # 关键分析：相位梯度很小，但相位范围是 2π
    # 这说明相位是平滑变化的，从 -π 到 +π
    # 这是 FFT 传播的特性：相位会累积
    
    print(f"\n关键分析：")
    print(f"  相位梯度很小（{np.max(np.abs(grad_x[valid_mask])):.6f} rad/pixel）")
    print(f"  但相位范围是 2π")
    print(f"  这说明相位是平滑变化的，从 -π 到 +π")
    
    # 检查相位的空间分布
    center = grid_size // 2
    phase_center = wfarr_phase[center, center]
    phase_edge = wfarr_phase[0, 0]
    
    print(f"\n相位空间分布:")
    print(f"  中心相位: {phase_center:.6f} rad")
    print(f"  角落相位: {phase_edge:.6f} rad")
    print(f"  差异: {phase_edge - phase_center:.6f} rad")
    
    # 检查沿对角线的相位变化
    diag_indices = np.arange(grid_size)
    diag_phase = wfarr_phase[diag_indices, diag_indices]
    
    print(f"\n沿对角线的相位变化:")
    print(f"  起点 (0,0): {diag_phase[0]:.6f} rad")
    print(f"  中点 ({center},{center}): {diag_phase[center]:.6f} rad")
    print(f"  终点 ({grid_size-1},{grid_size-1}): {diag_phase[-1]:.6f} rad")
    
    # 计算相位的累积变化
    diag_diff = np.diff(diag_phase)
    total_phase_change = np.sum(diag_diff)
    
    print(f"\n相位累积变化（沿对角线）:")
    print(f"  总变化: {total_phase_change:.6f} rad = {total_phase_change/(2*np.pi):.6f} waves")
    
    # 如果没有 2π 跳变，累积变化应该等于终点-起点
    expected_change = diag_phase[-1] - diag_phase[0]
    print(f"  期望变化 (终点-起点): {expected_change:.6f} rad")
    print(f"  差异: {abs(total_phase_change - expected_change):.6f} rad")


# ============================================================================
# 检查 PROPER 初始化问题
# ============================================================================

print_section("检查 PROPER 初始化问题")

print("""
问题：为什么 PROPER 的 w0 = 15 mm，而不是我们设置的 5 mm？

分析 PROPER 的 prop_begin 行为：
- beam_diameter 参数是网格的物理尺寸，不是光束直径
- PROPER 内部的 w0 = beam_diameter / 2 = 30 / 2 = 15 mm
- 这与我们期望的 w0 = 5 mm 不一致！

这是 SourceDefinition.create_initial_wavefront() 的问题：
- 它正确创建了 PilotBeamParams（w0 = 5 mm）
- 但没有同步 PROPER 对象的高斯光束参数
- PROPER 的 w0, z_Rayleigh, z_w0 仍然是默认值
""")

# 验证 PROPER 的 w0 计算
print(f"\n验证 PROPER 的 w0 计算:")
print(f"  physical_size_mm = {physical_size_mm} mm")
print(f"  PROPER w0 = physical_size_mm / 2 = {physical_size_mm / 2} mm")
print(f"  实际 PROPER w0 = {wfo.w0 * 1e3} mm")
print(f"  期望 w0 = {w0_mm} mm")

# 检查 SourceDefinition 的实现
print(f"\n检查 SourceDefinition:")
print(f"  source.w0_mm = {source.w0_mm} mm")
print(f"  source.physical_size_mm = {source.physical_size_mm} mm")


# ============================================================================
# 关键分析：理论相位应该是多少？
# ============================================================================

print_section("关键分析：理论相位应该是多少？")

if surface_3_entrance is not None:
    pilot = surface_3_entrance.pilot_beam_params
    wfo = surface_3_entrance.proper_wfo
    
    print("使用 Pilot Beam 参数（正确的 w0 = 5 mm）计算理论相位：")
    print(f"  w0 = {pilot.waist_radius_mm} mm")
    print(f"  z_R = {pilot.rayleigh_length_mm:.2f} mm")
    print(f"  R = {pilot.curvature_radius_mm:.2f} mm")
    
    # 理论相位范围
    wavelength_mm = wavelength_um * 1e-3
    k = 2 * np.pi / wavelength_mm
    half_size = physical_size_mm / 2
    r_max = half_size * np.sqrt(2)
    r_sq_max = r_max**2
    
    if np.isinf(pilot.curvature_radius_mm):
        phi_pilot_max = 0.0
    else:
        phi_pilot_max = k * r_sq_max / (2 * pilot.curvature_radius_mm)
    
    print(f"\n理论 Pilot Beam 最大相位: {phi_pilot_max:.6f} rad = {phi_pilot_max/(2*np.pi):.6f} waves")
    
    print("\n使用 PROPER 参数（错误的 w0 = 15 mm）计算理论相位：")
    print(f"  w0 = {wfo.w0 * 1e3} mm")
    print(f"  z_R = {wfo.z_Rayleigh * 1e3:.2f} mm")
    
    # PROPER 的曲率半径（严格公式）
    z_proper = (wfo.z - wfo.z_w0) * 1e3
    z_R_proper = wfo.z_Rayleigh * 1e3
    if abs(z_proper) < 1e-10:
        R_proper = np.inf
    else:
        R_proper = z_proper * (1 + (z_R_proper / z_proper)**2)
    
    print(f"  z = {z_proper:.2f} mm")
    print(f"  R (严格公式) = {R_proper:.2f} mm")
    
    if np.isinf(R_proper):
        phi_proper_max = 0.0
    else:
        phi_proper_max = k * r_sq_max / (2 * R_proper)
    
    print(f"\n理论 PROPER 最大相位: {phi_proper_max:.6f} rad = {phi_proper_max/(2*np.pi):.6f} waves")
    
    print("\n关键发现：")
    print(f"  Pilot Beam 理论相位: {phi_pilot_max:.6f} rad")
    print(f"  PROPER 理论相位: {phi_proper_max:.6f} rad")
    print(f"  PROPER 实际相位范围: {np.max(residual_phase) - np.min(residual_phase):.6f} rad")
    
    if phi_proper_max < 0.01:
        print(f"\n[关键] PROPER 理论相位非常小（{phi_proper_max:.6f} rad）")
        print(f"  但实际相位范围是 2pi！")
        print(f"  这说明 PROPER 的相位不是来自高斯光束的球面波前")
        print(f"  而是来自 FFT 传播过程中的数值累积")
    
    # 检查 PROPER 的 wfarr 是否真的是高斯光束
    print("\n检查 PROPER wfarr 的振幅分布：")
    wfarr_centered = proper.prop_shift_center(wfo.wfarr)
    wfarr_amplitude = np.abs(wfarr_centered)
    
    # 计算理论高斯振幅（使用 PROPER 的 w0）
    n = grid_size
    sampling_mm = physical_size_mm / n
    half_size = physical_size_mm / 2
    coords = np.linspace(-half_size, half_size, n)
    X, Y = np.meshgrid(coords, coords)
    R_sq = X**2 + Y**2
    
    # PROPER 的光斑大小
    w_proper = wfo.w0 * 1e3 * np.sqrt(1 + (z_proper / z_R_proper)**2) if z_R_proper > 0 else wfo.w0 * 1e3
    gaussian_amplitude_proper = np.exp(-R_sq / w_proper**2)
    
    # 用户期望的光斑大小
    w_pilot = pilot.spot_size_mm
    gaussian_amplitude_pilot = np.exp(-R_sq / w_pilot**2)
    
    print(f"  PROPER 光斑大小 w: {w_proper:.2f} mm")
    print(f"  Pilot Beam 光斑大小 w: {w_pilot:.2f} mm")
    
    # 比较振幅分布
    center = n // 2
    print(f"\n振幅分布比较（沿 X 轴）：")
    print(f"  位置 (mm) | wfarr | PROPER理论 | Pilot理论")
    for i in [0, n//4, n//2, 3*n//4, n-1]:
        x = coords[i]
        print(f"  {x:8.2f} | {wfarr_amplitude[center, i]:.4f} | {gaussian_amplitude_proper[center, i]:.4f} | {gaussian_amplitude_pilot[center, i]:.4f}")


# ============================================================================
# 结论
# ============================================================================

print_section("结论")

print("""
分析结果：

1. Pilot Beam 和 PROPER 的高斯光束参数严重不一致！
   - PROPER w0 = 15 mm (physical_size / 2)
   - Pilot Beam w0 = 5 mm (用户设置)
   - 差异 200%！

2. 理论上 Pilot Beam 在入射面的相位：
   - 由于曲率半径非常大（~5亿mm），相位应该非常小（~0.005 rad）
   - 这是因为 z << z_R，所以 R 约等于 z_R^2/z 非常大

3. PROPER 提取的相位：
   - 理论上应该更小（~0.00006 rad），因为 PROPER 的 z_R 更大
   - 但 PROPER 返回的相位范围是 2pi！
   - 相位梯度很小（0.000057 rad/pixel），没有 2pi 跳变
   - 这说明相位是平滑变化的，但范围覆盖了整个 [-pi, pi]

4. 根本原因：
   - PROPER 的 FFT 传播会在 wfarr 中累积相位
   - 即使理论相位很小，FFT 传播也会引入额外的相位
   - 这些相位可能来自数值误差或 FFT 的特性

5. 更重要的问题：
   - PROPER 和 Pilot Beam 的高斯光束参数不一致
   - 这会导致相位解包裹失败
   - 需要首先同步这两个参数

6. 解决方案：
   a. 首先同步 PROPER 和 Pilot Beam 的高斯光束参数
   b. 然后使用 Pilot Beam 参考相位进行解包裹
   c. T_unwrapped = T_pilot + angle(exp(1j * (T - T_pilot)))
""")
