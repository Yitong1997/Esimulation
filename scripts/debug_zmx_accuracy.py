"""
调试 ZMX 端到端精度问题

分析误差来源：
1. 初始波前与 Pilot Beam 的一致性
2. 自由空间传播后的误差
3. 反射镜处理后的误差
4. 振幅重建失败的原因

作者：混合光学仿真项目
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def print_section(title: str) -> None:
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================================
# 导入模块
# ============================================================================

print_section("导入模块")

from hybrid_optical_propagation import (
    SourceDefinition,
    HybridOpticalPropagator,
    PilotBeamParams,
    GridSampling,
    PropagationState,
    load_optical_system_from_zmx,
)

print("[OK] 模块导入成功")


# ============================================================================
# 加载光学系统
# ============================================================================

print_section("加载光学系统")

zmx_file_path = "optiland-master/tests/zemax_files/simple_fold_mirror_up.zmx"

if Path(zmx_file_path).exists():
    optical_system = load_optical_system_from_zmx(zmx_file_path)
    print(f"[OK] 加载 ZMX 文件成功")
    print(f"  表面数量: {len(optical_system)}")
    for surface in optical_system:
        print(f"  - 表面 {surface.index}: {surface.surface_type}, "
              f"R={surface.radius:.2f}mm, "
              f"mirror={surface.is_mirror}, "
              f"vertex={surface.vertex_position}")
else:
    print(f"[FAIL] ZMX 文件不存在: {zmx_file_path}")
    sys.exit(1)


# ============================================================================
# 创建光源
# ============================================================================

print_section("创建光源")

wavelength_um = 0.55
w0_mm = 5.0
grid_size = 256
physical_size_mm = 40.0

source = SourceDefinition(
    wavelength_um=wavelength_um,
    w0_mm=w0_mm,
    z0_mm=0.0,
    grid_size=grid_size,
    physical_size_mm=physical_size_mm,
)

# 计算瑞利长度
wavelength_mm = wavelength_um * 1e-3
z_R = np.pi * w0_mm**2 / wavelength_mm

print(f"""
光源参数:
  波长: {wavelength_um} μm
  束腰半径: {w0_mm} mm
  瑞利长度: {z_R:.1f} mm
  网格大小: {grid_size} × {grid_size}
  物理尺寸: {physical_size_mm} mm
""")


# ============================================================================
# 执行传播并收集详细信息
# ============================================================================

print_section("执行传播")

propagator = HybridOpticalPropagator(
    optical_system=optical_system,
    source=source,
    wavelength_um=wavelength_um,
    grid_size=grid_size,
    num_rays=150,
)

result = propagator.propagate()

if not result.success:
    print(f"[FAIL] 传播失败: {result.error_message}")
    sys.exit(1)

print(f"[OK] 传播成功")
print(f"  表面状态数量: {len(result.surface_states)}")


# ============================================================================
# 详细分析每个状态
# ============================================================================

print_section("详细分析每个状态")

def analyze_state_detail(state: PropagationState, name: str):
    """详细分析单个状态"""
    print(f"\n--- {name} ({state.position}) ---")
    
    # 基本信息
    amplitude = state.amplitude
    phase = state.phase
    pilot_params = state.pilot_beam_params
    grid_sampling = GridSampling.from_proper(state.proper_wfo)
    
    print(f"  网格大小: {amplitude.shape}")
    print(f"  物理尺寸: {grid_sampling.physical_size_mm:.2f} mm")
    print(f"  采样间隔: {grid_sampling.sampling_mm:.4f} mm")
    
    # 振幅统计
    max_amp = np.max(amplitude)
    print(f"  振幅最大值: {max_amp:.6f}")
    print(f"  振幅最小值: {np.min(amplitude):.6f}")
    
    # 相位统计
    print(f"  相位最大值: {np.max(phase):.4f} rad")
    print(f"  相位最小值: {np.min(phase):.4f} rad")
    print(f"  相位范围: {np.max(phase) - np.min(phase):.4f} rad")
    
    # Pilot Beam 参数
    print(f"  Pilot Beam 曲率半径: {pilot_params.curvature_radius_mm:.2f} mm")
    print(f"  Pilot Beam 光斑大小: {pilot_params.spot_size_mm:.4f} mm")
    print(f"  Pilot Beam 束腰位置: {pilot_params.waist_position_mm:.2f} mm")
    
    # 计算 Pilot Beam 参考相位
    pilot_phase = pilot_params.compute_phase_grid(
        grid_sampling.grid_size,
        grid_sampling.physical_size_mm,
    )
    
    print(f"  Pilot 相位最大值: {np.max(pilot_phase):.4f} rad")
    print(f"  Pilot 相位最小值: {np.min(pilot_phase):.4f} rad")
    
    # 计算相位误差
    valid_mask = amplitude > 0.01 * max_amp
    if np.sum(valid_mask) > 0:
        phase_diff = np.angle(np.exp(1j * (phase - pilot_phase)))
        phase_rms = np.sqrt(np.mean(phase_diff[valid_mask]**2))
        phase_pv = np.max(phase_diff[valid_mask]) - np.min(phase_diff[valid_mask])
        
        print(f"  相位误差 RMS: {phase_rms / (2*np.pi):.6f} waves")
        print(f"  相位误差 PV: {phase_pv / (2*np.pi):.6f} waves")
        
        # 检查是否有 NaN
        if np.any(np.isnan(amplitude)):
            print(f"  [WARNING] 振幅中存在 NaN!")
        if np.any(np.isnan(phase)):
            print(f"  [WARNING] 相位中存在 NaN!")
    else:
        print(f"  [WARNING] 无有效数据!")
    
    return {
        'amplitude': amplitude,
        'phase': phase,
        'pilot_phase': pilot_phase,
        'valid_mask': valid_mask,
        'grid_sampling': grid_sampling,
    }


# 分析所有状态
analysis_results = []
for state in result.surface_states:
    if state.surface_index < 0:
        name = "Initial"
    else:
        name = f"Surface_{state.surface_index}"
    
    data = analyze_state_detail(state, name)
    analysis_results.append((name, state.position, data))


# ============================================================================
# 绘制详细分析图
# ============================================================================

print_section("绘制详细分析图")

# 找出有问题的状态（相位误差 > 0.1 waves）
problem_states = []
for name, position, data in analysis_results:
    if np.sum(data['valid_mask']) > 0:
        phase_diff = np.angle(np.exp(1j * (data['phase'] - data['pilot_phase'])))
        phase_rms = np.sqrt(np.mean(phase_diff[data['valid_mask']]**2))
        if phase_rms / (2*np.pi) > 0.1:
            problem_states.append((name, position, data, phase_rms))

print(f"\n发现 {len(problem_states)} 个有问题的状态:")
for name, position, data, phase_rms in problem_states:
    print(f"  - {name} ({position}): 相位 RMS = {phase_rms / (2*np.pi):.4f} waves")


# 绘制第一个有问题的状态的详细分析
if problem_states:
    name, position, data, phase_rms = problem_states[0]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    extent = [-data['grid_sampling'].physical_size_mm/2,
              data['grid_sampling'].physical_size_mm/2,
              -data['grid_sampling'].physical_size_mm/2,
              data['grid_sampling'].physical_size_mm/2]
    
    # 振幅
    max_amp = np.max(data['amplitude'])
    amp_norm = data['amplitude'] / max_amp if max_amp > 0 else data['amplitude']
    im1 = axes[0, 0].imshow(amp_norm, extent=extent, cmap='viridis')
    axes[0, 0].set_title('Amplitude (normalized)')
    axes[0, 0].set_xlabel('X (mm)')
    axes[0, 0].set_ylabel('Y (mm)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 仿真相位
    im2 = axes[0, 1].imshow(data['phase'], extent=extent, cmap='twilight')
    axes[0, 1].set_title('Simulation Phase (rad)')
    axes[0, 1].set_xlabel('X (mm)')
    axes[0, 1].set_ylabel('Y (mm)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Pilot Beam 相位
    im3 = axes[0, 2].imshow(data['pilot_phase'], extent=extent, cmap='twilight')
    axes[0, 2].set_title('Pilot Beam Phase (rad)')
    axes[0, 2].set_xlabel('X (mm)')
    axes[0, 2].set_ylabel('Y (mm)')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # 相位差
    phase_diff = np.angle(np.exp(1j * (data['phase'] - data['pilot_phase'])))
    phase_diff_masked = np.where(data['valid_mask'], phase_diff, np.nan)
    vmax = max(0.1, np.nanmax(np.abs(phase_diff_masked)))
    im4 = axes[1, 0].imshow(phase_diff_masked, extent=extent, cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax)
    axes[1, 0].set_title(f'Phase Error (rad)\nRMS={phase_rms/(2*np.pi):.4f} waves')
    axes[1, 0].set_xlabel('X (mm)')
    axes[1, 0].set_ylabel('Y (mm)')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # 相位差直方图
    phase_diff_valid = phase_diff[data['valid_mask']]
    axes[1, 1].hist(phase_diff_valid.flatten(), bins=50, color='steelblue', alpha=0.7)
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_title('Phase Error Distribution')
    axes[1, 1].set_xlabel('Error (rad)')
    axes[1, 1].set_ylabel('Count')
    
    # 相位剖面（沿 Y=0）
    center_idx = data['phase'].shape[0] // 2
    x_coords = np.linspace(-data['grid_sampling'].physical_size_mm/2,
                           data['grid_sampling'].physical_size_mm/2,
                           data['phase'].shape[1])
    
    axes[1, 2].plot(x_coords, data['phase'][center_idx, :], 'b-', label='Simulation')
    axes[1, 2].plot(x_coords, data['pilot_phase'][center_idx, :], 'r--', label='Pilot Beam')
    axes[1, 2].set_title('Phase Profile (Y=0)')
    axes[1, 2].set_xlabel('X (mm)')
    axes[1, 2].set_ylabel('Phase (rad)')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    fig.suptitle(f'Detailed Analysis: {name} ({position})', fontsize=14)
    plt.tight_layout()
    fig.savefig('debug_zmx_accuracy_detail.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n[OK] 保存详细分析图: debug_zmx_accuracy_detail.png")


# ============================================================================
# 分析误差来源
# ============================================================================

print_section("误差来源分析")

print("""
根据测试结果，误差主要出现在以下位置：

1. Surface_3 exit（第一个反射镜 M1 的出射面）
   - 相位 RMS 误差 ~0.3 waves
   - 这是混合元件传播器处理反射镜后的结果

2. Surface_4 exit 和 Surface_5 entrance
   - 出现 NaN 值
   - 表示振幅重建失败

可能的误差来源：

A. 光线追迹误差
   - ElementRaytracer 可能没有正确处理反射镜
   - 光线方向计算可能有问题

B. OPD 计算误差
   - 反射镜的 OPD 计算可能不正确
   - 符号约定可能有问题

C. 振幅重建误差
   - RayToWavefrontReconstructor 的雅可比矩阵计算可能有问题
   - 网格重采样可能引入误差

D. Pilot Beam 更新误差
   - 反射镜后的 Pilot Beam 参数更新可能不正确
   - 曲率半径计算可能有问题

E. 相位解包裹误差
   - 出射面的相位解包裹可能失败
   - 残差相位过大导致 2π 跳变
""")


# ============================================================================
# 检查 Pilot Beam 参数演化
# ============================================================================

print_section("Pilot Beam 参数演化")

print("\n表面索引 | 位置     | 曲率半径 (mm) | 光斑大小 (mm) | 束腰位置 (mm)")
print("-" * 70)

for state in result.surface_states:
    if state.surface_index < 0:
        name = "Initial"
    else:
        name = f"Surface_{state.surface_index}"
    
    pilot = state.pilot_beam_params
    
    if np.isinf(pilot.curvature_radius_mm):
        R_str = "∞"
    else:
        R_str = f"{pilot.curvature_radius_mm:.2f}"
    
    print(f"{name:<12} | {state.position:<8} | {R_str:>13} | "
          f"{pilot.spot_size_mm:>13.4f} | {pilot.waist_position_mm:>13.2f}")


print("\n[完成] 调试分析完成")
