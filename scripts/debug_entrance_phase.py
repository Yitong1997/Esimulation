"""
调试入射相位的来源

检查入射相位是否正确包含 Pilot Beam 相位
"""

import sys
from pathlib import Path

# 添加项目路径
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
from hybrid_optical_propagation.material_detection import is_coordinate_break


def main():
    """主函数"""
    
    zmx_file = "optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx"
    
    print("加载光学系统...")
    optical_system = load_optical_system_from_zmx(zmx_file)
    
    # 创建光源定义
    source = SourceDefinition(
        wavelength_um=0.55,
        w0_mm=5.0,
        z0_mm=0.0,
        grid_size=256,
        physical_size_mm=40.0,
    )
    
    # 创建传播器
    propagator = HybridOpticalPropagator(
        optical_system=optical_system,
        source=source,
        wavelength_um=0.55,
        grid_size=256,
        num_rays=150,
    )
    
    # 初始化
    propagator._current_state = propagator._initialize_propagation()
    propagator._surface_states = [propagator._current_state]
    
    # 检查初始状态
    initial_state = propagator._current_state
    print("\n" + "=" * 60)
    print("初始状态（光源处）")
    print("=" * 60)
    
    grid_size = initial_state.grid_sampling.grid_size
    sampling_mm = initial_state.grid_sampling.sampling_mm
    x = np.arange(grid_size) - grid_size // 2
    y = np.arange(grid_size) - grid_size // 2
    xx, yy = np.meshgrid(x * sampling_mm, y * sampling_mm)
    r_sq = xx**2 + yy**2
    
    wavelength_mm = 0.55 * 1e-3
    k = 2 * np.pi / wavelength_mm
    
    # 初始 Pilot Beam 相位
    R_initial = initial_state.pilot_beam_params.curvature_radius_mm
    print(f"初始 Pilot Beam 曲率半径: {R_initial}")
    
    if np.isinf(R_initial):
        pilot_phase_initial = np.zeros_like(r_sq)
    else:
        pilot_phase_initial = k * r_sq / (2 * R_initial)
    
    # 初始仿真相位
    sim_phase_initial = initial_state.phase
    
    # 比较
    phase_diff_initial = sim_phase_initial - pilot_phase_initial
    mask = initial_state.amplitude > 0.01
    
    print(f"初始仿真相位范围: [{np.min(sim_phase_initial):.6f}, {np.max(sim_phase_initial):.6f}] rad")
    print(f"初始 Pilot Beam 相位范围: [{np.min(pilot_phase_initial):.6f}, {np.max(pilot_phase_initial):.6f}] rad")
    print(f"差异 RMS: {np.std(phase_diff_initial[mask]):.6f} rad")
    print(f"差异 RMS (waves): {np.std(phase_diff_initial[mask]) / (2 * np.pi):.6f}")
    
    # 传播到表面 3 入射面
    print("\n传播到表面 3 入射面...")
    for i in range(3):
        surface = optical_system[i]
        if is_coordinate_break(surface):
            continue
        propagator._propagate_to_surface(i)
    
    propagator._propagate_to_surface(3)
    
    # 找到表面 3 入射面状态
    state_entrance = None
    for state in propagator._surface_states:
        if state.surface_index == 3 and state.position == 'entrance':
            state_entrance = state
            break
    
    if state_entrance is None:
        print("未找到表面 3 入射面状态")
        return
    
    print("\n" + "=" * 60)
    print("表面 3 入射面状态")
    print("=" * 60)
    
    # 表面 3 入射面 Pilot Beam 相位
    R_entrance = state_entrance.pilot_beam_params.curvature_radius_mm
    print(f"Pilot Beam 曲率半径: {R_entrance:.2f} mm")
    print(f"Pilot Beam 束腰位置: {state_entrance.pilot_beam_params.waist_position_mm:.4f} mm")
    
    if np.isinf(R_entrance):
        pilot_phase_entrance = np.zeros_like(r_sq)
    else:
        pilot_phase_entrance = k * r_sq / (2 * R_entrance)
    
    # 仿真相位
    sim_phase_entrance = state_entrance.phase
    
    # 比较
    phase_diff_entrance = sim_phase_entrance - pilot_phase_entrance
    mask_entrance = state_entrance.amplitude > 0.01
    
    print(f"\n仿真相位范围: [{np.min(sim_phase_entrance):.6f}, {np.max(sim_phase_entrance):.6f}] rad")
    print(f"Pilot Beam 相位范围: [{np.min(pilot_phase_entrance[mask_entrance]):.6f}, {np.max(pilot_phase_entrance[mask_entrance]):.6f}] rad")
    print(f"差异范围: [{np.min(phase_diff_entrance[mask_entrance]):.6f}, {np.max(phase_diff_entrance[mask_entrance]):.6f}] rad")
    print(f"差异 RMS: {np.std(phase_diff_entrance[mask_entrance]):.6f} rad")
    print(f"差异 RMS (waves): {np.std(phase_diff_entrance[mask_entrance]) / (2 * np.pi):.6f}")
    
    # 检查仿真相位是否等于 Pilot Beam 相位
    print("\n" + "=" * 60)
    print("仿真相位与 Pilot Beam 相位的关系")
    print("=" * 60)
    
    # 在光瞳边缘（r = 20mm）处比较
    r_edge = 20.0  # mm
    r_sq_edge = r_edge**2
    
    if np.isinf(R_entrance):
        pilot_phase_edge = 0.0
    else:
        pilot_phase_edge = k * r_sq_edge / (2 * R_entrance)
    
    print(f"在 r = {r_edge} mm 处:")
    print(f"  Pilot Beam 相位: {pilot_phase_edge:.6f} rad = {pilot_phase_edge / (2 * np.pi):.6f} waves")
    
    # 找到最接近 r = 20mm 的像素
    r_grid = np.sqrt(r_sq)
    idx = np.argmin(np.abs(r_grid - r_edge))
    idx_2d = np.unravel_index(idx, r_grid.shape)
    
    sim_phase_at_edge = sim_phase_entrance[idx_2d]
    print(f"  仿真相位: {sim_phase_at_edge:.6f} rad = {sim_phase_at_edge / (2 * np.pi):.6f} waves")
    print(f"  差异: {sim_phase_at_edge - pilot_phase_edge:.6f} rad")
    
    # 检查仿真相位的形状
    print("\n" + "=" * 60)
    print("仿真相位的形状分析")
    print("=" * 60)
    
    # 沿 X 轴的相位剖面
    center_y = grid_size // 2
    phase_profile_x = sim_phase_entrance[center_y, :]
    pilot_profile_x = pilot_phase_entrance[center_y, :]
    
    print(f"沿 X 轴的相位剖面（y=0）:")
    print(f"  仿真相位: [{np.min(phase_profile_x):.6f}, {np.max(phase_profile_x):.6f}] rad")
    print(f"  Pilot Beam 相位: [{np.min(pilot_profile_x):.6f}, {np.max(pilot_profile_x):.6f}] rad")
    
    # 检查相位是否与 r² 成正比
    # 如果是理想高斯光束，相位应该是 k * r² / (2 * R)
    # 即相位与 r² 成正比
    
    # 在有效区域内拟合 phase = a * r²
    r_sq_flat = r_sq[mask_entrance]
    phase_flat = sim_phase_entrance[mask_entrance]
    
    # 线性拟合
    from numpy.polynomial import polynomial as P
    coeffs = np.polyfit(r_sq_flat, phase_flat, 1)
    a_fit = coeffs[0]
    b_fit = coeffs[1]
    
    print(f"\n相位拟合: phase = {a_fit:.10f} * r² + {b_fit:.6f}")
    print(f"理论值: phase = {k / (2 * R_entrance):.10f} * r²")
    print(f"拟合系数与理论值的比值: {a_fit / (k / (2 * R_entrance)):.6f}")
    
    print("\n完成")


if __name__ == '__main__':
    main()
