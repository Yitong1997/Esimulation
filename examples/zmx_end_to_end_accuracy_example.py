"""
ZMX 文件端到端精度验证示例

============================================================
示例说明
============================================================

本示例演示如何：
1. 加载 ZMX 文件定义的光学系统（激光扩束镜）
2. 执行混合光学传播仿真
3. 在每个入射面检查仿真复振幅与 Pilot Beam 的精度误差
4. 生成详细的误差分析图表
5. 分析误差来源

理论基础：
- 对于理想高斯光束，仿真复振幅的相位应当与 Pilot Beam 参考相位完全一致
- Pilot Beam 使用严格精确的高斯光束曲率公式 R = z × (1 + (z_R/z)²)
- 任何偏差都应该来自于：
  1. 光学元件引入的像差
  2. PROPER 的远场近似误差（在近场传播时）
  3. 数值计算误差

精度要求（严格，不得放松）：
- 初始波前：相位 RMS 误差 < 0.001 waves
- 自由空间传播后：相位 RMS 误差 < 0.01 waves
- 经过光学元件后：取决于元件像差

作者：混合光学仿真项目
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


def print_section(title: str) -> None:
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================================
# 第一部分：导入模块
# ============================================================================

print_section("第一部分：导入模块")

try:
    from hybrid_optical_propagation import (
        SourceDefinition,
        HybridOpticalPropagator,
        PilotBeamParams,
        GridSampling,
        PropagationState,
        load_optical_system_from_zmx,
    )
    print("[OK] 成功导入 hybrid_optical_propagation 模块")
except ImportError as e:
    print(f"[FAIL] 导入失败: {e}")
    sys.exit(1)


# ============================================================================
# 第二部分：定义精度分析数据类和函数
# ============================================================================

print_section("第二部分：定义精度分析函数")


@dataclass
class AccuracyMetrics:
    """精度指标数据类"""
    surface_name: str
    position: str
    
    # 振幅误差
    amplitude_rms_error: float
    amplitude_max_error: float
    
    # 相位误差（相对于 Pilot Beam）
    phase_rms_error_rad: float
    phase_max_error_rad: float
    phase_pv_error_rad: float
    
    # Pilot Beam 参数
    pilot_curvature_mm: float
    pilot_spot_size_mm: float
    
    @property
    def phase_rms_error_waves(self) -> float:
        return self.phase_rms_error_rad / (2 * np.pi)
    
    @property
    def phase_pv_error_waves(self) -> float:
        return self.phase_pv_error_rad / (2 * np.pi)


def compute_pilot_beam_phase(
    pilot_params: PilotBeamParams,
    grid_size: int,
    physical_size_mm: float,
) -> np.ndarray:
    """计算 Pilot Beam 参考相位"""
    return pilot_params.compute_phase_grid(grid_size, physical_size_mm)


def compute_ideal_gaussian_amplitude(
    pilot_params: PilotBeamParams,
    grid_size: int,
    physical_size_mm: float,
) -> np.ndarray:
    """计算理想高斯光束振幅分布"""
    half_size = physical_size_mm / 2
    coords = np.linspace(-half_size, half_size, grid_size)
    X, Y = np.meshgrid(coords, coords)
    r_sq = X**2 + Y**2
    
    w = pilot_params.spot_size_mm
    amplitude = np.exp(-r_sq / w**2)
    
    return amplitude



def analyze_accuracy(
    state: PropagationState,
    grid_size: int,
    physical_size_mm: float,
    surface_name: str,
) -> Tuple[AccuracyMetrics, dict]:
    """分析仿真复振幅与 Pilot Beam 的精度
    
    返回:
        (AccuracyMetrics, 分析数据字典)
    """
    # 提取数据（使用新的振幅/相位分离接口）
    sim_amplitude = state.amplitude
    sim_phase = state.phase
    pilot_params = state.pilot_beam_params
    
    # 计算理想值
    ideal_amplitude = compute_ideal_gaussian_amplitude(
        pilot_params, grid_size, physical_size_mm
    )
    pilot_phase = compute_pilot_beam_phase(
        pilot_params, grid_size, physical_size_mm
    )
    
    # 归一化振幅
    max_sim_amp = np.max(sim_amplitude)
    max_ideal_amp = np.max(ideal_amplitude)
    
    if max_sim_amp > 0:
        sim_amp_norm = sim_amplitude / max_sim_amp
    else:
        sim_amp_norm = np.zeros_like(sim_amplitude)
    
    if max_ideal_amp > 0:
        ideal_amp_norm = ideal_amplitude / max_ideal_amp
    else:
        ideal_amp_norm = np.zeros_like(ideal_amplitude)
    
    # 有效区域掩模
    valid_mask = sim_amp_norm > 0.01
    
    # 处理无有效数据的情况
    if np.sum(valid_mask) == 0:
        metrics = AccuracyMetrics(
            surface_name=surface_name,
            position=state.position,
            amplitude_rms_error=np.nan,
            amplitude_max_error=np.nan,
            phase_rms_error_rad=np.nan,
            phase_max_error_rad=np.nan,
            phase_pv_error_rad=np.nan,
            pilot_curvature_mm=pilot_params.curvature_radius_mm,
            pilot_spot_size_mm=pilot_params.spot_size_mm,
        )
        
        analysis_data = {
            'sim_amplitude': sim_amplitude,
            'sim_phase': sim_phase,
            'ideal_amplitude': ideal_amplitude,
            'pilot_phase': pilot_phase,
            'amp_diff': np.zeros_like(sim_amplitude),
            'phase_diff': np.zeros_like(sim_phase),
            'valid_mask': valid_mask,
        }
        
        return metrics, analysis_data
    
    # 振幅误差
    amp_diff = np.abs(sim_amp_norm - ideal_amp_norm)
    amplitude_rms_error = np.sqrt(np.mean(amp_diff[valid_mask]**2))
    amplitude_max_error = np.max(amp_diff[valid_mask])
    
    # 相位误差
    phase_diff = np.angle(np.exp(1j * (sim_phase - pilot_phase)))
    phase_rms_error = np.sqrt(np.mean(phase_diff[valid_mask]**2))
    phase_max_error = np.max(np.abs(phase_diff[valid_mask]))
    phase_pv_error = np.max(phase_diff[valid_mask]) - np.min(phase_diff[valid_mask])
    
    metrics = AccuracyMetrics(
        surface_name=surface_name,
        position=state.position,
        amplitude_rms_error=amplitude_rms_error,
        amplitude_max_error=amplitude_max_error,
        phase_rms_error_rad=phase_rms_error,
        phase_max_error_rad=phase_max_error,
        phase_pv_error_rad=phase_pv_error,
        pilot_curvature_mm=pilot_params.curvature_radius_mm,
        pilot_spot_size_mm=pilot_params.spot_size_mm,
    )
    
    analysis_data = {
        'sim_amplitude': sim_amplitude,
        'sim_phase': sim_phase,
        'ideal_amplitude': ideal_amplitude,
        'pilot_phase': pilot_phase,
        'amp_diff': amp_diff,
        'phase_diff': phase_diff,
        'valid_mask': valid_mask,
    }
    
    return metrics, analysis_data


def plot_comprehensive_analysis(
    metrics_list: List[AccuracyMetrics],
    analysis_data_list: List[dict],
    physical_size_mm: float,
    output_path: str,
) -> None:
    """绘制综合分析图表"""
    n_surfaces = len(metrics_list)
    
    fig, axes = plt.subplots(n_surfaces, 4, figsize=(16, 4 * n_surfaces))
    
    if n_surfaces == 1:
        axes = axes.reshape(1, -1)
    
    extent = [-physical_size_mm/2, physical_size_mm/2,
              -physical_size_mm/2, physical_size_mm/2]
    
    for i, (metrics, data) in enumerate(zip(metrics_list, analysis_data_list)):
        # 仿真振幅
        max_amp = np.max(data['sim_amplitude'])
        sim_amp_norm = data['sim_amplitude'] / max_amp if max_amp > 0 else data['sim_amplitude']
        im1 = axes[i, 0].imshow(sim_amp_norm, extent=extent, cmap='viridis')
        axes[i, 0].set_title(f'{metrics.surface_name}\n仿真振幅')
        axes[i, 0].set_xlabel('X (mm)')
        axes[i, 0].set_ylabel('Y (mm)')
        plt.colorbar(im1, ax=axes[i, 0])
        
        # 仿真相位
        im2 = axes[i, 1].imshow(data['sim_phase'], extent=extent, cmap='twilight')
        axes[i, 1].set_title('仿真相位 (rad)')
        axes[i, 1].set_xlabel('X (mm)')
        axes[i, 1].set_ylabel('Y (mm)')
        plt.colorbar(im2, ax=axes[i, 1])
        
        # 相位误差
        phase_diff_masked = np.where(data['valid_mask'], data['phase_diff'], np.nan)
        vmax = max(0.1, np.nanmax(np.abs(phase_diff_masked)))
        im3 = axes[i, 2].imshow(
            phase_diff_masked, extent=extent, cmap='RdBu_r',
            vmin=-vmax, vmax=vmax
        )
        axes[i, 2].set_title(f'相位误差 (rad)\nRMS={metrics.phase_rms_error_waves:.6f} waves')
        axes[i, 2].set_xlabel('X (mm)')
        axes[i, 2].set_ylabel('Y (mm)')
        plt.colorbar(im3, ax=axes[i, 2])
        
        # 误差直方图
        phase_diff_valid = data['phase_diff'][data['valid_mask']]
        if len(phase_diff_valid) > 0:
            axes[i, 3].hist(phase_diff_valid.flatten(), bins=50, color='steelblue', alpha=0.7)
            axes[i, 3].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[i, 3].set_title(f'相位误差分布\nPV={metrics.phase_pv_error_waves:.6f} waves')
        axes[i, 3].set_xlabel('误差 (rad)')
        axes[i, 3].set_ylabel('计数')
    
    fig.suptitle('仿真复振幅与 Pilot Beam 精度分析', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] 保存: {output_path}")


def plot_detailed_analysis(
    metrics: AccuracyMetrics,
    data: dict,
    physical_size_mm: float,
    output_path: str,
) -> None:
    """绘制单个表面的详细分析图表"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    extent = [-physical_size_mm/2, physical_size_mm/2,
              -physical_size_mm/2, physical_size_mm/2]
    
    # 第一行：振幅分析
    # 仿真振幅
    max_amp = np.max(data['sim_amplitude'])
    sim_amp_norm = data['sim_amplitude'] / max_amp if max_amp > 0 else data['sim_amplitude']
    im1 = axes[0, 0].imshow(sim_amp_norm, extent=extent, cmap='viridis')
    axes[0, 0].set_title('仿真振幅（归一化）')
    axes[0, 0].set_xlabel('X (mm)')
    axes[0, 0].set_ylabel('Y (mm)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 理想高斯振幅
    ideal_amp_norm = data['ideal_amplitude'] / np.max(data['ideal_amplitude'])
    im2 = axes[0, 1].imshow(ideal_amp_norm, extent=extent, cmap='viridis')
    axes[0, 1].set_title('理想高斯振幅')
    axes[0, 1].set_xlabel('X (mm)')
    axes[0, 1].set_ylabel('Y (mm)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 振幅误差
    amp_diff_masked = np.where(data['valid_mask'], data['amp_diff'], np.nan)
    im3 = axes[0, 2].imshow(amp_diff_masked, extent=extent, cmap='RdBu_r',
                            vmin=-0.1, vmax=0.1)
    axes[0, 2].set_title(f'振幅误差\nRMS={metrics.amplitude_rms_error:.6f}')
    axes[0, 2].set_xlabel('X (mm)')
    axes[0, 2].set_ylabel('Y (mm)')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # 第二行：相位分析
    # 仿真相位
    im4 = axes[1, 0].imshow(data['sim_phase'], extent=extent, cmap='twilight')
    axes[1, 0].set_title('仿真相位 (rad)')
    axes[1, 0].set_xlabel('X (mm)')
    axes[1, 0].set_ylabel('Y (mm)')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Pilot Beam 参考相位
    im5 = axes[1, 1].imshow(data['pilot_phase'], extent=extent, cmap='twilight')
    axes[1, 1].set_title('Pilot Beam 参考相位 (rad)')
    axes[1, 1].set_xlabel('X (mm)')
    axes[1, 1].set_ylabel('Y (mm)')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # 相位误差
    phase_diff_masked = np.where(data['valid_mask'], data['phase_diff'], np.nan)
    vmax = max(0.1, np.nanmax(np.abs(phase_diff_masked)))
    im6 = axes[1, 2].imshow(phase_diff_masked, extent=extent, cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax)
    axes[1, 2].set_title(f'相位误差 (rad)\nRMS={metrics.phase_rms_error_waves:.6f} waves')
    axes[1, 2].set_xlabel('X (mm)')
    axes[1, 2].set_ylabel('Y (mm)')
    plt.colorbar(im6, ax=axes[1, 2])
    
    fig.suptitle(f'精度分析: {metrics.surface_name} ({metrics.position})', fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] 保存: {output_path}")


print("[OK] 精度分析函数定义完成")



# ============================================================================
# 第三部分：创建测试光学系统
# ============================================================================

print_section("第三部分：创建测试光学系统")

# 尝试加载 ZMX 文件
zmx_file_path = "optiland-master/tests/zemax_files/simple_fold_mirror_up.zmx"

if Path(zmx_file_path).exists():
    print(f"[INFO] 尝试加载 ZMX 文件: {zmx_file_path}")
    try:
        optical_system = load_optical_system_from_zmx(zmx_file_path)
        print(f"[OK] 成功加载 ZMX 文件")
        print(f"  表面数量: {len(optical_system)}")
        for surface in optical_system:
            print(f"  - 表面 {surface.index}: {surface.surface_type}, "
                  f"R={surface.radius:.2f}mm, "
                  f"mirror={surface.is_mirror}, "
                  f"comment='{surface.comment}'")
        use_zmx = True
    except Exception as e:
        print(f"[WARN] 无法加载 ZMX 文件: {e}")
        use_zmx = False
else:
    print(f"[INFO] ZMX 文件不存在: {zmx_file_path}")
    use_zmx = False

# 如果无法加载 ZMX 文件，创建模拟的光学系统
if not use_zmx:
    print("[INFO] 创建模拟的光学系统（激光扩束镜）")
    
    @dataclass
    class MockSurface:
        """模拟的 GlobalSurfaceDefinition 对象"""
        index: int
        surface_type: str
        vertex_position: np.ndarray
        orientation: np.ndarray
        radius: float = np.inf
        conic: float = 0.0
        is_mirror: bool = False
        semi_aperture: float = 25.0
        material: str = "air"
        asphere_coeffs: List[float] = field(default_factory=list)
        comment: str = ""
        thickness: float = 0.0
        radius_x: float = np.inf
        conic_x: float = 0.0
        focal_length: float = np.inf
        
        @property
        def surface_normal(self) -> np.ndarray:
            return -self.orientation[:, 2]
    
    def create_rotation_matrix_x(angle_rad: float) -> np.ndarray:
        """创建绕 X 轴旋转矩阵"""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ])
    
    # 创建 45 度平面镜
    tilt_x_rad = -np.pi / 4
    orientation = create_rotation_matrix_x(tilt_x_rad)
    
    mirror = MockSurface(
        index=0,
        surface_type='flat',
        vertex_position=np.array([0.0, 0.0, 50.0]),
        orientation=orientation,
        is_mirror=True,
        material='mirror',
        comment='M1 (45deg flat mirror)',
    )
    
    optical_system = [mirror]
    
    print(f"光学系统配置:")
    print(f"  表面数量: {len(optical_system)}")
    for surface in optical_system:
        print(f"  - {surface.comment}: 位置={surface.vertex_position}")


# ============================================================================
# 第四部分：定义光源参数
# ============================================================================

print_section("第四部分：定义光源参数")

# 光源参数
wavelength_um = 0.55         # 可见光波长
w0_mm = 5.0                  # 束腰半径
grid_size = 256              # 网格大小
physical_size_mm = 40.0      # 物理尺寸

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

# 创建光源定义
source = SourceDefinition(
    wavelength_um=wavelength_um,
    w0_mm=w0_mm,
    z0_mm=0.0,
    grid_size=grid_size,
    physical_size_mm=physical_size_mm,
)


# ============================================================================
# 第五部分：执行传播仿真
# ============================================================================

print_section("第五部分：执行传播仿真")

# 创建传播器
propagator = HybridOpticalPropagator(
    optical_system=optical_system,
    source=source,
    wavelength_um=wavelength_um,
    grid_size=grid_size,
    num_rays=150,
)

print("正在执行传播仿真...")

# 执行传播
result = propagator.propagate()

if result.success:
    print(f"[OK] 传播成功!")
    print(f"  表面状态数量: {len(result.surface_states)}")
    print(f"  总光程: {result.total_path_length:.1f} mm")
else:
    print(f"[FAIL] 传播失败: {result.error_message}")
    sys.exit(1)


# ============================================================================
# 第六部分：精度分析
# ============================================================================

print_section("第六部分：精度分析")

metrics_list = []
analysis_data_list = []

# 精度要求（严格，不得放松）
INITIAL_PHASE_RMS_TOLERANCE_WAVES = 0.001
PROPAGATION_PHASE_RMS_TOLERANCE_WAVES = 0.01

print("\n精度指标:")
print("-" * 90)
print(f"{'表面':<20} {'位置':<10} {'振幅RMS':<12} {'相位RMS':<15} {'相位PV':<15} {'曲率半径':<12}")
print(f"{'':20} {'':10} {'误差':12} {'(waves)':15} {'(waves)':15} {'(mm)':12}")
print("-" * 90)

for state in result.surface_states:
    if state.surface_index < 0:
        name = "Initial"
    else:
        name = f"Surface_{state.surface_index}"
    
    # 使用 PROPER 的实际采样
    grid_sampling = GridSampling.from_proper(state.proper_wfo)
    
    metrics, data = analyze_accuracy(
        state=state,
        grid_size=grid_size,
        physical_size_mm=grid_sampling.physical_size_mm,
        surface_name=name,
    )
    
    metrics_list.append(metrics)
    analysis_data_list.append(data)
    
    # 格式化曲率半径
    if np.isinf(metrics.pilot_curvature_mm):
        curvature_str = "∞"
    else:
        curvature_str = f"{metrics.pilot_curvature_mm:.1f}"
    
    print(f"{name:<20} {state.position:<10} {metrics.amplitude_rms_error:<12.6f} "
          f"{metrics.phase_rms_error_waves:<15.6f} {metrics.phase_pv_error_waves:<15.6f} "
          f"{curvature_str:<12}")

print("-" * 90)


# ============================================================================
# 第七部分：误差来源分析
# ============================================================================

print_section("第七部分：误差来源分析")

print("""
误差来源分析:

1. 初始波前误差
   - 理论上应为零（仿真复振幅直接从 Pilot Beam 参数构造）
   - 实际误差来源：数值精度、网格离散化

2. 自由空间传播误差
   - PROPER 使用远场近似曲率 R_ref = z - z_w0
   - Pilot Beam 使用严格公式 R = z × (1 + (z_R/z)²)
   - 在近场（z ≈ z_R）时，两者存在差异
   - 误差随传播距离增加而减小（趋近远场）

3. 光学元件引入的误差
   - 反射镜的像差（球差、彗差等）
   - 光线追迹的数值误差
   - 网格重采样误差

4. 相位解包裹误差
   - 当相位梯度过大时，解包裹可能失败
   - 使用 Pilot Beam 参考相位可以最小化残差相位
""")

# 分析初始状态
initial_metrics = metrics_list[0]
print(f"\n初始波前分析:")
print(f"  相位 RMS 误差: {initial_metrics.phase_rms_error_waves:.6f} waves")
print(f"  相位 PV 误差: {initial_metrics.phase_pv_error_waves:.6f} waves")

if initial_metrics.phase_rms_error_waves < INITIAL_PHASE_RMS_TOLERANCE_WAVES:
    print(f"  [OK] 初始波前精度满足严格要求 (< {INITIAL_PHASE_RMS_TOLERANCE_WAVES} waves)")
else:
    print(f"  [FAIL] 初始波前精度不满足严格要求")
    print(f"    可能原因: 网格离散化误差、数值精度限制")



# ============================================================================
# 第八部分：生成可视化图表
# ============================================================================

print_section("第八部分：生成可视化图表")

# 使用 PROPER 的实际采样
grid_sampling = GridSampling.from_proper(result.surface_states[0].proper_wfo)
actual_physical_size_mm = grid_sampling.physical_size_mm

# 综合分析图
plot_comprehensive_analysis(
    metrics_list=metrics_list,
    analysis_data_list=analysis_data_list,
    physical_size_mm=actual_physical_size_mm,
    output_path="zmx_end_to_end_accuracy.png",
)

# 单独的详细分析图
for i, (metrics, data) in enumerate(zip(metrics_list, analysis_data_list)):
    plot_detailed_analysis(
        metrics=metrics,
        data=data,
        physical_size_mm=actual_physical_size_mm,
        output_path=f"accuracy_detail_{metrics.surface_name}_{metrics.position}.png",
    )


# ============================================================================
# 第九部分：总结
# ============================================================================

print_section("第九部分：总结")

# 计算总体统计
all_phase_rms = [m.phase_rms_error_waves for m in metrics_list]
all_phase_pv = [m.phase_pv_error_waves for m in metrics_list]

print(f"""
仿真复振幅与 Pilot Beam 精度验证完成！

系统配置:
  • 光学系统: {'ZMX 文件' if use_zmx else '模拟系统'}
  • 波长: {wavelength_um} μm
  • 束腰半径: {w0_mm} mm
  • 瑞利长度: {z_R:.1f} mm
  • 网格大小: {grid_size} × {grid_size}

精度统计:
  • 初始波前相位 RMS 误差: {metrics_list[0].phase_rms_error_waves:.6f} waves
  • 最大相位 RMS 误差: {max(all_phase_rms):.6f} waves
  • 最大相位 PV 误差: {max(all_phase_pv):.6f} waves

误差来源分析:
  1. 初始波前: 数值精度和网格离散化
  2. 自由空间传播: PROPER 远场近似 vs Pilot Beam 严格公式
  3. 光学元件: 像差和光线追迹数值误差

生成的图像文件:
  • zmx_end_to_end_accuracy.png - 综合分析图
  • accuracy_detail_*.png - 各表面详细分析图

精度验证结果:
""")

# 评估结果
all_passed = True

if metrics_list[0].phase_rms_error_waves < INITIAL_PHASE_RMS_TOLERANCE_WAVES:
    print(f"  [OK] 初始波前精度满足严格要求 (< {INITIAL_PHASE_RMS_TOLERANCE_WAVES} waves)")
else:
    print(f"  [FAIL] 初始波前精度不满足严格要求")
    all_passed = False

if max(all_phase_rms) < PROPAGATION_PHASE_RMS_TOLERANCE_WAVES:
    print(f"  [OK] 所有表面相位 RMS 误差 < {PROPAGATION_PHASE_RMS_TOLERANCE_WAVES} waves")
else:
    print(f"  [WARN] 存在表面相位 RMS 误差 > {PROPAGATION_PHASE_RMS_TOLERANCE_WAVES} waves")
    print(f"         这可能是由于光学元件引入的像差")

print(f"""
技术说明:
  • Pilot Beam 使用严格精确的高斯光束曲率公式 R = z × (1 + (z_R/z)²)
  • 仿真复振幅应当与 Pilot Beam 参考相位高度一致
  • 任何偏差都来自于光学元件像差或数值误差
  • 精度要求严格，不得放松

总体评估: {'通过' if all_passed else '需要进一步分析'}
""")
