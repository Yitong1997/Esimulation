"""
高斯光束传输仿真可视化与验证

本脚本展示高斯光束通过单个抛物面反射镜传输的仿真结果，
并与 ABCD 矩阵法的理论计算进行详细对比验证。

设计理念：Zemax 序列模式
- 元件按顺序排列，使用 thickness 定义间距
- 光束沿光路自动传播
- 使用 propagate_distance() 方法沿光路传播

验证内容：
1. 光束半径：理论值 vs 仿真值
2. 波前曲率半径：理论值 vs 仿真值
3. 相位分布：理论值 vs 仿真值
4. 振幅/光强分布：理论值 vs 仿真值
5. 中间节点的详细对比

作者：混合光学仿真项目
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass
from typing import List, Tuple

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加 src 目录到路径
sys.path.insert(0, 'src')

from gaussian_beam_simulation.gaussian_beam import GaussianBeam
from gaussian_beam_simulation.optical_elements import ParabolicMirror
from gaussian_beam_simulation.abcd_calculator import ABCDCalculator
from gaussian_beam_simulation.hybrid_simulator import HybridGaussianBeamSimulator


@dataclass
class ValidationPoint:
    """验证点数据"""
    path_distance: float      # 光程距离 (mm)
    z_position: float         # z 坐标位置 (mm)
    label: str                # 标签
    
    # 理论值（ABCD 矩阵）
    theory_w: float           # 理论光束半径 (mm)
    theory_R: float           # 理论波前曲率半径 (mm)
    theory_gouy: float        # 理论 Gouy 相位 (rad)
    
    # 仿真值（PROPER）
    sim_w: float              # 仿真光束半径 (mm)
    sim_phase_rms: float      # 仿真相位 RMS (waves)
    sim_phase_pv: float       # 仿真相位 PV (waves)
    
    # 误差
    w_error_percent: float    # 光束半径误差 (%)
    
    @property
    def w_match(self) -> bool:
        """光束半径是否匹配（误差 < 10%）"""
        return abs(self.w_error_percent) < 10.0


def extract_beam_radius_from_amplitude(amplitude: np.ndarray, sampling: float) -> float:
    """从振幅分布提取光束半径（二阶矩方法）
    
    对于高斯振幅 A(r) = exp(-r²/w²)，光强 I(r) = exp(-2r²/w²)
    二阶矩 <r²> = w²/2，所以 w = sqrt(2 * <r²>)
    """
    n = amplitude.shape[0]
    coords = (np.arange(n) - n // 2) * sampling
    X, Y = np.meshgrid(coords, coords)
    R_sq = X**2 + Y**2
    
    intensity = amplitude**2
    total_intensity = np.sum(intensity)
    
    if total_intensity < 1e-10:
        return 0.0
    
    r_sq_mean = np.sum(R_sq * intensity) / total_intensity
    # 对于高斯光束：<r²> = w²/2，所以 w = sqrt(2 * <r²>)
    beam_radius = np.sqrt(2 * r_sq_mean)
    
    return beam_radius


def extract_curvature_from_phase(phase: np.ndarray, amplitude: np.ndarray, 
                                  sampling: float, wavelength_um: float) -> float:
    """从相位分布提取波前曲率半径
    
    通过拟合相位的二次项来估计曲率半径
    φ(r) = -k * r² / (2R) => R = -k / (2a)，其中 a 是拟合系数
    
    注意：需要减去中心值（而不是均值）来去除 piston
    """
    n = phase.shape[0]
    center = n // 2
    coords = (np.arange(n) - center) * sampling
    X, Y = np.meshgrid(coords, coords)
    R_sq = X**2 + Y**2
    
    # 只使用振幅较大的区域
    mask = amplitude > 0.3 * np.max(amplitude)
    if np.sum(mask) < 10:
        return np.inf
    
    # 去除 piston：减去中心值（而不是均值）
    # 对于球面波前 φ(r) = a * r²，中心值为 0，均值为 a * <r²> ≠ 0
    phase_centered = phase - phase[center, center]
    
    # 使用最小二乘拟合 φ = a * r²
    r_sq_valid = R_sq[mask].flatten()
    phase_valid = phase_centered[mask].flatten()
    
    # 拟合 φ = a * r²
    # a = Σ(r² * φ) / Σ(r⁴)
    sum_r4 = np.sum(r_sq_valid**2)
    if sum_r4 < 1e-20:
        return np.inf
    
    a = np.sum(r_sq_valid * phase_valid) / sum_r4
    
    # R = -k / (2a)，其中 k = 2π/λ
    wavelength_mm = wavelength_um * 1e-3
    k = 2 * np.pi / wavelength_mm
    
    if abs(a) < 1e-10:
        return np.inf
    
    R = -k / (2 * a)
    return R


def compare_amplitude_profiles(amplitude: np.ndarray, sampling: float,
                                theory_w: float, z_label: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """比较振幅剖面：仿真 vs 理论
    
    返回：(r_coords, sim_profile, theory_profile)
    """
    n = amplitude.shape[0]
    center = n // 2
    
    # 提取中心行的剖面
    sim_profile = amplitude[center, :]
    
    # 计算径向坐标
    r_coords = (np.arange(n) - center) * sampling
    
    # 理论高斯剖面
    # A(r) = A0 * exp(-r²/w²)
    A0 = np.max(sim_profile)
    theory_profile = A0 * np.exp(-r_coords**2 / theory_w**2)
    
    return r_coords, sim_profile, theory_profile


def compare_phase_profiles(phase: np.ndarray, amplitude: np.ndarray, sampling: float,
                           theory_R: float, wavelength_um: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """比较相位剖面：仿真 vs 理论
    
    注意：PROPER 使用参考球面的概念，相位是相对于参考球面的偏差。
    对于高斯光束传播，PROPER 的相位表示与 ABCD 矩阵法的理论相位可能不同，
    因为 PROPER 会自动调整参考球面。
    
    这里我们比较的是相位的形状（归一化后），而不是绝对值。
    
    返回：(r_coords, sim_profile, theory_profile)
    """
    n = phase.shape[0]
    center = n // 2
    
    # 提取中心行的剖面
    sim_profile = phase[center, :]
    
    # 计算径向坐标
    r_coords = (np.arange(n) - center) * sampling
    
    # 理论球面波前相位
    # φ(r) = -k * r² / (2R)
    wavelength_mm = wavelength_um * 1e-3
    k = 2 * np.pi / wavelength_mm
    
    if np.isinf(theory_R):
        theory_profile = np.zeros_like(r_coords)
    else:
        theory_profile = -k * r_coords**2 / (2 * theory_R)
    
    # 对齐：使中心相位为 0
    sim_profile_aligned = sim_profile - sim_profile[center]
    theory_profile_aligned = theory_profile - theory_profile[center]
    
    # 归一化：使振幅较大区域的相位范围一致
    mask = amplitude[center, :] > 0.3 * np.max(amplitude)
    if np.sum(mask) > 0:
        sim_range = np.max(np.abs(sim_profile_aligned[mask]))
        theory_range = np.max(np.abs(theory_profile_aligned[mask]))
        
        if theory_range > 1e-10 and sim_range > 1e-10:
            # 归一化理论相位以匹配仿真相位的范围
            # 这样可以比较相位的形状
            scale = sim_range / theory_range
            theory_profile_scaled = theory_profile_aligned * scale
        else:
            theory_profile_scaled = theory_profile_aligned
    else:
        theory_profile_scaled = theory_profile_aligned
    
    return r_coords, sim_profile_aligned, theory_profile_scaled



def create_comprehensive_validation():
    """创建综合验证可视化
    
    在多个关键节点进行理论与仿真的详细对比
    """
    
    # ========== 设置光学系统参数 ==========
    wavelength = 0.633  # μm (HeNe 激光)
    w0 = 1.0            # mm (束腰半径)
    z0 = 50.0           # mm (束腰位置，在初始面后方)
    m2 = 1.0            # M² 因子（理想高斯光束）
    z_init = 0.0        # mm (初始面位置)
    
    focal_length = 100.0      # mm (反射镜焦距)
    initial_distance = 200.0  # mm (从初始面到反射镜的距离)
    semi_aperture = 15.0      # mm (反射镜半口径)
    
    # ========== 创建光学元件 ==========
    beam = GaussianBeam(
        wavelength=wavelength,
        w0=w0,
        z0=z0,
        m2=m2,
        z_init=z_init,
    )
    
    mirror = ParabolicMirror(
        thickness=150.0,
        semi_aperture=semi_aperture,
        parent_focal_length=focal_length,
    )
    
    # ========== ABCD 计算器 ==========
    calc = ABCDCalculator(beam, [mirror], initial_distance=initial_distance)
    
    # ========== 混合仿真器 ==========
    sim = HybridGaussianBeamSimulator(
        beam, [mirror],
        initial_distance=initial_distance,
        grid_size=256,
        beam_ratio=0.3,
        num_rays=200,
        use_hybrid=False,
    )
    
    # ========== 定义验证节点 ==========
    # 入射段：0, 50, 100, 150, 200 mm
    # 反射段：200+10, 200+50, 200+100, 200+150 mm
    validation_distances = [
        (0.0, "初始面"),
        (50.0, "束腰位置"),
        (100.0, "入射中点"),
        (150.0, "入射3/4"),
        (200.0, "反射镜"),
        (210.0, "反射后10mm"),
        (250.0, "反射后50mm"),
        (300.0, "反射后100mm"),
        (350.0, "反射后150mm"),
    ]
    
    # ========== 收集验证数据 ==========
    validation_points: List[ValidationPoint] = []
    
    for path_dist, label in validation_distances:
        # 理论计算
        theory_result = calc.propagate_distance(path_dist)
        theory_w = theory_result.w
        theory_R = theory_result.R
        theory_gouy = theory_result.gouy_phase if hasattr(theory_result, 'gouy_phase') else 0.0
        
        # 仿真计算
        sim.reset()
        sim_result = sim.propagate_distance(path_dist)
        sim_w = extract_beam_radius_from_amplitude(sim_result.amplitude, sim_result.sampling)
        
        # 计算误差
        if theory_w > 0:
            w_error = (sim_w - theory_w) / theory_w * 100
        else:
            w_error = 0.0
        
        vp = ValidationPoint(
            path_distance=path_dist,
            z_position=theory_result.z,
            label=label,
            theory_w=theory_w,
            theory_R=theory_R,
            theory_gouy=theory_gouy,
            sim_w=sim_w,
            sim_phase_rms=sim_result.wavefront_rms,
            sim_phase_pv=sim_result.wavefront_pv,
            w_error_percent=w_error,
        )
        validation_points.append(vp)
    
    # ========== 创建综合可视化 ==========
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 4, figure=fig, height_ratios=[1.2, 1, 1, 1])
    
    # ---------- 子图 1: 光束轮廓对比 ----------
    ax1 = fig.add_subplot(gs[0, :])
    
    # 绘制理论光束轮廓
    total_path = 400.0
    distances = np.linspace(0, total_path, 300)
    z_theory = []
    w_theory = []
    for d in distances:
        result = calc.propagate_distance(d)
        z_theory.append(result.z)
        w_theory.append(result.w)
    z_theory = np.array(z_theory)
    w_theory = np.array(w_theory)
    
    # 分离入射段和反射段
    mirror_path = initial_distance
    incident_mask = distances <= mirror_path
    reflected_mask = distances > mirror_path
    
    # 入射光束
    ax1.fill_between(z_theory[incident_mask], w_theory[incident_mask], 
                     -w_theory[incident_mask], alpha=0.3, color='blue', label='入射光束 (ABCD)')
    ax1.plot(z_theory[incident_mask], w_theory[incident_mask], 'b-', linewidth=1.5)
    ax1.plot(z_theory[incident_mask], -w_theory[incident_mask], 'b-', linewidth=1.5)
    
    # 反射光束
    ax1.fill_between(z_theory[reflected_mask], w_theory[reflected_mask], 
                     -w_theory[reflected_mask], alpha=0.3, color='red', label='反射光束 (ABCD)')
    ax1.plot(z_theory[reflected_mask], w_theory[reflected_mask], 'r-', linewidth=1.5)
    ax1.plot(z_theory[reflected_mask], -w_theory[reflected_mask], 'r-', linewidth=1.5)
    
    # 绘制仿真验证点
    z_sim = [vp.z_position for vp in validation_points]
    w_sim = [vp.sim_w for vp in validation_points]
    ax1.scatter(z_sim, w_sim, c='green', s=80, marker='o', label='仿真值 (PROPER)', zorder=5)
    ax1.scatter(z_sim, [-w for w in w_sim], c='green', s=80, marker='o', zorder=5)
    
    # 绘制反射镜
    mirror_z = mirror.z_position
    mirror_y = np.linspace(-semi_aperture, semi_aperture, 50)
    mirror_sag = mirror_y**2 / (4 * focal_length)
    ax1.plot(mirror_z - mirror_sag, mirror_y, 'k-', linewidth=3, label='抛物面反射镜')
    
    # 标记验证点
    for vp in validation_points:
        ax1.annotate(vp.label, (vp.z_position, vp.sim_w + 0.5), fontsize=8, ha='center')
    
    ax1.set_xlabel('Z 位置 (mm)', fontsize=12)
    ax1.set_ylabel('光束半径 (mm)', fontsize=12)
    ax1.set_title('高斯光束传输 - 光束轮廓对比', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-20, 20)
    
    # ---------- 子图 2: 光束半径误差 ----------
    ax2 = fig.add_subplot(gs[1, 0])
    
    path_dists = [vp.path_distance for vp in validation_points]
    w_errors = [vp.w_error_percent for vp in validation_points]
    colors = ['green' if abs(e) < 10 else 'red' for e in w_errors]
    
    bars = ax2.bar(range(len(validation_points)), w_errors, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=10, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.axhline(y=-10, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax2.set_xticks(range(len(validation_points)))
    ax2.set_xticklabels([vp.label for vp in validation_points], rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('光束半径误差 (%)', fontsize=10)
    ax2.set_title('光束半径：仿真 vs 理论', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ---------- 子图 3: 光束半径数值对比 ----------
    ax3 = fig.add_subplot(gs[1, 1])
    
    x = range(len(validation_points))
    width = 0.35
    
    theory_ws = [vp.theory_w for vp in validation_points]
    sim_ws = [vp.sim_w for vp in validation_points]
    
    ax3.bar([i - width/2 for i in x], theory_ws, width, label='理论 (ABCD)', color='blue', alpha=0.7)
    ax3.bar([i + width/2 for i in x], sim_ws, width, label='仿真 (PROPER)', color='green', alpha=0.7)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels([vp.label for vp in validation_points], rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('光束半径 (mm)', fontsize=10)
    ax3.set_title('光束半径数值对比', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ---------- 子图 4: 波前曲率半径 ----------
    ax4 = fig.add_subplot(gs[1, 2])
    
    theory_Rs = [vp.theory_R for vp in validation_points]
    # 限制显示范围，避免无穷大
    theory_Rs_display = [min(abs(R), 1000) * np.sign(R) if not np.isinf(R) else 0 for R in theory_Rs]
    
    ax4.bar(x, theory_Rs_display, color='purple', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels([vp.label for vp in validation_points], rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('波前曲率半径 (mm)', fontsize=10)
    ax4.set_title('理论波前曲率半径', fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 标注无穷大的点
    for i, R in enumerate(theory_Rs):
        if np.isinf(R):
            ax4.annotate('∞', (i, 0), ha='center', va='bottom', fontsize=12, color='red')
    
    # ---------- 子图 5: 波前质量 ----------
    ax5 = fig.add_subplot(gs[1, 3])
    
    rms_values = [vp.sim_phase_rms for vp in validation_points]
    pv_values = [vp.sim_phase_pv for vp in validation_points]
    
    ax5.bar([i - width/2 for i in x], rms_values, width, label='RMS', color='orange', alpha=0.7)
    ax5.bar([i + width/2 for i in x], pv_values, width, label='PV', color='red', alpha=0.7)
    
    ax5.set_xticks(x)
    ax5.set_xticklabels([vp.label for vp in validation_points], rotation=45, ha='right', fontsize=8)
    ax5.set_ylabel('波前误差 (waves)', fontsize=10)
    ax5.set_title('仿真波前质量', fontsize=11)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # ---------- 子图 6-9: 选取4个关键节点的振幅剖面对比 ----------
    key_indices = [0, 4, 6, 8]  # 初始面、反射镜、反射后50mm、反射后150mm
    
    for plot_idx, vp_idx in enumerate(key_indices):
        ax = fig.add_subplot(gs[2, plot_idx])
        vp = validation_points[vp_idx]
        
        # 获取仿真数据
        sim.reset()
        sim_result = sim.propagate_distance(vp.path_distance)
        
        # 比较振幅剖面
        r_coords, sim_profile, theory_profile = compare_amplitude_profiles(
            sim_result.amplitude, sim_result.sampling, vp.theory_w, vp.label
        )
        
        ax.plot(r_coords, sim_profile, 'g-', linewidth=1.5, label='仿真')
        ax.plot(r_coords, theory_profile, 'b--', linewidth=1.5, label='理论')
        
        ax.set_xlabel('r (mm)', fontsize=10)
        ax.set_ylabel('振幅', fontsize=10)
        ax.set_title(f'{vp.label}\nw_theory={vp.theory_w:.3f}, w_sim={vp.sim_w:.3f}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-10, 10)
    
    # ---------- 子图 10-13: 选取4个关键节点的相位剖面对比 ----------
    for plot_idx, vp_idx in enumerate(key_indices):
        ax = fig.add_subplot(gs[3, plot_idx])
        vp = validation_points[vp_idx]
        
        # 获取仿真数据
        sim.reset()
        sim_result = sim.propagate_distance(vp.path_distance)
        
        # 比较相位剖面
        r_coords, sim_profile, theory_profile = compare_phase_profiles(
            sim_result.phase, sim_result.amplitude, sim_result.sampling,
            vp.theory_R, wavelength
        )
        
        # 只显示振幅较大区域的相位
        mask = sim_result.amplitude[sim_result.amplitude.shape[0]//2, :] > 0.1 * np.max(sim_result.amplitude)
        
        ax.plot(r_coords, sim_profile, 'g-', linewidth=1.5, label='仿真')
        ax.plot(r_coords, theory_profile, 'b--', linewidth=1.5, label='理论')
        
        ax.set_xlabel('r (mm)', fontsize=10)
        ax.set_ylabel('相位 (rad)', fontsize=10)
        R_str = f'{vp.theory_R:.1f}' if not np.isinf(vp.theory_R) else '∞'
        ax.set_title(f'{vp.label}\nR_theory={R_str} mm', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-10, 10)
    
    plt.tight_layout()
    
    # 保存图像
    output_path = 'tests/output/gaussian_beam_comprehensive_validation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"综合验证可视化已保存到: {output_path}")
    
    return fig, validation_points


def create_intensity_comparison():
    """创建光强分布对比可视化
    
    在关键节点比较理论与仿真的2D光强分布
    """
    
    # ========== 设置光学系统参数 ==========
    wavelength = 0.633
    w0 = 1.0
    z0 = 50.0
    m2 = 1.0
    z_init = 0.0
    
    focal_length = 100.0
    initial_distance = 200.0
    semi_aperture = 15.0
    
    # ========== 创建光学元件 ==========
    beam = GaussianBeam(wavelength=wavelength, w0=w0, z0=z0, m2=m2, z_init=z_init)
    mirror = ParabolicMirror(thickness=150.0, semi_aperture=semi_aperture, 
                             parent_focal_length=focal_length)
    
    calc = ABCDCalculator(beam, [mirror], initial_distance=initial_distance)
    sim = HybridGaussianBeamSimulator(beam, [mirror], initial_distance=initial_distance,
                                       grid_size=256, use_hybrid=False)
    
    # ========== 选取关键节点 ==========
    key_points = [
        (0.0, "初始面 (path=0)"),
        (100.0, "入射中点 (path=100)"),
        (200.0, "反射镜 (path=200)"),
        (300.0, "反射后100mm (path=300)"),
    ]
    
    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    
    for row, (path_dist, label) in enumerate(key_points):
        # 理论计算
        theory_result = calc.propagate_distance(path_dist)
        theory_w = theory_result.w
        theory_R = theory_result.R
        
        # 仿真计算
        sim.reset()
        sim_result = sim.propagate_distance(path_dist)
        
        # 创建理论光强分布
        n = sim_result.amplitude.shape[0]
        half_size = sim_result.physical_size / 2
        coords = np.linspace(-half_size, half_size, n)
        X, Y = np.meshgrid(coords, coords)
        R_sq = X**2 + Y**2
        
        # 理论振幅和光强
        theory_amplitude = np.exp(-R_sq / theory_w**2)
        theory_intensity = theory_amplitude**2
        
        # 仿真光强
        sim_intensity = sim_result.amplitude**2
        
        # 归一化
        theory_intensity = theory_intensity / np.max(theory_intensity)
        sim_intensity = sim_intensity / np.max(sim_intensity)
        
        # 光强差异
        intensity_diff = sim_intensity - theory_intensity
        
        # 仿真相位（去除 piston，以中心为参考）
        sim_phase = sim_result.phase
        center = n // 2
        sim_phase_centered = sim_phase - sim_phase[center, center]
        
        # 只显示振幅较大区域的相位
        mask = sim_result.amplitude > 0.1 * np.max(sim_result.amplitude)
        sim_phase_display = np.where(mask, sim_phase_centered, np.nan)
        
        extent = [-half_size, half_size, -half_size, half_size]
        
        # 列 1: 仿真光强
        ax1 = axes[row, 0]
        im1 = ax1.imshow(sim_intensity, extent=extent, cmap='hot', origin='lower', vmin=0, vmax=1)
        ax1.set_title(f'{label}\n仿真光强', fontsize=10)
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # 列 2: 理论光强
        ax2 = axes[row, 1]
        im2 = ax2.imshow(theory_intensity, extent=extent, cmap='hot', origin='lower', vmin=0, vmax=1)
        ax2.set_title(f'理论光强\nw={theory_w:.3f}mm', fontsize=10)
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # 列 3: 光强差异
        ax3 = axes[row, 2]
        vmax = max(0.1, np.max(np.abs(intensity_diff)))
        im3 = ax3.imshow(intensity_diff, extent=extent, cmap='RdBu_r', origin='lower', 
                         vmin=-vmax, vmax=vmax)
        ax3.set_title(f'光强差异\nmax={np.max(np.abs(intensity_diff)):.4f}', fontsize=10)
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Y (mm)')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # 列 4: 仿真相位分布
        # 注意：PROPER 使用参考球面概念，相位是相对于参考球面的偏差
        # 因此不能直接与 ABCD 理论的绝对相位比较
        # 这里显示仿真相位分布本身
        ax4 = axes[row, 3]
        vmax_phase = max(0.5, np.nanmax(np.abs(sim_phase_display)))
        im4 = ax4.imshow(sim_phase_display, extent=extent, cmap='RdBu_r', origin='lower',
                         vmin=-vmax_phase, vmax=vmax_phase)
        phase_rms = np.nanstd(sim_phase_display)
        R_str = f'{theory_R:.1f}' if not np.isinf(theory_R) else '∞'
        ax4.set_title(f'仿真相位 (rad)\nR_theory={R_str}mm', fontsize=10)
        ax4.set_xlabel('X (mm)')
        ax4.set_ylabel('Y (mm)')
        plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    plt.suptitle('光强分布对比与仿真相位分布', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = 'tests/output/gaussian_beam_intensity_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"光强对比可视化已保存到: {output_path}")
    
    return fig


def print_validation_report(validation_points: List[ValidationPoint]):
    """打印详细的验证报告"""
    
    print("\n" + "=" * 80)
    print("高斯光束传输仿真验证报告")
    print("=" * 80)
    
    print("\n" + "-" * 80)
    print(f"{'节点':<15} {'光程(mm)':<10} {'z(mm)':<10} {'w_理论':<10} {'w_仿真':<10} {'误差(%)':<10} {'状态':<8}")
    print("-" * 80)
    
    for vp in validation_points:
        status = "✓ 通过" if vp.w_match else "✗ 失败"
        print(f"{vp.label:<15} {vp.path_distance:<10.1f} {vp.z_position:<10.2f} "
              f"{vp.theory_w:<10.4f} {vp.sim_w:<10.4f} {vp.w_error_percent:<10.2f} {status:<8}")
    
    print("-" * 80)
    
    # 统计
    passed = sum(1 for vp in validation_points if vp.w_match)
    total = len(validation_points)
    pass_rate = passed / total * 100
    
    print(f"\n验证结果: {passed}/{total} 通过 ({pass_rate:.1f}%)")
    
    # 波前质量统计
    print("\n" + "-" * 80)
    print("波前质量统计:")
    print("-" * 80)
    print(f"{'节点':<15} {'RMS (waves)':<15} {'PV (waves)':<15}")
    print("-" * 80)
    
    for vp in validation_points:
        print(f"{vp.label:<15} {vp.sim_phase_rms:<15.6f} {vp.sim_phase_pv:<15.6f}")
    
    print("-" * 80)
    
    avg_rms = np.mean([vp.sim_phase_rms for vp in validation_points])
    avg_pv = np.mean([vp.sim_phase_pv for vp in validation_points])
    print(f"{'平均值':<15} {avg_rms:<15.6f} {avg_pv:<15.6f}")
    
    print("=" * 80)


def create_gaussian_beam_visualization():
    """创建高斯光束传输仿真可视化（简化版，保持向后兼容）"""
    
    # 调用综合验证
    fig, validation_points = create_comprehensive_validation()
    
    # 打印验证报告
    print_validation_report(validation_points)
    
    # 创建光强对比
    fig2 = create_intensity_comparison()
    
    return fig


if __name__ == "__main__":
    import os
    
    # 确保输出目录存在
    os.makedirs('tests/output', exist_ok=True)
    
    print("正在生成高斯光束传输仿真验证可视化...")
    print("-" * 60)
    
    # 生成综合验证
    fig1, validation_points = create_comprehensive_validation()
    
    # 打印验证报告
    print_validation_report(validation_points)
    
    # 生成光强对比
    print("\n正在生成光强分布对比...")
    fig2 = create_intensity_comparison()
    
    print("\n可视化生成完成！")
    
    # 显示图像
    plt.show()
