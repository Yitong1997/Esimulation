"""
离轴抛物面反射镜扩束系统仿真与可视化

本脚本模拟由两个倾斜放置的抛物面反射镜构成的激光扩束系统。
采用离轴反射式望远镜结构（类似离轴卡塞格林/格里高利变体）。

系统参数：
- 倍率 (M): 2
- 入射束腰半径 (w1): 15 mm
- 出射束腰半径 (w2): 30 mm
- M^2 因子: 1.2

几何参数：
- M1（凸抛物面镜）: f1 = -100 mm, R1 = -200 mm, 离轴角 15
- M2（凹抛物面镜）: f2 = 200 mm, R2 = 400 mm, 离轴角 15
- 两镜共焦点配置，间距 d = |f1| + f2 = 300 mm

ABCD 矩阵分析：
对于共焦点扩束系统：
M_total = M_M2  M_free  M_M1
        = [[1,0],[-1/f2,1]]  [[1,d],[0,1]]  [[1,0],[-1/f1,1]]

作者：混合光学仿真项目
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass
from typing import List, Tuple, Optional

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加 src 目录到路径
sys.path.insert(0, 'src')

# 导入 PROPER
import proper


# ============================================================================
# ABCD 矩阵计算类
# ============================================================================

@dataclass
class BeamParameters:
    """高斯光束参数"""
    w: float          # 光束半径 (mm)
    R: float          # 波前曲率半径 (mm)
    z: float          # 位置 (mm)
    gouy_phase: float # Gouy 相位 (rad)
    q: complex        # 复光束参数


class OffAxisBeamExpanderABCD:
    """离轴扩束系统 ABCD 矩阵计算器
    
    系统配置：
    - M1: 凸抛物面镜 (f1 < 0)
    - M2: 凹抛物面镜 (f2 > 0)
    - 共焦点配置: 两镜焦点重合
    
    ABCD 矩阵：
    - 自由空间传播: [[1, d], [0, 1]]
    - 反射镜: [[1, 0], [-1/f, 1]]
    """
    
    def __init__(
        self,
        f1: float,           # M1 焦距 (mm)，凸面为负
        f2: float,           # M2 焦距 (mm)，凹面为正
        wavelength: float,   # 波长 (μm)
        w0_input: float,     # 入射束腰半径 (mm)
        m2: float = 1.0,     # M² 因子
        tilt_angle: float = 15.0,  # 离轴角度 (度)
    ):
        self.f1 = f1
        self.f2 = f2
        self.wavelength = wavelength
        self.wavelength_mm = wavelength * 1e-3
        self.w0_input = w0_input
        self.m2 = m2
        self.tilt_angle = np.radians(tilt_angle)
        
        # 共焦点间距
        # 对于开普勒型（两个凹面镜）：d = f1 + f2
        # 对于伽利略型（凸面 + 凹面）：d = |f1| + f2（但不能准直输出）
        if f1 > 0 and f2 > 0:
            # 开普勒型：两个凹面镜
            self.d_confocal = f1 + f2
            self.system_type = "开普勒型（有实焦点）"
        elif f1 < 0 and f2 > 0:
            # 伽利略型：凸面 + 凹面
            self.d_confocal = abs(f1) + f2
            self.system_type = "伽利略型（无实焦点，非准直输出）"
        else:
            self.d_confocal = abs(f1) + abs(f2)
            self.system_type = "其他配置"
        
        # 计算瑞利距离
        self.zR_input = np.pi * w0_input**2 / (m2 * self.wavelength_mm)
        
        # 理论放大倍率
        # 对于开普勒型：M = f2/f1（准直输出）
        # 对于伽利略型：M = f2/|f1|（准直输出，但需要 |f1| > f2）
        if f1 > 0 and f2 > 0:
            self.magnification = f2 / f1
        else:
            self.magnification = abs(f2 / f1)
        
        self.w0_output_theory = w0_input * self.magnification
        
        print(f"系统参数:")
        print(f"  M1 焦距: {f1} mm (凸面)")
        print(f"  M2 焦距: {f2} mm (凹面)")
        print(f"  共焦点间距: {self.d_confocal} mm")
        print(f"  放大倍率: {self.magnification}x")
        print(f"  入射束腰: {w0_input} mm")
        print(f"  理论出射束腰: {self.w0_output_theory} mm")
        print(f"  入射瑞利距离: {self.zR_input:.2f} mm")
    
    @staticmethod
    def free_space_matrix(d: float) -> np.ndarray:
        """自由空间传播矩阵"""
        return np.array([[1, d], [0, 1]], dtype=np.float64)
    
    @staticmethod
    def mirror_matrix(f: float) -> np.ndarray:
        """反射镜矩阵（等效薄透镜）"""
        return np.array([[1, 0], [-1/f, 1]], dtype=np.float64)
    
    def get_total_abcd(self) -> np.ndarray:
        """计算系统总 ABCD 矩阵
        
        M_total = M_M2  M_free  M_M1
        """
        M_M1 = self.mirror_matrix(self.f1)
        M_free = self.free_space_matrix(self.d_confocal)
        M_M2 = self.mirror_matrix(self.f2)
        
        # 矩阵乘法顺序：从右到左
        M_total = M_M2 @ M_free @ M_M1
        return M_total
    
    def q_to_params(self, q: complex, wavelength_mm: float) -> Tuple[float, float]:
        """从复光束参数计算 w 和 R"""
        inv_q = 1.0 / q
        
        # R = 1 / Re(1/q)
        if abs(inv_q.real) < 1e-15:
            R = np.inf
        else:
            R = 1.0 / inv_q.real
        
        # w = -λ / (π * Im(1/q))
        w_sq = -wavelength_mm / (np.pi * inv_q.imag)
        w = np.sqrt(abs(w_sq))
        
        return w, R
    
    def propagate_q(self, q: complex, abcd: np.ndarray) -> complex:
        """应用 ABCD 矩阵变换复光束参数
        
        q' = (A*q + B) / (C*q + D)
        """
        A, B = abcd[0, 0], abcd[0, 1]
        C, D = abcd[1, 0], abcd[1, 1]
        return (A * q + B) / (C * q + D)
    
    def trace_beam(
        self,
        z_waist_input: float,  # 入射束腰相对于 M1 的位置 (mm)，0 表示准直入射
        num_points: int = 200,
        extra_distance: float = 200.0,  # M2 之后的额外传播距离
    ) -> dict:
        """追迹光束通过整个系统
        
        坐标系定义（从 M1 开始）：
        - z = 0: M1 位置
        - z = d_confocal: M2 位置
        - z > d_confocal: M2 之后
        
        当 z_waist_input = 0 时，表示准直入射（束腰在 M1 位置）
        """
        results = {
            'z_input': [],      # 入射段 z 坐标（M1 之前，如果有的话）
            'w_input': [],      # 入射段光束半径
            'R_input': [],      # 入射段曲率半径
            'z_between': [],    # M1-M2 之间 z 坐标
            'w_between': [],    # M1-M2 之间光束半径
            'R_between': [],    # M1-M2 之间曲率半径
            'z_output': [],     # 出射段 z 坐标
            'w_output': [],     # 出射段光束半径
            'R_output': [],     # 出射段曲率半径
        }
        
        # ========== 入射段 ==========
        # 如果 z_waist_input < 0，则有入射段
        if z_waist_input < 0:
            z_input = np.linspace(z_waist_input, 0, num_points // 3)
            for z in z_input:
                dz = z - z_waist_input
                q = complex(dz, self.zR_input)
                w, R = self.q_to_params(q, self.wavelength_mm)
                results['z_input'].append(z)
                results['w_input'].append(w)
                results['R_input'].append(R)
            
            # M1 处的光束参数
            input_distance = abs(z_waist_input)
            q_at_M1 = complex(input_distance, self.zR_input)
        else:
            # z_waist_input = 0，准直入射，束腰在 M1 位置
            # q = i * zR（在束腰处）
            q_at_M1 = complex(0, self.zR_input)
        
        w_at_M1, R_at_M1 = self.q_to_params(q_at_M1, self.wavelength_mm)
        
        # ========== M1 到 M2 之间 ==========
        # 经过 M1 反射
        M_M1 = self.mirror_matrix(self.f1)
        q_after_M1 = self.propagate_q(q_at_M1, M_M1)
        
        z_between = np.linspace(0, self.d_confocal, num_points // 3)
        
        for z in z_between:
            # 从 M1 传播距离 z
            M_prop = self.free_space_matrix(z)
            q = self.propagate_q(q_after_M1, M_prop)
            w, R = self.q_to_params(q, self.wavelength_mm)
            
            results['z_between'].append(z)
            results['w_between'].append(w)
            results['R_between'].append(R)
        
        # M2 处的光束参数
        M_to_M2 = self.free_space_matrix(self.d_confocal)
        q_at_M2 = self.propagate_q(q_after_M1, M_to_M2)
        w_at_M2, R_at_M2 = self.q_to_params(q_at_M2, self.wavelength_mm)
        
        # ========== 出射段 ==========
        # 经过 M2 反射
        M_M2 = self.mirror_matrix(self.f2)
        q_after_M2 = self.propagate_q(q_at_M2, M_M2)
        
        z_output = np.linspace(self.d_confocal, self.d_confocal + extra_distance, num_points // 3)
        
        for z in z_output:
            d = z - self.d_confocal
            M_prop = self.free_space_matrix(d)
            q = self.propagate_q(q_after_M2, M_prop)
            w, R = self.q_to_params(q, self.wavelength_mm)
            
            results['z_output'].append(z)
            results['w_output'].append(w)
            results['R_output'].append(R)
        
        # 计算输出束腰
        w_output, R_output = self.q_to_params(q_after_M2, self.wavelength_mm)
        results['w_at_M1'] = w_at_M1
        results['w_at_M2'] = w_at_M2
        results['w_output_immediate'] = w_output
        results['q_after_M2'] = q_after_M2
        
        # 找到输出束腰位置和半径
        zR_output = abs(q_after_M2.imag)
        z_waist_from_M2 = -q_after_M2.real  # 束腰相对于 M2 的位置
        
        # 从瑞利距离计算束腰半径：zR = π * w0² / (M² * λ)
        # w0 = sqrt(zR * M² * λ / π)
        w0_output = np.sqrt(abs(zR_output * self.m2 * self.wavelength_mm / np.pi))
        
        # 对于准直光束，直接使用 M2 后的光束半径作为输出
        if zR_output > 1e6:  # 准直光束
            w0_output = w_output
            z_waist_from_M2 = np.inf
        
        results['zR_output'] = zR_output
        results['z_waist_output'] = z_waist_from_M2 + self.d_confocal
        results['w0_output'] = w0_output
        
        return results




# ============================================================================
# PROPER 物理光学仿真类
# ============================================================================

class OffAxisBeamExpanderPROPER:
    """使用 PROPER 进行离轴扩束系统的物理光学仿真
    
    仿真流程：
    1. 初始化准直高斯光束波前（在 M1 位置）
    2. 应用 M1 反射（凸面镜，使光束发散）
    3. 传播到 M2
    4. 应用 M2 反射（凹面镜，使光束重新准直）
    5. 传播到观察面
    
    对于准直光束扩束系统：
    - 入射光束直径 = 2 * w0_input
    - 出射光束直径 = 2 * w0_input * |f2/f1| = 2 * w0_input * M
    """
    
    def __init__(
        self,
        f1: float,           # M1 焦距 (mm)
        f2: float,           # M2 焦距 (mm)
        wavelength: float,   # 波长 (μm)
        w0_input: float,     # 入射光束半径 (mm)
        m2: float = 1.0,     # M² 因子
        grid_size: int = 512,
        beam_ratio: float = 0.25,
    ):
        self.f1 = f1
        self.f2 = f2
        self.wavelength = wavelength
        self.wavelength_m = wavelength * 1e-6
        self.w0_input = w0_input
        self.m2 = m2
        self.grid_size = grid_size
        self.beam_ratio = beam_ratio
        
        # 共焦点间距
        # 对于开普勒型（两个凹面镜）：d = f1 + f2
        # 对于伽利略型（凸面 + 凹面）：d = |f1| + f2
        if f1 > 0 and f2 > 0:
            # 开普勒型：两个凹面镜
            self.d_confocal = f1 + f2
        else:
            # 伽利略型或其他
            self.d_confocal = abs(f1) + f2
        
        # 放大倍率
        if f1 > 0 and f2 > 0:
            self.magnification = f2 / f1
        else:
            self.magnification = abs(f2 / f1)
        
        # 计算瑞利距离
        wavelength_mm = wavelength * 1e-3
        self.zR_input = np.pi * w0_input**2 / (m2 * wavelength_mm)
        
        # 输出光束半径（理论值）
        self.w0_output_theory = w0_input * self.magnification
        
        # 光束直径（用于 PROPER 初始化）
        # 使用输出光束直径来确保有足够的网格空间
        self.beam_diameter_m = 2 * self.w0_output_theory * 1e-3 * 1.5  # 留 50% 余量
    
    def simulate(
        self,
        z_waist_input: float,  # 入射束腰相对于 M1 的位置 (mm)，0 表示准直入射
        observation_points: List[float],  # 观察点列表（光程距离，从 M1 开始）
    ) -> dict:
        """运行仿真并在指定观察点收集数据
        
        光程距离定义（从 M1 开始）：
        - 0: M1 位置
        - d_confocal: M2 位置
        - > d_confocal: M2 之后
        """
        
        results = {
            'observation_points': observation_points,
            'amplitudes': [],
            'phases': [],
            'beam_radii': [],
            'samplings': [],
        }
        
        for obs_dist in observation_points:
            amp, phase, w, sampling = self._simulate_to_distance(obs_dist)
            results['amplitudes'].append(amp)
            results['phases'].append(phase)
            results['beam_radii'].append(w)
            results['samplings'].append(sampling)
        
        return results
    
    def _simulate_to_distance(
        self,
        total_distance: float,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """仿真到指定光程距离（从 M1 开始）
        
        光程定义：
        - 0: M1 位置（入射准直光束）
        - d_confocal: M2 位置
        - > d_confocal: M2 之后（出射准直光束）
        """
        
        m2_position = self.d_confocal
        
        # 初始化波前（在 M1 位置，准直光束）
        wfo = proper.prop_begin(
            self.beam_diameter_m,
            self.wavelength_m,
            self.grid_size,
            self.beam_ratio,
        )
        
        proper.prop_define_entrance(wfo)
        
        # 应用准直高斯振幅分布（入射光束）
        self._apply_gaussian_amplitude(wfo, self.w0_input)
        
        if total_distance <= 0:
            # 在 M1 位置或之前，返回入射光束
            pass
        elif total_distance <= m2_position:
            # M1 到 M2 之间
            # 先应用 M1 反射（凸面镜，f < 0，使光束发散）
            f1_m = self.f1 * 1e-3
            proper.prop_lens(wfo, f1_m)
            
            # 传播到目标位置
            if total_distance > 0:
                proper.prop_propagate(wfo, total_distance * 1e-3)
        else:
            # M2 之后
            # 应用 M1 反射
            f1_m = self.f1 * 1e-3
            proper.prop_lens(wfo, f1_m)
            
            # 传播到 M2
            proper.prop_propagate(wfo, m2_position * 1e-3)
            
            # 应用 M2 反射（凹面镜，f > 0，使光束准直）
            f2_m = self.f2 * 1e-3
            proper.prop_lens(wfo, f2_m)
            
            # 传播到目标位置
            d_after_m2 = total_distance - m2_position
            if d_after_m2 > 0:
                proper.prop_propagate(wfo, d_after_m2 * 1e-3)
        
        # 获取结果
        amplitude = proper.prop_get_amplitude(wfo)
        phase = proper.prop_get_phase(wfo)
        sampling_m = proper.prop_get_sampling(wfo)
        sampling_mm = sampling_m * 1e3
        
        # 计算光束半径
        beam_radius = self._estimate_beam_radius(amplitude, sampling_mm)
        
        return amplitude, phase, beam_radius, sampling_mm
    
    def _apply_gaussian_amplitude(self, wfo, w0_mm: float) -> None:
        """应用准直高斯振幅分布（平面波前）"""
        sampling_m = proper.prop_get_sampling(wfo)
        n = self.grid_size
        
        coords = (np.arange(n) - n // 2) * sampling_m
        X, Y = np.meshgrid(coords, coords)
        R_sq = X**2 + Y**2
        
        w0_m = w0_mm * 1e-3
        amplitude = np.exp(-R_sq / w0_m**2)
        
        amplitude_shifted = proper.prop_shift_center(amplitude)
        wfo.wfarr = amplitude_shifted.astype(np.complex128)
        
        # 归一化
        total_intensity = np.sum(np.abs(wfo.wfarr)**2)
        if total_intensity > 0:
            wfo.wfarr /= np.sqrt(total_intensity)
    
    def _estimate_beam_radius(self, amplitude: np.ndarray, sampling: float) -> float:
        """从振幅分布估计光束半径（二阶矩方法）"""
        n = amplitude.shape[0]
        coords = (np.arange(n) - n // 2) * sampling
        X, Y = np.meshgrid(coords, coords)
        R_sq = X**2 + Y**2
        
        intensity = amplitude**2
        total_intensity = np.sum(intensity)
        
        if total_intensity < 1e-10:
            return 0.0
        
        r_sq_mean = np.sum(R_sq * intensity) / total_intensity
        beam_radius = np.sqrt(2 * r_sq_mean)
        
        return beam_radius


# ============================================================================
# 可视化函数
# ============================================================================

def draw_parabolic_mirror(ax, z_pos: float, aperture: float, focal_length: float, 
                          tilt_angle: float = 15.0, color: str = 'purple',
                          label: str = None, mirror_thickness: float = 3.0):
    """绘制抛物面镜的2D截面
    
    参数:
        ax: matplotlib axes 对象
        z_pos: 镜面顶点的 z 位置 (mm)
        aperture: 镜面口径半径 (mm)
        focal_length: 焦距 (mm)，正值为凹面，负值为凸面
        tilt_angle: 离轴倾斜角度 (度)
        color: 镜面颜色
        label: 图例标签
        mirror_thickness: 镜面厚度 (mm)
    """
    # 抛物面方程: z = y^2 / (4f)
    # 对于凸面镜 (f < 0)，曲面向左凸出
    # 对于凹面镜 (f > 0)，曲面向右凹入
    
    y = np.linspace(-aperture, aperture, 100)
    
    # 计算抛物面矢高
    if abs(focal_length) > 1e-6:
        sag = y**2 / (4 * focal_length)
    else:
        sag = np.zeros_like(y)
    
    # 镜面前表面
    z_front = z_pos + sag
    
    # 镜面后表面（简化为平行偏移）
    if focal_length > 0:  # 凹面镜
        z_back = z_front - mirror_thickness
    else:  # 凸面镜
        z_back = z_front + mirror_thickness
    
    # 绘制镜面（填充区域）
    ax.fill_betweenx(y, z_front, z_back, color=color, alpha=0.6, label=label)
    ax.plot(z_front, y, color=color, linewidth=2)
    ax.plot(z_back, y, color=color, linewidth=1, linestyle='--', alpha=0.5)
    
    # 绘制镜面边缘
    ax.plot([z_front[0], z_back[0]], [y[0], y[0]], color=color, linewidth=1.5)
    ax.plot([z_front[-1], z_back[-1]], [y[-1], y[-1]], color=color, linewidth=1.5)
    
    # 添加反射面标记（短线表示反射面）
    n_marks = 7
    mark_indices = np.linspace(10, len(y)-10, n_marks, dtype=int)
    for idx in mark_indices:
        # 计算法向量方向
        if idx > 0 and idx < len(y) - 1:
            dz = z_front[idx+1] - z_front[idx-1]
            dy = y[idx+1] - y[idx-1]
            # 法向量（指向光束来的方向）
            norm = np.sqrt(dz**2 + dy**2)
            if norm > 0:
                nx, ny = -dy/norm, dz/norm
                mark_len = 2.0
                ax.plot([z_front[idx], z_front[idx] + nx*mark_len],
                       [y[idx], y[idx] + ny*mark_len],
                       color=color, linewidth=1, alpha=0.7)


def draw_gaussian_beam_2d(ax, z_array: np.ndarray, w_array: np.ndarray, 
                          color: str = 'red', alpha: float = 0.3,
                          edge_color: str = None, label: str = None,
                          show_rays: bool = True, n_rays: int = 5):
    """绘制高斯光束的2D表示
    
    参数:
        ax: matplotlib axes 对象
        z_array: z 坐标数组 (mm)
        w_array: 光束半径数组 (mm)
        color: 光束填充颜色
        alpha: 透明度
        edge_color: 边缘颜色
        label: 图例标签
        show_rays: 是否显示光线
        n_rays: 光线数量
    """
    if edge_color is None:
        edge_color = color
    
    # 绘制光束包络（填充区域）
    ax.fill_between(z_array, w_array, -w_array, color=color, alpha=alpha, label=label)
    
    # 绘制光束边缘
    ax.plot(z_array, w_array, color=edge_color, linewidth=1.5)
    ax.plot(z_array, -w_array, color=edge_color, linewidth=1.5)
    
    # 绘制光线（可选）
    if show_rays and len(z_array) > 1:
        # 在光束内绘制几条代表性光线
        ray_positions = np.linspace(0.2, 0.8, n_rays)
        for frac in ray_positions:
            ray_y = w_array * frac
            ax.plot(z_array, ray_y, color=edge_color, linewidth=0.5, alpha=0.5)
            ax.plot(z_array, -ray_y, color=edge_color, linewidth=0.5, alpha=0.5)


def draw_optical_system_2d(ax, abcd_calc, abcd_results: dict, proper_results: dict,
                           observation_points: list):
    """绘制完整的2D光学系统图
    
    包含：
    - 抛物面镜 M1 和 M2
    - 光束轮廓
    - 光线追迹
    - 焦点标记
    """
    f1 = abcd_calc.f1
    f2 = abcd_calc.f2
    d_confocal = abcd_calc.d_confocal
    w0_input = abcd_calc.w0_input
    
    # 确定镜面口径（比最大光束半径大一些）
    max_w_between = max(abcd_results['w_between']) if abcd_results['w_between'] else w0_input
    max_w_output = max(abcd_results['w_output']) if abcd_results['w_output'] else w0_input
    max_beam_radius = max(max_w_between, max_w_output, w0_input)
    
    m1_aperture = max_beam_radius * 1.3
    m2_aperture = max_beam_radius * 1.3
    
    # 绘制入射光束（M1 之前）
    z_input_ext = np.linspace(-80, 0, 50)
    w_input_ext = np.ones_like(z_input_ext) * w0_input
    draw_gaussian_beam_2d(ax, z_input_ext, w_input_ext, 
                          color='blue', alpha=0.2, edge_color='blue',
                          label='入射光束', show_rays=True, n_rays=3)
    
    # 绘制 M1-M2 之间的光束
    z_between = np.array(abcd_results['z_between'])
    w_between = np.array(abcd_results['w_between'])
    if len(z_between) > 0:
        draw_gaussian_beam_2d(ax, z_between, w_between,
                              color='orange', alpha=0.2, edge_color='darkorange',
                              label='M1-M2 之间', show_rays=True, n_rays=5)
    
    # 绘制出射光束（M2 之后）
    z_output = np.array(abcd_results['z_output'])
    w_output = np.array(abcd_results['w_output'])
    if len(z_output) > 0:
        draw_gaussian_beam_2d(ax, z_output, w_output,
                              color='red', alpha=0.2, edge_color='darkred',
                              label='出射光束', show_rays=True, n_rays=3)
    
    # 绘制 M1（根据焦距正负判断凹凸）
    m1_label = 'M1 (凹面)' if f1 > 0 else 'M1 (凸面)'
    draw_parabolic_mirror(ax, z_pos=0, aperture=m1_aperture, focal_length=f1,
                          color='purple', label=m1_label, mirror_thickness=4)
    
    # 绘制 M2
    m2_label = 'M2 (凹面)' if f2 > 0 else 'M2 (凸面)'
    draw_parabolic_mirror(ax, z_pos=d_confocal, aperture=m2_aperture, focal_length=f2,
                          color='brown', label=m2_label, mirror_thickness=4)
    
    # 绘制焦点位置
    # M1 的焦点
    f1_pos = f1  # 焦点在 z = f1 位置
    ax.plot(f1_pos, 0, 'x', color='purple', markersize=10, markeredgewidth=2, label='M1 焦点')
    
    # M2 的焦点（相对于 M2 位置）
    f2_pos = d_confocal + f2  # 焦点在 z = d_confocal + f2 位置
    ax.plot(f2_pos, 0, 'x', color='brown', markersize=10, markeredgewidth=2, label='M2 焦点')
    
    # 共焦点位置（两焦点应该重合）
    confocal_pos = abs(f1)  # = d_confocal - f2 (当共焦点配置时)
    ax.axvline(x=confocal_pos, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.annotate('共焦点', (confocal_pos, max_beam_radius * 0.9), 
                ha='center', fontsize=9, color='green')
    
    # 绘制 PROPER 仿真点
    w_proper = proper_results['beam_radii']
    ax.scatter(observation_points, w_proper, c='lime', s=80, marker='o',
               edgecolors='black', linewidths=1.5, zorder=10, label='PROPER 仿真')
    ax.scatter(observation_points, [-w for w in w_proper], c='lime', s=80, marker='o',
               edgecolors='black', linewidths=1.5, zorder=10)
    
    # 添加尺寸标注
    # 入射光束直径
    ax.annotate('', xy=(-60, w0_input), xytext=(-60, -w0_input),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
    ax.text(-65, 0, f'2w1={2*w0_input:.0f}mm', fontsize=9, color='blue',
            ha='right', va='center', rotation=90)
    
    # 出射光束直径（在最后位置）
    if len(z_output) > 0 and len(w_output) > 0:
        z_end = z_output[-1]
        w_end = w_output[-1]
        ax.annotate('', xy=(z_end + 20, w_end), xytext=(z_end + 20, -w_end),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        ax.text(z_end + 25, 0, f'2w2={2*w_end:.0f}mm', fontsize=9, color='red',
                ha='left', va='center', rotation=90)
    
    # 镜间距标注
    ax.annotate('', xy=(0, -max_beam_radius * 1.2), xytext=(d_confocal, -max_beam_radius * 1.2),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax.text(d_confocal / 2, -max_beam_radius * 1.35, f'd={d_confocal:.0f}mm',
            fontsize=9, color='gray', ha='center', va='top')
    
    # 设置坐标轴
    ax.set_xlabel('Z 位置 (mm)', fontsize=12)
    ax.set_ylabel('Y 位置 (mm)', fontsize=12)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, ncol=2)


def create_beam_expander_visualization():
    """创建离轴扩束系统的综合可视化
    
    系统设计说明：
    - 这是一个 2x 扩束系统，由两个共焦点的抛物面镜组成
    - M1（凸面）将入射准直光束发散
    - M2（凹面）将发散光束重新准直，同时扩大光束直径
    
    对于准直光束扩束系统：
    - 入射光束是准直的（平面波前，束腰在无穷远）
    - 经过 M1 后，光束发散，等效于从 M1 焦点发出的球面波
    - 经过 M2 后，光束重新准直，直径扩大 M = |f2/f1| 倍
    
    坐标系定义（从 M1 开始）：
    - z = 0: M1 位置
    - z = d_confocal (300mm): M2 位置
    - z > d_confocal: M2 之后
    """
    
 
    wavelength = 10.6        # μm (CO2 激光)
    w0_input = 15.0          # mm (入射光束半径)
    m2_factor = 1.0          # M² 因子（理想高斯光束）
    
    # 开普勒型扩束器（两个凹面镜，有实焦点）
    # M1（凹面镜，f1 > 0）：将准直光束聚焦
    # M2（凹面镜，f2 > 0）：将发散光束准直
    # 共焦点配置：d = f1 + f2
    # 放大倍率：M = f2/f1
    # 
    # 对于 2x 扩束：f2 = 2 × f1
    # 设 f1 = 100mm, f2 = 200mm, d = 300mm
    
    f1 = -150.0               # mm (M1 凹面镜焦距，正值)
    f2 = 300.0               # mm (M2 凹面镜焦距，正值)
    tilt_angle = 15.0        # 度 (离轴角度)
    
    # 入射束腰位置（相对于 M1）
    z_waist_input = -1000.0      # mm
    
    # ========== 创建计算器 ==========
    print("=" * 60)
    print("离轴抛物面反射镜扩束系统仿真")
    print("=" * 60)
    
    abcd_calc = OffAxisBeamExpanderABCD(
        f1=f1,
        f2=f2,
        wavelength=wavelength,
        w0_input=w0_input,
        m2=m2_factor,
        tilt_angle=tilt_angle,
    )
    
    proper_sim = OffAxisBeamExpanderPROPER(
        f1=f1,
        f2=f2,
        wavelength=wavelength,
        w0_input=w0_input,
        m2=m2_factor,
        grid_size=512,
        beam_ratio=0.25,
    )
    
    # ========== ABCD 矩阵分析 ==========
    print("\n" + "-" * 60)
    print("ABCD 矩阵分析")
    print("-" * 60)
    
    M_total = abcd_calc.get_total_abcd()
    print(f"系统总 ABCD 矩阵:")
    print(f"  A = {M_total[0,0]:.4f}")
    print(f"  B = {M_total[0,1]:.4f}")
    print(f"  C = {M_total[1,0]:.6f}")
    print(f"  D = {M_total[1,1]:.4f}")
    print(f"  det(M) = {np.linalg.det(M_total):.6f} (应为 1)")
    
    # 追迹光束
    abcd_results = abcd_calc.trace_beam(
        z_waist_input=z_waist_input,
        num_points=300,
        extra_distance=300.0,
    )
    
    print(f"\n光束参数:")
    print(f"  M1 处光束半径: {abcd_results['w_at_M1']:.3f} mm")
    print(f"  M2 处光束半径: {abcd_results['w_at_M2']:.3f} mm")
    print(f"  输出束腰半径: {abcd_results['w0_output']:.3f} mm")
    print(f"  输出束腰位置: {abcd_results['z_waist_output']:.2f} mm (相对于 M1)")
    print(f"  输出瑞利距离: {abcd_results['zR_output']:.2f} mm")
    
    # ========== PROPER 仿真 ==========
    print("\n" + "-" * 60)
    print("PROPER 物理光学仿真")
    print("-" * 60)
    
    # 定义观察点（从 M1 开始的光程距离）
    # 0: M1 位置
    # d_confocal: M2 位置
    observation_points = [
        0.0,                              # M1 位置（入射准直光束）
        abcd_calc.d_confocal * 0.25,      # M1-M2 之间 25%
        abcd_calc.d_confocal * 0.5,       # M1-M2 中点
        abcd_calc.d_confocal * 0.75,      # M1-M2 之间 75%
        abcd_calc.d_confocal,             # M2 位置
        abcd_calc.d_confocal + 0.1,       # M2 后 0.1mm（刚出射）
        abcd_calc.d_confocal + 100,       # M2 后 100mm
        abcd_calc.d_confocal + 300,       # M2 后 300mm
    ]
    
    proper_results = proper_sim.simulate(
        z_waist_input=z_waist_input,
        observation_points=observation_points,
    )
    
    # ========== 创建可视化 ==========
    fig = plt.figure(figsize=(22, 20))
    gs = GridSpec(5, 4, figure=fig, height_ratios=[1.5, 1.2, 0.8, 1, 1])
    
    # ---------- 子图 0: 2D 光学系统图 ----------
    ax0 = fig.add_subplot(gs[0, :])
    draw_optical_system_2d(ax0, abcd_calc, abcd_results, proper_results, observation_points)
    ax0.set_title(f'离轴抛物面反射镜扩束系统 - 2D 光学布局\n'
                  f'(M={abcd_calc.magnification:.1f}x, f1={f1}mm, f2={f2}mm, d={abcd_calc.d_confocal}mm)',
                  fontsize=14, fontweight='bold')
    
    # ---------- 子图 1: 光束轮廓对比（简化版） ----------
    ax1 = fig.add_subplot(gs[1, :])
    
    # 绘制 ABCD 理论光束轮廓
    # 入射段（z_waist_input 到 0，但 z_waist_input=0 时这段为空）
    z_input = np.array(abcd_results['z_input'])
    w_input = np.array(abcd_results['w_input'])
    if len(z_input) > 0:
        ax1.fill_between(z_input, w_input, -w_input, alpha=0.3, color='blue', label='入射光束 (ABCD)')
        ax1.plot(z_input, w_input, 'b-', linewidth=1.5)
        ax1.plot(z_input, -w_input, 'b-', linewidth=1.5)
    
    # M1-M2 之间
    z_between = np.array(abcd_results['z_between'])
    w_between = np.array(abcd_results['w_between'])
    ax1.fill_between(z_between, w_between, -w_between, alpha=0.3, color='orange', label='M1-M2 之间 (ABCD)')
    ax1.plot(z_between, w_between, 'orange', linewidth=1.5)
    ax1.plot(z_between, -w_between, 'orange', linewidth=1.5)
    
    # 出射段
    z_output = np.array(abcd_results['z_output'])
    w_output = np.array(abcd_results['w_output'])
    ax1.fill_between(z_output, w_output, -w_output, alpha=0.3, color='red', label='出射光束 (ABCD)')
    ax1.plot(z_output, w_output, 'r-', linewidth=1.5)
    ax1.plot(z_output, -w_output, 'r-', linewidth=1.5)
    
    # 绘制 PROPER 仿真点
    # 观察点已经是从 M1 开始的光程距离，直接使用
    z_proper = observation_points
    w_proper = proper_results['beam_radii']
    ax1.scatter(z_proper, w_proper, c='green', s=100, marker='o', 
                label='仿真值 (PROPER)', zorder=5, edgecolors='black')
    ax1.scatter(z_proper, [-w for w in w_proper], c='green', s=100, marker='o', 
                zorder=5, edgecolors='black')
    
    # 绘制反射镜位置
    ax1.axvline(x=0, color='purple', linestyle='--', linewidth=2, label='M1 (凸面)')
    ax1.axvline(x=abcd_calc.d_confocal, color='brown', linestyle='--', linewidth=2, label='M2 (凹面)')
    
    # 标注
    max_w = max(max(w_between), max(w_output)) if len(w_between) > 0 else max(w_output)
    ax1.annotate('M1\n(凸面镜)', (0, max_w*1.1), ha='center', fontsize=10, color='purple')
    ax1.annotate('M2\n(凹面镜)', (abcd_calc.d_confocal, max_w*1.1), ha='center', fontsize=10, color='brown')
    
    ax1.set_xlabel('Z 位置 (mm)', fontsize=12)
    ax1.set_ylabel('光束半径 (mm)', fontsize=12)
    ax1.set_title(f'离轴抛物面反射镜扩束系统 - 光束轮廓对比\n'
                  f'(M={abcd_calc.magnification}x, w1={w0_input}mm -> w2={abcd_results["w0_output"]:.1f}mm)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-20, max(z_output) + 20)
    
    # ---------- 子图 2: 光束半径对比 ----------
    ax2 = fig.add_subplot(gs[2, 0:2])
    
    # ABCD 理论值（在观察点）
    abcd_w_at_obs = []
    d_confocal = abcd_calc.d_confocal
    
    for obs in observation_points:
        if obs <= 0:
            # M1 位置或之前（准直入射）
            w = w0_input
        elif obs <= d_confocal:
            # M1-M2 之间
            # 在 M1 处：q = i * zR（准直光束）
            q_at_M1 = complex(0, abcd_calc.zR_input)
            M_M1 = abcd_calc.mirror_matrix(f1)
            q_after_M1 = abcd_calc.propagate_q(q_at_M1, M_M1)
            d = obs
            M_prop = abcd_calc.free_space_matrix(d)
            q = abcd_calc.propagate_q(q_after_M1, M_prop)
            w, _ = abcd_calc.q_to_params(q, abcd_calc.wavelength_mm)
        else:
            # 出射段
            q_at_M1 = complex(0, abcd_calc.zR_input)
            M_M1 = abcd_calc.mirror_matrix(f1)
            q_after_M1 = abcd_calc.propagate_q(q_at_M1, M_M1)
            M_to_M2 = abcd_calc.free_space_matrix(d_confocal)
            q_at_M2 = abcd_calc.propagate_q(q_after_M1, M_to_M2)
            M_M2 = abcd_calc.mirror_matrix(f2)
            q_after_M2 = abcd_calc.propagate_q(q_at_M2, M_M2)
            d = obs - d_confocal
            M_prop = abcd_calc.free_space_matrix(d)
            q = abcd_calc.propagate_q(q_after_M2, M_prop)
            w, _ = abcd_calc.q_to_params(q, abcd_calc.wavelength_mm)
        abcd_w_at_obs.append(w)
    
    x = np.arange(len(observation_points))
    width = 0.35
    
    ax2.bar(x - width/2, abcd_w_at_obs, width, label='理论 (ABCD)', color='blue', alpha=0.7)
    ax2.bar(x + width/2, w_proper, width, label='仿真 (PROPER)', color='green', alpha=0.7)
    
    # 8 个观察点的标签
    labels = ['M1', '25%', '50%', '75%', 'M2', 'M2出射', 'M2+100', 'M2+300']
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('光束半径 (mm)', fontsize=10)
    ax2.set_title('光束半径：理论 vs 仿真', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ---------- 子图 3: 光束半径误差 ----------
    ax3 = fig.add_subplot(gs[2, 2:4])
    
    errors = [(w_p - w_a) / w_a * 100 if w_a > 0 else 0 for w_p, w_a in zip(w_proper, abcd_w_at_obs)]
    colors = ['green' if abs(e) < 10 else 'red' for e in errors]
    
    ax3.bar(x, errors, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.axhline(y=10, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    ax3.axhline(y=-10, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('误差 (%)', fontsize=10)
    ax3.set_title('光束半径误差 (仿真 - 理论) / 理论', fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ---------- 子图 4-7: 振幅分布 ----------
    # 选择 4 个关键观察点：M1, 50%(中点), M2, M2+300
    key_indices = [0, 2, 4, 7]  # M1, 50%, M2, M2+300
    key_labels = ['M1 位置', 'M1-M2 中点', 'M2 位置', 'M2 后 300mm']
    
    for plot_idx, (obs_idx, label) in enumerate(zip(key_indices, key_labels)):
        ax = fig.add_subplot(gs[3, plot_idx])
        
        amp = proper_results['amplitudes'][obs_idx]
        sampling = proper_results['samplings'][obs_idx]
        n = amp.shape[0]
        half_size = sampling * n / 2
        extent = [-half_size, half_size, -half_size, half_size]
        
        # 归一化振幅
        amp_norm = amp / np.max(amp) if np.max(amp) > 0 else amp
        
        im = ax.imshow(amp_norm, extent=extent, cmap='hot', origin='lower', vmin=0, vmax=1)
        ax.set_xlabel('X (mm)', fontsize=9)
        ax.set_ylabel('Y (mm)', fontsize=9)
        ax.set_title(f'{label}\nw={proper_results["beam_radii"][obs_idx]:.2f}mm', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, label='归一化振幅')
        
        # 绘制光束半径圆
        w = proper_results['beam_radii'][obs_idx]
        circle = plt.Circle((0, 0), w, fill=False, color='cyan', linewidth=1.5, linestyle='--')
        ax.add_patch(circle)
    
    # ---------- 子图 8-11: 相位分布 ----------
    for plot_idx, (obs_idx, label) in enumerate(zip(key_indices, key_labels)):
        ax = fig.add_subplot(gs[4, plot_idx])
        
        phase = proper_results['phases'][obs_idx]
        amp = proper_results['amplitudes'][obs_idx]
        sampling = proper_results['samplings'][obs_idx]
        n = phase.shape[0]
        half_size = sampling * n / 2
        extent = [-half_size, half_size, -half_size, half_size]
        
        # 只显示振幅较大区域的相位
        mask = amp > 0.1 * np.max(amp) if np.max(amp) > 0 else np.ones_like(amp, dtype=bool)
        phase_display = np.where(mask, phase, np.nan)
        
        # 去除 piston
        center = n // 2
        if not np.isnan(phase_display[center, center]):
            phase_display = phase_display - phase_display[center, center]
        
        vmax = np.nanmax(np.abs(phase_display))
        if vmax < 0.1 or np.isnan(vmax):
            vmax = 0.1
        
        im = ax.imshow(phase_display, extent=extent, cmap='RdBu_r', origin='lower',
                       vmin=-vmax, vmax=vmax)
        ax.set_xlabel('X (mm)', fontsize=9)
        ax.set_ylabel('Y (mm)', fontsize=9)
        ax.set_title(f'{label}\n相位分布', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, label='相位 (rad)')
    
    plt.suptitle('离轴抛物面反射镜扩束系统 - ABCD 理论与 PROPER 仿真对比', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存图像
    output_path = 'tests/output/offaxis_beam_expander_visualization.png'
    os.makedirs('tests/output', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n可视化已保存到: {output_path}")
    
    return fig, abcd_results, proper_results, abcd_calc


def print_comparison_report(abcd_results: dict, proper_results: dict, abcd_calc):
    """打印详细的对比报告"""
    
    print("\n" + "=" * 60)
    print("理论与仿真对比报告")
    print("=" * 60)
    
    print(f"\n{'观察点':<15} {'理论 w (mm)':<15} {'仿真 w (mm)':<15} {'误差 (%)':<12}")
    print("-" * 60)
    
    d_confocal = abcd_calc.d_confocal
    f1 = abcd_calc.f1
    f2 = abcd_calc.f2
    
    labels = ['M1', '25%', '50%', '75%', 'M2', 'M2出射', 'M2+100', 'M2+300']
    observation_points = proper_results['observation_points']
    
    for i, (obs, label) in enumerate(zip(observation_points, labels)):
        # 计算理论值
        if obs <= 0:
            # M1 位置（准直入射）
            w_theory = abcd_calc.w0_input
        elif obs <= d_confocal:
            # M1-M2 之间
            q_at_M1 = complex(0, abcd_calc.zR_input)
            M_M1 = abcd_calc.mirror_matrix(f1)
            q_after_M1 = abcd_calc.propagate_q(q_at_M1, M_M1)
            d = obs
            M_prop = abcd_calc.free_space_matrix(d)
            q = abcd_calc.propagate_q(q_after_M1, M_prop)
            w_theory, _ = abcd_calc.q_to_params(q, abcd_calc.wavelength_mm)
        else:
            # 出射段
            q_at_M1 = complex(0, abcd_calc.zR_input)
            M_M1 = abcd_calc.mirror_matrix(f1)
            q_after_M1 = abcd_calc.propagate_q(q_at_M1, M_M1)
            M_to_M2 = abcd_calc.free_space_matrix(d_confocal)
            q_at_M2 = abcd_calc.propagate_q(q_after_M1, M_to_M2)
            M_M2 = abcd_calc.mirror_matrix(f2)
            q_after_M2 = abcd_calc.propagate_q(q_at_M2, M_M2)
            d = obs - d_confocal
            M_prop = abcd_calc.free_space_matrix(d)
            q = abcd_calc.propagate_q(q_after_M2, M_prop)
            w_theory, _ = abcd_calc.q_to_params(q, abcd_calc.wavelength_mm)
        
        w_sim = proper_results['beam_radii'][i]
        error = (w_sim - w_theory) / w_theory * 100 if w_theory > 0 else 0
        
        print(f"{label:<15} {w_theory:<15.4f} {w_sim:<15.4f} {error:<12.2f}")
    
    print("-" * 60)
    
    # 扩束比验证
    # 对于伽利略型扩束器，扩束比应该比较 M1 处和 M2 处的光束半径
    w_at_M1 = proper_results['beam_radii'][0]  # M1 位置（索引 0）
    w_at_M2 = proper_results['beam_radii'][4]  # M2 位置（索引 4）
    magnification_sim = w_at_M2 / w_at_M1 if w_at_M1 > 0 else 0
    magnification_theory = abcd_calc.magnification
    
    print(f"\n扩束比验证（M1 -> M2）:")
    print(f"  入射光束半径 (M1): {w_at_M1:.2f} mm")
    print(f"  出射光束半径 (M2): {w_at_M2:.2f} mm")
    print(f"  理论扩束比: {magnification_theory:.2f}x")
    print(f"  仿真扩束比: {magnification_sim:.2f}x")
    if magnification_theory > 0:
        print(f"  误差: {(magnification_sim - magnification_theory) / magnification_theory * 100:.2f}%")
    
    # 说明系统特性
    print(f"\n系统特性说明:")
    print(f"  - 这是伽利略型扩束器（无实焦点）")
    print(f"  - M1（凸面镜）使光束发散")
    print(f"  - M2（凹面镜）使光束重新准直")
    print(f"  - 光束在 M2 处达到最大直径")
    print(f"  - M2 之后光束保持准直（理想情况）")
    
    print("=" * 60)


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("正在生成离轴抛物面反射镜扩束系统仿真...")
    print("-" * 60)
    
    # 创建可视化（返回 abcd_calc）
    fig, abcd_results, proper_results, abcd_calc = create_beam_expander_visualization()
    
    # 打印对比报告
    print_comparison_report(abcd_results, proper_results, abcd_calc)
    
    print("\n仿真完成！")
    
    # 显示图像
    plt.show()


